from __future__ import annotations

import shutil
import stat
import threading
from pathlib import Path

import pytest

import tldw_chatbook.Chat.console_scratch_space as scratch_module
from tldw_chatbook.Chat.console_scratch_space import (
    ConsoleScratchSnapshot,
    ConsoleScratchSpaceManager,
    ConsoleScratchSpaceUnavailable,
)


def test_snapshots_are_distinct_owner_only_and_identifier_free(tmp_path: Path) -> None:
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)

    first = manager.snapshot("session-visible-id-a")
    second = manager.snapshot("session-visible-id-b")

    assert first.root != second.root
    assert "session-visible-id" not in first.root.name
    assert stat.S_IMODE(first.root.stat().st_mode) == 0o700
    assert not first.root.is_symlink()
    assert manager.dispose()


def test_chat_cannot_observe_another_chat_scratch_contents(tmp_path: Path) -> None:
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    first = manager.snapshot("a")
    second = manager.snapshot("b")

    with manager.lease(first) as root:
        (root / "marker.txt").write_text("a", encoding="utf-8")

    assert not (second.root / "marker.txt").exists()
    assert manager.dispose()


def test_close_rejects_new_lease_and_waits_for_last_active_lease(
    tmp_path: Path,
) -> None:
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    snapshot = manager.snapshot("a")
    lease = manager.lease(snapshot)
    lease.__enter__()

    manager.close("a")

    with pytest.raises(ConsoleScratchSpaceUnavailable):
        with manager.lease(snapshot):
            pass
    assert snapshot.root.exists()

    lease.__exit__(None, None, None)

    assert manager.wait_for_cleanup(timeout_seconds=2.0)
    assert not snapshot.root.exists()


def test_reopen_gets_new_generation_and_empty_root(tmp_path: Path) -> None:
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    old = manager.snapshot("session")
    (old.root / "marker").write_text("old", encoding="utf-8")

    manager.close("session")

    assert manager.wait_for_cleanup(timeout_seconds=2.0)
    fresh = manager.snapshot("session")
    assert fresh.token != old.token
    assert fresh.root != old.root
    assert not (fresh.root / "marker").exists()
    assert manager.dispose()


def test_replaced_root_fails_closed_without_deleting_replacement(
    tmp_path: Path,
) -> None:
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    snapshot = manager.snapshot("session")
    displaced = tmp_path / "original-root"
    snapshot.root.rename(displaced)
    snapshot.root.mkdir(mode=0o700)
    (snapshot.root / "keep.txt").write_text("replacement", encoding="utf-8")

    with pytest.raises(ConsoleScratchSpaceUnavailable):
        with manager.lease(snapshot):
            pass

    assert manager.wait_for_cleanup(timeout_seconds=2.0)
    assert (snapshot.root / "keep.txt").read_text(encoding="utf-8") == "replacement"
    shutil.rmtree(snapshot.root)
    shutil.rmtree(displaced)


def test_cleanup_failure_stays_tombstoned_for_later_dispose_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    snapshot = manager.snapshot("session")
    cleanup_attempted = threading.Event()
    real_rmtree = scratch_module.shutil.rmtree

    def fail_cleanup(path: Path) -> None:
        cleanup_attempted.set()
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(scratch_module.shutil, "rmtree", fail_cleanup)
    manager.close("session")

    assert cleanup_attempted.wait(timeout=2.0)
    assert not manager.wait_for_cleanup(timeout_seconds=0.01)
    with pytest.raises(ConsoleScratchSpaceUnavailable):
        with manager.lease(snapshot):
            pass

    monkeypatch.setattr(scratch_module.shutil, "rmtree", real_rmtree)
    assert manager.dispose(timeout_seconds=2.0)
    assert not snapshot.root.exists()


def test_dispose_is_bounded_with_active_lease_and_idempotent(tmp_path: Path) -> None:
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    snapshot = manager.snapshot("session")
    lease = manager.lease(snapshot)
    lease.__enter__()

    assert not manager.dispose(timeout_seconds=0.01)
    with pytest.raises(ConsoleScratchSpaceUnavailable):
        manager.snapshot("another-session")

    lease.__exit__(None, None, None)
    assert manager.wait_for_cleanup(timeout_seconds=2.0)
    assert manager.dispose(timeout_seconds=0.01)
    assert manager.dispose(timeout_seconds=0.01)


def test_snapshot_is_immutable(tmp_path: Path) -> None:
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    snapshot = manager.snapshot("session")

    with pytest.raises((AttributeError, TypeError)):
        snapshot.root = tmp_path  # type: ignore[misc]

    assert isinstance(snapshot, ConsoleScratchSnapshot)
    assert manager.is_live(snapshot)
    assert manager.dispose()
