from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from tldw_chatbook.Notes import sync_paths
from tldw_chatbook.Notes.sync_paths import (
    PinnedSyncRoot,
    SyncPathError,
    SyncPathPartialError,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="descriptor containment contract is POSIX-specific",
)


def _issue_reasons(issues) -> set[str]:
    return {issue.reason for issue in issues}


def test_selected_root_link_is_rejected(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    selected = tmp_path / "selected"
    selected.symlink_to(canonical, target_is_directory=True)
    with pytest.raises(SyncPathError, match="root_link_or_reparse"):
        PinnedSyncRoot(selected)


def test_scan_skips_cross_device_entry_and_keeps_safe_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    (root_path / "safe.md").write_text("safe", encoding="utf-8")
    (root_path / "mounted.md").write_text("mounted", encoding="utf-8")

    with PinnedSyncRoot(root_path) as root:
        original = root._same_device
        monkeypatch.setattr(
            root,
            "_same_device",
            lambda entry_stat: (
                False
                if entry_stat.st_ino == (root_path / "mounted.md").stat().st_ino
                else original(entry_stat)
            ),
        )
        files, issues = root.scan([".md"])

    assert set(files) == {Path("safe.md")}
    assert "cross_device" in _issue_reasons(issues)


def test_scan_skips_simulated_reparse_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    candidate = root_path / "reparse.md"
    candidate.write_text("must not import", encoding="utf-8")
    candidate_inode = candidate.stat().st_ino
    original = sync_paths._is_reparse
    monkeypatch.setattr(
        sync_paths,
        "_is_reparse",
        lambda entry_stat: entry_stat.st_ino == candidate_inode or original(entry_stat),
    )

    with PinnedSyncRoot(root_path) as root:
        files, issues = root.scan([".md"])

    assert files == {}
    assert _issue_reasons(issues) == {"link_or_reparse"}


@pytest.mark.parametrize("mode", [0o600, 0o640, 0o644])
def test_atomic_replacement_preserves_existing_mode(
    tmp_path: Path,
    mode: int,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_text("before", encoding="utf-8")
    target.chmod(mode)

    with PinnedSyncRoot(root_path) as root:
        result = root.write_text(Path("note.md"), "after")

    assert result.content == "after"
    assert target.read_text(encoding="utf-8") == "after"
    assert stat.S_IMODE(target.stat().st_mode) == mode


def test_new_file_and_parent_are_private(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()

    with PinnedSyncRoot(root_path) as root:
        root.write_text(Path("nested/note.md"), "private")

    assert stat.S_IMODE((root_path / "nested").stat().st_mode) == 0o700
    assert stat.S_IMODE((root_path / "nested/note.md").stat().st_mode) == 0o600


def test_final_target_replacement_race_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_text("original", encoding="utf-8")

    def replace_target(_relative_path: Path) -> None:
        target.unlink()
        target.write_text("racer", encoding="utf-8")

    with PinnedSyncRoot(root_path) as root:
        monkeypatch.setattr(root, "_before_replace", replace_target)
        with pytest.raises(SyncPathError, match="target_identity_changed"):
            root.write_text(Path("note.md"), "unsafe")

    assert target.read_text(encoding="utf-8") == "racer"


def test_intermediate_parent_replacement_race_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    parent = root_path / "parent"
    parent.mkdir(parents=True)
    target = parent / "note.md"
    target.write_text("original", encoding="utf-8")

    def replace_parent(_relative_path: Path) -> None:
        parent.rename(root_path / "moved-parent")
        parent.mkdir()

    with PinnedSyncRoot(root_path) as root:
        monkeypatch.setattr(root, "_before_replace", replace_parent)
        with pytest.raises(SyncPathError, match="parent_identity_changed"):
            root.write_text(Path("parent/note.md"), "unsafe")

    assert target.exists() is False
    assert (root_path / "moved-parent/note.md").read_text(encoding="utf-8") == (
        "original"
    )


def test_unsupported_descriptor_guards_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    monkeypatch.setattr(sync_paths, "_descriptor_guards_available", lambda: False)

    with PinnedSyncRoot(root_path) as root:
        files, issues = root.scan([".md"])
        with pytest.raises(SyncPathError, match="unsupported_platform"):
            root.write_text(Path("note.md"), "unsafe")

    assert files == {}
    assert _issue_reasons(issues) == {"unsupported_platform"}
    assert not (root_path / "note.md").exists()


def test_relative_escape_is_rejected(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()

    with PinnedSyncRoot(root_path) as root:
        with pytest.raises(SyncPathError, match="invalid_relative_path"):
            root.write_text(Path("../escape.md"), "unsafe")

    assert not (tmp_path / "escape.md").exists()


def test_descriptor_verified_byte_observation_retains_exact_bytes(
    tmp_path: Path,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    payload = b"\xef\xbb\xbfline one\r\nline two"
    (root_path / "note.md").write_bytes(payload)

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")

    assert observed.content == payload
    assert observed.identity.link_count == 1
    assert "note.md" not in repr(observed)
    assert str(observed.identity.device) not in repr(observed.identity)
    assert str(observed.identity.inode) not in repr(observed.identity)


def test_byte_observation_rejects_same_inode_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")

    with PinnedSyncRoot(root_path) as root:
        monkeypatch.setattr(
            root,
            "_after_read",
            lambda _relative_path: target.write_bytes(b"different-size"),
        )
        with pytest.raises(SyncPathError, match="target_changed_during_read"):
            root.read_bytes("note.md")


def test_guarded_byte_replacement_requires_observed_identity(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")
        target.unlink()
        target.write_bytes(b"racer")
        with pytest.raises(SyncPathError, match="target_identity_changed"):
            root.replace_bytes(
                "note.md",
                b"after",
                expected=observed,
                mode=0o640,
            )

    assert target.read_bytes() == b"racer"


def test_guarded_byte_replacement_does_not_clobber_final_boundary_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")

        def swap(_relative_path: Path) -> None:
            target.unlink()
            target.write_bytes(b"racer")

        monkeypatch.setattr(root, "_before_commit", swap)
        with pytest.raises(SyncPathError, match="target_identity_changed"):
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert target.read_bytes() == b"racer"


def test_exchange_verification_failure_rolls_back_without_losing_old_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")
        original = root._read_bytes

        def fail_displaced(parent_fd, leaf, relative_path, entry, max_bytes):
            if leaf.startswith(".note.md.tmp-"):
                raise SyncPathError("target_changed_during_read", relative_path)
            return original(parent_fd, leaf, relative_path, entry, max_bytes)

        monkeypatch.setattr(root, "_read_bytes", fail_displaced)
        with pytest.raises(SyncPathError, match="target_changed_during_read"):
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert target.read_bytes() == b"before"
    assert list(root_path.glob(".note.md.tmp-*")) == []


def test_exchange_rollback_failure_preserves_both_byte_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    original_rename = sync_paths._rename_with_flags
    exchanges = 0

    def fail_rollback(source_fd, source, destination_fd, destination, flags):
        nonlocal exchanges
        if flags == sync_paths._RENAME_EXCHANGE:
            exchanges += 1
            if exchanges == 2:
                raise OSError("rollback failed")
        return original_rename(source_fd, source, destination_fd, destination, flags)

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")
        monkeypatch.setattr(sync_paths, "_rename_with_flags", fail_rollback)
        monkeypatch.setattr(
            root,
            "_read_bytes",
            lambda *_args: (_ for _ in ()).throw(
                SyncPathError("target_changed_during_read", "note.md")
            ),
        )
        with pytest.raises(SyncPathError, match="replacement_rollback_failed"):
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert target.read_bytes() == b"after"
    preserved = list(root_path.glob(".note.md.tmp-*"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"before"


def test_displaced_cleanup_failure_preserves_old_bytes_for_attention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    original_unlink = sync_paths.os.unlink

    def fail_displaced_unlink(path, *args, **kwargs):
        if str(path).startswith(".note.md.tmp-"):
            raise OSError("cleanup failed")
        return original_unlink(path, *args, **kwargs)

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")
        monkeypatch.setattr(sync_paths.os, "unlink", fail_displaced_unlink)
        with pytest.raises(SyncPathPartialError) as raised:
            root.replace_bytes(
                "note.md",
                b"after",
                expected=observed,
                mode=0o600,
            )

    assert raised.value.reason == "replacement_cleanup_pending"
    assert raised.value.cleanup_leaf is not None
    assert "note.md" not in repr(raised.value)
    assert target.read_bytes() == b"after"
    preserved = list(root_path.glob(".note.md.tmp-*"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"before"


def test_parent_fsync_failure_after_exchange_preserves_old_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    at_commit = False
    original_fsync = sync_paths.os.fsync

    def mark_commit(_relative_path: Path) -> None:
        nonlocal at_commit
        at_commit = True

    def fail_after_commit(descriptor: int) -> None:
        if at_commit and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory fsync failed")
        original_fsync(descriptor)

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")
        monkeypatch.setattr(root, "_before_commit", mark_commit)
        monkeypatch.setattr(sync_paths.os, "fsync", fail_after_commit)
        with pytest.raises(
            SyncPathPartialError,
            match="replacement_rollback_unverified",
        ):
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert target.read_bytes() == b"before"
    preserved = list(root_path.glob(".note.md.tmp-*"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"after"


def test_race_after_displaced_verification_restores_old_and_preserves_racer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")
        monkeypatch.setattr(
            root,
            "_after_displaced_verification",
            lambda _relative_path: target.write_bytes(b"racer"),
        )
        with pytest.raises(SyncPathPartialError) as raised:
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert raised.value.reason == "replacement_raced_after_exchange"
    assert target.read_bytes() == b"before"
    preserved = list(root_path.glob(".note.md.tmp-*"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"racer"


def test_same_root_move_is_identity_guarded(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    source = root_path / "before.md"
    source.write_bytes(b"body")

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("before.md")
        moved = root.move_file(
            "before.md",
            "nested/after.md",
            expected=observed,
        )

    assert moved.relative_path == Path("nested/after.md")
    assert moved.identity == observed.identity
    assert not source.exists()
    assert (root_path / "nested/after.md").read_bytes() == b"body"


def test_same_root_move_rejects_existing_destination(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    (root_path / "before.md").write_bytes(b"before")
    (root_path / "after.md").write_bytes(b"after")

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("before.md")
        with pytest.raises(SyncPathError, match="destination_exists"):
            root.move_file(
                "before.md",
                "after.md",
                expected=observed,
            )

    assert (root_path / "before.md").read_bytes() == b"before"
    assert (root_path / "after.md").read_bytes() == b"after"


def test_same_root_move_does_not_clobber_destination_created_at_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    source = root_path / "before.md"
    source.write_bytes(b"before")
    destination = root_path / "after.md"

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("before.md")
        monkeypatch.setattr(
            root,
            "_before_commit",
            lambda _relative_path: destination.write_bytes(b"racer"),
        )
        with pytest.raises(SyncPathError, match="destination_exists"):
            root.move_file("before.md", "after.md", expected=observed)

    assert source.read_bytes() == b"before"
    assert destination.read_bytes() == b"racer"


def test_same_root_move_closes_source_parent_when_destination_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    (root_path / "before.md").write_bytes(b"before")
    captured: list[int] = []

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("before.md")
        original = root._open_parent

        def fail_second(relative_path: Path, *, create: bool) -> int:
            if captured:
                raise SyncPathError("missing_parent", relative_path)
            descriptor = original(relative_path, create=create)
            captured.append(descriptor)
            return descriptor

        monkeypatch.setattr(root, "_open_parent", fail_second)
        with pytest.raises(SyncPathError, match="missing_parent"):
            root.move_file("before.md", "nested/after.md", expected=observed)

    with pytest.raises(OSError):
        os.fstat(captured[0])


def test_move_fsync_failure_surfaces_committed_partial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    source = root_path / "before.md"
    source.write_bytes(b"before")
    at_commit = False
    original_fsync = sync_paths.os.fsync

    def mark_commit(_relative_path: Path) -> None:
        nonlocal at_commit
        at_commit = True

    def fail_directory_fsync(descriptor: int) -> None:
        if at_commit and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory fsync failed")
        original_fsync(descriptor)

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("before.md")
        monkeypatch.setattr(root, "_before_commit", mark_commit)
        monkeypatch.setattr(sync_paths.os, "fsync", fail_directory_fsync)
        with pytest.raises(SyncPathPartialError) as raised:
            root.move_file("before.md", "after.md", expected=observed)

    assert raised.value.reason == "move_commit_unverified"
    assert raised.value.cleanup_leaf == "after.md"
    assert not source.exists()
    assert (root_path / "after.md").read_bytes() == b"before"


def test_replace_maps_raw_precommit_oserror_to_bounded_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "PRIVATE-root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")
        monkeypatch.setattr(
            root,
            "_existing_target",
            lambda *_args: (_ for _ in ()).throw(OSError("PRIVATE path")),
        )
        with pytest.raises(SyncPathError) as raised:
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert raised.value.reason == "operation_failed"
    assert raised.value.__cause__ is None
    assert target.read_bytes() == b"before"


def test_replace_raw_postcommit_error_is_distinct_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    calls = 0

    with PinnedSyncRoot(root_path) as root:
        original = root._existing_target

        def fail_after_commit(parent_fd, leaf, relative_path):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("PRIVATE post-commit path")
            return original(parent_fd, leaf, relative_path)

        monkeypatch.setattr(root, "_existing_target", fail_after_commit)
        with pytest.raises(SyncPathPartialError) as raised:
            root.replace_bytes("note.md", b"after", expected=None, mode=0o600)

    assert raised.value.reason == "replacement_commit_unverified"
    assert raised.value.__cause__ is None
    assert (root_path / "note.md").read_bytes() == b"after"


def test_replace_bounded_postcommit_refusal_is_distinct_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    calls = 0

    with PinnedSyncRoot(root_path) as root:
        original = root._existing_target

        def hide_committed_target(parent_fd, leaf, relative_path):
            nonlocal calls
            calls += 1
            if calls == 3:
                return None
            return original(parent_fd, leaf, relative_path)

        monkeypatch.setattr(root, "_existing_target", hide_committed_target)
        with pytest.raises(SyncPathPartialError) as raised:
            root.replace_bytes("note.md", b"after", expected=None, mode=0o600)

    assert raised.value.reason == "replacement_postcondition_failed"
    assert (root_path / "note.md").read_bytes() == b"after"


def test_move_bounded_postcommit_refusal_is_distinct_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    source = root_path / "before.md"
    source.write_bytes(b"before")
    calls = 0

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("before.md")
        original = root._existing_target

        def fail_committed_observation(parent_fd, leaf, relative_path):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise SyncPathError("target_identity_changed", relative_path)
            return original(parent_fd, leaf, relative_path)

        monkeypatch.setattr(root, "_existing_target", fail_committed_observation)
        with pytest.raises(SyncPathPartialError) as raised:
            root.move_file("before.md", "after.md", expected=observed)

    assert raised.value.reason == "move_postcondition_failed"
    assert not source.exists()
    assert (root_path / "after.md").read_bytes() == b"before"


def test_legacy_engine_source_has_no_pathname_sync_escape_hatches() -> None:
    source = (
        Path(sync_paths.__file__)
        .with_name("sync_engine.py")
        .read_text(encoding="utf-8")
    )

    assert ".rglob(" not in source
    assert "file_path.read_text(" not in source
    assert "file_path.parent.mkdir(" not in source
    assert "atomic_write_text(" not in source


# --------------------------------------------------------------------------
# create_new_text -- the never-replace counterpart to write_text (task-19554)
# --------------------------------------------------------------------------
def test_create_new_text_refuses_an_existing_name_without_touching_it(
    tmp_path: Path,
) -> None:
    """The property the preserved-conflict-copy path depends on.

    ``write_text`` renames over its target; for a saved copy of user text
    that is destruction. ``create_new_text`` claims the name with
    ``O_CREAT|O_EXCL`` instead, so a taken name is reported, never replaced.
    """
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md.conflict-20260821T203015Z-disk.bak"
    target.write_text("someone else's preserved copy", encoding="utf-8")

    with PinnedSyncRoot(root_path) as root:
        with pytest.raises(FileExistsError):
            root.create_new_text(target.relative_to(root_path), "mine")

    assert target.read_text(encoding="utf-8") == "someone else's preserved copy"


def test_create_new_text_writes_a_private_file_and_reports_it(
    tmp_path: Path,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()

    with PinnedSyncRoot(root_path) as root:
        result = root.create_new_text(Path("note.md.conflict-x-db.bak"), "kept")

    created = root_path / "note.md.conflict-x-db.bak"
    assert created.read_text(encoding="utf-8") == "kept"
    assert stat.S_IMODE(created.stat().st_mode) == 0o600
    assert result.absolute_path == created
    assert result.content == "kept"


def test_create_new_text_rejects_a_symlinked_name_rather_than_following_it(
    tmp_path: Path,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    outside = tmp_path / "OUTSIDE-19554-SENTINEL.bak"
    outside.write_text("OUTSIDE-19554-SENTINEL", encoding="utf-8")
    (root_path / "note.md.conflict-x-db.bak").symlink_to(outside)

    with PinnedSyncRoot(root_path) as root:
        with pytest.raises(OSError):
            root.create_new_text(Path("note.md.conflict-x-db.bak"), "leaked")

    assert outside.read_text(encoding="utf-8") == "OUTSIDE-19554-SENTINEL"


def test_create_new_text_refuses_a_missing_parent(tmp_path: Path) -> None:
    """A sidecar goes beside an existing note; it never creates directories."""
    root_path = tmp_path / "root"
    root_path.mkdir()

    with PinnedSyncRoot(root_path) as root:
        with pytest.raises(SyncPathError, match="missing_parent"):
            root.create_new_text(Path("nope/note.md.conflict-x-db.bak"), "x")

    assert not (root_path / "nope").exists()


def test_create_new_text_leaves_nothing_behind_when_it_fails_after_creating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller told the write failed must not find a half-written file."""
    root_path = tmp_path / "root"
    root_path.mkdir()

    def explode(_file_fd: int, _content: bytes) -> None:
        raise OSError("short write")

    with PinnedSyncRoot(root_path) as root:
        monkeypatch.setattr(root, "_write_all", explode)
        with pytest.raises(OSError, match="short write"):
            root.create_new_text(Path("note.md.conflict-x-db.bak"), "partial")

    assert list(root_path.iterdir()) == []
