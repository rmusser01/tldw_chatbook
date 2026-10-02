"""PERF-06 (TASK-33265): warm settings reads must not enter the admission handshake.

TASK-32804.1 served warm ``get_cli_setting`` reads ahead of the ADR-126
admission handshake, but it covered 1 of 16 guarded entry points. Warm
``load_settings()`` and ``get_runtime_config_snapshot()`` still ran the full
handshake (~650 ``open()`` calls, 9-20 ms) on every call -- the 4 Hz Console
credential poll, 1-3 times per composer keystroke, ~120 times per Console
visit (2026-09-27 audit). A warm hit is a pure in-memory read; the handshake
protects the file/build lifecycle, which the miss path still enters.

Deterministic: a call counter on ``config_participants.operation``, not timing.
"""

from __future__ import annotations

import contextlib
import inspect
import os
from pathlib import Path

import pytest

import tldw_chatbook.config as config_module
from tldw_chatbook.Backup_Recovery import config_participants

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def operation_counter(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count entries into the admission handshake without disabling it.

    Args:
        monkeypatch: Wraps ``config_participants.operation`` with a counter.

    Returns:
        A mutable ``{"n": entries}`` mapping; tests reset ``n`` after warming
        the cache and then assert on it.
    """
    # Warm first: a cold-cache rebuild must run the real ``operation`` -- a
    # wrapped one fails the raw-participant provenance checks (Qodo, #2903).
    _warm_settings()
    real_operation = config_participants.operation
    calls = {"n": 0}

    @contextlib.contextmanager
    def counting_operation(*args, **kwargs):
        calls["n"] += 1
        with real_operation(*args, **kwargs):
            yield

    monkeypatch.setattr(config_participants, "operation", counting_operation)
    return calls


def _warm_settings() -> None:
    """Warm the settings cache, or skip where admission is unbound."""
    from tldw_chatbook.Backup_Recovery import bootstrap

    try:
        config_module.load_settings()
    except bootstrap.RecoveryRequired as exc:
        pytest.skip(f"settings cache cannot be warmed in this environment: {exc}")


#: ``os.open`` raises the ``open`` audit event with ``mode=None``. Counted via
#: an audit hook because replacing ``os.open`` itself trips the raw-participant
#: ``os.open in os.supports_dir_fd`` check (see lessons-testing-evidence.md).
_OS_OPENS: dict[str, object] = {"installed": False, "count": None}


def _count_os_open(event: str, args: tuple) -> None:
    counter = _OS_OPENS["count"]
    if counter is not None and event == "open" and args[1] is None:
        counter[0] += 1  # type: ignore[index]


def test_a_warm_load_settings_does_not_enter_the_handshake(
    operation_counter: dict[str, int],
) -> None:
    """Repeated warm ``load_settings()`` calls take no admission and open no files.

    Args:
        operation_counter: Admission-handshake entry counter (fixture).
    """
    import sys

    _warm_settings()
    operation_counter["n"] = 0
    if not _OS_OPENS["installed"]:
        sys.addaudithook(_count_os_open)  # permanent; inert while count is None
        _OS_OPENS["installed"] = True
    opens = [0]
    _OS_OPENS["count"] = opens
    try:
        results = [config_module.load_settings() for _ in range(10)]
    finally:
        _OS_OPENS["count"] = None

    assert operation_counter["n"] == 0, (
        f"10 warm load_settings() calls entered the handshake "
        f"{operation_counter['n']} time(s)"
    )
    assert opens[0] == 0, f"10 warm load_settings() calls made {opens[0]} os.open call(s)"
    assert all(result is results[0] for result in results)


def test_a_warm_runtime_snapshot_does_not_enter_the_handshake(
    operation_counter: dict[str, int],
) -> None:
    """A warm snapshot is still a defensive copy, without admission.

    Args:
        operation_counter: Admission-handshake entry counter (fixture).
    """
    _warm_settings()
    operation_counter["n"] = 0

    first = config_module.get_runtime_config_snapshot()
    second = config_module.get_runtime_config_snapshot()

    assert operation_counter["n"] == 0, (
        f"2 warm snapshots entered the handshake {operation_counter['n']} time(s)"
    )
    assert first.values == second.values
    assert first.values is not second.values, "the snapshot must stay a copy"


def test_a_forced_reload_still_rebuilds() -> None:
    """The handshake is amortised, not removed: a forced reload rebuilds.

    Counting ``operation`` entries on a real rebuild is not possible: a
    wrapped ``operation`` fails the raw-participant provenance checks the
    rebuild's file I/O performs. So this pins that the miss path really
    rebuilds (a new settings object), and ``test_the_miss_paths_stay_guarded``
    pins that the rebuild bodies carry the guard.
    """
    _warm_settings()
    warm = config_module.load_settings()

    rebuilt = config_module.load_settings(force_reload=True)

    assert rebuilt is not warm
    assert config_module.load_settings() is rebuilt


def test_the_miss_paths_stay_guarded() -> None:
    """Structural pin: only the warm hit bypasses the guarded bodies."""
    module_src = inspect.getsource(config_module)
    for guarded_body in (
        "@_config_participants.guarded\ndef _load_settings_guarded(",
        "@_config_participants.guarded\ndef _load_settings_uncached(",
        "@_config_participants.guarded\ndef _get_runtime_config_snapshot_guarded(",
    ):
        assert guarded_body in module_src, f"missing guard: {guarded_body!r}"



def test_a_warm_load_settings_does_not_stall_behind_an_in_flight_write() -> None:
    """A warm read stays lock-free while a config write holds its locks.

    TASK-21124's contract, extended to ``load_settings``: a write holds the
    settings rebuild and config-file locks through fsyncs and TOML parses,
    and the 4 Hz Console poll and composer keystrokes read settings on the
    event loop. A read overlapping a write may return the pre-write
    settings; the writer invalidates the cache before it returns.
    """
    import threading

    _warm_settings()
    finished = threading.Event()

    def read() -> None:
        config_module.load_settings()
        finished.set()

    with config_module._settings_rebuild_lock(), config_module._config_file_lock():
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        assert finished.wait(5.0), "the warm read stalled behind the write locks"
    reader.join(5.0)


def test_a_warm_snapshot_copies_the_hit_it_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The warm snapshot never looks the cache up twice (Qodo, #2903).

    A second lookup through ``load_settings`` could miss after a lock-free
    cache clear and run the guarded rebuild while REBUILD and FILE are held.

    Args:
        monkeypatch: Replaces ``load_settings`` with a tripwire.
    """
    _warm_settings()

    def tripwire(*args: object, **kwargs: object) -> dict:
        raise AssertionError("the warm snapshot looked the cache up a second time")

    monkeypatch.setattr(config_module, "load_settings", tripwire)
    snapshot = config_module.get_runtime_config_snapshot()
    assert snapshot.values


def test_a_warm_snapshot_never_blocks_behind_a_held_config_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A contended lock sends the snapshot to the guarded, pause-aware path.

    Blocking here would ignore a recovery pause the guarded wait observes
    (Qodo, #2903).

    Args:
        monkeypatch: Replaces the guarded snapshot with a marker.
    """
    import threading

    _warm_settings()
    marker = object()
    monkeypatch.setattr(
        config_module,
        "_get_runtime_config_snapshot_guarded",
        lambda **kwargs: marker,
    )
    for lock in (config_module._settings_rebuild_lock(), config_module._config_file_lock()):
        held, release = threading.Event(), threading.Event()

        def hold(lock=lock) -> None:
            with lock:
                held.set()
                release.wait(10)

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        assert held.wait(10)
        result: list[object] = []
        reader = threading.Thread(
            target=lambda: result.append(config_module.get_runtime_config_snapshot()),
            daemon=True,
        )
        reader.start()
        reader.join(5)
        blocked = reader.is_alive()
        release.set()
        holder.join(10)
        reader.join(10)
        assert not blocked, "the warm snapshot blocked behind a held lock"
        assert result == [marker]


@pytest.mark.parametrize("change", ["replaced", "symlinked", "parent-symlinked"])
def test_a_replaced_config_file_is_not_served_warm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: str
) -> None:
    """A warm hit re-checks the identity of the config path (Qodo, #2903).

    Before PERF-06 every read ran the guarded checks, which reject a config
    file swapped for another inode or a symlink, and a symlinked parent. A
    warm hit now compares an ``lstat`` stamp of every path component, so any
    such swap is a miss that reaches them.

    Args:
        monkeypatch: Replaces the guarded rebuild with a marker.
        tmp_path: pytest fixture; holds the swapped-out originals.
        change: What is swapped.
    """
    _warm_settings()
    path = config_module._get_effective_config_path()
    if not path.exists():
        pytest.skip("no config file to swap in this environment")
    marker: dict = {"guarded": True}
    monkeypatch.setattr(config_module, "_load_settings_guarded", lambda **kwargs: marker)
    assert config_module.load_settings() is not marker  # still warm

    original = path.read_bytes()
    # Everything the swap needs exists before the original moves, and the
    # finally block restores it whatever fails in between.
    target = tmp_path / "elsewhere.toml"
    target.write_bytes(original)
    if change == "parent-symlinked":
        moved, restore = path.parent, tmp_path / "moved-parent"
        os.replace(moved, restore)
        try:
            moved.symlink_to(restore, target_is_directory=True)
            assert (moved / path.name).read_bytes() == original  # same file, new route
            assert config_module.load_settings() is marker
        finally:
            if moved.is_symlink():
                moved.unlink()
            os.replace(restore, moved)
        return
    backup = tmp_path / "original.toml"
    os.replace(path, backup)
    try:
        if change == "replaced":
            os.replace(target, path)
        else:
            path.symlink_to(target)
        assert config_module.load_settings() is marker
    finally:
        if path.is_symlink() or path.exists():
            path.unlink()
        os.replace(backup, path)
