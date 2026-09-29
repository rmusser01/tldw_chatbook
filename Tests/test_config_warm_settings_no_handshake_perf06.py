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

import pytest

import tldw_chatbook.config as config_module
from tldw_chatbook.Backup_Recovery import config_participants

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def operation_counter(monkeypatch):
    """Count entries into the admission handshake without disabling it."""
    real_operation = config_participants.operation
    calls = {"n": 0}

    @contextlib.contextmanager
    def counting_operation(*args, **kwargs):
        calls["n"] += 1
        with real_operation(*args, **kwargs):
            yield

    monkeypatch.setattr(config_participants, "operation", counting_operation)
    return calls


def _warm_settings():
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


def test_a_warm_load_settings_does_not_enter_the_handshake(operation_counter):
    """Repeated warm ``load_settings()`` calls take no admission and open no files."""
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


def test_a_warm_runtime_snapshot_does_not_enter_the_handshake(operation_counter):
    """A warm snapshot is still a defensive copy, without admission."""
    _warm_settings()
    operation_counter["n"] = 0

    first = config_module.get_runtime_config_snapshot()
    second = config_module.get_runtime_config_snapshot()

    assert operation_counter["n"] == 0, (
        f"2 warm snapshots entered the handshake {operation_counter['n']} time(s)"
    )
    assert first.values == second.values
    assert first.values is not second.values, "the snapshot must stay a copy"


def test_a_forced_reload_still_rebuilds():
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


def test_the_miss_paths_stay_guarded():
    """Structural pin: only the warm hit bypasses the guarded bodies."""
    module_src = inspect.getsource(config_module)
    for guarded_body in (
        "@_config_participants.guarded\ndef _load_settings_guarded(",
        "@_config_participants.guarded\ndef _load_settings_uncached(",
        "@_config_participants.guarded\ndef _get_runtime_config_snapshot_guarded(",
    ):
        assert guarded_body in module_src, f"missing guard: {guarded_body!r}"



def test_a_warm_load_settings_does_not_stall_behind_an_in_flight_write():
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
