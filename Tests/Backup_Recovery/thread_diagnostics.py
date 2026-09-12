"""Bounded test diagnostics using Python-owned frames, without a C watchdog."""

from __future__ import annotations

import faulthandler
import json
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path


def _frames(frame, limit: int = 64) -> list[dict]:
    """Copy code metadata while Python retains every traversed frame."""
    result = []
    while frame is not None and len(result) < limit:
        result.append(
            {
                "file": Path(frame.f_code.co_filename).name[:128],
                "function": frame.f_code.co_name[:128],
                "line": frame.f_lineno,
            }
        )
        frame = frame.f_back
    return result


def _write(path: Path, records: list) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(records, output)


def _snapshot() -> list[dict]:
    current = sys._current_frames()
    return [
        {"thread": identifier, "frames": _frames(frame)}
        for identifier, frame in sorted(current.items())[:32]
    ]


def snapshot_threads(path: Path) -> None:
    """Write an immediate bounded snapshot without reading source or locals."""
    _write(Path(path), [_snapshot()])


def observe_threads(path: Path, *, interval: float = 60) -> Callable[[], None]:
    """Sample bounded metadata on a Python thread; retain native fatal reporting.

    The C dump_traceback_later watchdog can race active interpreter frames
    (CPython gh-140815). sys._current_frames returns owned Python frame objects.
    This observer never records source lines, local variables or exception text.
    """
    if interval <= 0:
        raise ValueError("positive_diagnostic_interval_required")
    path = Path(path)
    fatal = path.with_name(path.stem + "-fatal.log")
    descriptor = os.open(fatal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    fatal_output = os.fdopen(descriptor, "w", encoding="utf-8")
    was_enabled = faulthandler.is_enabled()
    faulthandler.enable(file=fatal_output)
    stopping = threading.Event()
    records, failures = [], []

    def sample():
        try:
            while not stopping.wait(interval):
                records.append(_snapshot())
                del records[:-4]
                _write(path, records)
        except Exception as error:  # noqa: BLE001 - stop() fails the test for observer errors.
            failures.append(type(error).__name__)

    worker = threading.Thread(
        target=sample, name="test-thread-diagnostics", daemon=True
    )
    worker.start()

    def stop():
        stopping.set()
        worker.join(timeout=5)
        if worker.is_alive():
            raise RuntimeError("thread_diagnostic_stop_failed")
        if was_enabled:
            faulthandler.enable()
        else:
            faulthandler.disable()
        fatal_output.close()
        if failures:
            raise RuntimeError("thread_diagnostic_failed:" + failures[0])

    return stop


def observe_recovery_failures(path: Path) -> Callable[[], None]:
    """Record only bounded original error metadata, then use the real mapper."""
    from tldw_chatbook.Backup_Recovery import recovery_service

    original = recovery_service.issue_code
    service_class = recovery_service.RecoveryService
    original_static = service_class.__dict__["issue_code"]
    records, lock = [], threading.Lock()

    def observed(error, *, kind=""):
        record = {"error_class": type(error).__name__[:80], "frames": []}
        for key in ("errno", "winerror"):
            value = getattr(error, key, None)
            record[key] = value if type(value) is int else None
        trace = error.__traceback__
        while trace is not None and len(record["frames"]) < 64:
            record["frames"].append(
                {
                    "file": Path(trace.tb_frame.f_code.co_filename).name[:128],
                    "function": trace.tb_frame.f_code.co_name[:128],
                    "line": trace.tb_lineno,
                }
            )
            trace = trace.tb_next
        with lock:
            records.append(record)
            del records[:-16]
            _write(Path(path), records)
        return original(error, kind=kind)

    recovery_service.issue_code = observed
    service_class.issue_code = staticmethod(observed)

    def stop():
        recovery_service.issue_code = original
        service_class.issue_code = original_static

    return stop
