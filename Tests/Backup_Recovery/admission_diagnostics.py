"""Test-only inclusive native/admission timing; never retain paths or SID values."""

from __future__ import annotations

import copy
import functools
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from Tests.Backup_Recovery.thread_diagnostics import _write


def observe_admission(path: Path, *, interval: float = 5) -> Callable[[], None]:
    """Aggregate real call durations and root counts without changing outcomes."""
    from tldw_chatbook.Backup_Recovery.admission import Admission
    from tldw_chatbook.Utils.windows_files import _Native

    return _observe(
        Path(path),
        [(Admission, name) for name in ("_groups", "_tokens")]
        + [
            (_Native, name)
            for name in ("open_handle", "ntfs", "security", "sid_string")
        ],
        interval=interval,
    )


def _observe(
    path: Path, targets: list[tuple[type, str]], *, interval: float
) -> Callable[[], None]:
    if interval <= 0:
        raise ValueError("positive_diagnostic_interval_required")
    lock, stopping = threading.Lock(), threading.Event()
    local = threading.local()
    metrics, groups, originals, failures = {}, [], [], []

    def instrument(owner, name):
        original = owner.__dict__[name]
        originals.append((owner, name, original))
        row = metrics[name] = {
            "started": 0,
            "completed": 0,
            "errors": 0,
            "wall_ns": 0,
            "max_ns": 0,
        }

        @functools.wraps(original)
        def measured(*args, **kwargs):
            detail = None
            if name == "_groups":
                try:
                    registry = args[1] if len(args) > 1 else kwargs["registry"]
                    roots = [
                        root
                        for entry in registry.entries.values()
                        for root in entry.roots + entry.proposed
                    ]
                    depths = [len(Path(root).parts) for root in roots]
                    detail = {
                        "namespaces": len(registry.entries),
                        "total_roots": len(roots),
                        "unique_roots": len(set(roots)),
                        "total_depth": sum(depths),
                        "max_depth": max(depths, default=0),
                        "unique_ancestors": len(
                            {
                                str(parent)
                                for root in roots
                                for parent in Path(root).parents
                            }
                        ),
                    }
                except (AttributeError, KeyError, TypeError):
                    detail = {"metadata_unavailable": 1}
                frame = sys._getframe(1)
                detail.update(
                    caller={
                        "file": Path(frame.f_code.co_filename).name[:128],
                        "function": frame.f_code.co_name[:128],
                        "line": frame.f_lineno,
                    },
                    completed=0,
                    errors=0,
                    wall_ns=0,
                    thread_cpu_ns=0,
                    process_cpu_ns=0,
                    calls={},
                )
                del frame
            started = time.perf_counter_ns()
            previous = getattr(local, "group", None)
            if detail is not None:
                thread_started, process_started = (
                    time.thread_time_ns(),
                    time.process_time_ns(),
                )
                detail["started_wall_ns"] = started
                local.group = detail
            active = getattr(local, "group", None)
            with lock:
                row["started"] += 1
                if detail is not None:
                    groups.append(detail)
                    del groups[:-16]
                elif active is not None:
                    scoped = active["calls"].setdefault(
                        name,
                        {
                            "started": 0,
                            "completed": 0,
                            "errors": 0,
                            "wall_ns": 0,
                        },
                    )
                    scoped["started"] += 1
            failed = 0
            try:
                return original(*args, **kwargs)
            except BaseException:
                failed = 1
                raise
            finally:
                elapsed = time.perf_counter_ns() - started
                if detail is not None:
                    thread_elapsed = time.thread_time_ns() - thread_started
                    process_elapsed = time.process_time_ns() - process_started
                    local.group = previous
                with lock:
                    row["completed"] += 1
                    row["errors"] += failed
                    row["wall_ns"] += elapsed
                    row["max_ns"] = max(row["max_ns"], elapsed)
                    if detail is not None:
                        detail.update(
                            completed=1,
                            errors=failed,
                            wall_ns=elapsed,
                            thread_cpu_ns=thread_elapsed,
                            process_cpu_ns=process_elapsed,
                        )
                    elif active is not None:
                        scoped["completed"] += 1
                        scoped["errors"] += failed
                        scoped["wall_ns"] += elapsed

        setattr(owner, name, measured)

    def write():
        with lock:
            snapshot = copy.deepcopy(
                {"inclusive_wall_times": 1, "calls": metrics, "groups": groups}
            )
        for group in snapshot["groups"]:
            if not group["completed"]:
                group["wall_ns"] = time.perf_counter_ns() - group["started_wall_ns"]
        _write(path, snapshot)

    def sample():
        try:
            while not stopping.wait(interval):
                write()
        except Exception as error:  # noqa: BLE001 - stop() reports observer failure.
            failures.append(type(error).__name__)

    for owner, name in targets:
        instrument(owner, name)
    write()
    worker = threading.Thread(target=sample, name="test-admission-timing", daemon=True)
    worker.start()

    def stop():
        stopping.set()
        worker.join(timeout=5)
        for owner, name, original in originals:
            setattr(owner, name, original)
        if worker.is_alive():
            raise RuntimeError("admission_diagnostic_stop_failed")
        write()
        if failures:
            raise RuntimeError("admission_diagnostic_failed:" + failures[0])

    return stop
