"""Test-only inclusive native/admission timing; never retain paths or SID values."""

from __future__ import annotations

import copy
import functools
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
            started = time.perf_counter_ns()
            with lock:
                row["started"] += 1
                if detail is not None:
                    groups.append(detail)
                    del groups[:-16]
            failed = 0
            try:
                return original(*args, **kwargs)
            except BaseException:
                failed = 1
                raise
            finally:
                elapsed = time.perf_counter_ns() - started
                with lock:
                    row["completed"] += 1
                    row["errors"] += failed
                    row["wall_ns"] += elapsed
                    row["max_ns"] = max(row["max_ns"], elapsed)

        setattr(owner, name, measured)

    def write():
        with lock:
            snapshot = copy.deepcopy(
                {"inclusive_wall_times": 1, "calls": metrics, "groups": groups}
            )
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
