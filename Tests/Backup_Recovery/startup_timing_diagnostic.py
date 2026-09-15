"""Compare one unchanged full-app test with bounded main-thread timing output.

This optional CI diagnostic uses cProfile, so its timings include profiling
overhead. It records code coordinates and aggregate times, never call values.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import sys
import threading
import time
from pathlib import Path


class ConfigTimings:
    """Observe finite code objects without wrapping config owners or reading values."""

    tool_id = 3

    def __init__(self, codes):
        self.codes = codes
        self.clock = time.perf_counter
        self.lock = threading.Lock()
        self.active = {}
        self.completed = []
        self.slowest = {}
        self.counts = {}
        self.errors = []
        self.dropped_active = 0
        self.owned = False

    def _error(self, error):
        self.errors.append(type(error).__name__)
        del self.errors[:-4]

    def _event(self, code, event, frame):
        if code not in self.codes:
            return
        try:
            now = self.clock()
            with self.lock:
                key = id(frame)
                if event == "started":
                    label = self.codes[code]
                    self.counts[label] = self.counts.get(label, 0) + 1
                    if len(self.active) >= 32:
                        self.dropped_active += 1
                        return
                    self.active[key] = {
                        "function": label, "started": now,
                        "cpu_started": time.thread_time(), "acquired": None,
                        "thread": threading.get_native_id(),
                        "thread_ident": threading.get_ident(),
                    }
                elif key in self.active:
                    row = self.active[key]
                    if event == "yielded":
                        if row["function"] in {"config_lock", "config_operation"} and row["acquired"] is None:
                            row["acquired"] = now
                    else:
                        self.active.pop(key)
                        done = {
                            "function": row["function"], "outcome": event,
                            "thread": row["thread"],
                            "elapsed_seconds": round(now - row["started"], 6),
                            "thread_cpu_seconds": round(time.thread_time() - row["cpu_started"], 6),
                        }
                        if row["acquired"] is not None:
                            # Entry includes native admission and all lock setup;
                            # exit includes body work and native/lock retirement.
                            # Neither number measures mutex contention alone.
                            done["acquisition_seconds"] = round(row["acquired"] - row["started"], 6)
                            done["body_and_release_seconds"] = round(now - row["acquired"], 6)
                        self.completed.append(done)
                        del self.completed[:-16]
                        old = self.slowest.get(row["function"])
                        if old is None or done["elapsed_seconds"] > old["elapsed_seconds"]:
                            self.slowest[row["function"]] = done
        except Exception as error:  # noqa: BLE001 - optional observation must preserve execution.
            self._error(error)

    def start(self):
        """Use an unclaimed monitoring slot; never replace another tool."""
        monitoring = sys.monitoring
        events = monitoring.events

        def started(code, _offset):
            self._event(code, "started", sys._getframe(1))

        def returned(code, _offset, _value):
            self._event(code, "returned", sys._getframe(1))

        def yielded(code, _offset, _value):
            self._event(code, "yielded", sys._getframe(1))

        def raised(code, _offset, _error):
            self._event(code, "raised", sys._getframe(1))

        try:
            monitoring.use_tool_id(self.tool_id, "backup-startup-config")
            self.owned = True
            for event, callback in (
                (events.PY_START, started), (events.PY_RETURN, returned),
                (events.PY_YIELD, yielded), (events.PY_UNWIND, raised),
            ):
                monitoring.register_callback(self.tool_id, event, callback)
            for code in self.codes:
                monitoring.set_local_events(
                    self.tool_id, code, events.PY_START | events.PY_RETURN | events.PY_YIELD
                )
            # Python 3.12 exposes unwind only globally; the callback immediately
            # ignores code outside the finite selection and never reads its error.
            monitoring.set_events(self.tool_id, events.PY_UNWIND)
        except Exception as error:  # noqa: BLE001 - test results remain authoritative.
            self._error(error)
            self.close()

    def snapshot(self):
        """Return bounded durations; ongoing CPU time is deliberately unknown."""
        now = time.perf_counter()
        with self.lock:
            active = list(self.active.values())
            result = {
                "active": [
                    {
                        "function": row["function"],
                        "thread": row["thread"],
                        "phase": "body_and_release" if row["acquired"] is not None else "running",
                        "elapsed_seconds": round(now - row["started"], 6),
                    }
                    for row in active
                ],
                "completed": list(self.completed), "counts": dict(self.counts),
                "slowest": dict(self.slowest),
                "dropped_active": self.dropped_active, "errors": list(self.errors),
            }
        # These are live code coordinates, not an atomic thread/lock snapshot.
        # Do not retain frames or inspect arguments, locals, errors or results.
        try:
            frames = sys._current_frames()
            for source, row in zip(active, result["active"], strict=True):
                frame = frames.get(source["thread_ident"])
                row["frames"] = coordinates = []
                while frame is not None and len(coordinates) < 25:
                    coordinates.append({
                        "file": Path(frame.f_code.co_filename).name[:128],
                        "function": frame.f_code.co_name[:128],
                        "line": frame.f_lineno,
                    })
                    frame = frame.f_back
        except Exception as error:  # noqa: BLE001 - optional code coordinates only.
            self._error(error)
        finally:
            frame = None
            frames = None
        result["errors"] = list(self.errors)
        return result

    def close(self):
        """Disable every owned event and callback before releasing the slot."""
        if not self.owned:
            return
        try:
            self._close_owned()
        except Exception as error:  # noqa: BLE001 - keep ownership for a cleanup retry.
            self._error(error)

    def _close_owned(self):
        monitoring = sys.monitoring
        events = monitoring.events
        monitoring.set_events(self.tool_id, 0)
        for code in self.codes:
            monitoring.set_local_events(self.tool_id, code, 0)
        for event in (events.PY_START, events.PY_RETURN, events.PY_YIELD, events.PY_UNWIND):
            monitoring.register_callback(self.tool_id, event, None)
        monitoring.free_tool_id(self.tool_id)
        self.owned = False


class ConfigTimingPlugin:
    """Attach after collection so selected config imports keep their original order."""

    def __init__(self, source):
        self.source = source
        self.timings = ConfigTimings({})

    def pytest_collection_finish(self):
        import inspect

        try:
            config = sys.modules.get("tldw_chatbook.config")
            if config is None:
                return
            names = {
                "_config_write_lock": "config_lock",
                "_apply_literal_settings_transaction_locked": "transaction",
                "_publish_runtime_config_unlocked": "publication",
                "_load_settings_uncached": "settings_rebuild",
                "get_user_data_dir": "user_directory",
            }
            for name, label in names.items():
                code = inspect.unwrap(getattr(config, name)).__code__
                if Path(code.co_filename).resolve() != self.source / "tldw_chatbook/config.py":
                    raise ValueError("config_timing_source_mismatch")
                self.timings.codes[code] = label
            participants = getattr(config, "_config_participants", None)
            if participants is not None:
                code = inspect.unwrap(participants.operation).__code__
                expected = self.source / "tldw_chatbook/Backup_Recovery/config_participants.py"
                if Path(code.co_filename).resolve() != expected:
                    raise ValueError("config_operation_timing_source_mismatch")
                self.timings.codes[code] = "config_operation"
            self.timings.start()
        except Exception as error:  # noqa: BLE001 - observation must not replace collection.
            self.timings._error(error)
            self.timings.close()


def snapshot(profile: cProfile.Profile, source: Path, label: str) -> dict:
    """Return the largest observed product costs without source or local values."""
    rows = []
    package = source / "tldw_chatbook"
    for entry in profile.getstats():
        code = entry.code
        if isinstance(code, str):
            continue
        try:
            relative = Path(code.co_filename).relative_to(package)
        except ValueError:
            continue
        rows.append(
            {
                "file": relative.as_posix(),
                "function": code.co_name,
                "line": code.co_firstlineno,
                "calls": entry.callcount,
                "total_seconds": round(entry.totaltime, 6),
                "self_seconds": round(entry.inlinetime, 6),
            }
        )
    module = sys.modules.get("tldw_chatbook")
    loaded = None if module is None else Path(module.__file__).resolve()
    return {
        "diagnostic": "backup_startup_profile",
        "label": label,
        "source_matches": None if loaded is None else loaded.is_relative_to(package),
        "largest_total": sorted(
            rows, key=lambda row: row["total_seconds"], reverse=True
        )[:20],
        "largest_self": sorted(rows, key=lambda row: row["self_seconds"], reverse=True)[
            :20
        ],
    }


def main() -> int:
    """Run one actual case with its existing 60-second ceiling in a fresh process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--label", choices=("dev", "candidate"), required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    os.chdir(source)
    sys.path[0] = str(source)
    os.environ["TLDW_TEST_PRIVATE_PROFILE_NODE"] = args.node
    import pytest

    profile = cProfile.Profile()
    stopping = threading.Event()
    started = time.monotonic()
    failures = []
    config_timing = ConfigTimingPlugin(source)

    def emit():
        try:
            record = snapshot(profile, source, args.label)
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            record["observer_errors"] = list(failures)
            record["config_operations"] = config_timing.timings.snapshot()
            print(json.dumps(record), file=sys.stderr, flush=True)
        except Exception as error:  # noqa: BLE001 - optional output cannot replace the observed test outcome.
            failures.append(type(error).__name__)
            del failures[:-4]

    def sample():
        while not stopping.wait(10):
            emit()

    observer = threading.Thread(
        target=sample, name="backup-startup-profile", daemon=True
    )
    profile.enable()
    try:
        try:
            observer.start()
        except RuntimeError as error:
            failures.append(type(error).__name__)
        return pytest.main(
            [args.node, "--timeout=60", "-q", "--capture=no"], plugins=[config_timing]
        )
    finally:
        config_timing.timings.close()
        profile.disable()
        stopping.set()
        if observer.ident is not None:
            observer.join(timeout=5)
        emit()


if __name__ == "__main__":
    raise SystemExit(main())
