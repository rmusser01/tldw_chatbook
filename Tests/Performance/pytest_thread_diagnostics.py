"""Opt-in RSS and project-thread diagnostics for bounded pytest investigations.

Load with ``-p Tests.Performance.pytest_thread_diagnostics`` and provide
``--thread-diagnostics-jsonl=PATH``. The default test suite is unaffected.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Iterable

import psutil
import pytest


_PROJECT_THREAD_PREFIXES = {"fleet-": "fleet", "tool-": "tool"}
_STATE_ATTR = "_tldw_thread_diagnostics_state"


def _project_threads(
    threads: Iterable[threading.Thread] | None = None,
) -> tuple[threading.Thread, ...]:
    """Return the project-owned thread objects from an inventory."""
    candidates = threading.enumerate() if threads is None else threads
    return tuple(
        thread
        for thread in candidates
        if any(thread.name.startswith(prefix) for prefix in _PROJECT_THREAD_PREFIXES)
    )


def project_thread_snapshots(
    threads: Iterable[threading.Thread] | None = None,
) -> tuple[dict[str, object], ...]:
    """Return stable metadata for AgentService-owned threads.

    Args:
        threads: Threads to classify. Defaults to the live process inventory.

    Returns:
        Sorted thread metadata for the ``fleet-*`` and ``tool-*`` owners.
    """
    rows: list[dict[str, object]] = []
    for thread in _project_threads(threads):
        owner = next(
            (
                owner
                for prefix, owner in _PROJECT_THREAD_PREFIXES.items()
                if thread.name.startswith(prefix)
            ),
            None,
        )
        if owner is None:
            continue
        rows.append(
            {
                "daemon": thread.daemon,
                "ident": thread.ident,
                "name": thread.name,
                "object_id": id(thread),
                "owner": owner,
            }
        )
    return tuple(sorted(rows, key=lambda row: (str(row["owner"]), str(row["name"]))))


def wait_for_project_thread_baseline(
    *,
    baseline_threads: frozenset[threading.Thread],
    timeout_seconds: float,
    poll_seconds: float = 0.05,
    quiescence_seconds: float = 0.1,
) -> tuple[dict[str, object], ...]:
    """Wait briefly for newly created project threads to leave the process.

    Args:
        baseline_threads: Project-owned thread objects present before the run.
        timeout_seconds: Maximum cleanup wait.
        poll_seconds: Delay between inventories.
        quiescence_seconds: Required continuously empty observation window.

    Returns:
        Project-owned threads still alive after the bounded wait.
    """
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    quiet_since: float | None = None
    while True:
        now = time.monotonic()
        survivor_threads = tuple(
            thread for thread in _project_threads() if thread not in baseline_threads
        )
        survivors = project_thread_snapshots(survivor_threads)
        if survivors:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = now
        elif now - quiet_since >= max(quiescence_seconds, 0.0):
            return ()
        if now >= deadline:
            return survivors
        time.sleep(min(max(poll_seconds, 0.001), max(deadline - now, 0.0)))


def first_unattributed_survivors(
    survivors: tuple[dict[str, object], ...],
    attributed_object_ids: set[int],
) -> tuple[dict[str, object], ...]:
    """Return survivors not already blamed on an earlier test."""
    first: list[dict[str, object]] = []
    for row in survivors:
        object_id = int(row["object_id"])
        if object_id in attributed_object_ids:
            continue
        attributed_object_ids.add(object_id)
        first.append(row)
    return tuple(first)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register diagnostics options without changing default test behavior."""
    group = parser.getgroup("tldw-thread-diagnostics")
    group.addoption(
        "--thread-diagnostics-jsonl",
        default=None,
        help="Write periodic RSS/current-test/project-thread samples as JSONL.",
    )
    group.addoption(
        "--thread-diagnostics-interval",
        type=float,
        default=0.5,
        help="Seconds between diagnostic samples (default: 0.5).",
    )
    group.addoption(
        "--thread-diagnostics-settle",
        type=float,
        default=3.0,
        help="Seconds to wait after teardown for project threads (default: 3).",
    )
    group.addoption(
        "--thread-diagnostics-stack-after",
        type=float,
        default=20.0,
        help="Dump Python stacks after one test phase remains active this long.",
    )
    group.addoption(
        "--thread-diagnostics-strict",
        action="store_true",
        help="Fail the pytest session if project threads survive teardown.",
    )


class _DiagnosticState:
    def __init__(self, config: pytest.Config, path: Path) -> None:
        self.path = path
        self.interval = max(config.getoption("thread_diagnostics_interval"), 0.05)
        self.settle = max(config.getoption("thread_diagnostics_settle"), 0.0)
        self.stack_after = max(config.getoption("thread_diagnostics_stack_after"), 0.1)
        self.strict = bool(config.getoption("thread_diagnostics_strict"))
        self.process = psutil.Process(os.getpid())
        self.baseline_threads = frozenset(_project_threads())
        self.current_test: str | None = None
        self.current_phase = "session"
        self.current_since = time.monotonic()
        self.started_at = self.current_since
        self.stack_emitted_for: tuple[str | None, str] | None = None
        self.failures: list[dict[str, object]] = []
        self._attributed_object_ids: set[int] = set()
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._monitor = threading.Thread(
            target=self._monitor_loop,
            name="pytest-thread-diagnostics",
            daemon=True,
        )

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.emit("session_start")
        self._monitor.start()

    def set_phase(self, nodeid: str | None, phase: str) -> None:
        with self._lock:
            identity = (nodeid, phase)
            if identity != (self.current_test, self.current_phase):
                self.current_test = nodeid
                self.current_phase = phase
                self.current_since = time.monotonic()
                self.stack_emitted_for = None

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            nodeid = self.current_test
            phase = self.current_phase
            phase_elapsed = time.monotonic() - self.current_since
        project_threads = project_thread_snapshots()
        return {
            "elapsed_seconds": round(time.monotonic() - self.started_at, 6),
            "phase": phase,
            "phase_elapsed_seconds": round(phase_elapsed, 6),
            "project_thread_count": len(project_threads),
            "project_threads": project_threads,
            "rss_bytes": self.process.memory_info().rss,
            "test": nodeid,
            "total_thread_count": len(threading.enumerate()),
        }

    def emit(self, event: str, **extra: object) -> None:
        if self._finished.is_set() and event != "session_finish":
            return
        record = {"event": event, **self.snapshot(), **extra}
        encoded = json.dumps(record, sort_keys=True)
        with self._write_lock:
            if self._finished.is_set() and event != "session_finish":
                return
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()

    def record_teardown(self, nodeid: str) -> None:
        settle_started = time.monotonic()
        survivors = wait_for_project_thread_baseline(
            baseline_threads=self.baseline_threads,
            timeout_seconds=self.settle,
        )
        settle_seconds = time.monotonic() - settle_started
        first_survivors = first_unattributed_survivors(
            survivors,
            self._attributed_object_ids,
        )
        self.emit(
            "test_teardown",
            test=nodeid,
            ownership_settle_seconds=round(settle_seconds, 6),
            surviving_project_threads=survivors,
        )
        if first_survivors:
            self.failures.append({"test": nodeid, "survivors": first_survivors})

    def record_final_ownership(self) -> None:
        """Make a final strict inventory after the last test teardown."""
        self.set_phase(None, "session_finish")
        survivors = wait_for_project_thread_baseline(
            baseline_threads=self.baseline_threads,
            timeout_seconds=0.0,
        )
        first_survivors = first_unattributed_survivors(
            survivors,
            self._attributed_object_ids,
        )
        self.emit("final_ownership", surviving_project_threads=survivors)
        if first_survivors:
            self.failures.append(
                {"test": "<session-finish>", "survivors": first_survivors}
            )

    def finish(self, exitstatus: int) -> None:
        self.set_phase(None, "session_finish")
        self._stop.set()
        if self._monitor.is_alive():
            self._monitor.join(max(self.interval * 2, 0.2))
        monitor_stopped = not self._monitor.is_alive()
        self._finished.set()
        self.emit(
            "session_finish",
            exitstatus=exitstatus,
            monitor_stopped=monitor_stopped,
            ownership_failures=self.failures,
        )

    def _monitor_loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.emit("sample")
            self._maybe_emit_stacks()

    def _maybe_emit_stacks(self) -> None:
        with self._lock:
            captured_test = self.current_test
            captured_phase = self.current_phase
            identity = (captured_test, captured_phase)
            elapsed = time.monotonic() - self.current_since
            if (
                self.current_test is None
                or elapsed < self.stack_after
                or self.stack_emitted_for == identity
            ):
                return
            self.stack_emitted_for = identity
        frames = sys._current_frames()
        names = {thread.ident: thread.name for thread in threading.enumerate()}
        stacks = {
            names.get(ident, f"thread-{ident}"): _normalize_stack_paths(
                "".join(traceback.format_stack(frame))
            )
            for ident, frame in frames.items()
        }
        self.emit(
            "stack_snapshot",
            phase=captured_phase,
            phase_elapsed_seconds=round(elapsed, 6),
            stacks=stacks,
            test=captured_test,
        )


def _normalize_stack_paths(value: str) -> str:
    replacements = (
        (str(Path.cwd()), "$REPO"),
        (sys.prefix, "$VENV"),
        (sys.base_prefix, "$PYTHON"),
        (str(Path.home()), "$HOME"),
    )
    for raw, replacement in replacements:
        value = value.replace(raw, replacement)
    return value


def _state(config: pytest.Config) -> _DiagnosticState | None:
    return getattr(config, _STATE_ATTR, None)


def pytest_configure(config: pytest.Config) -> None:
    """Start the monitor only when a report path was explicitly supplied."""
    raw_path = config.getoption("thread_diagnostics_jsonl", default=None)
    if not raw_path:
        return
    state = _DiagnosticState(config, Path(raw_path))
    setattr(config, _STATE_ATTR, state)
    state.start()


def pytest_runtest_setup(item: pytest.Item) -> None:
    state = _state(item.config)
    if state is not None:
        state.set_phase(item.nodeid, "setup")


def pytest_runtest_call(item: pytest.Item) -> None:
    state = _state(item.config)
    if state is not None:
        state.set_phase(item.nodeid, "call")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    state = _state(item.config)
    if state is not None:
        state.emit(
            "test_report",
            duration_seconds=round(report.duration, 6),
            outcome=report.outcome,
            report_phase=report.when,
            test=item.nodeid,
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None):
    state = _state(item.config)
    if state is not None:
        state.set_phase(item.nodeid, "teardown")
    yield
    if state is not None:
        state.record_teardown(item.nodeid)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    state = _state(session.config)
    if state is None:
        return
    state.record_final_ownership()
    if state.strict and state.failures and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    state.finish(int(session.exitstatus))
