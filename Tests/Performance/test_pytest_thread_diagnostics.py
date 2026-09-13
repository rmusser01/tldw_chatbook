from __future__ import annotations

import threading
import time
import json

from Tests.Performance import pytest_thread_diagnostics as diagnostics
from Tests.Performance.pytest_thread_diagnostics import (
    _DiagnosticState,
    first_unattributed_survivors,
    project_thread_snapshots,
    wait_for_project_thread_baseline,
)


def test_project_thread_snapshots_classify_only_agent_owned_prefixes() -> None:
    threads = (
        threading.Thread(name="tool-fs_read"),
        threading.Thread(name="fleet-1234abcd"),
        threading.Thread(name="pytest-thread-diagnostics"),
        threading.Thread(name="ThreadPoolExecutor-0_0"),
    )

    snapshots = project_thread_snapshots(threads)

    assert [row["name"] for row in snapshots] == [
        "fleet-1234abcd",
        "tool-fs_read",
    ]
    assert [row["owner"] for row in snapshots] == ["fleet", "tool"]


def test_wait_for_project_thread_baseline_observes_bounded_cleanup() -> None:
    release = threading.Event()
    worker = threading.Thread(
        target=lambda: release.wait(1.0),
        name="tool-bounded-probe",
        daemon=True,
    )
    worker.start()
    threading.Timer(0.05, release.set).start()

    survivors = wait_for_project_thread_baseline(
        baseline_threads=frozenset(),
        timeout_seconds=0.5,
        poll_seconds=0.01,
    )

    worker.join(0.5)
    assert survivors == ()


def test_wait_for_project_thread_baseline_reports_live_survivor() -> None:
    release = threading.Event()
    worker = threading.Thread(
        target=release.wait,
        name="fleet-stuckprobe",
        daemon=True,
    )
    worker.start()
    try:
        survivors = wait_for_project_thread_baseline(
            baseline_threads=frozenset(),
            timeout_seconds=0.01,
            poll_seconds=0.001,
        )
    finally:
        release.set()
        worker.join(0.5)

    assert [row["name"] for row in survivors] == ["fleet-stuckprobe"]


def test_wait_for_project_thread_baseline_observes_late_spawn() -> None:
    release = threading.Event()
    worker = threading.Thread(
        target=release.wait,
        name="tool-late-survivor",
        daemon=True,
    )
    producer = threading.Timer(0.03, worker.start)
    producer.start()
    try:
        survivors = wait_for_project_thread_baseline(
            baseline_threads=frozenset(),
            timeout_seconds=0.12,
            poll_seconds=0.005,
            quiescence_seconds=0.05,
        )
    finally:
        producer.join(0.5)
        release.set()
        worker.join(0.5)

    assert [row["name"] for row in survivors] == ["tool-late-survivor"]


def test_survivor_is_attributed_only_at_first_observation() -> None:
    survivor = {"name": "fleet-stuck", "object_id": 123}
    attributed: set[int] = set()

    first = first_unattributed_survivors((survivor,), attributed)
    second = first_unattributed_survivors((survivor,), attributed)

    assert first == (survivor,)
    assert second == ()


def test_baseline_uses_thread_objects_not_reusable_ident(monkeypatch) -> None:
    class FakeThread:
        daemon = True
        ident = 123

        def __init__(self, name: str) -> None:
            self.name = name

    baseline = FakeThread("tool-baseline")
    later = FakeThread("tool-later")
    monkeypatch.setattr(diagnostics, "_project_threads", lambda threads=None: (later,))

    survivors = wait_for_project_thread_baseline(
        baseline_threads=frozenset({baseline}),
        timeout_seconds=0.0,
    )

    assert [row["name"] for row in survivors] == ["tool-later"]


class _Config:
    def getoption(self, name: str) -> object:
        return {
            "thread_diagnostics_interval": 0.1,
            "thread_diagnostics_settle": 0.1,
            "thread_diagnostics_stack_after": 0.1,
            "thread_diagnostics_strict": True,
        }[name]


def test_stack_snapshot_keeps_captured_phase_during_transition(
    tmp_path,
    monkeypatch,
) -> None:
    state = _DiagnosticState(_Config(), tmp_path / "report.jsonl")
    state.path.parent.mkdir(parents=True, exist_ok=True)
    state.set_phase("old-node", "call")
    state.current_since = time.monotonic() - 1.0
    real_format_stack = diagnostics.traceback.format_stack
    transitioned = False

    def transition_during_format(frame):
        nonlocal transitioned
        if not transitioned:
            transitioned = True
            state.set_phase("new-node", "teardown")
        return real_format_stack(frame)

    monkeypatch.setattr(diagnostics.traceback, "format_stack", transition_during_format)

    state._maybe_emit_stacks()

    row = json.loads(state.path.read_text().splitlines()[0])
    assert row["event"] == "stack_snapshot"
    assert row["test"] == "old-node"
    assert row["phase"] == "call"
    assert row["phase_elapsed_seconds"] >= 1.0
    rendered_stacks = json.dumps(row["stacks"])
    assert str(diagnostics.Path.cwd()) not in rendered_stacks
    assert str(diagnostics.Path.home()) not in rendered_stacks
    assert diagnostics.sys.prefix not in rendered_stacks
    assert diagnostics.sys.base_prefix not in rendered_stacks


def test_final_ownership_inventory_records_late_survivor(tmp_path) -> None:
    state = _DiagnosticState(_Config(), tmp_path / "report.jsonl")
    release = threading.Event()
    worker = threading.Thread(
        target=release.wait,
        name="fleet-final-survivor",
        daemon=True,
    )
    worker.start()
    try:
        state.record_final_ownership()
    finally:
        release.set()
        worker.join(0.5)

    assert state.failures == [
        {
            "test": "<session-finish>",
            "survivors": (
                {
                    "daemon": True,
                    "ident": state.failures[0]["survivors"][0]["ident"],
                    "name": "fleet-final-survivor",
                    "object_id": id(worker),
                    "owner": "fleet",
                },
            ),
        }
    ]
    final_row = json.loads(state.path.read_text().splitlines()[-1])
    assert final_row["event"] == "final_ownership"
    assert final_row["test"] is None
    assert final_row["phase"] == "session_finish"


def test_session_finish_remains_terminal_when_monitor_is_blocked(tmp_path) -> None:
    state = _DiagnosticState(_Config(), tmp_path / "report.jsonl")
    entered = threading.Event()
    release = threading.Event()

    def blocked_stack_capture() -> None:
        entered.set()
        release.wait(1.0)

    state._maybe_emit_stacks = blocked_stack_capture
    state.start()
    assert entered.wait(0.5)

    state.finish(0)
    release.set()
    state._monitor.join(0.5)

    rows = [json.loads(line) for line in state.path.read_text().splitlines()]
    assert rows[-1]["event"] == "session_finish"
    assert rows[-1]["monitor_stopped"] is False
