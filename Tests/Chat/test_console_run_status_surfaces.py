"""TASK-347: status surfaces must reflect an active run, not read 'Ready'.

During a run (thinking gap and visible streaming) the header status chip
read 'Ready' and the Inspector read 'Status: Ready' / 'Live work: No active
work' — a direct contradiction of the streaming transcript (UX review
finding j4-status-surfaces-say-ready-during-run). The truthful indicators
(amber Stop, [streaming] suffix) existed, but the dedicated status surfaces
lied.
"""

from tldw_chatbook.Chat.console_display_state import (
    ConsoleControlState,
    ConsoleInspectorState,
)
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector
from tldw_chatbook.Widgets.Console.console_workbench_state import (
    build_console_workbench_state,
)


def _ready_control_state() -> ConsoleControlState:
    return ConsoleControlState(
        provider_label="Provider: llama_cpp",
        model_label="Model: local-gemma",
        persona_label="Assistant: General",
        rag_label="RAG: off",
        sources_label="Sources: 0",
        tools_label="Tools: 0",
        approvals_label="Approvals: 0",
    )


def test_workbench_header_status_is_running_during_active_run():
    running = build_console_workbench_state(
        control_state=_ready_control_state(),
        can_send=False,
        can_stop=True,
        run_active=True,
    )
    assert running.header.status == "running"

    idle = build_console_workbench_state(
        control_state=_ready_control_state(),
        can_send=True,
        can_stop=False,
        run_active=False,
    )
    assert idle.header.status == "ready"


def test_workbench_header_stays_blocked_over_running_when_not_active():
    blocked = build_console_workbench_state(
        control_state=_ready_control_state(),
        provider_blocker_copy="Provider setup needed: choose a model",
        run_active=False,
    )
    assert blocked.header.status == "blocked"


def test_inspector_live_work_and_status_reflect_active_run():
    running = ConsoleInspectorState.from_values(
        provider_label="llama_cpp",
        model_label="local-gemma",
        provider_ready=True,
        run_active=True,
    )
    rows = {row.label: row for row in running.rows}
    assert "No active work" not in rows["Live work"].value
    assert "Generating" in rows["Live work"].value

    inspector_status = ConsoleRunInspector._status_summary(
        ConsoleRunInspector, running
    )
    assert "Generating" in inspector_status
    assert inspector_status != "Status: Ready"


def test_inspector_returns_to_ready_when_run_inactive():
    idle = ConsoleInspectorState.from_values(
        provider_label="llama_cpp",
        model_label="local-gemma",
        provider_ready=True,
        run_active=False,
    )
    rows = {row.label: row for row in idle.rows}
    assert rows["Live work"].value == "No active work"
    assert (
        ConsoleRunInspector._status_summary(ConsoleRunInspector, idle)
        == "Status: Ready"
    )


def test_inspector_blocked_still_wins_over_running():
    """A mid-run provider block (or pending approval) is more important to
    surface than the generic running state."""
    blocked = ConsoleInspectorState.from_values(
        provider_ready=False,
        provider_recovery="Configure a provider before sending.",
        run_active=True,
    )
    assert (
        ConsoleRunInspector._status_summary(ConsoleRunInspector, blocked)
        == "Status: Blocked"
    )
