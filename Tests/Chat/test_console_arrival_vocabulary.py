"""TASK-34100.5 AC#9 (entry-exit-handoff-22): arrival uses plain readiness words.

Right after a successful setup the Console status strip read
'Library · Auto off · Agent blocked … Context unknown' and Home after Skip read
'Home | Blocked · Local'. 'Blocked' and 'unknown' read as errors for normal
states: the agent's Library access is simply off, an unknown window only
matters when it blocks a send (the send refusal then names the fix), and a
skipped setup is not set up rather than blocked.
"""

from __future__ import annotations

from types import SimpleNamespace

from tldw_chatbook.Chat.console_context_policy import ContextCompactionMode
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostState
from tldw_chatbook.Chat.console_display_state import ConsoleLibraryPolicyDisplayState
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Widgets.Console.console_context_controls import (
    build_console_context_cost_state,
)


def _library_chip(access: ConsoleAssistantLibraryAccess) -> str:
    snapshot = ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=access,
        policy_revision=None,
        source="missing",
    )
    return ConsoleLibraryPolicyDisplayState.from_snapshot(snapshot).chip_label


def test_library_chip_says_the_agents_access_is_off_not_blocked() -> None:
    blocked = _library_chip(ConsoleAssistantLibraryAccess.BLOCKED)
    allowed = _library_chip(ConsoleAssistantLibraryAccess.ALLOWED)

    assert blocked == "Library · Auto off · Agent access off"
    assert allowed == "Library · Auto off · Agent access on"
    assert "blocked" not in blocked.lower()


def _context(request_tokens, ceiling):
    return SimpleNamespace(
        request_tokens=request_tokens,
        safe_input_ceiling_tokens=ceiling,
        conversation_tokens=None,
        conversation_budget_tokens=None,
        compaction_trigger_tokens=None,
        resolved_policy=SimpleNamespace(
            policy=SimpleNamespace(compaction_mode=ContextCompactionMode.OFF)
        ),
    )


def test_an_unknown_window_does_not_label_the_status_chip() -> None:
    cost = ConsoleCostState(
        label="$0.00", compact_label="$0.00", tooltip="", alert=False, cold=False
    )

    unknown = build_console_context_cost_state(_context(None, None), cost)
    known = build_console_context_cost_state(_context(500, 1000), cost)

    assert "unknown" not in unknown.label.lower()
    assert "unknown" not in unknown.compact_label.lower()
    assert unknown.label.startswith("Current $0.00")
    # The detail stays one hover away, with the fix.
    assert "context window is unavailable" in unknown.tooltip
    assert known.label.startswith("Context 50% · Current $0.00")


def test_home_after_skip_says_not_set_up_instead_of_blocked() -> None:
    from tldw_chatbook.Home.dashboard_state import _header_line

    skipped = SimpleNamespace(model_ready=False, runtime_source="local", server_label="")
    ready = SimpleNamespace(model_ready=True, runtime_source="local", server_label="")

    assert _header_line(skipped) == "Home | Not set up · Local"
    assert _header_line(ready) == "Home | Ready · Local"
