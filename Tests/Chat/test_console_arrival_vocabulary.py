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
        # The real ConsoleContextControlState carries the model window and
        # whether it is verified (TASK-33007 #12 reads both).
        model_window_tokens=ceiling,
        model_window_verified=True,
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


def test_home_details_after_skip_say_not_set_up_too() -> None:
    """Review round 1 (F9): the header said 'Not set up' while the Details
    on the same screen still read 'Model: Blocked' and 'Model blocked'."""
    from tldw_chatbook.Home.dashboard_state import (
        HomeDashboardInput,
        _status_summary_line,
        _system_status_lines,
    )

    skipped = HomeDashboardInput(model_ready=False)
    summary = _status_summary_line(skipped)
    lines = " ".join(_system_status_lines(skipped))

    assert "Model: Not set up" in summary
    assert "Model: Blocked" not in summary
    assert "Model not set up" in lines
    assert "Model blocked" not in lines


def _ready_line(*, provider_label: str, model: str) -> str:
    from tldw_chatbook.Chat.console_onboarding_state import (
        build_console_setup_card_state,
    )

    state = build_console_setup_card_state(
        readiness=SimpleNamespace(operability="ready_to_send", blocker=None),
        provider_label=provider_label,
        has_model=bool(model),
        model=model,
        first_send_completed=False,
        has_messages=False,
        guidance_dismissed=False,
    )
    assert state.mode == "ready_line"
    return state.body_copy


def test_arrival_is_one_setup_complete_line_naming_the_pair() -> None:
    """TASK-34100.5 AC#11 (E10): arrival says what setup connected in one
    transcript line instead of a stack of toasts."""

    assert _ready_line(provider_label="OpenAI", model="gpt-4.1-mini") == (
        "Setup complete — OpenAI · gpt-4.1-mini. Ready — type a message to begin."
    )
    # A local model saved as a file path shows its file name, not the path.
    assert _ready_line(
        provider_label="llama.cpp",
        model="/Users/me/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    ) == (
        "Setup complete — llama.cpp · qwen2.5-0.5b-instruct-q4_k_m.gguf. "
        "Ready — type a message to begin."
    )
