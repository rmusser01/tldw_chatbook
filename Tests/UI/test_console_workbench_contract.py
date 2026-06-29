from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.Console.console_workbench_state import (
    build_console_workbench_state,
)


def _control_state() -> ConsoleControlState:
    return ConsoleControlState(
        provider_label="Provider: llama.cpp",
        model_label="Model: local-model",
        persona_label="Assistant: General",
        rag_label="RAG: off",
        sources_label="Sources: 0",
        tools_label="Tools: 0",
        approvals_label="Approvals: 0",
    )


def test_console_workbench_state_exposes_core_actions_visibly():
    state = build_console_workbench_state(
        control_state=_control_state(),
        provider_blocker_copy="",
        can_send=True,
        can_stop=False,
        can_save_chatbook=True,
    )

    actions = {action.id: action for action in state.actions}
    action_labels = {action.label for action in state.actions}

    assert {
        "Settings",
        "Attach context",
        "Run Library RAG",
        "Save Chatbook",
        "Help",
    } <= action_labels
    assert tuple(actions) == (
        "new-tab",
        "settings",
        "attach-context",
        "run-library-rag",
        "save-chatbook",
        "send",
        "stop",
        "help",
    )
    assert actions["save-chatbook"].disabled is False
    assert actions["send"].disabled is False
    assert actions["send"].primary is True
    assert actions["stop"].disabled is True
    assert state.route_id == "chat"
    assert state.density == "normal"
    assert state.header.title == "Console"
    assert tuple(pane.id for pane in state.panes) == (
        "context",
        "transcript",
        "inspector",
        "composer",
    )
    assert state.recovery is None


def test_console_workbench_state_surfaces_provider_recovery():
    state = build_console_workbench_state(
        control_state=ConsoleControlState(
            provider_label="Provider: OpenAI",
            model_label="Model: --",
            persona_label="Assistant: General",
            rag_label="RAG: off",
            sources_label="Sources: 0",
            tools_label="Tools: 0",
            approvals_label="Approvals: 0",
        ),
        provider_blocker_copy="Provider setup needed: choose a model",
        provider_action_label="Choose model",
        can_send=False,
        can_stop=False,
        can_save_chatbook=False,
    )

    assert state.recovery is not None
    assert "choose a model" in state.recovery.body.lower()
    assert state.recovery.action is not None
    assert state.recovery.action.label == "Choose model"
    assert state.recovery.action.id == "provider-recovery"
    assert state.recovery.action.primary is True

    modes = {mode.id: mode for mode in state.modes}
    assert modes["provider"].status == "blocked"
    assert modes["model"].status == "blocked"


def test_console_workbench_state_disables_send_when_provider_is_blocked():
    state = build_console_workbench_state(
        control_state=_control_state(),
        provider_blocker_copy="Provider setup needed: choose a model",
        can_send=True,
        can_stop=True,
        can_save_chatbook=True,
        density="compact",
    )

    actions = {action.id: action for action in state.actions}

    assert state.density == "compact"
    assert state.header.status == "blocked"
    assert actions["send"].disabled is True
    assert actions["send"].primary is False
    assert actions["stop"].disabled is False
