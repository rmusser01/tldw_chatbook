from tldw_chatbook.Chat.console_display_state import (
    ConsoleControlState,
    ConsoleInspectorState,
    ConsoleStagedContextState,
)
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch


def test_console_control_state_exposes_provider_model_and_context_labels():
    state = ConsoleControlState.from_values(
        provider="OpenAI",
        model="gpt-5.5",
        persona="Researcher",
        rag_enabled=True,
        staged_source_count=3,
        tool_count=4,
        approval_count=1,
    )

    assert state.provider_label == "Provider: OpenAI"
    assert state.model_label == "Model: gpt-5.5"
    assert state.persona_label == "Persona: Researcher"
    assert state.rag_label == "RAG: on"
    assert state.sources_label == "Sources: 3 staged"
    assert state.tools_label == "Tools: 4 ready"
    assert state.approvals_label == "Approvals: 1 pending"


def test_console_control_state_preserves_falsy_labels_and_general_assistant_fallback():
    state = ConsoleControlState.from_values(
        provider=0,
        model=False,
        persona="",
    )

    assert state.provider_label == "Provider: 0"
    assert state.model_label == "Model: False"
    assert state.persona_label == "Assistant: General"


def test_console_staged_context_state_preserves_live_work_payload_provenance():
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Transformer notes",
        status="ready",
        recovery="Review citations before sending.",
        payload={"source_id": "note-1", "citation_count": 2},
    )

    state = ConsoleStagedContextState.from_live_work(launch)

    assert state.heading == "Staged Context"
    assert "Transformer notes" in state.summary
    assert any(row.label == "source_id" and row.value == "note-1" for row in state.rows)
    assert state.recovery == "Review citations before sending."


def test_console_inspector_state_combines_readiness_artifact_and_recovery_rows():
    state = ConsoleInspectorState.from_values(
        live_work_title="Daily papers",
        provider_ready=False,
        provider_recovery="Configure a provider before sending.",
        rag_status="missing index",
        artifact_status="save available after response",
        approval_count=0,
    )

    text = state.to_plain_text()
    assert "Daily papers" in text
    assert "Provider: blocked" in text
    assert "Configure a provider before sending." in text
    assert "RAG/source: missing index" in text
    assert "Artifacts: save available after response" in text


def test_console_inspector_state_uses_explicit_chatbook_save_capability():
    state = ConsoleInspectorState.from_values(
        artifact_status="Chatbook save available",
        can_save_chatbook=True,
    )

    label_only_state = ConsoleInspectorState.from_values(
        artifact_status="Chatbook save available",
    )

    assert state.can_save_chatbook is True
    assert label_only_state.can_save_chatbook is False


def test_console_inspector_state_exposes_action_disabled_reasons():
    state = ConsoleInspectorState.from_values(
        provider_ready=False,
        provider_recovery="Select a provider and model before sending.",
        rag_status="missing source",
        artifact_status="No Chatbook artifact available",
        tool_count=0,
        approval_count=0,
        can_save_chatbook=False,
    )

    text = state.to_plain_text()
    actions_by_id = {action.widget_id: action for action in state.actions}

    assert "Tools: 0 ready" in text
    assert "RAG/source: missing source" in text
    assert actions_by_id["console-inspector-review-approval"].enabled is False
    assert (
        actions_by_id["console-inspector-review-approval"].disabled_reason
        == "No approval is pending."
    )
    assert actions_by_id["console-inspector-review-tool-call"].enabled is False
    assert (
        actions_by_id["console-inspector-review-tool-call"].disabled_reason
        == "No tool calls are ready for review."
    )
    assert actions_by_id["console-inspector-save-chatbook"].enabled is False
    assert (
        actions_by_id["console-inspector-save-chatbook"].disabled_reason
        == "No Chatbook artifact is available."
    )


def test_console_inspector_state_enables_pending_approval_tools_and_chatbook_actions():
    state = ConsoleInspectorState.from_values(
        live_work_title="Grounded answer",
        provider_ready=True,
        rag_status="staged from Library Search/RAG",
        artifact_status="Chatbook artifact available",
        tool_count=2,
        approval_count=1,
        can_save_chatbook=True,
    )

    actions_by_id = {action.widget_id: action for action in state.actions}

    assert state.has_pending_approval is True
    assert state.can_save_chatbook is True
    assert actions_by_id["console-inspector-review-approval"].enabled is True
    assert actions_by_id["console-inspector-review-tool-call"].enabled is True
    assert actions_by_id["console-inspector-save-chatbook"].enabled is True
