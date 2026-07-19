from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_INSPECTOR_NO_APPROVAL_REASON,
    CONSOLE_INSPECTOR_NO_CHATBOOK_ARTIFACT_REASON,
    CONSOLE_INSPECTOR_NO_TOOL_CALLS_REASON,
    CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
    CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID,
    CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
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


def test_console_control_state_counter_activity_flags():
    idle = ConsoleControlState.from_values()
    assert (idle.sources_active, idle.tools_active, idle.approvals_active) == (False, False, False)
    busy = ConsoleControlState.from_values(staged_source_count=2, tool_count=1, approval_count=3)
    assert (busy.sources_active, busy.tools_active, busy.approvals_active) == (True, True, True)


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
    assert state.is_empty is False


def test_console_staged_context_empty_state_uses_semantic_flag():
    state = ConsoleStagedContextState.empty()

    assert state.summary == "No sources attached."
    assert state.is_empty is True


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
    assert "Sources: missing index" in text
    assert "RAG/source:" not in text
    assert "Artifacts: save available after response" in text
    rows_by_label = {row.label: row for row in state.rows}
    assert rows_by_label["Provider"].status == "blocked"
    assert rows_by_label["Sources"].status == "blocked"
    assert "RAG/source" not in rows_by_label
    assert rows_by_label["Approvals"].status == "ready"


def test_console_inspector_state_omits_mcp_row_by_default():
    """`mcp_tool_count=None` (the default) means "no MCP service / kill
    switch on" -- the inspector must not show an "MCP" row at all."""
    state = ConsoleInspectorState.from_values()
    assert "MCP" not in {row.label for row in state.rows}


def test_console_inspector_state_shows_mcp_tools_ready_row():
    state = ConsoleInspectorState.from_values(mcp_tool_count=3)
    rows_by_label = {row.label: row for row in state.rows}
    assert rows_by_label["MCP"].value == "3 tools ready"
    assert rows_by_label["MCP"].status == "ready"


def test_console_inspector_state_shows_mcp_tools_ready_row_singular():
    """Pluralization fix (Finding I2): exactly one tool reads "1 tool
    ready", not "1 tools ready"."""
    state = ConsoleInspectorState.from_values(mcp_tool_count=1)
    rows_by_label = {row.label: row for row in state.rows}
    assert rows_by_label["MCP"].value == "1 tool ready"


def test_console_inspector_state_shows_mcp_not_connected_row_even_with_tools_ready():
    """Finding I2: the blocked "not connected" affordance must win
    whenever `mcp_not_connected_count > 0`, REGARDLESS of `mcp_tool_count`
    -- a stale (disconnected-with-snapshot) server still contributes its
    own tools to the eligible catalog (see
    `MCPToolProvider.compose_catalog`'s eligibility filter), so in the
    real mixed case `mcp_tool_count` is essentially never 0. The previous
    "tool_count == 0 and not_connected > 0" gate made this branch
    unreachable in production; pinning it at (5, 1) -- a REAL reachable
    state -- instead of the old (0, 2) case."""
    state = ConsoleInspectorState.from_values(
        mcp_tool_count=5, mcp_not_connected_count=1,
    )
    rows_by_label = {row.label: row for row in state.rows}
    assert rows_by_label["MCP"].value == "1 server enabled, not connected"
    assert rows_by_label["MCP"].status == "blocked"


def test_console_inspector_state_shows_mcp_not_connected_row_plural():
    state = ConsoleInspectorState.from_values(
        mcp_tool_count=5, mcp_not_connected_count=2,
    )
    rows_by_label = {row.label: row for row in state.rows}
    assert rows_by_label["MCP"].value == "2 servers enabled, not connected"
    assert rows_by_label["MCP"].status == "blocked"


def test_console_inspector_state_omits_mcp_row_when_zero_tools_and_zero_not_connected():
    state = ConsoleInspectorState.from_values(
        mcp_tool_count=0, mcp_not_connected_count=0,
    )
    assert "MCP" not in {row.label for row in state.rows}


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
    assert "Sources: missing source" in text
    assert "RAG/source:" not in text
    assert actions_by_id[CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID].enabled is False
    assert (
        actions_by_id[CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID].disabled_reason
        == CONSOLE_INSPECTOR_NO_APPROVAL_REASON
    )
    assert actions_by_id[CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID].enabled is False
    assert (
        actions_by_id[CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID].disabled_reason
        == CONSOLE_INSPECTOR_NO_TOOL_CALLS_REASON
    )
    assert actions_by_id[CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID].enabled is False
    assert (
        actions_by_id[CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID].disabled_reason
        == CONSOLE_INSPECTOR_NO_CHATBOOK_ARTIFACT_REASON
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
    assert actions_by_id[CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID].enabled is True
    assert actions_by_id[CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID].enabled is True
    assert actions_by_id[CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID].enabled is True
    rows_by_label = {row.label: row for row in state.rows}
    assert rows_by_label["Approvals"].status == "blocked"
