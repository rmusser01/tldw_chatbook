import inspect

import pytest

from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_INSPECTOR_NO_APPROVAL_REASON,
    CONSOLE_INSPECTOR_NO_CHATBOOK_ARTIFACT_REASON,
    CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
    CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID,
    CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
    CONSOLE_SYSTEM_PROMPT_LABEL_SET,
    CONSOLE_SYSTEM_PROMPT_LABEL_UNSET,
    ConsoleControlState,
    ConsoleInspectorState,
    ConsoleLibraryPolicyDisplayState,
    ConsoleStagedContextState,
    estimate_console_next_send_tokens,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch


def test_console_control_state_exposes_provider_model_and_context_labels():
    state = ConsoleControlState.from_values(
        provider="OpenAI",
        model="gpt-5.5",
        library_policy=ConsoleLibraryPolicySnapshot(
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
            policy_revision=2,
            source="durable",
        ),
        staged_source_count=3,
        tool_count=4,
        approval_count=1,
    )

    assert state.provider_label == "Provider: OpenAI"
    assert state.model_label == "Model: gpt-5.5"
    assert state.assistant_label == "Assistant: General"
    assert state.rag_label == "Library · Auto on · Agent blocked"
    assert state.sources_label == "Sources: 3"
    assert state.tools_label == "Tools: 4 ready"
    assert state.approvals_label == "Approvals: 1 pending"


@pytest.mark.parametrize(
    ("automatic", "assistant", "expected"),
    (
        (
            ConsoleAutoRetrieve.NEVER,
            ConsoleAssistantLibraryAccess.BLOCKED,
            "Library · Auto off · Agent blocked",
        ),
        (
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.BLOCKED,
            "Library · Auto on · Agent blocked",
        ),
        (
            ConsoleAutoRetrieve.NEVER,
            ConsoleAssistantLibraryAccess.ALLOWED,
            "Library · Auto off · Agent allowed",
        ),
        (
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.ALLOWED,
            "Library · Auto on · Agent allowed",
        ),
    ),
)
def test_library_policy_display_state_pins_the_four_fixed_order_combinations(
    automatic: ConsoleAutoRetrieve,
    assistant: ConsoleAssistantLibraryAccess,
    expected: str,
) -> None:
    state = ConsoleLibraryPolicyDisplayState.from_snapshot(
        ConsoleLibraryPolicySnapshot(
            auto_retrieve=automatic,
            assistant_access=assistant,
            policy_revision=1,
            source="durable",
        )
    )

    assert state.chip_label == expected
    assert state.auto_retrieve_label in {"Never", "Automatic"}
    assert state.assistant_access_label in {"Blocked", "Allowed"}
    assert "Sources" not in state.chip_label
    assert "ready" not in state.chip_label.lower()


def test_library_policy_display_state_fails_closed_when_authority_is_unavailable() -> None:
    state = ConsoleLibraryPolicyDisplayState.from_snapshot(
        ConsoleLibraryPolicySnapshot(
            auto_retrieve=ConsoleAutoRetrieve.NEVER,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
            policy_revision=None,
            source="unavailable",
            error_code="policy_read_error",
        )
    )

    assert state.chip_label == "Library: blocked · policy unavailable"
    assert state.source_status == "Unavailable — using Never and Blocked"
    assert state.editing_enabled is False
    assert state.save_enabled is False


def test_console_control_state_preserves_falsy_labels_and_generic_assistant_fallback():
    state = ConsoleControlState.from_values(
        provider=0,
        model=False,
    )

    assert state.provider_label == "Provider: 0"
    assert state.model_label == "Model: False"
    assert state.assistant_label == "Assistant: General"


def test_console_control_state_counter_activity_flags():
    idle = ConsoleControlState.from_values()
    assert (idle.sources_active, idle.tools_active, idle.approvals_active) == (
        False,
        False,
        False,
    )
    busy = ConsoleControlState.from_values(
        staged_source_count=2, tool_count=1, approval_count=3
    )
    assert (busy.sources_active, busy.tools_active, busy.approvals_active) == (
        True,
        True,
        True,
    )


def test_console_control_state_system_prompt_label_defaults_to_unset():
    state = ConsoleControlState.from_values()

    assert state.system_prompt_label == CONSOLE_SYSTEM_PROMPT_LABEL_UNSET


def test_console_control_state_system_prompt_label_reflects_set_prompt():
    state = ConsoleControlState.from_values(system_prompt_set=True)

    assert state.system_prompt_label == CONSOLE_SYSTEM_PROMPT_LABEL_SET


def test_console_control_state_tools_chip_includes_mcp_tools():
    """TASK-350: the header Tools chip must count the tools that can actually run
    — built-in AND MCP — not just built-in. It read 'Tools: 0 ready' while the
    inspector showed 'MCP: 10 tools ready'."""
    state = ConsoleControlState.from_values(tool_count=0, mcp_tool_count=10)
    assert state.tools_label == "Tools: 10 ready"
    assert state.tools_active is True


def test_console_control_state_tools_chip_sums_builtin_and_mcp():
    state = ConsoleControlState.from_values(tool_count=2, mcp_tool_count=10)
    assert state.tools_label == "Tools: 12 ready"


def test_console_control_state_tools_chip_without_mcp_seam_counts_builtin_only():
    # No MCP seam wired (mcp_tool_count default None) — chip is unchanged.
    state = ConsoleControlState.from_values(tool_count=3)
    assert state.tools_label == "Tools: 3 ready"
    assert state.tools_active is True


def test_console_control_state_tools_chip_shows_neutral_placeholder_at_zero():
    """Fleet-UX expert review F7 (task-1234): a zero effective tool count
    (the default -- the built-in count hook is never actually populated by
    production code, see `ChatScreen._console_tool_count`) must not read
    "Tools: 0 ready" -- live UAT read that as "no tools available" even
    though calculator/get_current_datetime are always registered builtins.
    `tools_active` (dim/emphasis) is UNCHANGED by this -- still False at
    zero, exactly as before.

    TASK-2154.12 (TX-04): the placeholder is now an inert dash and the chip
    itself hides at zero, so the lazy-loading detail never renders."""
    state = ConsoleControlState.from_values()
    assert state.tools_label == "Tools: —"
    assert "0 ready" not in state.tools_label
    assert "not loaded" not in state.tools_label
    assert state.tools_active is False

    # Explicit zero (not just the default) reads identically.
    explicit_zero = ConsoleControlState.from_values(tool_count=0, mcp_tool_count=None)
    assert explicit_zero.tools_label == "Tools: —"

    # A real mcp_tool_count of 0 (seam wired, catalog genuinely empty) is
    # NOT "no MCP seam" (that's `None`) -- still an honest zero, same copy.
    wired_but_empty = ConsoleControlState.from_values(tool_count=0, mcp_tool_count=0)
    assert wired_but_empty.tools_label == "Tools: —"


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

    # Task-400: no summary line for the empty state -- the tray widget owns
    # the "No sources attached. Stage sources from Library." guidance copy,
    # and a summary here rendered the same copy twice.
    assert state.summary == ""
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
    # TASK-24610: retrieval status is "Retrieval"; "Sources" is
    # staged context only.
    assert "Retrieval: missing index" in text
    assert "RAG/source:" not in text
    assert "Artifacts: save available after response" in text
    rows_by_label = {row.label: row for row in state.rows}
    assert rows_by_label["Provider"].status == "blocked"
    assert rows_by_label["Retrieval"].status == "blocked"
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
        mcp_tool_count=5,
        mcp_not_connected_count=1,
    )
    rows_by_label = {row.label: row for row in state.rows}
    assert rows_by_label["MCP"].value == "1 server enabled, not connected"
    assert rows_by_label["MCP"].status == "blocked"


def test_console_inspector_state_shows_mcp_not_connected_row_plural():
    state = ConsoleInspectorState.from_values(
        mcp_tool_count=5,
        mcp_not_connected_count=2,
    )
    rows_by_label = {row.label: row for row in state.rows}
    assert rows_by_label["MCP"].value == "2 servers enabled, not connected"
    assert rows_by_label["MCP"].status == "blocked"


def test_console_inspector_state_omits_mcp_row_when_zero_tools_and_zero_not_connected():
    state = ConsoleInspectorState.from_values(
        mcp_tool_count=0,
        mcp_not_connected_count=0,
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

    # TASK-1843: the Inspector row shares the chip's derivation. TASK-2154.12
    # (TX-04): the zero placeholder is the inert dash -- "0 ready" read as
    # "no tools available", "not loaded" exposed the lazy-loading detail.
    assert "Tools: —" in text
    assert "not loaded" not in text
    assert "Retrieval: missing source" in text
    assert "RAG/source:" not in text
    assert actions_by_id[CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID].enabled is False
    assert (
        actions_by_id[CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID].disabled_reason
        == CONSOLE_INSPECTOR_NO_APPROVAL_REASON
    )
    # TASK-1843: "Review tool call" was removed. It gated on a counter
    # production never populates, so it was permanently disabled while
    # permanently claiming a reason -- and its handler was a notify() stub.
    assert CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID not in actions_by_id
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
    # TASK-1843: removed -- see the note in the disabled-reasons test above.
    assert CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID not in actions_by_id
    assert actions_by_id[CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID].enabled is True
    rows_by_label = {row.label: row for row in state.rows}
    assert rows_by_label["Approvals"].status == "blocked"


def test_console_inspector_save_chatbook_action_is_blocked_when_ephemeral():
    """F2 (task-9 review): the run inspector's Save Chatbook action is a
    third door onto the same write the workbench and composer bar already
    gate. Must disable with the registry reason -- and still work normally
    otherwise (the control)."""
    from tldw_chatbook.Chat.console_ephemeral import blocked_reason

    blocked = ConsoleInspectorState.from_values(
        artifact_status="Chatbook artifact available",
        can_save_chatbook=True,
        ephemeral=True,
    )
    blocked_actions = {action.widget_id: action for action in blocked.actions}
    save_action = blocked_actions[CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID]
    assert save_action.enabled is False
    assert save_action.disabled_reason == blocked_reason(
        "save-chatbook", ephemeral=True
    )

    normal = ConsoleInspectorState.from_values(
        artifact_status="Chatbook artifact available",
        can_save_chatbook=True,
        ephemeral=False,
    )
    normal_actions = {action.widget_id: action for action in normal.actions}
    assert normal_actions[CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID].enabled is True


def test_assistant_label_names_the_active_character():
    state = ConsoleControlState.from_values(
        provider="llama_cpp", model="m", character="Seraphina"
    )

    assert state.assistant_label == "Character: Seraphina"


def test_assistant_chip_projects_character_identity_to_one_safe_line():
    state = ConsoleControlState.from_values(
        character="Nyx\n\tAdmin\x00[/bold]",
    )

    assert state.assistant_label == "Character: Nyx Admin?[/bold]"
    assert "\n" not in state.assistant_label
    assert "\t" not in state.assistant_label


def test_assistant_label_is_generic_without_an_identified_assistant():
    state = ConsoleControlState.from_values(provider="llama_cpp", model="m")

    assert state.assistant_label == "Assistant: General"


def test_assistant_label_uses_existing_persona_name():
    state = ConsoleControlState.from_values(
        provider="llama_cpp",
        model="m",
        assistant_kind="persona",
        assistant_name="Guide",
        assistant_id="persona-7",
    )

    assert state.assistant_label == "Persona: Guide"


def test_assistant_label_uses_existing_persona_id_when_name_is_missing():
    state = ConsoleControlState.from_values(
        provider="llama_cpp",
        model="m",
        assistant_kind="persona",
        assistant_id="persona-7",
    )

    assert state.assistant_label == "Persona: persona-7"


def test_assistant_label_normalizes_persona_values_and_keeps_character_precedence():
    normalized_kind = ConsoleControlState.from_values(
        assistant_kind=" Persona ",
        assistant_name=" Guide ",
    )
    id_fallback = ConsoleControlState.from_values(
        assistant_kind="persona",
        assistant_name=" \t ",
        assistant_id=" persona-7 ",
    )
    character = ConsoleControlState.from_values(
        character=" Ada ",
        assistant_kind="persona",
        assistant_name="Guide",
    )

    assert normalized_kind.assistant_label == "Persona: Guide"
    assert id_fallback.assistant_label == "Persona: persona-7"
    assert character.assistant_label == "Character: Ada"


def test_console_control_state_has_one_assistant_presentation_field() -> None:
    parameters = inspect.signature(ConsoleControlState.from_values).parameters

    assert "persona" not in parameters
    assert "user_profile_label" not in ConsoleControlState.__dataclass_fields__
    assert "character_label" not in ConsoleControlState.__dataclass_fields__
    assert "assistant_label" in ConsoleControlState.__dataclass_fields__


@pytest.mark.unit
def test_chip_and_inspector_report_the_same_tool_count():
    """TASK-1843: two surfaces in one panel must not contradict each other.

    `console_tool_count` is read in five places and assigned in NONE, so the
    built-in count is always 0. The chip was fixed to add the MCP count
    (`effective_tool_count`); the Inspector row was not, so it read
    "Tools: 0 ready" beside a chip reporting a real number.

    This is the same bug shape already fixed once on the chip and missed on
    the row -- so the fix belongs at the shared derivation, and this test
    asserts the two agree rather than asserting either one's literal text.
    """
    control = ConsoleControlState.from_values(
        provider="OpenAI", model="gpt-4o", tool_count=0, mcp_tool_count=12
    )
    inspector = ConsoleInspectorState.from_values(
        tool_count=0, mcp_tool_count=12
    )

    tools_rows = [r for r in inspector.rows if r.label == "Tools"]
    assert tools_rows, "inspector has no Tools row"
    assert "12" in tools_rows[0].value, (
        f"inspector says {tools_rows[0].value!r} while the chip says "
        f"{control.tools_label!r} -- same panel, two numbers"
    )
    assert "12" in control.tools_label

    # And the zero case uses the same neutral placeholder on both surfaces
    # (the chip additionally hides at zero, TX-04): "0 ready" read as "no
    # tools available", "not loaded" exposed the lazy-loading detail.
    zero_control = ConsoleControlState.from_values(
        provider="OpenAI", model="gpt-4o", tool_count=0, mcp_tool_count=None
    )
    zero_inspector = ConsoleInspectorState.from_values(tool_count=0)
    zero_row = [r for r in zero_inspector.rows if r.label == "Tools"][0]
    assert zero_control.tools_label == "Tools: —"
    assert zero_row.value == "—", (
        f"inspector zero-state says {zero_row.value!r}, chip says "
        f"{zero_control.tools_label!r}"
    )


@pytest.mark.unit
def test_permanently_dead_review_tool_call_action_is_gone():
    """TASK-1843: a control that can never enable must not advertise a reason.

    `Review tool call` gated on `console_tool_count`, which production never
    populates -- so it was permanently disabled while permanently claiming
    "No tool calls are ready for review", and its handler was a notify()
    stub. PRODUCT.md requires unavailable states to be honest; a permanently
    false one is the opposite.
    """
    inspector = ConsoleInspectorState.from_values(tool_count=0, mcp_tool_count=12)
    labels = [a.label for a in inspector.actions]
    assert not any("Review tool call" in lbl for lbl in labels), (
        f"the dead action is still advertised: {labels}"
    )


# --- Next-send token estimate (task-25836) ---------------------------------
#
# The Next Send tab's "~N tokens" header and the cost chip's first-send
# readout both need "what will the next request actually carry": system
# prompt + messages (draft included) + tool schemas + staged evidence.
# These tests pin the shared pure estimator's counting and its guards.


_FIRST_SEND_MESSAGES = [
    {"role": "system", "content": "You are a thorough assistant. " * 5},
    {"role": "user", "content": "hello, this is my first message"},
]


@pytest.mark.unit
def test_next_send_estimate_counts_tools_on_top_of_messages():
    without_tools = estimate_console_next_send_tokens(
        payload_messages=_FIRST_SEND_MESSAGES
    )
    with_tools = estimate_console_next_send_tokens(
        payload_messages=_FIRST_SEND_MESSAGES,
        tools_info={
            "native_schemas": [
                {
                    "name": "demo_tool",
                    "description": "does demo things",
                    "parameters": {"type": "object", "properties": {}},
                }
            ]
        },
    )

    assert without_tools is not None
    assert with_tools is not None
    assert with_tools > without_tools


@pytest.mark.unit
def test_next_send_estimate_does_not_double_count_duplicated_system_row():
    """The payload's `system` field duplicates the leading system row in
    `messages` (by design, so the viewer can show it at a glance) -- the
    estimate must not count it twice."""
    system_rows = [
        {"role": "system", "content": _FIRST_SEND_MESSAGES[0]["content"]}
    ]
    plain = estimate_console_next_send_tokens(
        payload_messages=_FIRST_SEND_MESSAGES
    )
    duplicated = estimate_console_next_send_tokens(
        payload_messages=_FIRST_SEND_MESSAGES,
        payload_system=system_rows,
    )

    assert duplicated == plain


@pytest.mark.unit
def test_next_send_estimate_counts_fallback_system_when_messages_have_none():
    """`build_context_snapshot` can hand a `system` field whose rows are NOT
    in `messages` (its fallback branch) -- that content still ships."""
    user_only = [{"role": "user", "content": "hello"}]
    without = estimate_console_next_send_tokens(payload_messages=user_only)
    with_system = estimate_console_next_send_tokens(
        payload_messages=user_only,
        payload_system=[
            {"role": "system", "content": "You are a thorough assistant." * 20}
        ],
    )

    assert without is not None
    assert with_system is not None
    assert with_system > without


@pytest.mark.unit
def test_next_send_estimate_folds_extra_texts_and_skips_blank_ones():
    """Staged evidence text rides along as an extra text (the preview payload
    lists staged sources as label-only metadata); blank texts add nothing."""
    base = estimate_console_next_send_tokens(
        payload_messages=_FIRST_SEND_MESSAGES
    )
    with_staged = estimate_console_next_send_tokens(
        payload_messages=_FIRST_SEND_MESSAGES,
        extra_texts=["", "   ", "staged evidence snippet " * 10],
    )

    assert base is not None
    assert with_staged is not None
    assert with_staged > base


@pytest.mark.unit
def test_next_send_estimate_ignores_tools_info_without_schemas():
    """`tools_info` carries prose notes (`mcp_note`/`preview_note`) that are
    not request content; an empty `native_schemas` contributes nothing."""
    with_notes = estimate_console_next_send_tokens(
        payload_messages=_FIRST_SEND_MESSAGES,
        tools_info={
            "native_schemas": [],
            "preview_note": "No native tools are configured for preview.",
        },
    )
    without = estimate_console_next_send_tokens(
        payload_messages=_FIRST_SEND_MESSAGES
    )

    assert with_notes == without


@pytest.mark.unit
def test_next_send_estimate_returns_none_when_nothing_to_send():
    assert estimate_console_next_send_tokens() is None
    assert (
        estimate_console_next_send_tokens(
            payload_messages=[],
            payload_system=[],
            tools_info={"native_schemas": []},
            extra_texts=["   "],
        )
        is None
    )


@pytest.mark.unit
def test_next_send_estimate_accepts_multimodal_part_list_content():
    """Message content may be a provider part-list (text + image) -- the
    estimator must not crash and must count the text part (plus the
    non-text part allowance)."""
    parts = [
        {"type": "text", "text": "describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    total = estimate_console_next_send_tokens(
        payload_messages=[{"role": "user", "content": parts}]
    )

    assert total is not None
    assert total > 0


@pytest.mark.unit
def test_next_send_estimate_skips_schemas_that_cannot_serialize():
    """Qodo finding 3: a schema object whose serialization raises must
    degrade to "no tools row", not propagate out of the estimator."""

    class _Unserializable:
        def __str__(self):  # pragma: no cover - exercised via dumps
            raise TypeError("no repr for you")

    total = estimate_console_next_send_tokens(
        payload_messages=_FIRST_SEND_MESSAGES,
        tools_info={"native_schemas": [{"name": _Unserializable()}]},
    )
    without = estimate_console_next_send_tokens(
        payload_messages=_FIRST_SEND_MESSAGES
    )

    assert total == without
