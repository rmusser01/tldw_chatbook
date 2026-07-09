from dataclasses import dataclass

from tldw_chatbook.Chat.console_display_state import (
    ConsoleInspectorState,
    ConsoleStagedContextState,
)
from tldw_chatbook.Chat.console_rail_state import (
    ConsoleRailPreferences,
    build_console_context_rail_badge,
    build_console_inspector_rail_badge,
    build_console_rail_preference_key,
    build_console_rail_state,
    coerce_console_rail_preferences,
    serialize_console_rail_preferences,
)


@dataclass(frozen=True)
class Row:
    label: str
    status: str = "ready"
    value: str = ""
    text: str = ""


def test_console_rail_state_uses_first_start_defaults():
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )
    state = build_console_rail_state(preference_key=key)

    assert state.left_open is True
    assert state.right_open is False
    assert state.preferred_left_open is True
    assert state.preferred_right_open is False
    assert state.persistence_key == "console_rail_state:workspace-1:session-1"


def test_console_rail_state_restores_stored_preferences():
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"left_open": False, "right_open": True},
        available_columns=220,
    )

    assert state.left_open is False
    assert state.right_open is True


def test_console_rail_state_invalid_stored_preferences_fall_back_to_defaults():
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    for invalid_preferences in (
        None,
        "bad",
        {"left_open": "bad"},
        {"right_open": []},
    ):
        state = build_console_rail_state(
            preference_key=key,
            stored_preferences=invalid_preferences,
        )

        assert state.left_open is True
        assert state.right_open is False


def test_console_rail_state_coerces_integer_preferences():
    preferences = coerce_console_rail_preferences(
        {"left_open": 0, "right_open": 1},
    )

    assert preferences.left_open is False
    assert preferences.right_open is True


def test_console_rail_preference_key_prefers_conversation_then_session_fallback():
    key = build_console_rail_preference_key(
        workspace_id="workspace 1",
        conversation_id="conv:1",
        session_id="session:1",
    )

    assert key.value == "console_rail_state:workspace_1:conv_1"
    assert key.fallback_value == "console_rail_state:workspace_1:session_1"


def test_console_rail_preference_key_covers_scope_fallbacks():
    session_key = build_console_rail_preference_key(
        workspace_id="workspace 1",
        session_id="session:1",
    )
    workspace_key = build_console_rail_preference_key(workspace_id="workspace 1")
    global_key = build_console_rail_preference_key()

    assert session_key.value == "console_rail_state:workspace_1:session_1"
    assert session_key.fallback_value is None
    assert workspace_key.value == "console_rail_state:workspace_1:global"
    assert workspace_key.fallback_value is None
    assert global_key.value == "console_rail_state:global:global"
    assert global_key.fallback_value is None


def test_console_rail_preference_key_treats_zero_scope_ids_as_present():
    key = build_console_rail_preference_key(
        workspace_id="workspace",
        conversation_id=0,
        session_id=0,
    )
    session_key = build_console_rail_preference_key(
        workspace_id="workspace",
        session_id=0,
    )

    assert key.value == "console_rail_state:workspace:0"
    assert key.fallback_value == "console_rail_state:workspace:0"
    assert session_key.value == "console_rail_state:workspace:0"


def test_console_rail_preference_key_treats_whitespace_scope_ids_as_absent():
    session_key = build_console_rail_preference_key(
        workspace_id="workspace",
        conversation_id="   ",
        session_id=0,
    )
    global_key = build_console_rail_preference_key(
        workspace_id="workspace",
        conversation_id="   ",
        session_id="\t",
    )

    assert session_key.value == "console_rail_state:workspace:0"
    assert session_key.fallback_value is None
    assert global_key.value == "console_rail_state:workspace:global"
    assert global_key.fallback_value is None


def test_console_context_rail_badge_prioritizes_available_context():
    assert build_console_context_rail_badge(staged_source_count=3) == "3 staged"
    assert (
        build_console_context_rail_badge(
            staged_source_count="bad",
            staged_summary="Ready staged citations",
        )
        == "staged"
    )
    assert (
        build_console_context_rail_badge(workspace_label="Research workspace")
        == "workspace"
    )
    assert build_console_context_rail_badge(session_label="Conversation 1") == "session"
    assert build_console_context_rail_badge() == ""


def test_console_context_rail_badge_ignores_workspace_fallback_labels():
    for workspace_label in ("", "local", "default", "no workspace", "No-workspace"):
        assert (
            build_console_context_rail_badge(
                workspace_label=workspace_label,
                session_label="Conversation 1",
            )
            == "session"
        )


def test_console_context_rail_badge_ignores_empty_staged_summary():
    empty_summary = ConsoleStagedContextState.empty().summary

    assert (
        build_console_context_rail_badge(
            staged_summary=empty_summary,
            session_label="Conversation 1",
        )
        == "session"
    )
    assert build_console_context_rail_badge(staged_summary=empty_summary) == ""


def test_console_context_rail_badge_ignores_default_workspace_display_labels():
    for workspace_label in (
        "No workspace selected",
        "Workspace: Local Default",
        "Workspace: Default",
    ):
        assert (
            build_console_context_rail_badge(
                workspace_label=workspace_label,
                session_label="Conversation 1",
            )
            == "session"
        )
        assert build_console_context_rail_badge(workspace_label=workspace_label) == ""


def test_console_inspector_rail_badge_prioritizes_run_and_review_state():
    assert build_console_inspector_rail_badge(run_status="failed") == "failed"
    assert (
        build_console_inspector_rail_badge(
            run_status="failed",
            inspector_rows=(Row("Policy", status="blocked"),),
        )
        == "failed"
    )
    assert (
        build_console_inspector_rail_badge(
            run_status="streaming",
            inspector_rows=(Row("Policy", status="blocked"),),
            approval_count=2,
        )
        == "blocked"
    )
    assert build_console_inspector_rail_badge(approval_count=1) == "1 approval"
    assert build_console_inspector_rail_badge(approval_count=2) == "2 approvals"
    assert (
        build_console_inspector_rail_badge(
            tool_count=3,
            inspector_rows=(Row("Artifacts", value="Chatbook artifact available"),),
        )
        == "tools"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Artifacts", value="Chatbook artifact available"),),
        )
        == "artifact"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("RAG/source", value="3 sources staged"),),
        )
        == "source"
    )
    assert build_console_inspector_rail_badge() == ""


def test_console_inspector_rail_badge_detects_blocked_from_row_fields():
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Policy blocked"),),
        )
        == "blocked"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Policy", value="BLOCKED by workspace"),),
        )
        == "blocked"
    )


def test_console_inspector_rail_badge_names_provider_setup_blockers():
    assert (
        build_console_inspector_rail_badge(
            run_status="blocked",
            inspector_rows=(Row("Provider", status="blocked"),),
        )
        == "setup"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Model", value="Missing", text="blocked"),),
        )
        == "setup"
    )


def test_console_inspector_rail_badge_detects_failed_from_row_fields():
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Run", text="failed"),),
        )
        == "failed"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("FAILED run"),),
        )
        == "failed"
    )


def test_console_inspector_rail_badge_ignores_idle_inspector_rows():
    state = ConsoleInspectorState.from_values(
        provider_ready=True,
        rag_status="not staged",
        artifact_status="unavailable",
    )

    assert build_console_inspector_rail_badge(inspector_rows=state.rows) == ""


def test_console_inspector_rail_badge_ignores_not_requested_source_rows():
    state = ConsoleInspectorState.from_values(
        provider_ready=True,
        rag_status="not requested",
        artifact_status="not available for this item",
    )

    assert build_console_inspector_rail_badge(inspector_rows=state.rows) == ""


def test_console_inspector_rail_badge_detects_positive_artifact_and_source_readiness():
    assert build_console_inspector_rail_badge(can_save_chatbook=True) == "artifact"
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Artifacts", value="available"),),
        )
        == "artifact"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Artifacts", value="Chatbook artifact available"),),
        )
        == "artifact"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("RAG/source", value="available"),),
        )
        == "source"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("RAG/source", value="staged from Library Search/RAG"),),
        )
        == "source"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("RAG/source", value="staged"),),
        )
        == "source"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("RAG/source", status="ready"),),
        )
        == "source"
    )


def test_console_inspector_rail_badge_does_not_treat_staged_as_source_category():
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Review", value="staged"),),
        )
        == ""
    )


def test_console_inspector_rail_badge_requires_label_category_for_artifact_and_source():
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Review", value="source available"),),
        )
        == ""
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Review", value="artifact available"),),
        )
        == ""
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("Artifacts", value="available"),),
        )
        == "artifact"
    )
    assert (
        build_console_inspector_rail_badge(
            inspector_rows=(Row("RAG/source", value="available"),),
        )
        == "source"
    )


def test_console_rail_preferences_accept_boolean_strings_case_insensitively():
    for raw_value in ("true", "yes", "1", "on", "TRUE", "Yes", "ON"):
        preferences = coerce_console_rail_preferences({"left_open": raw_value})

        assert preferences.left_open is True

    for raw_value in ("false", "no", "0", "off", "FALSE", "No", "OFF"):
        preferences = coerce_console_rail_preferences({"right_open": raw_value})

        assert preferences.right_open is False


def test_console_rail_preferences_serialize_to_public_dict_shape():
    assert serialize_console_rail_preferences(
        ConsoleRailPreferences(left_open=False, right_open=True),
    ) == {
        "left_open": False,
        "right_open": True,
        "session_open": True,
        "context_open": True,
        "model_open": True,
        "details_open": False,
    }


def test_console_rail_badges_do_not_mutate_open_booleans():
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"left_open": False, "right_open": False},
        staged_source_count=2,
        run_status="blocked",
        inspector_rows=(Row("Provider", status="blocked"),),
    )

    assert state.left_open is False
    assert state.right_open is False
    assert state.left_badge == "2 staged"
    assert state.right_badge == "setup"


def test_console_rail_state_compact_width_collapses_right_rail_effectively():
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )

    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"left_open": True, "right_open": True},
        available_columns=120,
    )

    assert state.left_open is True
    assert state.right_open is False
    assert state.preferred_right_open is True
    assert state.right_forced_collapsed is True


def test_console_rail_section_defaults():
    from tldw_chatbook.Chat.console_rail_state import CONSOLE_RAIL_SECTION_IDS
    prefs = ConsoleRailPreferences()
    assert CONSOLE_RAIL_SECTION_IDS == ("session", "context", "model", "details")
    assert prefs.session_open is True
    assert prefs.context_open is True
    assert prefs.model_open is True
    assert prefs.details_open is False


def test_coerce_console_rail_preferences_reads_section_fields():
    coerced = coerce_console_rail_preferences(
        {"left_open": True, "details_open": "true", "model_open": "off"}
    )
    assert coerced.details_open is True
    assert coerced.model_open is False
    assert coerced.session_open is True  # missing key -> default


def test_serialize_console_rail_preferences_round_trips_sections():
    prefs = ConsoleRailPreferences(details_open=True, context_open=False)
    serialized = serialize_console_rail_preferences(prefs)
    assert serialized["details_open"] is True
    assert serialized["context_open"] is False
    assert coerce_console_rail_preferences(serialized) == prefs


def test_build_console_rail_state_carries_section_flags():
    key = build_console_rail_preference_key(workspace_id="ws", session_id="s")
    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"details_open": True, "session_open": False},
    )
    assert state.details_open is True
    assert state.session_open is False
    assert state.context_open is True
    assert state.model_open is True
