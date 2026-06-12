"""Foundation coverage for the destination-native Personas workbench."""

from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    PersonaActionRequested,
    PersonaEntitySelected,
    PersonaModeChanged,
    PersonaSearchChanged,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_state import (
    PersonasWorkbenchState,
)


def test_personas_workbench_state_defaults_to_local_character_mode():
    state = PersonasWorkbenchState()

    assert state.active_mode == "characters"
    assert state.runtime_source == "local"
    assert state.search_query == ""
    assert state.filter_text == ""
    assert state.selected_entity_kind is None
    assert state.selected_entity_id is None
    assert state.selected_runtime_target is None
    assert state.is_loading is False
    assert state.has_unsaved_changes is False


def test_personas_workbench_mode_switch_clears_selection_and_dirty_state():
    state = PersonasWorkbenchState(
        active_mode="characters",
        selected_entity_kind="character",
        selected_entity_id="42",
        selected_entity_name="Ada",
        selected_runtime_target="local:character:42",
        has_unsaved_changes=True,
        status_message="Editing Ada",
    )

    state.switch_mode("personas")

    assert state.active_mode == "personas"
    assert state.selected_entity_kind is None
    assert state.selected_entity_id is None
    assert state.selected_runtime_target is None
    assert state.has_unsaved_changes is False
    assert state.status_message == "Mode: Personas"


def test_personas_workbench_selection_builds_console_target_metadata():
    state = PersonasWorkbenchState()

    state.select_entity(
        entity_kind="persona_profile",
        entity_id="persona.local.researcher",
        entity_name="Researcher",
    )

    assert state.selected_entity_kind == "persona_profile"
    assert state.selected_entity_id == "persona.local.researcher"
    assert state.selected_runtime_target == "local:persona_profile:persona.local.researcher"
    assert state.selected_metadata() == {
        "selected_kind": "persona_profile",
        "selected_record_id": "persona.local.researcher",
        "selected_name": "Researcher",
        "selected_target_id": "local:persona_profile:persona.local.researcher",
    }


def test_personas_workbench_state_reset_preserves_runtime_source():
    state = PersonasWorkbenchState(
        active_mode="prompts",
        runtime_source="server",
        search_query="qa",
        filter_text="draft",
        selected_entity_kind="prompt",
        selected_entity_id="p1",
        selected_entity_name="QA prompt",
        selected_runtime_target="server:prompt:p1",
        has_unsaved_changes=True,
        is_loading=True,
    )

    state.reset_for_runtime_source_change("local")

    assert state.runtime_source == "local"
    assert state.active_mode == "prompts"
    assert state.search_query == ""
    assert state.filter_text == ""
    assert state.selected_entity_id is None
    assert state.is_loading is False
    assert state.has_unsaved_changes is False


def test_personas_workbench_messages_are_screen_independent_payloads():
    mode_changed = PersonaModeChanged("dictionaries")
    selected = PersonaEntitySelected(
        entity_kind="character",
        entity_id="7",
        entity_name="Bean",
        runtime_target="local:character:7",
    )
    search_changed = PersonaSearchChanged(query="bean", filter_text="local")
    action = PersonaActionRequested(
        action="attach_to_console",
        entity_kind="character",
        entity_id="7",
    )

    assert mode_changed.mode == "dictionaries"
    assert selected.entity_kind == "character"
    assert selected.entity_id == "7"
    assert selected.runtime_target == "local:character:7"
    assert search_changed.query == "bean"
    assert search_changed.filter_text == "local"
    assert action.action == "attach_to_console"
    assert action.entity_kind == "character"
    assert action.entity_id == "7"
