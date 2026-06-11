# Tests/UI/test_personas_workbench.py
"""Mounted tests for the destination-native Personas workbench."""

from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App
from textual.widgets import Button, Static

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
import tldw_chatbook.UI.Persona_Modules.personas_conversations_controller as conversations_controller_module
import tldw_chatbook.UI.Screens.personas_screen as personas_screen_module
from tldw_chatbook.tldw_api import PersonaProfileCreate
from tldw_chatbook.UI.Navigation.shortcut_context import ShortcutAction, ShortcutContext
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import PersonaActionRequested
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    EditPersonaRequested,
    PersonaProfileSaveRequested,
)

pytestmark = pytest.mark.asyncio

CHARACTERS = [
    {
        "id": 1,
        "name": "Detective Sam",
        "description": "Noir detective",
        "first_message": "The name's {{char}}. Who's asking?",
        "version": 1,
    },
    {"id": 2, "name": "Lab Assistant", "description": "Helpful scientist", "version": 1},
]

PROFILE = {
    "id": "p-1",
    "name": "Archivist",
    "description": "Preserve and retrieve",
    "system_prompt": "You are a meticulous archivist.",
}


@pytest.fixture
def stub_scope_service(mock_app_instance):
    """Replace the MagicMock scope service with explicit AsyncMock methods."""
    service = Mock()
    service.list_persona_profiles = AsyncMock(
        return_value={"items": [dict(PROFILE)], "total": 1}
    )
    service.get_persona_profile = AsyncMock(return_value=dict(PROFILE))
    service.create_persona_profile = AsyncMock(
        return_value={"id": "p-9", "name": "Mentor"}
    )
    service.update_persona_profile = AsyncMock(
        return_value={"id": "p-1", "name": "Archivist 2"}
    )
    service.delete_persona_profile = AsyncMock(
        return_value={"status": "deleted", "persona_id": "p-1"}
    )
    mock_app_instance.character_persona_scope_service = service
    return service


@pytest.fixture
def stub_characters(monkeypatch):
    monkeypatch.setattr(
        character_handler_module, "fetch_all_characters", lambda: [dict(c) for c in CHARACTERS]
    )
    monkeypatch.setattr(
        character_handler_module,
        "fetch_character_by_id",
        lambda character_id: next(
            dict(c) for c in CHARACTERS if str(c["id"]) == str(character_id)
        ),
    )


class PersonasTestApp(App):
    def __init__(self, mock_app_instance):
        super().__init__()
        self._mock = mock_app_instance
        self.character_persona_scope_service = mock_app_instance.character_persona_scope_service

    # Delegating these to a MagicMock would make Textual see phantom dynamic
    # hooks (``compute_*``/``watch_*``/...) on the App and crash at mount.
    _NON_DELEGATED_PREFIXES = ("_", "watch_", "compute_", "validate_", "action_", "key_", "on_")

    def __getattr__(self, name):
        if name.startswith(self._NON_DELEGATED_PREFIXES):
            raise AttributeError(name)
        return getattr(self.__dict__["_mock"], name)

    def compose(self):
        # Mirrors the real app: the footer lives on the app's default screen
        # (see app.py compose), which is exactly where the screen's
        # ``self.app.query_one("AppFooterStatus")`` lookup resolves.
        yield AppFooterStatus(id="app-footer-status")

    def on_mount(self) -> None:
        self.push_screen(PersonasScreen(self))


def _row_text(item) -> str:
    """Visible text of a library/conversation row (the ListItem's inner Static)."""
    return str(item.query_one(Static).renderable)


async def _mounted(pilot):
    await pilot.pause()
    return pilot.app.screen


class TestWorkbenchShell:
    async def test_route_renders_destination_workbench(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            title = screen.query_one("#personas-title", Static)
            assert "Personas" in str(title.renderable)
            assert "ds-destination-header" in title.classes
            assert screen.query_one("#personas-mode-strip")
            assert screen.query_one("#personas-library-pane")
            assert screen.query_one("#personas-work-area")
            assert screen.query_one("#personas-inspector-pane")

    async def test_characters_mode_lists_library_rows(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam", "Lab Assistant"]

    async def test_footer_shortcut_context_set_and_cleared(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            context = screen._shortcut_context()
            rendered = context.render()
            assert "new" in rendered.lower()
            assert "search" in rendered.lower()
            assert "save" in rendered.lower()
            assert "attach" in rendered.lower()
            assert context.source == "personas"
            footer = pilot.app.query_one(AppFooterStatus)
            assert "new" in footer.shortcut_text.lower()
            assert "search" in footer.shortcut_text.lower()
            await pilot.app.pop_screen()
            await pilot.pause()
            assert footer.shortcut_text == AppFooterStatus.DEFAULT_SHORTCUT_TEXT

    async def test_unmount_clear_does_not_stomp_other_screens_context(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            footer = pilot.app.query_one(AppFooterStatus)
            # Another screen registers its context (switch_screen mounts the
            # incoming screen before unmounting the outgoing one).
            footer.set_shortcut_context(
                ShortcutContext(
                    source="console",
                    actions=(ShortcutAction("ctrl+enter", "send"),),
                )
            )
            screen._clear_footer_shortcuts()
            assert "ctrl+enter send" in footer.shortcut_text
            assert footer.shortcut_text != AppFooterStatus.DEFAULT_SHORTCUT_TEXT

    async def test_status_row_shows_live_counts_per_mode(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            status = screen.query_one("#personas-status-row", Static)
            assert "Characters: 2" in str(status.renderable)
            assert "Source: Local" in str(status.renderable)
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert "Personas: 1" in str(status.renderable)
            await pilot.click("#personas-mode-prompts")
            await pilot.pause()
            assert "Mode: Prompts" in str(status.renderable)

    async def test_placeholder_modes_show_placeholder_panel(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.click("#personas-mode-prompts")
            await pilot.pause()
            assert screen.state.active_mode == "prompts"
            placeholder = screen.query_one("#personas-mode-placeholder", Static)
            assert placeholder.display is True
            assert "not available yet" in str(placeholder.renderable)
            assert "is-active" in screen.query_one("#personas-mode-prompts", Button).classes


class TestCharacterSelectionAndEdit:
    async def test_row_selection_shows_card_and_inspector(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            assert screen.state.selected_entity_id == "1"
            assert screen._edit_mode == "view"
            assert "Selected: Detective Sam" in str(
                screen.query_one("#personas-selected-name", Static).renderable
            )

    async def test_card_body_populates_on_selection(self, mock_app_instance, stub_characters):
        """The card's BODY must populate: placeholder hidden, fields filled.

        Mirrors the screenshot QA defect where the inspector and preview
        populated but the center card kept its 'No character loaded.'
        placeholder with an empty details area.
        """
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.query_one("#ccp-character-card-view").display is True
            placeholder = screen.query_one("#personas-character-card-empty")
            assert placeholder.display is False
            body = screen.query_one("#personas-character-card-body")
            assert body.display is True
            name = screen.query_one("#personas-character-card-name", Static)
            assert "Detective Sam" in str(name.renderable)

    async def test_new_button_opens_editor_in_create_mode(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            # Select a character first: entering create mode must not leave
            # the previous selection's identity in the inspector.
            await pilot.click("#personas-library-row-character-1")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            assert screen._edit_mode == "create"
            editor = screen.query_one("#ccp-character-editor-view")
            assert editor.display is True
            selected_name = str(
                screen.query_one("#personas-selected-name", Static).renderable
            )
            assert "Detective Sam" not in selected_name
            # Unsaved gating must survive the identity reset.
            readiness = str(
                screen.query_one("#personas-readiness-console", Static).renderable
            )
            assert "Blocked" in readiness

    async def test_ctrl_n_opens_editor_in_create_mode(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.press("ctrl+n")
            await pilot.pause()
            assert screen._edit_mode == "create"
            assert screen.query_one("#ccp-character-editor-view").display is True

    async def test_ctrl_f_focuses_library_search(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.press("ctrl+f")
            await pilot.pause()
            focused = pilot.app.focused
            assert focused is not None
            assert focused.id == "personas-library-search"

    async def test_save_with_missing_name_blocks_and_shows_validation(self, mock_app_instance, stub_characters, monkeypatch):
        created = []
        monkeypatch.setattr(
            character_handler_module, "create_character",
            lambda data: created.append(data) or 99,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            from tldw_chatbook.Widgets.CCP_Widgets.ccp_character_editor_widget import (
                CharacterSaveRequested,
            )
            screen.post_message(CharacterSaveRequested({"name": "", "first_message": "Hi"}))
            await pilot.pause()
            summary = screen.query_one("#personas-validation-summary", Static)
            assert "name: required" in str(summary.renderable)
        assert created == []

    async def test_save_calls_create_and_refreshes(self, mock_app_instance, stub_characters, monkeypatch):
        created = []
        monkeypatch.setattr(
            character_handler_module, "create_character",
            lambda data: created.append(data) or 99,
        )

        def fetch_all_with_created():
            characters = [dict(c) for c in CHARACTERS]
            if created:
                characters.append({"id": 99, "name": "New Hero", "version": 1})
            return characters

        monkeypatch.setattr(
            character_handler_module, "fetch_all_characters", fetch_all_with_created
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            from tldw_chatbook.Widgets.CCP_Widgets.ccp_character_editor_widget import (
                CharacterSaveRequested,
            )
            screen.post_message(CharacterSaveRequested({"name": "New Hero", "first_message": "Hi"}))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.selected_entity_id == "99"
            assert screen._edit_mode == "view"
        assert created and created[0]["name"] == "New Hero"

    async def test_edit_requested_for_mismatched_character_is_ignored(self, mock_app_instance, stub_characters):
        from tldw_chatbook.Widgets.CCP_Widgets.ccp_character_card_widget import (
            EditCharacterRequested,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            screen.post_message(EditCharacterRequested(2))
            await pilot.pause()
            assert screen._edit_mode == "view"
            assert screen.query_one("#ccp-character-editor-view").display is False

    async def test_mode_switch_during_save_does_not_render_character_into_other_mode(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        import threading

        release = threading.Event()
        created = []

        def blocking_create(data):
            release.wait(timeout=5)
            created.append(data)
            return 99

        monkeypatch.setattr(character_handler_module, "create_character", blocking_create)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            from tldw_chatbook.Widgets.CCP_Widgets.ccp_character_editor_widget import (
                CharacterSaveRequested,
            )
            screen.post_message(CharacterSaveRequested({"name": "New Hero", "first_message": "Hi"}))
            await pilot.pause()  # Save worker is now blocked inside create_character.
            await screen._apply_mode("prompts")
            await pilot.pause()
            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert created and created[0]["name"] == "New Hero"
            assert screen.state.active_mode == "prompts"
            assert screen.state.selected_entity_id is None
            placeholder = screen.query_one("#personas-mode-placeholder", Static)
            assert placeholder.display is True
            assert screen.query_one("#ccp-character-card-view").display is False
            assert "New Hero" not in str(
                screen.query_one("#personas-selected-name", Static).renderable
            )


class TestPersonasMode:
    async def _enter_personas_mode(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-mode-personas")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def test_personas_mode_lists_profiles(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Archivist"]

    async def test_profile_selection_shows_card(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await pilot.click("#personas-library-row-persona_profile-p-1")
            await pilot.pause()
            assert screen.state.selected_entity_kind == "persona_profile"
            assert screen.query_one("#ccp-persona-card-view").display is True
            assert "Selected: Archivist" in str(
                screen.query_one("#personas-selected-name", Static).renderable
            )

    async def test_profile_save_calls_scope_service(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await pilot.click("#personas-library-new")
            await pilot.pause()
            assert screen._edit_mode == "create"
            screen.post_message(PersonaProfileSaveRequested({"name": "Mentor"}))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            stub_scope_service.create_persona_profile.assert_awaited_once()
            assert screen._edit_mode == "view"

    async def test_profile_edit_save_calls_update(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await pilot.click("#personas-library-row-persona_profile-p-1")
            await pilot.pause()
            screen.post_message(EditPersonaRequested("p-1"))
            await pilot.pause()
            assert screen._edit_mode == "edit"
            assert screen.query_one("#ccp-persona-editor-view").display is True
            screen.post_message(
                PersonaProfileSaveRequested({"id": "p-1", "name": "Archivist 2"})
            )
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            stub_scope_service.update_persona_profile.assert_awaited_once()
            assert stub_scope_service.update_persona_profile.await_args.args[0] == "p-1"
            assert screen._edit_mode == "view"

    async def test_profile_save_failure_keeps_editor_open(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        stub_scope_service.create_persona_profile.side_effect = RuntimeError("boom")
        notifications: list[tuple[str, str]] = []
        app = PersonasTestApp(mock_app_instance)
        app.notify = lambda message, severity="information", **kwargs: notifications.append(
            (str(message), severity)
        )
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await pilot.click("#personas-library-new")
            await pilot.pause()
            screen.post_message(PersonaProfileSaveRequested({"name": "Mentor"}))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any(
                "Save failed" in message and severity == "error"
                for message, severity in notifications
            )
            assert screen._edit_mode == "create"
            assert screen.query_one("#ccp-persona-editor-view").display is True

    async def test_profile_save_passes_schema_object(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await pilot.click("#personas-library-new")
            await pilot.pause()
            screen.post_message(
                PersonaProfileSaveRequested(
                    {"name": "Mentor", "description": "Guides new users"}
                )
            )
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            stub_scope_service.create_persona_profile.assert_awaited_once()
            request = stub_scope_service.create_persona_profile.await_args.args[0]
            assert isinstance(request, PersonaProfileCreate)
            assert request.name == "Mentor"
            assert request.description == "Guides new users"

    async def test_double_save_creates_once(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await pilot.click("#personas-library-new")
            await pilot.pause()
            screen.post_message(PersonaProfileSaveRequested({"name": "Mentor"}))
            screen.post_message(PersonaProfileSaveRequested({"name": "Mentor"}))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            stub_scope_service.create_persona_profile.assert_awaited_once()

    async def test_character_mode_unaffected(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await pilot.click("#personas-mode-characters")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.active_mode == "characters"
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam", "Lab Assistant"]


PROFILES_FOR_SEARCH = [
    {"id": "p-1", "name": "Archivist", "description": "Preserve and retrieve", "system_prompt": "You are a meticulous archivist."},
    {"id": "p-2", "name": "Navigator", "description": "Charts the course", "system_prompt": "You guide the user."},
]


class TestSearch:
    async def test_search_filters_loaded_characters_locally(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            # Type into the search input
            search_input = screen.query_one("#personas-library-search")
            search_input.value = "sam"
            await pilot.pause()
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam"]
            count = str(screen.query_one("#personas-library-count", Static).renderable)
            assert "1 of 2 characters" in count

    async def test_clearing_search_restores_all_rows(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            search_input = screen.query_one("#personas-library-search")
            # Filter first
            search_input.value = "sam"
            await pilot.pause()
            # Then clear
            search_input.value = ""
            await pilot.pause()
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam", "Lab Assistant"]
            count = str(screen.query_one("#personas-library-count", Static).renderable)
            assert "2 characters" in count
            assert "of" not in count

    async def test_search_is_case_insensitive(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            search_input = screen.query_one("#personas-library-search")
            search_input.value = "LAB"
            await pilot.pause()
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Lab Assistant"]

    async def test_search_filters_profiles_in_personas_mode(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        # Replace the scope service stub with two profiles
        stub_scope_service.list_persona_profiles = AsyncMock(
            return_value={"items": [dict(p) for p in PROFILES_FOR_SEARCH], "total": 2}
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            # Two profiles loaded; now search for "nav"
            search_input = screen.query_one("#personas-library-search")
            search_input.value = "nav"
            await pilot.pause()
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Navigator"]
            count = str(screen.query_one("#personas-library-count", Static).renderable)
            assert "1 of 2 persona profiles" in count

    async def test_mode_switch_clears_search(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            # Search in characters mode
            search_input = screen.query_one("#personas-library-search")
            search_input.value = "sam"
            await pilot.pause()
            assert len(screen.query(".personas-library-row")) == 1
            # Switch to personas mode and back
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-mode-characters")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            # All rows visible after round-trip
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam", "Lab Assistant"]
            # Search input is cleared
            assert screen.query_one("#personas-library-search").value == ""

    async def test_fts_path_used_for_large_libraries(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        fts_calls: list[str] = []

        def fake_fts(search_term: str, limit: int = 50):
            fts_calls.append(search_term)
            return [{"id": 1, "name": "Detective Sam"}]

        monkeypatch.setattr(character_handler_module, "search_characters_fts", fake_fts)

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            # Lower the threshold so the 2-character stub library triggers FTS
            screen.LIBRARY_FTS_THRESHOLD = 2
            search_input = screen.query_one("#personas-library-search")
            search_input.value = "sam"
            await pilot.pause()
            assert fts_calls == ["sam"]
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam"]

    async def test_concurrent_renders_do_not_duplicate_rows(
        self, mock_app_instance, stub_characters
    ):
        """Two back-to-back renders must serialize instead of double-mounting."""
        import asyncio

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await asyncio.gather(
                screen._render_library_rows(), screen._render_library_rows()
            )
            await pilot.pause()
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam", "Lab Assistant"]


class TestImportExport:
    """Path-based import/export flows; file dialogs are never exercised."""

    @pytest.fixture
    def stub_db(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: sentinel
        )
        return sentinel

    @staticmethod
    def _capture_notifications(app) -> list[tuple[str, str]]:
        """Shadow App.notify with an instance attribute, like _notify resolves it."""
        captured: list[tuple[str, str]] = []
        app.notify = lambda message, severity="information", **kwargs: captured.append(
            (str(message), severity)
        )
        return captured

    async def test_import_success_refreshes_selects_and_clears_search(
        self, mock_app_instance, stub_characters, monkeypatch, tmp_path
    ):
        imported_paths: list[str] = []

        def fake_import(file_path):
            imported_paths.append(file_path)
            return 3

        monkeypatch.setattr(
            character_handler_module, "import_character_card", fake_import
        )

        def fetch_all_with_imported():
            characters = [dict(c) for c in CHARACTERS]
            if imported_paths:
                characters.append({"id": 3, "name": "Imported Hero", "version": 1})
            return characters

        monkeypatch.setattr(
            character_handler_module, "fetch_all_characters", fetch_all_with_imported
        )
        monkeypatch.setattr(
            character_handler_module,
            "fetch_character_by_id",
            lambda character_id: next(
                (
                    dict(c)
                    for c in fetch_all_with_imported()
                    if str(c["id"]) == str(character_id)
                ),
                None,
            ),
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            search_input = screen.query_one("#personas-library-search")
            search_input.value = "sam"
            await pilot.pause()
            await screen._import_character_from_path(str(tmp_path / "card.json"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert imported_paths == [str(tmp_path / "card.json")]
            assert screen.state.selected_entity_id == "3"
            assert screen.query_one("#personas-library-search").value == ""
            rows = screen.query(".personas-library-row")
            assert "Imported Hero" in [_row_text(r) for r in rows]

    async def test_second_import_request_ignored_while_dialog_active(
        self, mock_app_instance, stub_characters
    ):
        """A queued import action must not start a second dialog worker."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            calls: list[int] = []

            def counting_worker():
                calls.append(1)

                async def _noop():
                    pass

                return _noop()

            screen._import_dialog_worker = counting_worker
            screen._io_dialog_active = True
            screen.post_message(PersonaActionRequested(action="import"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert calls == []
            # Sanity check: with the flag cleared the same wiring does fire.
            screen._io_dialog_active = False
            screen.post_message(PersonaActionRequested(action="import"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert calls == [1]

    async def test_import_existing_character_notifies_already_existed(
        self, mock_app_instance, stub_characters, monkeypatch, tmp_path
    ):
        """A name-conflict import returns an existing id; the copy must say so."""
        monkeypatch.setattr(
            character_handler_module, "import_character_card", lambda file_path: 1
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await screen._import_character_from_path(str(tmp_path / "dupe.json"))
            await pilot.pause()
            assert screen.state.selected_entity_id == "1"
            assert any(
                "already existed" in message and severity == "information"
                for message, severity in notifications
            )
            assert not any("Character imported." in message for message, _ in notifications)

    async def test_import_failure_shows_recovery_copy(
        self, mock_app_instance, stub_characters, monkeypatch, tmp_path
    ):
        def fake_import(file_path):
            raise ValueError("Unsupported card format")

        monkeypatch.setattr(
            character_handler_module, "import_character_card", fake_import
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await screen._import_character_from_path(str(tmp_path / "bad.json"))
            await pilot.pause()
            assert any(
                "Unsupported card format" in message and severity == "error"
                for message, severity in notifications
            )
            assert screen.state.selected_entity_id == "1"

    async def test_export_json_writes_file(
        self, mock_app_instance, stub_characters, stub_db, monkeypatch, tmp_path
    ):
        calls: list[tuple] = []

        def fake_export(db, character_id, include_image=True):
            calls.append((db, character_id, include_image))
            return '{"name": "Detective Sam"}'

        monkeypatch.setattr(
            personas_screen_module, "export_character_card_to_json", fake_export
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            target = tmp_path / "detective_sam.json"
            await screen._export_selected_character(str(target), fmt="json")
            await pilot.pause()
            assert calls and calls[0][0] is stub_db and calls[0][1] == 1
            assert target.read_text(encoding="utf-8") == '{"name": "Detective Sam"}'
            assert any(
                "Exported to" in message and severity == "information"
                for message, severity in notifications
            )

    async def test_export_png_delegates(
        self, mock_app_instance, stub_characters, stub_db, monkeypatch, tmp_path
    ):
        captured: dict = {}

        def fake_export_png(db, character_id, output_path, base_directory=None):
            captured.update(
                db=db,
                character_id=character_id,
                output_path=output_path,
                base_directory=base_directory,
            )
            return True

        monkeypatch.setattr(
            personas_screen_module, "export_character_card_to_png", fake_export_png
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            target = tmp_path / "detective_sam.png"
            await screen._export_selected_character(str(target), fmt="png")
            await pilot.pause()
            assert captured["db"] is stub_db
            assert captured["character_id"] == 1
            assert captured["output_path"] == str(target)
            assert any(
                "Exported to" in message and severity == "information"
                for message, severity in notifications
            )

    async def test_export_profile_json(
        self, mock_app_instance, stub_characters, stub_scope_service, tmp_path
    ):
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-library-row-persona_profile-p-1")
            await pilot.pause()
            target = tmp_path / "archivist.json"
            await screen._export_selected_character(str(target), fmt="json")
            await pilot.pause()
            assert "Archivist" in target.read_text(encoding="utf-8")
            assert any(
                "Exported to" in message and severity == "information"
                for message, severity in notifications
            )

    async def test_import_requires_characters_mode(
        self, mock_app_instance, stub_characters, stub_scope_service, monkeypatch
    ):
        import_calls: list[str] = []
        monkeypatch.setattr(
            character_handler_module,
            "import_character_card",
            lambda file_path: import_calls.append(file_path) or 3,
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            screen.post_message(PersonaActionRequested(action="import"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert import_calls == []
            assert not any(severity == "error" for _, severity in notifications)


class _FtsStubDB:
    """Captures the MATCH term handed to search_character_cards."""

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def search_character_cards(self, search_term, limit=10):
        self.calls.append((search_term, limit))
        return [{"id": 1, "name": "Match"}]


class TestFtsTermSafety:
    """Unit tests for the MATCH expression built by search_characters_fts."""

    @pytest.fixture
    def stub_db(self, monkeypatch):
        stub = _FtsStubDB()
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: stub
        )
        return stub

    async def test_normal_term_becomes_quoted_prefix_query(self, stub_db):
        results = character_handler_module.search_characters_fts("sam")
        assert [term for term, _ in stub_db.calls] == ['"sam"*']
        assert results and results[0]["name"] == "Match"

    async def test_apostrophe_term_is_safe(self, stub_db):
        character_handler_module.search_characters_fts("O'Brien")
        assert [term for term, _ in stub_db.calls] == ['"O\'Brien"*']

    async def test_embedded_double_quote_is_escaped(self, stub_db):
        character_handler_module.search_characters_fts('sam"')
        assert [term for term, _ in stub_db.calls] == ['"sam"""*']

    async def test_fts_operator_characters_are_quoted(self, stub_db):
        character_handler_module.search_characters_fts("(")
        assert [term for term, _ in stub_db.calls] == ['"("*']
        character_handler_module.search_characters_fts("sam-")
        assert [term for term, _ in stub_db.calls][-1] == '"sam-"*'

    async def test_empty_term_returns_empty_without_db_call(self, stub_db):
        assert character_handler_module.search_characters_fts("") == []
        assert character_handler_module.search_characters_fts("   ") == []
        assert stub_db.calls == []


class _NavCaptureApp(PersonasTestApp):
    """Test app that records NavigateToScreen routes bubbled from the screen."""

    def __init__(self, mock_app_instance):
        super().__init__(mock_app_instance)
        self.nav_routes: list[str] = []

    def on_navigate_to_screen(self, message) -> None:
        self.nav_routes.append(message.screen_name)


class TestConversationsPanel:
    @pytest.fixture
    def stub_conversations(self, monkeypatch):
        """Stub the DB resolver, conversation listing, and message retrieval."""
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: object()
        )
        monkeypatch.setattr(
            conversations_controller_module,
            "list_character_conversations",
            lambda db, character_id, limit=50, offset=0: [
                {"id": "conv-1", "title": "First case"}
            ],
        )
        monkeypatch.setattr(
            conversations_controller_module,
            "retrieve_conversation_messages_for_ui",
            lambda db, conversation_id, character_name, user_name, **kwargs: [
                ("Hello there", "Greetings, detective."),
            ],
        )

    async def _select_first_character(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-row-character-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def _open_conversation(self, pilot):
        screen = await self._select_first_character(pilot)
        await pilot.click("#personas-conversation-row-conv-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def test_selecting_character_lists_conversations(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            rows = screen.query(".personas-conversation-row")
            assert [_row_text(r) for r in rows] == ["First case"]

    async def test_conversation_listing_failure_is_tolerant(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        def boom(db, character_id, limit=50, offset=0):
            raise RuntimeError("listing failed")

        monkeypatch.setattr(
            conversations_controller_module, "list_character_conversations", boom
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            assert list(screen.query(".personas-conversation-row")) == []
            # Selection itself still succeeded.
            assert screen.state.selected_entity_id == "1"
            assert screen.query_one("#ccp-character-card-view").display is True

    async def test_conversation_row_opens_readonly_view(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            assert screen.query_one("#personas-conversation-transcript-view").display is True
            assert screen.query_one("#ccp-character-card-view").display is False

    async def test_back_returns_to_card(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            await pilot.click("#personas-conversation-back")
            await pilot.pause()
            assert screen.query_one("#ccp-character-card-view").display is True
            assert screen.query_one("#personas-conversation-transcript-view").display is False

    async def test_continue_in_console_stages_payload(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            await self._open_conversation(pilot)
            await pilot.click("#personas-conversation-continue-console")
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.source == "personas"
        assert payload.item_type == "character-conversation"
        assert payload.metadata["conversation_id"] == "conv-1"
        assert payload.metadata["selected_kind"] == "character"
        assert payload.metadata["selected_record_id"] == "1"
        assert "Detective Sam" in payload.title
        assert "First case" in payload.title
        assert "Greetings, detective." in payload.body

    async def test_open_in_library_navigates(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = _NavCaptureApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            await self._open_conversation(pilot)
            await pilot.click("#personas-conversation-open-library")
            await pilot.pause()
            assert app.nav_routes == ["conversation"]

    async def test_stale_conversation_rows_are_skipped(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Rows for a character other than the current selection are dropped."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            assert screen.state.selected_entity_id == "1"
            await screen.conversations.apply_conversation_rows("999", (("conv-x", "X"),))
            await pilot.pause()
            rows = screen.query(".personas-conversation-row")
            assert [_row_text(r) for r in rows] == ["First case"]
            assert "conv-x" not in screen.conversations._conversation_rows

    async def test_stale_conversation_view_is_skipped(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A view continuation for a superseded conversation id is dropped."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            assert screen.conversations._open_conversation_id == "conv-1"
            await screen.conversations.show_conversation_view(
                "conv-stale", [{"role": "user", "content": "stale"}], "stale", False
            )
            await pilot.pause()
            assert screen.conversations._loaded_conversation_id == "conv-1"
            assert screen.conversations._open_conversation_transcript != "stale"

    async def test_long_transcript_sets_body_truncated(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        monkeypatch.setattr(
            conversations_controller_module,
            "retrieve_conversation_messages_for_ui",
            lambda db, conversation_id, character_name, user_name, **kwargs: [
                ("u" * 500, "b" * 500) for _ in range(20)
            ],
        )
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            await self._open_conversation(pilot)
            await pilot.click("#personas-conversation-continue-console")
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.body_truncated is True
        assert len(payload.body) <= conversations_controller_module._HANDOFF_TRANSCRIPT_CHAR_LIMIT
        assert payload.source_id == "conv-1"

    async def test_continue_blocked_while_loading(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Continue in Console refuses to stage a transcript still in flight."""
        notifications: list[tuple[str, str]] = []
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        app.notify = lambda message, severity="information", **kwargs: notifications.append(
            (str(message), severity)
        )
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            screen.conversations._open_conversation_id = "conv-2"
            screen.conversations._loaded_conversation_id = "conv-1"
            await pilot.click("#personas-conversation-continue-console")
            await pilot.pause()
        app.open_chat_with_handoff.assert_not_called()
        assert any(
            "still loading" in message and severity == "warning"
            for message, severity in notifications
        )

    async def test_continue_blocked_during_same_conversation_reload(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        """Re-selecting the same conversation clears loaded state so Continue is blocked
        until the reload worker delivers its results.

        Regression: open_conversation() previously only reset _open_conversation_transcript
        but left _loaded_conversation_id and _open_conversation_truncated intact.  That meant
        the guard in continue_in_console() saw _loaded_conversation_id ==
        _open_conversation_id immediately after the reload started and would stage an empty
        body with a stale truncation flag.
        """
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            ConversationRowSelected as _CRS,
        )

        notifications: list[tuple[str, str]] = []
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        app.notify = lambda message, severity="information", **kwargs: notifications.append(
            (str(message), severity)
        )

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            # conv-1 is fully loaded at this point.
            assert screen.conversations._loaded_conversation_id == "conv-1"
            assert screen.conversations._open_conversation_transcript != ""

            # Stub the worker entry so it never calls show_conversation_view,
            # thus simulating an in-flight reload whose result hasn't arrived
            # yet. We patch on the controller instance so subsequent calls
            # within this test are bypassed.
            screen.conversations.load_conversation_messages = lambda *args, **kwargs: None

            # Re-select the same conversation by posting the message directly to the
            # screen — this exercises _handle_conversation_row_selected →
            # open_conversation() without relying on the inspector-pane button being
            # click-reachable while the conversation view is displayed on top.
            screen.post_message(_CRS("conv-1"))
            await pilot.pause()

            # open_conversation() should have cleared _loaded_conversation_id.
            assert screen.conversations._loaded_conversation_id is None, (
                "open_conversation() must reset _loaded_conversation_id so that "
                "re-selecting the same conversation doesn't bypass the "
                "still-loading guard"
            )

            # Try to continue — the reload is in flight so it must be blocked.
            await pilot.click("#personas-conversation-continue-console")
            await pilot.pause()

        app.open_chat_with_handoff.assert_not_called()
        assert any(
            "still loading" in message and severity == "warning"
            for message, severity in notifications
        ), f"Expected 'still loading' warning; got: {notifications}"

    async def test_profile_selection_shows_no_conversations(
        self, mock_app_instance, stub_characters, stub_conversations, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            assert len(screen.query(".personas-conversation-row")) == 1
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-library-row-persona_profile-p-1")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.selected_entity_kind == "persona_profile"
            assert list(screen.query(".personas-conversation-row")) == []


class TestConsoleActions:
    """Attach to Console and Start Chat from the inspector (Task 12)."""

    @pytest.fixture
    def stub_conversations(self, monkeypatch):
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: object()
        )
        monkeypatch.setattr(
            conversations_controller_module,
            "list_character_conversations",
            lambda db, character_id, limit=50, offset=0: [
                {"id": "conv-1", "title": "First case"}
            ],
        )
        monkeypatch.setattr(
            conversations_controller_module,
            "retrieve_conversation_messages_for_ui",
            lambda db, conversation_id, character_name, user_name, **kwargs: [
                ("Hello there", "Greetings, detective."),
            ],
        )

    async def _select_first_character(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-row-character-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def _select_profile(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-mode-personas")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.click("#personas-library-row-persona_profile-p-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def test_attach_stages_selected_character_payload(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.query_one("#personas-attach-to-console", Button).press()
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.source == "personas"
        assert payload.runtime_backend == "local"
        assert payload.source_owner == "local"
        assert payload.source_selector_state == "local"
        assert payload.metadata["selected_kind"] == "character"
        assert payload.metadata["selected_record_id"] == "1"
        assert payload.metadata["selected_target_id"] == "local:character:1"
        assert payload.metadata["backend"] == "local"
        assert "Detective Sam" in payload.title
        assert "Noir detective" in payload.body
        assert "Detective Sam" in payload.suggested_prompt

    async def test_attach_blocked_without_selection(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            await _mounted(pilot)
            await pilot.pause()
            await pilot.press("ctrl+enter")
            await pilot.pause()
        app.open_chat_with_handoff.assert_not_called()

    async def test_attach_blocked_with_unsaved_edits(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        from tldw_chatbook.Widgets.CCP_Widgets.ccp_character_card_widget import (
            EditCharacterRequested,
        )

        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.post_message(EditCharacterRequested("1"))
            await pilot.pause()
            assert screen._edit_mode == "edit"
            assert screen.state.has_unsaved_changes is True
            await pilot.press("ctrl+enter")
            await pilot.pause()
        app.open_chat_with_handoff.assert_not_called()

    async def test_start_chat_uses_real_mechanism(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Start Chat stages a handoff with start_chat intent metadata.

        The legacy CCP route launched a blank tab directly via the main chat
        tab container (`#chat-window` lookup), which is not mounted while a
        destination screen is active; the workbench therefore uses the
        app-level ``open_chat_with_handoff`` API with an intent marker.
        """
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.query_one("#personas-start-chat", Button).press()
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.source == "personas"
        assert payload.metadata["intent"] == "start_chat"
        assert payload.metadata["selected_target_id"] == "local:character:1"
        assert payload.suggested_prompt == "Respond as Detective Sam."

    async def test_attach_stages_profile_payload(
        self, mock_app_instance, stub_characters, stub_conversations, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_profile(pilot)
            screen.query_one("#personas-attach-to-console", Button).press()
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.metadata["selected_kind"] == "persona_profile"
        assert payload.metadata["selected_target_id"] == "local:persona_profile:p-1"
        assert "Archivist" in payload.title
        assert "You are a meticulous archivist." in payload.body

    async def test_attach_aborts_when_profile_fetch_degraded(
        self, mock_app_instance, stub_characters, stub_conversations, stub_scope_service
    ):
        """A fallback (list-row) profile record must not stage silently."""
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        captured: list[tuple[str, str]] = []
        app.notify = lambda message, severity="information", **kwargs: captured.append(
            (str(message), severity)
        )
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_profile(pilot)
            # The service degrades after listing/selection succeeded.
            stub_scope_service.get_persona_profile = AsyncMock(
                side_effect=RuntimeError("service down")
            )
            screen.query_one("#personas-attach-to-console", Button).press()
            await pilot.pause()
        app.open_chat_with_handoff.assert_not_called()
        assert (
            "Persona profile is not fully loaded; try reselecting it.",
            "warning",
        ) in captured
        assert not any(severity == "information" for _msg, severity in captured)

    async def test_ctrl_enter_attaches_selected_character(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Ctrl+Enter stages the selected character, same as the Attach button."""
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            await self._select_first_character(pilot)
            await pilot.press("ctrl+enter")
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.source == "personas"
        assert payload.metadata["selected_kind"] == "character"
        assert payload.metadata["selected_record_id"] == "1"

    async def test_attach_warns_when_handoff_unavailable(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A missing app handoff API warns instead of toasting success."""
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = None
        captured: list[tuple[str, str]] = []
        app.notify = lambda message, severity="information", **kwargs: captured.append(
            (str(message), severity)
        )
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.query_one("#personas-attach-to-console", Button).press()
            await pilot.pause()
        assert ("Console handoff is unavailable.", "warning") in captured
        assert not any(severity == "information" for _msg, severity in captured)

    async def test_conversation_continue_still_works(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """The refactored shared seam preserves the Continue-in-Console contract."""
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            await self._select_first_character(pilot)
            await pilot.click("#personas-conversation-row-conv-1")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-conversation-continue-console")
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.source == "personas"
        assert payload.item_type == "character-conversation"
        assert payload.source_id == "conv-1"
        assert payload.metadata["conversation_id"] == "conv-1"
        assert payload.metadata["selected_kind"] == "character"
        assert payload.metadata["selected_record_id"] == "1"
        assert payload.metadata["selected_target_id"] == "local:character:1"
        assert payload.metadata["backend"] == "local"
        assert "Greetings, detective." in payload.body

    async def test_footer_shortcut_attach_available(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            context = screen._shortcut_context()
            attach = next(a for a in context.actions if a.label == "attach")
            assert attach.available is True
            assert attach.key == "ctrl+enter"


class _FakePreviewGateway:
    """In-memory gateway double: ready resolution + scripted stream."""

    def __init__(self, chunks=("Hello, ", "world."), gate=None, error=None):
        self.chunks = chunks
        self.gate = gate
        self.error = error
        self.requests: list[list[dict]] = []
        self.closed = False

    async def resolve_for_send(self, selection):
        from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution

        return ConsoleProviderResolution(
            provider="openai", base_url="", model="test-model", ready=True
        )

    async def stream_chat(self, resolution, messages):
        self.requests.append([dict(m) for m in messages])
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


class TestPreviewIntegration:
    """Ephemeral preview-conversation pane wiring on the screen (Task 13)."""

    @pytest.fixture
    def stub_conversations(self, monkeypatch):
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: object()
        )
        monkeypatch.setattr(
            conversations_controller_module,
            "list_character_conversations",
            lambda db, character_id, limit=50, offset=0: [],
        )

    async def _select_first_character(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-row-character-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def test_preview_pane_is_mounted_in_work_area(
        self, mock_app_instance, stub_characters
    ):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            pane = screen.query_one("#personas-preview-pane", PersonasPreviewPane)
            work_area = screen.query_one("#personas-work-area")
            assert pane in work_area.children

    async def test_greeting_seeds_after_character_load(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """The greeting seeds once the load worker delivers the full card.

        ``load_character`` only schedules a thread worker, so the full record
        (with ``first_message``) is not available synchronously at selection
        time; the screen must seed from the load-completion message instead.
        """
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)
            assert (
                "character: The name's Detective Sam. Who's asking?"
                in pane.transcript_text()
            )

    async def test_reselect_does_not_duplicate_greeting(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Re-selecting a character seeds exactly one greeting line."""
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        async def _select(pilot, row_id):
            await pilot.click(row_id)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.pause()

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            await pilot.pause()
            await _select(pilot, "#personas-library-row-character-2")
            await _select(pilot, "#personas-library-row-character-1")
            pane = screen.query_one(PersonasPreviewPane)
            greeting_line = "character: The name's Detective Sam. Who's asking?"
            lines = [line for line in pane.transcript_text().splitlines() if line]
            assert lines == [greeting_line]

    async def test_blocked_provider_shows_readable_status(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """An unconfigured provider yields readable copy, never a traceback."""
        from textual.widgets import Static as _Static

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )

        # No character_defaults in config -> empty provider -> blocked.
        mock_app_instance.app_config = {}
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            status = str(
                screen.query_one("#personas-preview-status", _Static).renderable
            )
            assert status.strip()
            assert "Traceback" not in status

    async def test_reply_flow_appends_reply_and_history(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        fake = _FakePreviewGateway()
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen._ensure_preview_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi there"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)
            assert "character: Hello, world." in pane.transcript_text()
            assert screen._preview_history == [
                {"role": "user", "content": "Hi there"},
                {"role": "assistant", "content": "Hello, world."},
            ]
            from textual.widgets import Static as _Static

            assert (
                str(screen.query_one("#personas-preview-status", _Static).renderable)
                == "Ready"
            )
            # The provider saw the system prompt followed by the history.
            assert fake.requests and fake.requests[0][0]["role"] == "system"
            assert fake.requests[0][1] == {"role": "user", "content": "Hi there"}

    async def test_draft_aware_system_prompt_uses_editor_data(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        from textual.widgets import TextArea as _TextArea

        from tldw_chatbook.Widgets.CCP_Widgets.ccp_character_card_widget import (
            EditCharacterRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
            PersonasCharacterEditorWidget,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.post_message(EditCharacterRequested("1"))
            await pilot.pause()
            assert screen._edit_mode == "edit"
            editor = screen.query_one(PersonasCharacterEditorWidget)
            editor.query_one(
                "#personas-char-editor-description", _TextArea
            ).text = "Draft noir vibes, unsaved."
            await pilot.pause()
            prompt = screen._preview_system_prompt()
            assert "Draft noir vibes, unsaved." in prompt

    async def test_open_in_console_stages_preview_transcript(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        from textual.widgets import Button as _Button

        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            pane = screen.query_one(PersonasPreviewPane)
            pane.append_user("Hi")
            pane.append_reply("Hello.")
            await pilot.pause()
            screen.query_one("#personas-preview-open-console", _Button).press()
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.source == "personas"
        assert payload.item_type == "preview-conversation"
        assert payload.title == "Personas preview conversation"
        assert "you: Hi" in payload.body
        assert "character: Hello." in payload.body
        assert payload.suggested_prompt == "Continue this conversation in character."

    async def test_stale_reply_is_dropped_after_selection_change(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        import asyncio

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        gate = asyncio.Event()
        fake = _FakePreviewGateway(gate=gate)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen._ensure_preview_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            # Let the worker reach the gated stream deterministically.
            for _ in range(50):
                if fake.requests:
                    break
                await pilot.pause()
            assert fake.requests
            # Selection changes while the stream is in flight. The gated
            # preview worker is still running, so release the gate BEFORE
            # waiting for workers (waiting first deadlocks the test).
            await pilot.click("#personas-library-row-character-2")
            await pilot.pause()
            gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)
            assert "character: Hello, world." not in pane.transcript_text()
            assert not any(
                entry["role"] == "assistant" for entry in screen._preview_history
            )
            from textual.widgets import Static as _Static

            assert (
                str(screen.query_one("#personas-preview-status", _Static).renderable)
                != "Ready"
            )

    async def test_reset_and_mode_switch_clear_history(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewResetRequested,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen._preview_history.append({"role": "user", "content": "Hi"})
            screen.post_message(PreviewResetRequested())
            await pilot.pause()
            assert screen._preview_history == []
            screen._preview_history.append({"role": "user", "content": "Hi again"})
            await screen._apply_mode("prompts")
            await pilot.pause()
            assert screen._preview_history == []

    async def test_reset_mid_stream_drops_late_reply(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Reset while a reply streams must invalidate the in-flight worker.

        The selection key alone cannot catch this: Reset keeps the same
        (kind, id), so without a generation bump (and group cancel) the late
        reply would land in the freshly cleared history/transcript.
        """
        import asyncio

        from textual.widgets import Button as _Button

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        gate = asyncio.Event()
        fake = _FakePreviewGateway(gate=gate)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen._ensure_preview_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            # Let the worker reach the gated stream deterministically.
            for _ in range(50):
                if fake.requests:
                    break
                await pilot.pause()
            assert fake.requests
            # Reset while the stream is held at the gate.
            screen.query_one("#personas-preview-reset", _Button).press()
            await pilot.pause()
            gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)
            assert "character: Hello, world." not in pane.transcript_text()
            assert not any(
                entry["role"] == "assistant" for entry in screen._preview_history
            )

    async def test_error_pops_orphaned_user_history_entry(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A provider error must not leave an unanswered user turn in history.

        The transcript keeps the user's line (they did say it), but the
        history entry is popped so a retry does not send [user, user].
        """
        from textual.widgets import Button as _Button, Static as _Static
        from textual.widgets import Input as _Input

        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        fake = _FakePreviewGateway(error=RuntimeError("provider exploded"))
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen._ensure_preview_gateway = lambda: fake
            pane = screen.query_one(PersonasPreviewPane)
            pane.expand()
            await pilot.pause()
            screen.query_one("#personas-preview-input", _Input).value = "Hi"
            screen.query_one("#personas-preview-test-reply", _Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            # History: no trailing unanswered user entry.
            assert not any(
                entry["role"] == "user" for entry in screen._preview_history
            )
            # Transcript: the user line stays visible.
            assert "you: Hi" in pane.transcript_text()
            status = str(
                screen.query_one("#personas-preview-status", _Static).renderable
            )
            assert status.strip()
            assert "Traceback" not in status

    async def test_empty_reply_sets_status_without_bare_line(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """An empty stream must not append a bare transcript line or history entry."""
        from textual.widgets import Static as _Static

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        fake = _FakePreviewGateway(chunks=())
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen._ensure_preview_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)
            assert all(
                line.strip() != "character:"
                for line in pane.transcript_text().splitlines()
            )
            assert not any(
                entry["role"] == "assistant" for entry in screen._preview_history
            )
            assert (
                str(screen.query_one("#personas-preview-status", _Static).renderable)
                == "No reply received"
            )

    async def test_unmount_closes_gateway(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Leaving the screen releases the preview gateway's HTTP client."""
        fake = _FakePreviewGateway()
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen._preview_gateway = fake
        assert fake.closed is True

    async def test_double_fire_coalesces_user_turns(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """An exclusive-cancelled predecessor leaves back-to-back user turns;
        the replacement worker must coalesce them so strict providers never
        see [user, user]."""
        import asyncio

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )

        gate = asyncio.Event()
        fake = _FakePreviewGateway(gate=gate)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen._ensure_preview_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            for _ in range(50):
                if fake.requests:
                    break
                await pilot.pause()
            assert len(fake.requests) == 1
            # Second request while the first is gated: exclusive=True cancels
            # the first worker, and the second sees history [user, user].
            screen.post_message(PreviewReplyRequested("Again"))
            await pilot.pause()
            for _ in range(50):
                if len(fake.requests) >= 2:
                    break
                await pilot.pause()
            assert len(fake.requests) == 2
            gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            user_messages = [
                m for m in fake.requests[1] if m["role"] == "user"
            ]
            assert user_messages == [{"role": "user", "content": "Hi\nAgain"}]


class TestDelete:
    """Confirmed delete for characters and persona profiles (Task 14).

    The confirmation dialog itself is bypassed by replacing the screen's
    ``_confirm_delete`` helper, the same way the import/export tests bypass
    the file dialogs by calling the path-based methods directly.
    """

    @pytest.fixture
    def stub_conversations(self, monkeypatch):
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: object()
        )
        monkeypatch.setattr(
            conversations_controller_module,
            "list_character_conversations",
            lambda db, character_id, limit=50, offset=0: [
                {"id": "conv-1", "title": "First case"}
            ],
        )

    @staticmethod
    def _capture_notifications(app) -> list[tuple[str, str]]:
        captured: list[tuple[str, str]] = []
        app.notify = lambda message, severity="information", **kwargs: captured.append(
            (str(message), severity)
        )
        return captured

    @staticmethod
    def _bypass_confirm(screen, result: bool) -> None:
        async def _confirm(name: str) -> bool:
            return result

        screen._confirm_delete = _confirm

    async def _select_first_character(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-row-character-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def _select_profile(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-mode-personas")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.click("#personas-library-row-persona_profile-p-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    @staticmethod
    async def _press_delete(pilot, screen):
        screen.query_one("#personas-delete", Button).press()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_delete_character_soft_deletes_and_clears(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        deleted: list[tuple] = []

        def fake_delete(character_id, expected_version):
            deleted.append((character_id, expected_version))
            return True

        monkeypatch.setattr(character_handler_module, "delete_character", fake_delete)

        def fetch_all_post_delete():
            characters = [dict(c) for c in CHARACTERS]
            if deleted:
                characters = [c for c in characters if str(c["id"]) != "1"]
            return characters

        monkeypatch.setattr(
            character_handler_module, "fetch_all_characters", fetch_all_post_delete
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            # Sanity: the conversations panel has rows before the delete.
            assert screen.query_one("#personas-conversations-list").children
            self._bypass_confirm(screen, True)
            await self._press_delete(pilot, screen)
            # delete_character received the id and the FULL record's version.
            assert deleted == [("1", 1)]
            # Selection cleared, view mode, center pane empty.
            assert screen.state.selected_entity_id is None
            assert screen.state.selected_entity_kind is None
            assert screen._edit_mode == "view"
            assert screen.query_one("#ccp-character-card-view").display is False
            assert "Selected: none" in str(
                screen.query_one("#personas-selected-name", Static).renderable
            )
            # Conversations panel emptied.
            assert not screen.query_one("#personas-conversations-list").children
            # Library refreshed without the deleted record.
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Lab Assistant"]
        assert ("Deleted.", "information") in notifications

    async def test_delete_conflict_shows_recovery_copy(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        monkeypatch.setattr(
            character_handler_module,
            "delete_character",
            lambda character_id, expected_version: False,
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            self._bypass_confirm(screen, True)
            await self._press_delete(pilot, screen)
            assert any(
                "changed since it was loaded" in message and severity == "error"
                for message, severity in notifications
            )
            # The selection is retained so the user can reselect/retry.
            assert screen.state.selected_entity_id == "1"
            assert not any(message == "Deleted." for message, _ in notifications)

    async def test_delete_cancelled_is_noop(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        deleted: list[tuple] = []
        monkeypatch.setattr(
            character_handler_module,
            "delete_character",
            lambda character_id, expected_version: deleted.append(
                (character_id, expected_version)
            )
            or True,
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            self._bypass_confirm(screen, False)
            await self._press_delete(pilot, screen)
            assert deleted == []
            assert screen.state.selected_entity_id == "1"
            assert screen.query_one("#ccp-character-card-view").display is True
            assert not any(message == "Deleted." for message, _ in notifications)

    async def test_delete_profile_calls_scope_service(
        self, mock_app_instance, stub_characters, stub_conversations, stub_scope_service
    ):
        # The full record (with version) comes from get_persona_profile.
        stub_scope_service.get_persona_profile = AsyncMock(
            return_value={**PROFILE, "version": 3}
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_profile(pilot)
            self._bypass_confirm(screen, True)
            await self._press_delete(pilot, screen)
            stub_scope_service.delete_persona_profile.assert_awaited_once()
            await_args = stub_scope_service.delete_persona_profile.await_args
            assert await_args.args[0] == "p-1"
            assert await_args.kwargs == {"expected_version": 3, "mode": "local"}
            assert screen.state.selected_entity_id is None
            assert screen.query_one("#ccp-persona-card-view").display is False
        assert ("Deleted.", "information") in notifications

    async def test_delete_blocked_when_full_record_missing(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        deleted: list[tuple] = []
        monkeypatch.setattr(
            character_handler_module,
            "delete_character",
            lambda character_id, expected_version: deleted.append(
                (character_id, expected_version)
            )
            or True,
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            # Simulate a stale handler: the loaded record is another character,
            # so the optimistic-lock version cannot be sourced.
            screen.character_handler.current_character_id = "999"
            self._bypass_confirm(screen, True)
            await self._press_delete(pilot, screen)
            assert deleted == []
            assert screen.state.selected_entity_id == "1"
            assert any(
                "not loaded" in message and severity == "warning"
                for message, severity in notifications
            )
