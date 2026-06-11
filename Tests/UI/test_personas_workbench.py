# Tests/UI/test_personas_workbench.py
"""Mounted tests for the destination-native Personas workbench."""

from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App
from textual.widgets import Button, Static

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from tldw_chatbook.UI.Navigation.shortcut_context import ShortcutAction, ShortcutContext
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    EditPersonaRequested,
    PersonaProfileSaveRequested,
)

pytestmark = pytest.mark.asyncio

CHARACTERS = [
    {"id": 1, "name": "Detective Sam", "description": "Noir detective", "version": 1},
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
            assert [str(r.label) for r in rows] == ["Detective Sam", "Lab Assistant"]

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

    async def test_new_button_opens_editor_in_create_mode(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            assert screen._edit_mode == "create"
            editor = screen.query_one("#ccp-character-editor-view")
            assert editor.display is True

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
            assert [str(r.label) for r in rows] == ["Archivist"]

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
            assert [str(r.label) for r in rows] == ["Detective Sam", "Lab Assistant"]
