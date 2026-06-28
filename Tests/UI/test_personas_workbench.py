# Tests/UI/test_personas_workbench.py
"""Mounted tests for the destination-native Personas workbench."""

import inspect
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
import tldw_chatbook.UI.Persona_Modules.personas_conversations_controller as conversations_controller_module
import tldw_chatbook.UI.Screens.personas_screen as personas_screen_module
from tldw_chatbook.Constants import (
    LIBRARY_MODE_CONVERSATIONS,
    LIBRARY_NAV_CONTEXT_CONVERSATION_ID,
    LIBRARY_NAV_CONTEXT_MODE,
    TAB_LIBRARY,
)
from tldw_chatbook.tldw_api import PersonaProfileCreate
from tldw_chatbook.UI.Navigation.shortcut_context import ShortcutAction, ShortcutContext
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import PersonaActionRequested
from tldw_chatbook.Widgets.Persona_Widgets.personas_inspector_pane import (
    PersonasInspectorPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    CharacterImageUploadRequested,
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


class StyledPersonasTestApp(PersonasTestApp):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )


def _row_text(item) -> str:
    """Visible text of a library/conversation row (the ListItem's inner Static)."""
    return str(item.query_one(Static).renderable)


def _right_edge(widget) -> int:
    """Right edge of a mounted widget region."""
    return widget.region.x + widget.region.width


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
            assert (
                screen.query_one("#personas-library-rail-open", Button).tooltip
                == "Open Library rail"
            )
            assert (
                screen.query_one("#personas-inspector-rail-open", Button).tooltip
                == "Open Inspector rail"
            )

    async def test_personas_screen_sets_up_reused_ccp_enhancements(
        self,
        mock_app_instance,
        stub_characters,
    ):
        """Verify PersonasScreen installs loading/decorator support for CCP handlers."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)

            assert hasattr(screen, "loading_manager")
            assert (
                getattr(screen.character_handler.__class__, "_personas_character_enhanced", False)
                is True
            )

    async def test_workbench_columns_fit_80_column_terminal(
        self, mock_app_instance, stub_characters
    ):
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test(size=(80, 40)) as pilot:
            screen = await _mounted(pilot)
            workbench = screen.query_one("#personas-workbench")
            library = screen.query_one("#personas-library-pane")
            work_area = screen.query_one("#personas-work-area")
            inspector = screen.query_one("#personas-inspector-pane")
            readiness = screen.query_one("#personas-readiness-console", Static)

            assert workbench.has_class("personas-workbench-compact")
            assert library.size.width >= 12
            assert work_area.size.width >= 34
            assert inspector.size.width >= 18
            assert _right_edge(inspector) <= _right_edge(workbench)
            assert str(readiness.renderable).startswith("Console blocked:")

    async def test_library_rail_collapses_and_reopens_from_handle(
        self,
        mock_app_instance,
        stub_characters,
    ):
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)

            await pilot.click("#personas-library-rail-collapse")
            await pilot.pause()

            assert screen.query_one("#personas-library-pane").display is False
            assert screen.query_one("#personas-library-rail-handle").display is True
            assert screen.query_one("#personas-work-area").display is True

            await pilot.click("#personas-library-rail-open")
            await pilot.pause()

            assert screen.query_one("#personas-library-pane").display is True
            assert screen.query_one("#personas-library-rail-handle").display is False

    async def test_inspector_rail_collapses_and_reopens_from_handle(
        self,
        mock_app_instance,
        stub_characters,
    ):
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)

            await pilot.click("#personas-inspector-rail-collapse")
            await pilot.pause()

            assert screen.query_one("#personas-inspector-pane").display is False
            assert screen.query_one("#personas-inspector-rail-handle").display is True
            assert screen.query_one("#personas-work-area").display is True

            await pilot.click("#personas-inspector-rail-open")
            await pilot.pause()

            assert screen.query_one("#personas-inspector-pane").display is True
            assert screen.query_one("#personas-inspector-rail-handle").display is False

    async def test_collapsed_inspector_rail_handle_is_keyboard_reachable(
        self,
        mock_app_instance,
        stub_characters,
    ):
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.click("#personas-inspector-rail-collapse")
            await pilot.pause()

            open_button = screen.query_one("#personas-inspector-rail-open", Button)
            await pilot.press("shift+f6")
            await pilot.pause()
            assert pilot.app.focused is open_button

            await pilot.press("enter")
            await pilot.pause()

            assert screen.query_one("#personas-inspector-pane").display is True
            assert screen.query_one("#personas-inspector-rail-handle").display is False

    async def test_resize_sync_skips_work_when_compact_state_is_unchanged(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test(size=(80, 40)) as pilot:
            screen = await _mounted(pilot)
            assert screen.query_one("#personas-workbench").has_class(
                "personas-workbench-compact"
            )

            def fail_query(*_args, **_kwargs):
                raise AssertionError("unchanged compact state should not query panes")

            monkeypatch.setattr(screen, "query_one", fail_query)
            screen._sync_responsive_workbench()

    async def test_resize_hook_has_google_style_docstring(self):
        docstring = inspect.getdoc(PersonasScreen.on_resize)

        assert docstring is not None
        assert "Args:" in docstring
        assert "event:" in docstring

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
            assert "blocked" in readiness

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
        """Blocked saves render in the editor footer; the inspector says
        "editing..." while an editor is open (the footer is the single
        in-editor validation surface)."""
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
            from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
                CharacterSaveRequested,
            )
            screen.post_message(CharacterSaveRequested({"name": "", "first_message": "Hi"}))
            await pilot.pause()
            editor_validation = screen.query_one(
                "#personas-char-editor-validation", Static
            )
            assert "name: required" in str(editor_validation.renderable)
            summary = screen.query_one("#personas-validation-summary", Static)
            assert "Validation: editing..." in str(summary.renderable)
            assert "OK" not in str(summary.renderable)
        assert created == []

    async def test_character_book_errors_render_in_editor_footer(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """Screen-side validation (character_book) renders in the editor footer."""
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
            from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
                CharacterSaveRequested,
            )
            # entries is required (and must be a list) when a book is present.
            screen.post_message(
                CharacterSaveRequested(
                    {"name": "Bookish", "character_book": {"entries": "nope"}}
                )
            )
            await pilot.pause()
            editor_validation = screen.query_one(
                "#personas-char-editor-validation", Static
            )
            assert "character_book" in str(editor_validation.renderable)
            summary = screen.query_one("#personas-validation-summary", Static)
            assert "Validation: editing..." in str(summary.renderable)
        assert created == []

    async def test_inspector_validation_reads_editing_while_editor_open(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """Editor open -> inspector "editing..."; save success -> back to OK."""
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            EditCharacterRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            CharacterSaveRequested,
        )

        monkeypatch.setattr(
            character_handler_module, "update_character", lambda cid, data: True
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            summary = screen.query_one("#personas-validation-summary", Static)
            assert "Validation: OK" in str(summary.renderable)
            screen.post_message(EditCharacterRequested("1"))
            await pilot.pause()
            assert "Validation: editing..." in str(summary.renderable)
            screen.post_message(
                CharacterSaveRequested({"name": "Detective Sam", "version": 1})
            )
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "Validation: OK" in str(summary.renderable)

    async def test_inspector_validation_back_to_ok_on_editor_cancel(
        self, mock_app_instance, stub_characters
    ):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            EditCharacterRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            CharacterEditorCancelled,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            screen.post_message(EditCharacterRequested("1"))
            await pilot.pause()
            summary = screen.query_one("#personas-validation-summary", Static)
            assert "Validation: editing..." in str(summary.renderable)
            screen.post_message(CharacterEditorCancelled())
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "Validation: OK" in str(summary.renderable)

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
            from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
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
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
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
            from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
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

    async def test_personas_mode_service_failure_shows_recovery_state(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        stub_scope_service.list_persona_profiles.side_effect = RuntimeError("scope offline")

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)

            recovery = screen.query_one("#personas-service-error", Static)
            copy = str(recovery.renderable)
            assert "Persona profiles unavailable" in copy
            assert "Unavailable:" in copy
            assert "Recovery:" in copy
            assert "scope offline" in copy
            assert not list(screen.query("#personas-library-empty"))

    async def test_personas_mode_service_failure_replaces_stale_rows(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            assert [_row_text(r) for r in screen.query(".personas-library-row")] == ["Archivist"]

            stub_scope_service.list_persona_profiles.side_effect = RuntimeError("scope offline")
            screen._refresh_profile_rows_worker()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert screen.query_one("#personas-service-error", Static)
            assert not list(screen.query(".personas-library-row"))

    async def test_personas_mode_empty_state_copy_unchanged(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        stub_scope_service.list_persona_profiles.return_value = {"items": [], "total": 0}

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)

            empty = screen.query_one("#personas-library-empty", Static)
            assert str(empty.renderable) == "No persona profiles yet - use New to add one."
            assert not list(screen.query("#personas-service-error"))

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

    async def test_profile_save_refresh_failure_updates_status_row_and_recovery(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            assert "Personas: 1" in str(
                screen.query_one("#personas-status-row", Static).renderable
            )

            stub_scope_service.list_persona_profiles.side_effect = RuntimeError("scope offline")
            await pilot.click("#personas-library-new")
            await pilot.pause()
            screen.post_message(PersonaProfileSaveRequested({"name": "Mentor"}))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert screen.query_one("#personas-service-error", Static)
            assert not list(screen.query(".personas-library-row"))
            assert "Personas: 0" in str(
                screen.query_one("#personas-status-row", Static).renderable
            )

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
    async def _wait_for_search_render(self, pilot: Any) -> None:
        await pilot.pause(personas_screen_module.PERSONAS_SEARCH_DEBOUNCE_SECONDS + 0.05)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_search_input_debounces_rapid_changes(
        self,
        mock_app_instance: Any,
        stub_characters: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rapid search edits render only the final query after the debounce window."""

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()

            rendered_queries: list[str] = []
            original_render = screen._render_library_rows

            async def observe_render(
                *,
                expected_query: str | None = None,
                expected_mode: str | None = None,
            ) -> None:
                rendered_queries.append(screen.state.search_query)
                await original_render(
                    expected_query=expected_query,
                    expected_mode=expected_mode,
                )

            monkeypatch.setattr(screen, "_render_library_rows", observe_render)

            search_input = screen.query_one("#personas-library-search")
            search_input.value = "s"
            search_input.value = "sa"
            search_input.value = "sam"

            await pilot.pause(0.05)
            assert rendered_queries == []

            await self._wait_for_search_render(pilot)
            assert rendered_queries == ["sam"]
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam"]

    async def test_stale_fts_search_result_does_not_update_library_rows(
        self,
        mock_app_instance: Any,
        stub_characters: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A slower search result is dropped if the query changes while it awaits."""

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()

            screen.LIBRARY_FTS_THRESHOLD = 2
            library = screen.query_one("#personas-library-pane")
            original_update_rows = library.update_rows
            rendered_rows: list[tuple[str, ...]] = []

            async def observe_update_rows(rows: tuple[Any, ...], **kwargs: Any) -> None:
                rendered_rows.append(tuple(row.name for row in rows))
                await original_update_rows(rows, **kwargs)

            async def fake_to_thread(
                function: Any,
                query: str,
                *args: Any,
                **kwargs: Any,
            ) -> list[dict[str, Any]]:
                screen.state.search_query = "lab"
                return [{"id": 1, "name": "Detective Sam"}]

            monkeypatch.setattr(library, "update_rows", observe_update_rows)
            monkeypatch.setattr(personas_screen_module.asyncio, "to_thread", fake_to_thread)

            screen.state.search_query = "sam"
            await screen._render_search_query(query="sam", mode="characters")
            await pilot.pause()

            assert rendered_rows == []

    async def test_search_filters_loaded_characters_locally(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            # Type into the search input
            search_input = screen.query_one("#personas-library-search")
            search_input.value = "sam"
            await self._wait_for_search_render(pilot)
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
            await self._wait_for_search_render(pilot)
            # Then clear
            search_input.value = ""
            await self._wait_for_search_render(pilot)
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
            await self._wait_for_search_render(pilot)
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
            await self._wait_for_search_render(pilot)
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
            await self._wait_for_search_render(pilot)
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
            await self._wait_for_search_render(pilot)
            assert fts_calls == ["sam"]
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam"]

    async def test_fts_search_count_uses_unbounded_full_library_copy(
        self,
        mock_app_instance: Any,
        stub_characters: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """FTS search copy must not use the truncated loaded-page denominator.

        Args:
            mock_app_instance: Mounted test app fixture.
            stub_characters: Character-list fixture for the Personas library.
            monkeypatch: Pytest monkeypatch fixture used to force the FTS path.

        Returns:
            None.
        """

        def fake_fts(search_term: str, limit: int = 50) -> list[dict[str, Any]]:
            return [{"id": 1, "name": "Detective Sam"}]

        monkeypatch.setattr(character_handler_module, "search_characters_fts", fake_fts)

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            screen.LIBRARY_FTS_THRESHOLD = 2
            screen.query_one("#personas-library-search").value = "sam"
            await self._wait_for_search_render(pilot)

            count = str(screen.query_one("#personas-library-count", Static).renderable)
            assert count == "Showing 1 character match from full library"

    async def test_fts_search_runs_off_the_event_loop(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """The FTS query must not run synchronously on the UI loop."""
        import asyncio

        loop_seen: list[bool] = []

        def fake_fts(search_term: str, limit: int = 50):
            try:
                asyncio.get_running_loop()
                loop_seen.append(True)
            except RuntimeError:
                loop_seen.append(False)
            return [{"id": 1, "name": "Detective Sam"}]

        monkeypatch.setattr(character_handler_module, "search_characters_fts", fake_fts)

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            screen.LIBRARY_FTS_THRESHOLD = 2
            screen.query_one("#personas-library-search").value = "sam"
            await self._wait_for_search_render(pilot)
            assert loop_seen == [False]
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

    async def test_import_markdown_routes_through_character_import_helper(
        self, mock_app_instance, stub_characters, monkeypatch, tmp_path
    ):
        imported_paths: list[str] = []
        card_path = tmp_path / "card.md"
        card_path.write_text("# Character Card\n", encoding="utf-8")

        def fake_import(file_path):
            imported_paths.append(file_path)
            return 3

        monkeypatch.setattr(
            character_handler_module, "import_character_card", fake_import
        )

        def fetch_all_with_imported():
            characters = [dict(c) for c in CHARACTERS]
            if imported_paths:
                characters.append({"id": 3, "name": "Markdown Hero", "version": 1})
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
            await screen._import_character_from_path(str(card_path))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert imported_paths == [str(card_path)]
            assert screen.state.selected_entity_id == "3"

    async def test_import_invalid_markdown_uses_failure_path_without_selection_change(
        self, mock_app_instance, stub_characters, monkeypatch, tmp_path
    ):
        bad_path = tmp_path / "bad.md"
        bad_path.write_text("# Not a character card\n", encoding="utf-8")
        monkeypatch.setattr(
            character_handler_module, "import_character_card", lambda file_path: None
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await screen._import_character_from_path(str(bad_path))
            await pilot.pause()
            assert screen.state.selected_entity_id == "1"
            assert any(
                "valid character card" in message and severity == "error"
                for message, severity in notifications
            )

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

    async def test_stage_character_avatar_from_path_updates_editor_and_dirty_state(
        self, mock_app_instance, stub_characters, tmp_path
    ):
        avatar = tmp_path / "avatar.png"
        avatar.write_bytes(b"\x89PNG staged avatar")
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()

            await screen._stage_character_avatar_from_path(str(avatar))
            await pilot.pause()

            editor = screen.query_one(PersonasCharacterEditorWidget)
            assert editor.get_character_data()["image"] == b"\x89PNG staged avatar"
            assert (
                str(screen.query_one("#personas-char-editor-avatar-status", Static).renderable)
                == "Avatar: embedded"
            )
            assert screen.state.has_unsaved_changes is True

    async def test_stage_character_avatar_rejects_unsupported_extension_without_mutation(
        self, mock_app_instance, stub_characters, tmp_path
    ):
        bad = tmp_path / "avatar.txt"
        bad.write_text("not an image")
        app = PersonasTestApp(mock_app_instance)
        notifications = TestImportExport._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()

            await screen._stage_character_avatar_from_path(str(bad))
            await pilot.pause()

            editor = screen.query_one(PersonasCharacterEditorWidget)
            assert "image" not in editor.get_character_data()
            assert screen.state.has_unsaved_changes is False
            assert any("Unsupported avatar image type" in msg for msg, _ in notifications)

    async def test_stage_character_avatar_rejects_oversize_without_mutation(
        self, mock_app_instance, stub_characters, tmp_path
    ):
        avatar = tmp_path / "avatar.png"
        with avatar.open("wb") as avatar_file:
            avatar_file.truncate(personas_screen_module.PERSONAS_AVATAR_MAX_BYTES + 1)
        app = PersonasTestApp(mock_app_instance)
        notifications = TestImportExport._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()

            await screen._stage_character_avatar_from_path(str(avatar))
            await pilot.pause()

            editor = screen.query_one(PersonasCharacterEditorWidget)
            assert "image" not in editor.get_character_data()
            assert screen.state.has_unsaved_changes is False
            assert any("5 MB or smaller" in msg for msg, _ in notifications)

    async def test_stage_character_avatar_drops_stale_read_after_editor_restarts(
        self, mock_app_instance, stub_characters, monkeypatch, tmp_path
    ):
        avatar = tmp_path / "avatar.png"
        avatar.write_bytes(b"\x89PNG original avatar")
        app = PersonasTestApp(mock_app_instance)
        notifications = TestImportExport._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()

            async def fake_to_thread(function: Any, path: str) -> bytes:
                screen._finish_cancel_edit()
                await screen._begin_create_character()
                return b"\x89PNG stale avatar"

            monkeypatch.setattr(personas_screen_module.asyncio, "to_thread", fake_to_thread)

            await screen._stage_character_avatar_from_path(str(avatar))
            await pilot.pause()

            editor = screen.query_one(PersonasCharacterEditorWidget)
            assert "image" not in editor.get_character_data()
            assert screen.state.has_unsaved_changes is False
            assert not any("Avatar staged" in msg for msg, _ in notifications)

    async def test_stage_character_avatar_requires_open_editor(
        self, mock_app_instance, stub_characters, tmp_path
    ):
        avatar = tmp_path / "avatar.png"
        avatar.write_bytes(b"\x89PNG staged avatar")
        app = PersonasTestApp(mock_app_instance)
        notifications = TestImportExport._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()

            await screen._stage_character_avatar_from_path(str(avatar))
            await pilot.pause()

            assert screen.state.has_unsaved_changes is False
            assert any("Open a character editor" in msg for msg, _ in notifications)

    @staticmethod
    async def _open_persona_editor(pilot, mode: str):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-mode-personas")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        if mode == "create":
            await pilot.click("#personas-library-new")
            await pilot.pause()
        else:
            await pilot.click("#personas-library-row-persona_profile-p-1")
            await pilot.pause()
            screen.post_message(EditPersonaRequested("p-1"))
            await pilot.pause()
        assert screen._edit_mode == mode
        assert screen.state.has_unsaved_changes is False
        return screen

    @pytest.mark.parametrize("mode", ["create", "edit"])
    async def test_stage_character_avatar_ignores_persona_editor_session(
        self, mock_app_instance, stub_characters, stub_scope_service, tmp_path, mode
    ):
        avatar = tmp_path / f"persona-{mode}.png"
        avatar.write_bytes(b"\x89PNG persona editor avatar")
        app = PersonasTestApp(mock_app_instance)
        notifications = TestImportExport._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await self._open_persona_editor(pilot, mode)

            await screen._stage_character_avatar_from_path(str(avatar))
            await pilot.pause()

            editor = screen.query_one(PersonasCharacterEditorWidget)
            assert "image" not in editor.get_character_data()
            assert screen.state.has_unsaved_changes is False
            assert any("Open a character editor" in msg for msg, _ in notifications)

    @pytest.mark.parametrize("mode", ["create", "edit"])
    async def test_avatar_upload_request_ignores_persona_editor_session(
        self, mock_app_instance, stub_characters, stub_scope_service, mode
    ):
        calls: list[int] = []
        app = PersonasTestApp(mock_app_instance)
        notifications = TestImportExport._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await self._open_persona_editor(pilot, mode)

            def worker():
                calls.append(1)

                async def _noop():
                    pass

                return _noop()

            screen._avatar_upload_dialog_worker = worker
            screen.post_message(CharacterImageUploadRequested())
            await pilot.pause()
            await app.workers.wait_for_complete()

            assert calls == []
            assert screen._io_dialog_active is False
            assert screen.state.has_unsaved_changes is False
            assert any("Open a character editor" in msg for msg, _ in notifications)

    async def test_avatar_upload_request_launches_dialog_worker(
        self, mock_app_instance, stub_characters
    ):
        calls: list[int] = []
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()

            def worker():
                calls.append(1)

                async def _noop():
                    pass

                return _noop()

            screen._avatar_upload_dialog_worker = worker
            screen.post_message(CharacterImageUploadRequested())
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert calls == [1]

            screen.post_message(CharacterImageUploadRequested())
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert calls == [1]

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

    async def test_export_json_rejects_hidden_directory_destination(
        self, mock_app_instance, stub_characters, stub_db, monkeypatch, tmp_path
    ):
        """The JSON write path validates the destination like the PNG path."""
        monkeypatch.setattr(
            personas_screen_module,
            "export_character_card_to_json",
            lambda db, character_id, include_image=True: '{"name": "Detective Sam"}',
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            hidden_dir = tmp_path / ".sneaky"
            hidden_dir.mkdir()
            target = hidden_dir / "out.json"
            await screen._export_selected_character(str(target), fmt="json")
            await pilot.pause()
            assert not target.exists()
            assert any(
                "Export failed" in message and severity == "error"
                for message, severity in notifications
            )

    async def test_export_json_rejects_missing_destination_directory(
        self, mock_app_instance, stub_characters, stub_db, monkeypatch, tmp_path
    ):
        """A destination in a nonexistent directory fails readably."""
        monkeypatch.setattr(
            personas_screen_module,
            "export_character_card_to_json",
            lambda db, character_id, include_image=True: '{"name": "Detective Sam"}',
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            target = tmp_path / "missing" / "out.json"
            await screen._export_selected_character(str(target), fmt="json")
            await pilot.pause()
            assert not target.exists()
            assert any(
                "Export failed" in message and severity == "error"
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
        self.nav_contexts: list[dict[str, object]] = []

    def on_navigate_to_screen(self, message) -> None:
        self.nav_routes.append(message.screen_name)
        self.nav_contexts.append(dict(getattr(message, "screen_context", {}) or {}))


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

    async def test_conversations_panel_shows_loading_then_rows(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        """While the listing worker runs the panel says it is loading."""
        import threading

        release = threading.Event()

        def gated_listing(db, character_id, limit=50, offset=0):
            release.wait(timeout=5)
            return [{"id": "conv-1", "title": "First case"}]

        monkeypatch.setattr(
            conversations_controller_module,
            "list_character_conversations",
            gated_listing,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            panel = screen.query_one("#personas-conversations-list")
            texts = [str(s.renderable) for s in panel.query(Static)]
            assert any("Loading conversations..." in text for text in texts)
            release.set()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            rows = screen.query(".personas-conversation-row")
            assert [_row_text(r) for r in rows] == ["First case"]
            texts = [str(s.renderable) for s in panel.query(Static)]
            assert not any("Loading" in text for text in texts)

    async def test_conversations_panel_empty_shows_copy(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        monkeypatch.setattr(
            conversations_controller_module,
            "list_character_conversations",
            lambda db, character_id, limit=50, offset=0: [],
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            panel = screen.query_one("#personas-conversations-list")
            texts = [str(s.renderable) for s in panel.query(Static)]
            assert any("No saved conversations." in text for text in texts)
            assert list(screen.query(".personas-conversation-row")) == []

    async def test_open_conversation_shows_loading_placeholder(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        """Clicking a conversation gives instant feedback: the transcript view
        opens immediately with a loading placeholder."""
        import threading

        release = threading.Event()

        def gated_messages(db, conversation_id, character_name, user_name, **kwargs):
            release.wait(timeout=5)
            return [("Hello there", "Greetings, detective.")]

        monkeypatch.setattr(
            conversations_controller_module,
            "retrieve_conversation_messages_for_ui",
            gated_messages,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            await pilot.click("#personas-conversation-row-conv-1")
            await pilot.pause()
            view = screen.query_one("#personas-conversation-transcript-view")
            assert view.display is True
            texts = [str(s.renderable) for s in view.query(Static)]
            assert any("Loading transcript..." in text for text in texts)
            release.set()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            texts = [str(s.renderable) for s in view.query(Static)]
            assert not any("Loading transcript..." in text for text in texts)
            assert any("Greetings, detective." in text for text in texts)

    async def test_transcript_lines_use_speaker_names(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """The read-only transcript uses You/<character name>, not user/assistant."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            view = screen.query_one("#personas-conversation-transcript-view")
            texts = [
                str(line.renderable)
                for line in view.query(".personas-transcript-line")
            ]
            assert texts == [
                "You: Hello there",
                "Detective Sam: Greetings, detective.",
            ]

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
            assert app.nav_routes == [TAB_LIBRARY]
            assert app.nav_contexts == [
                {
                    LIBRARY_NAV_CONTEXT_MODE: LIBRARY_MODE_CONVERSATIONS,
                    LIBRARY_NAV_CONTEXT_CONVERSATION_ID: "conv-1",
                }
            ]

    async def test_open_in_library_requires_open_conversation(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        notifications: list[tuple[str, str]] = []
        app = _NavCaptureApp(mock_app_instance)
        app.notify = lambda message, severity="information", **kwargs: notifications.append(
            (str(message), severity)
        )
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            screen.conversations.open_in_library()
            await pilot.pause()

        assert app.nav_routes == []
        assert any(
            "Open a conversation" in message and severity == "warning"
            for message, severity in notifications
        )

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

    async def test_screen_gate_controls_visible_console_actions_after_selection(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            assert screen.query_one("#personas-attach-to-console", Button).disabled is False

            screen._console_action_allowed = lambda: False
            screen._console_action_block_reason = lambda: "prompts are not attachable"
            screen._register_footer_shortcuts()
            await pilot.pause()

            assert screen.query_one("#personas-attach-to-console", Button).disabled is True
            assert screen.query_one("#personas-start-chat", Button).disabled is True
            assert "Console blocked: prompts are not attachable" in str(
                screen.query_one("#personas-readiness-console", Static).renderable
            )

    async def test_selection_pushes_console_gate_before_async_followup(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        """Selection should not render as blocked before follow-up work completes."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            inspector = screen.query_one(PersonasInspectorPane)

            observed_enabled: list[bool] = []
            original_loading = inspector.show_conversations_loading

            async def assert_gate_synced_before_loading():
                observed_enabled.append(
                    not screen.query_one("#personas-attach-to-console", Button).disabled
                )
                return await original_loading()

            monkeypatch.setattr(
                inspector,
                "show_conversations_loading",
                assert_gate_synced_before_loading,
            )

            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

        assert observed_enabled == [True]

    async def test_character_save_pushes_console_gate_before_reload(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        """Save completion should expose valid Console actions before reload awaits."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()

            observed_enabled: list[bool] = []

            async def observe_load(_character_id):
                observed_enabled.append(
                    not screen.query_one("#personas-attach-to-console", Button).disabled
                )

            monkeypatch.setattr(screen.character_handler, "load_character", observe_load)

            await screen._after_character_save("1", "Detective Sam")
            await pilot.pause()

        assert observed_enabled == [True]

    async def test_profile_save_pushes_console_gate_before_row_render(
        self, mock_app_instance, stub_characters, stub_conversations, stub_scope_service, monkeypatch
    ):
        """Profile save completion should not wait for row rendering to sync gates."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            observed_enabled: list[bool] = []

            async def observe_render_rows():
                observed_enabled.append(
                    not screen.query_one("#personas-attach-to-console", Button).disabled
                )

            monkeypatch.setattr(screen, "_render_profile_rows", observe_render_rows)

            await screen._after_profile_save({"id": "p-1", "name": "Archivist"})
            await pilot.pause()

        assert observed_enabled == [True]

    async def test_attach_blocked_with_unsaved_edits(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            EditCharacterRequested,
        )

        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.post_message(EditCharacterRequested("1"))
            await pilot.pause()
            assert screen._edit_mode == "edit"
            # Change-based dirty tracking: make a real edit first.
            from textual.widgets import TextArea

            screen.query_one("#personas-char-editor-description", TextArea).text = "edited"
            await pilot.pause()
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
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """The attach hint is truthful: available only with a saved selection."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            context = screen._shortcut_context()
            attach = next(a for a in context.actions if a.label == "attach")
            assert attach.key == "ctrl+enter"
            assert attach.available is False  # nothing selected yet
            await self._select_first_character(pilot)
            attach = next(
                a for a in screen._shortcut_context().actions if a.label == "attach"
            )
            assert attach.available is True


class _FakePreviewGateway:
    """In-memory gateway double: ready resolution + scripted stream.

    ``gate`` holds the stream BEFORE the first chunk; ``mid_gate`` holds it
    between the first chunk and the rest; ``error`` raises before any chunk
    unless ``error_after_first`` is set, in which case the first chunk is
    yielded and the error raised on the next pull. ``stream_failures`` makes
    that many stream_chat calls raise before any chunk, then later calls
    succeed (exercises the non-streaming retry). ``selections`` records every
    selection passed to resolve_for_send so tests can assert the retry's
    ``streaming`` flag.
    """

    def __init__(
        self,
        chunks=("Hello, ", "world."),
        gate=None,
        mid_gate=None,
        error=None,
        error_after_first=False,
        stream_failures=0,
    ):
        self.chunks = chunks
        self.gate = gate
        self.mid_gate = mid_gate
        self.error = error
        self.error_after_first = error_after_first
        self.stream_failures = stream_failures
        self.requests: list[list[dict]] = []
        self.selections: list = []
        self.closed = False

    async def resolve_for_send(self, selection):
        from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution

        self.selections.append(selection)
        return ConsoleProviderResolution(
            provider="openai", base_url="", model="test-model", ready=True
        )

    async def stream_chat(self, resolution, messages):
        self.requests.append([dict(m) for m in messages])
        if self.gate is not None:
            await self.gate.wait()
        if self.stream_failures > 0:
            self.stream_failures -= 1
            raise RuntimeError("stream transport failed")
        if self.error is not None and not self.error_after_first:
            raise self.error
        for index, chunk in enumerate(self.chunks):
            if index == 1 and self.mid_gate is not None:
                await self.mid_gate.wait()
            yield chunk
            if index == 0 and self.error is not None and self.error_after_first:
                raise self.error

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

    async def test_reply_streams_progressively_into_one_line(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Chunks render as they arrive, updating ONE growing character line."""
        import asyncio

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        mid_gate = asyncio.Event()
        fake = _FakePreviewGateway(mid_gate=mid_gate)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen._ensure_preview_gateway = lambda: fake
            pane = screen.query_one(PersonasPreviewPane)
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            # The first chunk must be visible WHILE the stream is held open.
            for _ in range(50):
                if "character: Hello, " in pane.transcript_text():
                    break
                await pilot.pause()
            assert "character: Hello, " in pane.transcript_text()
            # History gets the consolidated entry only at the end.
            assert not any(
                entry["role"] == "assistant" for entry in screen._preview_history
            )
            mid_gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            lines = pane.transcript_text().splitlines()
            assert lines.count("character: Hello, world.") == 1
            assert "character: Hello, " not in [
                line for line in lines if line != "character: Hello, world."
            ]
            assert screen._preview_history[-1] == {
                "role": "assistant",
                "content": "Hello, world.",
            }

    async def test_reset_mid_stream_removes_partial_line(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Reset after the first chunk landed must drop the partial line."""
        import asyncio

        from textual.widgets import Button as _Button

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        mid_gate = asyncio.Event()
        fake = _FakePreviewGateway(mid_gate=mid_gate)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen._ensure_preview_gateway = lambda: fake
            pane = screen.query_one(PersonasPreviewPane)
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            for _ in range(50):
                if "character: Hello, " in pane.transcript_text():
                    break
                await pilot.pause()
            assert "character: Hello, " in pane.transcript_text()
            screen.query_one("#personas-preview-reset", _Button).press()
            await pilot.pause()
            mid_gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "Hello" not in pane.transcript_text()
            assert pane.transcript_text() == (
                "character: The name's Detective Sam. Who's asking?"
            )
            assert not any(
                entry["role"] == "assistant" for entry in screen._preview_history
            )

    async def test_selection_change_mid_stream_removes_partial_line(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A selection move after the first chunk must drop the partial line."""
        import asyncio

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        mid_gate = asyncio.Event()
        fake = _FakePreviewGateway(mid_gate=mid_gate)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen._ensure_preview_gateway = lambda: fake
            pane = screen.query_one(PersonasPreviewPane)
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            for _ in range(50):
                if "character: Hello, " in pane.transcript_text():
                    break
                await pilot.pause()
            assert "character: Hello, " in pane.transcript_text()
            await pilot.click("#personas-library-row-character-2")
            await pilot.pause()
            mid_gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "Hello" not in pane.transcript_text()
            assert not any(
                entry["role"] == "assistant" for entry in screen._preview_history
            )

    async def test_error_mid_stream_removes_partial_line(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A provider error after the first chunk must not leave a dangling
        partial line; status shows the recovery copy and the orphaned user
        history entry is popped."""
        from textual.widgets import Static as _Static

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        fake = _FakePreviewGateway(
            error=RuntimeError("provider exploded"), error_after_first=True
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen._ensure_preview_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)
            assert "Hello" not in pane.transcript_text()
            assert screen._preview_history == []
            status = str(
                screen.query_one("#personas-preview-status", _Static).renderable
            )
            assert "Provider error" in status

    async def test_draft_aware_system_prompt_uses_editor_data(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        from textual.widgets import TextArea as _TextArea

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            EditCharacterRequested,
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
            screen.query_one("#personas-preview-input", Input).value = "Hi"
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

    async def test_stream_failure_falls_back_to_non_streaming_once(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A failed stream retries exactly once with streaming disabled."""
        from textual.widgets import Static as _Static

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        fake = _FakePreviewGateway(stream_failures=1)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen._ensure_preview_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            # Both attempts resolved; the retry disabled streaming.
            assert [s.streaming for s in fake.selections] == [True, False]
            assert len(fake.requests) == 2
            pane = screen.query_one(PersonasPreviewPane)
            assert "character: Hello, world." in pane.transcript_text()
            assert screen._preview_history[-1] == {
                "role": "assistant",
                "content": "Hello, world.",
            }
            assert (
                str(screen.query_one("#personas-preview-status", _Static).renderable)
                == "Ready"
            )

    async def test_both_attempts_failing_keeps_error_semantics(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """When the non-streaming retry also fails the existing error
        semantics apply: orphan user turn popped, readable error status."""
        from textual.widgets import Static as _Static

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        fake = _FakePreviewGateway(error=RuntimeError("provider exploded"))
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen._ensure_preview_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            # Streaming attempt + one non-streaming retry, no more.
            assert [s.streaming for s in fake.selections] == [True, False]
            assert len(fake.requests) == 2
            assert screen._preview_history == []
            pane = screen.query_one(PersonasPreviewPane)
            assert "Hello" not in pane.transcript_text()
            status = str(
                screen.query_one("#personas-preview-status", _Static).renderable
            )
            assert "Provider error" in status

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


class TestKeyboardInteraction:
    """UX-E2: context-sensitive Escape, real Ctrl+S, mode keys, managed focus."""

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

    @staticmethod
    def _bypass_confirm(screen, answer: bool) -> list[bool]:
        """Stub the unsaved-changes confirm; returns a call log."""
        calls: list[bool] = []

        async def fake_confirm() -> bool:
            calls.append(answer)
            return answer

        screen._confirm_discard_unsaved = fake_confirm
        return calls

    async def _open_create_editor(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert screen._edit_mode == "create"
        return screen

    async def test_escape_cancels_editor_via_guard(
        self, mock_app_instance, stub_characters
    ):
        """Esc in the editor takes the SAME guarded cancel path as the button."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._open_create_editor(pilot)
            # Type into the focused Name input first: dirty tracking is
            # change-based, so a pristine editor would cancel dialog-free.
            await pilot.press("x")
            await pilot.pause()
            confirms = self._bypass_confirm(screen, True)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # The guard was consulted (real edit present), not bypassed.
            assert confirms == [True]
            assert screen._edit_mode == "view"
            assert screen.query_one("#ccp-character-editor-view").display is False

    async def test_escape_keeps_editor_when_guard_declined(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._open_create_editor(pilot)
            await pilot.press("x")  # real edit; the guard must fire
            await pilot.pause()
            confirms = self._bypass_confirm(screen, False)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert confirms == [False]
            assert screen._edit_mode == "create"
            assert screen.query_one("#ccp-character-editor-view").display is True

    async def test_escape_in_transcript_returns_to_card(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-conversation-row-conv-1")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            transcript = screen.query_one("#personas-conversation-transcript-view")
            assert transcript.display is True
            # Opening a transcript focuses its scroll so arrow keys scroll it.
            focused = pilot.app.focused
            assert focused is not None and focused.id == "personas-transcript-scroll"
            await pilot.press("escape")
            await pilot.pause()
            assert transcript.display is False
            assert screen.query_one("#ccp-character-card-view").display is True
            # Back returns focus to the conversations list in the inspector.
            focused = pilot.app.focused
            assert focused is not None and focused.id == "personas-conversations-list"

    async def test_escape_blurs_search_input(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.press("ctrl+f")
            await pilot.pause()
            assert pilot.app.focused.id == "personas-library-search"
            await pilot.press("escape")
            await pilot.pause()
            focused = pilot.app.focused
            assert focused is not None and focused.id == "personas-library-rows"
            # Still in view mode; nothing else changed.
            assert screen._edit_mode == "view"

    async def test_ctrl_s_saves_from_editor(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        created = []
        monkeypatch.setattr(
            character_handler_module, "create_character",
            lambda data: created.append(data) or 99,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._open_create_editor(pilot)
            screen.query_one("#personas-char-editor-name", Input).value = "New Hero"
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen._edit_mode == "view"
        assert created and created[0]["name"] == "New Hero"

    async def test_save_persists_staged_avatar_bytes(
        self, mock_app_instance, stub_characters, monkeypatch, tmp_path
    ):
        avatar = tmp_path / "avatar.png"
        avatar.write_bytes(b"\x89PNG staged avatar")
        created: list[dict[str, Any]] = []
        monkeypatch.setattr(
            character_handler_module,
            "create_character",
            lambda data: created.append(dict(data)) or 99,
        )
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            screen.query_one("#personas-char-editor-name", Input).value = "Avatar Hero"
            await screen._stage_character_avatar_from_path(str(avatar))
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

        assert created
        assert created[0]["name"] == "Avatar Hero"
        assert created[0]["image"] == b"\x89PNG staged avatar"

    async def test_ctrl_s_noop_in_view_mode(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        created = []
        monkeypatch.setattr(
            character_handler_module, "create_character",
            lambda data: created.append(data) or 99,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen._edit_mode == "view"
        assert created == []

    async def test_footer_save_hint_flips_with_edit_mode(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()

            def _save_action(context):
                return next(a for a in context.actions if a.label == "save")

            assert _save_action(screen._shortcut_context()).available is False
            await pilot.press("ctrl+n")
            await pilot.pause()
            assert _save_action(screen._shortcut_context()).available is True
            # The footer was re-registered on the transition.
            footer = pilot.app.query_one(AppFooterStatus)
            assert "ctrl+s save unavailable" not in footer.shortcut_text
            assert "ctrl+s save" in footer.shortcut_text
            confirms = self._bypass_confirm(screen, True)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # A pristine create session cancels without the discard dialog
            # (change-based dirty tracking).
            assert confirms == []
            assert _save_action(screen._shortcut_context()).available is False
            assert "ctrl+s save unavailable" in footer.shortcut_text

    async def test_mode_keys_switch_modes(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.press("ctrl+2")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.active_mode == "personas"
            # ]/[ cycle through the strip order from the active mode.
            await pilot.press("right_square_bracket")
            await pilot.pause()
            assert screen.state.active_mode == "prompts"
            await pilot.press("left_square_bracket")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.active_mode == "personas"

    async def test_focus_lands_in_editor_name_on_create(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            await self._open_create_editor(pilot)
            await pilot.pause()
            focused = pilot.app.focused
            assert focused is not None
            assert focused.id == "personas-char-editor-name"

    async def test_focus_returns_to_library_after_cancel(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._open_create_editor(pilot)
            self._bypass_confirm(screen, True)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            focused = pilot.app.focused
            assert focused is not None
            assert focused.id == "personas-library-rows"


class TestDirtyTracking:
    """UX-E3: change-based dirty tracking, live header state, row badges."""

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

    @staticmethod
    def _bypass_confirm(screen, answer: bool) -> list[bool]:
        """Stub the unsaved-changes confirm; returns a call log."""
        calls: list[bool] = []

        async def fake_confirm() -> bool:
            calls.append(answer)
            return answer

        screen._confirm_discard_unsaved = fake_confirm
        return calls

    async def _edit_first_character(self, pilot):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            EditCharacterRequested,
        )

        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-row-character-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        screen.post_message(EditCharacterRequested("1"))
        await pilot.pause()
        assert screen._edit_mode == "edit"
        return screen

    @staticmethod
    async def _type_in_description(pilot, screen):
        from textual.widgets import TextArea

        screen.query_one("#personas-char-editor-description", TextArea).focus()
        await pilot.pause()
        await pilot.press("x")
        await pilot.pause()

    async def test_edit_without_changes_switches_without_dialog(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Edit then click away with zero keystrokes: no discard dialog."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._edit_first_character(pilot)
            assert screen.state.has_unsaved_changes is False
            confirms = self._bypass_confirm(screen, True)
            await pilot.click("#personas-library-row-character-2")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert confirms == []
            assert screen.state.selected_entity_id == "2"
            assert screen._edit_mode == "view"

    async def test_typing_marks_dirty_and_guard_fires(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._edit_first_character(pilot)
            await self._type_in_description(pilot, screen)
            assert screen.state.has_unsaved_changes is True
            confirms = self._bypass_confirm(screen, True)
            await pilot.click("#personas-library-row-character-2")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert confirms == [True]
            assert screen.state.selected_entity_id == "2"

    async def test_persona_editor_typing_marks_dirty_and_guard_fires(
        self, mock_app_instance, stub_characters, stub_conversations, stub_scope_service
    ):
        """Carryover: PersonaProfileEditorWidget._field_changed parity with the
        character editor — typing posts EditorContentChanged exactly once, the
        screen marks the session unsaved, and leaving consults the guard."""
        from textual.widgets import TextArea

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-library-row-persona_profile-p-1")
            await pilot.pause()
            screen.post_message(EditPersonaRequested("p-1"))
            await pilot.pause()
            assert screen._edit_mode == "edit"
            # Programmatic population must not have marked the session dirty.
            assert screen.state.has_unsaved_changes is False
            screen.query_one("#personas-editor-description", TextArea).focus()
            await pilot.pause()
            await pilot.press("x")
            await pilot.pause()
            assert screen.state.has_unsaved_changes is True
            readiness = str(
                screen.query_one("#personas-readiness-console", Static).renderable
            )
            assert "unsaved" in readiness
            confirms = self._bypass_confirm(screen, True)
            await pilot.click("#personas-mode-characters")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert confirms == [True]
            assert screen.state.active_mode == "characters"

    async def test_programmatic_load_does_not_mark_dirty(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Populating the editor (load/new) must not count as a user change."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._edit_first_character(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.has_unsaved_changes is False
            # Same for a fresh create session (new_character population).
            await pilot.press("ctrl+n")
            await pilot.pause()
            await pilot.pause()
            assert screen._edit_mode == "create"
            assert screen.state.has_unsaved_changes is False

    async def test_title_reflects_editing_state(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        monkeypatch.setattr(
            character_handler_module, "update_character", lambda cid, data: True
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._edit_first_character(pilot)
            title = screen.query_one("#personas-title", Static)
            text = str(title.renderable)
            assert "Editing Detective Sam" in text
            assert "unsaved" not in text
            await self._type_in_description(pilot, screen)
            assert "Editing Detective Sam - unsaved" in str(title.renderable)
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # Back to Ready; "Local" stays out of the title (the status row
            # already carries "Source: Local").
            assert (
                str(title.renderable)
                == "Personas | Behavior profiles for chat and agents | Ready"
            )

    async def test_active_row_gets_unsaved_badge(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        monkeypatch.setattr(
            character_handler_module, "update_character", lambda cid, data: True
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._edit_first_character(pilot)
            row = screen.query_one("#personas-library-row-character-1")
            assert "is-unsaved" not in row.classes
            await self._type_in_description(pilot, screen)
            assert "is-unsaved" in row.classes
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # The save refresh rebuilds the rows; the badge must be gone.
            row = screen.query_one("#personas-library-row-character-1")
            assert "is-unsaved" not in row.classes

    async def test_unsaved_badge_cleared_on_discarded_switch(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Discarding edits while switching rows must drop the stale badge."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._edit_first_character(pilot)
            await self._type_in_description(pilot, screen)
            row = screen.query_one("#personas-library-row-character-1")
            assert "is-unsaved" in row.classes
            self._bypass_confirm(screen, True)
            await pilot.click("#personas-library-row-character-2")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.selected_entity_id == "2"
            assert not screen.query(".personas-library-row.is-unsaved")

    async def test_import_reregisters_footer(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        """UX-E2 carryover: import-selection must refresh the attach hint."""
        monkeypatch.setattr(
            character_handler_module, "import_character_card", lambda path: 1
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            attach = next(
                a for a in screen._shortcut_context().actions if a.label == "attach"
            )
            assert attach.available is False  # no prior selection
            footer = pilot.app.query_one(AppFooterStatus)
            assert "ctrl+enter attach unavailable" in footer.shortcut_text
            await screen._import_character_from_path("/tmp/card.json")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            attach = next(
                a for a in screen._shortcut_context().actions if a.label == "attach"
            )
            assert attach.available is True
            assert "ctrl+enter attach unavailable" not in footer.shortcut_text
            assert "ctrl+enter attach" in footer.shortcut_text


class TestConfirmationDialogEscape:
    """Keyboard users must be able to dismiss the shared confirm dialog."""

    async def test_confirmation_dialog_escape_cancels(self):
        from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

        results: list[bool] = []

        class DialogApp(App):
            def on_mount(self) -> None:
                self.push_screen(ConfirmationDialog(), callback=results.append)

        app = DialogApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(pilot.app.screen, ConfirmationDialog)
            await pilot.press("escape")
            await pilot.pause()
            assert results == [False]
            assert not isinstance(pilot.app.screen, ConfirmationDialog)


class TestImportExportFilters:
    """The file-picker filters must use callable testers, not glob strings.

    Regression guard for the P0 crash: ``Filter.__call__`` does
    ``self.tester(path)``; a glob STRING tester ("*.json") raises
    ``TypeError: 'str' object is not callable`` and tears down the session
    when Import / Export JSON / Export PNG is pressed. We drive the real
    import/export workers, capture the picker actually built, and assert
    every filter is callable and returns a bool without raising.
    """

    @staticmethod
    def _assert_filters_callable(filters) -> None:
        from pathlib import Path

        # ``selections`` enumerates every registered filter; each must be a
        # callable tester that survives being invoked on a Path.
        assert bool(filters)
        for _name, filter_id in filters.selections:
            entry = filters[filter_id]
            for sample in (Path("x.json"), Path("x.png"), Path("x.txt")):
                result = entry(sample)
                assert isinstance(result, bool)

    async def _capture_picker(self, pilot, screen, launch):
        from unittest.mock import AsyncMock

        captured: dict = {}

        async def _fake_push_screen_wait(picker):
            captured["picker"] = picker
            return None  # user cancels; the worker returns cleanly

        pilot.app.push_screen_wait = AsyncMock(side_effect=_fake_push_screen_wait)
        await launch()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert "picker" in captured, "picker was never pushed"
        return captured["picker"]

    async def test_import_filters_are_callable(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            picker = await self._capture_picker(
                pilot, screen, screen._import_dialog_worker
            )
            self._assert_filters_callable(picker.filters)

    async def test_import_filters_include_markdown(
        self, mock_app_instance, stub_characters
    ):
        from pathlib import Path

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            picker = await self._capture_picker(
                pilot, screen, screen._import_dialog_worker
            )
            filter_by_name = {
                name: picker.filters[filter_id]
                for name, filter_id in picker.filters.selections
            }

            assert "Markdown Files" in filter_by_name
            assert filter_by_name["Character Cards"](Path("character.md")) is True
            assert filter_by_name["Character Cards"](Path("character.markdown")) is True
            assert filter_by_name["Markdown Files"](Path("character.md")) is True
            assert filter_by_name["Markdown Files"](Path("character.markdown")) is True
            assert filter_by_name["Markdown Files"](Path("character.json")) is False

    async def test_export_json_filters_are_callable(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            picker = await self._capture_picker(
                pilot, screen, lambda: screen._export_dialog_worker("json")
            )
            self._assert_filters_callable(picker.filters)

    async def test_export_png_filters_are_callable(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            picker = await self._capture_picker(
                pilot, screen, lambda: screen._export_dialog_worker("png")
            )
            self._assert_filters_callable(picker.filters)

    @pytest.mark.parametrize("outcome", ["cancel", "error", "selection"])
    async def test_avatar_upload_worker_resets_flag_and_uses_avatar_context(
        self, mock_app_instance, stub_characters, tmp_path, outcome
    ):
        avatar = tmp_path / "avatar.png"
        avatar.write_bytes(b"\x89PNG selected avatar")
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            captured: dict = {}

            async def _fake_push_screen_wait(picker):
                captured["picker"] = picker
                if outcome == "error":
                    raise RuntimeError("dialog failed")
                if outcome == "selection":
                    return avatar
                return None

            pilot.app.push_screen_wait = AsyncMock(side_effect=_fake_push_screen_wait)
            screen._io_dialog_active = True

            await screen._avatar_upload_dialog_worker()
            await pilot.pause()

            assert screen._io_dialog_active is False
            picker = captured["picker"]
            assert picker.context == "character_avatar_upload"
            self._assert_filters_callable(picker.filters)
            editor = screen.query_one(PersonasCharacterEditorWidget)
            if outcome == "selection":
                assert editor.get_character_data()["image"] == b"\x89PNG selected avatar"
            else:
                assert "image" not in editor.get_character_data()
