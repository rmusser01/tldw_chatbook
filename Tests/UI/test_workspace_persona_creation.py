"""Native workspace creation/default controls retain explicit Persona choices."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, Checkbox, Input, Select, Static

from Tests.Workspaces.test_agent_provisioning import StubPersonaService, build
from tldw_chatbook.Widgets.workspace_create_modal import WorkspaceCreateModal


class Personas(StubPersonaService):
    def list_persona_profiles(self):
        return [{"id": "author", "name": "Author", "system_prompt": "Write clearly."}]

    def get_persona_profile(self, persona_id):
        return next(
            (item for item in self.list_persona_profiles() if item["id"] == persona_id),
            None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["none", "author"])
async def test_create_choice_survives_folder_recompose_and_saves(tmp_path, choice):
    personas = Personas()
    registry, _permissions = build(tmp_path, personas)
    app = App()
    async with app.run_test(size=(90, 36)) as pilot:
        await app.push_screen(
            WorkspaceCreateModal(registry_service=registry, persona_service=personas)
        )
        modal = app.screen
        modal.query_one("#workspace-create-name", Input).value = "Chosen"
        modal.query_one("#workspace-default-persona", Select).value = choice
        modal.query_one("#workspace-create-folder-path", Input).value = str(tmp_path)
        modal.query_one("#workspace-create-folder-add", Button).press()
        await pilot.pause()
        assert modal.query_one("#workspace-default-persona", Select).value == choice
        modal.query_one("#workspace-create-confirm", Button).press()
        await pilot.pause()
        saved = next(
            item for item in registry.list_workspaces() if item.name == "Chosen"
        )
        if choice == "none":
            assert saved.assistant_defaults is None
            assert saved.assistant_defaults_explicit_none
        else:
            assert saved.assistant_defaults.assistant_id == "author"
        assert personas.created == []


@pytest.mark.asyncio
async def test_read_write_create_requires_visible_confirmation(tmp_path):
    personas = Personas()
    registry, _permissions = build(tmp_path, personas)
    app = App()
    async with app.run_test(size=(90, 36)) as pilot:
        await app.push_screen(
            WorkspaceCreateModal(registry_service=registry, persona_service=personas)
        )
        modal = app.screen
        modal.query_one("#workspace-default-persona", Select).value = "author"
        modal.query_one("#workspace-default-memory", Select).value = "read_write"
        modal.query_one("#workspace-create-confirm", Button).press()
        await pilot.pause()
        assert registry.list_workspaces() == ()
        assert "Confirm" in str(
            modal.query_one("#workspace-create-error", Static).render()
        )
        modal.query_one("#workspace-default-memory-confirm", Checkbox).value = True
        modal.query_one("#workspace-create-confirm", Button).press()
        await pilot.pause()
        assert (
            registry.list_workspaces()[0].assistant_defaults.persona_memory_mode
            == "read_write"
        )


@pytest.mark.asyncio
async def test_default_modal_cancel_preserves_and_apply_clears(tmp_path):
    from tldw_chatbook.Widgets.workspace_persona_default import (
        WorkspacePersonaDefaultModal,
    )

    personas = Personas()
    registry, _permissions = build(tmp_path, personas)
    before = registry.create_workspace(workspace_id="chosen", name="Chosen")
    app = App()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(
            WorkspacePersonaDefaultModal(registry, personas, "chosen")
        )
        app.screen.query_one("#workspace-default-persona", Select).value = "none"
        app.screen.query_one("#workspace-default-cancel", Button).press()
        await pilot.pause()
        assert (
            registry.get_workspace("chosen").assistant_defaults
            == before.assistant_defaults
        )
        await app.push_screen(
            WorkspacePersonaDefaultModal(registry, personas, "chosen")
        )
        app.screen.query_one("#workspace-default-persona", Select).value = "none"
        app.screen.query_one("#workspace-default-apply", Button).press()
        await pilot.pause()
        assert registry.get_workspace("chosen").assistant_defaults is None
        assert registry.get_workspace("chosen").assistant_defaults_explicit_none


@pytest.mark.asyncio
async def test_console_bootstrap_workspace_activation_and_details_use_target_default():
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_console_workspace_action_row_geometry import StyledConsoleHarness
    from tldw_chatbook.Widgets.workspace_persona_default import (
        WorkspacePersonaDefaultModal,
    )
    from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults

    app = _build_test_app()
    app.local_character_persona_service = Personas()
    registry = app.workspace_registry_service
    defaults = WorkspaceAssistantDefaults(assistant_id="author")
    registry.create_workspace(
        workspace_id="first", name="First", assistant_defaults=defaults
    )
    registry.create_workspace(
        workspace_id="second", name="Second", assistant_defaults=defaults
    )
    registry.set_active_workspace("first")
    host = StyledConsoleHarness(app)
    async with host.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        screen = host.screen
        store = screen._ensure_console_chat_store()
        first = screen._session._active_native_console_session()
        assert first.workspace_id == "first"
        assert first.assistant_id == "author"
        screen._workspace._activate_console_session_for_workspace("second")
        second = screen._session._active_native_console_session()
        assert second.workspace_id == "second"
        assert second.assistant_id == "author"
        assert (
            second.persona_memory_mode
            == second.settings.persona_memory_mode
            == "read_only"
        )
        registry.set_active_workspace("second")
        await screen._sync_native_console_chat_ui()
        screen._set_console_rail_preference(left_open=True)
        if not screen._current_console_rail_state().details_open:
            screen._toggle_console_rail_section("details")
        await pilot.pause()
        button = screen.query_one("#console-workspace-default-persona", Button)
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        button.press()
        await pilot.pause()
        assert isinstance(host.screen, WorkspacePersonaDefaultModal)
        host.screen.query_one("#workspace-default-persona", Select).value = "none"
        host.screen.query_one("#workspace-default-apply", Button).press()
        await pilot.pause()
        assert registry.get_workspace("second").assistant_defaults is None
        assert store.ensure_session().assistant_id == "author"
        future = store.create_session(workspace_id="second")
        assert future.assistant_kind == "generic"
