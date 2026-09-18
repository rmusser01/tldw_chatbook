"""Workspace controls preserve real local Persona identity and catalog reach."""

from typing import ClassVar

import pytest
from textual.widgets import Button, Input, OptionList, Select

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from Tests.Workspaces.test_agent_provisioning import build
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.Widgets.workspace_create_modal import WorkspaceCreateModal
from tldw_chatbook.Widgets.workspace_persona_default import (
    WorkspacePersonaDefaultModal,
)
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults


class _Host(ConsolidatedCSSApp):
    CSS_PATH: ClassVar = [BUNDLED_STYLESHEET]


def _personas(tmp_path, ids):
    service = LocalCharacterPersonaService(
        None, persona_store_path=tmp_path / "personas.json"
    )
    for persona_id in ids:
        service.create_persona_profile(
            {
                "id": persona_id,
                "name": f"Writer [{persona_id}]",
                "system_prompt": "Help.",
            }
        )
    return service


def _selected_label(select):
    return str(next(label for label, value in select._options if value == select.value))


async def _press(host, pilot, selector):
    host.screen.query_one(selector, Button).focus()
    await pilot.pause()
    await pilot.wait_for_scheduled_animations()
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("persona_id", ["none", "auto"])
@private_profile_test
async def test_no_edit_apply_preserves_control_named_saved_persona(
    request, tmp_path, persona_id
):
    personas = _personas(tmp_path, [persona_id])
    registry, _store = build(tmp_path, personas)
    defaults = WorkspaceAssistantDefaults(
        assistant_id=persona_id, tool_policy_profile_id="default"
    )
    registry.create_workspace(
        workspace_id="chosen", name="Chosen", assistant_defaults=defaults
    )
    host = _Host()
    try:
        async with host.run_test(size=(80, 24)) as pilot:
            await host.push_screen(
                WorkspacePersonaDefaultModal(registry, personas, "chosen")
            )
            select = host.screen.query_one("#workspace-default-persona", Select)
            label = _selected_label(select)
            memory_disabled = host.screen.query_one(
                "#workspace-default-memory", Select
            ).disabled
            await _press(host, pilot, "#workspace-default-apply")
            saved = registry.get_workspace("chosen")
            assert (saved.assistant_defaults, label, memory_disabled) == (
                defaults,
                f"Writer [{persona_id}]",
                False,
            )
            assert not saved.assistant_defaults_explicit_none
    finally:
        registry.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("persona_id", ["none", "auto"])
@private_profile_test
async def test_create_control_named_persona_survives_folder_recomposition(
    request, tmp_path, persona_id
):
    personas = _personas(tmp_path, [persona_id])
    registry, _store = build(tmp_path, personas)
    host = _Host()
    try:
        async with host.run_test(size=(80, 24)) as pilot:
            await host.push_screen(
                WorkspaceCreateModal(
                    registry_service=registry, persona_service=personas
                )
            )
            modal = host.screen
            modal.query_one("#workspace-create-name", Input).value = "Chosen"
            modal.query_one("#workspace-default-persona", Select).value = persona_id
            modal.query_one("#workspace-create-folder-path", Input).value = str(
                tmp_path
            )
            await _press(host, pilot, "#workspace-create-folder-add")
            assert (
                modal.query_one("#workspace-default-persona", Select).value
                == persona_id
            )
            await _press(host, pilot, "#workspace-create-confirm")
            saved = next(w for w in registry.list_workspaces() if w.name == "Chosen")
            assert saved.assistant_defaults is not None
            assert saved.assistant_defaults.assistant_id == persona_id
            assert not saved.assistant_defaults_explicit_none
            assert len(personas.list_persona_profiles()) == 1
    finally:
        registry.db.close()


@pytest.mark.asyncio
@private_profile_test
async def test_modal_offers_all_personas_beyond_first_page(request, tmp_path):
    ids = [f"writer-{i:03}" for i in range(101)]
    personas = _personas(tmp_path, ids)
    assert ids[0] not in {p["id"] for p in personas.list_persona_profiles()}
    registry, _store = build(tmp_path, personas)
    registry.create_workspace(
        workspace_id="chosen",
        name="Chosen",
        assistant_defaults=WorkspaceAssistantDefaults(assistant_id=ids[0]),
    )
    host = _Host()
    try:
        async with host.run_test(size=(80, 24)):
            await host.push_screen(
                WorkspacePersonaDefaultModal(registry, personas, "chosen")
            )
            select = host.screen.query_one("#workspace-default-persona", Select)
            offered = {value for _, value in select._options}
            assert set(ids) <= offered
            assert _selected_label(select) == "Writer [writer-000]"
    finally:
        registry.db.close()


@pytest.mark.asyncio
@private_profile_test
async def test_settings_offers_and_selects_persona_beyond_first_page(request, tmp_path):
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_settings_overview_search_journeys import _category
    from Tests.UI.test_settings_provider_keyboard_journeys import _settle
    from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness

    ids = [f"writer-{i:03}" for i in range(101)]
    personas = _personas(tmp_path, ids)
    app = _build_test_app()
    app.local_character_persona_service = personas
    registry = app.workspace_registry_service
    registry.create_workspace(
        workspace_id="chosen",
        name="Chosen",
        assistant_defaults=WorkspaceAssistantDefaults(assistant_id=ids[0]),
    )
    host = _StyledDestinationHarness(app, "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        await _category(host, pilot, "Workspaces")
        await _press(host, pilot, "#settings-workspace-row-chosen")
        await _settle(host, pilot)
        picker = host.screen.query_one("#settings-workspace-persona-picker", OptionList)
        offered = {
            picker.get_option_at_index(i).persona_id for i in range(picker.option_count)
        }
        assert offered == set(ids)
        assert picker.get_option_at_index(picker.highlighted).persona_id == ids[0]
        picker.focus()
        await pilot.pause()
        await pilot.press("end", "enter")
        await _settle(host, pilot)
        assert host.screen._settings_workspace_assistant_pending["persona_id"] == ids[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("available", [True, False])
@private_profile_test
async def test_failed_list_resolves_saved_identity_without_silent_rebinding(
    request, tmp_path, available
):
    personas = _personas(tmp_path, ["saved"])
    if not available:
        personas.delete_persona_profile("saved", expected_version=1)
    registry, _store = build(tmp_path, personas)
    defaults = WorkspaceAssistantDefaults(assistant_id="saved")
    registry.create_workspace(
        workspace_id="chosen", name="Chosen", assistant_defaults=defaults
    )

    def unavailable_list(**_kwargs):
        raise RuntimeError("private list failure")

    personas.list_persona_profiles = unavailable_list
    host = _Host()
    try:
        async with host.run_test(size=(80, 24)) as pilot:
            await host.push_screen(
                WorkspacePersonaDefaultModal(registry, personas, "chosen")
            )
            select = host.screen.query_one("#workspace-default-persona", Select)
            assert _selected_label(select) == (
                "Writer [saved]" if available else "Saved Persona unavailable"
            )
            await _press(host, pilot, "#workspace-default-apply")
            assert registry.get_workspace("chosen").assistant_defaults == defaults
            if not available:
                assert isinstance(host.screen, WorkspacePersonaDefaultModal)
                text = "\n".join(
                    strip.text for strip in host.screen._compositor.render_strips()
                )
                assert "Selected Persona is unavailable" in text
                assert "private list failure" not in text
    finally:
        registry.db.close()


@pytest.mark.asyncio
@private_profile_test
async def test_settings_keeps_readable_saved_choice_when_catalog_fails(
    request, tmp_path
):
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_settings_overview_search_journeys import _category
    from Tests.UI.test_settings_provider_keyboard_journeys import _settle
    from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness

    personas = _personas(tmp_path, ["saved"])
    app = _build_test_app()
    app.local_character_persona_service = personas
    registry = app.workspace_registry_service
    defaults = WorkspaceAssistantDefaults(assistant_id="saved")
    registry.create_workspace(
        workspace_id="chosen", name="Chosen", assistant_defaults=defaults
    )

    def unavailable_list(**_kwargs):
        raise RuntimeError("private list failure")

    personas.list_persona_profiles = unavailable_list
    host = _StyledDestinationHarness(app, "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        await _category(host, pilot, "Workspaces")
        await _press(host, pilot, "#settings-workspace-row-chosen")
        await _settle(host, pilot)
        picker = host.screen.query_one("#settings-workspace-persona-picker", OptionList)
        assert picker.option_count == 1
        assert picker.get_option_at_index(0).persona_id == "saved"
        assert str(picker.get_option_at_index(0).prompt) == "Writer [saved]"
        assert picker.highlighted == 0
        assert registry.get_workspace("chosen").assistant_defaults == defaults
