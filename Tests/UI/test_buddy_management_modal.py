"""The Buddy management form stages intent and stays usable on compact terminals."""

import importlib.util

import pytest
from textual.app import App
from textual.widgets import Button, Input, Select, Switch


def modal_module():
    assert (
        importlib.util.find_spec(
            "tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal"
        )
        is not None
    ), "Buddy management surface is missing"
    from tldw_chatbook.Widgets.Persona_Widgets import buddy_management_modal

    return buddy_management_modal


@pytest.mark.asyncio
async def test_cancel_returns_no_mutation_after_staged_changes():
    m = modal_module()
    result = []
    app = App()
    async with app.run_test() as pilot:
        modal = m.BuddyManagementModal(buddies=(("Pixel Migu", "migu"),))
        app.push_screen(modal, result.append)
        await pilot.pause()
        modal.query_one("#buddy-enabled", Switch).value = True
        modal.query_one("#buddy-artwork", Select).value = "migu"
        await pilot.press("escape")
        assert result == [None]


@pytest.mark.asyncio
async def test_apply_captures_explicit_binding_and_static_preference():
    m = modal_module()
    from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding

    binding = BuddyBinding(kind="workspace", target_id="research")
    target = m.BuddyTargetChoice("workspace-research", "Workspace: Research", binding)
    result = []
    app = App()
    async with app.run_test() as pilot:
        modal = m.BuddyManagementModal(
            buddies=(("Pixel Migu", "migu"),), targets=(target,)
        )
        app.push_screen(modal, result.append)
        await pilot.pause()
        modal.query_one("#buddy-enabled", Switch).value = True
        modal.query_one("#buddy-artwork", Select).value = "migu"
        modal.query_one("#buddy-follow", Select).value = "workspace-research"
        modal.query_one("#buddy-motion", Select).value = "static"
        modal.query_one("#buddy-width", Input).value = "32"
        modal.query_one("#buddy-apply", Button).press()
        await pilot.pause()
        assert result[0].binding == binding
        assert result[0].animated is False
        assert result[0].width == 32
        assert result[0].persona_choice == m.PERSONA_UNCHANGED


@pytest.mark.asyncio
async def test_invalid_artwork_keeps_form_open_with_recovery_copy():
    m = modal_module()
    result = []
    app = App()
    async with app.run_test() as pilot:
        modal = m.BuddyManagementModal()
        app.push_screen(modal, result.append)
        await pilot.pause()
        modal.query_one("#buddy-enabled", Switch).value = True
        modal.query_one("#buddy-apply", Button).press()
        await pilot.pause()
        assert app.screen is modal
        assert "Choose artwork" in str(modal.query_one("#buddy-form-error").render())
        assert result == []


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 40), (60, 20)])
async def test_actions_remain_visible_and_fields_keyboard_accessible(size):
    m = modal_module()
    app = App()
    async with app.run_test(size=size) as pilot:
        modal = m.BuddyManagementModal(buddies=(("Pixel Migu", "migu"),))
        app.push_screen(modal)
        await pilot.pause()
        apply = modal.query_one("#buddy-apply", Button)
        assert apply.region.bottom <= size[1]
        assert apply.region.right <= size[0]
        modal.query_one("#buddy-height", Input).focus()
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        assert modal.query_one("#buddy-height").region.bottom <= apply.region.y
        assert modal.query_one("#buddy-height").has_focus
