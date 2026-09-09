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
        from textual.widgets import Collapsible

        modal.query_one(Collapsible).collapsed = False
        await pilot.pause()
        modal.query_one("#buddy-height", Input).focus()
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        assert modal.query_one("#buddy-height").region.bottom <= apply.region.y
        assert modal.query_one("#buddy-height").has_focus


@pytest.mark.asyncio
async def test_apply_failure_preserves_form_and_blocks_dismissal_while_saving():
    import asyncio

    m = modal_module()
    started, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def apply(choice):
        calls.append(choice)
        started.set()
        await finish.wait()
        raise ValueError("Cannot import this pack. Check its path and retry.")

    app = App()
    async with app.run_test() as pilot:
        modal = m.BuddyManagementModal(apply=apply)
        app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#buddy-import", Input).value = "/missing.zip"
        modal.query_one("#buddy-apply", Button).press()
        await started.wait()
        await pilot.press("escape")
        assert app.screen is modal
        assert modal.query_one("#buddy-apply", Button).disabled
        finish.set()
        await pilot.pause()
        assert app.screen is modal
        assert modal.query_one("#buddy-import", Input).value == "/missing.zip"
        assert "Check its path" in str(modal.query_one("#buddy-form-error").render())
        assert not modal.query_one("#buddy-apply", Button).disabled
        assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 20), (80, 24), (120, 40)])
async def test_preview_pointer_focus_and_disclosed_geometry_fit(size):
    from textual.widgets import Collapsible

    m = modal_module()
    previews = []
    app = App()
    async with app.run_test(size=size) as pilot:
        modal = m.BuddyManagementModal(
            buddies=(("Migu", "migu"),),
            initial=m.BuddyManagementChoice(buddy_id="migu"),
            preview=lambda *args: previews.append(args) or "Migu preview",
        )
        app.push_screen(modal)
        await pilot.pause()
        preview = modal.query_one("#buddy-preview-button", Button)
        preview.focus()
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        assert preview.has_focus
        assert (
            preview.region.right
            <= modal.query_one("#buddy-management-body").region.right
        )
        assert await pilot.click("#buddy-preview-button")
        await pilot.pause()
        assert previews == [("migu", "idle")]
        advanced = modal.query_one(Collapsible)
        assert advanced.collapsed
        advanced.query_one("CollapsibleTitle").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert not advanced.collapsed
        height = modal.query_one("#buddy-height", Input)
        height.focus()
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        assert height.has_focus
        assert (
            height.region.right
            <= modal.query_one("#buddy-management-body").region.right
        )
        assert height.region.bottom <= modal.query_one("#buddy-apply").region.y


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["conversation", "workspace"])
async def test_current_persona_name_is_literal_text(kind):
    from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding

    m = modal_module()
    binding = BuddyBinding(kind=kind, target_id="target")
    app = App()
    async with app.run_test() as pilot:
        modal = m.BuddyManagementModal(
            targets=(
                m.BuddyTargetChoice(
                    "target", "Named target", binding, current_persona="Archivist [/]"
                ),
            ),
            initial=m.BuddyManagementChoice(binding=binding),
        )
        app.push_screen(modal)
        await pilot.pause()
        assert "Archivist [/]" in str(modal.query_one("#buddy-persona-help").render())


@pytest.mark.asyncio
async def test_buddy_form_rejects_overlong_archive_at_shared_input_boundary():
    m = modal_module()
    app = App()
    async with app.run_test() as pilot:
        modal = m.BuddyManagementModal()
        app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#buddy-import", Input).value = "x" * 4097
        with pytest.raises(ValueError):
            modal._choice()


@pytest.mark.asyncio
async def test_artwork_paging_retains_selected_label_and_staged_choice():
    m = modal_module()
    all_rows = tuple((f"[bold]Buddy {index}[/bold]", str(index)) for index in range(5))

    async def page(offset):
        return all_rows[offset : offset + 2]

    app = App()
    async with app.run_test() as pilot:
        modal = m.BuddyManagementModal(
            buddies=all_rows[:2],
            artwork_page=page,
            artwork_page_size=2,
            selected_buddy=all_rows[-1],
            initial=m.BuddyManagementChoice(enabled=True, buddy_id="4"),
        )
        app.push_screen(modal)
        await pilot.pause()
        assert modal._choice().buddy_id == "4"
        modal.query_one("#buddy-artwork-next", Button).press()
        await pilot.pause()
        assert {key for _, key in modal._buddies} == {"2", "3"}
        assert modal._choice().buddy_id == "4"
        modal.query_one("#buddy-artwork", Select).value = "3"
        await pilot.pause()
        modal.query_one("#buddy-artwork-next", Button).press()
        await pilot.pause()
        assert {key for _, key in modal._buddies} == {"4"}
        assert modal._choice().buddy_id == "3"
        assert modal.query_one("#buddy-artwork-next", Button).disabled
        modal.query_one("#buddy-artwork-previous", Button).press()
        await pilot.pause()
        assert modal._choice().buddy_id == "3"
        labels = [
            str(label)
            for label, _ in modal.query_one("#buddy-artwork", Select)._options
        ]
        assert "[bold]Buddy 3[/bold]" in labels
