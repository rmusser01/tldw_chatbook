"""Artifacts participate in the shared Library layout and global focus cycle."""

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.test_library_artifacts_canvas import artifact_library

pytestmark = pytest.mark.ui


@pytest.mark.asyncio
@private_profile_test
async def test_artifact_shell_is_recognized_as_adaptive(tmp_path, request):
    async with artifact_library(tmp_path) as (screen, _pilot):
        assert screen._library_adaptive_reader_shell_active()


@pytest.mark.asyncio
@pytest.mark.parametrize("collapsed", [False, True])
@private_profile_test
async def test_artifact_f6_cycles_panes_and_closed_grips(tmp_path, request, collapsed):
    async with artifact_library(tmp_path) as (screen, pilot):
        controller = screen._artifacts_controller
        controller.focus_reader()
        await pilot.pause()
        if collapsed:
            controller.toggle_pane("library")
            controller.toggle_pane("items")
            await pilot.pause()
            assert not controller.layout.library_open
            assert not controller.layout.items_open
        body = screen.query_one("#library-artifacts-body")
        rows = screen.query_one("#library-artifacts-list")
        library = screen.query_one(
            "#library-artifacts-library-grip" if collapsed else "#library-search-input"
        )
        items = screen.query_one(
            "#library-artifacts-items-grip" if collapsed else "#library-artifacts-list"
        )
        selected = controller.selected
        body.focus()
        await pilot.pause()
        # The production app's global bindings delegate to these screen actions.
        # This CSS harness deliberately has no app-level F6 binding.
        for expected in (library, items, body):
            screen.action_focus_next_workbench_pane()
            await pilot.pause()
            assert screen.focused is expected
        for expected in (items, library, body):
            screen.action_focus_previous_workbench_pane()
            await pilot.pause()
            assert screen.focused is expected
        assert controller.selected == selected
        if collapsed:
            items.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert controller.layout.items_open
            assert rows.display
            screen.action_focus_next_workbench_pane()
            await pilot.pause()
            assert screen.focused is body


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(64, 30), (50, 25)])
@private_profile_test
async def test_narrow_f6_reveals_reader_and_returns_through_items_grip(
    tmp_path, request, size
):
    async with artifact_library(tmp_path, size=size) as (screen, pilot):
        controller = screen._artifacts_controller
        controller.focus_items()
        await pilot.pause()
        rows = screen.query_one("#library-artifacts-list")
        body = screen.query_one("#library-artifacts-body")
        assert screen.focused is rows
        assert not controller.reader_open
        screen.action_focus_next_workbench_pane()
        await pilot.pause()
        assert screen.focused is body
        assert controller.reader_open
        assert body.region.width > 20
        assert body.region.right <= screen.size.width
        screen.action_focus_previous_workbench_pane()
        await pilot.pause()
        assert screen.focused.id == "library-artifacts-items-grip"
        await pilot.press("enter")
        await pilot.pause()
        assert screen.focused is rows
        assert rows.region.right <= screen.size.width
        screen.action_focus_next_workbench_pane()
        await pilot.pause()
        assert screen.focused is body
        assert body.region.width > 20
        assert body.region.right <= screen.size.width


@pytest.mark.asyncio
@private_profile_test
async def test_narrow_empty_items_skips_zero_width_reader(tmp_path, request):
    async with artifact_library(tmp_path, size=(50, 25)) as (screen, pilot):
        controller = screen._artifacts_controller
        controller.search("no such report")
        for _ in range(100):
            await pilot.pause(0.02)
            if controller.page and controller.page.total == 0:
                break
        assert controller.selected is None
        controller.focus_items()
        await pilot.pause()
        screen.action_focus_next_workbench_pane()
        await pilot.pause()
        assert screen.focused.id == "library-artifacts-library-grip"
        screen.action_focus_next_workbench_pane()
        await pilot.pause()
        assert screen.focused.id == "library-artifacts-list"


@pytest.mark.asyncio
@private_profile_test
async def test_reader_reveal_does_not_refocus_after_a_newer_f6(
    tmp_path, request, monkeypatch
):
    async with artifact_library(tmp_path, size=(50, 25)) as (screen, pilot):
        controller = screen._artifacts_controller
        controller.focus_items()
        await pilot.pause()
        original = screen.call_after_refresh
        held = []

        def defer_reader_focus(callback, *args, **kwargs):
            if callback == controller._focus:
                held.append((callback, args, kwargs))
                return True
            return original(callback, *args, **kwargs)

        monkeypatch.setattr(screen, "call_after_refresh", defer_reader_focus)
        screen.action_focus_next_workbench_pane()
        await pilot.pause()
        body = screen.query_one("#library-artifacts-body")
        assert screen.focused is body
        assert body.region.right <= screen.size.width
        screen.action_focus_next_workbench_pane()
        await pilot.pause()
        grip = screen.query_one("#library-artifacts-library-grip")
        assert screen.focused is grip
        for callback, args, kwargs in held:
            callback(*args, **kwargs)
        await pilot.pause()
        assert screen.focused is grip
