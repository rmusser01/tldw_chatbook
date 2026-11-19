"""Artifacts participate in the shared Library layout and global focus cycle."""

import time

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


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 25), (64, 30), (50, 25)])
@private_profile_test
async def test_items_opened_beside_an_open_artifact_take_the_stage(
    tmp_path, request, size
):
    """TASK-38981: with an artifact open, opening Items beside it left the
    reader a ``width - 42`` sliver (38 cells at 80, 8 at 50). The list now
    takes the stage, and F6 still reveals the reader at full width."""
    async with artifact_library(tmp_path, size=size) as (screen, pilot):
        controller = screen._artifacts_controller
        controller.focus_items()
        await pilot.pause()
        screen.action_focus_next_workbench_pane()
        await pilot.pause()
        body = screen.query_one("#library-artifacts-body")
        assert controller.reader_open and screen.focused is body
        grip = screen.query_one("#library-artifacts-items-grip")
        screen.set_focus(grip)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        layout = controller.layout
        assert layout.items_open
        assert layout.reader_width == 0, layout
        screen.action_focus_next_workbench_pane()
        await pilot.pause()
        assert screen.focused is body
        assert body.region.width >= 30
        assert body.region.right <= screen.size.width

        # TASK-38981 re-review: with the list holding the stage again, Tab from
        # the list must never rest on a reader control the user cannot see.
        screen.set_focus(grip)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert controller.layout.reader_width == 0, controller.layout
        rows = screen.query_one("#library-artifacts-list")
        screen.set_focus(rows)
        await pilot.pause()
        for _ in range(8):
            await pilot.press("tab")
            await pilot.pause()
            await pilot.pause()
            focused = screen.focused
            assert focused is not None
            clip = screen.find_widget(focused).clip
            region = focused.region
            cells = min(region.right, clip.right, screen.size.width) - max(
                region.x, clip.x, 0
            )
            assert cells > 0, (size, focused.id, cells, controller.layout)


async def _until(pilot, predicate, what: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await pilot.pause(0.02)


def _visible_cells(screen, widget) -> int:
    """The cells of ``widget`` inside its compositor clip (lessons-textual)."""
    clip = screen.find_widget(widget).clip
    region = widget.region
    return min(region.right, clip.right, screen.size.width) - max(region.x, clip.x, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 24), (50, 25)])
@private_profile_test
async def test_empty_narrow_stage_tab_never_rests_on_an_unseen_reader_control(
    tmp_path, request, size
):
    """TASK-39062: with no artifact to open, the narrow stage gives the list the
    whole width (``reader_width`` 0). The reader stayed displayed, because
    focusing one of its controls reveals it, but with nothing selected there
    was nothing to reveal: Tab from the list walked Back, Preview, Details and
    the body at -1 to -21 visible cells."""
    async with artifact_library(tmp_path, size=size, reports=0) as (screen, pilot):
        controller = screen._artifacts_controller
        # Under load the empty page can land after the helper's 3 s wait, and
        # the list's focus lands a refresh after ``focus_items``: wait for
        # both, bounded, before reading them.
        await _until(pilot, lambda: controller.page is not None, "the empty page")
        assert not controller.page.items
        assert controller.selected is None
        controller.focus_items()
        rows = screen.query_one("#library-artifacts-list")
        await _until(pilot, lambda: screen.focused is rows, "focus on the list")
        assert controller.layout.items_open
        assert controller.layout.reader_width == 0, controller.layout
        stops = []
        for _ in range(40):  # one whole lap of the screen's focus chain
            await pilot.press("tab")
            await pilot.pause()
            focused = screen.focused
            assert focused is not None
            if focused is rows:
                break
            stops.append((focused.id, _visible_cells(screen, focused)))
        assert focused is rows, stops
        unseen = [(widget_id, cells) for widget_id, cells in stops if cells <= 0]
        assert unseen == [], (size, stops)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 24), (50, 25)])
@private_profile_test
async def test_narrow_tab_reaches_the_reader_once_the_first_artifact_lands(
    tmp_path, request, size
):
    """TASK-39062 review: entering Artifacts on a narrow stage resizes while no
    artifact is selected yet, so the zero-width reader hides. The page then
    lands and selects its first artifact without a resize. The shell's
    ``sync()`` re-applies the layout, so Tab from the list still reaches a
    reader control and reveals the reader. Nothing here calls
    ``focus_items()`` or resizes, so that re-apply is the only thing that can
    show the reader again."""
    async with artifact_library(tmp_path, size=size) as (screen, pilot):
        controller = screen._artifacts_controller
        await _until(pilot, lambda: controller.detail is not None, "the detail")
        assert controller.selected is not None
        assert controller.layout.reader_width == 0, controller.layout
        shell = screen.query_one("#library-artifacts-reader-shell")
        rows = screen.query_one("#library-artifacts-list")
        screen.set_focus(rows)
        await _until(pilot, lambda: screen.focused is rows, "focus on the list")
        stops = []
        for _ in range(40):  # at most one whole lap of the focus chain
            await pilot.press("tab")
            focused = screen.focused
            assert focused is not None
            stops.append(focused.id)
            if shell.work in focused.ancestors_with_self or focused is rows:
                break
        assert shell.work in focused.ancestors_with_self, (size, stops)
        await _until(
            pilot,
            lambda: controller.reader_open and _visible_cells(screen, focused) > 0,
            "the reader to be revealed around the focused control",
        )
