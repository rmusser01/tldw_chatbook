"""Shared structural contracts for Library adaptive reader shells."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual import on
from textual.containers import Vertical
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Utils.adaptive_reader_state import (
    PANE_GRIP_WIDTH,
    AdaptiveReaderEffectiveLayout,
    AdaptiveReaderLayoutPreferences,
    AdaptiveReaderLayoutProfile,
    resolve_adaptive_reader_layout,
)
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    AdaptiveReaderShellResized,
    LibraryAdaptiveReaderShell,
    PaneToggleRequested,
)
from tldw_chatbook.app import TldwCli


CSS_SOURCE = (
    Path(__file__).parents[2]
    / "tldw_chatbook"
    / "css"
    / "components"
    / "_agentic_terminal.tcss"
)


def _layout(
    *, library_open: bool = True, items_open: bool = True
) -> AdaptiveReaderEffectiveLayout:
    return AdaptiveReaderEffectiveLayout(
        library_open=library_open,
        items_open=items_open,
        library_width=28 if library_open else 0,
        items_width=40 if items_open else 0,
        reader_width=82,
        priority_pane=None,
    )


class _ProbeApp(ConsolidatedCSSApp):
    CSS_PATH = TldwCli.CSS_PATH

    def __init__(
        self,
        layout: AdaptiveReaderEffectiveLayout | None = None,
        *,
        focusable_content: bool = False,
        work_disabled: bool = False,
    ) -> None:
        super().__init__()
        self.layout = layout or _layout()
        self.focusable_content = focusable_content
        self.work_disabled = work_disabled
        self.toggles: list[str] = []
        self.resize_messages = 0

    def compose(self):
        if self.focusable_content:
            library = Vertical(Button("Library action", id="probe-library-action"))
            items = Vertical(Button("Items action", id="probe-items-action"))
        else:
            library = Static("Library")
            items = Static("Items")
        shell = LibraryAdaptiveReaderShell(
            library=library,
            items=items,
            work=Static("Work", disabled=self.work_disabled),
            layout=self.layout,
            id_prefix="probe",
            library_label="Library",
            items_label="Items",
            id="probe-shell",
        )
        yield shell

    @on(PaneToggleRequested)
    def _capture_toggle(self, event: PaneToggleRequested) -> None:
        event.stop()
        self.toggles.append(event.pane)

    @on(AdaptiveReaderShellResized)
    def _capture_resize(self, event: AdaptiveReaderShellResized) -> None:
        event.stop()
        self.resize_messages += 1


@pytest.mark.asyncio
async def test_shell_mounts_three_concrete_widgets_and_two_five_column_grips():
    app = _ProbeApp()

    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)

        assert list(shell.children) == [
            shell.library,
            shell.library_grip,
            shell.items,
            shell.items_grip,
            shell.work,
        ]
        assert [str(shell.library.renderable), str(shell.items.renderable)] == [
            "Library",
            "Items",
        ]
        assert str(shell.work.renderable) == "Work"
        assert [shell.library_grip.region.width, shell.items_grip.region.width] == [
            PANE_GRIP_WIDTH,
            PANE_GRIP_WIDTH,
        ]


@pytest.mark.asyncio
async def test_mount_and_terminal_resize_emit_shell_resize_messages():
    app = _ProbeApp()

    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        mounted_count = app.resize_messages

        assert mounted_count >= 1

        await pilot.resize_terminal(140, 30)
        await pilot.pause()

        assert app.resize_messages > mounted_count


@pytest.mark.asyncio
async def test_sync_layout_retains_every_mounted_child_identity():
    app = _ProbeApp()

    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)
        children = tuple(shell.children)

        shell.sync_layout(_layout(library_open=False, items_open=False))
        await pilot.pause()

        assert tuple(shell.children) == children
        assert shell.library is children[0]
        assert shell.items is children[2]
        assert shell.work is children[4]
        assert shell.work.is_mounted and shell.work.display


@pytest.mark.asyncio
async def test_sync_layout_preserves_destination_owned_work_disabled_state():
    app = _ProbeApp(work_disabled=True)

    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)

        shell.sync_layout(_layout(library_open=False, items_open=False))
        await pilot.pause()

        assert shell.work.is_mounted and shell.work.display
        assert shell.work.disabled


@pytest.mark.asyncio
async def test_grips_emit_correct_toggle_for_enter_space_and_pointer_click():
    app = _ProbeApp()

    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)
        app.toggles.clear()

        shell.library_grip.focus()
        await pilot.press("enter")
        shell.items_grip.focus()
        await pilot.press("space")
        await pilot.click("#probe-library-grip")

        assert app.toggles == ["library", "items", "library"]


@pytest.mark.asyncio
async def test_grips_do_not_drag_or_resize_any_region():
    app = _ProbeApp()

    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)
        before = tuple(child.region for child in shell.children)
        layout = shell.effective_layout

        await pilot.mouse_down("#probe-library-grip")
        await pilot.mouse_up("#probe-shell", offset=(159, 15))
        await pilot.pause()

        assert shell.effective_layout is layout
        assert tuple(child.region for child in shell.children) == before
        assert app.toggles == []


@pytest.mark.asyncio
async def test_hiding_focused_pane_moves_focus_to_truthful_restore_grip():
    app = _ProbeApp(focusable_content=True)

    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)
        library_action = shell.query_one("#probe-library-action", Button)
        library_action.focus()
        assert library_action.has_focus

        shell.sync_layout(_layout(library_open=False))
        await pilot.pause()

        assert not shell.library.display
        assert shell.library.disabled
        assert not library_action.has_focus
        assert shell.library_grip.has_focus
        assert shell.library_grip.can_focus
        assert shell.library_grip.display
        assert shell.library_grip.name == "Expand Library pane"
        assert str(shell.library_grip.tooltip) == "Expand Library pane"


@pytest.mark.asyncio
async def test_both_hidden_panes_keep_reachable_restore_controls_and_work_mounted():
    app = _ProbeApp(_layout(library_open=False, items_open=False))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)

        assert not shell.library.display and not shell.items.display
        assert shell.work.is_mounted and shell.work.display
        assert shell.work.region.width > 0
        for pane, grip in (
            ("library", shell.library_grip),
            ("items", shell.items_grip),
        ):
            assert grip.display and grip.can_focus
            assert grip.name == f"Expand {pane.title()} pane"
            assert str(grip.tooltip) == f"Expand {pane.title()} pane"
            grip.focus()
            await pilot.press("enter")

        assert app.toggles == ["library", "items"]
        assert not shell.query("#probe-work-grip")


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [160, 120, 100, 80, 60])
async def test_all_five_regions_remain_inside_representative_media_widths(width):
    layout = resolve_adaptive_reader_layout(
        width,
        AdaptiveReaderLayoutPreferences(),
        AdaptiveReaderLayoutProfile(),
    )
    app = _ProbeApp(layout)

    async with app.run_test(size=(width, 24)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)

        assert shell.region.width == width
        for child in shell.children:
            assert app.screen.region.contains_region(child.region), (
                width,
                child,
                child.region,
                app.screen.region,
            )
        assert shell.work.display and shell.work.region.right <= shell.region.right


def test_shared_shell_structure_is_owned_by_shared_tcss_selectors():
    source = CSS_SOURCE.read_text(encoding="utf-8")

    assert ".library-adaptive-reader-shell {" in source
    assert (
        ".library-adaptive-reader-shell > .library-adaptive-reader-work {" in source
    )
    assert (
        ".library-adaptive-reader-shell > .library-adaptive-reader-pane-grip {"
        in source
    )


def test_shared_tcss_is_structural_while_media_keeps_its_visual_contract():
    source = CSS_SOURCE.read_text(encoding="utf-8")
    shared_grip = source.split(
        ".library-adaptive-reader-shell > .library-adaptive-reader-pane-grip {",
        1,
    )[1].split("}", 1)[0]

    assert "background:" not in shared_grip
    assert "color:" not in shared_grip
    assert "text-style:" not in shared_grip
    assert "#library-media-reader-shell > .library-media-pane-grip {" in source
    assert (
        "#library-media-reader-shell > .library-media-pane-grip:focus {" in source
    )
