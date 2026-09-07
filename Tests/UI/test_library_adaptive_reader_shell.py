"""Shared structural contracts for Library adaptive reader shells."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual import on
from textual.containers import Vertical
from textual.css.styles import StylesBase
from textual.widget import Widget
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Library.library_media_reader_state import (
    MEDIA_READER_LAYOUT_PROFILE,
)
from tldw_chatbook.Utils.adaptive_reader_state import (
    PANE_GRIP_WIDTH,
    READER_COMFORT_WIDTH,
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
        hidden_items_subtree: bool = False,
        work_disabled: bool = False,
        grip_width: int = PANE_GRIP_WIDTH,
    ) -> None:
        super().__init__()
        self.layout = layout or _layout()
        self.focusable_content = focusable_content
        self.hidden_items_subtree = hidden_items_subtree
        self.work_disabled = work_disabled
        self.grip_width = grip_width
        self.toggles: list[str] = []
        self.resize_messages = 0

    def compose(self):
        if self.focusable_content:
            library = Vertical(Button("Library action", id="probe-library-action"))
            if self.hidden_items_subtree:
                hidden_items = Vertical(
                    Button("Hidden items action", id="probe-hidden-items-action")
                )
                hidden_items.display = False
                items = Vertical(
                    hidden_items,
                    Button("Items action", id="probe-items-action"),
                )
            else:
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
            grip_width=self.grip_width,
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


def _painted_rows_containing(app: _ProbeApp, widget: Widget, token: str) -> list[int]:
    """Return widget-relative compositor rows containing ``token``."""
    strips = list(app.screen._compositor.render_strips())
    return [
        y - widget.region.y
        for y in range(widget.region.y, widget.region.bottom)
        if token in strips[y].crop(widget.region.x, widget.region.right).text
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("grip_width", "arrow"),
    # task-31633 AC#2: the grip width is the destination profile's, and the
    # arrow is as wide as the grip. Five columns is the shared default every
    # destination but Media still uses; one column is Media's.
    [(PANE_GRIP_WIDTH, "<---"), (MEDIA_READER_LAYOUT_PROFILE.grip_width, "‹")],
)
async def test_shell_mounts_three_concrete_widgets_and_two_profile_width_grips(
    grip_width: int,
    arrow: str,
) -> None:
    app = _ProbeApp(grip_width=grip_width)

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
            grip_width,
            grip_width,
        ]
        assert _painted_rows_containing(app, shell.items_grip, arrow), arrow


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_size", "expected_library_rows"),
    [
        ((160, 50), [17, 32]),
        ((120, 35), [12, 22]),
        ((100, 30), [10, 19]),
        ((80, 24), [8, 15]),
    ],
)
async def test_grips_paint_library_arrows_at_35_and_65_percent_and_items_at_center(
    terminal_size: tuple[int, int],
    expected_library_rows: list[int],
) -> None:
    app = _ProbeApp()

    async with app.run_test(size=terminal_size) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)
        library_rows = _painted_rows_containing(app, shell.library_grip, "<---")
        items_rows = _painted_rows_containing(app, shell.items_grip, "<---")
        last_row = shell.library_grip.region.height - 1

        assert library_rows == expected_library_rows
        assert items_rows in ([last_row // 2], [(last_row + 1) // 2])


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
async def test_reopening_items_moves_focus_from_its_grip_into_the_items_pane():
    app = _ProbeApp(_layout(items_open=False), focusable_content=True)

    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)
        items_action = shell.query_one("#probe-items-action", Button)
        shell.items_grip.focus()
        await pilot.pause()
        assert shell.items_grip.has_focus

        shell.sync_layout(_layout(items_open=True))
        await pilot.pause()

        assert shell.items.display
        assert items_action.has_focus


@pytest.mark.asyncio
async def test_manual_reopen_restores_the_matching_panes_last_descendant():
    app = _ProbeApp(focusable_content=True)

    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)
        items_action = shell.query_one("#probe-items-action", Button)
        items_action.focus()
        await pilot.pause()
        shell.sync_layout(_layout(items_open=False))
        await pilot.pause()

        app.screen.set_focus(shell.library_grip, scroll_visible=False)
        shell.sync_layout(_layout(items_open=True), manual_reopen="items")
        await pilot.pause()

        assert items_action.has_focus


@pytest.mark.asyncio
async def test_collapse_focus_recovery_cannot_override_newer_explicit_focus():
    app = _ProbeApp(focusable_content=True)

    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)
        items_action = shell.query_one("#probe-items-action", Button)
        library_action = shell.query_one("#probe-library-action", Button)
        app.screen.set_focus(items_action, scroll_visible=False)

        shell.sync_layout(_layout(items_open=False))
        app.screen.set_focus(library_action, scroll_visible=False)
        await pilot.pause()

        assert library_action.has_focus
        assert not shell.items_grip.has_focus


@pytest.mark.asyncio
async def test_reopen_focus_recovery_cannot_override_newer_explicit_focus():
    app = _ProbeApp(_layout(items_open=False), focusable_content=True)

    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)
        library_action = shell.query_one("#probe-library-action", Button)
        app.screen.set_focus(shell.items_grip, scroll_visible=False)

        shell.sync_layout(_layout(items_open=True))
        app.screen.set_focus(library_action, scroll_visible=False)
        await pilot.pause()

        assert library_action.has_focus
        assert not shell.query_one("#probe-items-action", Button).has_focus


@pytest.mark.asyncio
async def test_reopen_focus_skips_focusable_descendants_of_hidden_subtrees():
    app = _ProbeApp(
        _layout(items_open=False),
        focusable_content=True,
        hidden_items_subtree=True,
    )

    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)
        visible_items_action = shell.query_one("#probe-items-action", Button)
        hidden_items_action = shell.query_one("#probe-hidden-items-action", Button)
        app.screen.set_focus(shell.items_grip, scroll_visible=False)

        shell.sync_layout(_layout(items_open=True))
        assert visible_items_action in app.screen.focus_chain
        assert app.screen.focused is visible_items_action
        await pilot.pause()

        assert visible_items_action.has_focus
        assert not hidden_items_action.has_focus


@pytest.mark.asyncio
async def test_unchanged_layout_skips_every_pane_geometry_assignment(monkeypatch):
    app = _ProbeApp()

    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)
        pane_styles = (shell.library.styles, shell.items.styles)
        assignments: list[str] = []
        original_setattr = StylesBase.__setattr__
        original_widget_setattr = Widget.__setattr__

        def track_pane_style_assignment(styles, name, value):
            if any(styles is pane_style for pane_style in pane_styles) and name in {
                "display",
                "width",
                "min_width",
                "max_width",
            }:
                assignments.append(name)
            original_setattr(styles, name, value)

        def track_pane_disabled_assignment(widget, name, value):
            if widget in (shell.library, shell.items) and name == "disabled":
                assignments.append(name)
            original_widget_setattr(widget, name, value)

        monkeypatch.setattr(StylesBase, "__setattr__", track_pane_style_assignment)
        monkeypatch.setattr(Widget, "__setattr__", track_pane_disabled_assignment)
        shell.sync_layout(shell.effective_layout)

        assert assignments == []


@pytest.mark.asyncio
async def test_unchanged_cached_layout_repairs_only_stale_physical_declarations(
    monkeypatch,
):
    app = _ProbeApp()

    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)
        layout = shell.effective_layout
        for pane in (shell.library, shell.items):
            pane.display = False
            pane.disabled = True
            pane.styles.width = 7
            pane.styles.min_width = 6
            pane.styles.max_width = 8

        shell.sync_layout(layout)

        for pane, width in (
            (shell.library, layout.library_width),
            (shell.items, layout.items_width),
        ):
            assert pane.display
            assert not pane.disabled
            assert pane.styles.width.value == width
            assert pane.styles.min_width.value == width
            assert pane.styles.max_width.value == width

        pane_styles = (shell.library.styles, shell.items.styles)
        assignments: list[str] = []
        original_setattr = StylesBase.__setattr__
        original_widget_setattr = Widget.__setattr__

        def track_pane_style_assignment(styles, name, value):
            if any(styles is pane_style for pane_style in pane_styles) and name in {
                "display",
                "width",
                "min_width",
                "max_width",
            }:
                assignments.append(name)
            original_setattr(styles, name, value)

        def track_pane_disabled_assignment(widget, name, value):
            if widget in (shell.library, shell.items) and name == "disabled":
                assignments.append(name)
            original_widget_setattr(widget, name, value)

        monkeypatch.setattr(StylesBase, "__setattr__", track_pane_style_assignment)
        monkeypatch.setattr(Widget, "__setattr__", track_pane_disabled_assignment)
        shell.sync_layout(layout)

        assert assignments == []


@pytest.mark.asyncio
async def test_collapsing_focused_items_moves_focus_to_its_grip():
    app = _ProbeApp(focusable_content=True)

    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)
        items_action = shell.query_one("#probe-items-action", Button)
        items_action.focus()
        await pilot.pause()
        assert items_action.has_focus

        shell.sync_layout(_layout(items_open=False))
        await pilot.pause()

        assert not shell.items.display
        assert shell.items_grip.has_focus


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
    assert ".library-adaptive-reader-shell > .library-adaptive-reader-work {" in source
    assert (
        ".library-adaptive-reader-shell > .library-adaptive-reader-pane-grip {"
        in source
    )


def test_shared_tcss_owns_the_calm_visual_contract_for_every_reader():
    source = CSS_SOURCE.read_text(encoding="utf-8")
    shared_grip = source.split(
        ".library-adaptive-reader-shell > .library-adaptive-reader-pane-grip {",
        1,
    )[1].split("}", 1)[0]

    assert "background: $ds-surface-raised;" in shared_grip
    assert "color: $ds-text-muted;" in shared_grip
    assert "text-style: none;" in shared_grip
    assert "outline: none;" in shared_grip
    assert (
        ".library-adaptive-reader-shell > .library-adaptive-reader-pane-grip:focus {"
        in source
    )
    # task-31276: the focused grip carries NO outline on any edge. The grip is
    # as tall as the shell, so "endcaps" land on the reader's first and last
    # content rows -- the top one abutted the Reader identity line and read as
    # a rendering artifact at the pane join (`┐─────Local Media item`, critique
    # #4 P2), not as focus. Focus is the accent recolour on the arrow glyph.
    assert "outline-top: solid $ds-action-focus;" not in source
    assert "outline-bottom: solid $ds-action-focus;" not in source
    assert "#library-media-reader-shell > .library-media-pane-grip" not in source


# ---------------------------------------------------------------------------
# task-31567 AC#2: Space on a focused Items row never collapses a pane.
#
# The grips are the first focusable widgets in the shell, so before the
# restore seam ANY media recompose handed them focus -- and in browse mode
# Space then activated the grip (wave 4 PR B's symptom, from the other end).
# The pin is the whole gesture, not the binding gate: stand on a row, leave
# select mode (a real canvas recompose), press Space, and the shell's pane
# allocation must be byte-identical.
# ---------------------------------------------------------------------------

from Tests.UI.test_library_media_side_by_side import (  # noqa: E402
    _build_media_test_app,
    _open_media_list,
    _two_media_items,
)
from Tests.UI.test_library_shell import (  # noqa: E402
    LibraryGlobalKeyProductionCSSHarness,
    LibraryProductionCSSHarness,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
)


@pytest.mark.parametrize("size", [(235, 52), (100, 30)])
@pytest.mark.asyncio
async def test_space_on_a_focused_media_row_never_collapses_a_pane(size):
    """task-31567 AC#2/AC#3: a row keeps focus across a recompose, so Space
    never reaches a pane grip."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryGlobalKeyProductionCSSHarness(app)

    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-row-0", Button).focus()
        await pilot.pause()

        await pilot.press("s")
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_select_mode,
            message="Select mode never engaged after 's'.",
        )
        for _ in range(3):
            await pilot.pause()
        await pilot.press("s")
        await _wait_for_condition(
            pilot,
            lambda: not screen._library_media_select_mode,
            message="Select mode never left after the second 's'.",
        )
        for _ in range(3):
            await pilot.pause()

        focused = screen.focused
        assert focused is not None, "the recompose left nothing focused"
        assert not focused.has_class("library-adaptive-reader-pane-grip"), focused
        assert focused.has_class("library-media-row"), focused

        layout_before = screen._library_media_reader_layout
        await pilot.press("space")
        for _ in range(3):
            await pilot.pause()

        assert screen._library_media_reader_layout == layout_before, (
            f"Space collapsed a pane: {layout_before} -> "
            f"{screen._library_media_reader_layout}"
        )
        assert not screen.focused.has_class(
            "library-adaptive-reader-pane-grip"
        ), screen.focused


# ---------------------------------------------------------------------------
# task-31633 AC#1/AC#4: the Items column grows with the terminal.
#
# Painted, not just resolved: critique #5 P1 measured a 98-character title
# truncated after 31 characters at 235x52 while the SAME title survived to 39
# at 100x30 -- the wider terminal painted the narrower list. The 100x30 pane
# width is pinned exactly, because PR D/F/G rows are sized against it.
# ---------------------------------------------------------------------------

MEDIA_LONG_TITLE = (
    "Quarterly roadmap interview recording with the leadership panel "
    "and appendix notes " * 2
)[:98]


def _media_items_with_a_long_title() -> list[dict[str, object]]:
    rows = _two_media_items()
    rows[0]["title"] = MEDIA_LONG_TITLE
    return rows


def _painted_lines(host, region) -> list[str]:
    strips = list(host.screen._compositor.render_strips())
    return [
        strips[y].crop(region.x, region.right).text
        for y in range(region.y, min(region.bottom, len(strips)))
    ]


@pytest.mark.parametrize(
    ("size", "expected_items_width", "expected_title_characters"),
    [
        # 235x52 was 40 cells / 31 painted title characters at badff73f1.
        # 56 is the profile's list comfort ceiling, the same one the
        # library-closed branch of the resolver already uses.
        ((235, 52), 56, 46),
        # 100x30 was 44 cells / 39 painted title characters while the two pane
        # grips still cost five columns each (task-31633 AC#2). The Reader is
        # on its 46-cell minimum here, so the eight cells the grips gave back
        # went to the list -- and the same 98-character title now survives to
        # the same 46 characters at BOTH sizes.
        ((100, 30), 52, 46),
    ],
)
@pytest.mark.asyncio
async def test_media_items_pane_grows_with_the_terminal_once_reader_is_comfortable(
    size: tuple[int, int],
    expected_items_width: int,
    expected_title_characters: int,
) -> None:
    assert len(MEDIA_LONG_TITLE) == 98
    app = _build_media_test_app()
    _seed_conversations(
        app, _two_conversations(), media=_media_items_with_a_long_title()
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        for _ in range(4):
            await pilot.pause()

        shell = screen.query_one(
            "#library-media-reader-shell", LibraryAdaptiveReaderShell
        )
        row = next(
            candidate
            for candidate in screen.query(".library-media-row").results(Button)
            if MEDIA_LONG_TITLE[:20] in str(candidate.label)
        )
        painted_title = _painted_lines(host, row.region)[0]

        assert "\u2026" in painted_title, painted_title
        visible = painted_title.split("\u2026", 1)[0].strip()
        assert MEDIA_LONG_TITLE.startswith(visible), (visible, painted_title)
        assert (shell.items.region.width, len(visible)) == (
            expected_items_width,
            expected_title_characters,
        ), (shell.items.region, painted_title)
        assert shell.work.region.width >= READER_COMFORT_WIDTH
        if size == (235, 52):
            # AC#4 floors: the wide list is at least as wide as the compact
            # one, and the 98-character title survives past 44 characters.
            assert shell.items.region.width >= 47
            assert len(visible) >= 44


# ---------------------------------------------------------------------------
# task-31633 AC#2: no 5-cell dead gutter between rail, list and Reader.
#
# Painted, not region-only: the gutter is the pane grip's own columns, and a
# region assertion is blind to what those columns actually carry. The slice
# bounds come from the pane regions, but every assertion below reads glyphs --
# the panes' own border glyphs anchor the slice, and the slice itself must be
# dead (the grip's arrow paints on its own rows, not on a list row).
# ---------------------------------------------------------------------------

MAX_PANE_GUTTER_CELLS = 2


def _dead_gutters(host, screen, shell) -> tuple[str, str]:
    """Return the painted columns flanking the Items pane on a list row."""
    row = screen.query_one("#library-media-row-0", Button)
    painted = list(host.screen._compositor.render_strips())[row.region.y].text
    return (
        painted[shell.library.region.right : shell.items.region.x],
        painted[shell.items.region.right : shell.work.region.x],
    )


@pytest.mark.asyncio
async def test_no_dead_gutter_flanks_the_media_items_pane_at_235x52() -> None:
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        for _ in range(4):
            await pilot.pause()

        shell = screen.query_one(
            "#library-media-reader-shell", LibraryAdaptiveReaderShell
        )
        assert shell.library.display, "the rail pane is closed at 235x52"
        left_gutter, right_gutter = _dead_gutters(host, screen, shell)
        row = screen.query_one("#library-media-row-0", Button)
        painted = list(host.screen._compositor.render_strips())[row.region.y].text

        # Anchors: the slice really is the run between two painted panes.
        assert painted[shell.library.region.right - 1] == "│", painted
        assert painted[shell.items.region.x] == "│", painted
        assert painted[shell.items.region.right - 1] == "│", painted

        assert left_gutter.strip() == "", (left_gutter, painted)
        assert right_gutter.strip() == "", (right_gutter, painted)
        assert len(left_gutter) <= MAX_PANE_GUTTER_CELLS, (left_gutter, painted)
        assert len(right_gutter) <= MAX_PANE_GUTTER_CELLS, (right_gutter, painted)
