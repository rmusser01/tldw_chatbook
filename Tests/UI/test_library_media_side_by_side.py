"""Library Media canvas side-by-side list|detail at wide widths (task-14900).

The Media canvas's list view renders list | preview side by side above the
screen's one measured width regime (``LIBRARY_NOTES_COMPACT_BREAKPOINT``,
applied as the ``library-notes-compact`` class on ``#library-canvas``) and
preserves the stacked layout below it. Geometry is asserted on REAL rendered
regions with the real ``LibraryScreen`` mounted in ``LibraryHarness`` (which
loads the real app stylesheet bundle) -- a canvas mounted alone in a bare App
is not measured against the tier that wins live.

Covers the three ACs: side-by-side wide / stacked narrow (geometry pins),
keyboard traversal incl. viewer entry in both layouts + footer honesty, and
Select mode / bulk toolbar usability in both layouts.
"""

import pytest
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)

#: Wide regime: the shell grid measures >= the compact breakpoint (120).
WIDE_SIZE = LIBRARY_TEST_SIZE  # (170, 48)
#: Stacked regime: below the breakpoint; a size test_library_shell already uses.
NARROW_SIZE = (100, 30)


def _build_media_test_app():
    """Return a populated-Library app rather than a fresh Starter profile."""
    app = _build_test_app()
    app.library_new_profile_admission = False
    return app


def _many_media_items(count: int = 45) -> list[dict[str, object]]:
    """Return deterministic production-shaped rows for paging/geometry."""
    return [
        {
            "id": f"media-{index + 1}",
            "title": f"Media item {index + 1:02d}",
            "type": ("video", "audio", "PDF")[index % 3],
            "last_modified": f"2026-07-{(index % 28) + 1:02d}T10:00:00Z",
            "content": f"Transcript for media item {index + 1}.",
            "version": 1,
        }
        for index in range(count)
    ]


async def _open_media_list(host, pilot):
    """Select the Media rail row and wait for the list rows to mount."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-media").press()
    await _wait_for_selector(screen, pilot, "#library-media-row-1")
    return screen


async def _wait_for_compact_class(screen, pilot, *, compact: bool):
    """Await the host's measured-crossing class matching the terminal size."""
    host_pane = screen.query_one("#library-canvas")
    await _wait_for_condition(
        pilot,
        lambda: host_pane.has_class("library-notes-compact") is compact,
        message=(
            f"#library-canvas never reached library-notes-compact={compact}; "
            f"classes: {host_pane.classes}"
        ),
    )
    await pilot.pause()


# ---------------------------------------------------------------------------
# AC#1: side-by-side at wide widths; stacked below the breakpoint.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_media_paints_five_one_line_rows_and_hides_preview():
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=NARROW_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(screen, pilot, compact=True)
        await _wait_for_selector(screen, pilot, "#library-media-row-19")

        rows = list(screen.query(".library-media-row"))
        painted = [row for row in rows if row.region.area > 0]
        assert len(painted) >= 5
        assert all(row.region.height == 1 for row in painted[:5])
        assert all(" · " in str(row.label) for row in painted[:5])
        assert all("\n" not in str(row.label) for row in painted[:5])
        assert all("▸" not in str(row.label) for row in painted[:5])
        assert not any(
            row.has_class("library-media-row-selected") for row in painted[:5]
        )

        preview = screen.query_one("#library-media-preview")
        open_viewer = screen.query_one("#library-media-open-viewer", Button)
        assert preview.region.area == 0
        assert open_viewer.can_focus is False


@pytest.mark.asyncio
async def test_wide_media_keeps_two_line_rows_and_preview():
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(screen, pilot, compact=False)

        row = screen.query_one("#library-media-row-0", Button)
        preview = screen.query_one("#library-media-preview")
        open_viewer = screen.query_one("#library-media-open-viewer", Button)
        assert row.region.height == 2
        assert "\n" in str(row.label)
        assert preview.region.area > 0
        assert open_viewer.can_focus is True


@pytest.mark.asyncio
async def test_media_list_and_preview_side_by_side_at_wide_width():
    """At 170 cols the preview renders BESIDE the list (same row band,
    strictly to its right), not below it -- the task-14900 defect was a
    blank right half-canvas while list and preview stacked vertically."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(screen, pilot, compact=False)

        media_list = screen.query_one("#library-media-list")
        preview = screen.query_one("#library-media-preview")
        assert preview.display is True

        list_region = media_list.region
        preview_region = preview.region
        assert list_region.width > 0 and preview_region.width > 0
        # Side by side: the preview starts at/right of the list's right edge…
        assert preview_region.x >= list_region.x + list_region.width
        # …in the same horizontal band (tops align inside the workbench).
        assert preview_region.y == list_region.y
        # Both halves get real width -- neither pane collapsed.
        assert list_region.width >= 30
        assert preview_region.width >= 30


@pytest.mark.asyncio
async def test_media_list_hides_preview_below_breakpoint():
    """At 100 cols the list owns the canvas and the preview is unpainted."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=NARROW_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(screen, pilot, compact=True)

        media_list = screen.query_one("#library-media-list")
        preview = screen.query_one("#library-media-preview")
        assert preview.display is False

        list_region = media_list.region
        preview_region = preview.region
        assert list_region.width > 0
        assert preview_region.area == 0
        assert list_region.width >= int(
            screen.query_one("#library-media-workbench").region.width * 0.9
        )


@pytest.mark.asyncio
async def test_trash_view_stays_single_column_at_wide_width():
    """The split applies to the LIST view only: Trash is a list-only surface
    (no detail half exists in its state), so at wide widths its rows keep the
    full canvas width and no media workbench container is present."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-trash-open").press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-title")

        assert not screen.query("#library-media-workbench")
        canvas = screen.query_one("#library-media-trash-canvas")
        trash_list = screen.query_one("#library-media-trash-list")
        assert trash_list.region.width >= int(canvas.region.width * 0.9)


# ---------------------------------------------------------------------------
# AC#2: keyboard traversal (rows, preview actions, viewer entry) + footer.
# ---------------------------------------------------------------------------


async def _assert_keyboard_traversal_and_viewer_entry(size):
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(
            screen, pilot, compact=size[0] < 120
        )

        # Footer honesty: the plain media list advertises the shared
        # list-canvas set through the one seam, in BOTH layouts.
        assert (
            screen._library_footer_shortcuts_for_current_state()
            == screen.LIBRARY_LIST_SHORTCUTS
        )

        # Rows: Up/Down move DOM focus between rows.
        row_0 = screen.query_one("#library-media-row-0", Button)
        row_1 = screen.query_one("#library-media-row-1", Button)
        row_0.focus()
        await pilot.pause()
        await pilot.press("down")
        assert screen.focused is row_1
        await pilot.press("up")
        assert screen.focused is row_0

        open_viewer = screen.query_one("#library-media-open-viewer", Button)
        previewed_title = str(
            screen.query_one("#library-media-preview-lines", Static).renderable
        ).splitlines()[0]
        if size[0] >= 120:
            # Wide: Tab from the last row reaches the preview action, whose
            # activation opens the preview-selected item.
            row_1.focus()
            await pilot.pause()
            await pilot.press("tab")
            assert screen.focused is open_viewer
            await pilot.press("enter")
        else:
            # Compact: the hidden preview action leaves traversal and a row
            # activation opens that row in the same full viewer.
            assert open_viewer.can_focus is False
            row_0.focus()
            await pilot.pause()
            await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-media-viewer-title")
        assert screen._library_media_view == "viewer"
        title = str(
            screen.query_one("#library-media-viewer-title", Static).renderable
        )
        assert title == previewed_title == "Product Demo Video"


@pytest.mark.asyncio
async def test_keyboard_traversal_and_viewer_entry_wide():
    await _assert_keyboard_traversal_and_viewer_entry(WIDE_SIZE)


@pytest.mark.asyncio
async def test_keyboard_traversal_and_viewer_entry_narrow():
    await _assert_keyboard_traversal_and_viewer_entry(NARROW_SIZE)


# ---------------------------------------------------------------------------
# AC#3: Select mode + bulk toolbar fully usable in both layouts.
# ---------------------------------------------------------------------------


async def _assert_select_mode_bulk_toolbar_usable(size):
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(
            screen, pilot, compact=size[0] < 120
        )

        screen.query_one("#library-media-select-toggle").press()
        await _wait_for_selector(screen, pilot, "#library-media-select-all")

        # Every bulk action renders and at least STARTS on-screen (the known
        # failure mode pushed the buttons entirely past the terminal's right
        # edge while keeping them in the DOM, where query_one still finds
        # them). In the WIDE layout the whole toolbar must fit; at 100 cols
        # "Delete selected"'s tail already overhung the edge BEFORE this
        # task (A/B-verified identical at base 345da0422: x=90, width=21 on
        # a 100-col terminal) -- that pre-existing stacked-layout overflow
        # is pinned as-is here and tracked in task-15140, not silently
        # blessed as the wide layout's contract.
        wide = size[0] >= 120
        for selector in (
            "#library-media-select-all",
            "#library-media-select-clear",
            "#library-media-export-selected",
            "#library-media-delete-selected",
            "#library-media-selected-count",
        ):
            widget = screen.query_one(selector)
            region = widget.region
            assert region.width > 0, f"{selector} has no rendered width"
            assert region.x < size[0], f"{selector} starts past the terminal edge"
            if wide:
                assert (
                    region.x + region.width <= size[0]
                ), f"{selector} extends past the terminal edge"

        # The toolbar is usable: toggle a row, watch the count patch in
        # place, arm bulk delete, then cancel it.
        screen.query_one("#library-media-row-0", Button).press()
        await pilot.pause()
        count = screen.query_one("#library-media-selected-count", Static)
        assert "1 selected" in str(count.renderable)

        screen.query_one("#library-media-delete-selected", Button).press()
        await _wait_for_selector(
            screen, pilot, "#library-media-bulk-delete-confirm-copy"
        )
        confirm = screen.query_one("#library-media-bulk-delete-confirm")
        cancel = screen.query_one("#library-media-bulk-delete-cancel")
        for widget in (confirm, cancel):
            region = widget.region
            assert region.width > 0
            assert region.x + region.width <= size[0]
        cancel.press()
        await pilot.pause()
        assert not screen.query("#library-media-bulk-delete-confirm-copy")


@pytest.mark.asyncio
async def test_select_mode_bulk_toolbar_usable_wide():
    await _assert_select_mode_bulk_toolbar_usable(WIDE_SIZE)


@pytest.mark.asyncio
async def test_select_mode_bulk_toolbar_usable_narrow():
    await _assert_select_mode_bulk_toolbar_usable(NARROW_SIZE)


async def _assert_select_mode_keyboard_toggle_and_footer(size):
    """Keyboard-only Select mode: Enter on a focused row toggles it through
    the in-place patcher (marker preserved -- the label rewrite must keep
    the ☑/☐ at position 0), and the footer seam stays honest through the
    armed bulk-delete sub-state in this layout."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(
            screen, pilot, compact=size[0] < 120
        )

        screen.query_one("#library-media-select-toggle").press()
        await _wait_for_selector(screen, pilot, "#library-media-select-all")

        # Keyboard toggle: focus a row, press Enter -- the in-place patcher
        # (_apply_library_row_toggle) must flip the marker to ☑ while
        # PRESERVING the rest of the label, and patch the count in place.
        row_0 = screen.query_one("#library-media-row-0", Button)
        row_0.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        label = str(row_0.label)
        assert label.startswith("☑"), f"toggle lost the marker: {label!r}"
        count = screen.query_one("#library-media-selected-count", Static)
        assert "1 selected" in str(count.renderable)

        # Down reaches the next row; Enter toggles it too (traversal is not
        # a browse-mode-only affordance).
        await pilot.press("down")
        row_1 = screen.query_one("#library-media-row-1", Button)
        assert screen.focused is row_1
        await pilot.press("enter")
        await pilot.pause()
        assert str(row_1.label).startswith("☑")
        assert "2 selected" in str(count.renderable)

        # Footer honesty through the armed sub-state: arming bulk delete
        # swaps the advertised set to the confirm context (esc cancels), in
        # this layout exactly as in the other.
        assert (
            screen._library_footer_shortcuts_for_current_state()
            == screen.LIBRARY_LIST_SHORTCUTS
        )
        screen.query_one("#library-media-delete-selected", Button).press()
        await _wait_for_selector(
            screen, pilot, "#library-media-bulk-delete-confirm-copy"
        )
        assert (
            screen._library_footer_shortcuts_for_current_state()
            == screen.LIBRARY_MEDIA_BULK_DELETE_CONFIRM_SHORTCUTS
        )
        await pilot.press("escape")
        await pilot.pause()
        assert not screen.query("#library-media-bulk-delete-confirm-copy")
        assert (
            screen._library_footer_shortcuts_for_current_state()
            == screen.LIBRARY_LIST_SHORTCUTS
        )


@pytest.mark.asyncio
async def test_select_mode_keyboard_toggle_and_footer_wide():
    await _assert_select_mode_keyboard_toggle_and_footer(WIDE_SIZE)


@pytest.mark.asyncio
async def test_select_mode_keyboard_toggle_and_footer_narrow():
    await _assert_select_mode_keyboard_toggle_and_footer(NARROW_SIZE)


# ---------------------------------------------------------------------------
# The wide-only detail placeholder (Collections' detail-pane grammar).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wide_select_mode_shows_detail_placeholder():
    """In the wide split, Select mode hides the preview (task-2853 AC4), so
    the detail half explains itself instead of sitting blank."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(screen, pilot, compact=False)

        # Browse mode with a selection: the preview owns the detail half,
        # the placeholder is hidden.
        placeholder = screen.query_one("#library-media-detail-empty", Static)
        assert placeholder.region.area == 0

        screen.query_one("#library-media-select-toggle").press()
        await _wait_for_selector(screen, pilot, "#library-media-select-all")

        preview = screen.query_one("#library-media-preview")
        assert preview.display is False
        placeholder = screen.query_one("#library-media-detail-empty", Static)
        assert placeholder.region.area > 0
        assert "No preview in Select mode." in str(placeholder.renderable)


@pytest.mark.asyncio
async def test_narrow_select_mode_keeps_placeholder_hidden():
    """The placeholder is a wide-layout affordance only -- the preserved
    stacked layout renders nothing below the list in Select mode, exactly
    as before this task."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=NARROW_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(screen, pilot, compact=True)

        screen.query_one("#library-media-select-toggle").press()
        await _wait_for_selector(screen, pilot, "#library-media-select-all")

        placeholder = screen.query_one("#library-media-detail-empty", Static)
        assert placeholder.region.area == 0
