"""Library Media canvas side-by-side list|detail at wide widths (task-14900).

The Media canvas's list view renders list | preview side by side above the
screen's one measured width regime (``LIBRARY_NOTES_COMPACT_BREAKPOINT``,
applied as the ``library-notes-compact`` class on ``#library-canvas``) and
uses a preview-free dense list below it. Geometry is asserted on REAL rendered
regions with the real ``LibraryScreen`` mounted in ``LibraryHarness`` (which
loads the real app stylesheet bundle) -- a canvas mounted alone in a bare App
is not measured against the tier that wins live.

Covers the three ACs: side-by-side wide / dense-list narrow (geometry pins),
keyboard traversal incl. viewer entry in both layouts + footer honesty, and
Select mode / bulk toolbar usability in both layouts.
"""

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    DoubleShrinkLibraryMediaScopeService,
    GatedFailingSecondLibraryMediaScopeService,
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
from tldw_chatbook.Library.library_media_state import MediaBrowseScope
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

#: Wide regime: the shell grid measures >= the compact breakpoint (120).
WIDE_SIZE = LIBRARY_TEST_SIZE  # (170, 48)
#: Stacked regime: below the breakpoint; a size test_library_shell already uses.
NARROW_SIZE = (100, 30)
COMPACT_SCROLL_SIZE = (100, 20)


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
async def test_media_resize_preserves_scope_focus_scroll_without_reads():
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    service = app.media_reading_scope_service
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=NARROW_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(screen, pilot, compact=True)
        await _wait_for_selector(screen, pilot, "#library-media-row-19")
        controller = screen._library_media_browse_controller
        row = screen.query_one("#library-media-row-19", Button)
        scroll = screen.query_one("#library-media-row-scroll", VerticalScroll)
        row.focus()
        await pilot.pause()
        initial_scroll = scroll.scroll_y
        initial_scope = controller.applied_scope
        initial_selection = screen._library_media_row_selection.ids
        initial_calls = (len(service.search_calls), len(service.type_calls))

        await pilot.resize_terminal(*WIDE_SIZE)
        await _wait_for_compact_class(screen, pilot, compact=False)
        assert row.region.height == 2
        assert "\n" in str(row.label)
        assert screen.focused is row

        await pilot.resize_terminal(*NARROW_SIZE)
        await _wait_for_compact_class(screen, pilot, compact=True)
        assert row.styles.height.value == 1
        assert "\n" not in str(row.label)
        assert screen.focused is row
        assert scroll.scroll_y == initial_scroll
        assert 0 <= scroll.scroll_y <= scroll.max_scroll_y
        assert controller.applied_scope == initial_scope
        assert screen._library_media_row_selection.ids == initial_selection
        assert (len(service.search_calls), len(service.type_calls)) == initial_calls


@pytest.mark.asyncio
async def test_media_preview_focus_moves_to_selected_row_on_compact_resize():
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(screen, pilot, compact=False)
        selected_id = screen._build_library_media_state().selected_id
        open_viewer = screen.query_one("#library-media-open-viewer", Button)
        open_viewer.focus()
        await pilot.pause()
        assert screen.focused is open_viewer

        await pilot.resize_terminal(*NARROW_SIZE)
        await _wait_for_compact_class(screen, pilot, compact=True)
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "media_id", None) == selected_id,
            message="Compact resize did not transfer preview focus to its row.",
        )
        assert open_viewer.can_focus is False


@pytest.mark.asyncio
async def test_media_resize_focus_restore_yields_to_newer_user_focus(monkeypatch):
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(screen, pilot, compact=False)
        open_viewer = screen.query_one("#library-media-open-viewer", Button)
        open_viewer.focus()
        await pilot.pause()

        pending = []
        restore = screen._restore_library_notes_focus_identity

        def hold_restore(identity, guard=None):
            pending.append((identity, guard))
            return False

        monkeypatch.setattr(
            screen,
            "_restore_library_notes_focus_identity",
            hold_restore,
        )
        await pilot.resize_terminal(*NARROW_SIZE)
        await _wait_for_compact_class(screen, pilot, compact=True)
        await _wait_for_condition(
            pilot,
            lambda: bool(pending),
            message="Resize did not queue a semantic Media focus restore.",
        )

        type_filter = screen.query_one("#library-media-type-filter", Button)
        screen._mark_library_notes_user_interaction()
        type_filter.focus()
        await pilot.pause()
        monkeypatch.setattr(
            screen,
            "_restore_library_notes_focus_identity",
            restore,
        )
        for identity, guard in pending:
            restore(identity, guard)
        assert screen.focused is type_filter


async def _open_scrolled_compact_media_viewer(host, pilot, *, row_index=15):
    """Open a non-first compact row after recording its real scroll owner."""
    screen = await _open_media_list(host, pilot)
    await _wait_for_compact_class(screen, pilot, compact=True)
    row = screen.query_one(f"#library-media-row-{row_index}", Button)
    scroll = screen.query_one("#library-media-row-scroll", VerticalScroll)
    row.focus()
    row.scroll_visible(animate=False, force=True, immediate=True)
    await _wait_for_condition(
        pilot,
        lambda: scroll.scroll_y > 0,
        message="Compact Media row pane never reached a nonzero scroll offset.",
    )
    media_id = row.media_id
    offset = (int(scroll.scroll_x), int(scroll.scroll_y))
    row.press()
    await _wait_for_selector(screen, pilot, "#library-media-back")
    return screen, media_id, offset


@pytest.mark.asyncio
async def test_compact_media_viewer_back_restores_semantic_row_and_scroll():
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    service = app.media_reading_scope_service
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, media_id, scroll_offset = await _open_scrolled_compact_media_viewer(
            host, pilot
        )
        applied_scope = screen._library_media_browse_controller.applied_scope
        reads_before_back = len(service.search_calls)
        screen.query_one("#library-media-back", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-row-scroll")
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "media_id", None) == media_id,
            message="Viewer Back did not restore the activated Media row.",
        )
        scroll = screen.query_one("#library-media-row-scroll", VerticalScroll)
        assert (int(scroll.scroll_x), int(scroll.scroll_y)) == scroll_offset
        assert screen._library_media_browse_controller.applied_scope == applied_scope
        assert len(service.search_calls) == reads_before_back


@pytest.mark.asyncio
async def test_compact_media_viewer_back_survives_authoritative_recompose():
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, media_id, scroll_offset = await _open_scrolled_compact_media_viewer(
            host, pilot
        )
        screen.query_one("#library-media-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "media_id", None) == media_id,
            message="Initial viewer return did not restore its Media row.",
        )

        screen.refresh(recompose=True)
        await _wait_for_selector(screen, pilot, "#library-media-row-scroll")
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "media_id", None) == media_id,
            message="Authoritative recompose lost the pending Media return row.",
        )
        scroll = screen.query_one("#library-media-row-scroll", VerticalScroll)
        assert (int(scroll.scroll_x), int(scroll.scroll_y)) == scroll_offset


@pytest.mark.asyncio
async def test_compact_media_viewer_back_survives_targeted_reorder():
    app = _build_media_test_app()
    media = _many_media_items()
    _seed_conversations(app, _two_conversations(), media=media)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, media_id, _scroll_offset = await _open_scrolled_compact_media_viewer(
            host, pilot
        )
        screen.query_one("#library-media-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "media_id", None) == media_id,
            message="Initial viewer return did not restore its Media row.",
        )

        backing_id = int(media_id.rsplit(":", 1)[1])
        moved = next(item for item in media if item["id"] == f"media-{backing_id}")
        moved["last_modified"] = "2030-01-01T00:00:00Z"
        controller = screen._library_media_browse_controller
        screen._request_library_media_browse(
            controller.mutation_refresh_scope,
            focus_identity=None,
        )
        await _wait_for_condition(
            pilot,
            lambda: not controller.loading,
            message="Targeted reorder never settled.",
        )
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "media_id", None) == media_id
            and getattr(screen.focused, "id", None) == "library-media-row-0",
            message=lambda: (
                "Targeted reorder restored a stale row index instead of Media "
                f"identity: focused={getattr(screen.focused, 'id', None)!r}/"
                f"{getattr(screen.focused, 'media_id', None)!r}; "
                f"pending={screen._library_pending_list_entry_focus!r}; "
                f"return={screen._library_pending_list_entry_media_return!r}; "
                f"restoring={screen._library_notes_restoring_focus!r}."
            ),
        )


@pytest.mark.asyncio
async def test_compact_media_viewer_back_falls_back_after_row_removed():
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, media_id, _scroll_offset = await _open_scrolled_compact_media_viewer(
            host, pilot
        )
        controller = screen._library_media_browse_controller
        service = app.media_reading_scope_service
        removed_backing = int(media_id.rsplit(":", 1)[1])
        service.media_items = [
            item
            for index, item in enumerate(service.media_items)
            if service._backing_id(item, index) != removed_backing
        ]
        screen._request_library_media_browse(
            controller.mutation_refresh_scope,
            focus_identity=None,
        )
        await _wait_for_condition(
            pilot,
            lambda: controller.applied_result is not None
            and controller.applied_result.total == 44
            and not controller.loading,
            message="Authoritative page without the removed row never applied.",
        )
        expected = controller.retained_items[0]["id"]

        screen.query_one("#library-media-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "media_id", None) == expected,
            message="Removed viewer row did not fall back to the first retained row.",
        )


@pytest.mark.asyncio
async def test_compact_media_viewer_back_follows_single_page_clamp():
    app = _build_media_test_app()
    media = _many_media_items(21)
    _seed_conversations(app, _two_conversations(), media=media)
    service = app.media_reading_scope_service
    screen = LibraryScreen(app)
    screen._library_selected_row_id = "browse-media"
    screen._library_media_browse_controller.invalidate(MediaBrowseScope(page=2))
    host = LibraryProductionCSSHarness(app, screen=screen)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-media-row-0")
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_browse_controller.applied_scope
            == MediaBrowseScope(page=2),
            message="Restored Media page 2 never applied.",
        )
        row = screen.query_one("#library-media-row-0", Button)
        target_id = row.media_id
        row.press()
        await _wait_for_selector(screen, pilot, "#library-media-back")
        reads_before_back = len(service.search_calls)
        screen.query_one("#library-media-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "media_id", None) == target_id,
            message="Viewer Back did not first restore the retained page-2 row.",
        )
        assert len(service.search_calls) == reads_before_back == 1

        target_backing = int(target_id.rsplit(":", 1)[1])
        service.media_items = [
            item
            for index, item in enumerate(service.media_items)
            if service._backing_id(item, index) != target_backing
        ]
        controller = screen._library_media_browse_controller
        screen._request_library_media_browse(
            controller.mutation_refresh_scope,
            focus_identity=None,
        )
        await _wait_for_condition(
            pilot,
            lambda: controller.applied_scope == MediaBrowseScope(page=1)
            and not controller.loading,
            message="Shrunken page 2 did not clamp once to page 1.",
        )
        assert [call["offset"] for call in service.search_calls] == [20, 20, 0]
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "media_id", None)
            == controller.retained_items[0]["id"],
            message="Clamped page did not fall back to its first authoritative row.",
        )


@pytest.mark.asyncio
async def test_compact_media_viewer_back_empty_page_focuses_recovery_control():
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    service = app.media_reading_scope_service
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, _media_id, _scroll_offset = await _open_scrolled_compact_media_viewer(
            host, pilot
        )
        controller = screen._library_media_browse_controller
        service.media_items = []
        screen._request_library_media_browse(
            controller.mutation_refresh_scope,
            focus_identity=None,
        )
        await _wait_for_condition(
            pilot,
            lambda: controller.applied_result is not None
            and controller.applied_result.total == 0
            and not controller.loading,
            message="Exact empty Media result never applied in the viewer.",
        )
        reads_before_back = len(service.search_calls)

        screen.query_one("#library-media-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None)
            == "library-media-empty-import",
            message="Empty viewer return did not focus the Import recovery action.",
        )
        assert len(service.search_calls) == reads_before_back


@pytest.mark.asyncio
async def test_compact_media_viewer_back_restore_yields_to_newer_user_focus(
    monkeypatch,
):
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, _media_id, _scroll_offset = await _open_scrolled_compact_media_viewer(
            host, pilot
        )
        pending = []
        focus_entry = screen._focus_library_list_entry_if_current

        def hold_focus_entry(generation=None):
            pending.append(generation)

        monkeypatch.setattr(
            screen,
            "_focus_library_list_entry_if_current",
            hold_focus_entry,
        )
        screen.query_one("#library-media-back", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-type-filter")
        await _wait_for_condition(
            pilot,
            lambda: bool(pending),
            message="Viewer Back did not queue its bounded focus settlement.",
        )
        type_filter = screen.query_one("#library-media-type-filter", Button)
        screen._mark_library_notes_user_interaction()
        type_filter.focus()
        await pilot.pause()

        monkeypatch.setattr(
            screen,
            "_focus_library_list_entry_if_current",
            focus_entry,
        )
        focus_entry(pending[-1])
        assert screen.focused is type_filter


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


@pytest.mark.asyncio
async def test_compact_media_stale_and_retry_actions_remain_truthful():
    media = _many_media_items()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=media)
    service = GatedFailingSecondLibraryMediaScopeService(media)
    app.media_reading_scope_service = service
    host = LibraryProductionCSSHarness(app)

    try:
        async with host.run_test(size=NARROW_SIZE) as pilot:
            screen = await _open_media_list(host, pilot)
            controller = screen._library_media_browse_controller
            screen.query_one("#library-media-next", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    service.page_two_entered.is_set()
                    and controller.loading
                    and len(screen.query(".library-media-row")) == 20
                ),
                message="Compact page-2 loading never retained its mounted rows.",
            )
            assert controller.loading is True
            assert len(screen.query(".library-media-row")) == 20
            assert screen.query_one("#library-media-next", Button).disabled
            assert screen.query_one("#library-media-previous", Button).disabled

            service.page_two_release.set()
            await _wait_for_condition(
                pilot,
                lambda: (
                    bool(controller.error_copy)
                    and bool(screen.query("#library-media-retry"))
                ),
                message="Compact page failure never exposed retained Retry.",
            )
            assert len(screen.query(".library-media-row")) == 20
            assert not screen.query_one("#library-media-retry", Button).disabled

            screen._library_media_delete_receipt_ids = ("local:media:1",)
            app.media_reading_scope_service = DoubleShrinkLibraryMediaScopeService(
                media
            )
            screen._request_library_media_page(3, focus_identity=None)
            await _wait_for_condition(
                pilot,
                lambda: (
                    controller.freshness == "stale"
                    and bool(screen.query("#library-media-retry"))
                ),
                message="Compact double shrink never reached stale recovery.",
            )
            for selector in (
                "#library-media-row-0",
                "#library-media-select-toggle",
                "#library-media-export",
                "#library-media-bulk-delete-undo",
            ):
                action = screen.query_one(selector, Button)
                assert action.disabled, selector
                assert str(action.label).startswith("○"), selector
                assert action.tooltip == controller.stale_copy, selector
            assert not screen.query_one("#library-media-retry", Button).disabled
            assert not screen.query_one("#library-media-type-filter", Button).disabled

            await pilot.resize_terminal(*WIDE_SIZE)
            await _wait_for_compact_class(screen, pilot, compact=False)
            assert str(screen.query_one("#library-media-row-0", Button).label).startswith(
                "○"
            )
            await pilot.resize_terminal(*NARROW_SIZE)
            await _wait_for_compact_class(screen, pilot, compact=True)
            assert str(screen.query_one("#library-media-row-0", Button).label).startswith(
                "○"
            )

            screen._sync_library_media_browse_state(None)
            await _wait_for_condition(
                pilot,
                lambda: screen.query_one("#library-media-row-0", Button).disabled,
                message="Compact stale action gate did not survive recompose.",
            )
    finally:
        service.page_two_release.set()


@pytest.mark.asyncio
async def test_compact_media_pager_receipt_and_empty_states_remain_contained():
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=NARROW_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        canvas = screen.query_one("#library-media-canvas")
        pager = screen.query_one("#library-media-pager")
        assert "1-20 of 45" in str(
            screen.query_one("#library-media-page-status", Static).renderable
        )
        assert canvas.region.contains_region(pager.region)

        screen._library_media_delete_receipt_ids = ("local:media:1",)
        screen._sync_library_media_browse_state(None)
        receipt = await _wait_for_selector(
            screen, pilot, "#library-media-bulk-delete-receipt"
        )
        canvas = screen.query_one("#library-media-canvas")
        assert canvas.region.contains_region(receipt.region)
        assert screen.query_one("#library-media-bulk-delete-undo", Button).can_focus

    empty_app = _build_media_test_app()
    _seed_conversations(empty_app, _two_conversations(), media=[])
    empty_host = LibraryProductionCSSHarness(empty_app)
    async with empty_host.run_test(size=NARROW_SIZE) as pilot:
        empty_screen = _active_library_screen(empty_host)
        await _wait_for_library_shell(empty_screen, pilot)
        empty_screen.query_one("#library-row-browse-media", Button).press()
        empty_action = await _wait_for_selector(
            empty_screen, pilot, "#library-media-empty-import"
        )
        empty_canvas = empty_screen.query_one("#library-media-canvas")
        assert empty_canvas.region.contains_region(empty_action.region)
        assert empty_action.can_focus
        assert not empty_screen.query("#library-media-pager")


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
