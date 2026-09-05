"""Library Media Items and permanent Reader production-layout checks.

Media keeps its two-line Items rows and a separate permanent Reader at every
responsive width where Items is open. Geometry is asserted on real rendered
regions with the production ``LibraryScreen`` and consolidated stylesheet.
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
    if screen.query("#library-media-reader-shell"):
        compact = False
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
async def test_narrow_media_paints_five_two_line_rows_without_embedded_preview() -> None:
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
        assert all(row.region.height == 2 for row in painted[:5])
        assert all("\n" in str(row.label) for row in painted[:5])

        preview = screen.query_one("#library-media-preview")
        open_viewer = screen.query_one("#library-media-open-viewer", Button)
        assert preview.region.area == 0
        assert open_viewer.can_focus is False


@pytest.mark.asyncio
async def test_wide_media_keeps_two_line_rows_and_permanent_reader() -> None:
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
        assert preview.region.area == 0
        assert open_viewer.can_focus is False
        assert screen.query_one("#library-media-viewer").region.area > 0


@pytest.mark.asyncio
async def test_reader_mode_toolbar_has_one_body_and_reachable_primary_actions_at_80x24() -> None:
    """The compact Reader keeps primary actions on screen without hidden mode DOM."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(80, 24)) as pilot:
        screen = await _open_media_list(host, pilot)
        row = screen.query_one("#library-media-row-0", Button)
        row.press()
        await _wait_for_selector(screen, pilot, "#library-media-reader-mode-read")

        for selector in (
            "#library-media-reader-find",
            "#library-media-read-later",
            "#library-media-use-in-chat",
            "#library-media-reader-more",
        ):
            widget = screen.query_one(selector, Button)
            assert widget.region.width > 0
            assert widget.region.x < 80

        bodies = [
            "#library-media-reader-mode-read",
            "#library-media-reader-mode-analysis",
            "#library-media-reader-mode-highlights",
            "#library-media-reader-mode-info",
        ]
        assert sum(bool(screen.query(selector)) for selector in bodies) == 1


@pytest.mark.asyncio
async def test_media_resize_preserves_scope_focus_scroll_without_reads() -> None:
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
        assert row.styles.height.value == 2
        assert "\n" in str(row.label)
        assert screen.focused is row
        # Textual may minimally re-scroll the focused row as the viewport
        # height changes; the semantic row and valid scroll owner survive.
        assert scroll.scroll_y >= 0
        assert 0 <= scroll.scroll_y <= scroll.max_scroll_y
        assert controller.applied_scope == initial_scope
        assert screen._library_media_row_selection.ids == initial_selection
        assert (len(service.search_calls), len(service.type_calls)) == initial_calls


@pytest.mark.asyncio
async def test_media_row_focus_moves_to_items_grip_when_resize_hides_items() -> None:
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(screen, pilot, compact=False)
        row = screen.query_one("#library-media-row-0", Button)
        row.focus()
        await pilot.pause()
        assert screen.focused is row

        await pilot.resize_terminal(80, 24)
        await _wait_for_compact_class(screen, pilot, compact=True)
        grip = screen.query_one("#library-media-items-grip", Button)
        reader = screen.query_one("#library-media-viewer")
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is grip
            or (
                screen.focused is not None
                and (screen.focused is reader or reader in screen.focused.ancestors)
            ),
            message=lambda: (
                "Resize did not transfer hidden Items focus to a visible role: "
                f"focused={screen.focused!r}, focused_display="
                f"{getattr(screen.focused, 'display', None)!r}, layout="
                f"{screen._library_media_reader_layout!r}."
            ),
        )
        assert screen.query_one("#library-canvas").display is False
        assert row is not screen.focused


@pytest.mark.asyncio
async def test_media_resize_focus_restore_yields_to_newer_user_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(screen, pilot, compact=False)
        row = screen.query_one("#library-media-row-0", Button)
        row.focus()
        await pilot.pause()

        pending = []
        transfer_focus = screen._focus_library_media_grip_if_current

        def hold_transfer(generation, hidden_focus, grip):
            pending.append((generation, hidden_focus, grip))

        monkeypatch.setattr(
            screen,
            "_focus_library_media_grip_if_current",
            hold_transfer,
        )
        await pilot.resize_terminal(80, 24)
        await _wait_for_compact_class(screen, pilot, compact=True)
        await _wait_for_condition(
            pilot,
            lambda: bool(pending),
            message="Resize did not queue a semantic Media focus restore.",
        )

        library_grip = screen.query_one("#library-media-library-grip", Button)
        screen._mark_library_notes_user_interaction()
        library_grip.focus()
        await pilot.pause()
        monkeypatch.setattr(
            screen,
            "_focus_library_media_grip_if_current",
            transfer_focus,
        )
        for generation, hidden_focus, grip in pending:
            transfer_focus(generation, hidden_focus, grip)
        assert screen.focused is library_grip


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
async def test_compact_media_viewer_back_restores_semantic_row_and_scroll() -> None:
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
        await _wait_for_condition(
            pilot,
            lambda: screen._library_pending_list_entry_focus,
            message="Viewer Back did not arm its semantic Media return.",
        )
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
async def test_compact_media_viewer_back_survives_authoritative_recompose() -> None:
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
            lambda: screen._library_pending_list_entry_focus,
            message="Viewer Back did not arm its semantic Media return.",
        )
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
            message=lambda: (
                "Authoritative recompose lost the pending Media return row: "
                f"focused={screen.focused!r}, pending="
                f"{screen._library_pending_list_entry_focus!r}, return="
                f"{screen._library_pending_list_entry_media_return!r}, layout="
                f"{screen._library_media_reader_layout!r}."
            ),
        )
        scroll = screen.query_one("#library-media-row-scroll", VerticalScroll)
        assert (int(scroll.scroll_x), int(scroll.scroll_y)) == scroll_offset


@pytest.mark.asyncio
async def test_compact_media_viewer_back_wheel_scroll_cancels_stored_restore() -> None:
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, media_id, _scroll_offset = await _open_scrolled_compact_media_viewer(
            host, pilot
        )
        screen.query_one("#library-media-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_pending_list_entry_focus,
            message="Viewer Back did not arm its semantic Media return.",
        )
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "media_id", None) == media_id,
            message="Initial viewer return did not restore its Media row.",
        )
        scroll = screen.query_one("#library-media-row-scroll", VerticalScroll)
        scroll.scroll_to(y=0, animate=False, force=True, immediate=True)
        screen.on_mouse_scroll_up(object())
        await pilot.pause()

        assert screen._library_pending_list_entry_focus is False, (
            screen.focused,
            screen._library_pending_list_entry_media_return,
            screen._library_list_entry_focus_generation,
        )
        screen.refresh(recompose=True)
        await _wait_for_selector(screen, pilot, "#library-media-row-scroll")
        await pilot.pause()
        scroll = screen.query_one("#library-media-row-scroll", VerticalScroll)
        assert int(scroll.scroll_y) == 0


@pytest.mark.asyncio
async def test_compact_media_viewer_back_survives_targeted_reorder() -> None:
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
            lambda: screen._library_pending_list_entry_focus,
            message="Viewer Back did not arm its semantic Media return.",
        )
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
async def test_media_reader_back_keeps_a_retained_row_after_origin_removed() -> None:
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
        retained_ids = {item["id"] for item in controller.retained_items}

        screen.query_one("#library-media-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_pending_list_entry_focus,
            message="Viewer Back did not arm its semantic Media return.",
        )
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "media_id", None) in retained_ids,
            message=lambda: (
                "Removed Reader origin did not retain usable Items focus: "
                f"retained={retained_ids!r}, focused={screen.focused!r}, "
                f"focused_media={getattr(screen.focused, 'media_id', None)!r}, "
                f"pending={screen._library_pending_list_entry_focus!r}, return="
                f"{screen._library_pending_list_entry_media_return!r}."
            ),
        )
        assert getattr(screen.focused, "media_id", None) != media_id


@pytest.mark.asyncio
async def test_compact_media_viewer_back_follows_single_page_clamp() -> None:
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
            lambda: screen._library_pending_list_entry_focus,
            message="Viewer Back did not arm its semantic Media return.",
        )
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "media_id", None) == target_id,
            message=lambda: (
                "Viewer Back did not first restore the retained page-2 row: "
                f"target={target_id!r}, focused={screen.focused!r}, "
                f"focused_media={getattr(screen.focused, 'media_id', None)!r}, "
                f"pending={screen._library_pending_list_entry_focus!r}, return="
                f"{screen._library_pending_list_entry_media_return!r}, layout="
                f"{screen._library_media_reader_layout!r}."
            ),
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
async def test_compact_media_viewer_back_empty_page_focuses_recovery_control() -> None:
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
async def test_media_items_and_reader_are_side_by_side_at_wide_width() -> None:
    """At 170 columns Items and the permanent Reader occupy one row band."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(screen, pilot, compact=False)

        items = screen.query_one("#library-canvas")
        reader = screen.query_one("#library-media-viewer")

        assert items.region.width > 0 and reader.region.width > 0
        assert reader.region.x >= items.region.x + items.region.width
        assert reader.region.y == items.region.y
        assert items.region.width >= 32
        assert reader.region.width >= 44


@pytest.mark.asyncio
async def test_media_toolbar_actions_fit_the_items_panel_at_wide_width() -> None:
    """task-28025: Trash/Select fit the narrow Items panel, not clipped off it."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(screen, pilot, compact=False)

        def _panel_right() -> int:
            canvas = screen.query_one("#library-media-canvas")
            return canvas.region.x + canvas.region.width

        for selector in (
            "#library-media-type-filter",
            "#library-media-sort",
            "#library-media-export",
            "#library-media-trash-open",
            "#library-media-select-toggle",
        ):
            button = screen.query_one(selector, Button)
            assert button.region.width > 0, selector
            assert button.region.x + button.region.width <= _panel_right(), selector

        # Select mode's bulk bar must fit the panel too (same overflow class).
        screen.query_one("#library-media-select-toggle", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-select-all")
        for selector in (
            "#library-media-select-all",
            "#library-media-select-clear",
            "#library-media-export-selected",
        ):
            button = screen.query_one(selector, Button)
            assert button.region.width > 0, selector
            assert button.region.x + button.region.width <= _panel_right(), selector


@pytest.mark.asyncio
async def test_media_list_hides_preview_below_breakpoint() -> None:
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
async def test_trash_view_stays_single_column_at_wide_width() -> None:
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


@pytest.mark.asyncio
async def test_media_trash_collapse_choices_survive_filter_and_page_refresh() -> None:
    """Real pane-grip activation remains durable across Trash recomposes."""
    from textual.widgets import Input

    from Tests.UI.test_library_media_trash import (
        _MountedTrashFeed,
        _canonical_trash_items,
    )
    from tldw_chatbook.Library.library_media_state import MediaTrashScope

    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items())
    feed.install(app.media_reading_scope_service)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(160, 50)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-trash-open", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-row-0")
        controller = screen._library_media_trash_browse_controller

        items = screen.query_one("#library-canvas")
        initial_width = items.region.width
        library_grip = screen.query_one("#library-media-library-grip", Button)
        library_grip.focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: not screen._library_media_reader_layout.library_open,
            message="Library grip did not collapse the Library pane.",
        )
        assert items.region.width > initial_width

        search = screen.query_one("#library-media-trash-search", Input)
        search.focus()
        await pilot.press("t", "r", "a", "s", "h", "enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.state.applied_result is not None
                and controller.state.applied_result.scope.query == "trash"
            ),
            message="Trash filter refresh never settled.",
        )
        assert screen._library_media_reader_layout.library_open is False

        library_grip = screen.query_one("#library-media-library-grip", Button)
        library_grip.focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_layout.library_open,
            message="Library grip did not reopen the Library pane.",
        )

        items_grip = screen.query_one("#library-media-items-grip", Button)
        items_grip.focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: not screen._library_media_reader_layout.items_open,
            message="Items grip did not collapse the Items pane.",
        )
        screen._request_library_media_trash_page(
            2, focus_identity="#library-media-trash-next"
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.state.applied_result is not None
                and controller.state.applied_result.scope
                == MediaTrashScope(query="trash", page=2)
            ),
            message="Hidden Items page refresh never settled.",
        )
        assert screen._library_media_reader_layout.items_open is False


@pytest.mark.asyncio
async def test_media_trash_compact_pane_priority_survives_page_and_filter_refresh() -> (
    None
):
    """Compact recomposes retain whichever mutually exclusive pane was opened."""
    from textual.widgets import Input

    from Tests.UI.test_library_media_trash import (
        _MountedTrashFeed,
        _canonical_trash_items,
    )
    from tldw_chatbook.Library.library_media_state import MediaTrashScope

    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items())
    feed.install(app.media_reading_scope_service)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(80, 24)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_layout.reader_width > 0,
            message="Compact Media layout never settled.",
        )
        if not screen._library_media_reader_layout.items_open:
            items_grip = screen.query_one("#library-media-items-grip", Button)
            items_grip.focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: screen._library_media_reader_layout.items_open,
                message="Compact Items pane never opened.",
            )
        await pilot.pause()
        opener = screen.query_one("#library-media-trash-open", Button)
        opener.focus()
        await pilot.press("enter")
        controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.state.applied_result is not None,
            message="Compact Trash page never applied.",
        )

        library_grip = screen.query_one("#library-media-library-grip", Button)
        library_grip.focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_media_reader_layout.library_open
                and not screen._library_media_reader_layout.items_open
            ),
            message="Compact Library pane did not become the explicit priority.",
        )
        screen._sync_library_media_reader_layout_from_shell()
        await pilot.pause()
        assert screen._library_media_reader_layout.library_open is True
        assert screen._library_media_reader_layout.items_open is False
        screen._request_library_media_trash_page(
            2, focus_identity="#library-media-trash-next"
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.state.applied_result is not None
                and controller.state.applied_result.scope == MediaTrashScope(page=2)
            ),
            message="Compact page refresh never settled.",
        )
        assert screen._library_media_reader_layout.library_open is True
        assert screen._library_media_reader_layout.items_open is False

        items_grip = screen.query_one("#library-media-items-grip", Button)
        items_grip.focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_media_reader_layout.items_open
                and not screen._library_media_reader_layout.library_open
            ),
            message="Compact Items pane did not become the explicit priority.",
        )
        search = screen.query_one("#library-media-trash-search", Input)
        search.focus()
        await pilot.press("t", "r", "a", "s", "h", "enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.state.applied_result is not None
                and controller.state.applied_result.scope
                == MediaTrashScope(query="trash")
            ),
            message="Compact filter refresh never settled.",
        )
        assert screen._library_media_reader_layout.items_open is True
        assert screen._library_media_reader_layout.library_open is False


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

        # Footer honesty: the plain media list advertises its own set
        # (task-28012 adds the "s: select" key), in BOTH layouts.
        assert (
            screen._library_footer_shortcuts_for_current_state()
            == screen.LIBRARY_MEDIA_LIST_SHORTCUTS
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
        assert open_viewer.can_focus is False
        row_0.focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-media-viewer-title")
        assert screen._library_media_view == "viewer"
        assert row_0.is_mounted
        title = str(
            screen.query_one("#library-media-viewer-title", Static).renderable
        )
        assert title == "Product Demo Video"


@pytest.mark.asyncio
async def test_keyboard_traversal_and_viewer_entry_wide() -> None:
    await _assert_keyboard_traversal_and_viewer_entry(WIDE_SIZE)


@pytest.mark.asyncio
async def test_keyboard_traversal_and_viewer_entry_narrow() -> None:
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
async def test_select_mode_bulk_toolbar_usable_wide() -> None:
    await _assert_select_mode_bulk_toolbar_usable(WIDE_SIZE)


@pytest.mark.asyncio
async def test_select_mode_bulk_toolbar_usable_narrow() -> None:
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

        # Footer honesty through the armed sub-state: while selecting the
        # footer advertises the select keys (task-28012); arming bulk delete
        # swaps to the confirm context (esc cancels), in this layout exactly
        # as in the other.
        assert (
            screen._library_footer_shortcuts_for_current_state()
            == screen.LIBRARY_MEDIA_SELECT_SHORTCUTS
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
            == screen.LIBRARY_MEDIA_SELECT_SHORTCUTS
        )


@pytest.mark.asyncio
async def test_select_mode_keyboard_toggle_and_footer_wide() -> None:
    await _assert_select_mode_keyboard_toggle_and_footer(WIDE_SIZE)


@pytest.mark.asyncio
async def test_select_mode_keyboard_toggle_and_footer_narrow() -> None:
    await _assert_select_mode_keyboard_toggle_and_footer(NARROW_SIZE)


@pytest.mark.asyncio
async def test_compact_media_stale_and_retry_actions_remain_truthful() -> None:
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
            # task-31220: rows carry the mutation gate only -- reading an item
            # is how you recover from a stale page, so it is never gated by
            # that staleness.
            assert not screen.query_one("#library-media-row-0", Button).disabled

            # The density crossings still matter: ``apply_compact_presentation``
            # re-gates every row IN PLACE, so a crossing must not re-disable
            # (or re-mark) a row the stale gate no longer owns.
            for size, compact in ((WIDE_SIZE, False), (NARROW_SIZE, True)):
                await pilot.resize_terminal(*size)
                await _wait_for_compact_class(screen, pilot, compact=compact)
                row = screen.query_one("#library-media-row-0", Button)
                assert not row.disabled
                assert not str(row.label).startswith("○")

            screen._sync_library_media_browse_state(None)
            await _wait_for_condition(
                pilot,
                lambda: screen.query_one("#library-media-export", Button).disabled,
                message="Compact stale action gate did not survive recompose.",
            )
            assert not screen.query_one("#library-media-row-0", Button).disabled
    finally:
        service.page_two_release.set()


@pytest.mark.asyncio
async def test_compact_media_pager_receipt_and_empty_states_remain_contained() -> None:
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


@pytest.mark.asyncio
async def test_single_page_media_list_drops_pager_boundary_noise() -> None:
    # task-28016: one page of results has nowhere to page to, so the
    # "Page 1 of 1" counter and the boundary reasons ("Already on the first
    # page.", "No more results.") are suppressed. task-31237 (critique #3,
    # supersedes 28016's keep-the-disabled-controls choice): the two dead
    # "○ Previous ○ Next" forms are dropped entirely -- the item range
    # stays, and the controls return the moment a second page exists.
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items(3))
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        status = await _wait_for_selector(
            screen, pilot, "#library-media-page-status"
        )
        assert str(status.renderable) == "1-3 of 3"
        assert not screen.query("#library-media-disabled-reason")
        assert not screen.query("#library-media-next")
        assert not screen.query("#library-media-previous")


@pytest.mark.asyncio
async def test_multi_page_media_list_keeps_pager_controls() -> None:
    """A second page brings Previous/Next back (task-31237 negative control)."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items(45))
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_selector(screen, pilot, "#library-media-next")
        assert screen.query_one("#library-media-next", Button).disabled is False
        assert (
            screen.query_one("#library-media-previous", Button).disabled is True
        )


# ---------------------------------------------------------------------------
# The wide-only detail placeholder (Collections' detail-pane grammar).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wide_select_mode_keeps_permanent_reader() -> None:
    """Select mode changes Items controls without removing the Reader."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_compact_class(screen, pilot, compact=False)

        placeholder = screen.query_one("#library-media-detail-empty", Static)
        assert placeholder.region.area == 0
        reader = screen.query_one("#library-media-viewer")
        assert reader.region.area > 0

        screen.query_one("#library-media-select-toggle").press()
        await _wait_for_selector(screen, pilot, "#library-media-select-all")

        preview = screen.query_one("#library-media-preview")
        assert preview.display is False
        placeholder = screen.query_one("#library-media-detail-empty", Static)
        assert placeholder.region.area == 0
        assert screen.query_one("#library-media-viewer") is reader
        assert reader.region.area > 0


@pytest.mark.asyncio
async def test_narrow_select_mode_keeps_placeholder_hidden() -> None:
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
