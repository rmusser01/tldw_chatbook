"""Library media Trash view (task-4025): widget, handlers, restore seam.

The Trash view is the third ``_library_media_view`` value of the Browse ▸
Media canvas. These tests mirror ``test_library_multiselect_media.py``'s
structure: pilot tests for the canvas widget's rendered DOM, plain handler
tests over a SimpleNamespace fake, and real-DB tests (file-backed
``MediaDatabase``, never ``:memory:`` -- the restore worker hops threads via
``isolate_in_worker=True``) for the restore path, including a chunked item.
"""

import asyncio
import dataclasses
import threading
import types
from types import SimpleNamespace

import pytest

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Media import LocalMediaReadingService, MediaReadingScopeService
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Library.library_media_state import (
    LIBRARY_MEDIA_TRASH_EMPTY_COPY,
    LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_EMPTY_TOOLTIP,
    LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_ERROR_TOOLTIP,
    LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_LOADING_TOOLTIP,
    LIBRARY_MEDIA_TRASH_RESTORE_TOOLTIP,
    MediaBrowseScope,
    MediaTrashBrowseState,
    MediaTrashMutationTarget,
    MediaTrashScope,
    apply_media_trash_result,
    begin_media_trash_mutation,
    begin_media_trash_request,
    build_media_trash_result,
    build_library_media_trash_state,
    commit_media_trash_mutation,
    fail_media_trash_mutation,
    fail_media_trash_request,
)
from tldw_chatbook.Library.library_pager_state import build_library_pager_display
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
from tldw_chatbook.Widgets.Library.library_media_trash_canvas import (
    LibraryMediaTrashCanvas,
)
from tldw_chatbook.UI.Library_Modules.library_media_trash_browse_controller import (
    LibraryMediaTrashBrowseController,
    MediaTrashMutationClaim,
)


def _canonical_trash_items(count: int = 45) -> list[dict[str, object]]:
    return [
        {
            "id": f"local:media:{index}",
            "backing_media_id": index,
            "title": f"Trash {index:02d}",
            "media_type": "audio" if index % 2 else "video",
            "trash_date": f"2026-08-{(index % 28) + 1:02d}T00:00:00+00:00",
        }
        for index in range(1, count + 1)
    ]


class _MountedTrashFeed:
    """Production-shaped exact Trash source with deterministic gates/failures."""

    def __init__(
        self,
        items: list[dict[str, object]],
        *,
        fail_counts: dict[tuple[str, int], int] | None = None,
        gate_first: bool = False,
    ) -> None:
        self.items = items
        self.calls: list[dict[str, object]] = []
        self.fail_counts = dict(fail_counts or {})
        self.entered = threading.Event()
        self.release = threading.Event()
        if not gate_first:
            self.release.set()

    async def list_library_media_trash(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        if len(self.calls) == 1:
            self.entered.set()
            await asyncio.to_thread(self.release.wait, 10.0)
        query = str(kwargs.get("query") or "")
        offset = int(kwargs["offset"])
        key = (query, offset)
        remaining_failures = self.fail_counts.get(key, 0)
        if remaining_failures:
            self.fail_counts[key] = remaining_failures - 1
            raise RuntimeError("private-mounted-trash-failure")
        media_type = kwargs.get("media_type")
        rows = [
            item
            for item in self.items
            if query.casefold() in str(item["title"]).casefold()
            and (media_type is None or item["media_type"] == media_type)
        ]
        limit = int(kwargs["limit"])
        return {
            "items": [dict(item) for item in rows[offset : offset + limit]],
            "total": len(rows),
            "limit": limit,
            "offset": offset,
            "types": sorted(
                {
                    str(item["media_type"])
                    for item in self.items
                    if item["media_type"] is not None
                }
            ),
        }

    def install(self, service: object) -> None:
        async def _list_library_media_trash(_service: object, **kwargs: object):
            return await self.list_library_media_trash(**kwargs)

        service.list_library_media_trash = types.MethodType(  # type: ignore[attr-defined]
            _list_library_media_trash,
            service,
        )


# ---------------------------------------------------------------------------
# Canvas widget pilot tests
# ---------------------------------------------------------------------------


def _trash_state(**kwargs):
    records = kwargs.pop(
        "records",
        [
            {
                "id": "11",
                "title": "First trashed",
                "type": "pdf",
                "trash_date": "2026-08-11T11:00:00+00:00",
            },
            {"id": "12", "title": "Second trashed", "type": "video"},
        ],
    )
    kwargs.setdefault("total", len(records or ()))
    return build_library_media_trash_state(records, **kwargs)


class _TrashCanvasApp(ConsolidatedCSSApp):
    def __init__(self, state, **presentation):
        super().__init__()
        self._state = state
        self._presentation = presentation

    def compose(self):
        yield LibraryMediaTrashCanvas(
            canvas=self._state,
            **self._presentation,
            id="library-media-trash-canvas",
        )


@pytest.mark.asyncio
async def test_trash_canvas_renders_heading_rows_and_enabled_restore():
    app = _TrashCanvasApp(_trash_state())
    async with app.run_test() as pilot:
        await pilot.pause()
        title = app.query_one("#library-media-trash-title", Static)
        assert "Trash (2)" in str(title.render())
        assert app.query_one("#library-media-trash-back", Button)
        rows = list(app.query(".library-media-trash-row"))
        assert len(rows) == 2
        # Selected-row grammar: leading "▸ " on the auto-selected first row.
        assert str(rows[0].label).startswith("▸ First trashed")
        assert str(rows[1].label).startswith("  Second trashed")
        restore = app.query_one("#library-media-trash-restore", Button)
        assert restore.disabled is False
        assert str(restore.label) == "Restore"
        assert restore.tooltip == LIBRARY_MEDIA_TRASH_RESTORE_TOOLTIP


@pytest.mark.asyncio
async def test_media_trash_permanent_confirmation_disambiguates_full_long_title():
    """Duplicate truncated rows still expose one complete captured identity."""
    shared_prefix = "A" * 140
    full_title = f"{shared_prefix} selected tail"
    other_title = f"{shared_prefix} other tail"
    records = [
        {
            "id": "local:media:11",
            "title": full_title,
            "type": "video",
            "trash_date": "2026-08-11T11:00:00+00:00",
        },
        {
            "id": "local:media:12",
            "title": other_title,
            "type": "video",
            "trash_date": "2026-08-12T12:00:00+00:00",
        },
    ]
    state = _trash_state(records=records, selected_id="local:media:11")
    target = MediaTrashMutationTarget(
        stable_id="local:media:11",
        backing_media_id=11,
        title=full_title,
        media_type="video",
        trash_date="2026-08-11T11:00:00+00:00",
        page_index=0,
    )
    app = _TrashCanvasApp(state, confirmation_target=target)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        row_labels = [str(row.label) for row in app.query(".library-media-trash-row")]
        assert full_title not in row_labels[0]
        assert other_title not in row_labels[1]
        assert row_labels[0].splitlines()[0].endswith("...")
        assert row_labels[1].splitlines()[0].endswith("...")

        details = app.query_one(
            "#library-media-trash-delete-confirm-details", VerticalScroll
        )
        assert 0 < details.region.height <= 2
        assert (
            app.query_one(
                "#library-media-trash-delete-confirm-title", Static
            ).renderable
            == full_title
        )
        assert (
            app.query_one("#library-media-trash-delete-confirm-type", Static).renderable
            == "video"
        )
        assert (
            app.query_one("#library-media-trash-delete-confirm-time", Static).renderable
            == "2026-08-11T11:00:00+00:00"
        )
        assert not app.query("#library-media-trash-restore")
        assert not app.query("#library-media-trash-delete")
        assert app.query_one("#library-media-trash-delete-cancel", Button)
        assert app.query_one("#library-media-trash-delete-confirm", Button)


@pytest.mark.asyncio
async def test_media_trash_permanent_confirmation_names_missing_identity_fields():
    state = _trash_state(
        records=[
            {
                "id": "local:media:11",
                "title": "Untyped",
                "type": None,
                "trash_date": None,
            }
        ],
        selected_id="local:media:11",
    )
    target = MediaTrashMutationTarget(
        stable_id="local:media:11",
        backing_media_id=11,
        title="Untyped",
        media_type=None,
        trash_date=None,
        page_index=0,
    )
    app = _TrashCanvasApp(state, confirmation_target=target)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert (
            app.query_one("#library-media-trash-delete-confirm-type", Static).renderable
            == "Unknown type"
        )
        assert (
            app.query_one("#library-media-trash-delete-confirm-time", Static).renderable
            == "Unknown deletion time"
        )


@pytest.mark.asyncio
async def test_trash_canvas_empty_state_disables_restore_with_reason():
    app = _TrashCanvasApp(_trash_state(records=[], total=0))
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.query_one("#library-media-trash-status", Static)
        assert LIBRARY_MEDIA_TRASH_EMPTY_COPY in str(status.render())
        restore = app.query_one("#library-media-trash-restore", Button)
        assert restore.disabled is True
        # F-018 + the non-colour disabled marker.
        assert str(restore.label) == "○ Restore"
        assert restore.tooltip == LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_EMPTY_TOOLTIP


@pytest.mark.asyncio
async def test_trash_canvas_loading_state_never_claims_empty():
    app = _TrashCanvasApp(_trash_state(records=None, total=0))
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.query_one("#library-media-trash-status", Static)
        rendered = str(status.render())
        assert "Loading Trash…" in rendered
        assert LIBRARY_MEDIA_TRASH_EMPTY_COPY not in rendered
        restore = app.query_one("#library-media-trash-restore", Button)
        assert restore.disabled is True
        assert restore.tooltip == LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_LOADING_TOOLTIP


@pytest.mark.asyncio
async def test_trash_canvas_truncation_status_and_notice_render():
    records = [{"id": str(i), "title": f"T{i}", "type": "pdf"} for i in range(3)]
    app = _TrashCanvasApp(
        _trash_state(records=records, total=9, notice="Restored 'A doc'.")
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.query_one("#library-media-trash-status", Static)
        assert "showing 3 of 9" in str(status.render())
        notice = app.query_one("#library-media-trash-notice", Static)
        assert "Restored 'A doc'." in str(notice.render())
        title = app.query_one("#library-media-trash-title", Static)
        assert "Trash (9)" in str(title.render())


@pytest.mark.asyncio
async def test_trash_canvas_error_state_shows_error_not_empty_copy():
    app = _TrashCanvasApp(
        _trash_state(records=[], total=0, error="Could not load Trash.")
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.query_one("#library-media-trash-status", Static)
        rendered = str(status.render())
        assert "Could not load Trash." in rendered
        assert LIBRARY_MEDIA_TRASH_EMPTY_COPY not in rendered


@pytest.mark.asyncio
async def test_trash_canvas_error_state_restore_tooltip_names_the_failure():
    """PR-1505 review: with a failed fetch (``canvas.error`` set) the
    disabled Restore's tooltip must state the TRUE blocker -- the load
    failure -- never the empty-trash copy (F-018 accuracy: an unreadable
    Trash is not an empty Trash)."""
    app = _TrashCanvasApp(
        _trash_state(records=[], total=0, error="Could not load Trash.")
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        restore = app.query_one("#library-media-trash-restore", Button)
        assert restore.disabled is True
        assert str(restore.label) == "○ Restore"
        assert restore.tooltip != LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_EMPTY_TOOLTIP
        assert restore.tooltip == LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_ERROR_TOOLTIP


def _fresh_trash_pager(*, page: int = 1, total: int = 45, rows: int = 20):
    return build_library_pager_display(
        applied_page=page,
        requested_page=page,
        page_size=20,
        row_count=rows,
        total=total,
        freshness="fresh",
    )


@pytest.mark.asyncio
async def test_media_trash_render_pager_filter_and_disabled_action_semantics():
    """The bounded canvas names only applied fresh authority and true blockers."""
    records = [
        {
            "id": str(index),
            "title": f"Trashed item {index:02d}",
            "type": "audio",
            "trash_date": "2026-08-10T12:00:00+00:00",
        }
        for index in range(1, 21)
    ]
    app = _TrashCanvasApp(
        _trash_state(records=records, total=45, selected_id=""),
        pager=_fresh_trash_pager(),
        types=("audio", "video"),
        query_draft="unapplied draft",
        applied_scope_label="",
        applied_type=None,
        type_choices_visible=False,
        action_disabled_reason="Select a Trash item first.",
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        title = app.query_one("#library-media-trash-title", Static)
        assert str(title.renderable) == "Local Trash · 45 items"
        assert app.query_one("#library-media-trash-search", Input).value == (
            "unapplied draft"
        )
        assert (
            str(app.query_one("#library-media-trash-range", Static).renderable)
            == "1-20 of 45"
        )
        assert (
            str(app.query_one("#library-media-trash-page", Static).renderable)
            == "Page 1 of 3"
        )
        previous = app.query_one("#library-media-trash-previous", Button)
        next_button = app.query_one("#library-media-trash-next", Button)
        assert previous.disabled is True
        assert previous.tooltip == "Already on the first page."
        assert next_button.disabled is False
        assert list(app.query(".library-media-trash-row"))[0].region.height == 2
        assert len(list(app.query(".library-media-trash-row"))) == 20
        trash_list = app.query_one("#library-media-trash-list")
        assert str(trash_list.styles.overflow_x) == "hidden"

        for selector in (
            "#library-media-trash-restore",
            "#library-media-trash-delete",
        ):
            action = app.query_one(selector, Button)
            assert action.disabled is True
            assert str(action.label).startswith("○ ")
            assert action.tooltip == "Select a Trash item first."


@pytest.mark.asyncio
async def test_media_trash_type_filter_uses_one_bounded_complete_facet_chooser():
    """Sixty complete-source facets stay in one scrollable keyboard chooser."""
    types = tuple(f"type-{index:02d}" for index in range(60))
    app = _TrashCanvasApp(
        _trash_state(),
        pager=build_library_pager_display(
            applied_page=1,
            requested_page=1,
            page_size=20,
            row_count=2,
            total=2,
            freshness="fresh",
        ),
        types=types,
        query_draft="",
        applied_scope_label="Type: type-59",
        applied_type="type-59",
        type_choices_visible=True,
        action_disabled_reason="",
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert not app.query("#library-media-trash-type-filter")
        chooser = app.query_one("#library-media-trash-type-choices", OptionList)
        assert chooser.option_count == 61
        assert not app.query("#library-media-trash-type-choices Button")
        assert str(chooser.get_option_at_index(0).prompt) == "All types"
        assert str(chooser.get_option_at_index(60).prompt).startswith("✓ type-59")
        chooser.focus()
        await pilot.press("end")
        await pilot.pause()
        assert chooser.highlighted == 60
        assert "type-59" in _compositor_text(app.export_screenshot(simplify=True))


# ---------------------------------------------------------------------------
# Fold reachability (PR-1505 review): with a full page of trashed items the
# list must scroll -- the repo's L3a clipping lesson (LibraryExportCanvas's
# docstring: a plain Vertical canvas clips content past the fold) applies
# here too, and the whole-branch live smoke only ever had ONE trash row, so
# this was never exercised. Measured in the REAL LibraryHarness (real app
# stylesheet, real #library-canvas host) -- a canvas mounted alone in a bare
# App is not measured against the tier that wins live (task-14900's lesson).
# ---------------------------------------------------------------------------


def _forty_trash_items():
    return [
        {
            "id": f"local:media:{index}",
            "backing_media_id": index,
            "title": f"Trashed item {index:02d}",
            "media_type": "document",
            "trash_date": "2026-08-10T12:00:00+00:00",
        }
        for index in range(1, 21)
    ]


def _compositor_text(svg: str) -> str:
    """Rejoin an exported-screenshot SVG's `<text>` nodes into plain text.

    The established compositor-honest idiom (see
    ``test_console_fleet_discoverability._compositor_text``): scroll-clipped
    or off-screen content never becomes a `<text>` node at all, unlike a
    widget's ``.renderable``/``region``, which exist regardless of paint.
    """
    import re
    from html import unescape

    joined = "".join(re.findall(r"<text[^>]*>([^<]*)</text>", svg))
    return unescape(joined).replace("\xa0", " ")


async def _open_media_trash_with_items(host, pilot, trash_items):
    """Drive the real screen into the Trash view over a seeded trash page."""
    from Tests.UI.test_library_shell import (
        _active_library_screen,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)

    feed = _MountedTrashFeed(list(trash_items))
    feed.install(host.app_instance.media_reading_scope_service)

    await _wait_for_selector(screen, pilot, "#library-row-browse-media")
    screen.query_one("#library-row-browse-media").press()
    await _wait_for_selector(screen, pilot, "#library-media-trash-open")
    screen.query_one("#library-media-trash-open").press()
    await _wait_for_selector(
        screen, pilot, f"#library-media-trash-row-{len(trash_items) - 1}"
    )
    await pilot.pause()
    return screen


async def _assert_trash_rows_and_restore_reachable(host, pilot):
    screen = await _open_media_trash_with_items(host, pilot, _forty_trash_items())
    trash_list = screen.query_one("#library-media-trash-list")
    restore = screen.query_one("#library-media-trash-restore", Button)
    last_row = screen.query_one("#library-media-trash-row-19", Button)
    screen_height = host.size.height

    # The Restore toolbar stays inside the terminal: 20 two-line rows
    # must not push it below the fold.
    assert restore.region.height > 0
    assert restore.region.y + restore.region.height <= screen_height

    # The list owns the vertical scroll -- its content overflows the pane
    # and the overflow is reachable rather than clipped.
    assert trash_list.max_scroll_y > 0

    # Scroll reachability: after scroll_end the LAST row is actually
    # painted (compositor-honest -- clipped content never paints).
    trash_list.scroll_end(animate=False)
    await pilot.pause()
    await pilot.pause()
    painted = _compositor_text(host.export_screenshot(simplify=True))
    assert "Trashed item 20" in painted

    # Keyboard reachability: from the top, focusing the last row
    # auto-scrolls it into view (Screen.set_focus -> scroll_visible).
    trash_list.scroll_home(animate=False)
    await pilot.pause()
    painted_top = _compositor_text(host.export_screenshot(simplify=True))
    assert "Trashed item 20" not in painted_top
    last_row.focus()
    await pilot.pause()
    await pilot.pause()
    painted = _compositor_text(host.export_screenshot(simplify=True))
    assert "Trashed item 20" in painted


@pytest.mark.asyncio
async def test_trash_page_past_fold_reachable_wide_split_layout():
    """Wide (>= the compact breakpoint) on a 24-row terminal: the full page
    trash page scrolls; the last row and Restore both stay reachable."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _seed_conversations,
        _two_conversations,
        _two_media_items,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)
    async with host.run_test(size=(170, 24)) as pilot:
        await _assert_trash_rows_and_restore_reachable(host, pilot)


@pytest.mark.asyncio
async def test_trash_page_past_fold_reachable_stacked_layout():
    """Below the breakpoint (the ``library-notes-compact`` stacked regime),
    same 24-row terminal: identical reachability contract."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _seed_conversations,
        _two_conversations,
        _two_media_items,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)
    async with host.run_test(size=(100, 24)) as pilot:
        await _assert_trash_rows_and_restore_reachable(host, pilot)


# ---------------------------------------------------------------------------
# Exact-page screen lifecycle (task-18918 Task 4). The canvas gains its
# visible filter/pager controls in Task 5; these tests pin the mounted Screen
# authority and real event handlers that those controls consume.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_media_trash_entry_requests_one_independent_initial_page():
    """Entry ignores normal Media filters and exposes no fabricated result."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items(), gate_first=True)
    feed.install(app.media_reading_scope_service)
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            media_row = await _wait_for_selector(
                screen, pilot, "#library-row-browse-media"
            )
            media_row.focus()
            await pilot.press("enter")
            await _wait_for_selector(screen, pilot, "#library-media-trash-open")
            # These are genuine normal-Media scope values. Trash must never
            # inherit them when its independent page owner begins.
            screen._request_library_media_type("audio", focus_identity=None)
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_media_browse_controller.applied_scope
                    == MediaBrowseScope(media_type="audio")
                ),
                message="Normal Media type scope never applied.",
            )
            screen._request_library_media_filter("Interview")
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_media_browse_controller.applied_scope
                    == MediaBrowseScope(query="Interview", media_type="audio")
                ),
                message="Normal Media query scope never applied.",
            )

            opener = screen.query_one("#library-media-trash-open", Button)
            opener.focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                feed.entered.is_set,
                message="Trash entry request never reached the service.",
            )

            controller = screen._library_media_trash_browse_controller
            assert len(feed.calls) == 1
            assert feed.calls[0] == {
                "mode": "local",
                "query": "",
                "media_type": None,
                "limit": 20,
                "offset": 0,
            }
            assert controller.state.requested_scope == MediaTrashScope()
            assert controller.state.applied_result is None
            assert controller.state.loading is True
            assert (
                screen.query_one("#library-media-trash-status", Static).renderable
                == "Loading Trash…"
            )

            feed.release.set()
            await _wait_for_condition(
                pilot,
                lambda: controller.state.applied_result is not None,
                message="Trash entry page never applied.",
            )
            assert len(feed.calls) == 1
            assert controller.state.applied_result.scope == MediaTrashScope()
    finally:
        feed.release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trash_items", "expected_focus"),
    (
        (_canonical_trash_items(2), "library-media-trash-row-0"),
        ([], "library-media-trash-back"),
    ),
)
async def test_media_trash_focus_initial_success_and_empty_fallback(
    trash_items, expected_focus
):
    """The opening Enter lands on one mounted success/empty recovery target."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(trash_items)
    feed.install(app.media_reading_scope_service)
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        media_row = screen.query_one("#library-row-browse-media", Button)
        media_row.focus()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        await pilot.pause()
        opener = screen.query_one("#library-media-trash-open", Button)
        opener.focus()
        await pilot.press("enter")

        controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.state.applied_result is not None,
            message="Initial Trash success never settled.",
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                getattr(screen.focused, "id", None) == expected_focus
                and screen.focused is screen.query_one(f"#{expected_focus}", Button)
            ),
            message="Initial Trash focus did not settle on its mounted fallback.",
        )
        assert screen._library_media_view == "trash"
        assert len(feed.calls) == 1


@pytest.mark.asyncio
async def test_media_trash_initial_entry_failure_sets_retry_focus_intent():
    """An initial read failure mounts one Retry and gives it real DOM focus."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items(), fail_counts={("", 0): 1})
    feed.install(app.media_reading_scope_service)
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        await pilot.pause()
        opener = screen.query_one("#library-media-trash-open", Button)
        opener.focus()
        await pilot.press("enter")
        controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.state.error_copy == "Could not load Trash.",
            message="Initial Trash failure never settled.",
        )

        assert len(feed.calls) == 1
        assert controller.state.applied_result is None
        assert controller.state.failed_scope == MediaTrashScope()
        assert (
            screen._library_media_trash_focus_identity == "#library-media-trash-retry"
        )
        status = await _wait_for_selector(screen, pilot, "#library-media-trash-status")
        assert status.renderable == "Could not load Trash · Retry"
        await _wait_for_selector(screen, pilot, "#library-media-trash-retry")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen.focused is screen.query_one("#library-media-trash-retry", Button)
            ),
            message="Initial Trash failure did not focus mounted Retry.",
        )
        assert len(list(screen.query("#library-media-trash-retry"))) == 1
        retry = screen.query_one("#library-media-trash-retry", Button)
        assert screen.focused is retry
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.state.applied_result is not None
                and controller.state.applied_result.scope == MediaTrashScope()
            ),
            message="Keyboard Retry did not apply the initial page.",
        )
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None) == "library-media-trash-row-0",
            message="Retry success did not settle focus on the mounted first row.",
        )


@pytest.mark.asyncio
async def test_media_trash_focus_falls_back_when_next_becomes_disabled():
    """A real Next Enter lands on Previous when the last page removes Next."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items(21))
    feed.install(app.media_reading_scope_service)
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        await pilot.pause()
        opener = screen.query_one("#library-media-trash-open", Button)
        opener.focus()
        await pilot.press("enter")
        controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.state.applied_result is not None,
            message="Initial Trash page never applied.",
        )

        await _wait_for_selector(screen, pilot, "#library-media-trash-next")
        await pilot.pause()
        next_button = screen.query_one("#library-media-trash-next", Button)
        assert next_button.disabled is False
        next_button.focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.state.applied_result is not None
                and controller.state.applied_result.scope == MediaTrashScope(page=2)
            ),
            message="Last Trash page never became authoritative.",
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                screen.focused
                is screen.query_one("#library-media-trash-previous", Button)
            ),
            message="Disabled Next did not fall back to mounted Previous.",
        )
        assert screen.query_one("#library-media-trash-next", Button).disabled is True


@pytest.mark.asyncio
async def test_media_trash_filter_retry_page_and_type_use_applied_scope():
    """Real submitted/chooser events preserve requested/applied separation."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(
        _canonical_trash_items(),
        fail_counts={("failed", 0): 2},
    )
    feed.install(app.media_reading_scope_service)
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        screen.query_one("#library-media-trash-open", Button).press()
        controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.state.applied_result is not None,
            message="Initial Trash page never applied.",
        )
        await _wait_for_selector(screen, pilot, "#library-media-trash-row-1")

        screen.query_one("#library-media-trash-row-1", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: controller.state.selected_id == "local:media:2",
            message="Trash row selection never settled.",
        )

        calls_before_bound = len(feed.calls)
        long_input = Input(id="library-media-trash-search")
        long_event = Input.Submitted(long_input, "x" * 201)
        screen.handle_library_media_trash_search_submitted(long_event)
        await pilot.pause()
        assert len(feed.calls) == calls_before_bound
        assert controller.state.applied_result.scope == MediaTrashScope()
        assert controller.state.failed_scope is None
        assert (
            screen._library_media_trash_focus_identity == "#library-media-trash-search"
        )
        assert (
            screen.query_one("#library-media-trash-status", Static).renderable
            == "Search is limited to 200 characters."
        )

        failed_input = Input(id="library-media-trash-search")
        screen.handle_library_media_trash_search_submitted(
            Input.Submitted(failed_input, "failed")
        )
        await _wait_for_condition(
            pilot,
            lambda: bool(controller.state.error_copy),
            message="Failed Trash filter never exposed Retry state.",
        )
        assert controller.state.selected_id == ""
        assert controller.state.applied_result.scope == MediaTrashScope()
        assert controller.state.requested_scope == MediaTrashScope(query="failed")
        assert (
            screen.query_one("#library-media-trash-status", Static).renderable
            == "Filter not applied — showing All Trash · Retry"
        )

        # Validation remains a Search-owned concern even when a prior
        # failed browse target is still retained for a later Retry.
        calls_before_failed_bound = len(feed.calls)
        screen.handle_library_media_trash_search_submitted(
            Input.Submitted(long_input, "y" * 201)
        )
        await pilot.pause()
        assert len(feed.calls) == calls_before_failed_bound
        assert controller.state.failed_scope == MediaTrashScope(query="failed")
        assert (
            screen._library_media_trash_focus_identity == "#library-media-trash-search"
        )
        assert (
            screen.query_one("#library-media-trash-status", Static).renderable
            == "Search is limited to 200 characters."
        )

        retry = Button("Retry", id="library-media-trash-retry")
        screen.handle_library_media_trash_retry(Button.Pressed(retry))
        await _wait_for_condition(
            pilot,
            lambda: len(feed.calls) == calls_before_bound + 2,
            message="Trash Retry did not repeat the failed target.",
        )
        assert feed.calls[-1]["query"] == "failed"
        assert controller.state.applied_result.scope == MediaTrashScope()

        # After that second failure, Next derives from the visible applied
        # All-Trash page, never the failed query draft.
        next_button = Button("Next", id="library-media-trash-next")
        screen.handle_library_media_trash_next(Button.Pressed(next_button))
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.state.applied_result is not None
                and controller.state.applied_result.scope == MediaTrashScope(page=2)
            ),
            message="Trash Next did not use the visible applied scope.",
        )
        assert feed.calls[-1]["query"] == ""
        assert feed.calls[-1]["offset"] == 20

        # The bounded chooser event changes only the visible applied scope
        # and clears the current-page selection before dispatch.
        controller.select("local:media:22")
        option_list = OptionList(id="library-media-trash-type-choices")
        option = Option("audio")
        option.choice_value = "audio"
        screen.handle_library_media_trash_type_choice(
            OptionList.OptionSelected(option_list, option, 0)
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.state.applied_result is not None
                and controller.state.applied_result.scope
                == MediaTrashScope(media_type="audio")
            ),
            message="Trash type choice never applied page 1.",
        )
        assert controller.state.selected_id == ""


@pytest.mark.asyncio
async def test_media_trash_filter_type_focus_and_escape_use_real_keyboard_inputs():
    """Search/type intents reacquire current controls; chooser Escape is inert."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items())
    feed.install(app.media_reading_scope_service)
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        await pilot.pause()
        opener = screen.query_one("#library-media-trash-open", Button)
        opener.focus()
        await pilot.press("enter")
        controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.state.applied_result is not None,
            message="Initial Trash page never applied.",
        )

        await _wait_for_selector(screen, pilot, "#library-media-trash-search")
        await pilot.pause()
        search = screen.query_one("#library-media-trash-search", Input)
        search.focus()
        await pilot.press("t", "r", "a", "s", "h", "enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.state.applied_result is not None
                and controller.state.applied_result.scope.query == "trash"
                and not controller.state.loading
            ),
            message="Submitted Trash query never became applied authority.",
        )
        assert getattr(screen.focused, "id", None) == "library-media-trash-search"
        assert (
            screen.query_one("#library-media-trash-scope", Static).renderable
            == "Query: trash"
        )

        calls_before_open = len(feed.calls)
        await pilot.press("tab")
        assert getattr(screen.focused, "id", None) == "library-media-trash-type-filter"
        await pilot.press("enter")
        chooser = await _wait_for_selector(
            screen, pilot, "#library-media-trash-type-choices"
        )
        assert isinstance(chooser, OptionList)
        assert len(feed.calls) == calls_before_open
        assert chooser.highlighted == 0
        await _wait_for_condition(
            pilot,
            lambda: (
                screen.focused
                is screen.query_one("#library-media-trash-type-choices", OptionList)
            ),
            message="Type chooser did not receive mounted focus after paint.",
        )
        chooser = screen.query_one("#library-media-trash-type-choices", OptionList)
        await pilot.press("down", "enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.state.applied_result is not None
                and controller.state.applied_result.scope.media_type == "audio"
            ),
            message="Keyboard type choice never became applied authority.",
        )
        assert getattr(screen.focused, "id", None) == (
            "library-media-trash-type-filter"
        )

        applied_before_escape = controller.state.applied_result.scope
        requested_before_escape = controller.state.requested_scope
        type_filter = screen.query_one("#library-media-trash-type-filter", Button)
        assert screen.focused is type_filter
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-media-trash-type-choices")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen.focused
                is screen.query_one("#library-media-trash-type-choices", OptionList)
            ),
            message="Reopened type chooser did not receive mounted focus.",
        )
        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen.focused
                is screen.query_one("#library-media-trash-type-filter", Button)
            ),
            message="Escape did not restore mounted type-filter focus.",
        )
        assert controller.state.applied_result.scope == applied_before_escape
        assert controller.state.requested_scope == requested_before_escape
        assert getattr(screen.focused, "id", None) == (
            "library-media-trash-type-filter"
        )


@pytest.mark.asyncio
async def test_media_trash_type_filter_uses_complete_production_facets():
    """The mounted chooser includes facets absent from the visible page."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    items = [
        {
            "id": f"local:media:{index}",
            "backing_media_id": index,
            "title": f"Facet source {index:02d}",
            "media_type": f"type-{(index - 1) % 60:02d}",
            "trash_date": "2026-08-10T12:00:00+00:00",
        }
        for index in range(1, 81)
    ]
    feed = _MountedTrashFeed(items)
    feed.install(app.media_reading_scope_service)
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        await pilot.pause()
        opener = screen.query_one("#library-media-trash-open", Button)
        opener.focus()
        await pilot.press("enter")
        controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.state.applied_result is not None,
            message=lambda: (
                "Complete-facet Trash page never applied: "
                f"calls={feed.calls!r}, state={controller.state!r}."
            ),
        )
        assert len(list(screen.query(".library-media-trash-row"))) == 20
        type_filter = await _wait_for_selector(
            screen, pilot, "#library-media-trash-type-filter"
        )
        type_filter.focus()
        await pilot.press("enter")
        chooser = await _wait_for_selector(
            screen, pilot, "#library-media-trash-type-choices"
        )
        assert isinstance(chooser, OptionList)
        assert chooser.option_count == 61
        assert str(chooser.get_option_at_index(60).prompt) == "type-59"
        chooser = screen.query_one("#library-media-trash-type-choices", OptionList)
        chooser.focus()
        await pilot.pause()
        assert screen.focused is chooser
        await pilot.press("end")
        assert chooser.highlighted == 60


@pytest.mark.asyncio
async def test_media_trash_focus_completion_does_not_steal_newer_tab_intent():
    """A gated page completion yields to a later real keyboard focus choice."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items(), gate_first=True)
    feed.install(app.media_reading_scope_service)
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-media", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-trash-open")
            screen.query_one("#library-media-trash-open", Button).focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                feed.entered.is_set,
                message="Initial Trash request never reached its gate.",
            )
            await _wait_for_selector(screen, pilot, "#library-media-trash-search")
            screen.query_one("#library-media-trash-search", Input).focus()
            await pilot.press("tab")
            newer_focus = screen.focused
            assert getattr(newer_focus, "id", None) == (
                "library-media-trash-type-filter"
            )

            feed.release.set()
            controller = screen._library_media_trash_browse_controller
            await _wait_for_condition(
                pilot,
                lambda: controller.state.applied_result is not None,
                message="Gated Trash page never applied.",
            )
            await pilot.pause()
            assert getattr(screen.focused, "id", None) == (
                "library-media-trash-type-filter"
            )
            assert screen.focused is screen.query_one(
                "#library-media-trash-type-filter", Button
            )
            assert screen.focused is not newer_focus
    finally:
        feed.release.set()


@pytest.mark.asyncio
async def test_media_trash_render_retry_matches_recoverable_browse_authority():
    """Retry copy, control, and focus exist only for a retryable browse."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items())
    feed.install(app.media_reading_scope_service)
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        screen.query_one("#library-media-trash-open", Button).press()
        controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.state.applied_result is not None,
            message="Initial Trash page never applied.",
        )
        ordinary = controller.state
        first = ordinary.retained_items[0]
        target = MediaTrashMutationTarget(
            stable_id=str(first["id"]),
            backing_media_id=int(first["backing_media_id"]),
            title=str(first["title"]),
            media_type=str(first["media_type"]),
            trash_date=str(first["trash_date"]),
            page_index=0,
        )

        mutation_error = fail_media_trash_mutation(
            begin_media_trash_mutation(ordinary),
            target,
            copy="Could not restore this media item.",
        )
        controller.state = mutation_error
        screen._library_media_trash_focus_identity = "#library-media-trash-restore"
        screen._sync_library_media_trash_state(None)
        await pilot.pause()
        assert (
            screen.query_one("#library-media-trash-status", Static).renderable
            == "Could not restore this media item."
        )
        assert not screen.query("#library-media-trash-retry")
        assert (
            screen._library_media_trash_focus_identity == "#library-media-trash-restore"
        )

        stale_loading = dataclasses.replace(
            ordinary,
            freshness="stale",
            loading=True,
            stale_copy="List may be out of date.",
            selected_id="",
            request_origin="mutation",
            failed_scope=None,
            failed_origin=None,
        )
        controller.state = stale_loading
        screen._library_media_trash_focus_identity = "#library-media-trash-row-0"
        screen._sync_library_media_trash_state(None)
        await pilot.pause()
        loading_status = str(
            screen.query_one("#library-media-trash-status", Static).renderable
        )
        assert "Retry" not in loading_status
        assert not screen.query("#library-media-trash-retry")
        assert (
            screen._library_media_trash_focus_identity != "#library-media-trash-retry"
        )

        stale_failure = fail_media_trash_request(
            stale_loading,
            stale_loading.requested_scope,
            copy="List may be out of date.",
        )
        controller.state = stale_failure
        screen._sync_library_media_trash_state(None)
        await pilot.pause()
        assert (
            screen.query_one("#library-media-trash-status", Static).renderable
            == "List may be out of date · Retry"
        )
        assert len(list(screen.query("#library-media-trash-retry"))) == 1
        assert (
            screen._library_media_trash_focus_identity == "#library-media-trash-retry"
        )


@pytest.mark.asyncio
async def test_media_trash_filter_draft_survives_background_recompose():
    """A typed, unsubmitted draft survives an unrelated request completion."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items(), gate_first=True)
    feed.install(app.media_reading_scope_service)
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-media", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-trash-open")
            opener = screen.query_one("#library-media-trash-open", Button)
            opener.focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                feed.entered.is_set,
                message="Initial Trash request never reached its gate.",
            )

            search_before = screen.query_one("#library-media-trash-search", Input)
            search_before.focus()
            await pilot.press("d", "r", "a", "f", "t")
            await pilot.pause()
            assert search_before.value == "draft"

            feed.release.set()
            controller = screen._library_media_trash_browse_controller
            await _wait_for_condition(
                pilot,
                lambda: controller.state.applied_result is not None,
                message="Background Trash completion never applied.",
            )
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen.query_one("#library-media-trash-search", Input)
                    is not search_before
                ),
                message="Trash completion did not mount a current Search input.",
            )
            search_after = screen.query_one("#library-media-trash-search", Input)
            assert search_after.value == "draft"
            assert screen._library_media_trash_query_draft == "draft"
            assert controller.state.applied_result.scope == MediaTrashScope()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen.focused
                    is screen.query_one("#library-media-trash-search", Input)
                ),
                message="Draft-preserving completion did not restore current Search.",
            )
    finally:
        feed.release.set()


@pytest.mark.asyncio
async def test_media_trash_compact_status_folds_after_two_readable_rows():
    """An overflowing compact status paints two rows plus a clear fold."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryProductionCSSHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items())
    feed.install(app.media_reading_scope_service)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_layout.reader_width > 0,
            message="Compact Media layout never settled.",
        )
        if not screen._library_media_reader_layout.items_open:
            grip = screen.query_one("#library-media-items-grip", Button)
            grip.focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: screen._library_media_reader_layout.items_open,
                message="Compact Items pane never opened.",
            )
        opener = screen.query_one("#library-media-trash-open", Button)
        opener.focus()
        await pilot.press("enter")
        controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.state.applied_result is not None,
            message="Initial Trash page never applied.",
        )
        assert "▼ more status" not in _compositor_text(
            host.export_screenshot(simplify=True)
        )

        long_copy = (
            "Could not load this Trash page. The local source stayed available "
            "but its full recovery detail does not fit in the compact status area."
        )
        controller.state = fail_media_trash_request(
            begin_media_trash_request(
                MediaTrashBrowseState(), MediaTrashScope(), origin="entry"
            ),
            MediaTrashScope(),
            copy=long_copy,
        )
        screen._sync_library_media_trash_state(None)
        await pilot.pause()
        await pilot.pause()

        status = screen.query_one("#library-media-trash-status", Static)
        fold = screen.query_one("#library-media-trash-status-fold", Static)
        trash_list = screen.query_one("#library-media-trash-list")
        pager = screen.query_one("#library-media-trash-pager")
        actions = screen.query_one("#library-media-trash-actions")
        items = screen.query_one("#library-canvas")
        assert status.region.height == 2
        assert fold.region.height == 1
        assert fold.region.y == status.region.bottom
        assert fold.tooltip == f"{long_copy.rstrip('.')} · Retry"
        assert trash_list.region.height >= 4
        assert items.region.contains_region(pager.region)
        assert items.region.contains_region(actions.region)
        painted = _compositor_text(host.export_screenshot(simplify=True))
        assert "▼ more status" in painted


@pytest.mark.asyncio
async def test_media_trash_focus_recovers_after_suppressed_canvas_sync(monkeypatch):
    """A suppressed sync releases focus authority to a later real Tab."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items())
    feed.install(app.media_reading_scope_service)
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        screen.query_one("#library-media-trash-open", Button).press()
        controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.state.applied_result is not None,
            message="Initial Trash page never applied.",
        )
        await _wait_for_selector(screen, pilot, "#library-media-trash-search")
        search = screen.query_one("#library-media-trash-search", Input)
        search.focus()
        await pilot.pause()
        screen._library_notes_programmatic_focus_target = search
        older_generation = screen._library_notes_focus_intent_generation

        with monkeypatch.context() as patch:
            patch.setattr(
                "tldw_chatbook.UI.Screens.library_screen._sync_library_canvas",
                lambda *args, **kwargs: False,
            )
            screen._sync_library_media_trash_state("#library-media-trash-search")

        assert screen._library_notes_restoring_focus is False
        assert screen._library_notes_programmatic_focus_target is None
        await pilot.press("tab")
        await _wait_for_condition(
            pilot,
            lambda: (
                getattr(screen.focused, "id", None) == "library-media-trash-type-filter"
            ),
            message="Real Tab did not leave Search after a suppressed sync.",
        )
        assert screen._library_notes_focus_intent_generation > older_generation
        screen._focus_library_media_trash_after_paint(
            "#library-media-trash-search", older_generation
        )
        await pilot.pause()
        assert getattr(screen.focused, "id", None) == "library-media-trash-type-filter"


@pytest.mark.asyncio
async def test_media_trash_disabled_browse_controls_name_mutation_interlock():
    """Every conflicting browse control is visibly disabled during mutation."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items())
    feed.install(app.media_reading_scope_service)
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        screen.query_one("#library-media-trash-open", Button).press()
        controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.state.applied_result is not None,
            message="Initial Trash page never applied.",
        )
        failed_scope = MediaTrashScope(query="failed")
        controller.state = fail_media_trash_request(
            begin_media_trash_request(controller.state, failed_scope, origin="search"),
            failed_scope,
            copy="Filter not applied — showing All Trash.",
        )
        screen._library_media_bulk_delete_in_flight = True
        screen._sync_library_media_trash_state(None)
        await pilot.pause()

        reason = "Trash is refreshing."
        search = screen.query_one("#library-media-trash-search", Input)
        assert search.disabled is True
        assert search.tooltip == reason
        expected_labels = {
            "#library-media-trash-type-filter": "○ Type: All",
            "#library-media-trash-previous": "○ Previous",
            "#library-media-trash-next": "○ Next",
            "#library-media-trash-retry": "○ Retry",
        }
        for selector, label in expected_labels.items():
            control = screen.query_one(selector, Button)
            assert control.disabled is True
            assert str(control.label) == label
            assert control.tooltip == reason
        screen._library_media_bulk_delete_in_flight = False


@pytest.mark.asyncio
async def test_media_trash_confirmation_focus_cancel_escape_and_explicit_commit():
    """Cancel owns initial focus and the opener Enter never commits deletion."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items(3))
    feed.install(app.media_reading_scope_service)
    permanent_calls = []

    async def permanently_delete(_service, *, mode, media_id):
        permanent_calls.append({"mode": mode, "media_id": media_id})
        feed.items[:] = [
            item for item in feed.items if item["backing_media_id"] != media_id
        ]
        return {"ok": True, "media_id": media_id}

    app.media_reading_scope_service.permanently_delete_media_item = types.MethodType(
        permanently_delete,
        app.media_reading_scope_service,
    )
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        screen.query_one("#library-media-trash-open", Button).press()
        controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.state.applied_result is not None,
            message="Initial Trash page never applied.",
        )
        await _wait_for_selector(screen, pilot, "#library-media-trash-delete")

        delete = screen.query_one("#library-media-trash-delete", Button)
        delete.focus()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-media-trash-delete-cancel")
        await _wait_for_condition(
            pilot,
            lambda: (
                getattr(screen.focused, "id", None)
                == "library-media-trash-delete-cancel"
            ),
            message="Cancel did not receive initial confirmation focus.",
        )
        assert permanent_calls == []

        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: (
                not screen.query("#library-media-trash-delete-confirmation")
                and getattr(screen.focused, "id", None) == "library-media-trash-delete"
            ),
            message="Escape did not safely cancel and restore opener focus.",
        )
        assert permanent_calls == []

        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                getattr(screen.focused, "id", None)
                == "library-media-trash-delete-cancel"
            ),
            message="Reopened confirmation did not focus Cancel.",
        )
        await pilot.press("tab")
        assert getattr(screen.focused, "id", None) == (
            "library-media-trash-delete-confirm"
        )
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: len(permanent_calls) == 1,
            message="Later explicit confirmation did not commit.",
        )
        assert permanent_calls == [{"mode": "local", "media_id": 1}]


@pytest.mark.asyncio
async def test_media_trash_restore_preserves_normal_page_and_marks_only_it_stale():
    """A ranked normal-Media page is retained byte-for-byte after Restore."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items(3))
    feed.install(app.media_reading_scope_service)

    async def restore_item(_service, *, mode, media_id, **_kwargs):
        assert mode == "local"
        restored = next(
            item for item in feed.items if item["backing_media_id"] == media_id
        )
        feed.items[:] = [
            item for item in feed.items if item["backing_media_id"] != media_id
        ]
        return {
            "id": media_id,
            "title": restored["title"],
            "type": restored["media_type"],
            "deleted": 0,
            "is_trash": 0,
            "last_modified": "2026-08-30T00:00:00+00:00",
        }

    app.media_reading_scope_service.restore_media_item = types.MethodType(
        restore_item,
        app.media_reading_scope_service,
    )
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        normal = screen._library_media_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: normal.applied_result is not None,
            message="Normal Media page never applied.",
        )
        retained_applied = normal.applied_result
        retained_items = normal.retained_items
        retained_requested_scope = normal.requested_scope
        retained_selected_id = screen._selected_media_id

        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        screen.query_one("#library-media-trash-open", Button).press()
        trash = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: trash.state.applied_result is not None,
            message="Initial Trash page never applied.",
        )
        await _wait_for_selector(screen, pilot, "#library-media-trash-restore")
        screen.query_one("#library-media-trash-restore", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                trash.state.applied_result is not None
                and trash.state.applied_result.total == 2
                and trash.state.freshness == "fresh"
            ),
            message="Authoritative post-Restore Trash page never applied.",
        )

        assert normal.applied_result is retained_applied
        assert normal.retained_items is retained_items
        assert normal.requested_scope == retained_requested_scope
        assert screen._selected_media_id == retained_selected_id
        assert normal.freshness == "stale"
        assert normal.pager.title_count is None
        assert normal.pager.retry_visible is True
        # The successful Trash read owns only Trash and cannot clear Media stale.
        assert trash.state.freshness == "fresh"
        assert normal.freshness == "stale"


@pytest.mark.asyncio
async def test_media_trash_commit_unknown_blocks_back_but_refresh_can_be_abandoned():
    """Only the irreversible-call phase owns the temporary Back exclusion."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items(3))
    commit_entered = threading.Event()
    release_commit = threading.Event()
    refresh_entered = threading.Event()
    release_refresh = threading.Event()
    list_calls = 0

    async def list_trash(_service, **kwargs):
        nonlocal list_calls
        list_calls += 1
        if list_calls > 1:
            refresh_entered.set()
            await asyncio.to_thread(release_refresh.wait, 10.0)
        return await feed.list_library_media_trash(**kwargs)

    async def permanently_delete(_service, *, mode, media_id):
        assert mode == "local"
        commit_entered.set()
        await asyncio.to_thread(release_commit.wait, 10.0)
        feed.items[:] = [
            item for item in feed.items if item["backing_media_id"] != media_id
        ]
        return {"ok": True, "media_id": media_id}

    scope_service = app.media_reading_scope_service
    scope_service.list_library_media_trash = types.MethodType(list_trash, scope_service)
    scope_service.permanently_delete_media_item = types.MethodType(
        permanently_delete, scope_service
    )
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-media", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-trash-open")
            screen.query_one("#library-media-trash-open", Button).press()
            controller = screen._library_media_trash_browse_controller
            await _wait_for_condition(
                pilot,
                lambda: controller.state.applied_result is not None,
                message="Initial Trash page never applied.",
            )
            await _wait_for_selector(screen, pilot, "#library-media-trash-delete")
            screen.query_one("#library-media-trash-delete", Button).press()
            await _wait_for_selector(
                screen, pilot, "#library-media-trash-delete-confirm"
            )
            screen.query_one("#library-media-trash-delete-confirm", Button).press()
            await _wait_for_condition(
                pilot,
                commit_entered.is_set,
                message="Permanent delete did not reach its commit gate.",
            )

            assert controller.state.mutation_pending is True
            assert screen._library_media_bulk_delete_in_flight is True
            back = screen.query_one("#library-media-trash-back", Button)
            assert back.disabled is True
            assert back.tooltip == "Finishing this action…"
            assert (
                screen.query_one("#library-media-trash-status", Static).renderable
                == "Finishing this action…"
            )
            assert screen.check_action("library_media_trash_back", ()) is False
            screen.action_library_media_trash_back()
            assert screen._library_media_view == "trash"

            release_commit.set()
            await _wait_for_condition(
                pilot,
                refresh_entered.is_set,
                message="Committed delete did not begin its Trash refresh.",
            )
            assert controller.state.mutation_pending is False
            assert controller.state.loading is True
            assert screen._library_media_bulk_delete_in_flight is False
            back = screen.query_one("#library-media-trash-back", Button)
            assert back.disabled is False
            assert screen.check_action("library_media_trash_back", ()) is True

            await pilot.press("escape")
            await _wait_for_condition(
                pilot,
                lambda: screen._library_media_view == "list",
                message="Post-commit refresh could not be abandoned.",
            )
            release_refresh.set()
            await pilot.pause()
            assert screen._library_media_view == "list"
    finally:
        release_commit.set()
        release_refresh.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "error_copy", "expected_focus_id"),
    (
        (
            "restore",
            "Could not restore this media item.",
            "library-media-trash-restore",
        ),
        (
            "permanent-delete",
            "Could not delete this media item permanently.",
            "library-media-trash-delete",
        ),
    ),
    ids=("restore", "permanent-delete"),
)
async def test_media_trash_precommit_failure_releases_mounted_controls_without_refresh(
    operation, error_copy, expected_focus_id
):
    """A settled pre-commit failure republishes Trash after releasing its lease."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items())
    feed.install(app.media_reading_scope_service)
    service_calls = []

    async def fail_mutation(_service, *, mode, media_id, **_kwargs):
        service_calls.append({"mode": mode, "media_id": media_id})
        raise PermissionError("private policy detail")

    service = app.media_reading_scope_service
    if operation == "restore":
        service.restore_media_item = types.MethodType(fail_mutation, service)
    else:
        service.permanently_delete_media_item = types.MethodType(fail_mutation, service)
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        screen.query_one("#library-media-trash-open", Button).press()
        controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.state.applied_result is not None,
            message="Initial Trash page never applied.",
        )
        await _wait_for_selector(screen, pilot, "#library-media-trash-restore")

        retained_applied = controller.state.applied_result
        retained_items = controller.state.retained_items
        retained_selected_id = controller.state.selected_id
        assert retained_applied is not None
        assert retained_applied.total == 45
        assert len(feed.calls) == 1

        if operation == "restore":
            action = screen.query_one("#library-media-trash-restore", Button)
            action.focus()
            await pilot.press("enter")
        else:
            action = screen.query_one("#library-media-trash-delete", Button)
            action.focus()
            await pilot.press("enter")
            await _wait_for_selector(
                screen, pilot, "#library-media-trash-delete-confirm"
            )
            await _wait_for_condition(
                pilot,
                lambda: (
                    getattr(screen.focused, "id", None)
                    == "library-media-trash-delete-cancel"
                ),
                message="Permanent-delete confirmation never focused Cancel.",
            )
            await pilot.press("tab", "enter")

        def settled_with_enabled_controls() -> bool:
            if (
                controller.state.mutation_pending
                or screen._library_media_bulk_delete_in_flight
                or controller.state.error_copy != error_copy
            ):
                return False
            selectors = (
                "#library-media-trash-restore",
                "#library-media-trash-delete",
                "#library-media-trash-search",
                "#library-media-trash-type-filter",
                "#library-media-trash-next",
            )
            controls = [screen.query_one(selector) for selector in selectors]
            return all(not control.disabled for control in controls) and (
                getattr(screen.focused, "id", None) == expected_focus_id
            )

        await _wait_for_condition(
            pilot,
            settled_with_enabled_controls,
            message=(
                f"{operation} failure never republished enabled Trash controls "
                "with recoverable action focus."
            ),
        )

        assert service_calls == [{"mode": "local", "media_id": 1}]
        assert len(feed.calls) == 1
        assert controller.state.applied_result is retained_applied
        assert controller.state.retained_items is retained_items
        assert controller.state.selected_id == retained_selected_id
        assert controller.state.freshness == "fresh"
        assert controller.state.loading is False
        assert controller.state.applied_result.total == 45
        assert (
            screen.query_one("#library-media-trash-status", Static).renderable
            == error_copy
        )
        assert (
            screen.query_one("#library-media-trash-previous", Button).disabled is True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "outcome"),
    (
        ("permanent-delete", {"ok": True, "media_id": 999}),
        ("permanent-delete", {"ok": False, "media_id": 1}),
        ("permanent-delete", {"ok": True}),
        ("permanent-delete", {"ok": True, "media_id": True}),
        ("permanent-delete", ["malformed"]),
        (
            "restore",
            {
                "id": 999,
                "title": "Wrong identity",
                "type": "audio",
                "deleted": 0,
                "is_trash": 0,
            },
        ),
        ("restore", {"ok": False, "media_id": 999}),
        (
            "restore",
            {"id": 1, "title": "Missing state", "type": "audio"},
        ),
        (
            "restore",
            {
                "id": True,
                "title": "Wrong type",
                "type": "audio",
                "deleted": 0,
                "is_trash": 0,
            },
        ),
        ("restore", ["malformed"]),
    ),
)
async def test_media_trash_malformed_mutation_results_fail_closed(
    operation, outcome
):
    """Malformed acknowledgements retain truthful fresh action authority."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items(1))
    feed.install(app.media_reading_scope_service)
    mutation_calls: list[dict[str, object]] = []

    async def mutate(_service, **kwargs):
        mutation_calls.append(dict(kwargs))
        return outcome

    service = app.media_reading_scope_service
    method_name = (
        "restore_media_item"
        if operation == "restore"
        else "permanently_delete_media_item"
    )
    setattr(service, method_name, types.MethodType(mutate, service))
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        screen.query_one("#library-media-trash-open", Button).press()
        controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.state.applied_result is not None,
            message="Initial Trash page never applied.",
        )
        await _wait_for_selector(
            screen, pilot, "#library-media-trash-restore"
        )
        retained_applied = controller.state.applied_result
        retained_items = controller.state.retained_items
        retained_selected_id = controller.state.selected_id
        records_before = screen._local_source_records.get("media", ())

        if operation == "restore":
            action = screen.query_one("#library-media-trash-restore", Button)
            expected_error = "Could not restore this media item."
        else:
            action = screen.query_one("#library-media-trash-delete", Button)
            expected_error = "Could not delete this media item permanently."
        action.focus()
        await pilot.press("enter")
        if operation == "permanent-delete":
            await _wait_for_selector(
                screen, pilot, "#library-media-trash-delete-confirm"
            )
            await pilot.press("tab", "enter")

        await _wait_for_condition(
            pilot,
            lambda: (
                not controller.state.mutation_pending
                and not screen._library_media_bulk_delete_in_flight
            ),
            message="Malformed mutation response did not settle.",
        )

        assert mutation_calls
        if operation == "restore":
            assert mutation_calls == [
                {
                    "mode": "local",
                    "media_id": 1,
                    "include_content": False,
                    "include_versions": False,
                }
            ]
        else:
            assert mutation_calls == [{"mode": "local", "media_id": 1}]
        assert controller.state.applied_result is retained_applied
        assert controller.state.retained_items is retained_items
        assert controller.state.selected_id == retained_selected_id
        assert controller.state.freshness == "fresh"
        assert controller.state.error_copy == expected_error
        assert controller.state.committed_notice == ""
        assert screen._local_source_records.get("media", ()) is records_before
        assert len(feed.calls) == 1
        assert not action.disabled


def test_media_trash_restore_summary_accepts_blank_persisted_type():
    """A committed restore normalizes the schema-valid blank type to None."""
    summary = LibraryScreen._validated_library_media_trash_restore_summary(
        {
            "id": 7,
            "title": "Restored",
            "type": "   ",
            "deleted": 0,
            "is_trash": 0,
        },
        media_id=7,
    )

    assert summary == {
        "id": 7,
        "title": "Restored",
        "type": None,
        "deleted": 0,
        "is_trash": 0,
    }


@pytest.mark.asyncio
async def test_media_trash_restore_bounds_request_and_retained_summary():
    """A successful restore never requests or retains unbounded detail."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    restored_id = 41
    trash_item = {
        "id": f"local:media:{restored_id}",
        "backing_media_id": restored_id,
        "title": "Large restored item",
        "media_type": "document",
        "trash_date": "2026-08-30T00:00:00+00:00",
    }
    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed([trash_item])
    feed.install(app.media_reading_scope_service)
    restore_calls: list[dict[str, object]] = []

    async def restore(_service, **kwargs):
        restore_calls.append(dict(kwargs))
        feed.items.clear()
        return {
            "id": restored_id,
            "title": "Large restored item",
            "type": "document",
            "deleted": 0,
            "is_trash": 0,
            "content": "private-large-content" * 10_000,
            "versions": [{"content": "private-version-content" * 10_000}],
            "documents": [{"body": "private-document-body" * 10_000}],
        }

    app.media_reading_scope_service.restore_media_item = types.MethodType(
        restore,
        app.media_reading_scope_service,
    )
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        normal = screen._library_media_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: normal.applied_result is not None,
            message="Normal Media page never applied.",
        )
        retained_normal_result = normal.applied_result
        retained_normal_items = normal.retained_items
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        screen.query_one("#library-media-trash-open", Button).press()
        trash = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: trash.state.applied_result is not None,
            message="Initial Trash page never applied.",
        )
        await _wait_for_selector(
            screen, pilot, "#library-media-trash-restore"
        )
        screen.query_one("#library-media-trash-restore", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                trash.state.applied_result is not None
                and trash.state.applied_result.total == 0
                and trash.state.freshness == "fresh"
            ),
            message="Post-restore Trash page never settled empty.",
        )

        assert restore_calls == [
            {
                "mode": "local",
                "media_id": restored_id,
                "include_content": False,
                "include_versions": False,
            }
        ]
        restored_summaries = tuple(
            record
            for record in screen._local_source_records.get("media", ())
            if screen._source_record_id(record) == str(restored_id)
        )
        assert restored_summaries == (
            {
                "id": restored_id,
                "title": "Large restored item",
                "type": "document",
                "deleted": 0,
                "is_trash": 0,
            },
        )
        assert "private-" not in repr(restored_summaries)
        assert normal.applied_result is retained_normal_result
        assert normal.retained_items is retained_normal_items
        assert normal.freshness == "stale"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((160, 50), (120, 35), (100, 30), (80, 24)))
async def test_media_trash_geometry_four_sizes_paints_all_fixed_controls(size):
    """All four postures keep the bounded vertical grammar in the Items pane."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryProductionCSSHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items())
    feed.install(app.media_reading_scope_service)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        media_row = screen.query_one("#library-row-browse-media", Button)
        media_row.focus()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_layout.reader_width > 0,
            message="Media reader allocation never settled for geometry inspection.",
        )
        if not screen._library_media_reader_layout.items_open:
            items_grip = screen.query_one("#library-media-items-grip", Button)
            items_grip.focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_media_reader_layout.items_open
                    and screen.query_one("#library-canvas").region.area > 0
                ),
                message="Items pane never opened for compact Trash inspection.",
            )
        await pilot.pause()
        opener = screen.query_one("#library-media-trash-open", Button)
        opener.focus()
        await pilot.press("enter")
        controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.state.applied_result is not None,
            message="Trash page never applied for geometry inspection.",
        )

        ordinary = controller.state
        first_item = ordinary.retained_items[0]
        full_confirmation_title = f"{'A' * 140} selected tail"
        confirmation_target = MediaTrashMutationTarget(
            stable_id=str(first_item["id"]),
            backing_media_id=int(first_item["backing_media_id"]),
            title=full_confirmation_title,
            media_type=str(first_item["media_type"]),
            trash_date=str(first_item["trash_date"]),
            page_index=0,
        )
        postures = {
            "ordinary": ordinary,
            "confirmation": dataclasses.replace(
                ordinary,
                selected_id=confirmation_target.stable_id,
                confirmation_target=confirmation_target,
            ),
            "stale": dataclasses.replace(
                ordinary,
                freshness="stale",
                stale_copy="List may be out of date.",
                selected_id="",
                failed_scope=MediaTrashScope(page=2),
                failed_origin="next",
            ),
            "initial-error": fail_media_trash_request(
                begin_media_trash_request(
                    MediaTrashBrowseState(), MediaTrashScope(), origin="entry"
                ),
                MediaTrashScope(),
                copy="Could not load Trash.",
            ),
        }
        for posture, state in postures.items():
            controller.state = state
            screen._sync_library_media_trash_state(None)
            await pilot.pause()
            await pilot.pause()

            heading = screen.query_one("#library-media-trash-heading")
            filters = screen.query_one("#library-media-trash-filters")
            status = screen.query_one("#library-media-trash-status")
            trash_list = screen.query_one("#library-media-trash-list")
            pager = screen.query_one("#library-media-trash-pager")
            action_region = screen.query_one(
                "#library-media-trash-delete-confirmation"
                if posture == "confirmation"
                else "#library-media-trash-actions"
            )
            items_pane = screen.query_one("#library-canvas")
            assert (
                heading.region.y
                <= filters.region.y
                <= status.region.y
                <= trash_list.region.y
                < pager.region.y
                < action_region.region.y
            ), (
                f"{posture}: heading={heading.region!r}, "
                f"filters={filters.region!r}, status={status.region!r}, "
                f"list={trash_list.region!r}, pager={pager.region!r}, "
                f"actions={action_region.region!r}, "
                f"layout={screen._library_media_reader_layout!r}, "
                f"items_display={items_pane.display!r}, "
                f"items={items_pane.region!r}, "
                "canvas="
                f"{screen.query_one('#library-media-trash-canvas').region!r}"
            )
            assert status.region.height <= 3
            assert trash_list.region.height >= (4 if size == (80, 24) else 1)
            previous = screen.query_one("#library-media-trash-previous", Button)
            next_button = screen.query_one("#library-media-trash-next", Button)
            checked_controls = [
                pager,
                action_region,
                previous,
                next_button,
            ]
            if posture == "confirmation":
                details = screen.query_one(
                    "#library-media-trash-delete-confirm-details", VerticalScroll
                )
                title = screen.query_one(
                    "#library-media-trash-delete-confirm-title", Static
                )
                media_type = screen.query_one(
                    "#library-media-trash-delete-confirm-type", Static
                )
                deletion_time = screen.query_one(
                    "#library-media-trash-delete-confirm-time", Static
                )
                cancel = screen.query_one("#library-media-trash-delete-cancel", Button)
                confirm = screen.query_one(
                    "#library-media-trash-delete-confirm", Button
                )
                assert 0 < details.region.height <= 2
                assert title.renderable == full_confirmation_title
                checked_controls.extend(
                    (details, media_type, deletion_time, cancel, confirm)
                )
            else:
                restore = screen.query_one("#library-media-trash-restore", Button)
                delete = screen.query_one("#library-media-trash-delete", Button)
                checked_controls.extend((restore, delete))
            if screen.query("#library-media-trash-retry"):
                checked_controls.append(
                    screen.query_one("#library-media-trash-retry", Button)
                )
            for widget in checked_controls:
                assert items_pane.region.contains_region(widget.region), (
                    f"{posture}: {widget.id}={widget.region!r}, "
                    f"items={items_pane.region!r}"
                )
                hit, _ = screen.get_widget_at(*widget.region.center)
                assert (
                    hit is widget or widget in hit.ancestors or hit in widget.ancestors
                )

            painted = _compositor_text(host.export_screenshot(simplify=True))
            if size == (80, 24):
                assert items_pane.region.width == 32
            assert "Previous" in painted
            assert "Next" in painted
            if screen.query("#library-media-trash-retry"):
                assert "Retry" in painted
            if posture == "confirmation":
                assert "cannot be undone" in painted
                assert confirmation_target.media_type in painted
                assert confirmation_target.trash_date in painted
                assert "Cancel" in painted
                assert "Delete permanently" in painted
            else:
                assert "Restore" in painted
                assert "Delete permanently" in painted, (
                    f"{posture}: items={items_pane.region!r}, "
                    f"actions={action_region.region!r}, "
                    f"delete={delete.region!r}, label={str(delete.label)!r}"
                )
            assert trash_list.styles.height.is_fraction
            assert trash_list.styles.min_height.value == 0
            if posture == "initial-error":
                assert (
                    screen.query_one("#library-media-trash-title", Static).renderable
                    == "Local Trash"
                )
                assert not screen.query_one(
                    "#library-media-trash-page", Static
                ).renderable
            elif posture == "stale":
                assert "List may be out of date" in painted


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_key", ("button", "escape"))
async def test_media_trash_back_and_escape_restore_distinct_media_return(exit_key):
    """A Trash round-trip restores Media without consuming Viewer receipt."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    media = [
        {
            "id": f"media-{index}",
            "title": f"Media {index:02d}",
            "type": "video",
            "last_modified": f"2026-08-{(index % 28) + 1:02d}T00:00:00Z",
        }
        for index in range(1, 46)
    ]
    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=media)
    feed = _MountedTrashFeed(_canonical_trash_items())
    feed.install(app.media_reading_scope_service)
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        controller = screen._library_media_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.applied_scope == MediaBrowseScope(),
            message="Initial Media page never applied.",
        )
        screen._request_library_media_page(2, focus_identity=None)
        await _wait_for_condition(
            pilot,
            lambda: controller.applied_scope == MediaBrowseScope(page=2),
            message="Media page 2 never applied.",
        )
        selected_id = str(controller.retained_items[4]["id"])
        screen._selected_media_id = selected_id
        row_scroll = screen.query_one("#library-media-row-scroll")
        row_scroll.scroll_to(y=4, animate=False, force=True, immediate=True)
        await pilot.pause()
        scroll_offset = (int(row_scroll.scroll_x), int(row_scroll.scroll_y))
        assert scroll_offset[1] > 0

        viewer_receipt = ("viewer-sentinel", (0, 1))
        screen._library_media_viewer_return = viewer_receipt
        opener = screen.query_one("#library-media-trash-open", Button)
        opener.focus()
        await pilot.press("enter")
        trash_controller = screen._library_media_trash_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: trash_controller.state.applied_result is not None,
            message="Trash page never applied.",
        )
        screen._request_library_media_trash_page(
            2, focus_identity="#library-media-trash-next"
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                trash_controller.state.applied_result is not None
                and trash_controller.state.applied_result.scope
                == MediaTrashScope(page=2)
            ),
            message="Independent Trash page 2 never applied.",
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                list(screen.query("#library-media-trash-row-0"))
                and getattr(
                    screen.query_one("#library-media-trash-row-0"),
                    "media_id",
                    "",
                )
                == "local:media:21"
            ),
            message="Independent Trash page 2 never reached the mounted canvas.",
        )
        await pilot.pause()

        if exit_key == "escape":
            assert screen.check_action("library_media_trash_back", ()) is True
            await pilot.press("escape")
        else:
            back = screen.query_one("#library-media-trash-back", Button)
            back.focus()
            await pilot.pause()
            back.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_view == "list",
            message=f"Trash {exit_key} exit never reached normal Media.",
        )
        await _wait_for_selector(screen, pilot, "#library-media-canvas")
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None) == "library-media-trash-open",
            message="Trash return did not restore toolbar focus.",
        )

        assert controller.applied_scope == MediaBrowseScope(page=2)
        assert screen._selected_media_id == selected_id
        assert screen._library_media_viewer_return == viewer_receipt
        restored_scroll = screen.query_one("#library-media-row-scroll")
        assert (int(restored_scroll.scroll_x), int(restored_scroll.scroll_y)) == (
            scroll_offset
        )


@pytest.mark.asyncio
async def test_media_trash_back_generation_fences_late_reactivated_completion():
    """Back invalidation survives a later reactivation of the same subview."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items(), gate_first=True)
    feed.install(app.media_reading_scope_service)
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-media", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-trash-open")
            screen.query_one("#library-media-trash-open", Button).press()
            await _wait_for_condition(
                pilot,
                feed.entered.is_set,
                message="Late Trash request never reached its gate.",
            )
            trash_controller = screen._library_media_trash_browse_controller
            screen._exit_library_media_trash()
            await _wait_for_selector(screen, pilot, "#library-media-canvas")

            # Model the same retained Screen becoming Trash-active again before
            # any new request generation starts. Only Back's explicit
            # invalidation can fence the first session in this interval.
            screen._library_media_view = "trash"
            feed.release.set()
            await pilot.pause()
            await pilot.pause()
            screen._library_media_view = "list"

            assert trash_controller.state.applied_result is None
            assert screen.query("#library-media-canvas")
    finally:
        feed.release.set()


@pytest.mark.asyncio
async def test_media_trash_unmount_generation_fences_late_completion():
    """Unmount invalidation rejects a local read even if its route reactivates."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items(), gate_first=True)
    feed.install(app.media_reading_scope_service)
    host = LibraryHarness(app)
    invalidated = asyncio.Event()
    release_after_unmount = None

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-media", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-trash-open")
            screen.query_one("#library-media-trash-open", Button).press()
            await _wait_for_condition(
                pilot,
                feed.entered.is_set,
                message="Late Trash request never reached its gate.",
            )
            controller = screen._library_media_trash_browse_controller
            original_invalidate = controller.invalidate

            def invalidate_and_signal():
                generation = original_invalidate()
                invalidated.set()
                return generation

            controller.invalidate = invalidate_and_signal

            async def reactivate_and_release_after_unmount():
                await invalidated.wait()
                # Match the Back inverse: make the retained predicates active
                # again so only the explicit generation fence can reject the
                # first session's completion.
                screen._library_media_view = "trash"
                feed.release.set()
                await asyncio.sleep(0)

            release_after_unmount = asyncio.create_task(
                reactivate_and_release_after_unmount()
            )

        await asyncio.wait_for(release_after_unmount, timeout=2.0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert invalidated.is_set()
        assert controller.state.applied_result is None
    finally:
        feed.release.set()
        if release_after_unmount is not None and not release_after_unmount.done():
            release_after_unmount.cancel()


@pytest.mark.asyncio
async def test_media_trash_unmount_fences_inflight_restore_completion():
    """A detached Trash route releases its interlock without projecting success."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_media_items,
        _wait_for_condition,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    class RestoreGate:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()
            self.calls: list[dict[str, object]] = []

        async def restore(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(dict(kwargs))
            self.entered.set()
            await asyncio.to_thread(self.release.wait, 10.0)
            media_id = int(kwargs["media_id"])
            return {
                "id": media_id,
                "title": "Trash 01",
                "type": "audio",
                "deleted": 0,
                "is_trash": 0,
            }

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=_two_media_items())
    feed = _MountedTrashFeed(_canonical_trash_items(1))
    feed.install(app.media_reading_scope_service)
    restore_gate = RestoreGate()

    async def restore_media_item(_service: object, **kwargs: object):
        return await restore_gate.restore(**kwargs)

    app.media_reading_scope_service.restore_media_item = types.MethodType(
        restore_media_item,
        app.media_reading_scope_service,
    )
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-media", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-trash-open")
            screen.query_one("#library-media-trash-open", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-trash-row-0")
            await _wait_for_condition(
                pilot,
                lambda: screen._library_media_trash_browse_controller.state.freshness
                == "fresh",
                message="Trash page did not settle before restore.",
            )
            screen.query_one("#library-media-trash-restore", Button).press()
            await _wait_for_condition(
                pilot,
                restore_gate.entered.is_set,
                message="Restore mutation never reached its gate.",
            )
            controller = screen._library_media_trash_browse_controller
            records_before = screen._local_source_records.get("media", ())
            count_before = screen._local_source_counts.get("media", 0)
            list_calls_before = len(feed.calls)

            await screen.on_unmount()
            invalidated_state = controller.state
            sync_calls: list[str | None] = []
            refresh_calls: list[dict[str, object]] = []
            screen._sync_library_media_trash_state = sync_calls.append
            original_refresh = screen.refresh

            def record_refresh(*args, **kwargs):
                refresh_calls.append(dict(kwargs))
                return original_refresh(*args, **kwargs)

            screen.refresh = record_refresh
            screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
            screen._library_media_view = "trash"
            restore_gate.release.set()
            await _wait_for_condition(
                pilot,
                lambda: not screen._library_media_bulk_delete_in_flight,
                message="Stale restore did not release the shared interlock.",
            )

            assert controller.state is invalidated_state
            assert controller.state.committed_notice == ""
            assert screen._local_source_records.get("media", ()) is records_before
            assert screen._local_source_counts.get("media", 0) == count_before
            assert len(feed.calls) == list_calls_before
            assert sync_calls == []
            assert not any(call.get("recompose") is True for call in refresh_calls)
    finally:
        restore_gate.release.set()


# ---------------------------------------------------------------------------
# The media list toolbar's "Trash" entry point (AC#1 reachability)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_media_list_toolbar_offers_trash_outside_select_mode():
    """AC#1: the Trash surface is reachable from the Media canvas -- the
    list toolbar carries a "Trash" action (hidden in select mode like
    "Export…"; Select's toolbar acts on the selection, never navigates)."""
    import dataclasses as _dataclasses

    from tldw_chatbook.Widgets.Library.library_media_canvas import (
        LibraryMediaCanvas,
    )
    from tldw_chatbook.Library.library_media_state import build_library_media_state

    list_state = build_library_media_state([{"id": "1", "title": "One", "type": "pdf"}])

    class _ListApp(ConsolidatedCSSApp):
        def __init__(self, state):
            super().__init__()
            self._state = state

        def compose(self):
            yield LibraryMediaCanvas(canvas=self._state, id="library-media-canvas")

    app = _ListApp(list_state)
    async with app.run_test() as pilot:
        await pilot.pause()
        trash_btn = app.query_one("#library-media-trash-open", Button)
        assert trash_btn.display is True
        assert str(trash_btn.label) == "Trash"

    select_state = _dataclasses.replace(list_state, select_mode=True)
    app = _ListApp(select_state)
    async with app.run_test() as pilot:
        await pilot.pause()
        trash_btn = app.query_one("#library-media-trash-open", Button)
        assert trash_btn.display is False


@pytest.mark.asyncio
async def test_confirm_copies_and_receipt_point_at_trash():
    """AC#3 (ADR-055 Pattern A): both delete confirmations and the receipt
    name the durable recovery path -- Trash -- instead of the old
    "there's no Trash view" copy."""
    import dataclasses as _dataclasses

    from tldw_chatbook.Widgets.Library.library_media_canvas import (
        LibraryMediaCanvas,
    )
    from tldw_chatbook.Library.library_media_state import build_library_media_state
    from tldw_chatbook.Library.library_media_viewer_state import (
        build_library_media_viewer_state,
    )
    from tldw_chatbook.Widgets.Library.library_media_viewer import (
        LibraryMediaViewer,
    )

    confirming_state = _dataclasses.replace(
        build_library_media_state(
            [{"id": "1", "title": "One", "type": "pdf"}],
            select_mode=True,
            selected_ids=frozenset({"1"}),
        ),
        confirming_bulk_delete=True,
    )
    receipt_state = _dataclasses.replace(
        build_library_media_state([{"id": "1", "title": "One", "type": "pdf"}]),
        delete_receipt_count=2,
    )

    class _ConfirmApp(ConsolidatedCSSApp):
        def __init__(self, state):
            super().__init__()
            self._state = state

        def compose(self):
            yield LibraryMediaCanvas(canvas=self._state, id="library-media-canvas")

    app = _ConfirmApp(confirming_state)
    async with app.run_test() as pilot:
        await pilot.pause()
        copy = str(
            app.query_one("#library-media-bulk-delete-confirm-copy", Static).renderable
        )
        assert "restore later from Trash" in copy
        assert "no Trash view" not in copy

    app = _ConfirmApp(receipt_state)
    async with app.run_test() as pilot:
        await pilot.pause()
        receipt = str(
            app.query_one("#library-media-bulk-delete-receipt-copy", Static).renderable
        )
        assert receipt == "✓ deleted · 2 items · in Trash"

    class _ViewerApp(ConsolidatedCSSApp):
        def compose(self):
            yield LibraryMediaViewer(
                build_library_media_viewer_state(
                    {"id": "1", "title": "One", "type": "pdf", "content": "x"}
                ),
                confirming_delete=True,
                id="library-media-viewer",
            )

    app = _ViewerApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        copy = str(
            app.query_one("#library-media-delete-confirm-copy", Static).renderable
        )
        assert "restore later from Trash" in copy
        assert "no Trash view" not in copy


# ---------------------------------------------------------------------------
# Handler tests (SimpleNamespace fakes, mirroring test_library_multiselect_media)
# ---------------------------------------------------------------------------


def _bind_trash_mutation_seams(fake):
    """Give direct restore fakes the production mutation boundary shape."""
    events = []
    scope = MediaBrowseScope()
    fake._mutation_events = events
    fake._library_media_browse_controller = SimpleNamespace(
        mutation_refresh_scope=scope,
        begin_mutation=lambda: events.append(("begin",)) or scope,
        mark_stale_after_trash_restore=lambda: events.append(("mark-stale",)),
        reconcile_committed_mutation=lambda **kwargs: events.append(
            ("reconcile", kwargs)
        ),
        request=lambda requested, **kwargs: events.append(
            ("request", requested, kwargs)
        ),
        request_facets=lambda **kwargs: events.append(("facets", kwargs)),
    )
    fake._library_media_mutation_scope = None
    fake._library_media_mutation_authority = None
    fake._library_media_lifecycle_generation = 0
    fake._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    fake._library_media_type_choices_visible = False
    fake._sync_library_media_browse_state = lambda *_args: events.append(("sync",))
    fake._sync_library_media_viewer_mutation_gate = lambda: None
    fake._begin_library_media_mutation = types.MethodType(
        LibraryScreen._begin_library_media_mutation, fake
    )
    fake._library_media_backing_id = types.MethodType(
        LibraryScreen._library_media_backing_id, fake
    )
    fake._required_library_media_backing_id = types.MethodType(
        LibraryScreen._required_library_media_backing_id, fake
    )
    fake._library_media_mutation_summary = types.MethodType(
        LibraryScreen._library_media_mutation_summary, fake
    )
    fake._complete_library_media_mutation = types.MethodType(
        LibraryScreen._complete_library_media_mutation, fake
    )
    if fake._library_media_bulk_delete_in_flight:
        fake._begin_library_media_mutation()
    return fake


def _trash_view_fake(
    *,
    records=None,
    total=0,
    selected_id="",
    view="trash",
    in_flight=False,
):
    from tldw_chatbook.Library.row_selection import RowSelection

    notified = []
    refresh_calls = []
    worker_calls = []
    after_refresh = []
    trash_controller_calls = []
    first_record = next(iter(records or ()), None)
    target_id = selected_id or (
        LibraryScreen._source_record_id(first_record)
        if isinstance(first_record, dict)
        else ""
    )
    mutation_target = (
        _trash_mutation_target(target_id, str(first_record.get("title") or "Untitled"))
        if target_id and isinstance(first_record, dict)
        else None
    )
    trash_controller = SimpleNamespace(
        state=MediaTrashBrowseState(
            retained_items=tuple(records or ()), selected_id=selected_id
        ),
        invalidate=lambda: trash_controller_calls.append(("invalidate",)),
        request=lambda scope, **kwargs: trash_controller_calls.append(
            ("request", scope, kwargs)
        ),
        select=lambda stable_id: trash_controller_calls.append(("select", stable_id)),
        claim_mutation=lambda: (
            _trash_mutation_claim(mutation_target)
            if mutation_target is not None
            else None
        ),
    )
    fake = SimpleNamespace(
        _library_media_select_mode=False,
        _library_media_row_selection=RowSelection("media"),
        _library_media_confirming_bulk_delete=False,
        _library_media_bulk_delete_in_flight=in_flight,
        _library_media_delete_receipt_ids=("stale-receipt-id",),
        _library_media_view=view,
        _library_media_trash_browse_controller=trash_controller,
        _library_media_trash_query_draft="",
        _library_media_trash_input_error="",
        _library_media_trash_type_choices_visible=False,
        _library_media_trash_focus_identity="#library-media-trash-row-0",
        _library_notes_focus_intent_generation=0,
        _library_media_trash_return=None,
        _trash_controller_calls=trash_controller_calls,
        app_instance=SimpleNamespace(notify=lambda msg, **k: notified.append((msg, k))),
        _notified=notified,
        _refresh_calls=refresh_calls,
        _worker_calls=worker_calls,
        _after_refresh=after_refresh,
        is_mounted=True,
        refresh=lambda **k: refresh_calls.append(k),
        run_worker=lambda coro, **k: worker_calls.append((coro, k)),
        call_after_refresh=lambda fn, *a: after_refresh.append(fn),
        _focus_library_media_trash_entry=lambda: None,
        _arm_library_list_entry_focus=lambda: None,
        _capture_library_media_trash_return=lambda: None,
        _disarm_library_media_return_for_route_change=lambda: None,
    )
    fake._exit_library_media_select_mode = types.MethodType(
        LibraryScreen._exit_library_media_select_mode, fake
    )
    fake._notify_library_media_selection_discarded = types.MethodType(
        LibraryScreen._notify_library_media_selection_discarded, fake
    )
    fake._build_library_media_trash_state = types.MethodType(
        LibraryScreen._build_library_media_trash_state, fake
    )
    fake._exit_library_media_trash = types.MethodType(
        LibraryScreen._exit_library_media_trash, fake
    )
    return _bind_trash_mutation_seams(fake)


def _failed_trash_restore_screen_fake():
    record = _canonical_trash_items(1)[0]
    scope = MediaTrashScope()
    loading = begin_media_trash_request(MediaTrashBrowseState(), scope, origin="entry")
    applied = apply_media_trash_result(
        loading,
        build_media_trash_result(
            scope,
            {
                "items": [record],
                "total": 1,
                "limit": 20,
                "offset": 0,
                "types": ["audio"],
            },
        ),
    )
    target = MediaTrashMutationTarget(
        stable_id="local:media:1",
        backing_media_id=1,
        title="Trash 01",
        media_type="audio",
        trash_date=str(record["trash_date"]),
        page_index=0,
    )
    state = fail_media_trash_mutation(
        begin_media_trash_mutation(applied),
        target,
        copy="Could not restore this media item.",
    )
    fake = SimpleNamespace(
        _library_selected_row_id=LIBRARY_ROW_BROWSE_MEDIA,
        _library_media_view="trash",
        _library_media_trash_browse_controller=SimpleNamespace(state=state),
        _library_media_trash_input_error="",
        _library_media_trash_focus_identity="#library-media-trash-restore",
        _focus_library_media_trash_intent=lambda: None,
    )
    fake._library_media_trash_retry_visible = types.MethodType(
        LibraryScreen._library_media_trash_retry_visible, fake
    )
    return fake


def test_media_trash_restore_failure_does_not_offer_browse_retry():
    """A failed mutation keeps its selected row and has no list-read action."""
    fake = _failed_trash_restore_screen_fake()

    presentation = LibraryScreen._build_library_media_trash_state(fake)

    assert presentation.error == "Could not restore this media item."
    assert "Retry" not in presentation.error
    assert presentation.selected_id == "local:media:1"
    assert len(presentation.rows) == 1
    assert presentation.rows[0].selected is True


def test_media_trash_restore_failure_preserves_restore_focus_not_retry(monkeypatch):
    """Mutation errors keep Restore authority rather than targeting read Retry."""
    fake = _failed_trash_restore_screen_fake()
    sync_calls = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen._sync_library_canvas",
        lambda *args, **kwargs: sync_calls.append((args, kwargs)),
    )

    LibraryScreen._sync_library_media_trash_state(fake, None)

    assert fake._library_media_trash_focus_identity == "#library-media-trash-restore"
    assert len(sync_calls) == 1


def test_media_trash_committed_refresh_failure_keeps_success_notice_and_retry():
    """A follow-up read failure cannot recast a committed Restore as failed."""
    scope = MediaTrashScope()
    applied = apply_media_trash_result(
        begin_media_trash_request(MediaTrashBrowseState(), scope, origin="entry"),
        build_media_trash_result(
            scope,
            {
                "items": _canonical_trash_items(2),
                "total": 2,
                "limit": 20,
                "offset": 0,
                "types": ["audio", "video"],
            },
        ),
    )
    target = MediaTrashMutationTarget(
        stable_id="local:media:1",
        backing_media_id=1,
        title="Trash 01",
        media_type="audio",
        trash_date="2026-08-02T00:00:00+00:00",
        page_index=0,
    )
    committed = commit_media_trash_mutation(
        begin_media_trash_mutation(applied),
        target,
        notice="Restored 'Trash 01'.",
    )
    failed = fail_media_trash_request(
        committed,
        committed.requested_scope,
        copy="List may be out of date.",
    )
    fake = SimpleNamespace(
        _library_media_trash_browse_controller=SimpleNamespace(state=failed),
        _library_media_trash_input_error="",
    )
    fake._library_media_trash_retry_visible = types.MethodType(
        LibraryScreen._library_media_trash_retry_visible, fake
    )

    presentation = LibraryScreen._build_library_media_trash_state(fake)

    assert presentation.error == (
        "Restored 'Trash 01'. List may be out of date · Retry"
    )
    assert "Could not restore" not in presentation.error
    assert presentation.notice == "Restored 'Trash 01'."


def test_media_trash_mutation_focus_falls_to_same_position_then_previous_then_back():
    fake = SimpleNamespace()

    assert LibraryScreen._library_media_trash_focus_selectors(
        fake, "#library-media-trash-row-3"
    ) == (
        "#library-media-trash-row-3",
        "#library-media-trash-row-2",
        "#library-media-trash-back",
    )
    assert LibraryScreen._library_media_trash_focus_selectors(
        fake, "#library-media-trash-row-0"
    ) == (
        "#library-media-trash-row-0",
        "#library-media-trash-back",
    )


def test_trash_open_enters_view_resets_state_and_kicks_fetch():
    """Entering Trash clears the stale receipt (the Trash view IS the
    durable path the receipt points at), resets the fetch/session state,
    recomposes, and schedules exactly one fetch worker."""
    fake = _trash_view_fake(view="list", records=("stale",), total=7)

    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_trash_open(fake, event)

    assert fake._library_media_view == "trash"
    assert fake._library_media_delete_receipt_ids == ()
    assert fake._library_media_trash_browse_controller.state == MediaTrashBrowseState()
    assert fake._library_media_trash_query_draft == ""
    assert fake._library_media_trash_input_error == ""
    assert fake._refresh_calls == [{"recompose": True}]
    assert fake._trash_controller_calls == [
        ("invalidate",),
        (
            "request",
            MediaTrashScope(),
            {
                "origin": "entry",
                "focus_identity": "#library-media-trash-row-0",
            },
        ),
    ]


def test_trash_row_press_updates_selection():
    fake = _trash_view_fake(
        records=(
            {"id": "1", "title": "A", "type": "pdf"},
            {"id": "2", "title": "B", "type": "pdf"},
        ),
        total=2,
    )
    event = SimpleNamespace(button=SimpleNamespace(media_id="2"), stop=lambda: None)

    LibraryScreen.handle_library_media_trash_row(fake, event)

    assert fake._trash_controller_calls == [("select", "2")]


def test_trash_back_returns_to_list_and_drops_session_state():
    events = []
    controller = SimpleNamespace(
        state=SimpleNamespace(retained_items=({"id": "local:media:1"},)),
        invalidate=lambda: events.append("invalidate"),
    )
    fake = SimpleNamespace(
        _library_media_bulk_delete_in_flight=False,
        _library_media_trash_browse_controller=controller,
        _library_media_view="trash",
        _library_media_trash_query_draft="old",
        _library_media_trash_input_error="old error",
        _library_media_trash_type_choices_visible=True,
        _library_media_trash_focus_identity="#old-focus",
        _library_media_trash_return=None,
        _apply_library_media_list_return=LibraryScreen._apply_library_media_list_return,
        call_next=lambda callback, receipt: events.append(
            ("return", callback, receipt)
        ),
    )
    fake._exit_library_media_trash = types.MethodType(
        LibraryScreen._exit_library_media_trash, fake
    )
    fake._library_media_return_candidate = types.MethodType(
        LibraryScreen._library_media_return_candidate, fake
    )
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_trash_back(fake, event)

    assert fake._library_media_view == "list"
    assert controller.state.requested_scope == MediaTrashScope()
    assert fake._library_media_trash_query_draft == ""
    assert fake._library_media_trash_input_error == ""
    assert fake._library_media_trash_type_choices_visible is False
    assert events[0] == "invalidate"
    assert events[1][0] == "return"
    assert events[1][1] == LibraryScreen._apply_library_media_list_return


def test_trash_restore_reads_resolved_selection_and_claims_shared_flag():
    """The restore worker joins the ONE bulk-delete interlock (PR-1473's
    one-flag rule): same flag, same exclusive worker group -- it is the
    fourth mutator of the shared records/counts state."""
    fake = _trash_view_fake(
        records=(
            {"id": "5", "title": "A", "type": "pdf"},
            {"id": "6", "title": "B", "type": "pdf"},
        ),
        total=2,
    )
    restore_calls = []

    async def _restore(media_id):
        restore_calls.append(media_id)

    fake._restore_library_media_from_trash = _restore
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_trash_restore(fake, event)

    assert fake._library_media_bulk_delete_in_flight is True
    assert len(fake._worker_calls) == 1
    coro, kwargs = fake._worker_calls[0]
    assert kwargs.get("exclusive") is True
    assert kwargs.get("group") == "library_media_bulk_delete"
    import asyncio

    asyncio.run(coro)
    # No explicit selection -> the state builder's first-row fallback,
    # exactly what the "▸" marker shows.
    assert len(restore_calls) == 1
    assert restore_calls[0].target.stable_id == "5"
    assert restore_calls[0].target.backing_media_id == 5


def test_trash_restore_refused_while_delete_or_undo_in_flight():
    fake = _trash_view_fake(
        records=({"id": "5", "title": "A", "type": "pdf"},),
        total=1,
        in_flight=True,
    )
    fake.run_worker = lambda coro, **k: pytest.fail("must not start a worker")
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_trash_restore(fake, event)

    assert fake._library_media_bulk_delete_in_flight is True


def test_trash_restore_noop_when_trash_empty():
    fake = _trash_view_fake(records=(), total=0)
    fake.run_worker = lambda coro, **k: pytest.fail("must not start a worker")
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_trash_restore(fake, event)

    assert fake._library_media_bulk_delete_in_flight is False


def test_media_trash_completion_flags_preserve_normal_media_presentation():
    """Trash-only completion never reorders or refreshes the retained Media page."""
    restore = _trash_view_fake(
        records=({"id": "5", "title": "A", "type": "pdf"},),
        total=1,
        in_flight=True,
    )
    restore._selected_media_id = "local:media:22"
    restore._library_media_focus_identity = "#library-media-row-1"
    restore._library_media_scroll_offset = (0, 7)
    retained_presentation = (
        restore._selected_media_id,
        restore._library_media_focus_identity,
        restore._library_media_scroll_offset,
    )

    LibraryScreen._complete_library_media_mutation(
        restore,
        committed=True,
        refresh_normal_media=False,
        stale_normal_media=True,
    )

    assert (
        restore._selected_media_id,
        restore._library_media_focus_identity,
        restore._library_media_scroll_offset,
    ) == retained_presentation
    assert restore._library_media_bulk_delete_in_flight is False
    assert ("mark-stale",) in restore._mutation_events
    assert not any(
        event[0] in {"reconcile", "request", "facets"}
        for event in restore._mutation_events
    )

    permanent = _trash_view_fake(
        records=({"id": "5", "title": "A", "type": "pdf"},),
        total=1,
        in_flight=True,
    )
    LibraryScreen._complete_library_media_mutation(
        permanent,
        committed=True,
        refresh_normal_media=False,
        stale_normal_media=False,
    )

    assert permanent._library_media_bulk_delete_in_flight is False
    assert not any(
        event[0] in {"mark-stale", "reconcile", "request", "facets"}
        for event in permanent._mutation_events
    )


def test_media_trash_delete_opener_is_consumed_and_only_opens_confirmation():
    fake = _trash_view_fake(
        records=({"id": "5", "title": "Same visible title", "type": "pdf"},),
        total=1,
        selected_id="5",
    )
    target = _trash_mutation_target(5, "Same visible title")
    calls = []
    fake._library_media_trash_browse_controller.open_delete_confirmation = lambda: (
        calls.append(("open", target.stable_id)) or target
    )
    stopped = []
    event = SimpleNamespace(stop=lambda: stopped.append(True))

    LibraryScreen.handle_library_media_trash_delete(fake, event)

    assert stopped == [True]
    assert calls == [("open", "5")]
    assert fake._worker_calls == []
    assert fake._library_media_bulk_delete_in_flight is False
    assert fake._library_media_trash_focus_identity == (
        "#library-media-trash-delete-cancel"
    )


def test_media_trash_permanent_confirm_double_press_schedules_one_shared_worker():
    fake = _trash_view_fake(
        records=({"id": "5", "title": "Captured", "type": "pdf"},),
        total=1,
        selected_id="5",
    )
    target = _trash_mutation_target(5, "Captured")
    fake._library_media_trash_browse_controller.state = dataclasses.replace(
        fake._library_media_trash_browse_controller.state,
        confirmation_target=target,
    )
    claims = []
    fake._library_media_trash_browse_controller.claim_mutation = lambda: (
        claims.append(target.stable_id) or _trash_mutation_claim(target)
    )

    async def delete_permanently(_target):
        return None

    fake._permanently_delete_library_media_from_trash = delete_permanently
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_trash_delete_confirm(fake, event)
    LibraryScreen.handle_library_media_trash_delete_confirm(fake, event)

    assert claims == ["5"]
    assert fake._library_media_bulk_delete_in_flight is True
    assert len(fake._worker_calls) == 1
    coroutine, worker_kwargs = fake._worker_calls[0]
    assert worker_kwargs == {
        "exclusive": True,
        "group": "library_media_bulk_delete",
    }
    coroutine.close()


@pytest.mark.asyncio
async def test_media_trash_permanent_delete_uses_only_scope_service_target_seam():
    target = MediaTrashMutationTarget(
        stable_id="local:media:41",
        backing_media_id=41,
        title="Duplicate visible title",
        media_type="document",
        trash_date="2026-08-30T00:00:00+00:00",
        page_index=3,
    )
    service_calls = []

    class ScopeService:
        async def permanently_delete_media_item(self, **kwargs):
            service_calls.append(("permanently_delete_media_item", kwargs))
            return {"ok": True, "media_id": kwargs["media_id"]}

        def __getattr__(self, name):
            if name in {
                "empty_media_trash",
                "permanently_delete_backing_media_item",
                "delete_media_item",
            }:
                pytest.fail(f"unexpected delete seam: {name}")
            raise AttributeError(name)

    events = []
    fake = SimpleNamespace(
        app_instance=SimpleNamespace(
            media_reading_scope_service=ScopeService(),
            notify=lambda *_args, **_kwargs: None,
        ),
        _library_media_trash_browse_controller=SimpleNamespace(
            finish_mutation_failure=lambda *args: events.append(("failure", args))
            or True,
            finish_mutation_commit=lambda *args: events.append(("commit", args))
            or True,
            request_after_mutation=lambda *_args, **kwargs: events.append(
                ("trash-request", kwargs)
            ),
        ),
        _library_media_bulk_delete_in_flight=True,
        _library_media_mutation_scope=MediaBrowseScope(page=2),
        _library_media_mutation_authority=7,
        _library_media_lifecycle_generation=7,
        _library_selected_row_id=LIBRARY_ROW_BROWSE_MEDIA,
        _library_media_browse_controller=SimpleNamespace(
            mutation_refresh_scope=MediaBrowseScope(page=2),
            mark_stale_after_trash_restore=lambda: events.append(("mark-stale",)),
            reconcile_committed_mutation=lambda **kwargs: events.append(
                ("reconcile", kwargs)
            ),
            request=lambda *args, **kwargs: events.append(
                ("media-request", args, kwargs)
            ),
            request_facets=lambda **kwargs: events.append(("facets", kwargs)),
        ),
        _sync_library_media_browse_state=lambda *_args: events.append(("sync",)),
        _run_library_service_call=LibraryScreen._run_library_service_call,
    )
    fake._complete_library_media_mutation = types.MethodType(
        LibraryScreen._complete_library_media_mutation, fake
    )

    claim = _trash_mutation_claim(target)
    await LibraryScreen._permanently_delete_library_media_from_trash(fake, claim)

    assert service_calls == [
        (
            "permanently_delete_media_item",
            {"mode": "local", "media_id": 41},
        )
    ]
    assert (
        "commit",
        (claim, "Deleted 'Duplicate visible title' permanently."),
    ) in events
    assert ("trash-request", {"focus_identity": "#library-media-trash-row-3"}) in events
    assert not any(
        event[0] in {"mark-stale", "reconcile", "media-request", "facets"}
        for event in events
    )
    assert fake._library_media_bulk_delete_in_flight is False


@pytest.mark.asyncio
async def test_media_trash_permanent_failure_keeps_fresh_row_and_skips_refresh():
    scope = MediaTrashScope()
    item = _canonical_trash_items(1)[0]

    # Use the real controller reducer boundary so row/selection/total authority
    # is exercised rather than represented by a permissive callback fake.
    class ControllerScreen:
        pending = []

        def run_worker(self, work, **_kwargs):
            self.pending.append(work)
            return work

    class ListService:
        async def list_library_media_trash(self, **_kwargs):
            return {
                "items": [item],
                "total": 1,
                "limit": 20,
                "offset": 0,
                "types": [str(item["media_type"])],
            }

    async def call_service(fn, **kwargs):
        assert kwargs.pop("isolate_in_worker") is True
        return await fn(**kwargs)

    controller_screen = ControllerScreen()
    controller = LibraryMediaTrashBrowseController(
        screen=controller_screen,
        run_service_call=lambda: call_service,
        media_service=lambda: ListService(),
        sync_view=lambda: lambda _focus: None,
        request_is_active=lambda: True,
    )
    controller.request(scope, origin="entry", focus_identity=None)
    await controller_screen.pending.pop()
    claim = controller.claim_mutation()
    assert claim is not None
    target = claim.target

    class FailingScopeService:
        async def permanently_delete_media_item(self, **_kwargs):
            raise PermissionError("private policy detail")

    events = []
    fake = SimpleNamespace(
        app_instance=SimpleNamespace(
            media_reading_scope_service=FailingScopeService(),
            notify=lambda message, **_kwargs: events.append(("notify", message)),
        ),
        _library_media_trash_browse_controller=controller,
        _library_media_bulk_delete_in_flight=True,
        _library_media_mutation_scope=MediaBrowseScope(),
        _library_media_mutation_authority=2,
        _library_media_lifecycle_generation=2,
        _library_selected_row_id=LIBRARY_ROW_BROWSE_MEDIA,
        _library_media_browse_controller=SimpleNamespace(
            mutation_refresh_scope=MediaBrowseScope(),
            mark_stale_after_trash_restore=lambda: events.append(("mark-stale",)),
            reconcile_committed_mutation=lambda **kwargs: events.append(
                ("reconcile", kwargs)
            ),
            request=lambda *args, **kwargs: events.append(
                ("media-request", args, kwargs)
            ),
            request_facets=lambda **kwargs: events.append(("facets", kwargs)),
        ),
        _sync_library_media_browse_state=lambda *_args: events.append(("sync",)),
        _run_library_service_call=LibraryScreen._run_library_service_call,
        _notify_library_media_delete_warning=lambda message: events.append(
            ("warning", message)
        ),
    )
    fake._complete_library_media_mutation = types.MethodType(
        LibraryScreen._complete_library_media_mutation, fake
    )

    await LibraryScreen._permanently_delete_library_media_from_trash(fake, claim)

    assert controller.state.retained_items == (item,)
    assert controller.state.selected_id == target.stable_id
    assert controller.state.freshness == "fresh"
    assert controller.state.applied_result is not None
    assert controller.state.applied_result.total == 1
    assert (
        controller.state.error_copy == "Could not delete this media item permanently."
    )
    assert not any(
        event[0]
        in {"trash-request", "media-request", "facets", "reconcile", "mark-stale"}
        for event in events
    )
    assert fake._library_media_bulk_delete_in_flight is False


def test_escape_gate_only_passes_in_trash_view():
    """check_action: ``library_media_trash_back`` fires only while the
    media canvas genuinely shows its Trash view; the viewer's own gate is
    False there (disjoint-gate contract of the escape chain)."""
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_ROW_BROWSE_MEDIA,
    )

    in_trash = SimpleNamespace(
        _library_selected_row_id=LIBRARY_ROW_BROWSE_MEDIA,
        _library_media_view="trash",
        _library_media_bulk_delete_in_flight=False,
    )
    assert LibraryScreen.check_action(in_trash, "library_media_trash_back", ()) is True
    assert (
        LibraryScreen.check_action(in_trash, "library_media_viewer_back", ()) is False
    )

    in_list = SimpleNamespace(
        _library_selected_row_id=LIBRARY_ROW_BROWSE_MEDIA,
        _library_media_view="list",
    )
    assert LibraryScreen.check_action(in_list, "library_media_trash_back", ()) is False

    other_row = SimpleNamespace(
        _library_selected_row_id="browse-notes",
        _library_media_view="trash",
    )
    assert (
        LibraryScreen.check_action(other_row, "library_media_trash_back", ()) is False
    )


# ---------------------------------------------------------------------------
# The restore seam against a REAL (file-backed) MediaDatabase -- never
# :memory: (the worker hops threads via ``isolate_in_worker=True``; a fresh
# :memory: connection on a new thread is a distinct, empty database).
# ---------------------------------------------------------------------------


def _restore_fake(*, db, trash_records, media_records, media_count):
    local_service = LocalMediaReadingService(db)
    scope_service = MediaReadingScopeService(local_service, None)
    notified = []
    refresh_calls = []
    after_refresh = []
    trash_controller_events = []
    trash_controller = SimpleNamespace(
        finish_mutation_failure=lambda claim, copy: trash_controller_events.append(
            ("failure", claim.target, copy)
        )
        or True,
        finish_mutation_commit=lambda claim, notice: trash_controller_events.append(
            ("commit", claim.target, notice)
        )
        or True,
        request_after_mutation=lambda *_args, **kwargs: trash_controller_events.append(
            ("request", kwargs)
        ),
    )
    fake = SimpleNamespace(
        app_instance=SimpleNamespace(
            media_reading_scope_service=scope_service,
            notify=lambda msg, **k: notified.append((msg, k)),
        ),
        _notified=notified,
        _refresh_calls=refresh_calls,
        _after_refresh=after_refresh,
        _trash_controller_events=trash_controller_events,
        _library_media_trash_browse_controller=trash_controller,
        _local_source_records={"media": tuple(media_records)},
        _local_source_counts={"media": media_count},
        # The real caller (``handle_library_media_trash_restore``) claims
        # the shared flag BEFORE scheduling this coroutine.
        _library_media_bulk_delete_in_flight=True,
        is_mounted=True,
        refresh=lambda **k: refresh_calls.append(k),
        call_after_refresh=lambda fn, *a: after_refresh.append(fn),
        _focus_library_media_trash_entry=lambda: None,
        _run_library_service_call=LibraryScreen._run_library_service_call,
        _source_record_id=LibraryScreen._source_record_id,
    )
    fake._notify_library_media_delete_warning = types.MethodType(
        LibraryScreen._notify_library_media_delete_warning, fake
    )
    return _bind_trash_mutation_seams(fake)


def _trash_mutation_target(media_id: int | str, title: str) -> MediaTrashMutationTarget:
    return MediaTrashMutationTarget(
        stable_id=str(media_id),
        backing_media_id=int(media_id),
        title=title,
        media_type="document",
        trash_date=None,
        page_index=0,
    )


def _trash_mutation_claim(
    target: MediaTrashMutationTarget, generation: int = 1
) -> MediaTrashMutationClaim:
    return MediaTrashMutationClaim(target=target, generation=generation)


@pytest.mark.asyncio
async def test_restore_via_real_db_moves_item_back_and_updates_counts(tmp_path):
    """AC#2: restore flips ``is_trash`` back through the existing seam
    (``restore_media_item`` -> ``restore_from_trash``, never raw SQL). The
    broad rail snapshot may update, but the exact normal-Media page is retained
    in place and marked stale rather than receiving an unranked row."""
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-4025-restore"
    )
    keep_id, _, _ = db.add_media_with_keywords(
        title="Keep", content="keep", media_type="article", keywords=[]
    )
    trashed_id, _, _ = db.add_media_with_keywords(
        title="Trashed Doc", content="body", media_type="document", keywords=[]
    )
    assert db.mark_as_trash(trashed_id) is True

    fake = _restore_fake(
        db=db,
        trash_records=(
            {"id": str(trashed_id), "title": "Trashed Doc", "type": "document"},
        ),
        media_records=({"id": str(keep_id), "title": "Keep"},),
        media_count=1,
    )

    target = _trash_mutation_target(trashed_id, "Trashed Doc")
    await LibraryScreen._restore_library_media_from_trash(
        fake, _trash_mutation_claim(target)
    )

    row = db.get_media_by_id(trashed_id)
    assert row is not None
    assert not row["is_trash"]

    restored_ids = {
        LibraryScreen._source_record_id(r) for r in fake._local_source_records["media"]
    }
    assert str(trashed_id) in restored_ids
    assert fake._local_source_counts["media"] == 2
    assert ("commit", target, "Restored 'Trashed Doc'.") in (
        fake._trash_controller_events
    )
    assert any(event[0] == "request" for event in fake._trash_controller_events)
    assert fake._notified == []
    assert fake._refresh_calls == [{"recompose": True}]
    assert fake._library_media_bulk_delete_in_flight is False
    assert fake._mutation_events[0] == ("begin",)
    assert ("mark-stale",) in fake._mutation_events
    assert not any(event[0] == "reconcile" for event in fake._mutation_events)
    assert not any(event[0] == "request" for event in fake._mutation_events)
    assert not any(event[0] == "facets" for event in fake._mutation_events)

    db.close_connection()


@pytest.mark.asyncio
async def test_restore_via_real_db_chunked_item_keeps_chunks_and_url(tmp_path):
    """Chunked variant (the batch's real-DB requirement) + the
    url-canonicalization decision pin: Trash-surface restore is an
    ``is_trash`` flag flip on the existing row -- it must leave the stored
    chunks AND the row's url byte-identical (task-4026's one-directional
    url-canonicalization edge lives only in ``add_media_with_keywords``'s
    restore-by-re-import path, which this surface never takes)."""
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-4025-restore-chunked"
    )
    url = "https://example.com/task-4025/chunked.txt"
    media_id, _, _ = db.add_media_with_keywords(
        title="Chunked Doc",
        content="chunked content",
        media_type="document",
        keywords=[],
        url=url,
        chunks=[
            {"text": "chunk one", "chunk_type": "text"},
            {"text": "chunk two", "chunk_type": "text"},
        ],
    )
    assert db.mark_as_trash(media_id) is True

    def _chunk_texts():
        return {
            r["chunk_text"]
            for r in db.execute_query(
                "SELECT chunk_text FROM UnvectorizedMediaChunks "
                "WHERE media_id = ? AND deleted = 0",
                (media_id,),
            ).fetchall()
        }

    assert _chunk_texts() == {"chunk one", "chunk two"}

    fake = _restore_fake(
        db=db,
        trash_records=(
            {"id": str(media_id), "title": "Chunked Doc", "type": "document"},
        ),
        media_records=(),
        media_count=0,
    )

    await LibraryScreen._restore_library_media_from_trash(
        fake, _trash_mutation_claim(_trash_mutation_target(media_id, "Chunked Doc"))
    )

    row = db.get_media_by_id(media_id)
    assert row is not None
    assert not row["is_trash"]
    assert row["url"] == url
    assert row["content"] == "chunked content"
    assert _chunk_texts() == {"chunk one", "chunk two"}
    assert fake._local_source_counts["media"] == 1

    db.close_connection()


@pytest.mark.asyncio
async def test_restore_failure_warns_keeps_row_and_clears_flag(tmp_path):
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-4025-restore-fail"
    )
    trash_records = ({"id": "999999", "title": "Ghost", "type": "document"},)
    fake = _restore_fake(
        db=db,
        trash_records=trash_records,
        media_records=(),
        media_count=0,
    )

    target = _trash_mutation_target(999999, "Ghost")
    await LibraryScreen._restore_library_media_from_trash(
        fake, _trash_mutation_claim(target)
    )

    assert fake._local_source_records["media"] == ()
    assert fake._local_source_counts["media"] == 0
    assert len(fake._notified) == 1
    assert "Could not restore" in fake._notified[0][0]
    assert ("failure", target, "Could not restore this media item.") in (
        fake._trash_controller_events
    )
    assert fake._library_media_bulk_delete_in_flight is False

    db.close_connection()


@pytest.mark.asyncio
async def test_trash_controller_fetches_page_through_real_seam(tmp_path):
    """AC#1: the fetch worker lists every ``is_trash=1`` item through the
    existing ``list_media_trash`` seam and maps items + total into the
    view's session state (trash_date included, for the "trashed <age>"
    secondary)."""
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="task-4025-load")
    active_id, _, _ = db.add_media_with_keywords(
        title="Active", content="a", media_type="article", keywords=[]
    )
    trashed_a, _, _ = db.add_media_with_keywords(
        title="Trash A", content="b", media_type="document", keywords=[]
    )
    trashed_b, _, _ = db.add_media_with_keywords(
        title="Trash B", content="c", media_type="video", keywords=[]
    )
    assert db.mark_as_trash(trashed_a) is True
    assert db.mark_as_trash(trashed_b) is True

    scope_service = MediaReadingScopeService(LocalMediaReadingService(db), None)
    controller = LibraryMediaTrashBrowseController(
        screen=SimpleNamespace(run_worker=lambda *_args, **_kwargs: None),
        run_service_call=lambda: LibraryScreen._run_library_service_call,
        media_service=lambda: scope_service,
        sync_view=lambda: lambda _focus: None,
        request_is_active=lambda: True,
    )
    result = await controller._list(MediaTrashScope())

    listed_ids = {int(r["backing_media_id"]) for r in result.items}
    assert listed_ids == {trashed_a, trashed_b}
    assert active_id not in listed_ids
    assert result.total == 2
    assert all(r.get("trash_date") for r in result.items)

    db.close_connection()


# ---------------------------------------------------------------------------
# The FTS decision (AC#4's explicit question), pinned against a real DB:
# trashed items are EXCLUDED from search results. The FTS5 index itself
# retains the trashed row's content (``mark_as_trash`` documents "does not
# affect FTS" -- an index-level property that makes restore instant), but
# every reachable query path filters ``is_trash = 0`` at query time:
# ``search_media_db`` (the Library keyword-search seam AND the RAG
# ``search_media_fts5`` leg both route through it with the default
# ``include_trash=False``) and ``rag_service``'s direct media_fts query
# (hard-coded ``m.is_trash = 0``). These tests pin both directions so the
# exclusion can never silently regress.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trashed_item_excluded_from_search_until_restored(tmp_path):
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="task-4025-fts")
    active_id, _, _ = db.add_media_with_keywords(
        title="Active zebra notes",
        content="the zebra grazes quietly",
        media_type="document",
        keywords=[],
    )
    trashed_id, _, _ = db.add_media_with_keywords(
        title="Trashed zebra notes",
        content="the zebra sleeps soundly",
        media_type="document",
        keywords=[],
    )
    assert db.mark_as_trash(trashed_id) is True
    service = LocalMediaReadingService(db)

    # The Library keyword-search seam (library_local_rag_search_service ->
    # scope service -> this): the trashed item must not surface.
    hits = service.search_media(query="zebra", limit=10)
    hit_ids = {int(item["id"]) for item in hits["items"]}
    assert hit_ids == {active_id}

    # The DB layer's own default, which the RAG fts5 leg also uses.
    rows, total = db.search_media_db(search_query="zebra")
    assert {int(r["id"]) for r in rows} == {active_id}
    assert total == 1

    # Symmetric: restore brings it straight back into search results (the
    # index never dropped it, so no re-index is needed).
    assert db.restore_from_trash(trashed_id) is True
    hits = service.search_media(query="zebra", limit=10)
    hit_ids = {int(item["id"]) for item in hits["items"]}
    assert hit_ids == {active_id, trashed_id}

    db.close_connection()
