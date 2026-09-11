"""Critique #10: the Collections and Trash pagers, name and empty state.

Covers task-32352 (one name, a true empty state, a disabled pager at
``0–0 of 0``) and the Collections/Trash half of task-32354 (every Library
pager goes through ``library_pager_layout``'s single-page rule).

The Skills half of task-32354 lives in ``test_library_skills_canvas.py``
beside the pin it reverses.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_library_collections_capture_reader import (
    AUTHORITY,
    _capabilities,
)
from Tests.UI.test_library_media_trash import (
    _TrashCanvasApp,
    _fresh_trash_pager,
    _trash_state,
)
from tldw_chatbook.Library.collections_capture_models import (
    CAPTURE_PAGE_SIZE,
    CaptureIdentity,
    CapturePage,
    CapturePageRequest,
    CaptureSummary,
)
from tldw_chatbook.UI.Library_Modules.library_collections_capture_controller import (
    CollectionsCaptureControllerState,
)
from tldw_chatbook.Widgets.Library.library_collections_capture_reader import (
    CollectionsCaptureReaderPresentation,
    LibraryCollectionsItemsPane,
)


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


class _ItemsApp(ConsolidatedCSSApp):
    """Mounts the Collections Items pane alone -- the surface under test."""

    def __init__(self, presentation: CollectionsCaptureReaderPresentation) -> None:
        super().__init__()
        self.presentation = presentation

    def compose(self):
        yield LibraryCollectionsItemsPane(self.presentation, id="items")


def _capture(index: int) -> CaptureSummary:
    return CaptureSummary(
        identity=CaptureIdentity(AUTHORITY, f"capture-{index:03d}"),
        canonical_url=f"https://example.com/{index}",
        title=f"Capture {index}",
        domain="example.com",
        summary="A compact summary.",
        published_at="2026-08-30T12:00:00Z",
        status="reading",
        favorite=False,
        tags=(),
        processing_state="ready",
        created_at="2026-08-31T12:00:00Z",
        updated_at="2026-08-31T12:00:00Z",
    )


def _collections_app(
    *, total: int, rows: int = 0, stale: bool = False, **scope
) -> _ItemsApp:
    """Build an Items pane over a page of ``rows`` items out of ``total``."""
    request = CapturePageRequest(AUTHORITY, **scope)
    page = CapturePage(request, tuple(_capture(index) for index in range(rows)), total)
    state = CollectionsCaptureControllerState(
        authority_key=AUTHORITY,
        requested_scope=request,
        applied_scope=request,
        page=page,
        page_stale=stale,
    )
    return _ItemsApp(
        CollectionsCaptureReaderPresentation(
            state=state,
            capabilities=_capabilities("browse", "capture"),
        )
    )


@pytest.mark.asyncio
async def test_collections_canvas_is_named_collections_not_quick_capture():
    """task-32352 AC#1: one name in the rail and on the canvas.

    The rail row says "Collections (N)"; the canvas led with the Quick
    Capture button and no heading at all, so the first painted line read
    as the canvas title. "Quick Capture" survives only on the button that
    saves a URL -- it is a verb, not a place.
    """
    app = _collections_app(total=0)
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        heading = app.query_one("#library-collections-header", Static)
        assert str(heading.renderable) == "Collections"
        titled = [
            widget
            for widget in app.query(Static)
            if str(getattr(widget, "renderable", "")).strip() == "Quick Capture"
        ]
        assert not titled, "Quick Capture is a button label, not a title"
        assert "Quick Capture" in str(
            app.query_one("#library-collections-quick-capture", Button).label
        )


@pytest.mark.asyncio
async def test_collections_empty_state_never_blames_an_unset_filter():
    """task-32352 AC#2: a profile that never captured anything.

    One sentence used to cover both cases, so an empty profile was told
    to clear filters it had never set and pointed at Quick Capture as if
    it were somewhere else on the screen.
    """
    app = _collections_app(total=0)
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        empty = app.query_one("#library-collections-items-empty", Static)
        assert str(empty.renderable) == (
            "No saved captures yet · press Quick Capture above to save a page by URL."
        )


@pytest.mark.asyncio
async def test_collections_empty_state_names_filters_only_when_one_is_set():
    """task-32352 AC#2, the other half: a filter IS set, so say so."""
    app = _collections_app(total=0, search="nothing matches this")
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        empty = app.query_one("#library-collections-items-empty", Static)
        assert str(empty.renderable) == (
            "No captures match these filters · clear them to see everything saved."
        )


@pytest.mark.parametrize(
    "scope",
    ({"favorite": True}, {"statuses": ("archived",)}),
    ids=("favorites", "archived"),
)
@pytest.mark.asyncio
async def test_collections_empty_state_names_an_empty_rail_scope(scope):
    """task-32352 AC#2, the third case: the rail's own scope narrowed it.

    The rail's scope rows set ``statuses``/``favorite`` rather than any of
    the filter-form fields, so an empty Favorites beside a rail reading
    "Collections (57)" would otherwise claim nothing was ever saved and
    point at an action that does not leave the scope.
    """
    app = _collections_app(total=0, **scope)
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        empty = app.query_one("#library-collections-items-empty", Static)
        assert str(empty.renderable) == (
            "Nothing in this scope yet · choose All Captures in the rail to see "
            "everything saved."
        )


@pytest.mark.asyncio
async def test_collections_empty_state_prefers_the_filter_copy_inside_a_scope():
    """A filter set inside a scope is the thing the reader can clear."""
    app = _collections_app(total=0, favorite=True, search="nothing matches this")
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        empty = app.query_one("#library-collections-items-empty", Static)
        assert str(empty.renderable) == (
            "No captures match these filters · clear them to see everything saved."
        )


@pytest.mark.asyncio
async def test_collections_shows_no_pager_chrome_at_zero_of_zero():
    """task-32352 AC#3 + task-32354 AC#1 for this canvas.

    Previous/Next rendered enabled-looking at "0–0 of 0" while the
    identical Trash pager rendered disabled (B D6 cap 52). There is
    nowhere to page to, so the shared rule drops the controls entirely.
    """
    app = _collections_app(total=0)
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        assert not app.query("#library-collections-page-toolbar")
        assert str(
            app.query_one("#library-collections-page-range", Static).renderable
        ) == "0–0 of 0"


@pytest.mark.asyncio
async def test_collections_keeps_paused_controls_when_the_page_is_stale():
    """The suppression's precondition: nowhere to page to, not paging paused.

    A stale page withholds totals, so both directions are unavailable --
    but that is "we cannot page right now", not "this is the only page".
    The controls stay, disabled, with the reason the old pager gave
    (pinned by ``test_items_keep_capture_controls_rows_and_stale_recovery_
    reachable``).
    """
    app = _collections_app(total=40, rows=CAPTURE_PAGE_SIZE, stale=True)
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        previous = app.query_one("#library-collections-page-previous", Button)
        assert previous.disabled
        assert previous.tooltip == "No current previous page is available."
        assert str(
            app.query_one("#library-collections-page-range", Static).renderable
        ) == "Page 1 · total unavailable"


@pytest.mark.asyncio
async def test_collections_keeps_the_pager_when_a_second_page_exists():
    """The suppression is the one-page case only -- two pages keep both
    controls, the page copy, and the ``○`` marker every other Library
    pager uses for a disabled one."""
    app = _collections_app(total=CAPTURE_PAGE_SIZE + 1, rows=CAPTURE_PAGE_SIZE)
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        assert app.query("#library-collections-page-toolbar")
        assert str(
            app.query_one("#library-collections-page-range", Static).renderable
        ) == f"1–{CAPTURE_PAGE_SIZE} of {CAPTURE_PAGE_SIZE + 1} · Page 1 of 2"
        previous = app.query_one("#library-collections-page-previous", Button)
        assert previous.disabled
        assert str(previous.label).startswith("○ ")
        assert not app.query_one(
            "#library-collections-page-next", Button
        ).disabled


# ---------------------------------------------------------------------------
# Trash
# ---------------------------------------------------------------------------


def _trash_app(*, total: int, rows: int) -> _TrashCanvasApp:
    """Reuse the Trash suite's own canvas host and record shape."""
    records = [
        {
            "id": str(index),
            "title": f"Trashed {index}",
            "type": "audio",
            "trash_date": "2026-09-01T00:00:00+00:00",
        }
        for index in range(rows)
    ]
    return _TrashCanvasApp(
        _trash_state(records=records, total=total, selected_id=""),
        pager=_fresh_trash_pager(total=total, rows=rows),
    )


@pytest.mark.asyncio
async def test_trash_shows_no_pager_controls_on_a_single_page():
    """task-32354 AC#1 for Trash, at the width where the rows matter."""
    app = _trash_app(total=1, rows=1)
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause()
        assert not app.query("#library-media-trash-pager-controls")
        assert not app.query("#library-media-trash-page")
        assert str(
            app.query_one("#library-media-trash-range", Static).renderable
        ) == "1-1 of 1"
        pager = app.query_one("#library-media-trash-pager")
        assert pager.styles.height.value == 1


@pytest.mark.asyncio
async def test_trash_keeps_its_two_row_pager_when_a_second_page_exists():
    app = _trash_app(total=45, rows=20)
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause()
        assert app.query("#library-media-trash-pager-controls")
        assert str(
            app.query_one("#library-media-trash-range", Static).renderable
        ) == "1-20 of 45"
        assert str(
            app.query_one("#library-media-trash-page", Static).renderable
        ) == "Page 1 of 3"
        pager = app.query_one("#library-media-trash-pager")
        assert pager.styles.height.value == 2
