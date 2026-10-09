"""B22: the chatbooks grid renders one bounded page at a time.

Mounting one widget subtree per matching chatbook made a 150-chatbook
search application hundreds of mounts in one synchronous rebuild. The
grid now mounts at most ``CHATBOOK_RENDER_PAGE_SIZE`` (60) cards/list
items per render, with a "Load more" control for the rest. Search and
filter semantics are unchanged (the full match set is still computed by
``_filter_chatbooks``), and paging resets whenever the match set is
re-rendered from scratch (filter/search/view/data changes).
"""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, ListItem, ListView

# The autouse Tests/UI catalog-refresh fixture lazily imports
# `tldw_chatbook.app`; under the per-test config redirect that first import
# fails the raw-source admission (RecoveryRequired). Importing it here, at
# collection time while the bootstrap config is still bound, is the repo's
# established standalone-run pattern (see Tests/UI/conftest.py's notes).
import tldw_chatbook.app  # noqa: F401,E402
from tldw_chatbook.UI.Chatbooks_Window_Improved import (
    CHATBOOK_RENDER_PAGE_SIZE,
    ChatbooksWindowImproved,
)
from Tests.UI.consolidated_css import ConsolidatedCSSApp

pytestmark = pytest.mark.bootstrap_profile


def _chatbooks(n: int) -> list[dict]:
    return [
        {
            "name": f"Chatbook {i:03d}",
            "description": f"description {i}",
            "tags": [],
            "path": f"/tmp/cb-{i}",
            "size_mb": 1.0,
        }
        for i in range(n)
    ]


class ChatbooksHostApp(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield from ()


async def _mounted_window(
    app: ChatbooksHostApp, pilot, n: int
) -> ChatbooksWindowImproved:
    """Mount the window with its scan worker serving the fixture.

    The on_mount scan worker loads the fixture itself: assigning
    ``chatbooks`` after the mount races the worker's own (empty) result.
    """
    window = ChatbooksWindowImproved(app)
    window._scan_chatbooks = lambda: _chatbooks(n)  # type: ignore[method-assign]
    await app.mount(window)
    await pilot.pause()
    await pilot.pause()
    return window


def _card_count(window: ChatbooksWindowImproved) -> int:
    return len(window.query("ChatbookCard"))


def _load_more_visible(window: ChatbooksWindowImproved) -> bool:
    """Whether the Load more control is currently shown."""
    try:
        return bool(window.query_one("#chatbooks-load-more", Button).display)
    except Exception:
        return False


async def _advance_page(pilot, window: ChatbooksWindowImproved) -> None:
    window.query_one("#chatbooks-load-more", Button).press()
    await pilot.pause()


@pytest.mark.asyncio
async def test_first_render_mounts_one_page_and_a_load_more_control():
    app = ChatbooksHostApp()
    async with app.run_test() as pilot:
        window = await _mounted_window(app, pilot, 150)
        await pilot.pause()

        assert CHATBOOK_RENDER_PAGE_SIZE == 60
        assert _card_count(window) == CHATBOOK_RENDER_PAGE_SIZE, (
            f"expected one page of cards, got {_card_count(window)}"
        )
        assert _load_more_visible(window), "expected a visible Load more control"

        # The full match set is intact in state; only the render is bounded.
        assert len(window._filter_chatbooks()) == 150


@pytest.mark.asyncio
async def test_load_more_mounts_next_page_then_disappears_at_the_end():
    app = ChatbooksHostApp()
    async with app.run_test() as pilot:
        window = await _mounted_window(app, pilot, 150)
        await pilot.pause()

        await _advance_page(pilot, window)

        assert _card_count(window) == 2 * CHATBOOK_RENDER_PAGE_SIZE
        assert _load_more_visible(window)

        await _advance_page(pilot, window)

        # Final page: everything is mounted and the control is hidden.
        assert _card_count(window) == 150
        assert not _load_more_visible(window)


@pytest.mark.asyncio
async def test_filter_change_resets_paging_to_first_page():
    app = ChatbooksHostApp()
    async with app.run_test() as pilot:
        window = await _mounted_window(app, pilot, 150)
        await pilot.pause()

        await _advance_page(pilot, window)
        assert _card_count(window) == 2 * CHATBOOK_RENDER_PAGE_SIZE

        # A new search re-runs the (unchanged) filter over the full match
        # set and renders its first page only. Names are zero-padded, so
        # "Chatbook 1" matches exactly 100-149 (50 items) -- which fits in
        # one page.
        window.search_query = "Chatbook 1"
        await pilot.pause()

        matched = len(window._filter_chatbooks())
        assert matched == 50
        assert _card_count(window) == 50
        assert not _load_more_visible(window)


@pytest.mark.asyncio
async def test_fitting_match_set_mounts_everything_without_a_load_more():
    app = ChatbooksHostApp()
    async with app.run_test() as pilot:
        window = await _mounted_window(app, pilot, 45)
        await pilot.pause()

        assert _card_count(window) == 45
        assert not _load_more_visible(window)


@pytest.mark.asyncio
async def test_list_mode_is_capped_the_same_way():
    app = ChatbooksHostApp()
    async with app.run_test() as pilot:
        window = await _mounted_window(app, pilot, 150)
        await pilot.pause()

        window.view_mode = "list"
        await pilot.pause()

        list_view = window.query_one(".chatbooks-list", ListView)
        assert len(list_view.children) == CHATBOOK_RENDER_PAGE_SIZE
        items = list_view.children
        assert all(isinstance(item, ListItem) for item in items)
        assert _load_more_visible(window)

        await _advance_page(pilot, window)
        list_view = window.query_one(".chatbooks-list", ListView)
        assert len(list_view.children) == 2 * CHATBOOK_RENDER_PAGE_SIZE


@pytest.mark.asyncio
async def test_zero_match_search_hides_previous_load_more_control():
    app = ChatbooksHostApp()
    async with app.run_test() as pilot:
        window = await _mounted_window(app, pilot, 150)
        assert _load_more_visible(window)

        window.search_query = "no-chatbook-matches-this"
        await pilot.pause()

        assert window._filter_chatbooks() == []
        assert _card_count(window) == 0
        assert not _load_more_visible(window)
