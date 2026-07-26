import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
    TreeScope,
    TreeScopeChanged,
    WatchlistTree,
)


def _tree_data():
    return {
        "watchlists": [
            {"id": 1, "name": "Morning AI Brief", "tags": ["ai", "daily"]},
            {"id": 2, "name": "Security", "tags": ["sec"]},
        ],
        "counts": {
            # total != unread everywhere, deliberately: a swapped-field bug
            # (displaying total where unread belongs, or vice versa) must
            # make some assertion fail.
            1: {"total": 30, "unread": 24},
            2: {"total": 8, "unread": 3},
            -1: {"total": 4, "unread": 1},
            -2: {"total": 52, "unread": 37},
        },
    }


class _TreeApp(App):
    def __init__(self, data, source_rows=None):
        super().__init__()
        self._data = data
        self._source_rows = source_rows or {}
        self.scopes: list[TreeScope] = []

    def compose(self) -> ComposeResult:
        yield WatchlistTree(
            watchlists=self._data["watchlists"],
            counts=self._data["counts"],
            source_rows_loader=lambda wid: self._source_rows.get(wid, []),
            id="wl-tree",
        )

    def on_tree_scope_changed(self, message: TreeScopeChanged) -> None:
        self.scopes.append(message.scope)


@pytest.mark.asyncio
async def test_permanent_roots_always_render():
    app = _TreeApp(_tree_data())
    async with app.run_test():
        assert app.query("#wl-tree-node-all")
        assert app.query("#wl-tree-node-unassigned")


@pytest.mark.asyncio
async def test_roots_render_even_with_no_watchlists():
    # First run: the folder migration is effectively a no-op, so there are none.
    app = _TreeApp({"watchlists": [], "counts": {-1: {"total": 0, "unread": 0},
                                                 -2: {"total": 0, "unread": 0}}})
    async with app.run_test():
        assert app.query("#wl-tree-node-all")
        assert app.query("#wl-tree-node-unassigned")


@pytest.mark.asyncio
async def test_watchlists_render_with_their_unread_counts():
    app = _TreeApp(_tree_data())
    async with app.run_test():
        # Textual 8.2.7: Button subclasses Widget directly (not Static) and
        # exposes its text via the `label` reactive, not `.renderable` — so
        # node text lives on Button widgets, matching this repo's own
        # convention (str(button.label)) elsewhere in the test suite.
        text = " ".join(str(n.label) for n in app.query(Button))
        assert "Morning AI Brief" in text
        assert "24" in text
        assert "Security" in text


@pytest.mark.asyncio
async def test_permanent_roots_render_their_unread_counts():
    app = _TreeApp(_tree_data())
    async with app.run_test():
        all_button = app.query_one("#wl-tree-node-all", Button)
        unassigned_button = app.query_one("#wl-tree-node-unassigned", Button)
        # Exact match, not a substring check: -2's total (52) and -1's total
        # (4) must not leak in where unread (37 / 1) belongs.
        assert str(all_button.label) == "All sources  37"
        assert str(unassigned_button.label) == "Unassigned  1"


@pytest.mark.asyncio
async def test_selecting_a_watchlist_posts_its_scope():
    app = _TreeApp(_tree_data())
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-node-watchlist-1")
        await pilot.pause()
        assert app.scopes[-1] == TreeScope(kind="watchlist", watchlist_id=1, source_id=None)


@pytest.mark.asyncio
async def test_selecting_all_sources_posts_the_all_scope():
    app = _TreeApp(_tree_data())
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-node-all")
        await pilot.pause()
        assert app.scopes[-1].kind == "all"


@pytest.mark.asyncio
async def test_sources_load_only_when_a_watchlist_is_expanded():
    calls: list[int] = []

    class _App(_TreeApp):
        def compose(self) -> ComposeResult:
            def loader(wid):
                calls.append(wid)
                return [{"id": 10, "name": "ArXiv: AI", "type": "rss"}]

            yield WatchlistTree(
                watchlists=self._data["watchlists"],
                counts=self._data["counts"],
                source_rows_loader=loader,
                id="wl-tree",
            )

    app = _App(_tree_data())
    async with app.run_test() as pilot:
        assert calls == [], "no watchlist is expanded yet, so nothing should have loaded"
        await pilot.click("#wl-tree-expand-1")
        await pilot.pause()
        assert calls == [1]
        # Id is qualified by watchlist (fix-round 1, finding 2): a source
        # may belong to more than one watchlist, so "source 10" alone is
        # not a unique id.
        assert app.query("#wl-tree-node-source-1-10")


@pytest.mark.asyncio
async def test_re_expanding_a_watchlist_does_not_reload_its_sources():
    calls: list[int] = []

    class _App(_TreeApp):
        def compose(self) -> ComposeResult:
            def loader(wid):
                calls.append(wid)
                return [{"id": 10, "name": "ArXiv: AI", "type": "rss"}]

            yield WatchlistTree(
                watchlists=self._data["watchlists"],
                counts=self._data["counts"],
                source_rows_loader=loader,
                id="wl-tree",
            )

    app = _App(_tree_data())
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-expand-1")  # expand
        await pilot.pause()
        await pilot.click("#wl-tree-expand-1")  # collapse
        await pilot.pause()
        await pilot.click("#wl-tree-expand-1")  # re-expand
        await pilot.pause()
        assert calls == [1], "the cache must serve the second expand, not refetch"


@pytest.mark.asyncio
async def test_selecting_a_source_posts_a_source_scope():
    app = _TreeApp(_tree_data(), source_rows={1: [{"id": 10, "name": "ArXiv: AI", "type": "rss"}]})
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-expand-1")
        await pilot.pause()
        await pilot.click("#wl-tree-node-source-1-10")
        await pilot.pause()
        assert app.scopes[-1] == TreeScope(kind="source", watchlist_id=1, source_id=10)


@pytest.mark.asyncio
async def test_tag_filter_narrows_which_watchlists_render():
    app = _TreeApp(_tree_data())
    async with app.run_test() as pilot:
        # Tags are ids by position, not by text: watchlist 1's tags are
        # ["ai", "daily"] (indices 0, 1) and watchlist 2's tag "sec" is
        # index 2, in first-seen order across _all_tags().
        await pilot.click("#wl-tree-tag-2")
        await pilot.pause()
        assert app.query("#wl-tree-node-watchlist-2")
        assert not app.query("#wl-tree-node-watchlist-1")
        # The permanent roots survive filtering — they are not watchlists.
        assert app.query("#wl-tree-node-all")


@pytest.mark.asyncio
async def test_toggling_a_tag_off_shows_all_watchlists_again():
    app = _TreeApp(_tree_data())
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-tag-2")  # filter to "sec"
        await pilot.pause()
        assert not app.query("#wl-tree-node-watchlist-1")

        await pilot.click("#wl-tree-tag-2")  # toggle the same tag off
        await pilot.pause()
        assert app.query("#wl-tree-node-watchlist-1")
        assert app.query("#wl-tree-node-watchlist-2")


# --- fix-round 1: reproduction of the two reviewer-found defects, run
# against the pre-fix implementation to capture RED before patching. ---


def _tricky_tag_data():
    return {
        "watchlists": [
            {"id": 1, "name": "A", "tags": ["must read"]},
            {"id": 2, "name": "B", "tags": ["ai/ml"]},
            {"id": 3, "name": "C", "tags": ["café"]},
        ],
        "counts": {
            1: {"total": 1, "unread": 1},
            2: {"total": 1, "unread": 1},
            3: {"total": 1, "unread": 1},
            -1: {"total": 0, "unread": 0},
            -2: {"total": 3, "unread": 3},
        },
    }


@pytest.mark.asyncio
async def test_tags_with_id_illegal_characters_render_and_filter():
    """A tag containing a space, a slash, or a non-ASCII character must not
    break id construction — Textual ids are restricted to
    [a-zA-Z_-][a-zA-Z0-9_-]*, but tag text is free-form user data.
    """
    app = _TreeApp(_tricky_tag_data())
    async with app.run_test() as pilot:
        assert app.query("#wl-tree-tag-0")
        assert app.query("#wl-tree-tag-1")
        assert app.query("#wl-tree-tag-2")

        # Clicking the "ai/ml" tag (index 1) narrows to watchlist B only.
        await pilot.click("#wl-tree-tag-1")
        await pilot.pause()
        assert app.query("#wl-tree-node-watchlist-2")
        assert not app.query("#wl-tree-node-watchlist-1")
        assert not app.query("#wl-tree-node-watchlist-3")


@pytest.mark.asyncio
async def test_shared_source_across_watchlists_gets_distinct_ids_and_correct_scope():
    """A source that belongs to more than one watchlist must render distinct
    ids per watchlist and post the correct watchlist_id per click — not
    whichever watchlist happened to be expanded (and thus cached) first.
    """
    shared_row = {"id": 10, "name": "ArXiv: AI", "type": "rss"}
    app = _TreeApp(_tree_data(), source_rows={1: [shared_row], 2: [shared_row]})
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-expand-1")
        await pilot.pause()
        await pilot.click("#wl-tree-expand-2")
        await pilot.pause()

        assert app.query("#wl-tree-node-source-1-10")
        assert app.query("#wl-tree-node-source-2-10")

        await pilot.click("#wl-tree-node-source-1-10")
        await pilot.pause()
        assert app.scopes[-1] == TreeScope(kind="source", watchlist_id=1, source_id=10)

        await pilot.click("#wl-tree-node-source-2-10")
        await pilot.pause()
        assert app.scopes[-1] == TreeScope(kind="source", watchlist_id=2, source_id=10)
