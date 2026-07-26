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
            1: {"total": 24, "unread": 24},
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
        assert app.query("#wl-tree-node-source-10")


@pytest.mark.asyncio
async def test_selecting_a_source_posts_a_source_scope():
    app = _TreeApp(_tree_data(), source_rows={1: [{"id": 10, "name": "ArXiv: AI", "type": "rss"}]})
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-expand-1")
        await pilot.pause()
        await pilot.click("#wl-tree-node-source-10")
        await pilot.pause()
        assert app.scopes[-1] == TreeScope(kind="source", watchlist_id=1, source_id=10)


@pytest.mark.asyncio
async def test_tag_filter_narrows_which_watchlists_render():
    app = _TreeApp(_tree_data())
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-tag-sec")
        await pilot.pause()
        assert app.query("#wl-tree-node-watchlist-2")
        assert not app.query("#wl-tree-node-watchlist-1")
        # The permanent roots survive filtering — they are not watchlists.
        assert app.query("#wl-tree-node-all")
