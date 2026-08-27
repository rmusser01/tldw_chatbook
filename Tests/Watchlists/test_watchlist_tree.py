import pytest
from textual.app import App, ComposeResult
from textual.message import Message
from textual.widgets import Button, Static

from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
    STARRED_BUCKET,
    TODAY_BUCKET,
    AddSourceToWatchlistRequested,
    CreateWatchlistRequested,
    DeleteWatchlistRequested,
    RemoveSourceFromWatchlistRequested,
    RenameWatchlistRequested,
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


def test_source_scope_identity_includes_its_parent_context() -> None:
    all_occurrence = TreeScope(
        kind="source",
        parent_context="all",
        source_id=10,
    )
    unassigned_occurrence = TreeScope(
        kind="source",
        parent_context="unassigned",
        source_id=10,
    )

    assert all_occurrence != unassigned_occurrence
    assert all_occurrence.parent_context == "all"
    assert unassigned_occurrence.parent_context == "unassigned"


class _TreeApp(App):
    def __init__(
        self,
        data,
        source_rows=None,
        active_scope=None,
        expanded=(),
        write_disabled_reason=None,
        source_counts=None,
        all_source_rows=None,
        unassigned_source_rows=None,
        expanded_root_kinds=(),
        unread_pin_source_id=None,
    ):
        super().__init__()
        self._data = data
        self._source_rows = source_rows or {}
        self._active_scope = active_scope
        self._expanded = expanded
        self._write_disabled_reason = write_disabled_reason
        self._source_counts = source_counts
        self._all_source_rows = all_source_rows or []
        self._unassigned_source_rows = unassigned_source_rows or []
        self._expanded_root_kinds = expanded_root_kinds
        self._unread_pin_source_id = unread_pin_source_id
        self.scopes: list[TreeScope] = []
        self.write_requests: list[Message] = []

    def compose(self) -> ComposeResult:
        yield WatchlistTree(
            watchlists=self._data["watchlists"],
            counts=self._data["counts"],
            source_rows_loader=lambda wid: self._source_rows.get(wid, []),
            active_scope=self._active_scope,
            expanded=self._expanded,
            write_disabled_reason=self._write_disabled_reason,
            source_counts=self._source_counts,
            all_source_rows=self._all_source_rows,
            unassigned_source_rows=self._unassigned_source_rows,
            expanded_root_kinds=self._expanded_root_kinds,
            unread_pin_source_id=self._unread_pin_source_id,
            id="wl-tree",
        )

    def on_tree_scope_changed(self, message: TreeScopeChanged) -> None:
        self.scopes.append(message.scope)

    def on_create_watchlist_requested(self, message) -> None:
        self.write_requests.append(message)

    def on_rename_watchlist_requested(self, message) -> None:
        self.write_requests.append(message)

    def on_delete_watchlist_requested(self, message) -> None:
        self.write_requests.append(message)

    def on_add_source_to_watchlist_requested(self, message) -> None:
        self.write_requests.append(message)

    def on_remove_source_from_watchlist_requested(self, message) -> None:
        self.write_requests.append(message)


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
async def test_source_node_shows_unread_badge_only_when_positive():
    """task-2513 Task 8: per-feed unread badges, NNW-style.

    A source with unread items shows the number after its name; a source
    with none shows no number at all (roots and watchlists keep their
    always-show behaviour). The tooltip always says it in words, including
    the zero case.
    """
    app = _TreeApp(
        _tree_data(),
        source_rows={
            1: [
                {"id": 10, "name": "Feed A", "type": "rss"},
                {"id": 11, "name": "Feed B", "type": "rss"},
            ]
        },
        expanded=frozenset({1}),
        source_counts={
            10: {"total": 5, "unread": 3},
            11: {"total": 2, "unread": 0},
        },
    )
    async with app.run_test():
        feed_a = app.query_one("#wl-tree-node-source-1-10", Button)
        feed_b = app.query_one("#wl-tree-node-source-1-11", Button)
        assert str(feed_a.label) == "    Feed A  3"
        assert str(feed_b.label) == "    Feed B"
        assert "3 unread items" in (feed_a.tooltip or "")
        assert "No unread items" in (feed_b.tooltip or "")


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
async def test_all_sources_caret_expands_without_selecting_and_child_keeps_context():
    app = _TreeApp(
        _tree_data(),
        all_source_rows=[{"id": 10, "name": "ArXiv: AI", "type": "rss"}],
    )
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-expand-root-all")
        await pilot.pause()

        assert app.scopes == []
        assert app.query("#wl-tree-node-source-all-10")

        await pilot.click("#wl-tree-node-source-all-10")
        await pilot.pause()
        assert app.scopes[-1] == TreeScope(
            kind="source",
            parent_context="all",
            source_id=10,
        )


@pytest.mark.asyncio
async def test_expanded_empty_aggregate_branches_explain_their_context():
    app = _TreeApp(
        _tree_data(),
        expanded_root_kinds=frozenset({"all", "unassigned", "unread"}),
    )
    async with app.run_test():
        expected = {
            "#wl-tree-empty-all": "No Watchlists sources yet.",
            "#wl-tree-empty-unassigned": "No unassigned feeds",
            "#wl-tree-empty-unread": "No unread feeds",
        }
        for selector, copy in expected.items():
            empty = app.query_one(selector, Static)
            assert str(empty.renderable) == copy


@pytest.mark.asyncio
async def test_all_unread_keeps_only_positive_feeds_plus_the_zero_count_pin():
    rows = [
        {"id": 10, "name": "Pinned", "type": "rss"},
        {"id": 11, "name": "Unread", "type": "rss"},
        {"id": 12, "name": "Read", "type": "rss"},
    ]
    app = _TreeApp(
        _tree_data(),
        all_source_rows=rows,
        expanded_root_kinds=frozenset({"unread"}),
        unread_pin_source_id=10,
        source_counts={
            10: {"total": 1, "unread": 0},
            11: {"total": 1, "unread": 1},
            12: {"total": 1, "unread": 0},
        },
    )
    async with app.run_test():
        assert app.query("#wl-tree-node-source-unread-10")
        assert app.query("#wl-tree-node-source-unread-11")
        assert not app.query("#wl-tree-node-source-unread-12")


@pytest.mark.asyncio
async def test_aggregate_children_are_stably_sorted_and_keep_exact_occurrences():
    rows = [
        {"id": 12, "name": "zebra", "type": "rss"},
        {"id": 11, "name": "Alpha", "type": "rss"},
        {"id": 10, "name": "alpha", "type": "rss"},
    ]
    app = _TreeApp(
        _tree_data(),
        all_source_rows=rows,
        unassigned_source_rows=[rows[0]],
        expanded_root_kinds=frozenset({"all", "unassigned"}),
        active_scope=TreeScope(
            kind="source",
            parent_context="unassigned",
            source_id=12,
        ),
    )
    async with app.run_test() as pilot:
        all_ids = [
            button.id
            for button in app.query(Button)
            if button.id and button.id.startswith("wl-tree-node-source-all-")
        ]
        assert all_ids == [
            "wl-tree-node-source-all-10",
            "wl-tree-node-source-all-11",
            "wl-tree-node-source-all-12",
        ]
        assert app.query_one(
            "#wl-tree-node-source-unassigned-12", Button
        ).has_class("is-active")
        assert not app.query_one(
            "#wl-tree-node-source-all-12", Button
        ).has_class("is-active")

        await pilot.click("#wl-tree-node-source-unassigned-12")
        await pilot.pause()
        assert app.scopes[-1] == TreeScope(
            kind="source",
            parent_context="unassigned",
            source_id=12,
        )


@pytest.mark.asyncio
async def test_contextual_child_keeps_focus_when_scope_commit_recomposes_tree():
    app = _TreeApp(
        _tree_data(),
        all_source_rows=[{"id": 10, "name": "Feed", "type": "rss"}],
        expanded_root_kinds=frozenset({"all"}),
    )
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-node-source-all-10")
        await pilot.pause()
        tree = app.query_one("#wl-tree", WatchlistTree)
        tree.active_scope = TreeScope(
            kind="source",
            parent_context="all",
            source_id=10,
        )
        await pilot.pause()

        assert app.focused is app.query_one("#wl-tree-node-source-all-10", Button)


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
        assert app.scopes[-1] == TreeScope(
            kind="source",
            parent_context="watchlist",
            watchlist_id=1,
            source_id=10,
        )


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
        assert app.scopes[-1] == TreeScope(
            kind="source",
            parent_context="watchlist",
            watchlist_id=1,
            source_id=10,
        )

        await pilot.click("#wl-tree-node-source-2-10")
        await pilot.pause()
        assert app.scopes[-1] == TreeScope(
            kind="source",
            parent_context="watchlist",
            watchlist_id=2,
            source_id=10,
        )


# --- task-876: `active_scope` marks the node the screen is scoped to -------
#
# `WatchlistTree` rendered every node as a plain Button and never read
# `tree_scope` at all, so the left rail gave no sign of which node the
# centre was scoped to -- the Feeds heading was the only feedback. These
# pin the widget's own class-assignment logic (which node gets `is-active`
# for a given `active_scope`); the production-stylesheet rendering check
# for the same highlight lives in
# Tests/UI/test_destination_visual_parity_correction.py, alongside the
# other geometry/rendering assertions that require the real CSS bundle.


@pytest.mark.asyncio
async def test_active_scope_all_marks_the_all_root_active():
    app = _TreeApp(_tree_data(), active_scope=TreeScope(kind="all"))
    async with app.run_test():
        assert app.query_one("#wl-tree-node-all", Button).has_class("is-active")
        assert not app.query_one("#wl-tree-node-unassigned", Button).has_class(
            "is-active"
        )


@pytest.mark.asyncio
async def test_active_scope_unassigned_marks_the_unassigned_root_active():
    app = _TreeApp(_tree_data(), active_scope=TreeScope(kind="unassigned"))
    async with app.run_test():
        assert app.query_one("#wl-tree-node-unassigned", Button).has_class(
            "is-active"
        )
        assert not app.query_one("#wl-tree-node-all", Button).has_class("is-active")


@pytest.mark.asyncio
async def test_active_scope_watchlist_marks_only_that_watchlist_node_active():
    app = _TreeApp(
        _tree_data(),
        active_scope=TreeScope(kind="watchlist", watchlist_id=2),
    )
    async with app.run_test():
        assert app.query_one(
            "#wl-tree-node-watchlist-2", Button
        ).has_class("is-active")
        assert not app.query_one(
            "#wl-tree-node-watchlist-1", Button
        ).has_class("is-active")
        assert not app.query_one("#wl-tree-node-all", Button).has_class("is-active")


@pytest.mark.asyncio
async def test_active_scope_source_marks_only_that_source_node_active():
    app = _TreeApp(
        _tree_data(),
        source_rows={1: [{"id": 10, "name": "ArXiv: AI", "type": "rss"}]},
        active_scope=TreeScope(
            kind="source",
            parent_context="watchlist",
            watchlist_id=1,
            source_id=10,
        ),
        expanded=frozenset({1}),
    )
    async with app.run_test():
        assert app.query_one(
            "#wl-tree-node-source-1-10", Button
        ).has_class("is-active")
        # The source's own parent watchlist node must NOT also read as
        # active -- only the single node matching the scope exactly.
        assert not app.query_one(
            "#wl-tree-node-watchlist-1", Button
        ).has_class("is-active")


@pytest.mark.asyncio
async def test_active_scope_none_marks_nothing_active():
    app = _TreeApp(_tree_data(), active_scope=None)
    async with app.run_test():
        assert not any(
            button.has_class("is-active")
            for button in app.query(Button)
            if button.id and button.id.startswith("wl-tree-node-")
        )


@pytest.mark.asyncio
async def test_setting_active_scope_after_mount_moves_the_highlight():
    """The screen pushes a new scope into the already-mounted tree (a real
    click's own scope, or a breadcrumb promotion, neither of which rebuilds
    this widget) -- see `WatchlistsCollectionsScreen.watch_tree_scope`.
    """
    app = _TreeApp(_tree_data(), active_scope=TreeScope(kind="all"))
    async with app.run_test() as pilot:
        tree = app.query_one("#wl-tree", WatchlistTree)
        tree.active_scope = TreeScope(kind="watchlist", watchlist_id=1)
        await pilot.pause()

        assert app.query_one("#wl-tree-node-watchlist-1", Button).has_class(
            "is-active"
        )
        assert not app.query_one("#wl-tree-node-all", Button).has_class("is-active")


# --- TASK-895: the tree's five write verbs --------------------------------
#
# `create`, `rename`, `delete`, `add_source` and `remove_source` had no
# production caller at all: the rail could be browsed but nothing in it
# could be changed. These pin the widget half -- which action is armed for a
# given scope, what it posts, and that a blocked action is disabled *and*
# says why. The service calls and dialogs live on the screen and are
# covered in Tests/Watchlists/test_watchlists_collections_screen.py.

_ACTION_IDS = (
    "#wl-tree-new",
    "#wl-tree-rename",
    "#wl-tree-delete",
    "#wl-tree-add-source",
    "#wl-tree-remove-source",
)


@pytest.mark.asyncio
async def test_the_five_write_verbs_render_in_the_rail():
    app = _TreeApp(_tree_data())
    async with app.run_test():
        for action_id in _ACTION_IDS:
            assert app.query(action_id), f"{action_id} is missing from the rail"


@pytest.mark.asyncio
async def test_only_create_is_armed_when_nothing_is_selected():
    """Rename/Delete/Add-source need a watchlist and Remove needs a source,
    so with no scope in view only New can do anything -- and the other four
    must be visibly off, with a tooltip saying what to select. A disabled
    button that renders as though it were live is a defect this program has
    already fixed once.
    """
    app = _TreeApp(_tree_data(), active_scope=None)
    async with app.run_test():
        assert not app.query_one("#wl-tree-new", Button).disabled
        for action_id in _ACTION_IDS[1:]:
            button = app.query_one(action_id, Button)
            assert button.disabled, f"{action_id} should be disabled with no scope"
            assert button.tooltip, f"{action_id} is disabled without saying why"


@pytest.mark.asyncio
async def test_a_watchlist_scope_arms_rename_delete_and_add_source():
    app = _TreeApp(
        _tree_data(), active_scope=TreeScope(kind="watchlist", watchlist_id=2)
    )
    async with app.run_test():
        for action_id in ("#wl-tree-rename", "#wl-tree-delete", "#wl-tree-add-source"):
            assert not app.query_one(action_id, Button).disabled
        # Removing a source still needs a *source* selected, not a watchlist.
        assert app.query_one("#wl-tree-remove-source", Button).disabled


@pytest.mark.asyncio
async def test_a_source_scope_arms_remove_only():
    app = _TreeApp(
        _tree_data(),
        source_rows={1: [{"id": 10, "name": "ArXiv: AI", "type": "rss"}]},
        active_scope=TreeScope(
            kind="source",
            parent_context="watchlist",
            watchlist_id=1,
            source_id=10,
        ),
        expanded=frozenset({1}),
    )
    async with app.run_test():
        assert not app.query_one("#wl-tree-remove-source", Button).disabled
        for action_id in ("#wl-tree-rename", "#wl-tree-delete", "#wl-tree-add-source"):
            assert app.query_one(action_id, Button).disabled


@pytest.mark.asyncio
async def test_pressing_new_posts_a_create_request():
    app = _TreeApp(_tree_data())
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-new")
        await pilot.pause()
        assert isinstance(app.write_requests[-1], CreateWatchlistRequested)


@pytest.mark.asyncio
async def test_rename_delete_and_add_source_carry_the_scoped_watchlist_id():
    app = _TreeApp(
        _tree_data(), active_scope=TreeScope(kind="watchlist", watchlist_id=2)
    )
    async with app.run_test() as pilot:
        for action_id, message_type in (
            ("#wl-tree-rename", RenameWatchlistRequested),
            ("#wl-tree-delete", DeleteWatchlistRequested),
            ("#wl-tree-add-source", AddSourceToWatchlistRequested),
        ):
            await pilot.click(action_id)
            await pilot.pause()
            message = app.write_requests[-1]
            assert isinstance(message, message_type)
            assert message.watchlist_id == 2


@pytest.mark.asyncio
async def test_remove_carries_both_ids_because_membership_is_many_to_many():
    """"Source 10" alone does not say which watchlist it is leaving -- the
    same reason the source node's own id is watchlist-qualified.
    """
    shared_row = {"id": 10, "name": "ArXiv: AI", "type": "rss"}
    app = _TreeApp(
        _tree_data(),
        source_rows={1: [shared_row], 2: [shared_row]},
        active_scope=TreeScope(
            kind="source",
            parent_context="watchlist",
            watchlist_id=2,
            source_id=10,
        ),
        expanded=frozenset({1, 2}),
    )
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-remove-source")
        await pilot.pause()
        message = app.write_requests[-1]
        assert isinstance(message, RemoveSourceFromWatchlistRequested)
        assert (message.watchlist_id, message.source_id) == (2, 10)


@pytest.mark.asyncio
async def test_a_write_disabled_reason_turns_every_verb_off_and_states_itself():
    """AC #5's widget half: on the server backend there is no wire path for
    any of these, so all five are disabled -- not hidden -- and the reason
    is both the tooltip and a line the user can read without hovering.
    """
    reason = "Server backend: no wire path for watchlist membership edits."
    app = _TreeApp(
        _tree_data(),
        active_scope=TreeScope(kind="watchlist", watchlist_id=2),
        write_disabled_reason=reason,
    )
    async with app.run_test():
        for action_id in _ACTION_IDS:
            button = app.query_one(action_id, Button)
            assert button.disabled, f"{action_id} must be disabled, not hidden"
            assert str(button.tooltip) == reason

        note = app.query_one("#wl-tree-actions-unavailable", Static)
        assert reason in str(getattr(note.renderable, "plain", note.renderable))


@pytest.mark.asyncio
async def test_no_unavailable_note_when_writes_are_available():
    app = _TreeApp(_tree_data(), write_disabled_reason=None)
    async with app.run_test():
        assert not app.query("#wl-tree-actions-unavailable")


@pytest.mark.asyncio
async def test_a_blocked_verb_posts_nothing_even_if_it_is_pressed_directly():
    """Belt-and-braces for the guard in `_post_action`: a disabled Button
    never emits `Pressed`, but the screen can push a new `active_scope` into
    a still-mounted tree between renders, so the handler re-checks rather
    than trusting the `disabled=` flag `compose()` baked in.
    """
    app = _TreeApp(
        _tree_data(),
        active_scope=TreeScope(kind="watchlist", watchlist_id=2),
        write_disabled_reason="Watchlists services are unavailable in this runtime.",
    )
    async with app.run_test() as pilot:
        tree = app.query_one("#wl-tree", WatchlistTree)
        tree.on_button_pressed(
            Button.Pressed(app.query_one("#wl-tree-rename", Button))
        )
        await pilot.pause()
        assert app.write_requests == []


# --- TASK-3072 plan task 6: the Starred smart feed ---------------------------
#
# NNW's first-class smart feed: one root above the watchlists whose badge is
# `SubscriptionsDB.get_flagged_items_count` (starred items, status-agnostic)
# and whose scope maps to `{"is_flagged": True}` in the items query.


def _starred_data(flagged: int):
    data = _tree_data()
    data["counts"][STARRED_BUCKET] = {"total": flagged, "unread": flagged}
    return data


@pytest.mark.asyncio
async def test_starred_root_renders_above_the_watchlists_with_its_count():
    app = _TreeApp(_starred_data(5))
    async with app.run_test():
        starred = app.query_one("#wl-tree-node-starred", Button)
        assert str(starred.label) == "★ Starred  5"
        # DOM order is the rail's reading order: starred sits with the roots,
        # above every watchlist node.
        ids = [node.id for node in app.query(Button)]
        assert ids.index("wl-tree-node-starred") < ids.index(
            "wl-tree-node-watchlist-1"
        )


@pytest.mark.asyncio
async def test_starred_root_without_a_count_renders_zero():
    """Before any item is starred there is no STARRED_BUCKET entry at all --
    the badge must read 0, not crash the rail."""
    app = _TreeApp(_tree_data())
    async with app.run_test():
        starred = app.query_one("#wl-tree-node-starred", Button)
        assert str(starred.label) == "★ Starred  0"


@pytest.mark.asyncio
async def test_starred_root_tooltip_says_starred_not_unread():
    """The legend row labels the rail's numbers "unread items"; the starred
    badge counts STARRED items, so its own tooltip must say the true word --
    the TASK-2304 AC#3 honesty rule applied to the one node the legend does
    not describe."""
    app = _TreeApp(_starred_data(5))
    async with app.run_test():
        starred = app.query_one("#wl-tree-node-starred", Button)
        tooltip = starred.tooltip or ""
        assert "5 starred items" in tooltip
        assert "unread" not in tooltip


@pytest.mark.asyncio
async def test_selecting_starred_posts_the_starred_scope():
    app = _TreeApp(_starred_data(5))
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-node-starred")
        await pilot.pause()
        assert app.scopes[-1] == TreeScope(kind="starred")


@pytest.mark.asyncio
async def test_active_scope_starred_marks_the_starred_root_active():
    app = _TreeApp(_starred_data(5), active_scope=TreeScope(kind="starred"))
    async with app.run_test():
        assert app.query_one("#wl-tree-node-starred", Button).has_class("is-active")
        assert not app.query_one("#wl-tree-node-all", Button).has_class("is-active")


# --- TASK-3791 plan task 4: All Unread + Today smart feeds ---------------------


def _phase3_data(today: int = 2):
    data = _starred_data(5)
    data["counts"][TODAY_BUCKET] = {"total": today, "unread": today}
    return data


@pytest.mark.asyncio
async def test_unread_and_today_roots_render_in_the_smart_feed_cluster():
    """NNW's standing smart feeds: All Unread and Today sit with Starred,
    above the watchlists. All Unread's badge is the same unread count the
    All sources root shows (one fact, two angles); Today's is the
    unread-since-local-midnight count the screen inserts."""
    app = _TreeApp(_phase3_data())
    async with app.run_test():
        assert str(app.query_one("#wl-tree-node-unread", Button).label) == "All Unread  37"
        assert str(app.query_one("#wl-tree-node-today", Button).label) == "Today  2"
        ids = [node.id for node in app.query(Button)]
        assert ids.index("wl-tree-node-unassigned") < ids.index("wl-tree-node-unread")
        assert ids.index("wl-tree-node-unread") < ids.index("wl-tree-node-today")
        assert ids.index("wl-tree-node-today") < ids.index("wl-tree-node-starred")
        assert ids.index("wl-tree-node-starred") < ids.index("wl-tree-node-watchlist-1")


@pytest.mark.asyncio
async def test_today_root_without_a_count_renders_zero():
    app = _TreeApp(_starred_data(5))
    async with app.run_test():
        assert str(app.query_one("#wl-tree-node-today", Button).label) == "Today  0"


@pytest.mark.asyncio
async def test_selecting_unread_and_today_post_their_scopes():
    app = _TreeApp(_phase3_data())
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-node-unread")
        await pilot.pause()
        assert app.scopes[-1] == TreeScope(kind="unread")
        await pilot.click("#wl-tree-node-today")
        await pilot.pause()
        assert app.scopes[-1] == TreeScope(kind="today")


@pytest.mark.asyncio
async def test_active_scope_today_marks_the_today_root_active():
    app = _TreeApp(_phase3_data(), active_scope=TreeScope(kind="today"))
    async with app.run_test():
        assert app.query_one("#wl-tree-node-today", Button).has_class("is-active")
        assert not app.query_one("#wl-tree-node-unread", Button).has_class("is-active")
