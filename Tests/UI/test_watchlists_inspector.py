"""Tests for the Watchlists inspector pane wiring."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import DestinationHarness, StaticWatchlistsScopeService
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.watchlists_collections_screen import WatchlistsCollectionsScreen
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import BreadcrumbScopeSelected, InspectorPane
from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemSelected
from tldw_chatbook.UI.Watchlists_Modules.notifications_pane import (
    NotificationSelected,
    RefreshNotificationsRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope, TreeScopeChanged


def _app_with_watchlists(watch_items):
    app = _build_test_app()
    app.watchlist_scope_service = SimpleNamespace(
        list_watch_items=StaticWatchlistsScopeService(watch_items).list_watch_items,
    )
    return app


@pytest.mark.asyncio
async def test_inspector_pane_mounts_in_screen():
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        assert screen.query_one("#watchlists-entity-inspector", InspectorPane)


@pytest.mark.asyncio
async def test_selecting_source_updates_inspector_actions():
    sources = [
        {
            "id": "source-1",
            "name": "AI News RSS",
            "source_type": "rss",
            "url": "http://example.com/feed",
            "active": True,
        },
    ]
    app = _app_with_watchlists(sources)
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "sources"
        await pilot.pause()

        sources_pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        sources_pane.sources = sources
        await pilot.pause()
        sources_pane.select_source_by_id("source-1")
        await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        assert inspector.query_one("#inspector-preview-button", Button)
        assert inspector.query_one("#inspector-check-now-button", Button)
        assert inspector.query_one("#inspector-stage-console-button", Button)
        assert inspector.query_one("#inspector-delete-button", Button)


@pytest.mark.asyncio
async def test_selecting_run_updates_inspector_actions():
    runs = [
        {
            "id": "run-1",
            "source_title": "AI News RSS",
            "status": "completed",
            "found_count": 5,
            "processed_count": 4,
            "filtered_count": 1,
            "error_count": 0,
        },
    ]
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "runs"
        await pilot.pause()

        from tldw_chatbook.UI.Watchlists_Modules.runs_pane import RunsPane
        runs_pane = screen.query_one("#watchlists-runs-pane", RunsPane)
        runs_pane.runs = runs
        await pilot.pause()
        runs_pane.select_run_by_id("run-1")
        await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        assert not inspector.query("#inspector-preview-button")
        assert not inspector.query("#inspector-check-now-button")
        assert inspector.query_one("#inspector-stage-console-button", Button)
        assert inspector.query_one("#inspector-delete-button", Button)


@pytest.mark.asyncio
async def test_inspector_delete_button_posts_delete_requested():
    sources = [
        {
            "id": "source-1",
            "name": "AI News RSS",
            "source_type": "rss",
            "url": "http://example.com/feed",
            "active": True,
        },
    ]
    app = _app_with_watchlists(sources)
    host = DestinationHarness(app, "watchlists_collections")
    captured = []

    def capture_message(message):
        captured.append(message)

    async with host.run_test(size=(180, 50), message_hook=capture_message) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "sources"
        await pilot.pause()

        sources_pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        sources_pane.sources = sources
        await pilot.pause()
        sources_pane.select_source_by_id("source-1")
        await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        inspector.query_one("#inspector-delete-button", Button).press()
        await pilot.pause()

        assert any(
            msg.__class__.__name__ == "DeleteRequested"
            and (msg.entity or {}).get("id") == "source-1"
            for msg in captured
        )


# -- Task 5: breadcrumb stack -------------------------------------------------
#
# `scope`/`breadcrumb_labels` are set directly on the mounted `InspectorPane`
# rather than via the tree (`TreeScopeChanged` -> `screen.selected_scope`):
# the screen does not yet push its `selected_scope` into the Inspector (that
# wiring, like `WatchlistBundleService`'s in Task 1, is deliberately left for
# a later task -- this one only has to build the pane's own capability). This
# still fully exercises the reactive contract Task 7 (or a follow-up) will
# drive from the tree.


@pytest.mark.asyncio
async def test_breadcrumb_shows_each_selected_level():
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)

        inspector.scope = TreeScope(kind="source", watchlist_id=1, source_id=10)
        inspector.breadcrumb_labels = ["Morning AI Brief", "ArXiv: AI"]
        await pilot.pause()

        texts = [str(s.renderable) for s in inspector.query(Static)]
        texts += [str(b.label) for b in inspector.query(Button)]
        combined = " ".join(texts)
        assert "Morning AI Brief" in combined
        assert "ArXiv: AI" in combined

        # The shallower level (watchlist) collapses to one clickable
        # breadcrumb line; the deepest (source) is full detail, not a
        # breadcrumb button of its own.
        assert inspector.query_one("#inspector-breadcrumb-0", Button)
        assert not inspector.query("#inspector-breadcrumb-1")


@pytest.mark.asyncio
async def test_actions_belong_to_the_deepest_level():
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)

        inspector.scope = TreeScope(kind="watchlist", watchlist_id=1)
        await pilot.pause()

        assert inspector.query_one("#inspector-check-now-button", Button)
        assert inspector.query_one("#inspector-delete-button", Button)
        # "Mark reviewed" was removed entirely (Task 5 fix round 1), so an
        # absence check for it here would no longer discriminate anything --
        # Ingest/Ignore below are the item actions that still exist and must
        # not show while a watchlist is the deepest selection.
        assert not inspector.query("#inspector-ingest-button"), (
            "an item action must not show while a watchlist is the deepest selection"
        )
        assert not inspector.query("#inspector-ignore-button")
        assert not inspector.query("#inspector-preview-button")


@pytest.mark.asyncio
async def test_selected_entity_is_deeper_than_scope():
    """A row picked within a pane (an item, here) is one level deeper than
    the tree scope that got the user there: the scope's own levels collapse
    to breadcrumbs, and the entity becomes the expanded deepest level with
    ITS actions -- never the scope's."""
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)

        inspector.scope = TreeScope(kind="source", watchlist_id=1, source_id=10)
        inspector.breadcrumb_labels = ["Morning AI Brief", "ArXiv: AI"]
        inspector.selected_entity = {"item_id": "item-1", "title": "RAG Evaluation"}
        await pilot.pause()

        # Both scope levels are now ancestors (collapsed breadcrumbs) --
        # the item selected within the source is deeper than the source
        # itself.
        assert inspector.query_one("#inspector-breadcrumb-0", Button)
        assert inspector.query_one("#inspector-breadcrumb-1", Button)
        # "Mark reviewed" was removed (Task 5 fix round 1); Ingest is the
        # still-present item action used here to prove the item's action
        # set is showing.
        assert inspector.query_one("#inspector-ingest-button", Button)
        assert not inspector.query("#inspector-check-now-button")


@pytest.mark.asyncio
async def test_clicking_breadcrumb_requests_scope_promotion():
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    captured = []

    def capture_message(message):
        captured.append(message)

    async with host.run_test(size=(180, 50), message_hook=capture_message) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)

        inspector.scope = TreeScope(kind="source", watchlist_id=1, source_id=10)
        inspector.breadcrumb_labels = ["Morning AI Brief", "ArXiv: AI"]
        await pilot.pause()

        inspector.query_one("#inspector-breadcrumb-0", Button).press()
        await pilot.pause()

        promoted = [m for m in captured if isinstance(m, BreadcrumbScopeSelected)]
        assert promoted, "clicking the shallower breadcrumb should request promotion"
        assert promoted[0].scope == TreeScope(kind="watchlist", watchlist_id=1)


# -- Task 5, fix round 1: wire scope/breadcrumb_labels to the live tree ------
#
# Unlike the tests above (which drive `InspectorPane` directly), these two
# go through the real path a user's tree click takes: `TreeScopeChanged`
# reaches the SCREEN, not the pane, so this is what proves the wiring the
# coordinator asked for -- as opposed to the pane's own already-tested
# capability to render whatever `scope`/`breadcrumb_labels` it is given.


@pytest.mark.asyncio
async def test_tree_scope_reaching_screen_populates_inspector_breadcrumb():
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        # Seeds the same data `_load_tree_data` would have loaded, so the
        # breadcrumb shows the real watchlist name rather than the
        # `Watchlist {id}` fallback `_resolve_breadcrumb_labels` uses when
        # no matching row is found.
        screen._tree_watchlists = [{"id": 7, "name": "Morning AI Brief"}]

        screen.post_message(TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=7)))
        await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        assert inspector.scope == TreeScope(kind="watchlist", watchlist_id=7)
        assert inspector.breadcrumb_labels == ["Morning AI Brief"]

        texts = [str(s.renderable) for s in inspector.query(Static)]
        texts += [str(b.label) for b in inspector.query(Button)]
        assert "Morning AI Brief" in " ".join(texts)


@pytest.mark.asyncio
async def test_inspector_breadcrumb_survives_a_left_rail_toggle():
    """`[` toggles the LEFT rail, not the right rail the Inspector lives in
    -- but `region_layout` is screen-level `recompose=True`, so ANY region
    toggle rebuilds the whole workbench, constructing a brand new
    `InspectorPane` via `_build_inspector_pane`'s factory (see
    `test_scope_survives_a_region_toggle` in the guard file, which proves
    the same thing for `screen.selected_scope` alone). Without seeding the
    fresh pane from screen state, the breadcrumb would go blank on any
    rail/region toggle -- unrelated to the tree selection it just lost.
    """
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen._tree_watchlists = [{"id": 7, "name": "Morning AI Brief"}]
        screen.post_message(TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=7)))
        await pilot.pause()

        await pilot.press("[")
        await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        assert inspector.scope == TreeScope(kind="watchlist", watchlist_id=7), (
            "the Inspector rebuilt by the rail toggle must be re-seeded from "
            "screen state, not start back at its class default"
        )
        assert inspector.breadcrumb_labels == ["Morning AI Brief"]


# -- Task 5, fix round 2 -----------------------------------------------------
#
# Finding 1: `scope` and `selected_entity` drifted -- selecting an entity
#   never touched `selected_scope`, and changing the tree scope never
#   touched `selected_entity`, so a stale ancestor and a fresh entity (or
#   vice versa) could both be shown together, describing two different
#   things. Finding 2: the watchlist-level Check now/Delete buttons were
#   enabled but every consumer handler silently no-ops on `entity=None`.
#   Finding 3: `BreadcrumbScopeSelected` had zero consumers -- clicking a
#   shallower breadcrumb did nothing.


@pytest.mark.asyncio
async def test_changing_scope_clears_a_stale_entity_selection():
    """Finding 1's exact reproduction: select an item under one watchlist,
    then switch the tree to a different watchlist. Before this fix the
    breadcrumb named the new watchlist while the actions still targeted the
    old item, with nothing on screen indicating the mismatch.
    """
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen._tree_watchlists = [
            {"id": 1, "name": "First Watchlist"},
            {"id": 2, "name": "Second Watchlist"},
        ]

        screen.post_message(TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=1)))
        await pilot.pause()
        screen.post_message(ItemSelected({"item_id": "item-1", "title": "RAG Eval"}))
        await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        # "Mark reviewed" was removed (Task 5 fix round 1); Ingest is the
        # still-present item action used here to prove item-level actions
        # are showing.
        assert inspector.query_one("#inspector-ingest-button", Button)

        screen.post_message(TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=2)))
        await pilot.pause()

        assert screen.selected_entity is None, (
            "switching the tree scope must drop the now-stale entity selection"
        )
        assert inspector.breadcrumb_labels == ["Second Watchlist"]
        assert not inspector.query("#inspector-ingest-button"), (
            "the breadcrumb now names Watchlist 2 -- Watchlist 1's item "
            "actions must not still be showing beneath it"
        )
        assert inspector.query_one("#inspector-check-now-button", Button), (
            "the deepest level is now the bare watchlist scope, so its own "
            "action set (not the stale item's) should render"
        )


@pytest.mark.asyncio
async def test_selecting_an_entity_clears_a_stale_watchlist_ancestor():
    """The other write direction of Finding 1: browse into a source via the
    tree, then select an unrelated item from a pane row (its ancestry is
    not actually known here -- see `_select_entity`'s docstring). The stale
    source/watchlist breadcrumb must not linger above it.
    """
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen._tree_watchlists = [{"id": 1, "name": "Morning AI Brief"}]

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="source", watchlist_id=1, source_id=10))
        )
        await pilot.pause()
        screen.post_message(ItemSelected({"item_id": "item-1", "title": "RAG Eval"}))
        await pilot.pause()

        assert screen.selected_scope == TreeScope(kind="all")
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        assert inspector.breadcrumb_labels == []
        assert not inspector.query("#inspector-breadcrumb-0"), (
            "no breadcrumb ancestor is known for a pane-selected item in "
            "this slice -- it must not keep showing the tree's old one"
        )
        # "Mark reviewed" was removed (Task 5 fix round 1); Ingest is the
        # still-present item action used here to prove the item's action
        # set is showing.
        assert inspector.query_one("#inspector-ingest-button", Button)


@pytest.mark.asyncio
async def test_watchlist_level_actions_are_disabled_not_silently_broken():
    """Finding 2: Check now/Delete render for a bare watchlist scope but
    must not be clickable -- no consumer handler acts on `entity=None`, so
    an enabled button there produced no action, no error, and no toast.
    """
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.post_message(TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=1)))
        await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        check_now = inspector.query_one("#inspector-check-now-button", Button)
        delete = inspector.query_one("#inspector-delete-button", Button)
        assert check_now.disabled
        assert delete.disabled
        assert "not implemented yet" in str(check_now.tooltip or "")
        assert "not implemented yet" in str(delete.tooltip or "")


@pytest.mark.asyncio
async def test_clicking_breadcrumb_actually_promotes_the_scope():
    """Finding 3: the click reaching the screen must actually change what's
    shown, not just get posted into the void -- `BreadcrumbScopeSelected`
    had zero consumers before this fix.
    """
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen._tree_watchlists = [{"id": 1, "name": "Morning AI Brief"}]

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="source", watchlist_id=1, source_id=10))
        )
        await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        assert inspector.query_one("#inspector-breadcrumb-0", Button)

        inspector.query_one("#inspector-breadcrumb-0", Button).press()
        await pilot.pause()

        assert screen.selected_scope == TreeScope(kind="watchlist", watchlist_id=1)
        assert inspector.scope == TreeScope(kind="watchlist", watchlist_id=1)
        assert inspector.breadcrumb_labels == ["Morning AI Brief"]
        # Promoted to the deepest level now -- it is the whole breadcrumb,
        # not a collapsed ancestor of something deeper anymore.
        assert not inspector.query("#inspector-breadcrumb-0")


@pytest.mark.asyncio
async def test_stale_notification_mirror_does_not_resurrect_under_a_new_scope():
    """Task 5 fix round 3, Finding 1's remaining gap.

    `_apply_tree_scope` cleared `selected_entity` but left
    `selected_notification` standing -- a persisted shadow of that same
    selection. Reachable path: select a notification (scope reconciles to
    "all"), click a tree node for a watchlist (scope moves, entity cleared,
    `selected_notification` left set), then anything that reruns
    `_load_notifications` (Refresh, mark-read, dismiss, a section
    round-trip) re-derives `selected_entity` from the surviving mirror
    without touching `selected_scope` -- the breadcrumb names the watchlist
    while the detail/actions belong to the notification again.
    """
    row = {
        "id": 7,
        "title": "Research complete",
        "message": "The synthesis is ready.",
        "category": "research",
        "severity": "info",
        "is_read": False,
    }
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen._tree_watchlists = [{"id": 1, "name": "Morning AI Brief"}]
        screen._notifications_controller.load_rows = AsyncMock(return_value=[row])

        screen.post_message(NotificationSelected(row))
        await pilot.pause()
        assert screen.selected_scope == TreeScope(kind="all"), (
            "selecting a notification reconciles scope to 'all', same as "
            "any other pane-selected entity"
        )

        screen.post_message(TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=1)))
        await pilot.pause()

        screen.post_message(RefreshNotificationsRequested())
        for _ in range(20):
            await pilot.pause()
            if screen._notifications_controller.load_rows.await_count:
                break

        assert screen.selected_scope == TreeScope(kind="watchlist", watchlist_id=1), (
            "the tree scope the user navigated to must not be clobbered by "
            "the notifications reload"
        )
        assert screen.selected_entity is None, (
            "the notification mirror must not resurrect selected_entity "
            "under a scope the tree has since moved away from -- the "
            "breadcrumb and the detail/actions must agree"
        )


@pytest.mark.asyncio
async def test_apply_tree_scope_clears_all_persisted_selection_shadows():
    """Pins Task 5 fix round 3's core change directly.

    `selected_source`/`selected_run`/`selected_notification` are persisted
    shadows of the same selection `selected_entity` represents -- one per
    pane, kept so a highlighted row survives that pane's own
    reactive-recompose. `_apply_tree_scope` must clear all three alongside
    the entity, or a surviving mirror can repopulate the selection under a
    scope the tree has since moved away from (see the notifications
    resurrection test above; the sources/runs panes reselecting a stale row
    on rebuild are the visual half of the same gap).
    """
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen._tree_watchlists = [{"id": 1, "name": "Morning AI Brief"}]

        screen.selected_source = {"id": "source-1", "name": "AI News RSS"}
        screen.selected_run = {"id": "run-1", "status": "completed"}
        screen.selected_notification = {"id": 7, "title": "Research complete"}

        screen.post_message(TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=1)))
        await pilot.pause()

        assert screen.selected_source is None
        assert screen.selected_run is None
        assert screen.selected_notification is None
