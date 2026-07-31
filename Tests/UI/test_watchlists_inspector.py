"""Tests for the Watchlists inspector pane wiring."""

import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from textual.geometry import Region
from textual.widgets import Button, DataTable, Static, TextArea

# The end-to-end check harness (TASK-1362 tests below): the real service, the
# real DB and the real `URLMonitor.check_url` persistence path. See its own
# module docstring for why a hand-built item dict would prove nothing.
from Tests.Subscriptions.test_watchlist_content_kind_producer import (
    _check,
    _serve,
    _stored_items,
)
from Tests.Subscriptions.test_watchlist_noise_not_volume import (
    _counts,
    _direct_check,
    _dispositions,
)
from Tests.UI.full_app_destination_context import (
    FullAppDestinationContext as DestinationHarness,
    StaticWatchlistsScopeService,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_watchlists_item_actions import OTHER_ENTITIES, REAL_ITEM
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.Subscriptions.noise_defaults import DEFAULT_IGNORE_SELECTORS
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Watchlists_Modules import inspector_pane as inspector_pane_module
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import (
    BreadcrumbScopeSelected,
    InspectorPane,
    SaveNoiseSelectorsRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemSelected, ItemsPane
from tldw_chatbook.UI.Watchlists_Modules.notifications_pane import (
    NotificationSelected,
    RefreshNotificationsRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.rules_pane import RuleSelected
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
    TreeScope,
    TreeScopeChanged,
)

# Whole-branch review (Important): without this, CI's `pytest -m unit` run
# DESELECTS this entire module. See the identical note in
# `test_watchlists_item_actions.py`.
pytestmark = pytest.mark.unit


def _assert_visible_in_viewport(
    widget,
    *,
    height: int,
    context: str,
    viewport_width: int | None = None,
) -> None:
    """Assert that a production widget is fully inside the test viewport."""
    x, y, widget_width, widget_height = tuple(widget.region)
    assert x >= 0, context
    if viewport_width is not None:
        assert x < viewport_width, context
        assert x + widget_width <= viewport_width, context
    assert y >= 0, context
    assert y < height, context
    assert y + widget_height <= height, context


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
        return True

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
        original_post_message = inspector.post_message
        inspector.post_message = capture_message
        try:
            button = inspector.query_one("#inspector-delete-button", Button)
            inspector.on_button_pressed(Button.Pressed(button))
        finally:
            inspector.post_message = original_post_message

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
        return True

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)

        inspector.scope = TreeScope(kind="source", watchlist_id=1, source_id=10)
        inspector.breadcrumb_labels = ["Morning AI Brief", "ArXiv: AI"]
        await pilot.pause()

        original_post_message = inspector.post_message
        inspector.post_message = capture_message
        try:
            button = inspector.query_one("#inspector-breadcrumb-0", Button)
            inspector.on_button_pressed(Button.Pressed(button))
        finally:
            inspector.post_message = original_post_message

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

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=7))
        )
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
        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=7))
        )
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

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=1))
        )
        await pilot.pause()
        screen.post_message(ItemSelected({"item_id": "item-1", "title": "RAG Eval"}))
        await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        # "Mark reviewed" was removed (Task 5 fix round 1); Ingest is the
        # still-present item action used here to prove item-level actions
        # are showing.
        assert inspector.query_one("#inspector-ingest-button", Button)

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=2))
        )
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

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=1))
        )
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

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=1))
        )
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

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=1))
        )
        await pilot.pause()

        assert screen.selected_source is None
        assert screen.selected_run is None
        assert screen.selected_notification is None


# -- TASK-1362 (spec §2): the Inspector's noise-selector editor --------------
#
# The only edit path a source has. Before this, nothing on the Watchlists
# screen could change a source at all (only alert rules had Edit), so the
# spec's core loop -- a noisy item's diff names what churned, the user adds
# one rule to silence it -- meant deleting the source and recreating it,
# losing its history.
#
# The three assertions that matter are on the REAL stored row, the REAL check
# disposition after the edit, and the screen surviving the save. A test that
# only pressed the button and inspected a posted message would pass whether or
# not the text ever reached `subscriptions.ignore_selectors`.

_NOISY_PAGE = """<html><body>
<h1>Anthropic status</h1>
<div class="ad">BUY NOW</div>
<div class="promo">Limited time offer, ends today</div>
<p>All systems operational.</p>
<p>Latest release: Opus 4.5 is available.</p>
</body></html>"""


async def _seed_url_source(app, *, ignore_selectors: str = ".ad"):
    """Create one real url-family source in the app's real subscriptions DB.

    Goes through `LocalWatchlistsService.create_source` rather than
    `db.add_subscription` so the row is built by the same code path the create
    form uses -- including `_subscription_config_fields`, which is what
    `ignore_selectors` has to survive.

    Returns:
        `(db, service, source_id)`.
    """
    service = app.local_watchlists_service
    source = await service.create_source(
        {
            "name": "Anthropic status",
            "url": "https://example.com/page",
            "source_type": "site",
            "ignore_selectors": ignore_selectors,
        }
    )
    return service._db(), service, int(source["source_id"])


async def _select_real_source(pilot, screen, source_id: int) -> SourcesPane:
    """Open Sources, wait for the real list, and select `source_id`'s row."""
    screen.active_section = "sources"
    await pilot.pause(0.3)
    pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
    for _ in range(40):
        await pilot.pause()
        if pane.sources:
            break
    assert pane.sources, "the real source list must reach the Sources pane"
    pane.select_source_by_id(f"local:subscription:{source_id}")
    await pilot.pause()
    return pane


async def _save_selectors(pilot, screen, text: str) -> None:
    """Type `text` into the Inspector's field and press Save, as a user does.

    The press goes through `InspectorPane.on_button_pressed` ->
    `SaveNoiseSelectorsRequested` -> the screen's `@on` handler -> the real
    controller. Nothing is called directly.
    """
    inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
    field = inspector.query_one("#inspector-noise-selectors", TextArea)
    field.text = text
    await pilot.pause()
    inspector.query_one("#inspector-save-selectors-button", Button).press()


@pytest.mark.asyncio
async def test_saving_selectors_writes_them_to_the_database():
    """Step 1: the field, the real message path, and the stored row."""
    app = _build_test_app()
    db, _service, source_id = await _seed_url_source(app)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        await _select_real_source(pilot, screen, source_id)

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        field = inspector.query_one("#inspector-noise-selectors", TextArea)
        assert field.text == ".ad", (
            "the field must open seeded with what the source actually stores, "
            "not blank and not the shipped default"
        )

        await _save_selectors(pilot, screen, ".ad\n.promo")
        for _ in range(40):
            await pilot.pause()
            if db.get_subscription(source_id)["ignore_selectors"] != ".ad":
                break

        assert db.get_subscription(source_id)["ignore_selectors"] == ".ad\n.promo", (
            "the saved text must reach the subscriptions row -- the whole "
            "point of the affordance"
        )


@pytest.mark.asyncio
async def test_the_editor_renders_only_for_url_family_sources():
    """Step 1's second half: `ignore_selectors` shapes `URLMonitor.check_url`
    and nothing else, so a feed source, an item and a rule must not offer it.

    A feed's items come from the feed's own entries; no selector is consulted
    anywhere on that path, so the control there would be a field that silently
    does nothing. The url source in the same test is the positive control --
    without it, deleting the affordance entirely would pass this test.
    """
    sources = [
        {
            "id": "source-rss",
            "name": "AI News RSS",
            "source_type": "rss",
            "url": "http://example.com/feed",
            "active": True,
        },
        {
            "id": "source-url",
            "name": "Anthropic status",
            "source_type": "url",
            "url": "http://example.com/page",
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
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)

        sources_pane.select_source_by_id("source-url")
        await pilot.pause()
        assert inspector.query_one("#inspector-noise-selectors", TextArea)
        assert inspector.query_one("#inspector-save-selectors-button", Button)
        # Geometry is NOT asserted here. `DestinationHarness` declares no
        # `CSS_PATH`, so none of the shipped rules
        # (`#inspector-noise-selectors { max-height: 4 }`) are applied and a
        # measurement taken on it describes framework defaults, not the
        # product -- see `test_the_editor_fits_on_screen_with_the_source_actions`
        # below, which runs on the production stylesheet instead.

        sources_pane.select_source_by_id("source-rss")
        await pilot.pause()
        assert not inspector.query("#inspector-noise-selectors"), (
            "a feed source has no extraction settings for CSS selectors to "
            "shape -- offering the field there would be inert"
        )
        assert not inspector.query("#inspector-save-selectors-button")

        screen.post_message(ItemSelected({"item_id": "item-1", "title": "RAG Eval"}))
        await pilot.pause()
        assert inspector.query_one("#inspector-ingest-button", Button), (
            "precondition: the item's own action set is what is showing"
        )
        assert not inspector.query("#inspector-noise-selectors")

        screen.post_message(
            RuleSelected(
                {"rule_id": 3, "name": "Price drop", "condition_type": "keyword"}
            )
        )
        await pilot.pause()
        assert inspector.query_one("#inspector-edit-rule-button", Button), (
            "precondition: the rule's own action set is what is showing"
        )
        assert not inspector.query("#inspector-noise-selectors")


#: Both ends of the range the Watchlists parity suite covers. The small size
#: is the one that constrains this: the right rail has to hold the source's
#: four action buttons AND a five-row editor, and 42 rows is where the rest of
#: the screen has already been shown to run out (see `SIZES` in
#: `test_watchlists_source_create_form.py`, whose form had zero spare rows).
_GEOMETRY_SIZES = [(160, 42), (180, 50)]

#: Two stored values a real source holds, because ONE of them cannot exercise
#: the rule under test. `shipped-default` is what every migrated url source
#: carries after Task 2's migration -- six rules, so an uncapped `height:
#: auto` field wants eight rows, and the right rail has roughly nine rows
#: spare below the Inspector's content at both sizes. Measured: removing the
#: cap changes that layout and evicts nothing. So the default set alone cannot
#: tell a working cap from an absent one.
#:
#: `accumulated` is the case the cap exists for. `_IGNORE_SELECTORS_MAX_LENGTH`
#: is 4000 characters precisely because "a long-watched page legitimately
#: accumulates rules" (`sources_pane.py`), and thirty of them in an uncapped
#: field is thirty-two rows in a thirty-two-row rail: the four source actions
#: have nowhere left to go.
_GEOMETRY_SELECTOR_SETS = {
    "shipped-default": list(DEFAULT_IGNORE_SELECTORS),
    "accumulated": [f".noise-rule-{index}" for index in range(30)],
}


@pytest.mark.parametrize("size", _GEOMETRY_SIZES)
@pytest.mark.parametrize(
    "selectors", _GEOMETRY_SELECTOR_SETS.values(), ids=_GEOMETRY_SELECTOR_SETS
)
@pytest.mark.asyncio
async def test_the_editor_fits_on_screen_with_the_source_actions(selectors, size):
    """The editor must not push a source action off the bottom of the screen.

    Fix round 1 (Important). Two things were wrong with the first version of
    this check, and each on its own made it unable to fail for the defect it
    names:

    * It asserted `region.height > 0`, which detects a zero-height collapse
      and nothing else. Run at 160x18 every control was at y=28..40 --
      entirely below the screen -- and it passed. On-screen placement is
      asserted here instead, via the parity suite's own
      `_assert_visible_in_viewport` (x/y >= 0, and both far edges inside the
      viewport).
    * The previous surrogate harness declared no `CSS_PATH`. None of the
      shipped rules applied -- `styles.max_height` was `None`, i.e. the probe
      measured framework defaults, not `#inspector-noise-selectors {
      max-height: 4 }`. This runs the production destination inside the full
      `TldwCli`, so the geometry measured is the geometry that ships.

    The source carries real stored selectors (see `_GEOMETRY_SELECTOR_SETS`),
    which is what makes `max-height` load-bearing rather than decorative: an
    EMPTY field resolves `height: auto` to three rows and never reaches the
    cap at all, so a fixture without selectors measures a control the rule
    does not constrain.
    """
    sources = [
        {
            "id": "source-url",
            "name": "Anthropic status",
            "source_type": "url",
            "url": "http://example.com/page",
            "active": True,
            "settings": {"ignore_selectors": list(selectors)},
        },
    ]
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService(sources)
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=size) as pilot:
        await pilot.pause(0.2)
        screen = host.context_screen
        screen.active_section = "sources"
        await pilot.pause(0.3)

        sources_pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        sources_pane.sources = sources
        await pilot.pause()
        sources_pane.select_source_by_id("source-url")
        await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        field = inspector.query_one("#inspector-noise-selectors", TextArea)
        # The precondition that makes the rest meaningful: the production
        # stylesheet really did reach this widget. Without it a regression
        # that silently drops the rule (a renamed id, a rebuilt bundle that
        # lost the block) would leave every assertion below measuring
        # TextArea's own defaults and still pass.
        assert field.styles.max_height is not None, (
            "the shipped #inspector-noise-selectors rule did not apply -- this "
            "harness is not measuring the product"
        )

        # Whole-branch review, Important 5: the border title must FIT. Textual
        # truncates an over-wide border label with an ellipsis and reports
        # nothing, so the 65-character label this field started with painted as
        # "Ignore elements (CSS s…" in the ~26-column rail -- the truncation ate
        # the syntax note that was the whole reason the label was long, and the
        # 75-character border subtitle below it fared no better.
        #
        # Measured off the painted strip rather than off `field.border_title`,
        # because the attribute holds the full string whether or not it fits:
        # only `render_lines` knows what the user actually sees.
        top_border = field.render_lines(Region(0, 0, field.outer_size.width, 1))[0].text
        assert "…" not in top_border, (
            f"the noise field's border title is truncated at {size[0]}x{size[1]}: "
            f"{top_border!r} -- shorten the label, do not widen the rail"
        )
        assert str(field.border_title) in top_border, (
            f"the whole title must be painted, not merely fit: {top_border!r}"
        )
        assert field.border_subtitle in (None, ""), (
            "a second rail-width border label is a second silent truncation, "
            "and this one duplicated the Save button's tooltip one row below it"
        )
        assert "silence" in str(field.tooltip), (
            "the guidance the shortened title dropped has to live somewhere -- "
            "the tooltip, which has no width budget"
        )

        width, height = size
        for control_id in (
            "#inspector-noise-selectors",
            "#inspector-save-selectors-button",
            "#inspector-preview-button",
            "#inspector-check-now-button",
            "#inspector-stage-console-button",
            "#inspector-delete-button",
        ):
            control = inspector.query_one(control_id)
            assert control.region.height > 0, f"{control_id} collapsed to no height"
            _assert_visible_in_viewport(
                control,
                height=height,
                viewport_width=width,
                context=(
                    f"{control_id} is not on screen at {width}x{height} -- the "
                    "noise editor pushed a source action past the edge"
                ),
            )


@pytest.mark.asyncio
async def test_saving_selectors_makes_the_next_check_rebaseline(monkeypatch):
    """The payoff (spec §3): the edit re-baselines instead of firing a phantom.

    Same page for every fetch, so nothing a human wrote ever changes. Adding
    `.promo` -- which this page HAS -- changes the extracted text, so the
    stored hash (computed under the old selectors) no longer matches. Without
    Task 3's fingerprint comparison that produces an item whose entire diff is
    the promo banner disappearing: the app reporting its own setting change
    back to the user as news from the site.

    The check before the save is the precondition that makes the assertion
    mean something -- it proves the page really is unchanged, so
    `baseline_stored` afterwards can only have come from the edit.
    """
    app = _build_test_app()
    db, service, source_id = await _seed_url_source(app)
    _serve(monkeypatch, [_NOISY_PAGE])

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        assert _dispositions(await _check(service, source_id)) == _counts(baseline=1)
        assert _dispositions(await _check(service, source_id)) == _counts(
            unchanged=1
        ), "precondition: the served page does not change between checks"

        await _select_real_source(pilot, screen, source_id)
        await _save_selectors(pilot, screen, ".ad\n.promo")
        for _ in range(40):
            await pilot.pause()
            if db.get_subscription(source_id)["ignore_selectors"] != ".ad":
                break

        after = await _check(service, source_id)
        assert _dispositions(after) == _counts(rebaselined=1), (
            "the check following a selector edit must re-baseline -- not "
            "report the now-stripped noise as a change the site made, and "
            "not report `unchanged` off a hash computed under the old settings. "
            "It counts as `rebaselined`, not `baseline`: this save cost the "
            "user a real diff window (whole-branch review, Critical 1)"
        )
        assert _stored_items(db, source_id) == [], "no phantom item may be stored"

        assert _dispositions(await _check(service, source_id)) == _counts(
            unchanged=1
        ), "and once re-baselined the very next check compares normally"

        # Fix round 1 (Minor 3): the `reason`, off the disposition dict the
        # aggregated run counts deliberately cannot carry. A second save --
        # this time a rule matching nothing on the page, so the extracted text
        # and therefore the hash are IDENTICAL -- is what distinguishes "the
        # fingerprint decided" from "the text happened to differ": only a
        # fingerprint comparison running BEFORE the hash comparison can
        # re-baseline here at all.
        await _save_selectors(pilot, screen, ".ad\n.promo\n.matches-nothing-at-all")
        for _ in range(40):
            await pilot.pause()
            if "matches-nothing" in db.get_subscription(source_id)["ignore_selectors"]:
                break

        item, disposition = await _direct_check(db, source_id)
        assert item is None
        assert disposition == {
            "kind": "baseline_stored",
            "reason": "extraction_settings_changed",
            "withheld_percentage": None,
        }, (
            "the re-baseline must say WHY it happened -- the Runs pane has to "
            "distinguish a first-ever check from a settings change the user "
            "just made in this very Inspector"
        )
        assert _stored_items(db, source_id) == []


@pytest.mark.asyncio
async def test_saving_selectors_does_not_recompose_the_screen():
    """Phase D Task 5's regression class, guarded for this new write path.

    `_refresh_overview_data()` sets `overview_data`, `reactive({}, recompose=
    True)` on the screen -- calling it (as `_create_source` does) rebuilds
    every region through its factory and replaces the mounted panes wholesale,
    which was proven live to detach the `ItemsPane`, reset the `DataTable`
    cursor and drop keyboard focus. Nothing visible is derived from a source's
    selectors, so `_save_noise_selectors` patches the entity dict in place
    instead; this pins that it stays that way.
    """
    app = _build_test_app()
    db, _service, source_id = await _seed_url_source(app)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        sources_pane = await _select_real_source(pilot, screen, source_id)
        table = sources_pane.query_one("#sources-table", DataTable)
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        field = inspector.query_one("#inspector-noise-selectors", TextArea)

        await _save_selectors(pilot, screen, ".ad\n.promo")
        for _ in range(40):
            await pilot.pause()
            if db.get_subscription(source_id)["ignore_selectors"] != ".ad":
                break
        for _ in range(20):
            await pilot.pause()

        assert sources_pane.is_attached and table.is_attached
        assert (
            screen.query_one("#watchlists-sources-pane", SourcesPane) is sources_pane
        ), (
            "saving selectors must not rebuild the screen's regions -- the "
            "same SourcesPane instance must still be mounted"
        )
        assert sources_pane.query_one("#sources-table", DataTable) is table
        assert screen.query_one("#watchlists-entity-inspector", InspectorPane) is inspector
        assert inspector.query_one("#inspector-noise-selectors", TextArea) is field, (
            "not even the Inspector may recompose: the entity dict is patched "
            "in place, so the field the user is typing in survives the save"
        )

        # The in-place patch is the reason no rebuild is needed -- the entity
        # every surface holds already reports the saved value.
        assert (
            InspectorPane._ignore_selectors_text(screen.selected_entity)
            == ".ad\n.promo"
        )


@pytest.mark.asyncio
async def test_a_save_that_cannot_write_says_so():
    """Fix round 1 (Minor 4): no silent no-op behind the Save button.

    An entity carrying no `id` cannot be written, and the handler used to
    `return` on it -- no write, no error, no toast, indistinguishable from a
    broken button. Driven through the real message rather than the private
    handler, since the message is the reachable surface.
    """
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    toasts = []
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        app.notify = lambda message, **kwargs: toasts.append((message, kwargs))

        screen.post_message(SaveNoiseSelectorsRequested(None, ".ad"))
        await pilot.pause()

        assert toasts, "a Save that cannot write anything must still say so"
        message, kwargs = toasts[-1]
        assert "Nothing to save" in message
        assert kwargs.get("severity") == "warning"


@pytest.mark.asyncio
async def test_a_successful_save_warns_that_the_next_check_loses_a_window():
    """Whole-branch review, Critical 1, third leg: warn at the source.

    Spec §3 sanctions the re-baseline's cost -- a change the page makes before
    the next check is compared against nothing and is never reported -- on the
    strength of the user being told. The Runs pane now says it after the fact,
    which is too late to act on; this toast says it at the one moment the user
    could still decide to wait for a check before saving.

    Asserted on the delivered message's SUBSTANCE, deliberately not against
    `NOISE_SELECTORS_SAVED_TOAST`: comparing to the constant would stay green
    if the warning sentence were deleted from the constant itself, which is the
    exact regression this guards.
    """
    app = _build_test_app()
    db, _service, source_id = await _seed_url_source(app)

    host = DestinationHarness(app, "watchlists_collections")
    toasts: list[tuple[str, dict]] = []
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        await _select_real_source(pilot, screen, source_id)
        app.notify = lambda message, **kwargs: toasts.append((str(message), kwargs))

        await _save_selectors(pilot, screen, ".ad\n.promo")
        for _ in range(40):
            await pilot.pause()
            if db.get_subscription(source_id)["ignore_selectors"] != ".ad":
                break
        for _ in range(10):
            await pilot.pause()

        assert toasts, "a successful save must confirm itself"
        message, kwargs = toasts[-1]
        assert kwargs.get("severity") == "information"
        assert "saved" in message.lower(), "precondition: this is the success toast"
        assert "re-baselines" in message
        # The added sentence: the consequence, not just the mechanism.
        # "re-baselines" alone is jargon that does not tell the user a change
        # can be lost.
        assert "will not be reported" in message, (
            "the toast must state the consequence of the re-baseline, not only "
            "that one will happen -- spec §3's cost is only acceptable if the "
            "user is told about it while they can still act on it"
        )
        assert "before" in message.lower(), (
            "and it must say WHEN: a change landing before the next check is "
            "the window that is lost"
        )


@pytest.mark.asyncio
async def test_a_save_with_an_unparseable_selector_is_refused_and_says_why(
    monkeypatch: pytest.MonkeyPatch,
):
    """Whole-branch fix F1, UI side (Inspector half).

    Same refusal as the create form, and the same reason: `soup.select` raises
    on anything CSS cannot parse, so an unparseable line silently suppresses
    nothing forever. Writing it would leave the source carrying a dead rule the
    user believes is working -- the extraction guard's log warning is not a
    place a TUI user looks. Mutation: delete the `first_invalid_selector` check
    in `_post_noise_selectors_save` and this reddens (the row is overwritten
    and no error toast arrives).

    Driven through the real button, so the message path and the screen handler
    are the ones that would run for a user.
    """
    app = _build_test_app()
    db, _service, source_id = await _seed_url_source(app)

    host = DestinationHarness(app, "watchlists_collections")
    toasts: list[tuple[str, dict]] = []
    diagnostic_warnings: list[str] = []
    monkeypatch.setattr(
        inspector_pane_module,
        "logger",
        SimpleNamespace(
            warning=lambda message: diagnostic_warnings.append(str(message))
        ),
    )
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        await _select_real_source(pilot, screen, source_id)
        # `FullAppDestinationContext` now runs the real TldwCli directly, so
        # the pane's `self.app` and the screen's `app_instance` are both `app`.
        app.notify = lambda message, **kwargs: toasts.append((str(message), kwargs))

        await _save_selectors(pilot, screen, ".ad\n:::nonsense\n.promo")
        for _ in range(30):
            await pilot.pause()

        assert db.get_subscription(source_id)["ignore_selectors"] == ".ad", (
            "the stored row must be untouched -- a refused save that still "
            "wrote would be worse than no validation at all"
        )

        assert toasts, "the refusal must be visible, not a silent no-op button"
        message, kwargs = toasts[-1]
        assert kwargs.get("severity") == "error"
        assert ":::nonsense" in message, (
            f"the toast must name the offending line; got {message!r}"
        )
        assert kwargs.get("markup") is False, (
            "selectors carry `[`, which Textual's toast markup would consume"
        )
        assert diagnostic_warnings
        assert ":::nonsense" not in diagnostic_warnings[-1]
        # The field keeps the user's text so the fix is one edit away.
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        assert (
            inspector.query_one("#inspector-noise-selectors", TextArea).text
            == ".ad\n:::nonsense\n.promo"
        )


@pytest.mark.asyncio
async def test_the_inspector_still_saves_every_shipped_default_selector():
    """The refusal's other direction, on the real prefill plus the hard cases.

    `DEFAULT_IGNORE_SELECTORS` is what a user saves first, so it must pass. It
    is NOT sufficient on its own: every default's comma groups happen to be
    valid selectors individually, so a validator that re-split lines on commas
    would still accept all of them (measured -- that mutation left this test
    green until `:is(.a, .b)` and `[data-x="a,b"]` were added, where splitting
    produces `:is(.a` and `[data-x="a`, neither of which parses).
    """
    app = _build_test_app()
    db, _service, source_id = await _seed_url_source(app)

    host = DestinationHarness(app, "watchlists_collections")
    toasts: list[tuple[str, dict]] = []
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        await _select_real_source(pilot, screen, source_id)
        # Both sinks: the pane's refusal would land on `host`, the screen's
        # success toast on `app`. Watching only one would let a refusal here
        # pass as "no error toast".
        host.notify = lambda message, **kwargs: toasts.append((str(message), kwargs))
        app.notify = lambda message, **kwargs: toasts.append((str(message), kwargs))

        wanted = "\n".join((*DEFAULT_IGNORE_SELECTORS, ":is(.a, .b)", '[data-x="a,b"]'))
        await _save_selectors(pilot, screen, wanted)
        for _ in range(40):
            await pilot.pause()
            if db.get_subscription(source_id)["ignore_selectors"] != ".ad":
                break

        assert db.get_subscription(source_id)["ignore_selectors"] == wanted, (
            "every shipped default line is valid CSS and must save"
        )
        assert not [t for t in toasts if t[1].get("severity") == "error"], (
            f"the shipped defaults produced an error toast: {toasts!r}"
        )


# --- Task 5: the "Queue for briefing" affordance (spec #2 phase 1) ---------
#
# The button, the row indicator, and the honest-failure path all reuse the
# same rules Task 1362's noise-selectors save proved out on this stream: no
# full-screen recompose, an in-place patch, and a targeted repaint instead of
# a rebuild.


def _seed_new_item(app, *, content_hash: str) -> tuple[Any, int, int]:
    """Seed one subscription with one "new" item through the real database.

    Seeded through `app.local_watchlists_service._db()` -- the connection
    the Items pane's own real load path resolves to (`_load_items` -> the
    controller -> `LocalWatchlistsService`) -- and NOT
    `app.watchlist_bundle_service.db`, even though the latter is what
    `WatchlistsCollectionsScreen._briefings_db()` (the queue-toggle write
    path) reaches. In the running app both resolve to the identical on-disk
    file, but in THIS harness they do not:
    `_build_test_app()`'s `get_subscriptions_db_path` patch only lives for
    the duration of `TldwCli.__init__`, so `watchlist_bundle_service`'s
    connection (built EAGERLY, inside that init, while the patch is live)
    and `local_watchlists_service`'s connection (built LAZILY, per call, once
    the patch has already exited) resolve to two DIFFERENT temp files here.
    `_open_items_with_seeded_item` below points `watchlist_bundle_service`
    at THIS connection once the screen's initial load has settled, so
    `_briefings_db()` agrees with what the Items pane reads.

    Returns:
        `(db, source_id, item_id)` -- `db` is the connection to read the
        written flag back from.
    """
    db = app.local_watchlists_service._db()
    source_id = db.add_subscription(
        name="Summit Route", type="rss", source="https://summitroute.com/blog/feed.xml"
    )
    with db.transaction() as conn:
        item_id = persist_subscription_item(
            conn,
            source_id,
            {
                "url": f"https://summitroute.com/blog/2024/{content_hash}/",
                "title": "Lightsail object storage concerns - Part 2",
                "content_hash": content_hash,
                "status": "new",
            },
            run_id=None,
            now="2026-07-30T09:00:00+00:00",
        )
    return db, source_id, item_id


def _queued_flag(db, item_id: int, *, status: str = "reviewed") -> bool | None:
    """A fresh read of one item's stored flag -- never the widget's state."""
    for row in db.get_new_items(status=status, limit=50):
        if row["id"] == item_id:
            return bool(row["queued_for_briefing"])
    return None


async def _open_items_with_seeded_item(pilot, screen, app, db):
    """Open Items, wait for the real load, then align the queue-write path.

    `app.watchlist_bundle_service._db` is pointed at `db` only AFTER the
    initial mount's background loads have settled (`pane.items` populated),
    never before: mounting the screen fires several CONCURRENT background
    loads that each construct a fresh `SubscriptionsDB(...)` against this
    same brand-new file, and reassigning earlier lets those races hit the
    one-time schema-migration gate on this connection's cached schema view.
    Observed directly: an early reassignment intermittently made the very
    next write on `db` raise `OperationalError: no such table:
    subscription_items`, which then self-healed on an immediate retry --
    proof it was a startup race, not a real absence of the table. Waiting
    for the settle removes the race instead of papering over it with a
    retry loop.
    """
    screen.active_section = "items"
    await pilot.pause(0.3)
    pane = screen.query_one("#watchlists-items-pane", ItemsPane)
    for _ in range(40):
        await pilot.pause()
        if pane.items:
            break
    assert pane.items, "the seeded item must reach the Items pane"
    table = pane.query_one("#items-table", DataTable)
    app.watchlist_bundle_service._db = db
    return pane, table


@pytest.mark.asyncio
async def test_pressing_queue_for_briefing_writes_the_flag_and_repaints_the_row():
    """Step 1: press the REAL button and follow the write all the way
    through -- the database flag, the Items-table indicator, and instance
    survival (the same `ItemsPane` -- Phase D pattern).

    Mutation (a): stub the handler into a no-op and this reddens on the
    database assertion, since nothing was ever written.
    """
    app = _build_test_app()
    db, _source_id, item_id = _seed_new_item(app, content_hash="queue-write")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane, table = await _open_items_with_seeded_item(pilot, screen, app, db)

        row_key = str(pane.items[0]["id"])
        assert table.get_row(row_key)[4] == "", (
            "precondition: a freshly seeded item is not queued"
        )

        pane.select_item_by_id(row_key)
        for _ in range(20):
            await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        button = inspector.query_one("#inspector-queue-briefing-button", Button)
        assert str(button.label) == "Queue for briefing", (
            "precondition: a not-yet-queued item's button names the action "
            "it is about to take"
        )
        button.press()
        # Fix round 1: the write now happens in a worker (`asyncio.to_thread`
        # then the in-place patch + repaint), so settle on the LAST
        # observable effect of that sequence -- the repainted cell -- rather
        # than the DB flag alone. The repaint only ever runs after the write
        # has already been awaited, so waiting on it also guarantees the DB
        # assertion below is no longer racing the worker.
        for _ in range(40):
            await pilot.pause()
            if table.get_row(row_key)[4] == ItemsPane._QUEUED_GLYPH:
                break

        assert _queued_flag(db, item_id) is True, (
            "the press must reach SubscriptionsDB.set_item_briefing_queued"
        )
        assert table.get_row(row_key)[4] == ItemsPane._QUEUED_GLYPH, (
            "the row indicator must repaint in place once the write succeeds"
        )
        assert str(button.label) == "Unqueue from briefing", (
            "the button's own label must flip too -- it is the only control "
            "and states the CURRENT value"
        )

        # Phase D pattern: no full-screen recompose. The very instances the
        # user was looking at must still be the ones mounted.
        assert screen.query_one("#watchlists-items-pane", ItemsPane) is pane
        assert pane.query_one("#items-table", DataTable) is table
        assert (
            screen.query_one("#watchlists-entity-inspector", InspectorPane) is inspector
        )


@pytest.mark.asyncio
async def test_the_queue_write_runs_off_the_event_loop_thread():
    """Fix round 1: pin the load-bearing part of moving the write off the
    UI thread. `run_worker` alone only *schedules* a coroutine back onto the
    SAME event loop -- it is `asyncio.to_thread` inside `_toggle_briefing_queue`
    that actually gets `set_item_briefing_queued` off it. A mutation that
    drops `to_thread` and calls the DB method directly (still correctly,
    still successfully) passes every other Task 5 test unchanged, since the
    end state -- flag set, cell repainted -- is identical either way; only
    watching WHICH thread executes the call can tell the two apart.
    """
    app = _build_test_app()
    db, _source_id, item_id = _seed_new_item(app, content_hash="queue-thread-ident")

    loop_thread_id = threading.get_ident()
    write_thread_ids: list[int] = []
    real_write = db.set_item_briefing_queued

    def _spy(item_id_arg, queued_arg):
        write_thread_ids.append(threading.get_ident())
        return real_write(item_id_arg, queued_arg)

    db.set_item_briefing_queued = _spy

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane, table = await _open_items_with_seeded_item(pilot, screen, app, db)
        row_key = str(pane.items[0]["id"])

        pane.select_item_by_id(row_key)
        for _ in range(20):
            await pilot.pause()
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        button = inspector.query_one("#inspector-queue-briefing-button", Button)
        button.press()
        for _ in range(40):
            await pilot.pause()
            if write_thread_ids:
                break

        assert write_thread_ids, "the write must have run at all"
        assert write_thread_ids[0] != loop_thread_id, (
            "set_item_briefing_queued must run off the event-loop thread "
            "(asyncio.to_thread), not synchronously inside the worker on "
            "the same thread that runs the event loop"
        )


@pytest.mark.asyncio
async def test_pressing_queue_for_briefing_again_unqueues_and_relabels():
    """Step 1's other half: toggling back clears the flag, the indicator,
    and restores the button's original label -- not a one-way ratchet."""
    app = _build_test_app()
    db, _source_id, item_id = _seed_new_item(app, content_hash="queue-toggle")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane, table = await _open_items_with_seeded_item(pilot, screen, app, db)
        row_key = str(pane.items[0]["id"])

        pane.select_item_by_id(row_key)
        for _ in range(20):
            await pilot.pause()
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        button = inspector.query_one("#inspector-queue-briefing-button", Button)

        # Fix round 1: settle on the repainted cell, the LAST effect of the
        # worker's write-then-patch sequence -- see the comment in
        # `test_pressing_queue_for_briefing_writes_the_flag_and_repaints_the_row`.
        button.press()
        for _ in range(40):
            await pilot.pause()
            if table.get_row(row_key)[4] == ItemsPane._QUEUED_GLYPH:
                break
        assert _queued_flag(db, item_id) is True, "precondition: first press queued it"
        assert table.get_row(row_key)[4] == ItemsPane._QUEUED_GLYPH

        button.press()
        for _ in range(40):
            await pilot.pause()
            if table.get_row(row_key)[4] == "":
                break

        assert _queued_flag(db, item_id) is False, (
            "the second press must clear the flag, not queue it again"
        )
        assert table.get_row(row_key)[4] == "", (
            "the indicator must clear along with the flag"
        )
        assert str(button.label) == "Queue for briefing", (
            "the label must read exactly as it did before either press"
        )


@pytest.mark.asyncio
async def test_the_queue_button_only_renders_for_item_selections():
    """AC#3 / mutation (c): a discriminator that always answers "item" (or
    that renders the button unconditionally) would leak "Queue for
    briefing" onto sources, runs, rules and notifications -- entities
    `set_item_briefing_queued` has no meaning for at all. Mirrors
    `test_the_editor_renders_only_for_url_family_sources`'s shape: real
    entities of every OTHER selectable kind (`OTHER_ENTITIES`, shared with
    `test_watchlists_item_actions.py` so this cannot be fixed for items by
    breaking one of those fixtures) are the negative control, and a real
    item (`REAL_ITEM`) is the positive one.
    """
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)

        for _kind, entity in OTHER_ENTITIES:
            screen.selected_entity = entity
            await pilot.pause()
            assert not inspector.query("#inspector-queue-briefing-button"), (
                f"a {_kind} selection must not offer the briefing queue toggle"
            )

        screen.selected_entity = REAL_ITEM
        await pilot.pause()
        assert inspector.query_one("#inspector-queue-briefing-button", Button), (
            "precondition: an item selection DOES offer it -- otherwise the "
            "loop above would pass even with the button deleted entirely"
        )


@pytest.mark.asyncio
async def test_a_failed_queue_write_leaves_the_flag_and_indicator_unchanged():
    """Honest failure: a DB error must not move the flag, must not repaint
    the indicator, and must say so -- and nothing may escape the handler.

    The mock replaces `set_item_briefing_queued` itself with a `Mock` that
    raises, so the write never reaches the real database at all; `_queued_flag`
    here confirms the stored flag stayed exactly as seeded, and the failure
    toast plus `screen.is_attached` confirm nothing escaped the worker.
    """
    app = _build_test_app()
    db, _source_id, item_id = _seed_new_item(app, content_hash="queue-fail")
    # Patches the INSTANCE method on `db` itself, so it stays in effect once
    # `_open_items_with_seeded_item` below points `_briefings_db()` -- the
    # screen handler's write path -- at this exact object.
    db.set_item_briefing_queued = Mock(side_effect=RuntimeError("disk full"))

    host = DestinationHarness(app, "watchlists_collections")
    toasts: list[tuple[str, dict]] = []
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane, table = await _open_items_with_seeded_item(pilot, screen, app, db)
        row_key = str(pane.items[0]["id"])

        pane.select_item_by_id(row_key)
        for _ in range(20):
            await pilot.pause()
        app.notify = lambda message, **kwargs: toasts.append((str(message), kwargs))

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        button = inspector.query_one("#inspector-queue-briefing-button", Button)
        button.press()
        # Fix round 1: the write now happens in a worker (`asyncio.to_thread`),
        # so settle on the observable outcome -- the error toast the failure
        # path fires -- rather than a fixed count of no-op pauses.
        for _ in range(60):
            await pilot.pause()
            if toasts:
                break

        assert _queued_flag(db, item_id) is False, (
            "a failed write must leave the stored flag exactly as it was"
        )
        assert table.get_row(row_key)[4] == "", (
            "a failed write must not repaint the indicator"
        )
        assert str(button.label) == "Queue for briefing", (
            "a failed write must not relabel the button either"
        )
        assert toasts, "the failure must be visible, not a silent no-op"
        message, kwargs = toasts[-1]
        assert kwargs.get("severity") == "error"
        assert "queue" in message.lower() or "briefing" in message.lower()
        # Nothing escaped the handler and crashed the app: reaching this
        # line at all, with the screen still mounted, is the proof.
        assert screen.is_attached


@pytest.mark.asyncio
async def test_the_queued_indicator_renders_from_the_normalized_flag_on_load():
    """Requirement 5: an item already queued (via a prior session, say)
    shows the glyph on a plain (first) load -- pinning Task 1's read path
    (`queued_for_briefing` surviving `get_new_items` ->
    `normalize_watchlist_item`) end to end, through the real controller,
    with no button press anywhere in this test. (This test never navigates
    away and back, so "load" rather than "reload" is the accurate word for
    what it exercises.)"""
    app = _build_test_app()
    db, _source_id, item_id = _seed_new_item(app, content_hash="queue-preloaded")
    db.set_item_briefing_queued(item_id, True)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane, table = await _open_items_with_seeded_item(pilot, screen, app, db)

        assert pane.items[0].get("queued_for_briefing") is True, (
            "the normalized item handed to the pane must already carry the "
            "flag -- the read path, not a write this test performs"
        )
        row_key = str(pane.items[0]["id"])
        assert table.get_row(row_key)[4] == ItemsPane._QUEUED_GLYPH, (
            "a pre-queued item must show the glyph as soon as it loads"
        )
