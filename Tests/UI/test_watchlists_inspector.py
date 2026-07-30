"""Tests for the Watchlists inspector pane wiring."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
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
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticWatchlistsScopeService,
    _active_destination_screen,
)
from Tests.UI.test_destination_visual_parity_correction import (
    _assert_visible_in_viewport,
    _visual_destination_harness,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Subscriptions.noise_defaults import DEFAULT_IGNORE_SELECTORS
from tldw_chatbook.UI.Screens.watchlists_collections_screen import WatchlistsCollectionsScreen
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import (
    BreadcrumbScopeSelected,
    InspectorPane,
    SaveNoiseSelectorsRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemSelected
from tldw_chatbook.UI.Watchlists_Modules.notifications_pane import (
    NotificationSelected,
    RefreshNotificationsRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.rules_pane import RuleSelected
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope, TreeScopeChanged

# Whole-branch review (Important): without this, CI's `pytest -m unit` run
# DESELECTS this entire module. See the identical note in
# `test_watchlists_item_actions.py`.
pytestmark = pytest.mark.unit


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
            RuleSelected({"rule_id": 3, "name": "Price drop", "condition_type": "keyword"})
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
    * It ran on `DestinationHarness`, which declares no `CSS_PATH`. None of
      the shipped rules applied -- `styles.max_height` was `None`, i.e. the
      probe measured framework defaults, not `#inspector-noise-selectors {
      max-height: 4 }`. This runs on `_visual_destination_harness`, the
      production-stylesheet harness `test_watchlists_source_create_form.py`
      and the visual parity suite already use, so the geometry measured is
      the geometry that ships.

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
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=size) as pilot:
        await pilot.pause(0.2)
        screen = _active_destination_screen(host)
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
        assert _dispositions(await _check(service, source_id)) == _counts(unchanged=1), (
            "precondition: the served page does not change between checks"
        )

        await _select_real_source(pilot, screen, source_id)
        await _save_selectors(pilot, screen, ".ad\n.promo")
        for _ in range(40):
            await pilot.pause()
            if db.get_subscription(source_id)["ignore_selectors"] != ".ad":
                break

        after = await _check(service, source_id)
        assert _dispositions(after) == _counts(baseline=1), (
            "the check following a selector edit must re-baseline -- not "
            "report the now-stripped noise as a change the site made, and "
            "not report `unchanged` off a hash computed under the old settings"
        )
        assert _stored_items(db, source_id) == [], "no phantom item may be stored"

        assert _dispositions(await _check(service, source_id)) == _counts(unchanged=1), (
            "and once re-baselined the very next check compares normally"
        )

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
        assert screen.query_one("#watchlists-sources-pane", SourcesPane) is sources_pane, (
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
        assert InspectorPane._ignore_selectors_text(screen.selected_entity) == ".ad\n.promo"


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
