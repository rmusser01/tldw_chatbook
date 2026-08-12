"""One scoped rebuild per Watchlists section or tree interaction (task-15461).

Evidence for the three acceptance criteria, counted off the real screen
rather than reasoned about:

* A **section switch** rebuilds the section's own pane and the centre header,
  and nothing else -- not the screen, not the navigation bar, not the footer,
  not either rail, and not the reader.
* A **tree-node click** updates each affected pane at most once.
* **z / Z / [ / ]** rebuild only the region whose rendered form actually
  changed.

Every count comes from `Widget.recompose` and `Widget.mount`, patched for the
duration of one interaction (`_RebuildCounter`). Against the pre-task code
each of these fails: a section switch was a whole-screen
`refresh(recompose=True)` (measured: 75-176 mounted widgets, the navigation
bar and footer included) and any layout key recomposed the whole workbench
(36-99), because `WatchlistsWorkbench.region_layout` was `recompose=True`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from textual.widget import Widget
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Watchlists_Modules.artifacts_pane import ArtifactsPane
from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
    TreeScope,
    TreeScopeChanged,
    WatchlistTree,
)

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


class _RebuildCounter:
    """Counts `recompose()`/`mount()` per widget while it is installed.

    Patches the two `Widget` methods rather than watching for DOM diffs
    because "how many times was this torn down and rebuilt" is exactly the
    quantity these ACs are about, and a before/after DOM comparison cannot
    tell one rebuild from three.
    """

    def __init__(self) -> None:
        self.recomposes: Counter = Counter()
        self.mounts: Counter = Counter()
        self._original_recompose = Widget.recompose
        self._original_mount = Widget.mount

    @staticmethod
    def _key(widget: Widget) -> str:
        return f"{type(widget).__name__}#{widget.id or '-'}"

    def __enter__(self) -> "_RebuildCounter":
        counter = self

        async def _counting_recompose(widget):
            counter.recomposes[counter._key(widget)] += 1
            return await counter._original_recompose(widget)

        def _counting_mount(widget, *widgets, **kwargs):
            counter.mounts[counter._key(widget)] += len(widgets)
            return counter._original_mount(widget, *widgets, **kwargs)

        Widget.recompose = _counting_recompose
        Widget.mount = _counting_mount
        return self

    def __exit__(self, *exc_info) -> None:
        Widget.recompose = self._original_recompose
        Widget.mount = self._original_mount

    @property
    def total_mounts(self) -> int:
        return sum(self.mounts.values())

    def report(self) -> str:
        return f"recomposes={dict(self.recomposes)} mounts={dict(self.mounts)}"


def _seed(app, *, items: int = 2, briefings: int = 0) -> int:
    """A watchlist with a source, some items and optionally some briefings."""
    service = app.watchlist_bundle_service
    db = service.db
    watchlist = service.create("Morning AI Brief")
    watchlist_id = watchlist["id"]
    source_id = db.add_subscription(
        name="AI News", type="rss", source="https://ai-news.example/feed.xml"
    )
    service.add_source(watchlist_id, source_id)
    created = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    for index in range(items):
        with db.transaction() as conn:
            persist_subscription_item(
                conn,
                source_id,
                {
                    "url": f"https://ai-news.example/{index}",
                    "title": f"Story {index}",
                    "content": f"body of story {index}",
                    "content_hash": f"hash-{index}",
                    "content_kind": "article",
                    "content_format": "text",
                },
                run_id=None,
                now=created,
            )
    for n in range(briefings):
        briefing_id = db.insert_briefing(watchlist_id)
        db.update_briefing(
            briefing_id,
            status="complete",
            body_markdown=f"# Briefing {n}\n\nBody paragraph {n}.",
        )
    return watchlist_id


@asynccontextmanager
async def _open(app, watchlist_id: int | None = None, *, section: str = "items"):
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        if watchlist_id is not None:
            screen.tree_scope = TreeScope(
                kind="watchlist", watchlist_id=watchlist_id
            )
        screen.active_section = section
        await pilot.pause(0.4)
        yield screen, pilot, host


async def _settle(pilot, host) -> None:
    """Let both the message pump and every worker finish."""
    await pilot.pause()
    await host.workers.wait_for_complete()
    await pilot.pause()


# --------------------------------------------------------------------------
# AC#1 -- a section switch rebuilds the changed section only
# --------------------------------------------------------------------------


async def test_a_section_switch_never_recomposes_the_whole_screen():
    """The screen, its navigation bar and its footer are shared chrome that
    a tab click has nothing to say about.

    `watch_active_section` used to call `self.refresh(recompose=True)`, and
    `BaseAppScreen.compose` yields `MainNavigationBar` and `AppFooterStatus`
    around `compose_content`, so every tab click tore down and rebuilt both
    of them along with everything else.
    """
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
        from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar

        nav_before = screen.query_one(MainNavigationBar)
        footer_before = screen.query_one(AppFooterStatus)

        with _RebuildCounter() as counted:
            screen.active_section = "sources"
            await _settle(pilot, host)

        assert counted.recomposes["WatchlistsCollectionsScreen#-"] == 0, (
            f"a tab click must not recompose the screen: {counted.report()}"
        )
        assert screen.query_one(MainNavigationBar) is nav_before, (
            "the navigation bar must survive a tab click"
        )
        assert screen.query_one(AppFooterStatus) is footer_before, (
            "the footer must survive a tab click"
        )


async def test_a_section_switch_leaves_both_rails_standing():
    """The rails carry no `active_section`-derived state at all.

    The left rail matters most: `WatchlistTree.compose` runs one synchronous
    `list_source_rows` query per expanded watchlist, so rebuilding it for a
    tab click is database work on the UI thread with no reason behind it.
    """
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        tree_before = screen.query_one("#wl-tree", WatchlistTree)
        inspector_before = screen.query_one(
            "#watchlists-entity-inspector", InspectorPane
        )

        with _RebuildCounter() as counted:
            screen.active_section = "runs"
            await _settle(pilot, host)

        assert screen.query_one("#wl-tree", WatchlistTree) is tree_before, (
            f"the watchlist tree must survive a tab click: {counted.report()}"
        )
        assert (
            screen.query_one("#watchlists-entity-inspector", InspectorPane)
            is inspector_before
        ), f"the Inspector must survive a tab click: {counted.report()}"
        assert counted.recomposes["WatchlistTree#wl-tree"] == 0, counted.report()
        assert (
            counted.recomposes["InspectorPane#watchlists-entity-inspector"] == 0
        ), counted.report()


async def test_a_section_switch_builds_the_sections_pane_exactly_once():
    """One scoped rebuild, not two.

    The old path was a whole-screen recompose that built the new pane from
    whatever rows the screen happened to be holding, followed by the
    section's own loader landing a frame later and recomposing that same
    fresh pane a second time. The pane is now mounted with the data (see
    `_reseed_active_section_pane`), so the loader's push is a no-op.
    """
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        # Patched on the WORKBENCH's factory map, not on the screen: the map
        # captured the bound method at construction time, so rebinding the
        # screen attribute would count nothing.
        from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
            WatchlistsWorkbench,
        )

        workbench = screen.query_one(WatchlistsWorkbench)
        builds: list[str] = []
        real_build = workbench._content[Region.ITEMS]

        def _counting_build():
            builds.append(screen.active_section)
            return real_build()

        workbench._content[Region.ITEMS] = _counting_build

        with _RebuildCounter() as counted:
            screen.active_section = "rules"
            await _settle(pilot, host)

        assert builds == ["rules"], (
            f"the section's pane must be built exactly once: {builds}"
        )
        assert counted.recomposes["RulesPane#watchlists-rules-pane"] == 0, (
            "the freshly mounted pane must already carry the section's rows, "
            f"so the loader's push finds nothing to change: {counted.report()}"
        )


async def test_a_section_switch_moves_the_tab_strip_and_the_backend_control():
    """Behaviour, not just widget counts: the scoped swap must still render
    the section it was asked for, mark its tab, and disable the backend
    picker on a local-only section (`_LOCAL_ONLY_SECTIONS`).

    The backend Select is the one `active_section`-derived control outside
    the workbench, and it is now patched in place rather than rebuilt.
    """
    app = _build_test_app()
    watchlist_id = _seed(app, briefings=1)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        backend_select = screen.query_one("#watchlists-backend-select")
        assert backend_select.disabled is False

        screen.active_section = "artifacts"
        await _settle(pilot, host)

        assert screen.query("#watchlists-artifacts-pane"), (
            "the Artifacts pane must actually be on screen"
        )
        assert not screen.query("#watchlists-rules-pane"), (
            "the previous section's pane must be gone"
        )
        assert screen.query_one("#watchlists-backend-select") is backend_select, (
            "the backend Select is patched in place, never rebuilt"
        )
        assert backend_select.disabled is True, (
            "Artifacts is local-only, so the backend picker must be disabled"
        )
        assert screen.query("#watchlists-backend-label"), (
            "a local-only section must explain why the picker is disabled"
        )
        assert not screen.query("#wl-region-content"), (
            "CONTENT is unmounted off the Read tab"
        )

        screen.active_section = "items"
        await _settle(pilot, host)

        assert screen.query("#wl-region-content"), (
            "CONTENT must be mounted back on the Read tab"
        )
        assert backend_select.disabled is False, (
            "the backend picker must be re-enabled off a local-only section"
        )
        assert not screen.query("#watchlists-backend-label"), (
            "and the local-only explanation must go away with it"
        )


# --------------------------------------------------------------------------
# AC#2 -- one update per affected pane
# --------------------------------------------------------------------------


async def test_a_tree_click_updates_each_affected_pane_at_most_once():
    """A tree click fans out to the Inspector, the tree's own highlight, the
    centre header and the section's rows. Each of those is one update; none
    of them is two, and the screen is not involved.
    """
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id, section="sources") as (screen, pilot, host):
        with _RebuildCounter() as counted:
            screen.post_message(TreeScopeChanged(TreeScope(kind="all")))
            await _settle(pilot, host)

        assert counted.recomposes["WatchlistsCollectionsScreen#-"] == 0, (
            f"a tree click must not recompose the screen: {counted.report()}"
        )
        assert counted.recomposes["WatchlistsWorkbench#wl-workbench"] == 0, (
            f"nor the whole workbench: {counted.report()}"
        )
        for key, count in counted.recomposes.items():
            assert count <= 1, (
                f"{key} was rebuilt {count} times by one tree click: "
                f"{counted.report()}"
            )
        assert screen.tree_scope == TreeScope(kind="all"), (
            "the click still has to move the scope"
        )


async def test_a_briefing_selection_costs_one_pane_recompose():
    """The select->clear->reload pipeline, coalesced.

    Selecting a briefing moved three things in three separate instants: the
    selection itself, then the screen clearing the previous briefing's
    scripts/audio/citations off the pane one reactive at a time, then the
    reload landing. The clearing now rides the recompose the selection has
    already queued (`ArtifactsPane._clear_selection_derived_state`), so a
    selection whose stale state has to be dropped costs ONE rebuild.
    """
    app = _build_test_app()
    watchlist_id = _seed(app, briefings=2)
    async with _open(app, watchlist_id, section="artifacts") as (screen, pilot, host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        rows = list(pane.briefings)
        assert len(rows) == 2, f"the fixture needs two briefings: {rows}"

        # Stand in for "the previously selected briefing had a cast script and
        # a resolved citation": without stale state to drop, the clearing is
        # a no-op and this test would pass for the wrong reason.
        pane.selected_briefing = rows[0]
        await _settle(pilot, host)
        pane.set_reactive(
            ArtifactsPane.scripts, [{"id": 1, "status": "complete", "turns": []}]
        )
        pane.set_reactive(
            ArtifactsPane.citations,
            [{"item_id": 1, "label": "item 1", "available": True}],
        )

        with _RebuildCounter() as counted:
            pane.selected_briefing = rows[1]
            await _settle(pilot, host)

        assert counted.recomposes["ArtifactsPane#watchlists-artifacts-pane"] == 1, (
            f"one selection must cost one pane rebuild: {counted.report()}"
        )
        assert pane.scripts == [], "the previous briefing's scripts must be gone"
        assert pane.citations == [], "and so must its citations"
        assert pane.selected_briefing == rows[1]


# --------------------------------------------------------------------------
# AC#3 -- a layout key rebuilds only the toggled region
# --------------------------------------------------------------------------


async def test_a_rail_toggle_rebuilds_only_the_toggled_region():
    """`[` collapses the left rail. Nothing in the centre or on the right has
    changed, so nothing there may be rebuilt."""
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        items_region = screen.query_one("#wl-region-items")
        content_region = screen.query_one("#wl-region-content")
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        header = screen.query_one("#wl-centre-status")

        with _RebuildCounter() as counted:
            await pilot.press("[")
            await _settle(pilot, host)

        assert screen.query("#wl-header-left_rail"), "the rail really did collapse"
        assert not screen.query("#wl-region-left_rail")
        assert screen.query_one("#wl-region-items") is items_region, counted.report()
        assert screen.query_one("#wl-region-content") is content_region, (
            counted.report()
        )
        assert (
            screen.query_one("#watchlists-entity-inspector", InspectorPane)
            is inspector
        ), counted.report()
        assert screen.query_one("#wl-centre-status") is header, counted.report()
        assert counted.recomposes["WatchlistsWorkbench#wl-workbench"] == 0, (
            f"a rail toggle must not recompose the workbench: {counted.report()}"
        )

        with _RebuildCounter() as counted_back:
            await pilot.press("[")
            await _settle(pilot, host)

        assert screen.query("#wl-region-left_rail"), "and it expands again"
        assert screen.query_one("#wl-region-items") is items_region, (
            counted_back.report()
        )


async def test_collapsing_items_rebuilds_neither_rail_nor_the_reader():
    """`z` on ITEMS. CONTENT stays expanded, so it keeps its instance -- and
    picks up the sole-centre marker in place rather than by being rebuilt."""
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        tree = screen.query_one("#wl-tree", WatchlistTree)
        reader = screen.query_one("#watchlists-content-pane", ContentPane)
        content_region = screen.query_one("#wl-region-content")
        assert not content_region.has_class("watchlists-region-sole-centre")

        screen.focused_region = Region.ITEMS
        with _RebuildCounter() as counted:
            screen.action_toggle_region()
            await _settle(pilot, host)

        assert screen.query("#wl-header-items"), "ITEMS really did collapse"
        assert not screen.query("#wl-region-items")
        assert screen.query_one("#wl-tree", WatchlistTree) is tree, counted.report()
        assert screen.query_one("#watchlists-content-pane", ContentPane) is reader, (
            f"the reader must not be rebuilt by its neighbour collapsing: "
            f"{counted.report()}"
        )
        assert screen.query_one("#wl-region-content") is content_region
        assert content_region.has_class("watchlists-region-sole-centre"), (
            "the sole-expanded marker must be applied in place"
        )


async def test_soloing_content_relabels_the_reader_without_rebuilding_it():
    """`Z` on CONTENT collapses ITEMS around it. CONTENT's own form does not
    change, so the reader survives -- and its Expand/Restore button has to be
    relabelled in place or it would keep offering the action it just did."""
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        items_pane = screen.query_one("#watchlists-items-pane")
        items_pane.items = [
            {
                "id": "local:watchlist_item:1",
                "item_id": 1,
                "title": "Story 0",
                "source_name": "AI News",
                "status": "new",
                "content": "body",
                "content_kind": "article",
                "content_format": "text",
            }
        ]
        await _settle(pilot, host)
        items_pane.select_item_by_id("local:watchlist_item:1")
        await _settle(pilot, host)

        reader = screen.query_one("#watchlists-content-pane", ContentPane)
        assert str(reader.query_one("#content-expand-button", Button).label) == "Expand"

        screen.focused_region = Region.CONTENT
        with _RebuildCounter() as counted:
            screen.action_solo_region()
            await _settle(pilot, host)

        assert screen.region_layout.solo_region is Region.CONTENT
        assert not screen.query("#wl-region-items"), "solo really collapsed ITEMS"
        assert screen.query_one("#watchlists-content-pane", ContentPane) is reader, (
            f"soloing CONTENT must not rebuild CONTENT: {counted.report()}"
        )
        assert (
            str(reader.query_one("#content-expand-button", Button).label) == "Restore"
        ), "the surviving reader must be relabelled in place"


async def test_an_inspector_toggle_leaves_the_centre_and_the_left_rail_alone():
    """`]` collapses the right rail: the mirror image of the `[` test, so a
    regression that scoped one rail and not the other cannot hide."""
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        tree = screen.query_one("#wl-tree", WatchlistTree)
        items_region = screen.query_one("#wl-region-items")

        with _RebuildCounter() as counted:
            await pilot.press("]")
            await _settle(pilot, host)

        assert screen.query("#wl-header-right_rail"), "the Inspector really collapsed"
        assert not screen.query("#wl-region-right_rail")
        assert screen.query_one("#wl-tree", WatchlistTree) is tree, counted.report()
        assert screen.query_one("#wl-region-items") is items_region, counted.report()
        assert counted.recomposes["WatchlistsWorkbench#wl-workbench"] == 0, (
            counted.report()
        )


async def test_a_layout_toggle_mounts_far_fewer_widgets_than_a_full_rebuild():
    """The headline number, asserted rather than described.

    A `[` used to run every region factory: on this fixture the pre-task code
    mounted 52-76 widgets for one rail toggle (the tree, the tab strip, the
    article list and the Inspector among them). The scoped path rebuilds one
    region. The bound is deliberately loose -- this is a guard against the
    reactive silently going back to `recompose=True`, not a golden number.
    """
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        with _RebuildCounter() as counted:
            await pilot.press("[")
            await _settle(pilot, host)

        assert counted.total_mounts <= 12, (
            f"one rail toggle mounted {counted.total_mounts} widgets: "
            f"{counted.report()}"
        )


async def test_layout_keys_still_persist_and_restore_the_layout():
    """Scoping the rebuild must not change what the keys MEAN.

    Collapse state, solo/restore and the refusal off the Read tab are all
    behaviour the perf work is not allowed to move.
    """
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        await pilot.press("[")
        await _settle(pilot, host)
        assert Region.LEFT_RAIL in screen.region_layout.collapsed

        await pilot.press("]")
        await _settle(pilot, host)
        assert Region.RIGHT_RAIL in screen.region_layout.collapsed
        assert Region.LEFT_RAIL in screen.region_layout.collapsed, (
            "one rail's toggle must not disturb the other's"
        )

        screen.focused_region = Region.CONTENT
        screen.action_solo_region()
        await _settle(pilot, host)
        assert screen.region_layout.solo_region is Region.CONTENT
        assert not screen.query("#wl-region-items")

        screen.focused_region = Region.CONTENT
        screen.action_solo_region()
        await _settle(pilot, host)
        assert screen.region_layout.solo_region is None, "a second Z restores"
        assert screen.query("#wl-region-items")

        # Off Read, the centre regions are not the user's to collapse.
        screen.active_section = "sources"
        await _settle(pilot, host)
        before = screen.region_layout
        screen.focused_region = Region.ITEMS
        screen.action_toggle_region()
        await _settle(pilot, host)
        assert screen.region_layout == before, (
            "a centre-region gesture off the Read tab must still be refused"
        )
        assert screen.query("#wl-region-items"), (
            "and the section's own pane must still be showing"
        )


async def test_a_collapsed_region_still_expands_from_its_header_button() -> None:
    """The chevron route through `RegionToggled`, not just the keybinding."""
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        await pilot.press("[")
        await _settle(pilot, host)
        header = screen.query_one("#wl-header-left_rail", Button)

        header.press()
        await _settle(pilot, host)

        assert screen.query("#wl-region-left_rail"), (
            "clicking a collapsed region's header must expand it"
        )
        assert screen.query("#wl-tree"), "and rebuild its content"
