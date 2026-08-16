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

import pytest
from textual.widget import Widget
from textual.css.query import NoMatches
from textual.widgets import Button, DataTable

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.Subscriptions.briefing_cast import dump_roster
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Watchlists_Modules.artifacts_pane import ArtifactsPane
from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region
from tldw_chatbook.UI.Watchlists_Modules.rules_pane import RulesPane
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


async def _seed_alert_rule(app, name: str = "Rule One") -> None:
    """One alert rule, through the real service.

    Review minor (1): without a rule, `RulesPane.rules` is `[]` both before
    and after the loader lands, so "the loader's push found nothing to
    change" would hold for a pane that was never seeded at all. A row makes
    the assertion able to tell those two apart.
    """
    await app.local_watchlists_service.create_alert_rule(
        name=name, condition_type="no_items"
    )


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
    """One scoped region build per switch, and no duplicate pane rebuild.

    The old path was a whole-screen recompose that built the new pane from
    whatever rows the screen happened to be holding, followed by the section's
    own loader landing a frame later and recomposing that same fresh pane a
    second time.

    What is left, stated exactly. Review minor (1) added the seeded alert rule
    below, and it earned its place immediately: with a real row in play the
    pane still rebuilds ONCE per switch, and root-causing that turned out to
    be worth doing.

    * The **region** is built exactly once per switch -- that is `builds`.
    * The **pane** rebuilds at most once, and the one is not a data-arrival
      duplicate. `_build_detail_pane` used to seed `rules_pane.rules` by
      plain assignment on a freshly-constructed pane whose class default is
      `[]`, so Textual queued a recompose (`[] != [row]`) that fired just
      after the pane mounted -- the residual this task recorded, closed by
      task-15778 (the factories now seed `recompose=True` reactives with
      `set_reactive`, per-reactive, keeping `RunsPane`'s load-bearing
      `selected_run` watcher on the plain path). What `<= 1` still allows on
      a COLD visit is the genuine data arrival: the loader can land after
      the mount and push `[] -> [row]`, which is one honest rebuild.

    The claim this test therefore pins is the one AC#1 makes: **one scoped
    region build, and no SECOND rebuild from the loader landing** -- verified
    on a warm revisit, where the rows are already on screen state (and where,
    since task-15778, the count is exactly zero).
    """
    app = _build_test_app()
    watchlist_id = _seed(app)
    await _seed_alert_rule(app)
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
        rules_pane = screen.query_one("#watchlists-rules-pane", RulesPane)
        assert len(rules_pane.rules) == 1, (
            "the fixture's rule has to reach the pane, or the recompose count "
            f"below cannot discriminate: {rules_pane.rules}"
        )
        assert counted.recomposes["RulesPane#watchlists-rules-pane"] <= 1, (
            f"never a second full rebuild of the pane: {counted.report()}"
        )

        # Warm revisit: the rows are already on screen state, so the pane is
        # built carrying them and the loader's push must change nothing. This
        # is the half that used to cost a second full rebuild every time.
        builds.clear()
        screen.active_section = "sources"
        await _settle(pilot, host)
        with _RebuildCounter() as warm:
            screen.active_section = "rules"
            await _settle(pilot, host)

        assert builds == ["sources", "rules"], builds
        assert warm.recomposes["RulesPane#watchlists-rules-pane"] == 0, (
            "a warm revisit must not rebuild the freshly built pane at all "
            "-- the pre-mount seeding recompose this used to tolerate was "
            f"removed by task-15778: {warm.report()}"
        )
        assert warm.recomposes["WatchlistsCollectionsScreen#-"] == 0, warm.report()
        assert (
            screen.query_one("#rules-table", DataTable).row_count == 1
        ), "and the row still has to be on screen"


async def test_a_section_switch_shows_rows_that_land_while_the_swap_runs():
    """The window `_reseed_active_section_pane` closes.

    `watch_active_section` dispatches the section's loader and the swap in the
    same breath, and `refresh_region_content` calls the region factory BEFORE
    its remove/mount awaits (so a raising factory leaves the screen intact).
    A loader landing in that gap writes rows to `self._loaded_*` and then
    fails to find its pane -- built, not yet mounted -- and nothing is left to
    correct it. Textual's own `Widget.recompose` never had the gap: it removes
    children first and calls `compose()` afterwards.

    Neutering `_reseed_active_section_pane` to a no-op reds this test: the
    Alert-rules table stays empty over a `_loaded_rules` holding the row.
    """
    from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
        WatchlistsWorkbench,
    )

    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id, section="sources") as (screen, pilot, host):
        assert not screen.query("#watchlists-rules-pane")

        # Reconstruct the window rather than race for it, the same way
        # `test_run_detail_lands_when_the_push_happens_in_the_mount_window`
        # reconstructs the mount window. The loader's two observable effects
        # are "mirror the rows onto the screen" and "push them into the pane";
        # replaying them from inside the factory call puts them exactly where
        # the real gap is -- after the factory has read screen state, before
        # its replacement is mounted, so the push finds nothing.
        row = {
            "id": "r1",
            "name": "Rule One",
            "condition_type": "no_items",
            "severity": "warning",
            "enabled": True,
        }
        # No real loader may run: it would land AFTER the mount and repair the
        # pane itself, which is not what this test is about.
        screen._load_active_section_data = lambda: None

        workbench = screen.query_one(WatchlistsWorkbench)
        real_build = workbench._content[Region.ITEMS]

        def _build_then_land():
            built = real_build()
            screen._loaded_rules = [row]
            try:  # the loader's push, from inside the gap
                screen.query_one("#watchlists-rules-pane", RulesPane).rules = [row]
            except NoMatches:
                pass
            return built

        workbench._content[Region.ITEMS] = _build_then_land

        screen.active_section = "rules"
        await _settle(pilot, host)

        assert screen._loaded_rules == [row], "precondition: the rows did land"
        table = screen.query_one("#rules-table", DataTable)
        assert table.row_count == 1, (
            "rows that landed while the swap was mid-flight must reach the "
            "pane the swap mounted, not be stranded in `_loaded_rules`"
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


async def test_clicking_a_tab_leaves_focus_on_that_tab_so_z_stays_refused():
    """A tab click must not hand the left rail to the next keypress.

    Review finding (Important 1), measured A/B. A tab click focuses the tab
    `Button`, which lives in the centre header -- and the swap rebuilds that
    header, so the focused widget is removed from under the user. Textual's
    `Screen._reset_focus` then picks the first focusable replacement it finds,
    which on this screen is a LEFT RAIL tree node; `on_descendant_focus` reads
    that and sets `focused_region = LEFT_RAIL`, so the very next `z` collapsed
    the rail AND persisted the collapse to config. Pre-task, the whole-screen
    recompose left focus at `None` and `z` was refused.

    Re-focusing the rebuilt tab restores the refusal by the honest route:
    focus is inside `#wl-centre-status`, so `_focus_in_centre_header` is set,
    which is what `action_toggle_region` already consults. (`focused_region`
    may still read LEFT_RAIL from the instant Textual re-homed focus -- that
    stale value is precisely what `_focus_in_centre_header` exists to
    neutralise, per task-1344.)
    """
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        await pilot.click("#wl-tab-sources")
        await _settle(pilot, host)

        assert screen.active_section == "sources", "precondition: the tab switched"
        focused = screen.focused
        assert focused is not None and focused.id == "wl-tab-sources", (
            f"focus must stay on the tab the user clicked, not be re-homed by "
            f"Textual: {focused!r}"
        )
        assert screen._focus_in_centre_header is True, (
            "focus in the centre header is what makes z/Z refuse"
        )

        before = screen.region_layout
        await pilot.press("z")
        await _settle(pilot, host)

        assert screen.region_layout == before, (
            "z straight after a tab click must change nothing at all"
        )
        assert Region.LEFT_RAIL not in screen.region_layout.collapsed, (
            "and above all it must not collapse -- and persist -- the rail the "
            "user never aimed at"
        )
        assert screen.query("#wl-region-left_rail"), "the rail is still expanded"


async def test_a_section_switch_driven_from_elsewhere_does_not_steal_focus():
    """The other half of the focus rule: only re-home focus the swap destroyed.

    A section change can come from a deep link or `EditRuleRequested` while
    the user is working somewhere the swap does not touch. Moving them to the
    tab strip then would be its own defect, so the restore is gated on
    `_swap_will_destroy_focus`.
    """
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        tree_node = screen.query_one("#wl-tree", WatchlistTree).query(Button).first()
        tree_node.focus()
        await _settle(pilot, host)
        assert screen.focused is tree_node, "precondition: focus is in the rail"

        screen.active_section = "runs"
        await _settle(pilot, host)

        assert screen.focused is tree_node, (
            "a section switch that did not unmount the focused widget must "
            f"leave focus alone: {screen.focused!r}"
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


async def test_a_briefing_selection_never_recomposes_the_pane():
    """The select->clear->reload pipeline, fully scoped.

    Selecting a briefing moved three things in three separate instants: the
    selection itself, then the screen clearing the previous briefing's
    scripts/audio/citations off the pane one reactive at a time, then the
    reload landing. Task-15461 coalesced that to ONE pane recompose (the
    clearing rides the selection's own rebuild) -- and recorded that the one
    remaining recompose still destroyed the briefings table under the
    user's cursor. Task-15779 retires it: the selection-derived reactives
    no longer recompose the pane at all; they rebuild only
    `BriefingDetailRegion`, so a selection costs ZERO pane rebuilds (the
    table survives -- pinned end-to-end by
    `test_watchlists_artifacts_selection_in_place.py`) and at most one
    region rebuild per message-pump drain, with the clearing unchanged.
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

        assert counted.recomposes["ArtifactsPane#watchlists-artifacts-pane"] == 0, (
            "task-15779: a selection must never rebuild the pane -- that "
            "rebuild is what destroyed the briefings table (and its focus, "
            f"cursor and scroll) under the user: {counted.report()}"
        )
        assert (
            counted.recomposes["BriefingDetailRegion#artifacts-detail-region"] == 1
        ), (
            "the selection, its synchronous clearing and its reload landing "
            "must coalesce into ONE detail-region rebuild: "
            f"{counted.report()}"
        )
        assert pane.scripts == [], "the previous briefing's scripts must be gone"
        assert pane.citations == [], "and so must its citations"
        assert pane.selected_briefing == rows[1]


_SCRIPT_ROSTER = [{"name": "Narrator", "role_prompt": "Calm narration."}]


async def test_a_script_selection_never_recomposes_the_briefing_detail_region():
    """Task-16852: the identical guarantee, one level down.

    Task-15779 retired the briefing-selection rebuild but disclosed, as
    deliberately unexpanded scope, that a SCRIPT selection still recomposed
    the whole `BriefingDetailRegion` -- scripts `DataTable` included.
    `selected_script`/`script_audio` now rebuild only `ScriptDetailRegion`,
    a second boundary nested inside `BriefingDetailRegion`, so a script
    selection costs ZERO region rebuilds and ZERO pane rebuilds -- only the
    nested sub-region.
    """
    app = _build_test_app()
    watchlist_id = _seed(app, briefings=1)
    db = app.watchlist_bundle_service.db
    briefing_id = db.list_briefings(watchlist_id)[0]["id"]
    script_ids = [
        db.insert_briefing_script(
            briefing_id,
            preset_id=None,
            preset_name=f"Preset-{index}",
            roster_snapshot_json=dump_roster(_SCRIPT_ROSTER),
            status="complete",
        )
        for index in range(2)
    ]

    async with _open(app, watchlist_id, section="artifacts") as (screen, pilot, host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.selected_briefing = pane.briefings[0]
        await _settle(pilot, host)
        assert len(pane.scripts) == 2, f"the fixture needs two scripts: {pane.scripts}"

        with _RebuildCounter() as counted:
            pane.select_script_by_id(str(script_ids[0]))
            await _settle(pilot, host)

        assert counted.recomposes["ArtifactsPane#watchlists-artifacts-pane"] == 0, (
            "a script selection must never rebuild the pane: "
            f"{counted.report()}"
        )
        assert (
            counted.recomposes["BriefingDetailRegion#artifacts-detail-region"] == 0
        ), (
            "task-16852: a script selection must never rebuild the WHOLE "
            "detail region either -- that rebuild is what destroyed the "
            f"scripts table (and its focus, cursor and scroll): {counted.report()}"
        )
        assert (
            counted.recomposes[
                "ScriptDetailRegion#artifacts-script-detail-region"
            ]
            == 1
        ), (
            "the script selection and its audio reload landing must "
            f"coalesce into ONE script-detail-region rebuild: {counted.report()}"
        )
        assert pane.selected_script is not None
        assert pane.selected_script["id"] == script_ids[0]


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


async def test_z_collapsing_a_centre_region_is_not_one_way():
    """The keyboard round trip, not just the chevron.

    A collapsed region renders as a focusable header (`#wl-header-items`) and
    `on_descendant_focus` maps that id back to the region, so `z` with the
    header focused has to expand it again. Worth pinning next to the scoping
    work: the scoped path removes the widget that had focus and mounts a
    different one in its place, which is exactly where a "collapse is one
    way" regression would come from.
    """
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        screen.focused_region = Region.ITEMS
        screen.action_toggle_region()
        await _settle(pilot, host)
        assert screen.query("#wl-header-items"), "ITEMS collapsed"

        screen.query_one("#wl-header-items", Button).focus()
        await _settle(pilot, host)
        assert screen.focused_region is Region.ITEMS, (
            "focusing a collapsed region's header must point the keybinding "
            "at that region"
        )

        await pilot.press("z")
        await _settle(pilot, host)

        assert screen.query("#wl-region-items"), "z must expand it again"
        assert screen.query("#watchlists-items-pane"), (
            "and the region has to come back with its pane, not empty"
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
