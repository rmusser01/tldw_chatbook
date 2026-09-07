"""One scoped rebuild per Watchlists section or tree interaction (task-15461).

Evidence for the three acceptance criteria, counted off the real screen
rather than reasoned about:

* A **section switch** rebuilds the section's own pane and the centre header,
  and nothing else -- not the screen, not the navigation bar, not the footer,
  not either rail, and not the reader.
* A **tree-node click** updates each affected pane at most once.
* Side-pane layout actions mount or remove only the affected pane body.

Every count comes from `Widget.recompose` and `Widget.mount`, patched for the
duration of one interaction (`_RebuildCounter`). Against the pre-task code
each of these fails: a section switch was a whole-screen
`refresh(recompose=True)` (measured: 75-176 mounted widgets, the navigation
bar and footer included) and any layout key recomposed the whole workbench
(36-99), because `WatchlistsWorkbench.region_layout` was `recompose=True`.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections import Counter
from datetime import datetime, timedelta, timezone
import threading
from unittest.mock import AsyncMock

import pytest
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.css.query import NoMatches
from textual.widgets import Button, DataTable, ListView

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness, _static_text
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.Subscriptions.briefing_cast import dump_roster
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    ManualLayoutRollback,
    ResponsivePriorityLease,
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane
from tldw_chatbook.UI.Watchlists_Modules.artifacts_pane import ArtifactsPane
from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane
from tldw_chatbook.UI.Watchlists_Modules.pane_grip import WatchlistsPaneGrip
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region, RegionLayout
from tldw_chatbook.UI.Watchlists_Modules.rules_pane import RulesPane
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
    TreeScope,
    TreeScopeChanged,
    WatchlistTree,
)
from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
    RegionLayoutApplied,
    RegionLayoutApplyFailed,
    WatchlistsWorkbench,
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


async def _mount_failed_reader_return(screen, pilot, host, scope):
    """Commit a management scope, then mount its failed Reader retry state."""
    original_list_page = screen._controller.list_reader_items_page
    screen.active_section = "sources"
    await _settle(pilot, host)
    screen._request_tree_scope(scope)
    failed_list_page = AsyncMock(side_effect=RuntimeError("offline"))
    screen._controller.list_reader_items_page = failed_list_page
    screen.active_section = "items"
    await _settle(pilot, host)
    return original_list_page, failed_list_page


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


async def test_inspector_preference_and_grip_action_are_shared_across_all_tabs(
    monkeypatch,
) -> None:
    """One Inspector preference governs Read and every management tab."""
    writes: list[RegionLayout] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.load_region_layout",
        lambda: RegionLayout(collapsed=frozenset({Region.RIGHT_RAIL})),
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: writes.append(layout) or True,
    )
    app = _build_test_app()
    pane_ids = {
        "sources": "watchlists-sources-pane",
        "runs": "watchlists-runs-pane",
        "rules": "watchlists-rules-pane",
        "notifications": "watchlists-notifications-pane",
        "artifacts": "watchlists-artifacts-pane",
        "overview": "watchlists-overview-pane",
    }

    async with _open(app) as (screen, pilot, host):
        collapsed_grip = screen.query_one(
            "#wl-grip-right_rail", WatchlistsPaneGrip
        )
        assert str(collapsed_grip.label) == "<---"
        assert collapsed_grip.tooltip == "Expand Inspector"

        collapsed_grip.press()
        await _settle(pilot, host)
        assert writes == [RegionLayout()]

        for section, pane_id in pane_ids.items():
            screen.active_section = section
            await _settle(pilot, host)

            assert screen.region_layout == RegionLayout(), section
            grip = screen.query_one("#wl-grip-right_rail", WatchlistsPaneGrip)
            assert grip.expanded is True, section
            assert str(grip.label) == "--->", section
            assert grip.tooltip == "Collapse Inspector", section
            assert screen.query(f"#{pane_id}"), section
            assert screen.query("#wl-region-items"), section
            assert not screen.query("#watchlists-items-pane"), section
            assert not screen.query("#wl-grip-items"), section
            assert writes == [RegionLayout()], (
                f"visiting {section} must not persist an unchanged preference"
            )

        screen.query_one("#wl-grip-right_rail", WatchlistsPaneGrip).press()
        await _settle(pilot, host)
        expected_collapsed = RegionLayout(
            collapsed=frozenset({Region.RIGHT_RAIL})
        )
        assert writes == [RegionLayout(), expected_collapsed]

        screen.active_section = "items"
        await _settle(pilot, host)
        grip = screen.query_one("#wl-grip-right_rail", WatchlistsPaneGrip)
        assert screen.region_layout == expected_collapsed
        assert grip.expanded is False
        assert str(grip.label) == "<---"
        assert grip.tooltip == "Expand Inspector"
        assert not screen.query("#wl-region-right_rail")
        assert screen.query("#watchlists-items-pane")
        assert screen.query("#wl-grip-items")
        assert writes == [RegionLayout(), expected_collapsed]


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
        # Section intents now capture the screen factory at dispatch time.
        builds: list[str] = []
        real_build = screen._build_detail_pane

        def _counting_build(section=None):
            requested = section or screen.active_section
            builds.append(requested)
            return real_build(requested)

        screen._build_detail_pane = _counting_build

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

        real_build = screen._build_detail_pane

        def _build_then_land(section=None):
            built = real_build(section)
            screen._loaded_rules = [row]
            try:  # the loader's push, from inside the gap
                screen.query_one("#watchlists-rules-pane", RulesPane).rules = [row]
            except NoMatches:
                pass
            return built

        screen._build_detail_pane = _build_then_land

        screen.active_section = "rules"
        await _settle(pilot, host)

        assert screen._loaded_rules == [row], "precondition: the rows did land"
        table = screen.query_one("#rules-table", DataTable)
        assert table.row_count == 1, (
            "rows that landed while the swap was mid-flight must reach the "
            "pane the swap mounted, not be stranded in `_loaded_rules`"
        )


async def test_rapid_management_section_requests_keep_captured_intent_on_failure(
    monkeypatch,
) -> None:
    """A delayed A completion cannot build or relabel mutable intent B."""
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        workbench = screen.query_one(WatchlistsWorkbench)
        original_apply = workbench.apply_section_view
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        apply_calls = 0

        async def delayed_apply(**kwargs):
            nonlocal apply_calls
            apply_calls += 1
            if apply_calls == 1:
                first_entered.set()
                await release_first.wait()
            return await original_apply(**kwargs)

        monkeypatch.setattr(workbench, "apply_section_view", delayed_apply)
        original_build = screen._build_detail_pane
        fail_rules = True

        def captured_build(section=None):
            requested = section or screen.active_section
            if requested == "rules" and fail_rules:
                raise RuntimeError("rules centre failed")
            try:
                return original_build(section)
            except TypeError:
                return original_build()

        monkeypatch.setattr(screen, "_build_detail_pane", captured_build)
        workbench._content[Region.ITEMS] = captured_build

        screen.active_section = "sources"
        await asyncio.wait_for(first_entered.wait(), timeout=1)
        screen.active_section = "rules"
        release_first.set()
        await _settle(pilot, host)

        assert screen.active_section == "sources"
        assert screen._rendered_section == "sources"
        assert workbench.read_mode is False
        assert screen.query("#watchlists-sources-pane")
        assert not screen.query("#watchlists-rules-pane")

        fail_rules = False
        screen.active_section = "rules"
        await _settle(pilot, host)

        assert screen.active_section == "rules"
        assert screen._rendered_section == "rules"
        assert workbench.read_mode is False
        assert screen.query("#watchlists-rules-pane")


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
        assert backend_select.disabled is True, (
            "Read is local-only, so its truthful backend selector is locked"
        )

        screen.active_section = "sources"
        await _settle(pilot, host)
        assert screen.query_one("#watchlists-backend-select") is backend_select
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
        from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
            WatchlistsWorkbench,
        )

        workbench = screen.query_one(WatchlistsWorkbench)
        body = screen.query_one("#wl-workbench-body")
        assert workbench.read_mode is False
        assert [child.id for child in workbench.children] == [
            "wl-centre-status",
            "wl-workbench-body",
        ]
        assert [child.id for child in body.children] == [
            "wl-region-left_rail",
            "wl-grip-left_rail",
            "wl-region-items",
            "wl-grip-right_rail",
            "wl-region-right_rail",
        ]

        screen.active_section = "items"
        await _settle(pilot, host)

        assert screen.query("#wl-region-content"), (
            "CONTENT must be mounted back on the Read tab"
        )
        assert workbench.read_mode is True
        assert [child.id for child in body.children] == [
            "wl-region-left_rail",
            "wl-grip-left_rail",
            "wl-region-items",
            "wl-grip-items",
            "wl-region-content",
            "wl-grip-right_rail",
            "wl-region-right_rail",
        ]
        assert backend_select.disabled is True, (
            "Read is local-only, so its backend picker must remain disabled"
        )
        assert screen.query("#watchlists-backend-label"), (
            "Read must keep its local-only explanation beside the selector"
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


@pytest.mark.parametrize("failure", ["false", "exception"])
async def test_browser_worker_failure_does_not_rebuild_reader(
    monkeypatch, failure: str
):
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        reader = screen.query_one("#watchlists-content-pane", ContentPane)
        item = {
            "url": "https://example.com/story",
            "title": "Reader stays put",
            "content_kind": "article",
            "content": "body",
        }
        reader.item = item
        await _settle(pilot, host)
        ui_thread = threading.get_ident()
        notifications: list[tuple[int, str, dict]] = []
        app.notify = lambda message, **kwargs: notifications.append(
            (threading.get_ident(), message, kwargs)
        )

        def fail_to_open(_url: str) -> bool:
            if failure == "exception":
                raise RuntimeError("browser unavailable")
            return False

        monkeypatch.setattr("webbrowser.open", fail_to_open)
        synchronous_errors: list[Exception] = []
        with _RebuildCounter() as counted:
            try:
                screen._open_item_in_browser(item)
            except Exception as exc:
                synchronous_errors.append(exc)
            await _settle(pilot, host)

        assert synchronous_errors == []
        assert notifications
        assert notifications[-1][0] == ui_thread
        assert notifications[-1][2].get("severity") == "error"
        assert screen.query_one("#watchlists-content-pane", ContentPane) is reader
        assert reader.item is item
        assert counted.recomposes["ContentPane#watchlists-content-pane"] == 0, (
            counted.report()
        )


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

        grip = screen.query_one("#wl-grip-left_rail", Button)
        assert getattr(grip, "expanded") is False
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
    """`z` on Feed Items removes only that side body."""
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        tree = screen.query_one("#wl-tree", WatchlistTree)
        reader = screen.query_one("#watchlists-content-pane", ContentPane)
        content_region = screen.query_one("#wl-region-content")
        screen.query_one("#wl-region-items").focus()
        await pilot.pause()
        with _RebuildCounter() as counted:
            screen.action_toggle_region()
            await _settle(pilot, host)

        grip = screen.query_one("#wl-grip-items", Button)
        assert getattr(grip, "expanded") is False
        assert not screen.query("#wl-region-items")
        assert screen.query_one("#wl-tree", WatchlistTree) is tree, counted.report()
        assert screen.query_one("#watchlists-content-pane", ContentPane) is reader, (
            f"the reader must not be rebuilt by its neighbour collapsing: "
            f"{counted.report()}"
        )
        assert screen.query_one("#wl-region-content") is content_region


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

        grip = screen.query_one("#wl-grip-right_rail", Button)
        assert getattr(grip, "expanded") is False
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


async def test_layout_keys_keep_independent_side_pane_state():
    """Scoping the rebuild must not couple the side-pane preferences."""
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

        # Off Read, the centre regions are not the user's to collapse.
        screen.active_section = "sources"
        await _settle(pilot, host)
        before = screen.region_layout
        screen.query_one("#wl-region-items").focus()
        await pilot.pause()
        screen.action_toggle_region()
        await _settle(pilot, host)
        assert screen.region_layout == before, (
            "a centre-region gesture off the Read tab must still be refused"
        )
        assert screen.query("#wl-region-items"), (
            "and the section's own pane must still be showing"
        )


async def test_a_collapsed_region_still_expands_from_its_grip_button() -> None:
    """The grip route through `RegionToggled`, not just the keybinding."""
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        await pilot.press("[")
        await _settle(pilot, host)
        grip = screen.query_one("#wl-grip-left_rail", Button)

        grip.press()
        await _settle(pilot, host)

        assert screen.query("#wl-region-left_rail"), (
            "clicking a collapsed region's grip must expand it"
        )
        assert screen.query("#wl-tree"), "and rebuild its content"


async def test_resize_derives_effective_layout_without_persisting_preference(
    monkeypatch,
) -> None:
    writes: list[RegionLayout] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: writes.append(layout) or True,
    )
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        preferred = screen.region_layout
        writes.clear()

        await pilot.resize_terminal(90, 50)
        await _settle(pilot, host)

        workbench = screen.query_one(WatchlistsWorkbench)
        assert screen.region_layout == preferred
        assert screen._effective_region_layout != preferred
        assert workbench.region_layout == screen._effective_region_layout
        assert writes == []


async def test_resize_hysteresis_suppresses_sub_band_layout_requests(
    monkeypatch,
) -> None:
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        await pilot.resize_terminal(144, 50)
        await _settle(pilot, host)

        workbench = screen.query_one(WatchlistsWorkbench)
        assert screen._responsive_region_layout is not None
        assert screen._responsive_region_layout.is_collapsed(Region.RIGHT_RAIL)
        grip = screen.query_one("#wl-grip-right_rail", WatchlistsPaneGrip)
        focused = screen.query_one("#items-table", ListView)
        focused.focus()
        await pilot.pause()
        assert screen.focused is focused

        requests: list[tuple[int, RegionLayout]] = []
        real_request = workbench.request_region_layout

        def record_request(layout: RegionLayout, *, token: int) -> None:
            requests.append((token, layout))
            real_request(layout, token=token)

        monkeypatch.setattr(workbench, "request_region_layout", record_request)

        for width in (145, 146, 147, 148, 147, 148, 147, 148):
            await pilot.resize_terminal(width, 50)
            await _settle(pilot, host)
            assert screen._responsive_region_layout.is_collapsed(
                Region.RIGHT_RAIL
            )
            assert screen.query_one(
                "#wl-grip-right_rail", WatchlistsPaneGrip
            ) is grip
            assert screen.focused is focused

        assert requests == []

        await pilot.resize_terminal(149, 50)
        await _settle(pilot, host)
        assert len(requests) == 1
        assert not screen._responsive_region_layout.is_collapsed(
            Region.RIGHT_RAIL
        )
        right_rail_body = screen.query_one("#wl-region-right_rail")
        assert screen.query_one(
            "#wl-grip-right_rail", WatchlistsPaneGrip
        ) is grip
        assert screen.focused is focused

        await pilot.resize_terminal(145, 50)
        await _settle(pilot, host)
        assert len(requests) == 1
        assert not screen._responsive_region_layout.is_collapsed(
            Region.RIGHT_RAIL
        )
        assert screen.query_one("#wl-region-right_rail") is right_rail_body
        assert screen.focused is focused

        await pilot.resize_terminal(144, 50)
        await _settle(pilot, host)
        assert len(requests) == 2
        assert screen._responsive_region_layout.is_collapsed(Region.RIGHT_RAIL)
        assert not screen.query("#wl-region-right_rail")
        assert screen.query_one(
            "#wl-grip-right_rail", WatchlistsPaneGrip
        ) is grip
        assert screen.focused is focused


async def test_zero_width_never_seeds_or_replaces_responsive_history(
    monkeypatch,
) -> None:
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        workbench = screen.query_one(WatchlistsWorkbench)
        requests: list[tuple[int, RegionLayout]] = []
        real_request = workbench.request_region_layout

        def record_request(layout: RegionLayout, *, token: int) -> None:
            requests.append((token, layout))
            real_request(layout, token=token)

        monkeypatch.setattr(workbench, "request_region_layout", record_request)
        monkeypatch.setattr(screen, "_available_layout_width", lambda: None)
        screen._responsive_region_layout = None
        before_zero = (
            screen._effective_region_layout,
            screen._current_layout_request_token,
            screen._layout_request_generation,
        )

        screen.on_resize(None)
        await _settle(pilot, host)

        assert screen._responsive_region_layout is None
        assert (
            screen._effective_region_layout,
            screen._current_layout_request_token,
            screen._layout_request_generation,
        ) == before_zero
        assert requests == []

        monkeypatch.setattr(screen, "_available_layout_width", lambda: 144)
        screen.on_resize(None)
        await _settle(pilot, host)
        baseline = screen._responsive_region_layout
        assert baseline is not None
        positive_state = (
            screen._effective_region_layout,
            screen._current_layout_request_token,
            screen._layout_request_generation,
        )
        positive_request_count = len(requests)

        monkeypatch.setattr(screen, "_available_layout_width", lambda: None)
        screen.on_resize(None)
        await _settle(pilot, host)

        assert screen._responsive_region_layout == baseline
        assert (
            screen._effective_region_layout,
            screen._current_layout_request_token,
            screen._layout_request_generation,
        ) == positive_state
        assert len(requests) == positive_request_count


async def test_article_focus_is_transient_and_restores_exact_preference() -> None:
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        screen.action_toggle_left_rail()
        await _settle(pilot, host)
        preferred = screen.region_layout

        await pilot.press("Z")
        await _settle(pilot, host)

        assert screen._article_focus_active is True
        assert screen.region_layout == preferred
        assert screen._effective_region_layout.collapsed == frozenset(
            {Region.LEFT_RAIL, Region.ITEMS, Region.RIGHT_RAIL}
        )

        await pilot.press("Z")
        await _settle(pilot, host)

        assert screen._article_focus_active is False
        assert screen.region_layout == preferred
        assert screen._effective_region_layout == preferred


async def test_article_focus_is_refused_off_read_without_layout_change() -> None:
    app = _build_test_app()
    async with _open(app, section="sources") as (screen, pilot, host):
        before = (screen.region_layout, screen._effective_region_layout)

        await pilot.press("Z")
        await _settle(pilot, host)

        assert screen._article_focus_active is False
        assert (screen.region_layout, screen._effective_region_layout) == before


async def test_responsive_grip_open_protects_preferred_open_pane(
    monkeypatch,
) -> None:
    writes: list[RegionLayout] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: writes.append(layout) or True,
    )
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        await pilot.resize_terminal(144, 50)
        await _settle(pilot, host)
        await pilot.resize_terminal(145, 50)
        await _settle(pilot, host)
        assert screen._effective_region_layout.is_collapsed(Region.RIGHT_RAIL)
        assert not screen.region_layout.is_collapsed(Region.RIGHT_RAIL)
        writes.clear()

        screen.query_one("#wl-grip-right_rail", Button).press()
        await _settle(pilot, host)

        expected_lease = ResponsivePriorityLease(
            Region.RIGHT_RAIL, read_mode=True
        )
        assert screen._responsive_priority_lease == expected_lease
        assert not screen.region_layout.is_collapsed(Region.RIGHT_RAIL)
        assert not screen._effective_region_layout.is_collapsed(Region.RIGHT_RAIL)
        assert writes == []

        screen.on_resize(None)
        await _settle(pilot, host)

        assert screen._responsive_priority_lease == expected_lease
        assert not screen._effective_region_layout.is_collapsed(Region.RIGHT_RAIL)


async def test_article_focus_preserves_priority_lease_and_hidden_baseline(
    monkeypatch,
) -> None:
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        await pilot.resize_terminal(144, 50)
        await _settle(pilot, host)
        await pilot.resize_terminal(145, 50)
        await _settle(pilot, host)
        screen.query_one("#wl-grip-right_rail", Button).press()
        await _settle(pilot, host)

        lease = screen._responsive_priority_lease
        responsive_before_focus = screen._responsive_region_layout
        screen.action_article_focus()
        await _settle(pilot, host)

        assert screen._article_focus_active is True
        assert screen._responsive_priority_lease == lease
        assert screen._responsive_region_layout == responsive_before_focus
        assert screen._effective_region_layout.collapsed == frozenset(
            {Region.LEFT_RAIL, Region.ITEMS, Region.RIGHT_RAIL}
        )

        workbench = screen.query_one(WatchlistsWorkbench)
        requests: list[tuple[int, RegionLayout]] = []
        real_request = workbench.request_region_layout

        def record_request(layout: RegionLayout, *, token: int) -> None:
            requests.append((token, layout))
            real_request(layout, token=token)

        monkeypatch.setattr(workbench, "request_region_layout", record_request)
        await pilot.resize_terminal(90, 50)
        await _settle(pilot, host)

        hidden_baseline = screen._responsive_region_layout
        assert hidden_baseline != responsive_before_focus
        assert screen._responsive_priority_lease == lease
        assert screen._effective_region_layout.collapsed == frozenset(
            {Region.LEFT_RAIL, Region.ITEMS, Region.RIGHT_RAIL}
        )
        assert requests == []

        await pilot.resize_terminal(180, 50)
        await _settle(pilot, host)

        hidden_baseline = screen._responsive_region_layout
        assert hidden_baseline == RegionLayout()
        assert screen._responsive_priority_lease == lease
        assert screen._effective_region_layout.collapsed == frozenset(
            {Region.LEFT_RAIL, Region.ITEMS, Region.RIGHT_RAIL}
        )
        assert requests == []

        screen.action_article_focus()
        await _settle(pilot, host)

        assert screen._article_focus_active is False
        assert screen._responsive_priority_lease == lease
        assert screen._effective_region_layout == hidden_baseline


async def test_priority_lease_parks_across_management_and_replaces_in_read() -> None:
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        await pilot.resize_terminal(144, 50)
        await _settle(pilot, host)
        await pilot.resize_terminal(145, 50)
        await _settle(pilot, host)
        screen.query_one("#wl-grip-right_rail", Button).press()
        await _settle(pilot, host)
        read_lease = screen._responsive_priority_lease

        screen.active_section = "sources"
        await _settle(pilot, host)
        assert screen._responsive_priority_lease == read_lease

        await pilot.resize_terminal(180, 50)
        await _settle(pilot, host)
        assert screen._responsive_priority_lease == read_lease

        screen.active_section = "items"
        await _settle(pilot, host)
        assert screen._responsive_priority_lease == read_lease

        await pilot.resize_terminal(90, 50)
        await _settle(pilot, host)
        screen.query_one("#wl-grip-left_rail", Button).press()
        await _settle(pilot, host)

        left_lease = ResponsivePriorityLease(Region.LEFT_RAIL, read_mode=True)
        assert screen._responsive_priority_lease == left_lease
        assert not screen._effective_region_layout.is_collapsed(Region.LEFT_RAIL)

        screen.query_one("#wl-grip-left_rail", Button).press()
        await _settle(pilot, host)
        assert screen._responsive_priority_lease is None


async def test_priority_lease_clears_only_after_origin_mode_fits_past_dead_band() -> None:
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        await pilot.resize_terminal(144, 50)
        await _settle(pilot, host)
        await pilot.resize_terminal(145, 50)
        await _settle(pilot, host)
        screen.query_one("#wl-grip-right_rail", Button).press()
        await _settle(pilot, host)
        lease = screen._responsive_priority_lease

        for width in (145, 146, 147, 148, 147, 148):
            await pilot.resize_terminal(width, 50)
            await _settle(pilot, host)
            assert screen._responsive_priority_lease == lease

        await pilot.resize_terminal(149, 50)
        await _settle(pilot, host)
        assert screen._responsive_priority_lease is None


async def test_manual_grip_during_article_focus_exits_before_opening() -> None:
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        preferred = screen.region_layout
        screen.action_article_focus()
        await _settle(pilot, host)

        screen.query_one("#wl-grip-left_rail", Button).press()
        await _settle(pilot, host)

        assert screen._article_focus_active is False
        assert screen.region_layout == preferred
        assert not screen._effective_region_layout.is_collapsed(Region.LEFT_RAIL)


async def test_z_targets_only_collapsible_side_panes() -> None:
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        before = screen.region_layout
        screen.query_one("#wl-region-content").focus()
        await pilot.pause()
        screen.action_toggle_region()
        assert screen.region_layout == before

        screen.active_section = "sources"
        await _settle(pilot, host)
        screen.query_one("#wl-region-items").focus()
        await pilot.pause()
        screen.action_toggle_region()
        assert screen.region_layout == before


async def test_z_ignores_stale_region_after_focus_moves_outside_workbench(
    monkeypatch,
) -> None:
    writes: list[RegionLayout] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: writes.append(layout) or True,
    )
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        screen.query_one("#wl-region-items").focus()
        await pilot.pause()
        assert screen.focused_region is Region.ITEMS

        outside = screen.query_one("#watchlists-backend-select")
        outside.disabled = False
        outside.focus()
        await pilot.pause()
        assert screen.focused is outside
        before = (screen.region_layout, screen._effective_region_layout)
        writes.clear()

        await pilot.press("z")
        await _settle(pilot, host)

        assert (screen.region_layout, screen._effective_region_layout) == before
        assert writes == []


async def test_mounted_layout_cycles_preserve_complete_reader_and_list_state() -> None:
    app = _build_test_app()
    watchlist_id = _seed(app, items=40)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        rows = pane.displayed_items()
        assert rows
        pane.select_and_reveal(rows[min(1, len(rows) - 1)])
        await _settle(pilot, host)
        pane.status_filter = "unread"
        pane.search_query = "Story"
        # Let the real debounce and reload settle. Cancelling the production
        # timer here used to hide the anchor/load interleaving this survival
        # test is meant to exercise.
        await pilot.pause(0.4)
        await _settle(pilot, host)
        table = pane.query_one("#items-table", ListView)
        table.scroll_to(y=6, animate=False)
        table.focus()
        await pilot.pause()
        selected_id = str(pane.selected_item["id"])
        anchor_id = getattr(table.highlighted_child, "item_id_key", None)
        reader = screen.query_one("#watchlists-content-pane", ContentPane)
        screen._selected_content_item["content"] = "\n".join(
            f"Reader line {index}" for index in range(120)
        )
        reader.item = dict(screen._selected_content_item)
        await pilot.pause()
        reader.query_one("#content-body").styles.height = 200
        await pilot.pause()
        reader_scroll = reader.query_one("#content-body-scroll", VerticalScroll)
        reader_scroll.scroll_to(y=8, animate=False)
        await pilot.pause()
        scope = screen.tree_scope
        screen._items_page_index = 2
        pane.page_number = 3
        page_index = 2
        list_scroll_y = float(table.scroll_y)
        reader_scroll_y = float(reader_scroll.scroll_y)
        assert list_scroll_y > 0
        assert reader_scroll_y > 0

        async def assert_restored() -> None:
            restored = screen.query_one(
                "#watchlists-items-pane", ArticleListPane
            )
            restored_table = restored.query_one("#items-table", ListView)
            assert str(restored.selected_item["id"]) == selected_id
            assert restored.status_filter == "unread"
            assert restored.search_query == "Story"
            assert restored.page_number == 3
            assert getattr(
                restored_table.highlighted_child, "item_id_key", None
            ) == anchor_id
            assert float(restored_table.scroll_y) == list_scroll_y
            assert restored_table.has_focus
            assert screen.tree_scope == scope
            assert screen._items_page_index == page_index
            assert screen.query_one(
                "#watchlists-content-pane", ContentPane
            ) is reader
            assert float(
                reader.query_one("#content-body-scroll", VerticalScroll).scroll_y
            ) == reader_scroll_y

        for key in ("[", "]"):
            items_identity = screen.query_one(
                "#watchlists-items-pane", ArticleListPane
            )
            await pilot.press(key)
            await _settle(pilot, host)
            assert screen.query_one(
                "#watchlists-items-pane", ArticleListPane
            ) is items_identity
            await pilot.press(key)
            await _settle(pilot, host)
            assert screen.query_one(
                "#watchlists-items-pane", ArticleListPane
            ) is items_identity
            await assert_restored()

        await pilot.press("z")
        await _settle(pilot, host)
        assert screen.focused is screen.query_one("#wl-grip-items")
        assert not screen.query("#watchlists-items-pane")
        await pilot.press("z")
        await _settle(pilot, host)
        await assert_restored()

        for _ in range(2):
            await pilot.resize_terminal(90, 50)
            await _settle(pilot, host)
            assert screen.focused is screen.query_one("#wl-grip-items")
            assert not screen.query("#watchlists-items-pane")
            await pilot.resize_terminal(180, 50)
            await _settle(pilot, host)
            await assert_restored()

        screen.action_article_focus()
        await _settle(pilot, host)
        assert screen.focused is screen.query_one("#wl-grip-items")
        assert screen._items_view_anchor_id == anchor_id
        assert screen._items_view_had_focus is True
        assert screen.query_one("#watchlists-content-pane", ContentPane) is reader

        screen.action_article_focus()
        await _settle(pilot, host)
        await assert_restored()


@pytest.mark.parametrize("failure", [False, OSError("disk full")])
async def test_layout_persistence_advances_only_after_current_success(
    monkeypatch, failure,
) -> None:
    results = iter([failure, True])

    def save(_layout: RegionLayout) -> bool:
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        save,
    )
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        baseline = screen._last_persisted_collapsed
        screen.action_toggle_left_rail()
        await _settle(pilot, host)

        assert screen._last_persisted_collapsed == baseline
        assert screen._pending_persist_layout == screen.region_layout

        screen.action_toggle_right_rail()
        await _settle(pilot, host)

        assert screen._last_persisted_collapsed == screen.region_layout.collapsed
        assert screen._pending_persist_layout is None


@pytest.mark.parametrize("failure", [False, OSError("disk full")])
async def test_manual_focus_exit_retries_pending_preference_without_changing_it(
    monkeypatch, failure,
) -> None:
    writes: list[RegionLayout] = []
    results = iter([failure, True])

    def save(layout: RegionLayout) -> bool:
        writes.append(layout)
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        save,
    )
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        screen.action_toggle_right_rail()
        await _settle(pilot, host)
        pending = screen.region_layout
        assert screen._pending_persist_layout == pending
        assert len(writes) == 1

        screen.action_article_focus()
        await _settle(pilot, host)
        assert len(writes) == 1, "Article Focus itself is never a retry gesture"
        assert not pending.is_collapsed(Region.LEFT_RAIL)

        screen.query_one("#wl-grip-left_rail", Button).press()
        await _settle(pilot, host)

        assert screen.region_layout == pending
        assert screen._article_focus_active is False
        assert writes == [pending, pending]
        assert screen._last_persisted_collapsed == pending.collapsed
        assert screen._pending_persist_layout is None
        assert screen._manual_layout_rollback is None


async def test_layout_persistence_rapid_toggles_drain_latest_request(
    monkeypatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    writes: list[RegionLayout] = []

    def save(layout: RegionLayout) -> bool:
        writes.append(layout)
        if len(writes) == 1:
            started.set()
            assert release.wait(2)
        return True

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        save,
    )
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        screen.action_toggle_left_rail()
        for _ in range(20):
            if started.is_set():
                break
            await pilot.pause(0.01)
        assert started.is_set()

        screen.action_toggle_right_rail()
        newest = screen.region_layout
        release.set()
        await _settle(pilot, host)

        assert writes[-1] == newest
        assert screen._last_persisted_collapsed == newest.collapsed
        assert screen._pending_persist_layout is None
        assert screen._layout_persist_draining is False


async def test_layout_persist_disarm_is_atomic_with_a_newer_generation(
    monkeypatch,
) -> None:
    """A request in the old decision/finally gap must start or be drained."""
    gap_open = threading.Event()
    release_exit = threading.Event()
    writes: list[RegionLayout] = []

    class GapLock:
        """Expose the old worker gap after its return decision releases lock."""

        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._worker_exits = 0

        def __enter__(self):
            self._lock.acquire()
            return self

        def __exit__(self, *_exc_info) -> None:
            self._lock.release()
            if threading.current_thread() is threading.main_thread():
                return
            self._worker_exits += 1
            if self._worker_exits == 2:
                gap_open.set()
                assert release_exit.wait(2)

    def save(layout: RegionLayout) -> bool:
        writes.append(layout)
        return True

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        save,
    )
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        screen._layout_persist_lock = GapLock()
        screen.action_toggle_left_rail()
        first = screen.region_layout

        try:
            assert await asyncio.to_thread(gap_open.wait, 2)
            screen.action_toggle_right_rail()
            newest = screen.region_layout
        finally:
            release_exit.set()

        for _ in range(100):
            if (
                writes[-1:] == [newest]
                and screen._pending_persist_layout is None
                and not screen._layout_persist_draining
            ):
                break
            await pilot.pause(0.01)

        assert writes == [first, newest]
        assert screen._last_persisted_collapsed == newest.collapsed
        assert screen._pending_persist_layout is None
        assert screen._layout_persist_draining is False


async def test_layout_acknowledgements_ignore_stale_tokens_and_clear_current_noop() -> None:
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        layout = screen._effective_region_layout
        token = screen._next_layout_request_token()
        rollback = ManualLayoutRollback(
            token=token,
            attempted_layout=layout,
            attempted_preferred=screen.region_layout,
            preferred_before=screen.region_layout,
            effective_before=screen._effective_region_layout,
            responsive_before=screen._responsive_region_layout,
            article_focus_before=screen._article_focus_active,
            priority_lease_before=screen._responsive_priority_lease,
        )
        screen._manual_layout_rollback = rollback

        screen.post_message(
            RegionLayoutApplyFailed(
                token=token - 1,
                attempted=layout,
                fallback=layout.toggle_preferred(Region.LEFT_RAIL),
            )
        )
        screen.post_message(
            RegionLayoutApplied(
                token=token - 1,
                previous=layout,
                layout=layout,
            )
        )
        await _settle(pilot, host)

        assert screen._effective_region_layout == layout
        assert screen._manual_layout_rollback == rollback

        screen.post_message(
            RegionLayoutApplied(token=token, previous=layout, layout=layout)
        )
        await _settle(pilot, host)

        assert screen._manual_layout_rollback is None


async def test_stale_failure_cannot_rollback_rekeyed_manual_intent(
    monkeypatch,
) -> None:
    """Only the latest correlated token owns a manual rollback."""
    persisted: list[RegionLayout] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: persisted.append(layout) or True,
    )
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        preferred = screen.region_layout
        effective = screen._effective_region_layout
        lease_before = ResponsivePriorityLease(
            Region.RIGHT_RAIL, read_mode=False
        )
        effective_before = effective.toggle_preferred(Region.RIGHT_RAIL)
        responsive_before = effective.toggle_preferred(Region.LEFT_RAIL)
        token1 = screen._next_layout_request_token()
        screen._manual_layout_rollback = ManualLayoutRollback(
            token=token1,
            attempted_layout=effective,
            attempted_preferred=preferred,
            preferred_before=preferred.toggle_preferred(Region.LEFT_RAIL),
            effective_before=effective_before,
            responsive_before=responsive_before,
            article_focus_before=screen._article_focus_active,
            priority_lease_before=lease_before,
        )

        token2 = screen._next_layout_request_token()
        assert screen._manual_layout_rollback is not None
        assert (
            screen._manual_layout_rollback.priority_lease_before
            == lease_before
        )
        assert screen._manual_layout_rollback.effective_before == effective_before
        assert (
            screen._manual_layout_rollback.responsive_before
            == responsive_before
        )
        screen.post_message(
            RegionLayoutApplyFailed(
                token=token1,
                attempted=effective,
                fallback=effective.toggle_preferred(Region.LEFT_RAIL),
            )
        )
        await _settle(pilot, host)

        assert screen._current_layout_request_token == token2
        assert screen.region_layout == preferred
        assert screen._effective_region_layout == effective
        assert persisted == []
        assert screen._manual_layout_rollback is not None

        screen.post_message(
            RegionLayoutApplied(
                token=token2,
                previous=effective,
                layout=effective,
            )
        )
        await _settle(pilot, host)

        assert screen._manual_layout_rollback is None


async def test_suppressed_manual_preference_change_does_not_reuse_stale_token(
    monkeypatch,
) -> None:
    """A no-DOM preference change must not belong to an older request."""
    persisted: list[RegionLayout] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: persisted.append(layout) or True,
    )
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        if not screen.region_layout.is_collapsed(Region.RIGHT_RAIL):
            screen.action_toggle_right_rail()
            await _settle(pilot, host)
        assert screen.region_layout.is_collapsed(Region.RIGHT_RAIL)

        await pilot.resize_terminal(60, 50)
        await _settle(pilot, host)
        workbench = screen.query_one(WatchlistsWorkbench)
        stale_token = screen._current_layout_request_token
        effective_before = screen._effective_region_layout
        assert stale_token > 0
        assert effective_before.is_collapsed(Region.RIGHT_RAIL)
        persisted.clear()

        screen.query_one("#wl-grip-right_rail", Button).press()
        await _settle(pilot, host)

        preferred = screen.region_layout
        assert not preferred.is_collapsed(Region.RIGHT_RAIL)
        assert screen._effective_region_layout == effective_before
        assert screen._current_layout_request_token == stale_token
        assert screen._manual_layout_rollback is None
        assert screen._responsive_priority_lease == ResponsivePriorityLease(
            Region.RIGHT_RAIL, read_mode=True
        )
        assert persisted == [preferred]

        workbench.post_message(
            RegionLayoutApplyFailed(
                token=stale_token,
                attempted=effective_before,
                fallback=effective_before,
            )
        )
        await _settle(pilot, host)

        assert screen.region_layout == preferred
        assert persisted == [preferred]


async def test_failed_responsive_inspector_open_restores_layout_snapshots(
    monkeypatch,
) -> None:
    """A failed dead-band reopen keeps controller and DOM fallbacks aligned."""
    persisted: list[RegionLayout] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: persisted.append(layout) or True,
    )
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        await pilot.resize_terminal(144, 50)
        await _settle(pilot, host)
        await pilot.resize_terminal(145, 50)
        await _settle(pilot, host)

        workbench = screen.query_one(WatchlistsWorkbench)
        preferred_before = screen.region_layout
        effective_before = screen._effective_region_layout
        responsive_before = screen._responsive_region_layout
        focus_before = screen._article_focus_active
        parked_lease = ResponsivePriorityLease(
            Region.LEFT_RAIL, read_mode=False
        )
        screen._responsive_priority_lease = parked_lease

        assert not preferred_before.is_collapsed(Region.RIGHT_RAIL)
        assert effective_before.is_collapsed(Region.RIGHT_RAIL)
        assert responsive_before == effective_before
        assert workbench.region_layout == effective_before

        requests: list[tuple[int, RegionLayout]] = []
        real_request = workbench.request_region_layout

        def record_request(layout: RegionLayout, *, token: int) -> None:
            requests.append((token, layout))
            real_request(layout, token=token)

        monkeypatch.setattr(workbench, "request_region_layout", record_request)
        original_factory = workbench._content[Region.RIGHT_RAIL]

        def fail_inspector_factory():
            raise RuntimeError("inspector failed")

        workbench._content[Region.RIGHT_RAIL] = fail_inspector_factory
        persisted.clear()
        screen.query_one("#wl-grip-right_rail", Button).press()
        await _settle(pilot, host)

        assert len(requests) == 1
        assert not requests[0][1].is_collapsed(Region.RIGHT_RAIL)
        assert workbench.region_layout == effective_before
        assert screen._effective_region_layout == effective_before
        assert screen._responsive_region_layout == responsive_before
        assert screen.region_layout == preferred_before
        assert screen._article_focus_active is focus_before
        assert screen._responsive_priority_lease == parked_lease
        assert screen._manual_layout_rollback is None
        assert persisted == []

        requests.clear()
        screen.on_resize(None)
        await _settle(pilot, host)

        assert requests == []
        assert workbench.region_layout == effective_before
        assert screen._effective_region_layout == effective_before
        assert screen._responsive_region_layout == responsive_before
        assert screen._responsive_priority_lease == parked_lease
        workbench._content[Region.RIGHT_RAIL] = original_factory


async def test_failed_manual_expansion_rolls_back_full_layout_intent(
    monkeypatch,
) -> None:
    """A real screen owns the same fallback that the workbench still renders."""
    from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
        RegionLayoutApplyFailed,
        WatchlistsWorkbench,
    )

    persisted: list[RegionLayout] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: persisted.append(layout) or True,
    )
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        if not screen.region_layout.is_collapsed(Region.LEFT_RAIL):
            screen.action_toggle_left_rail()
            await _settle(pilot, host)
        fallback = screen.region_layout
        persisted.clear()

        parked_lease = ResponsivePriorityLease(
            Region.RIGHT_RAIL, read_mode=False
        )
        screen._responsive_priority_lease = parked_lease
        screen._article_focus_active = True

        workbench = screen.query_one(WatchlistsWorkbench)
        grip = screen.query_one("#wl-grip-left_rail", Button)
        original_factory = workbench._content[Region.LEFT_RAIL]
        fail_factory = True

        def flaky_factory():
            if fail_factory:
                raise RuntimeError("navigation failed")
            return original_factory()

        workbench._content[Region.LEFT_RAIL] = flaky_factory
        grip.press()
        await _settle(pilot, host)

        assert screen.region_layout == fallback
        assert screen._article_focus_active is True
        assert screen._responsive_priority_lease == parked_lease
        assert workbench.region_layout == screen._effective_region_layout
        assert getattr(grip, "expanded") is False
        assert not screen.query("#wl-region-left_rail")
        assert screen._pending_persist_layout is None
        assert screen._last_persisted_collapsed == fallback.collapsed
        assert persisted
        assert persisted[-1] == fallback

        fail_factory = False
        grip.press()
        await _settle(pilot, host)

        assert not screen.region_layout.is_collapsed(Region.LEFT_RAIL)
        assert workbench.region_layout == screen._effective_region_layout
        assert getattr(grip, "expanded") is True
        assert screen.query("#wl-region-left_rail")

        newer = screen.region_layout
        workbench.post_message(
            RegionLayoutApplyFailed(
                token=screen._current_layout_request_token - 1,
                attempted=fallback,
                fallback=fallback.toggle_preferred(Region.RIGHT_RAIL),
            )
        )
        await _settle(pilot, host)
        assert screen.region_layout == newer


async def test_failed_manual_expansion_survives_resize_request_supersession(
    monkeypatch,
) -> None:
    """An automatic token must not orphan an in-flight manual rollback."""
    persisted: list[RegionLayout] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: persisted.append(layout) or True,
    )
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        if not screen.region_layout.is_collapsed(Region.LEFT_RAIL):
            screen.action_toggle_left_rail()
            await _settle(pilot, host)
        fallback = screen.region_layout
        persisted.clear()

        workbench = screen.query_one(WatchlistsWorkbench)
        original_factory = workbench._content[Region.LEFT_RAIL]
        failures = 0

        def fail_while_resize_supersedes():
            nonlocal failures
            failures += 1
            if failures == 1:
                screen._recompute_effective_layout(cause="resize")
            if failures <= 2:
                raise RuntimeError("navigation failed after resize")
            return original_factory()

        workbench._content[Region.LEFT_RAIL] = fail_while_resize_supersedes
        screen.query_one("#wl-grip-left_rail", Button).press()
        await _settle(pilot, host)

        grip = screen.query_one("#wl-grip-left_rail", Button)
        assert screen.region_layout == fallback
        assert screen._pending_persist_layout is None
        assert screen._last_persisted_collapsed == fallback.collapsed
        assert persisted[-1] == fallback
        assert workbench.region_layout == screen._effective_region_layout
        assert not screen.query("#wl-region-left_rail")
        assert getattr(grip, "expanded") is False


@pytest.mark.parametrize("focus_id", ["items-search-input", "items-status-select"])
async def test_reopening_items_restores_exact_focused_descendant(focus_id) -> None:
    app = _build_test_app()
    watchlist_id = _seed(app, items=6)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        focused_child = screen.query_one(f"#{focus_id}")
        focused_child.focus()
        await pilot.pause()
        assert screen.focused is focused_child

        screen._toggle_preferred_region(Region.ITEMS)
        await _settle(pilot, host)
        assert not screen.query("#watchlists-items-pane")

        screen.query_one("#wl-grip-items", Button).press()
        await _settle(pilot, host)

        assert screen.focused is screen.query_one(f"#{focus_id}")


async def test_items_anchor_is_abandoned_once_when_query_context_changes(
    monkeypatch,
) -> None:
    app = _build_test_app()
    watchlist_id = _seed(app, items=6)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        pane.select_and_reveal(pane.displayed_items()[0])
        await _settle(pilot, host)
        table = screen.query_one("#items-table", ListView)
        table.focus()
        await pilot.pause()
        screen._toggle_preferred_region(Region.ITEMS)
        await _settle(pilot, host)
        assert screen._items_view_anchor_id is not None

        screen._items_search_query = "definitely-not-a-story"
        screen._reset_items_paging_for_context(loading=True)
        await screen._replace_items_snapshot(reason="search")

        scheduled: list[float] = []
        original_set_timer = screen.set_timer

        def recording_timer(delay, callback, *args, **kwargs):
            scheduled.append(float(delay))
            return original_set_timer(delay, callback, *args, **kwargs)

        monkeypatch.setattr(screen, "set_timer", recording_timer)
        screen.query_one("#wl-grip-items", Button).press()
        await _settle(pilot, host)

        assert screen._items_view_anchor_id is None
        assert 0.01 not in scheduled


async def test_management_scope_invalidates_reader_return_to_read_failure_is_honest():
    """A failed return never relabels rows parked under the prior scope."""
    app = _build_test_app()
    watchlist_id = _seed(app, items=2)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        await _settle(pilot, host)
        assert screen._loaded_items
        open_item = screen._loaded_items[0]
        screen._selected_content_item = open_item
        content = screen.query_one("#watchlists-content-pane", ContentPane)
        content.item = open_item

        screen.active_section = "sources"
        await _settle(pilot, host)
        committed = TreeScope(kind="unassigned")
        screen._request_tree_scope(committed)

        assert screen.tree_scope == committed
        assert screen._items_snapshot is None
        assert screen._loaded_items == []
        assert screen._selected_content_item is None

        screen._controller.list_reader_items_page = AsyncMock(
            side_effect=RuntimeError("offline")
        )
        screen.active_section = "items"
        await _settle(pilot, host)

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert screen.tree_scope == committed
        assert screen._items_snapshot is None
        assert screen._loaded_items == []
        assert screen._items_snapshot_count == 0
        assert screen._items_pending_arrivals == 0
        assert screen._selected_content_item is None
        assert pane.items == []
        assert pane.new_items_note == ""
        assert pane.page_loading is False
        assert pane.display is False
        assert _static_text(
            screen.query_one("#watchlists-items-retry-state")
        ) == "Couldn't load Unassigned. Retry to load Feed Items."
        retry = screen.query_one("#watchlists-items-retry-button", Button)
        assert str(retry.label) == "Retry"
        assert retry.disabled is False
        assert "No matching items" not in host.export_screenshot()
        assert screen.query_one("#watchlists-content-pane", ContentPane).item is None


async def test_reader_retry_copy_renders_hostile_scope_label_literally():
    app = _build_test_app()
    watchlist_id = _seed(app)
    hostile_label = "[bold]Unbalanced watchlist"
    app.watchlist_bundle_service.rename(watchlist_id, hostile_label)

    async with _open(app, watchlist_id) as (screen, pilot, host):
        await _settle(pilot, host)
        scope = TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        await _mount_failed_reader_return(screen, pilot, host, scope)

        state = screen.query_one("#watchlists-items-retry-state")
        expected = f"Couldn't load {hostile_label}. Retry to load Feed Items."
        assert _static_text(state) == expected
        rendered = state.render()
        assert getattr(rendered, "plain", str(rendered)) == expected


async def test_reader_retry_click_successfully_publishes_committed_scope():
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        await _settle(pilot, host)
        original, _failed = await _mount_failed_reader_return(
            screen, pilot, host, TreeScope(kind="all")
        )
        successful_retry = AsyncMock(side_effect=original)
        screen._controller.list_reader_items_page = successful_retry

        screen.query_one("#watchlists-items-retry-button", Button).press()
        await _settle(pilot, host)

        successful_retry.assert_awaited_once()
        assert screen._items_snapshot is not None
        assert screen._loaded_items
        assert screen.query_one("#watchlists-items-pane", ArticleListPane).display
        with pytest.raises(NoMatches):
            screen.query_one("#watchlists-items-retry-state")


async def test_reader_retry_repeated_failure_retains_scoped_retry_authority():
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        await _settle(pilot, host)
        _original, failed = await _mount_failed_reader_return(
            screen, pilot, host, TreeScope(kind="unassigned")
        )

        screen.query_one("#watchlists-items-retry-button", Button).press()
        await _settle(pilot, host)

        assert failed.await_count == 2
        assert _static_text(
            screen.query_one("#watchlists-items-retry-state")
        ) == "Couldn't load Unassigned. Retry to load Feed Items."
        assert screen.query_one(
            "#watchlists-items-retry-button", Button
        ).disabled is False


async def test_reader_retry_rapid_presses_coalesce_around_one_producer():
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        await _settle(pilot, host)
        original, _failed = await _mount_failed_reader_return(
            screen, pilot, host, TreeScope(kind="all")
        )
        started = asyncio.Event()
        release = asyncio.Event()

        async def controlled_retry(**kwargs):
            started.set()
            await release.wait()
            return await original(**kwargs)

        retry_loader = AsyncMock(side_effect=controlled_retry)
        screen._controller.list_reader_items_page = retry_loader
        retry = screen.query_one("#watchlists-items-retry-button", Button)

        retry.press()
        retry.press()
        await asyncio.wait_for(started.wait(), timeout=2)
        await pilot.pause()

        assert retry_loader.await_count == 1
        assert screen._items_retry_message is not None
        assert screen.query_one(
            "#watchlists-items-retry-button", Button
        ).disabled is True

        release.set()
        await _settle(pilot, host)
        assert retry_loader.await_count == 1
        assert screen._items_snapshot is not None
        with pytest.raises(NoMatches):
            screen.query_one("#watchlists-items-retry-state")


async def test_reader_retry_scheduling_failure_keeps_retry_available(monkeypatch):
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        await _settle(pilot, host)
        await _mount_failed_reader_return(
            screen, pilot, host, TreeScope(kind="unassigned")
        )

        monkeypatch.setattr(
            screen,
            "run_worker",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("worker scheduling failed")
            ),
        )
        screen.query_one("#watchlists-items-retry-button", Button).press()
        await pilot.pause()

        assert _static_text(
            screen.query_one("#watchlists-items-retry-state")
        ) == "Couldn't load Unassigned. Retry to load Feed Items."
        assert screen.query_one(
            "#watchlists-items-retry-button", Button
        ).disabled is False


async def test_layout_persist_scheduler_and_thread_handoff_failures_are_retryable(
    monkeypatch,
) -> None:
    writes: list[RegionLayout] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: writes.append(layout) or True,
    )
    app = _build_test_app()
    async with _open(app) as (screen, pilot, host):
        requested = screen.region_layout.toggle_preferred(Region.LEFT_RAIL)
        original_run_worker = screen.run_worker

        def fail_schedule(*args, **kwargs):
            raise RuntimeError("worker scheduling failed")

        monkeypatch.setattr(screen, "run_worker", fail_schedule)
        screen._schedule_layout_persist(requested)
        assert screen._layout_persist_draining is False
        assert screen._pending_persist_layout == requested

        monkeypatch.setattr(screen, "run_worker", original_run_worker)
        original_call_from_thread = screen.app.call_from_thread
        monkeypatch.setattr(
            screen.app,
            "call_from_thread",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("handoff failed")
            ),
        )
        screen._layout_persist_draining = True
        screen._persist_layout_worker()
        assert screen._layout_persist_draining is False
        assert screen._pending_persist_layout == requested

        monkeypatch.setattr(screen.app, "call_from_thread", original_call_from_thread)
        screen._schedule_layout_persist(requested)
        await _settle(pilot, host)

        assert writes[-1] == requested
        assert screen._pending_persist_layout is None
        assert screen._layout_persist_draining is False


async def test_management_expansion_failure_preserves_parked_feed_preference(
    monkeypatch,
) -> None:
    """Rendered rollback merges into, rather than replaces, preferred state."""
    from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
        WatchlistsWorkbench,
    )

    persisted: list[RegionLayout] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: persisted.append(layout) or True,
    )
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id, section="sources") as (
        screen,
        pilot,
        host,
    ):
        fallback = RegionLayout(
            collapsed=frozenset({Region.LEFT_RAIL, Region.ITEMS})
        )
        screen._apply_layout(fallback)
        await _settle(pilot, host)
        persisted.clear()

        workbench = screen.query_one(WatchlistsWorkbench)
        grip = screen.query_one("#wl-grip-left_rail", Button)
        original_factory = workbench._content[Region.LEFT_RAIL]
        fail_factory = True

        def flaky_factory():
            if fail_factory:
                raise RuntimeError("management navigation failed")
            return original_factory()

        workbench._content[Region.LEFT_RAIL] = flaky_factory
        grip.press()
        await _settle(pilot, host)

        assert screen.region_layout == fallback
        assert screen.region_layout.is_collapsed(Region.ITEMS)
        assert workbench.region_layout == screen._effective_region_layout
        assert getattr(grip, "expanded") is False
        assert not screen.query("#wl-region-left_rail")
        assert screen._pending_persist_layout is None
        assert screen._last_persisted_collapsed == fallback.collapsed
        assert persisted
        assert persisted[-1] == fallback

        fail_factory = False
        grip.press()
        await _settle(pilot, host)

        assert screen.region_layout.is_collapsed(Region.ITEMS)
        assert not screen.region_layout.is_collapsed(Region.LEFT_RAIL)
        assert workbench.region_layout == screen._effective_region_layout
        assert getattr(grip, "expanded") is True
        assert screen.query("#wl-region-left_rail")


async def test_section_factory_failure_rolls_back_mode_and_can_retry(
    monkeypatch,
) -> None:
    persisted: list[RegionLayout] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: persisted.append(layout) or True,
    )
    app = _build_test_app()
    watchlist_id = _seed(app)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        await pilot.resize_terminal(144, 50)
        await _settle(pilot, host)
        await pilot.resize_terminal(145, 50)
        await _settle(pilot, host)
        screen.query_one("#wl-grip-right_rail", Button).press()
        await _settle(pilot, host)
        parked_lease = screen._responsive_priority_lease
        assert parked_lease == ResponsivePriorityLease(
            Region.RIGHT_RAIL, read_mode=True
        )
        await pilot.resize_terminal(90, 50)
        await _settle(pilot, host)

        workbench = screen.query_one(WatchlistsWorkbench)
        reader = screen.query_one("#watchlists-content-pane", ContentPane)
        items_grip = screen.query_one("#wl-grip-items", WatchlistsPaneGrip)
        before_layout = screen._effective_region_layout
        assert before_layout.collapsed == frozenset(
            {Region.LEFT_RAIL, Region.ITEMS}
        )
        original_factory = screen._build_detail_pane
        fail = True

        def flaky_factory(section=None):
            requested = section or screen.active_section
            if fail and requested == "sources":
                raise RuntimeError("sources centre failed")
            return original_factory(requested)

        screen._build_detail_pane = flaky_factory
        persisted.clear()

        screen.active_section = "sources"
        await _settle(pilot, host)

        assert screen.active_section == "items"
        assert screen._effective_region_layout == before_layout
        assert workbench.read_mode is True
        assert workbench.region_layout == before_layout
        assert screen.query_one("#watchlists-content-pane", ContentPane) is reader
        assert screen.query_one(
            "#wl-grip-items", WatchlistsPaneGrip
        ) is items_grip
        assert screen._responsive_priority_lease == parked_lease
        assert screen._responsive_region_layout == before_layout
        assert persisted == []

        screen.action_article_focus()
        await _settle(pilot, host)
        screen.action_article_focus()
        await _settle(pilot, host)

        assert screen._effective_region_layout == before_layout
        assert screen._responsive_region_layout == before_layout
        assert screen._responsive_priority_lease == parked_lease

        fail = False
        screen.active_section = "sources"
        await _settle(pilot, host)

        assert screen.active_section == "sources"
        assert workbench.read_mode is False
        assert not screen.query("#watchlists-content-pane")
        assert screen._responsive_priority_lease == parked_lease
        assert persisted == []
