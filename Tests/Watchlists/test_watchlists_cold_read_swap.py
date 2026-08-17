"""Batched section swap + recompose-free pane seeding (task-15778).

Two related residuals from task-15461's Implementation Notes:

1. **The section swap's DOM work runs under one `App.batch_update`.** The
   cold Read switch is the one section move that mounts the CONTENT region
   back, and the swap performs it as discrete awaited remove/mount cycles.
   Measured at HEAD the screen never actually painted mid-swap even before
   this task -- the whole sequence runs inside `_drain_surface_refresh`'s
   single `call_next` callback, so the pump never idles and the paused
   update timer never fires (task-15461's own `run_worker` -> `call_next`
   move bought that silently). The batch makes the one-pass property an
   explicit contract instead of a scheduling accident: it survives a future
   factory that awaits, or the drain moving off a single callback.

2. **`_build_detail_pane`/`_build_content_pane` seed `recompose=True`
   reactives with `set_reactive`.** A plain assignment on the freshly
   constructed, unmounted pane queues `refresh(recompose=True)` the moment
   the seeded value differs from the class default (`[] != [row]`), which
   tears the pane straight back down just after it mounts -- one full extra
   pane rebuild per data-carrying region build (measured: sources 1->0,
   rules 1->0, overview 1->0, artifacts 2->1 recomposes per switch; the
   remaining artifacts 1 is the briefings loader landing post-mount, a real
   data arrival). The conversion is per-reactive, not blanket:
   non-recompose reactives keep their plain assignments so load-bearing
   watcher side effects survive -- `RunsPane.selected_run` must keep
   clearing the stale detail (hence detail-after-selection order) and must
   keep starting the status poll for a still-running run.
"""

from __future__ import annotations

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.Watchlists.test_watchlists_scoped_rebuilds import (
    _RebuildCounter,
    _open,
    _seed,
    _seed_alert_rule,
    _settle,
)
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Watchlists_Modules.artifacts_pane import ArtifactsPane
from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
from tldw_chatbook.UI.Watchlists_Modules.notifications_pane import NotificationsPane
from tldw_chatbook.UI.Watchlists_Modules.overview_pane import OverviewPane
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region, RegionLayout
from tldw_chatbook.UI.Watchlists_Modules.rules_pane import RulesPane
from tldw_chatbook.UI.Watchlists_Modules.runs_pane import RunsPane
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane
from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
    WatchlistsWorkbench,
)

pytestmark = pytest.mark.asyncio

_LOAD_REGION_LAYOUT_TARGET = (
    "tldw_chatbook.UI.Screens.watchlists_collections_screen.load_region_layout"
)


# ---------------------------------------------------------------------------
# AC#1 -- the cold Read swap's DOM work runs inside one batch_update
# ---------------------------------------------------------------------------


class _BatchObserver:
    """Records `app._batch_count > 0` at entry to each swap DOM-work method.

    Call-through wrappers on the three methods `apply_section_view` drives:
    the region sync (which mounts CONTENT back on a cold Read switch), the
    section pane rebuild and the header rebuild. Only entries made while
    `apply_section_view` itself is on the stack are recorded -- the drainer
    can legitimately run e.g. a rail `refresh_region_content` outside the
    swap (and outside the batch) in the same settle window, and that is not
    what the AC is about. Asserting on the batch counter at entry is
    deterministic -- unlike counting layout passes, it cannot be masked by
    the pump happening never to idle mid-swap (which is exactly what HEAD
    measured; see the module docstring).
    """

    _METHODS = ("_sync_regions", "refresh_region_content", "refresh_header_content")

    def __init__(self) -> None:
        self.entries: list[tuple[str, bool]] = []
        self._in_swap = False
        self._originals = {
            name: getattr(WatchlistsWorkbench, name) for name in self._METHODS
        }
        self._original_apply = WatchlistsWorkbench.apply_section_view

    def __enter__(self) -> "_BatchObserver":
        observer = self
        original_apply = self._original_apply

        async def _bracketed_apply(widget_self, **kwargs):
            observer._in_swap = True
            try:
                return await original_apply(widget_self, **kwargs)
            finally:
                observer._in_swap = False

        def _wrap(name, original):
            async def _observing(widget_self, *args, **kwargs):
                if observer._in_swap:
                    observer.entries.append((name, widget_self.app._batch_count > 0))
                return await original(widget_self, *args, **kwargs)

            return _observing

        WatchlistsWorkbench.apply_section_view = _bracketed_apply
        for name, original in self._originals.items():
            setattr(WatchlistsWorkbench, name, _wrap(name, original))
        return self

    def __exit__(self, *exc_info) -> None:
        WatchlistsWorkbench.apply_section_view = self._original_apply
        for name, original in self._originals.items():
            setattr(WatchlistsWorkbench, name, original)


@pytest.mark.parametrize(
    "layout",
    [
        # The shipped first-run default and a deliberately customised
        # layout: the batch must hold in both config states, not just the
        # default the fixture happens to produce.
        RegionLayout(collapsed=frozenset({Region.RIGHT_RAIL})),
        RegionLayout(collapsed=frozenset({Region.LEFT_RAIL})),
    ],
    ids=["first-run-default", "user-customised"],
)
async def test_the_cold_read_swap_runs_inside_one_batch_update(monkeypatch, layout):
    """Every piece of the swap's DOM work must see an open batch.

    Born red against the pre-task code: without the `batch_update` in
    `apply_section_view`, every entry records `_batch_count == 0`.
    """
    monkeypatch.setattr(_LOAD_REGION_LAYOUT_TARGET, lambda: layout)
    app = _build_test_app()
    watchlist_id = _seed(app, items=3)
    async with _open(app, watchlist_id, section="sources") as (screen, pilot, host):
        await _settle(pilot, host)
        assert not screen.query("#wl-region-content"), (
            "precondition: CONTENT must be unmounted off Read, or the switch "
            "below is not the cold one"
        )
        with _BatchObserver() as observed:
            screen.active_section = "items"
            await _settle(pilot, host)

        assert screen.query("#wl-region-content"), "CONTENT must be back on Read"
        surfaces = {name for name, _ in observed.entries}
        assert {"_sync_regions", "refresh_region_content"} <= surfaces, (
            f"the swap must actually have exercised its DOM work: {observed.entries}"
        )
        unbatched = [name for name, batched in observed.entries if not batched]
        assert unbatched == [], (
            "every DOM-work entry of the section swap must run inside "
            f"App.batch_update; these ran outside it: {unbatched}"
        )


# ---------------------------------------------------------------------------
# AC#2 + AC#3 + content-pane AC -- pre-mount seeding queues no recompose,
# per pane branch
# ---------------------------------------------------------------------------


def _unmounted_screen(app) -> WatchlistsCollectionsScreen:
    return WatchlistsCollectionsScreen(app)


def _pane_of(built, pane_type):
    for child in built._pending_children:
        if isinstance(child, pane_type):
            return child
    raise AssertionError(f"no {pane_type.__name__} among {built._pending_children!r}")


_RULE_ROW = {
    "id": "r1",
    "name": "Rule One",
    "condition_type": "no_items",
    "severity": "warning",
    "enabled": True,
}


def _primed_screen(app, section: str) -> WatchlistsCollectionsScreen:
    """An unmounted screen whose state gives `section`'s pane NON-default
    seeds, so a plain-assignment seeding would queue the extra recompose."""
    screen = _unmounted_screen(app)
    screen.active_section = section
    if section == "overview":
        screen.set_reactive(
            WatchlistsCollectionsScreen.overview_data,
            {"sources": 1, "items": 2, "runs": 0},
        )
        screen._tree_watchlists = [{"id": 1, "name": "One"}]
    elif section == "sources":
        screen._loaded_sources = [
            {"id": "s1", "name": "AI News", "type": "rss", "source": "https://x/f"}
        ]
    elif section == "runs":
        run = {"id": "run-1", "status": "complete", "source_name": "AI News"}
        screen._loaded_runs = [run]
        # App order: the selection watcher clears the detail mirrors, the
        # loader then fills them.
        screen.selected_run = run
        screen._run_detail_items = [{"title": "Story", "status": "kept"}]
        screen._run_detail_logs = "log line"
        screen._run_detail_items_note = "1 item"
    elif section == "rules":
        screen._loaded_rules = [_RULE_ROW]
    elif section == "rules-editing":
        screen.active_section = "rules"
        screen._loaded_rules = [_RULE_ROW]
        screen._rule_form_open = True
        screen._rule_form_editing = _RULE_ROW
    elif section == "notifications":
        row = {"id": 7, "title": "Alert", "severity": "warning"}
        screen._loaded_notifications = [row]
        screen.set_reactive(WatchlistsCollectionsScreen.selected_notification, row)
    elif section == "artifacts":
        briefing = {"id": 3, "status": "complete", "title": "Briefing"}
        screen._loaded_briefings = [briefing]
        screen._selected_briefing = briefing
        screen._loaded_scripts = [{"id": 5, "briefing_id": 3}]
    elif section == "items":
        # Primed through the SYNC-watcher reactives rather than `items`
        # rows: `watch_items` is async, and on a pane this probe never
        # mounts the queued coroutine would simply be garbage-collected
        # un-awaited (a test artifact, not a product path -- production
        # always mounts the built pane). The mounted warm-revisit test
        # below bounces through Read with real rows.
        screen._items_search_query = "story"
        screen._selected_content_item = {"id": "i1", "title": "Story 1"}
    return screen


_PANE_BY_SECTION = {
    "overview": OverviewPane,
    "sources": SourcesPane,
    "runs": RunsPane,
    "rules": RulesPane,
    "rules-editing": RulesPane,
    "notifications": NotificationsPane,
    "artifacts": ArtifactsPane,
    "items": None,  # ArticleListPane: no recompose reactives -- guard only
}


@pytest.mark.parametrize("section", list(_PANE_BY_SECTION))
async def test_detail_pane_seeding_queues_no_recompose(section):
    """A freshly built, data-seeded pane must not be scheduled for an
    immediate rebuild.

    `refresh(recompose=True)` marks `_recompose_required` on the unmounted
    pane and queues `_check_recompose`, which fires just after the mount and
    tears the pane straight back down. Born red for every data-carrying
    branch against the pre-task plain-assignment seeding (`items` is the
    audited exception -- `ArticleListPane` has no `recompose=True` reactives
    at all, so that branch is a regression guard, not a born-red pin).
    """
    app = _build_test_app()
    screen = _primed_screen(app, section)

    built = screen._build_detail_pane()

    flagged = [
        f"{type(child).__name__}#{child.id or '-'}"
        for child in built._pending_children
        if child._recompose_required
    ]
    assert flagged == [], (
        "pre-mount seeding must not queue a recompose on the freshly built "
        f"pane; flagged: {flagged}"
    )
    pane_type = _PANE_BY_SECTION[section]
    if pane_type is not None:
        _pane_of(built, pane_type)  # the seeded pane really is in there


async def test_rules_editing_seed_still_prefills_the_form_state():
    """`edit_rule`'s pre-mount route must seed the same reactives the
    mounted route assigns -- `compose()` reads them to pre-fill the form."""
    app = _build_test_app()
    screen = _primed_screen(app, "rules-editing")

    pane = _pane_of(screen._build_detail_pane(), RulesPane)

    assert pane.show_rule_form is True
    assert pane.selected_rule == _RULE_ROW
    assert pane._editing_rule_id == "r1"
    assert pane._recompose_required is False


async def test_content_pane_item_seeding_queues_no_recompose():
    """`_build_content_pane` (the CONTENT region the cold Read swap mounts
    back): `item` is `recompose=True`, so a plain assignment re-rendered the
    whole article a second time immediately after the swap whenever an item
    was selected. Born red against the pre-task code."""
    app = _build_test_app()
    screen = _unmounted_screen(app)
    screen._selected_content_item = {
        "id": "i1",
        "title": "Story 1",
        "content": "body",
    }

    pane = screen._build_content_pane()

    assert isinstance(pane, ContentPane)
    assert pane.item == screen._selected_content_item, "the reader still seeds"
    assert pane._recompose_required is False, (
        "seeding `item` must not schedule an immediate second render of the article"
    )


# ---------------------------------------------------------------------------
# AC#2 -- RunsPane's load-bearing seeding order and watcher side effects
# ---------------------------------------------------------------------------


async def test_runs_pane_seeding_keeps_the_detail_set_after_the_selection():
    """`selected_run`'s watcher clears the detail (a run's items must never
    outlive the run they belong to), so `_build_detail_pane` must keep
    setting the detail AFTER the selection. Reordering the seeding -- or
    silencing the watcher's clear without re-seeding -- reds this."""
    app = _build_test_app()
    screen = _primed_screen(app, "runs")

    pane = _pane_of(screen._build_detail_pane(), RunsPane)

    assert pane.selected_run == screen.selected_run
    assert pane.run_items == [{"title": "Story", "status": "kept"}], (
        "the detail must survive the selection watcher's clear -- it is "
        "seeded after `selected_run` on purpose"
    )
    assert pane.run_logs == "log line"
    assert pane.run_items_note == "1 item"


async def test_runs_pane_seeding_still_starts_the_poll_for_a_running_run():
    """`watch_selected_run` starts the status poll when the seeded run is
    still running -- the reason `selected_run` stays a PLAIN assignment in
    `_build_detail_pane` (task-15778's per-reactive audit). A blind
    `set_reactive` conversion would silently freeze a mid-flight run's
    status on every region rebuild; this pins the watcher route."""
    polled: list[dict] = []

    original = RunsPane._start_run_poll
    RunsPane._start_run_poll = lambda self, run: polled.append(run)
    try:
        app = _build_test_app()
        screen = _unmounted_screen(app)
        screen.active_section = "runs"
        running = {"id": "run-9", "status": "running", "source_name": "AI News"}
        screen._loaded_runs = [running]
        screen.selected_run = running

        pane = _pane_of(screen._build_detail_pane(), RunsPane)
    finally:
        RunsPane._start_run_poll = original

    assert pane.selected_run == running
    assert polled == [running], (
        "the pane's `watch_selected_run` must fire during seeding and start "
        "the poll for a still-running run -- a region rebuild mid-run "
        f"depends on it; got {polled!r}"
    )


# ---------------------------------------------------------------------------
# End-to-end: a mounted, warm revisit costs zero pane recomposes
# ---------------------------------------------------------------------------


async def test_warm_revisits_cost_zero_pane_recomposes():
    """Mounted proof of the seeding fix: revisiting a section whose rows are
    already on screen state builds the pane once, carrying the rows, and
    recomposes it zero times. Pre-task this was exactly 1 per data-carrying
    revisit -- the pre-mount seeding recompose."""
    app = _build_test_app()
    watchlist_id = _seed(app, items=2, briefings=1)
    await _seed_alert_rule(app)
    async with _open(app, watchlist_id, section="items") as (screen, pilot, host):
        await _settle(pilot, host)
        # First visits load each section's rows onto screen state.
        for section in ("rules", "artifacts", "sources"):
            screen.active_section = section
            await _settle(pilot, host)

        for section, pane_key in (
            ("rules", "RulesPane#watchlists-rules-pane"),
            ("artifacts", "ArtifactsPane#watchlists-artifacts-pane"),
            ("sources", "SourcesPane#watchlists-sources-pane"),
        ):
            screen.active_section = "items"
            await _settle(pilot, host)
            with _RebuildCounter() as counted:
                screen.active_section = section
                await _settle(pilot, host)
            assert counted.recomposes[pane_key] == 0, (
                f"warm {section} revisit must not rebuild the freshly built "
                f"pane: {counted.report()}"
            )
