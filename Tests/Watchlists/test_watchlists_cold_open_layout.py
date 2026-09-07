"""TASK-15775: `region_layout`'s reactive default must agree with the layout
a cold open actually renders.

Before this task, `WatchlistsCollectionsScreen.region_layout`'s class-level
reactive default was `RegionLayout()` (nothing collapsed), while a genuinely
fresh config's effective layout -- `region_layout_store._FIRST_RUN_DEFAULT`
-- collapses RIGHT_RAIL (the Inspector). `compose_content` reads
`self.region_layout` to build the initial `WatchlistsWorkbench`, and compose
always runs before `on_mount`, so every cold open composed the full,
expanded 13-widget Inspector pane and then, the instant `on_mount` fired
`_apply_layout(load_region_layout())`, tore it straight back down for the
one-line collapsed header the fresh config actually wants -- a
`_swap_region_widget` call and a discarded/rebuilt widget subtree on a
completely ordinary first paint (task-15462's profiling: ~5-10ms, 1-2% of a
~450ms push).

Fixed by seeding `region_layout` (and the `_last_persisted_collapsed`
persistence marker) from a single `load_region_layout()` call made at
construction time, before `compose_content` ever runs -- see
`WatchlistsCollectionsScreen.__init__`'s own comment for the ordering
guarantee this closes (task-15462's flagged migration-write risk).

Note on `Tests/conftest.py`'s `isolate_test_environment` fixture: it
blanket-patches `watchlists_collections_screen.load_region_layout` to
`lambda: RegionLayout()` (nothing collapsed) for every test that has the
screen module imported, so pre-task-2513 screen tests do not have to care
about collapse state. Every test below that cares what `load_region_layout`
actually returns overrides that patch explicitly -- either back to a fixed
value or back to the real function -- exactly like the established
`test_persisted_layout_is_applied_on_mount` (`Tests/UI/
test_watchlists_destination_shell.py`) already does; a test that skipped
this would silently measure the conftest stub, not this task's fix.
"""

from __future__ import annotations

import pytest
import tomllib

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.config import get_cli_setting, save_setting_to_cli_config
from tldw_chatbook import config as config_module
from tldw_chatbook.config import save_settings_to_cli_config
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    ResponsivePriorityLease,
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Watchlists_Modules import region_layout_store
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region, RegionLayout
from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
    WatchlistsWorkbench,
)

pytestmark = pytest.mark.asyncio

_LOAD_REGION_LAYOUT_TARGET = (
    "tldw_chatbook.UI.Screens.watchlists_collections_screen.load_region_layout"
)


class _RegionBuildCounter:
    """Count pane factory construction during the cold-open lifecycle."""

    def __init__(self) -> None:
        self.regions: list[Region] = []
        self._original = WatchlistsWorkbench._region_body

    def __enter__(self) -> "_RegionBuildCounter":
        counter = self
        original = self._original

        def _counting_build(widget_self, region):
            counter.regions.append(region)
            return original(widget_self, region)

        WatchlistsWorkbench._region_body = _counting_build
        return self

    def __exit__(self, *exc_info) -> None:
        WatchlistsWorkbench._region_body = self._original


def _right_rail_is_collapsed(screen) -> bool:
    grip = screen.query_one("#wl-grip-right_rail")
    body = screen.query("#wl-region-right_rail")
    assert getattr(grip, "expanded") is bool(body)
    return not bool(body)


# ---------------------------------------------------------------------------
# AC#1 + AC#2 -- construction seeds the reactive from a single load, and
# `_last_persisted_collapsed` agrees with it immediately (task-15462's
# flagged ordering risk).
# ---------------------------------------------------------------------------


async def test_construction_seeds_region_layout_from_a_single_persisted_load(
    monkeypatch,
):
    """A fresh config's `region_layout` must already equal the persisted/
    first-run layout the instant `__init__` returns -- before
    `compose_content` (which reads `self.region_layout` to build the
    workbench) ever runs. `load_region_layout()` must run exactly once for
    the whole construction, not zero (stale class default) or more than
    once (re-running its one-time migration branch).

    Explicitly restores the real `load_region_layout` (see module
    docstring): this test's whole point is exercising the REAL config read,
    not the conftest's legacy-compat stub.
    """
    calls: list[None] = []
    original_load = region_layout_store.load_region_layout

    def _counting_load():
        calls.append(None)
        return original_load()

    monkeypatch.setattr(_LOAD_REGION_LAYOUT_TARGET, _counting_load)

    app = _build_test_app()
    screen = WatchlistsCollectionsScreen(app)

    assert len(calls) == 1, (
        "load_region_layout must run exactly once at construction, not the "
        f"class-default-then-on_mount pattern this replaces (got {len(calls)} calls)"
    )
    assert screen.region_layout == RegionLayout(
        collapsed=frozenset({Region.RIGHT_RAIL})
    ), (
        "the reactive must already hold the FIRST-RUN layout before compose "
        f"runs, not the stale class default; got {screen.region_layout!r}"
    )
    assert screen._last_persisted_collapsed == frozenset({Region.RIGHT_RAIL}), (
        "the persistence marker must be primed from the SAME call, before "
        "anything else (a keypress, a later _schedule_layout_persist call) "
        "can race it"
    )
    assert get_cli_setting("watchlists", "layout_version", None) == (
        region_layout_store.LAYOUT_VERSION
    )
    assert get_cli_setting("watchlists", "content_reader_migrated", None) is None


async def test_construction_seeds_an_explicitly_saved_non_default_layout(monkeypatch):
    """The same construction-time seed must reflect a REAL persisted choice,
    not just the first-run default -- proving the fix derives from
    `load_region_layout()` rather than hard-coding RIGHT_RAIL collapsed.

    Explicitly restores the real `load_region_layout` (see module
    docstring) before seeding config, so this exercises the true read.
    """
    monkeypatch.setattr(
        _LOAD_REGION_LAYOUT_TARGET, region_layout_store.load_region_layout
    )
    save_setting_to_cli_config("watchlists", "collapsed_regions", [])
    save_setting_to_cli_config("watchlists", "content_reader_migrated", True)
    app = _build_test_app()

    screen = WatchlistsCollectionsScreen(app)

    assert screen.region_layout == RegionLayout(), (
        "a user who explicitly re-expanded everything (saved `[]`) must see "
        "that choice reflected immediately, not the first-run default"
    )
    assert screen._last_persisted_collapsed == frozenset()


# ---------------------------------------------------------------------------
# AC#3 -- `_apply_layout` performs zero `_swap_region_widget` calls on a
# normal screen visit, verified against BOTH config states.
# ---------------------------------------------------------------------------


async def test_cold_open_with_the_first_run_default_shows_the_collapsed_rail_with_no_swap(
    monkeypatch,
):
    """The shipped first-run default (RIGHT_RAIL collapsed): it must render
    COLLAPSED on the very first paint, with zero `_swap_region_widget`
    calls -- not expanded-then-collapsed a moment later.
    """
    monkeypatch.setattr(
        _LOAD_REGION_LAYOUT_TARGET,
        lambda: RegionLayout(collapsed=frozenset({Region.RIGHT_RAIL})),
    )
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    with _RegionBuildCounter() as builds:
        async with host.run_test(size=(180, 50)) as pilot:
            await pilot.pause(0.2)
            screen = host.screen_stack[-1]
            assert isinstance(screen, WatchlistsCollectionsScreen)
            await pilot.pause(0.2)
            await host.workers.wait_for_complete()
            await pilot.pause()

            assert _right_rail_is_collapsed(screen), (
                "RIGHT_RAIL must be collapsed on the FIRST paint, per "
                "_FIRST_RUN_DEFAULT"
            )

    assert Region.RIGHT_RAIL not in builds.regions


async def test_cold_open_honors_an_explicit_non_default_layout_with_no_swap(
    monkeypatch,
):
    """A config where the user explicitly chose a layout that disagrees
    with BOTH the pre-fix class default (nothing collapsed) and the
    shipped first-run default (RIGHT_RAIL collapsed) -- collapsing
    LEFT_RAIL instead. Their choice must still apply on the very first
    paint, with zero swaps: the fix must not pin either default harder.
    """
    monkeypatch.setattr(
        _LOAD_REGION_LAYOUT_TARGET,
        lambda: RegionLayout(collapsed=frozenset({Region.LEFT_RAIL})),
    )
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    with _RegionBuildCounter() as builds:
        async with host.run_test(size=(180, 50)) as pilot:
            await pilot.pause(0.2)
            screen = host.screen_stack[-1]
            assert isinstance(screen, WatchlistsCollectionsScreen)
            await pilot.pause(0.2)
            await host.workers.wait_for_complete()
            await pilot.pause()

            assert not _right_rail_is_collapsed(screen), (
                "RIGHT_RAIL was not part of the user's saved collapse set "
                "and must render expanded"
            )
            assert not screen.query_one("#wl-grip-left_rail").expanded, (
                "LEFT_RAIL is the region the user actually collapsed and "
                "must render collapsed on the first paint"
            )
            assert not screen.query("#wl-region-left_rail")

    assert Region.LEFT_RAIL not in builds.regions


async def test_cold_open_with_a_fresh_real_config_shows_the_collapsed_rail_with_no_swap(
    monkeypatch,
):
    """End-to-end version of the two tests above: the REAL
    `load_region_layout`, reading the REAL (empty, per-test-sandboxed)
    config file, through a REAL mounted pilot -- proving the production
    wiring (not just the isolated logic) never composes the expanded
    Inspector on a genuinely fresh install.
    """
    monkeypatch.setattr(
        _LOAD_REGION_LAYOUT_TARGET, region_layout_store.load_region_layout
    )
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    with _RegionBuildCounter() as builds:
        async with host.run_test(size=(180, 50)) as pilot:
            await pilot.pause(0.2)
            screen = host.screen_stack[-1]
            assert isinstance(screen, WatchlistsCollectionsScreen)
            await pilot.pause(0.2)
            await host.workers.wait_for_complete()
            await pilot.pause()

            assert _right_rail_is_collapsed(screen), (
                "a genuinely fresh, real config must cold-open with RIGHT_RAIL "
                "already collapsed"
            )

    assert Region.RIGHT_RAIL not in builds.regions


# ---------------------------------------------------------------------------
# AC#2 (continued) -- on_mount's reuse of the construction-time load must
# not schedule a redundant persist write.
# ---------------------------------------------------------------------------


async def test_mount_schedules_no_persist_worker_when_nothing_changed(monkeypatch):
    """`on_mount`'s `_apply_layout(self.region_layout)` call must be a true
    no-op: `_last_persisted_collapsed` already agrees (primed at
    construction from the same load), so `_schedule_layout_persist`'s
    no-op guard must never set `_pending_persist_layout`. A regression here
    would reopen exactly the race task-15462 flagged: two unordered
    writers on the same config key.

    Uses a non-default persisted value (RIGHT_RAIL collapsed) rather than
    the conftest stub's trivial `RegionLayout()`, so the no-op guard is
    genuinely exercised against a real divergent-from-empty value.
    """
    monkeypatch.setattr(
        _LOAD_REGION_LAYOUT_TARGET,
        lambda: RegionLayout(collapsed=frozenset({Region.RIGHT_RAIL})),
    )
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        await pilot.pause(0.2)
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert screen._pending_persist_layout is None, (
            "on_mount must not schedule a persist worker when construction "
            "already loaded and primed this exact layout"
        )


async def test_preferred_layout_survives_an_isolated_fresh_restart(
    monkeypatch, tmp_path
) -> None:
    """Only preferred side-pane state crosses a real config restart."""
    profile = tmp_path / "restart-profile"
    home = profile / "home"
    config_home = profile / "config-home"
    config_path = config_home / "tldw_cli" / "config.toml"
    home.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    for name in (
        "_CONFIG_CACHE",
        "_CONFIG_CACHE_SOURCE",
        "_SETTINGS_CACHE",
        "_SETTINGS_CACHE_SOURCE",
    ):
        monkeypatch.setattr(config_module, name, None)
    monkeypatch.setattr(
        _LOAD_REGION_LAYOUT_TARGET, region_layout_store.load_region_layout
    )

    desired = RegionLayout(
        collapsed=frozenset({Region.LEFT_RAIL, Region.ITEMS})
    )
    assert save_settings_to_cli_config(
        {
            "watchlists": {
                "collapsed_regions": [
                    "content",
                    Region.LEFT_RAIL.value,
                    Region.ITEMS.value,
                ],
                "layout_version": 1,
            }
        }
    )

    first = WatchlistsCollectionsScreen(_build_test_app())
    assert first.region_layout == desired
    first._article_focus_active = True
    first._effective_region_layout = RegionLayout(
        collapsed=frozenset(
            {Region.LEFT_RAIL, Region.ITEMS, Region.RIGHT_RAIL}
        )
    )
    first._responsive_region_layout = RegionLayout(
        collapsed=frozenset({Region.RIGHT_RAIL})
    )
    first._responsive_priority_lease = ResponsivePriorityLease(
        Region.RIGHT_RAIL, read_mode=True
    )

    config_module._invalidate_config_caches()
    restarted = WatchlistsCollectionsScreen(_build_test_app())
    assert restarted.region_layout == desired
    assert restarted._effective_region_layout == desired
    assert restarted._article_focus_active is False
    assert restarted._responsive_region_layout is None
    assert restarted._responsive_priority_lease is None

    persisted = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["watchlists"]["collapsed_regions"] == [
        Region.LEFT_RAIL.value,
        Region.ITEMS.value,
    ]
    assert persisted["watchlists"]["layout_version"] == (
        region_layout_store.LAYOUT_VERSION
    )
    assert "content" not in persisted["watchlists"]["collapsed_regions"]

    config_module._invalidate_config_caches()
    restarted_again = WatchlistsCollectionsScreen(_build_test_app())
    assert restarted_again.region_layout == desired
    assert Region.CONTENT not in restarted_again.region_layout.collapsed
