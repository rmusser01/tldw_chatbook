"""task-21113: the whole-registry screen pre-importer is a polite citizen.

The pass is a GIL-holding CPU burst on a daemon thread starting 0.2 s after
first paint. A live A/B on multi-core hardware measured NO first-interaction
penalty, so this is design hardening for 1-2-core machines, not a fix for a
measured regression -- and the measurement that drove its shape is the
distribution, not the total: with the initial screen already warm (the real
boot condition), 21 routes cost 361 ms of which three modules are 80%.

So the gap between routes is proportional to what the previous import took
rather than flat, the pass steps aside while a navigation is resolving, and
a machine below `SCREEN_PREIMPORT_LOW_CORE_THRESHOLD` usable CPUs gets a much
heavier version of the same throttle. What none of it does is touch
task-21110's single-route initial-screen warm-up, which shares this method:
gaps are strictly BETWEEN routes, so a one-route list is unchanged.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest

from tldw_chatbook import app as app_module
from tldw_chatbook.UI.Navigation import screen_registry

from Tests.UI.app_factory import _build_test_app


def _cheap_routes(count: int) -> list:
    """`count` routes whose `load_screen_class()` is a no-op dict hit."""
    base = screen_registry._SCREEN_ROUTES["chat"]
    return [
        replace(base, screen_name=f"task-21113-route-{index}") for index in range(count)
    ]


@pytest.fixture
def sleep_spy(monkeypatch):
    """Record every `time.sleep` the pre-importer performs, without sleeping."""
    recorded: list[float] = []

    def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(app_module.time, "sleep", fake_sleep)
    return recorded


# --- AC #1: yield between routes, proportional to the cost just paid --------


def test_gap_is_inserted_between_routes_only(monkeypatch, sleep_spy):
    """N routes produce N-1 pauses: none before the first, none after the last."""
    app = _build_test_app()
    monkeypatch.setattr(app, "_screen_preimport_pacing", lambda: (1.0, 0.10))
    pauses: list[float] = []
    monkeypatch.setattr(app, "_pause_between_preimports", pauses.append)

    app._preimport_screens(_cheap_routes(4))

    assert len(pauses) == 3
    assert sleep_spy == []  # the pause itself is stubbed above


def test_single_route_preimport_is_untouched(monkeypatch, sleep_spy):
    """task-21110's initial-screen warm-up races the splash: add nothing to it.

    Asserted on the pause CALL, not just on the sleep: a pause invoked with a
    zero gap sleeps nothing but still probes the navigation lock, and the
    guarantee for the single-route warm-up is that it reaches
    ``load_screen_class()`` with nothing at all in front of it.
    """
    app = _build_test_app()
    monkeypatch.setattr(app, "_screen_preimport_pacing", lambda: (3.0, 1.5))
    pauses: list[float] = []
    monkeypatch.setattr(app, "_pause_between_preimports", pauses.append)

    app._preimport_screens(_cheap_routes(1))

    assert pauses == []
    assert sleep_spy == []


def test_gap_tracks_the_previous_import_cost_and_respects_the_cap(monkeypatch):
    """The gap is `min(previous_cost * ratio, cap)`, not a flat constant.

    Driven with a fake clock so the assertion is on the arithmetic, not on
    how long a real import happens to take on the runner. Asserted on the
    pause CALL rather than on raw `time.sleep` values because the pause
    slices its sleep (task-22214) -- the arithmetic and the slicing are
    separate contracts with separate tests.
    """
    app = _build_test_app()
    monkeypatch.setattr(app, "_screen_preimport_pacing", lambda: (1.0, 0.10))

    # Each `load_screen_class()` "costs" the next value in this list.
    costs = [0.005, 0.400, 0.020]
    clock = {"now": 0.0}
    pauses: list[float] = []

    def fake_monotonic() -> float:
        return clock["now"]

    monkeypatch.setattr(app_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(app, "_pause_between_preimports", pauses.append)

    routes = _cheap_routes(4)
    real_load = type(routes[0]).load_screen_class
    calls = {"n": 0}

    def costed_load(self):
        index = calls["n"]
        calls["n"] += 1
        if index < len(costs):
            clock["now"] += costs[index]
        return real_load(self)

    monkeypatch.setattr(type(routes[0]), "load_screen_class", costed_load)

    app._preimport_screens(routes)

    # 5 ms import -> 5 ms gap; 400 ms import -> capped at 100 ms; 20 ms -> 20 ms.
    assert pauses == pytest.approx([0.005, 0.10, 0.020])


# --- task-22214: the cap must not turn the proportional yield into a no-op ---


def test_normal_tier_cap_covers_the_heaviest_measured_route():
    """The gap cap must sit ABOVE any real single-route import cost.

    task-22214: the payload grew until the heaviest route (library, 615 ms on
    a bytecode-compiling boot, M-series -- slower hardware proportionally
    worse) dwarfed the then-0.10 s cap, silently flattening the proportional
    yield into a ~90%-duty near-no-op for exactly the routes that matter.
    1.0 s is a floor, not the tuned value (2.0 s): it is above every
    single-route cost measured on fast hardware with margin, so a future
    "optimization" that quietly restores a sub-second cap reds here and has
    to bring new measurements.
    """
    assert app_module.SCREEN_PREIMPORT_MAX_ROUTE_GAP_SECONDS >= 1.0
    assert (
        app_module.SCREEN_PREIMPORT_LOW_CORE_MAX_ROUTE_GAP_SECONDS
        >= 3.0 * app_module.SCREEN_PREIMPORT_MAX_ROUTE_GAP_SECONDS / 2.0
    )


def test_gap_sleep_is_sliced_to_the_navigation_poll_size(monkeypatch):
    """A multi-second gap sleeps in poll-sized slices, never one big sleep.

    With the caps at 2.0 s / 6.0 s a single `time.sleep(gap)` would leave a
    quitting app waiting out the whole gap; sliced, the shutdown check
    between slices fires within 0.05 s (the 22200 `_interruptible_sleep`
    precedent).
    """
    app = _build_test_app()
    monkeypatch.setattr(app, "_screen_navigation_in_progress", lambda: False)
    slept: list[float] = []
    monkeypatch.setattr(app_module.time, "sleep", slept.append)

    app._pause_between_preimports(0.17)

    poll = app_module.SCREEN_PREIMPORT_NAVIGATION_POLL_SECONDS
    assert sum(slept) == pytest.approx(0.17)
    assert max(slept) <= poll + 1e-9
    assert slept == pytest.approx([poll, poll, poll, 0.17 - 3 * poll])


def test_shutdown_breaks_out_of_a_long_gap_mid_sleep(monkeypatch):
    """Quit during a capped-length gap: the thread stops within one slice."""
    app = _build_test_app()
    monkeypatch.setattr(app, "_screen_navigation_in_progress", lambda: False)

    slept: list[float] = []

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        if len(slept) == 2:
            app._shutting_down = True

    monkeypatch.setattr(app_module.time, "sleep", fake_sleep)

    app._pause_between_preimports(
        app_module.SCREEN_PREIMPORT_MAX_ROUTE_GAP_SECONDS
    )

    # Two slices happened, then the shutdown check stopped the gap AND the
    # navigation park never ran.
    assert len(slept) == 2


# --- AC #3a: low-core throttle ----------------------------------------------


def test_pacing_switches_to_the_low_core_tier_below_the_threshold(monkeypatch):
    app = _build_test_app()

    monkeypatch.setattr(
        app_module,
        "_usable_cpu_count",
        lambda: app_module.SCREEN_PREIMPORT_LOW_CORE_THRESHOLD - 1,
    )
    assert app._screen_preimport_pacing() == (
        app_module.SCREEN_PREIMPORT_LOW_CORE_YIELD_RATIO,
        app_module.SCREEN_PREIMPORT_LOW_CORE_MAX_ROUTE_GAP_SECONDS,
    )

    monkeypatch.setattr(
        app_module,
        "_usable_cpu_count",
        lambda: app_module.SCREEN_PREIMPORT_LOW_CORE_THRESHOLD,
    )
    assert app._screen_preimport_pacing() == (
        app_module.SCREEN_PREIMPORT_YIELD_RATIO,
        app_module.SCREEN_PREIMPORT_MAX_ROUTE_GAP_SECONDS,
    )


def test_low_core_tier_yields_strictly_more_than_the_normal_tier():
    """A throttle that is not heavier than the default is not a throttle."""
    assert (
        app_module.SCREEN_PREIMPORT_LOW_CORE_YIELD_RATIO
        > app_module.SCREEN_PREIMPORT_YIELD_RATIO
    )
    assert (
        app_module.SCREEN_PREIMPORT_LOW_CORE_MAX_ROUTE_GAP_SECONDS
        > app_module.SCREEN_PREIMPORT_MAX_ROUTE_GAP_SECONDS
    )


def test_usable_cpu_count_never_reports_zero(monkeypatch):
    """An unanswerable platform must throttle, not divide-by-nothing."""
    monkeypatch.delattr(app_module.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(app_module.os, "cpu_count", lambda: None)
    assert app_module._usable_cpu_count() == 1


def test_low_core_throttle_does_not_disable_the_feature(monkeypatch):
    """Below the threshold the pass still runs -- it only paces itself.

    Disabling would push each screen's import back onto the event loop at
    first navigation, on the machines least able to absorb it.
    """
    app = _build_test_app()
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1")
    monkeypatch.setattr(app_module, "_usable_cpu_count", lambda: 1)

    assert app._screen_preimport_enabled() is True

    imported: list[str] = []
    monkeypatch.setattr(app_module.time, "sleep", lambda _s: None)
    routes = _cheap_routes(3)
    monkeypatch.setattr(
        type(routes[0]),
        "load_screen_class",
        lambda self: imported.append(self.screen_name),
    )

    app._preimport_screens(routes)

    assert len(imported) == 3


# --- AC #3b: park while a navigation is resolving ---------------------------


def test_pass_parks_while_the_navigation_lock_is_held(monkeypatch):
    """The pre-importer waits out a navigation instead of racing it."""
    import asyncio

    app = _build_test_app()
    app._screen_navigation_lock_instance = asyncio.Lock()
    monkeypatch.setattr(app, "_screen_preimport_pacing", lambda: (0.0, 0.0))

    # Pretend the lock is held for the first two polls, then released.
    held = {"remaining": 2}

    def navigating() -> bool:
        if held["remaining"] > 0:
            held["remaining"] -= 1
            return True
        return False

    monkeypatch.setattr(app, "_screen_navigation_in_progress", navigating)
    polls: list[float] = []
    monkeypatch.setattr(app_module.time, "sleep", polls.append)

    app._pause_between_preimports(0.0)

    assert polls == [
        app_module.SCREEN_PREIMPORT_NAVIGATION_POLL_SECONDS,
        app_module.SCREEN_PREIMPORT_NAVIGATION_POLL_SECONDS,
    ]


def test_navigation_park_is_bounded(monkeypatch):
    """A navigation lock that is never released throttles, never strands."""
    app = _build_test_app()
    monkeypatch.setattr(app, "_screen_navigation_in_progress", lambda: True)
    polls: list[float] = []
    monkeypatch.setattr(app_module.time, "sleep", polls.append)

    app._pause_between_preimports(0.0)

    assert len(polls) == app_module.SCREEN_PREIMPORT_MAX_NAVIGATION_POLLS
    parked_for = len(polls) * app_module.SCREEN_PREIMPORT_NAVIGATION_POLL_SECONDS
    assert parked_for == pytest.approx(
        app_module.SCREEN_PREIMPORT_NAVIGATION_PARK_LIMIT_SECONDS
    )


def test_navigation_probe_reads_a_real_lock_without_touching_the_loop():
    """`_screen_navigation_in_progress` reflects the real asyncio lock."""
    import asyncio

    app = _build_test_app()
    assert app._screen_navigation_in_progress() is False

    async def drive() -> tuple[bool, bool]:
        lock = app._screen_navigation_lock()
        async with lock:
            # Read it the way the daemon thread does: from another thread.
            observed: list[bool] = []
            worker = threading.Thread(
                target=lambda: observed.append(app._screen_navigation_in_progress())
            )
            worker.start()
            worker.join(timeout=5)
            during = observed[0]
        return during, app._screen_navigation_in_progress()

    during, after = asyncio.run(drive())
    assert during is True
    assert after is False


# --- Shutdown ----------------------------------------------------------------


def test_shutdown_stops_the_pass_before_the_next_route(monkeypatch):
    """Quit mid-pass: the daemon thread drops out instead of finishing 21 imports."""
    app = _build_test_app()
    monkeypatch.setattr(app, "_screen_preimport_pacing", lambda: (0.0, 0.0))
    monkeypatch.setattr(app_module.time, "sleep", lambda _s: None)

    imported: list[str] = []
    routes = _cheap_routes(5)

    def load(self):
        imported.append(self.screen_name)
        if len(imported) == 2:
            app._shutting_down = True

    monkeypatch.setattr(type(routes[0]), "load_screen_class", load)

    app._preimport_screens(routes)

    assert len(imported) == 2


def test_shutdown_breaks_out_of_a_navigation_park(monkeypatch):
    """A quit during the park does not wait out the bound."""
    app = _build_test_app()
    monkeypatch.setattr(app, "_screen_navigation_in_progress", lambda: True)

    polls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        polls.append(seconds)
        if len(polls) == 3:
            app._shutting_down = True

    monkeypatch.setattr(app_module.time, "sleep", fake_sleep)

    app._pause_between_preimports(0.0)

    assert len(polls) == 3


def test_preimport_thread_terminates_promptly_on_shutdown(monkeypatch):
    """End to end on a real thread: set `_shutting_down`, the thread exits."""
    app = _build_test_app()
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1")
    monkeypatch.setattr(app, "_screen_preimport_pacing", lambda: (1.0, 0.05))

    started = threading.Event()
    routes = _cheap_routes(200)

    def load(self):
        started.set()
        time.sleep(0.005)

    monkeypatch.setattr(type(routes[0]), "load_screen_class", load)
    monkeypatch.setattr(app, "_screen_preimport_route_order", lambda: routes)

    app._schedule_screen_preimport()
    thread = app._screen_preimport_thread
    assert thread is not None
    assert started.wait(timeout=5)

    app._shutting_down = True
    thread.join(timeout=5)
    assert not thread.is_alive()


# --- AC #4: the env override still works both ways ---------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", False), ("false", False), ("no", False), ("1", True), ("true", True)],
)
def test_env_override_survives_the_pacing_change(monkeypatch, value, expected):
    app = _build_test_app()
    monkeypatch.setattr(app_module, "_usable_cpu_count", lambda: 1)
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", value)
    assert app._screen_preimport_enabled() is expected
