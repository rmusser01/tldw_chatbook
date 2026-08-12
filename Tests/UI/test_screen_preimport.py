"""Tests for task-15472: background pre-import of screen modules after first
paint.

Before this task, the first navigation to any tab paid for a synchronous,
UI-thread `import_module` call inside the FIFO-locked navigation worker
(`UI/Navigation/screen_registry.py`'s `ScreenRoute.load_screen_class`) --
chat_screen.py is ~20k lines, library_screen.py ~26k, settings_screen.py
~19k (Docs/Design/2026-08-11-input-latency-audit.md). `TldwCli` now schedules
a background daemon thread, strictly after first paint (`_ui_ready = True`),
that warms `sys.modules` for every registered screen route so that cost is
paid once, off the UI thread, instead of on the user's first click.

Coverage:
  - `_screen_preimport_route_order` prioritizes chat/library/settings and
    dedupes routes that share one module.
  - `_preimport_screens` (the per-route worker body) leaves a warmed route's
    subsequent `load_screen_class()` call effectively free (AC #1).
  - A module that fails to import is swallowed by the pre-importer, and a
    real navigation's `load_screen_class()` call for that same route still
    fails/degrades identically to an unpatched baseline (AC #3).
  - `_screen_preimport_enabled` defaults off under pytest and respects the
    `TLDW_SCREEN_PREIMPORT` override in both directions; `_schedule_screen_
    preimport` is idempotent and truly runs off the calling thread.
  - Live, in-process `app.run_test()` proof that the preimport thread is
    never scheduled before `_ui_ready` flips true (AC #2's ordering claim),
    plus a loose timing sanity check that forcing it on doesn't move
    time-to-`_ui_ready` (honest-numbers A/B, per the task's evidence list).
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from dataclasses import replace

import pytest

from tldw_chatbook.UI.Navigation import screen_registry
from tldw_chatbook.UI.Navigation.screen_registry import (
    ScreenRoute,
    registered_screen_routes,
)

from Tests.UI.app_factory import _build_test_app


async def _wait_until(condition, *, pause, timeout_seconds: float = 5.0) -> None:
    """Poll `condition` via the pilot's own clock instead of sleeping blind."""

    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await pause(0.02)
    if condition():
        return
    raise AssertionError(f"condition was not met within {timeout_seconds:.1f}s")


# --- Route ordering ----------------------------------------------------------


def test_screen_preimport_route_order_prioritizes_and_dedupes():
    """chat/library/settings come first; every distinct module appears once."""
    app = _build_test_app()

    order = app._screen_preimport_route_order()
    ordered_ids = [route.screen_name for route in order]

    assert ordered_ids[:3] == ["chat", "library", "settings"]

    module_paths = [route.module_path for route in order]
    assert len(module_paths) == len(set(module_paths)), (
        "each shared-module alias (e.g. ccp/personas, tools_settings/mcp) "
        "must be scheduled once, not once per canonical route id"
    )

    all_module_paths = {route.module_path for route in registered_screen_routes()}
    assert set(module_paths) == all_module_paths, (
        "every registered module must still be covered exactly once"
    )


# --- AC #1: a pre-imported route's next load is effectively free -------------


def test_load_after_preimport_is_effectively_free():
    """Warming a route via `_preimport_screens` makes its next load ~free.

    `load_screen_class()` always calls `import_module()` -- Python's own
    `sys.modules` cache is what makes a warm call cheap, not a call-count
    difference -- so the honest evidence is timing: a cold import measurably
    costs real work, the same call immediately afterward does not.
    """
    app = _build_test_app()
    route = next(
        r for r in app._screen_preimport_route_order() if r.screen_name == "chat"
    )
    sys.modules.pop(route.module_path, None)

    cold_start = time.perf_counter()
    cold_class = route.load_screen_class()
    cold_duration = time.perf_counter() - cold_start
    assert cold_class is not None
    assert route.module_path in sys.modules

    warm_start = time.perf_counter()
    warm_class = route.load_screen_class()
    warm_duration = time.perf_counter() - warm_start

    assert warm_class is cold_class, "the cached module object must be reused"
    # Generous bound: the warm call must be dramatically cheaper than the
    # cold one (sys.modules hit vs. real parse+exec), without pinning an
    # absolute millisecond figure that would flake on slow CI hardware.
    assert warm_duration < max(cold_duration / 4, 0.002), (
        f"warm load ({warm_duration * 1000:.3f}ms) was not near-free "
        f"relative to the cold load ({cold_duration * 1000:.3f}ms)"
    )


def test_preimport_screens_warms_every_target_module():
    """`_preimport_screens` actually lands each route's module in `sys.modules`."""
    app = _build_test_app()
    targets = [
        r
        for r in app._screen_preimport_route_order()
        if r.screen_name in ("acp", "stats", "logs")
    ]
    assert len(targets) == 3
    for route in targets:
        sys.modules.pop(route.module_path, None)

    app._preimport_screens(targets)

    for route in targets:
        assert route.module_path in sys.modules


# --- AC #3: a failing pre-import changes nothing about nav-time behavior -----


def test_preimport_swallows_a_broken_module_and_navtime_behavior_is_unchanged(
    monkeypatch,
):
    """A module that raises something other than Import/AttributeError at
    import time must not crash the pre-import thread -- and a real
    navigation attempt for that same route must still fail exactly as it
    would have with no pre-import ever attempted."""
    app = _build_test_app()

    broken_route = ScreenRoute(
        screen_name="task-15472-broken-route",
        canonical_tab="task-15472-broken-route",
        module_path="tldw_chatbook.UI.Screens.chat_screen",
        class_name="ChatScreen",
    )
    real_import_module = screen_registry.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == broken_route.module_path:
            raise RuntimeError("boom: simulated broken screen module")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(screen_registry, "import_module", fake_import_module)

    # Baseline: what a real navigation attempt does today, before any
    # pre-import is ever attempted.
    with pytest.raises(RuntimeError, match="boom"):
        broken_route.load_screen_class()

    # The pre-importer must swallow the same failure without raising.
    app._preimport_screens([broken_route])

    # A real navigation attempt afterward must fail IDENTICALLY -- proving
    # the swallowed pre-import attempt left no trace (no partial module
    # cached, no altered exception).
    with pytest.raises(RuntimeError, match="boom"):
        broken_route.load_screen_class()


def test_preimport_of_a_missing_module_matches_existing_none_degrade():
    """The already-existing ImportError -> None degrade (missing optional
    screen) must be unaffected by having been pre-imported first."""
    app = _build_test_app()
    real_route = screen_registry._SCREEN_ROUTES["chat"]
    missing_route = replace(
        real_route,
        screen_name="task-15472-missing-route",
        module_path="tldw_chatbook.UI.Screens.no_such_screen_xyz_15472",
    )

    assert missing_route.load_screen_class() is None  # today's baseline

    app._preimport_screens([missing_route])  # must not raise

    assert missing_route.load_screen_class() is None  # unchanged after


# --- Gating: default-off under pytest, env override, idempotent, off-thread --


def test_screen_preimport_enabled_defaults_off_under_pytest(monkeypatch):
    app = _build_test_app()
    monkeypatch.delenv("TLDW_SCREEN_PREIMPORT", raising=False)
    assert "PYTEST_CURRENT_TEST" in os.environ, (
        "this assertion only makes sense while pytest is actually running "
        "the test -- if it ever fails, the whole premise of the default-off "
        "gate needs re-examination, not just this test"
    )
    assert app._screen_preimport_enabled() is False


@pytest.mark.parametrize(
    "value,expected",
    [("1", True), ("true", True), ("TRUE", True), ("0", False), ("false", False)],
)
def test_screen_preimport_enabled_env_override(monkeypatch, value, expected):
    app = _build_test_app()
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", value)
    assert app._screen_preimport_enabled() is expected


def test_schedule_screen_preimport_noop_when_disabled(monkeypatch):
    app = _build_test_app()
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "0")
    app._schedule_screen_preimport()
    assert app._screen_preimport_thread is None


def test_schedule_screen_preimport_runs_off_the_calling_thread_and_is_idempotent(
    monkeypatch,
):
    app = _build_test_app()
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1")

    calls: list[int] = []
    worker_thread_idents: list[int] = []
    caller_ident = threading.get_ident()

    def fake_preimport() -> None:
        calls.append(1)
        worker_thread_idents.append(threading.get_ident())
        time.sleep(0.02)

    monkeypatch.setattr(app, "_preimport_heavy_screens", fake_preimport)

    app._schedule_screen_preimport()
    first_thread = app._screen_preimport_thread
    assert first_thread is not None
    assert first_thread.daemon is True

    # A second call while the first is still running (or after) must not
    # spawn a second thread.
    app._schedule_screen_preimport()
    assert app._screen_preimport_thread is first_thread

    first_thread.join(timeout=5)
    assert not first_thread.is_alive()
    assert calls == [1]
    assert worker_thread_idents == [first_thread.ident]
    assert worker_thread_idents[0] != caller_ident, (
        "the pre-import work must run on a real OS thread, never inline on "
        "the caller (which, in production, is the asyncio loop's own timer "
        "callback)"
    )


# --- End-to-end: the real thread actually warms the priority routes ----------


def test_schedule_screen_preimport_end_to_end_warms_priority_routes(monkeypatch):
    priority_modules = (
        "tldw_chatbook.UI.Screens.chat_screen",
        "tldw_chatbook.UI.Screens.library_screen",
        "tldw_chatbook.UI.Screens.settings_screen",
    )
    for module_name in priority_modules:
        sys.modules.pop(module_name, None)

    app = _build_test_app()
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1")

    app._schedule_screen_preimport()
    thread = app._screen_preimport_thread
    assert thread is not None
    thread.join(timeout=30)
    assert not thread.is_alive()

    for module_name in priority_modules:
        assert module_name in sys.modules

    # The real navigation path resolves through the now-warm module.
    _name, _tab, screen_class = screen_registry.resolve_screen_target("chat")
    assert screen_class is not None
    assert screen_class.__module__ == "tldw_chatbook.UI.Screens.chat_screen"


# --- AC #2: pre-import never starts before first paint (`_ui_ready`) ---------


@pytest.mark.asyncio
async def test_screen_preimport_scheduled_only_after_ui_ready_flips(monkeypatch):
    """`_schedule_screen_preimport` never runs while `_ui_ready` is still False.

    A real-time poll racing the 0.2s deferred-timer callback against
    wall-clock jitter (splash animation, logging overhead, scheduler noise)
    is not a reliable way to prove an ordering guarantee -- an earlier
    version of this test polled `_ui_ready`/`_screen_preimport_thread` from
    outside and flaked because enough real time had already elapsed by the
    time the poll first ran. Instead, spy directly on the call: capture what
    `self._ui_ready` was AT THE INSTANT the scheduler actually invoked
    `_schedule_screen_preimport` (wired via `self.set_timer(...)` inside
    `_schedule_deferred_startup_work()`, the last statement of
    `_post_mount_setup()`, which sets `_ui_ready = True` earlier in that same
    never-awaited-in-between coroutine body). No race window: the assertion
    is about a value observed synchronously inside the real call itself.
    """
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "0")  # only the ordering matters here
    app = _build_test_app()

    ui_ready_at_call_time: list[bool] = []
    real_schedule = type(app)._schedule_screen_preimport

    def spy_schedule(self) -> None:
        ui_ready_at_call_time.append(self._ui_ready)
        return real_schedule(self)

    monkeypatch.setattr(type(app), "_schedule_screen_preimport", spy_schedule)

    async with app.run_test(size=(120, 36)) as pilot:
        await _wait_until(
            lambda: ui_ready_at_call_time, pause=pilot.pause, timeout_seconds=5.0
        )
        assert ui_ready_at_call_time == [True], (
            "screen pre-import must only be scheduled once _ui_ready is True"
        )
        await pilot.pause(0.05)


@pytest.mark.asyncio
async def test_cold_start_time_to_ui_ready_is_not_regressed_by_preimport(monkeypatch):
    """Loose A/B sanity check: forcing the pre-importer on vs off must not
    move time-to-`_ui_ready` by more than a generous margin.

    This is a secondary, "honest numbers" check on top of the deterministic
    ordering test above -- structurally the preimport cannot run before
    `_ui_ready` (see that test), so this is not expected to catch anything
    the ordering test wouldn't, but it's cheap insurance against a future
    refactor that moves the trigger earlier.
    """

    async def timed_ui_ready(enabled: bool) -> float:
        monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1" if enabled else "0")
        app = _build_test_app()
        start = time.perf_counter()
        async with app.run_test(size=(120, 36)) as pilot:
            await _wait_until(lambda: app._ui_ready, pause=pilot.pause)
            elapsed = time.perf_counter() - start
            thread = app._screen_preimport_thread
            await pilot.pause(0.05)
        if thread is not None:
            thread.join(timeout=15)
        return elapsed

    off_duration = await timed_ui_ready(False)
    on_duration = await timed_ui_ready(True)

    # Generous absolute+relative slack: this asserts "not regressed", not
    # "identical" -- CI hardware and scheduling jitter make tight bounds
    # flaky, and the real guarantee is the structural ordering test above.
    assert on_duration <= off_duration * 2 + 0.5, (
        f"time-to-ui_ready grew suspiciously with pre-import enabled: "
        f"off={off_duration * 1000:.1f}ms on={on_duration * 1000:.1f}ms"
    )
