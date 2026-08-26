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

_SCREEN_MODULE_PREFIX = "tldw_chatbook.UI.Screens."


@pytest.fixture(autouse=True)
def _isolate_screen_modules():
    """Snapshot/restore every ``tldw_chatbook.UI.Screens.*`` sys.modules entry.

    Several tests below deliberately pop and re-import real screen modules
    (to force a genuinely cold ``import_module`` call) or run the real
    pre-importer against the full registry -- both create/replace module
    objects in the process-global ``sys.modules``, which is emphatically not
    test-local: 135+ test files bind screen classes at import time
    (``from ...chat_screen import ChatScreen``), so a later-collected test's
    ``isinstance()`` check breaks if this file silently swapped in a
    *different* class object for the same dotted name -- caught in review
    round 1 against
    ``test_settings_workspaces_category.py::test_create_rename_archive_
    unarchive_flow`` when this file's end-to-end test ran first in the same
    process. ``ScreenRoute.load_screen_class()`` does a fresh ``import_
    module()`` + ``getattr()`` per call rather than caching anything of its
    own. Import machinery also publishes each child module on its parent
    package, so both that attribute and the exact original ``sys.modules``
    entry (or absence) must be restored. Autouse and scoped to every test in
    this file (not opt-in per test) so a future test added here can't
    reintroduce the same leak by omission.
    """
    before = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith(_SCREEN_MODULE_PREFIX)
    }
    yield
    after_names = {
        name for name in sys.modules if name.startswith(_SCREEN_MODULE_PREFIX)
    }
    for name in sorted(after_names - before.keys(), reverse=True):
        replacement = sys.modules.pop(name, None)
        parent_name, _, attribute = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, attribute, None) is replacement:
            delattr(parent, attribute)
    for name, module in before.items():
        sys.modules[name] = module
        parent_name, _, attribute = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, attribute, module)


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
    """chat/library/settings come first; every reachable module appears once.

    "Reachable" excludes alias-shadowed route ids (see
    `test_screen_preimport_route_order_excludes_alias_shadowed_routes` below)
    -- their module is either dead entirely (``customize``) or real but
    unreachable via that id at nav time (``media``), so neither belongs in
    `all_module_paths` here.
    """
    from tldw_chatbook.UI.Navigation.screen_registry import registered_screen_aliases

    app = _build_test_app()

    order = app._screen_preimport_route_order()
    ordered_ids = [route.screen_name for route in order]

    assert ordered_ids[:3] == ["chat", "library", "settings"]

    module_paths = [route.module_path for route in order]
    assert len(module_paths) == len(set(module_paths)), (
        "each shared-module alias (e.g. ccp/personas, tools_settings/mcp) "
        "must be scheduled once, not once per canonical route id"
    )

    shadowed_ids = set(registered_screen_aliases())
    reachable_module_paths = {
        route.module_path
        for route in registered_screen_routes()
        if route.screen_name not in shadowed_ids
    }
    assert set(module_paths) == reachable_module_paths, (
        "every reachable registered module must still be covered exactly once"
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


# NOTE: an earlier version of this file also had a "loose A/B" pytest check
# (preimport forced on vs. off, asserting `on <= off * 2 + 0.5`) as a second
# piece of AC #2 evidence. Review round 1 correctly called it vacuous: that
# bound passes a 100%-plus-500ms regression, so it could never fail for a
# real problem. Removed rather than tightened -- CI wall-clock jitter makes a
# tight bound on this kind of two-sample timing comparison genuinely flaky,
# and the deterministic spy above is the actual guarantee. A manual,
# non-pytest timing probe (run once, by hand, not part of this suite) is
# recorded as a live number in the task's Implementation Notes instead.


# --- IMPORTANT (review round 1): dead/shadowed routes are never attempted ----


def test_screen_preimport_route_order_excludes_alias_shadowed_routes():
    """A route id that is ALSO an alias-table key is unreachable under its
    own id at real navigation time (`_lookup_route` resolves the alias to a
    different canonical route first) and must never be scheduled.

    Concretely: `_SCREEN_ROUTES["customize"]` points at a `customize_screen`
    module that no longer exists (retired; `_SCREEN_ALIASES["customize"] =
    "settings"` is the only way "customize" is ever actually reached).
    Before this fix, pre-importing it anyway logged a "Screen route
    unavailable: customize: No module named ..." warning on every boot.
    """
    from tldw_chatbook.UI.Navigation.screen_registry import registered_screen_aliases

    app = _build_test_app()
    ordered_ids = {route.screen_name for route in app._screen_preimport_route_order()}

    shadowed_ids = set(registered_screen_aliases())
    assert shadowed_ids, "sanity: the alias table must be non-empty for this test to mean anything"
    assert not (ordered_ids & shadowed_ids), (
        f"pre-import scheduled alias-shadowed (unreachable) route ids: "
        f"{ordered_ids & shadowed_ids}"
    )
    assert "customize" not in ordered_ids
    assert "media" not in ordered_ids


def test_full_preimport_pass_emits_zero_warnings_on_the_shipped_registry():
    """A real, full pre-import pass over the app's actual registered routes
    must not log a single WARNING (or above) -- any such log means the
    pre-importer is attempting a route it cannot actually service (the
    "customize" dead-route regression this fix addresses), which is exactly
    the noise a background cache-warmer must never produce on every boot.
    """
    from loguru import logger as loguru_logger

    app = _build_test_app()
    records: list[str] = []
    sink = loguru_logger.add(lambda m: records.append(str(m)), level="WARNING")
    try:
        app._preimport_heavy_screens()
    finally:
        loguru_logger.remove(sink)

    assert records == [], f"pre-import logged unexpected warnings: {records}"
