"""Tests for task-21110: overlap the splash with the initial screen's import.

Before this task, a boot with the splash enabled (the default) was strictly
serial: the splash owned the event loop for its full duration doing nothing
else, and only when it closed did `_push_initial_screen` synchronously
`import_module` the initial route's module on that same loop -- ~0.31s warm,
~0.94s on a first boot after an upgrade (measured, isolated-profile probe,
`Docs/Design/2026-08-22-holistic-perf-review.md` finding 21110). The existing
background pre-importer (task-15472) structurally could not help: it is armed
by `_schedule_deferred_startup_work()` at the tail of `_post_mount_setup`,
which itself only runs AFTER that push.

`TldwCli` now schedules the SAME `_preimport_screens` worker, for the single
resolved initial route, on a daemon thread 0.2s after the splash mounts.

Coverage:
  - The route warmed is exactly the route `_push_initial_screen` will import
    (same alias / shell-destination resolution), including for alias routes
    and for the first-run Home redirect.
  - The worker runs off the calling thread, is idempotent, and every gate
    (feature off, shutting down, splash already closed, screen already
    pushed, unroutable target) is a no-op.
  - A pre-import that FAILS changes nothing about what the real push does.
  - Live, in-process `app.run_test()` proof that the initial screen's module
    is imported by the pre-import thread and NOT by the event loop, and that
    a splash-disabled boot never starts the thread at all.
  - The zero-delay branch, which exists because Textual 8's `set_timer(0.0)`
    raises ZeroDivisionError inside the timer's own task and never fires.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from dataclasses import replace

import pytest

from tldw_chatbook import app as app_module
from tldw_chatbook.Constants import TAB_CHAT, TAB_HOME
from tldw_chatbook.UI.Navigation import screen_registry
from tldw_chatbook.UI.Navigation.screen_registry import (
    ScreenRoute,
    resolve_screen_route,
    resolve_screen_target,
)

from Tests.UI.app_factory import _build_test_app

_SCREEN_MODULE_PREFIX = "tldw_chatbook.UI.Screens."
CHAT_MODULE = "tldw_chatbook.UI.Screens.chat_screen"


@pytest.fixture(autouse=True)
def _isolate_screen_modules():
    """Snapshot/restore every ``tldw_chatbook.UI.Screens.*`` sys.modules entry.

    Same reasoning (and the same regression) as the fixture of the same name
    in ``test_screen_preimport.py``: tests here deliberately pop and re-import
    real screen modules, and ``sys.modules`` is process-global -- 135+ test
    files bind screen classes at import time, so leaving a *different* class
    object behind under the same dotted name breaks a later test's
    ``isinstance`` check.
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


def _arm(app, *, initial_tab: str = TAB_CHAT):
    """Put an app in the state the splash-mount timer fires in."""
    app.splash_screen_active = True
    app._initial_screen_pushed = False
    app._initial_tab_value = initial_tab
    return app


async def _wait_until(condition, *, pause=None, timeout_seconds: float = 30.0) -> bool:
    """Poll `condition`, driving the app's own clock when a pilot is given."""
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return True
        if pause is None:
            await asyncio.sleep(0.01)
        else:
            await pause(0.02)
    return bool(condition())


# --- the warmed route is the route the push will use -------------------------


@pytest.mark.parametrize(
    "configured_tab",
    ["chat", "library", "home", "settings", "search", "notes", "coding"],
)
def test_warmed_route_module_matches_the_module_the_push_would_import(
    configured_tab,
):
    """`_initial_screen_preimport_route()` must agree with the real push.

    `_push_initial_screen` resolves through `resolve_screen_target()`, which
    imports; the warm-up resolves through `resolve_screen_route()`, which does
    not. Warming a different module than the push then imports would leave the
    finding's cost exactly where it was -- silently. Alias routes
    ("search"/"notes" -> Library, "coding" -> Chat) are the interesting cases.
    """
    app = _arm(_build_test_app(), initial_tab=configured_tab)

    warmed = app._initial_screen_preimport_route()
    _name, _tab, screen_class = resolve_screen_target(configured_tab)

    assert warmed is not None
    assert screen_class is not None
    assert warmed.module_path == screen_class.__module__


def test_warmed_route_follows_the_first_run_home_redirect():
    """A first-run boot lands on Home, so Home is what must be warmed.

    `_resolve_initial_shell_route()` -- not the raw configured default -- is
    the decision the push makes, so the warm-up has to go through it too.
    """
    app = _arm(_build_test_app(), initial_tab="library")
    app.app_config["_first_run"] = True

    assert app._resolve_initial_shell_route() == TAB_HOME
    warmed = app._initial_screen_preimport_route()
    assert warmed is not None
    assert warmed.module_path == resolve_screen_route(TAB_HOME).module_path


def test_warmed_route_is_none_for_an_unroutable_target(monkeypatch):
    """An unroutable configured default warms nothing, and does not raise."""
    app = _arm(_build_test_app())
    monkeypatch.setattr(
        type(app),
        "_resolve_initial_shell_route",
        lambda self: "task-21110-no-such-route",
    )

    assert app._initial_screen_preimport_route() is None

    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1")
    app._schedule_initial_screen_preimport()
    assert app._initial_screen_preimport_thread is None


def test_warmed_route_is_none_when_route_resolution_raises(monkeypatch):
    """A raising route resolution is swallowed, not propagated into on_mount."""
    app = _arm(_build_test_app())

    def boom(self):
        raise RuntimeError("boom: route resolution exploded")

    monkeypatch.setattr(type(app), "_resolve_initial_shell_route", boom)

    assert app._initial_screen_preimport_route() is None


# --- the worker: off-thread, idempotent, gated -------------------------------


def test_schedule_warms_the_initial_module_on_a_background_thread(monkeypatch):
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1")
    app = _arm(_build_test_app(), initial_tab="chat")
    for name in [n for n in list(sys.modules) if n.startswith(CHAT_MODULE)]:
        sys.modules.pop(name, None)
    caller_ident = threading.get_ident()
    # Record where the WORK ran, not merely that a thread object exists: an
    # implementation that imports inline and then spawns an idle thread would
    # satisfy every ident/daemon assertion while stalling the event loop.
    worker_idents: list[int] = []
    real_preimport = app._preimport_screens

    def recording_preimport(routes):
        worker_idents.append(threading.get_ident())
        return real_preimport(routes)

    monkeypatch.setattr(app, "_preimport_screens", recording_preimport)

    app._schedule_initial_screen_preimport()
    thread = app._initial_screen_preimport_thread

    assert thread is not None
    assert thread.daemon is True
    thread.join(timeout=60)
    assert worker_idents == [thread.ident], (
        f"the import ran on {worker_idents} but the pre-import thread is "
        f"{thread.ident}; in production the caller is the event loop's own "
        "timer callback, which is exactly what must not block"
    )
    assert worker_idents[0] != caller_ident
    assert not thread.is_alive()
    assert CHAT_MODULE in sys.modules

    # ...and the real navigation path now resolves through the warm module.
    _name, _tab, screen_class = resolve_screen_target("chat")
    assert screen_class is not None
    assert screen_class.__module__ == CHAT_MODULE


def test_schedule_is_idempotent(monkeypatch):
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1")
    app = _arm(_build_test_app())
    calls: list[tuple] = []
    monkeypatch.setattr(
        app, "_preimport_screens", lambda routes: calls.append(tuple(routes))
    )

    app._schedule_initial_screen_preimport()
    first = app._initial_screen_preimport_thread
    app._schedule_initial_screen_preimport()

    assert first is not None
    assert app._initial_screen_preimport_thread is first
    first.join(timeout=10)
    assert len(calls) == 1
    assert len(calls[0]) == 1, "exactly one route is warmed, not the registry"


@pytest.mark.parametrize("order", ["initial_first", "registry_first"])
def test_the_two_preimporters_never_suppress_each_other(monkeypatch, order):
    """The two pre-importers must not cancel each other out.

    They are separate mechanisms with separate once-guards. Sharing one
    thread handle would make whichever ran first silently suppress the other
    -- and the direction that matters is not obvious from the production
    ordering alone, so both orders are exercised: what is asserted is that
    BOTH bodies actually ran, not merely that two thread objects exist.
    """
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1")
    app = _arm(_build_test_app())
    ran: list[str] = []
    monkeypatch.setattr(app, "_preimport_screens", lambda routes: ran.append("initial"))
    monkeypatch.setattr(app, "_preimport_heavy_screens", lambda: ran.append("registry"))

    if order == "initial_first":
        app._schedule_initial_screen_preimport()
        app._schedule_screen_preimport()
    else:
        app._schedule_screen_preimport()
        app._schedule_initial_screen_preimport()

    assert app._initial_screen_preimport_thread is not None
    assert app._screen_preimport_thread is not None
    assert app._initial_screen_preimport_thread is not app._screen_preimport_thread
    for thread in (app._initial_screen_preimport_thread, app._screen_preimport_thread):
        thread.join(timeout=10)
    assert sorted(ran) == ["initial", "registry"], (
        f"one pre-importer suppressed the other (ran={ran}, order={order})"
    )


@pytest.mark.parametrize(
    "gate",
    ["feature_disabled", "shutting_down", "splash_already_closed", "already_pushed"],
)
def test_schedule_is_a_noop_behind_every_gate(monkeypatch, gate):
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1")
    app = _arm(_build_test_app())
    monkeypatch.setattr(
        app,
        "_preimport_screens",
        lambda routes: pytest.fail("the gate should have stopped this"),
    )

    if gate == "feature_disabled":
        monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "0")
    elif gate == "shutting_down":
        app._shutting_down = True
    elif gate == "splash_already_closed":
        app.splash_screen_active = False
    elif gate == "already_pushed":
        app._initial_screen_pushed = True

    app._schedule_initial_screen_preimport()

    assert app._initial_screen_preimport_thread is None


def test_failed_initial_preimport_leaves_navtime_behaviour_unchanged(monkeypatch):
    """A pre-import that raises must not change what the real push then does.

    The push must fail identically -- same exception, same message -- and the
    failure must not be swallowed into a silently broken first screen.
    """
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1")
    app = _arm(_build_test_app())
    for name in [n for n in list(sys.modules) if n.startswith(CHAT_MODULE)]:
        sys.modules.pop(name, None)

    real_import_module = screen_registry.import_module
    attempts: list[str] = []

    def fake_import_module(name, *args, **kwargs):
        if name == CHAT_MODULE:
            attempts.append(threading.current_thread().name)
            raise RuntimeError("boom: simulated broken initial screen module")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(screen_registry, "import_module", fake_import_module)
    route = resolve_screen_route("chat")

    # Baseline: what the real push does with no pre-import at all.
    with pytest.raises(RuntimeError, match="boom"):
        route.load_screen_class()

    app._schedule_initial_screen_preimport()
    thread = app._initial_screen_preimport_thread
    assert thread is not None
    thread.join(timeout=60)
    assert not thread.is_alive(), "a raising import must not wedge the thread"

    # Identical afterwards: no partial module cached, no altered exception.
    with pytest.raises(RuntimeError, match="boom"):
        route.load_screen_class()
    assert attempts.count("tldw-initial-screen-preimport") == 1
    assert CHAT_MODULE not in sys.modules


def test_missing_module_degrade_is_unchanged_by_a_pre_import_attempt(monkeypatch):
    """The existing ImportError -> None degrade survives being pre-imported."""
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1")
    app = _arm(_build_test_app())
    missing_route = replace(
        screen_registry._SCREEN_ROUTES["chat"],
        screen_name="task-21110-missing-route",
        module_path="tldw_chatbook.UI.Screens.no_such_screen_xyz_21110",
    )
    monkeypatch.setattr(
        app, "_initial_screen_preimport_route", lambda: missing_route
    )

    assert missing_route.load_screen_class() is None  # today's baseline

    app._schedule_initial_screen_preimport()
    app._initial_screen_preimport_thread.join(timeout=30)

    assert missing_route.load_screen_class() is None  # unchanged after


# --- the delay constant and Textual's zero-timer landmine --------------------


def test_delay_constant_leaves_real_overlap_inside_the_default_splash():
    """The delay must be > 0 and small relative to the splash it hides behind.

    > 0 because Textual 8's `set_timer(0.0)` never fires (see below); small
    because the whole point is the overlap -- the measured warm import is
    ~0.31s and the cold one ~0.94s, both of which have to fit inside the
    default 1.5s splash after this delay has elapsed.
    """
    delay = app_module.SPLASH_INITIAL_SCREEN_PREIMPORT_DELAY_SECONDS
    assert 0 < delay <= 0.5
    assert delay + 0.94 < 1.5, (
        "a first-boot-after-upgrade import must still finish before the "
        "default 1.5s splash closes"
    )


@pytest.mark.asyncio
async def test_zero_delay_falls_back_to_call_after_refresh(monkeypatch):
    """A 0s delay must still schedule the warm-up.

    Textual 8's `Timer._run` computes `(now - start) / interval`, so
    `set_timer(0.0)` raises ZeroDivisionError inside the timer's own task and
    the callback NEVER fires -- silently, because nobody retrieves that task's
    exception. This was not theoretical: the 0.0s arm of this feature's own
    delay A/B looked like a stutter-free win purely because no pre-import had
    happened at all.
    """
    monkeypatch.setattr(
        app_module, "SPLASH_INITIAL_SCREEN_PREIMPORT_DELAY_SECONDS", 0.0
    )
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1")
    app = _build_test_app()
    scheduled: list[str] = []
    monkeypatch.setattr(
        type(app),
        "_schedule_initial_screen_preimport",
        lambda self: scheduled.append("called"),
    )

    async with app.run_test(size=(120, 36)):
        assert await _wait_until(lambda: bool(scheduled), timeout_seconds=20.0), (
            "a 0s delay must not silently drop the warm-up on the floor"
        )


# --- live: the loop never pays the initial import ----------------------------


@pytest.mark.asyncio
async def test_initial_screen_module_is_imported_by_the_thread_not_the_loop(
    monkeypatch,
):
    """The money test: which thread pays for the initial screen's import.

    Asserting only "the module ends up in sys.modules" would pass on the
    unfixed code -- `_push_initial_screen` puts it there too, just on the
    event loop, 1.5s later. So record the thread of the first `import_module`
    call for that module and require it to be the pre-import thread.
    """
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1")
    for name in [n for n in list(sys.modules) if n.startswith(CHAT_MODULE)]:
        sys.modules.pop(name, None)

    importers: list[tuple[str, float]] = []
    real_import_module = screen_registry.import_module

    def spy_import_module(name, *args, **kwargs):
        if name == CHAT_MODULE:
            importers.append(
                (threading.current_thread().name, time.perf_counter())
            )
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(screen_registry, "import_module", spy_import_module)

    app = _build_test_app(configured_default="chat")
    async with app.run_test(size=(120, 36)) as pilot:
        assert await _wait_until(
            lambda: bool(importers), pause=pilot.pause, timeout_seconds=30.0
        ), "the initial screen module was never imported"
        first_importer, _at = importers[0]
        assert first_importer == "tldw-initial-screen-preimport", (
            f"the initial screen's import was paid by {first_importer!r}; the "
            "whole point of this task is that the event loop must not pay it"
        )
        assert app.splash_screen_active, (
            "the warm-up must start while the splash is still on screen -- "
            "otherwise nothing is being overlapped"
        )
        assert not getattr(app, "_initial_screen_pushed", False)

        thread = app._initial_screen_preimport_thread
        assert thread is not None
        assert await _wait_until(
            lambda: getattr(app, "_ui_ready", False),
            pause=pilot.pause,
            timeout_seconds=60.0,
        )
        assert type(app.screen).__name__ == "ChatScreen"
        await pilot.pause(0.2)


@pytest.mark.asyncio
async def test_no_overlap_thread_when_the_splash_is_disabled(monkeypatch):
    """Splash off: the boot path is untouched and no extra thread appears.

    The splash switch is patched at `app.py`'s own `get_cli_setting` (what
    `compose()` actually reads) rather than written into the sandbox config
    file: this test must not leave a config mutation behind for whatever runs
    next in the same session.
    """
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1")

    app = _build_test_app(configured_default="chat")
    real_get_cli_setting = app_module.get_cli_setting

    def splash_off(section, key=None, default=None, *args, **kwargs):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default, *args, **kwargs)

    monkeypatch.setattr(app_module, "get_cli_setting", splash_off)
    async with app.run_test(size=(120, 36)) as pilot:
        assert await _wait_until(
            lambda: getattr(app, "_ui_ready", False),
            pause=pilot.pause,
            timeout_seconds=60.0,
        )
        assert app.splash_screen_active is False
        assert app._initial_screen_preimport_thread is None, (
            "no splash means nothing to overlap -- the thread must not exist"
        )
        assert type(app.screen).__name__ == "ChatScreen"
        await pilot.pause(0.2)


@pytest.mark.parametrize(
    "schedule",
    ["_schedule_initial_screen_preimport", "_schedule_screen_preimport"],
)
def test_a_thread_that_refuses_to_start_degrades_instead_of_raising(
    monkeypatch, schedule
):
    """A refused `Thread.start()` must not escape the splash-path callback.

    Both schedulers run from a timer / deferred-startup callback during boot,
    where an exception is an unhandled error in a Textual timer task rather
    than something a caller can catch. Under thread exhaustion (or a start
    during interpreter shutdown) the right outcome is losing a speculative
    warm-up: every module involved is imported normally on first navigation
    anyway. Recording the handle only after a successful start also leaves the
    once-guard able to try again rather than latching on a thread that never
    ran.
    """
    monkeypatch.setenv("TLDW_SCREEN_PREIMPORT", "1")
    app = _arm(_build_test_app())
    monkeypatch.setattr(
        app,
        "_preimport_screens",
        lambda routes: pytest.fail("the thread never started"),
    )
    monkeypatch.setattr(
        app,
        "_preimport_heavy_screens",
        lambda: pytest.fail("the thread never started"),
    )

    def _refuse_to_start(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", _refuse_to_start)

    getattr(app, schedule)()  # must not raise

    assert app._initial_screen_preimport_thread is None
    assert app._screen_preimport_thread is None
