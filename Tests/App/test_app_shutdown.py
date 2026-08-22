"""Tests for bounded, graceful process termination (task-19561).

Two things are pinned here that the code they replaced could not do:

1. The exit watchdog is a real daemon thread that really fires, and it
   fires *after* the grace period, not instead of it. `_hard_exit` is the
   single seam that lets the test observe the decision without the test
   process being what dies.
2. The reason the three removed `thread.daemon = True` blocks were dead is
   a CPython rule, not a style opinion -- so it is asserted directly. If a
   future Python ever allowed it, this test tells us before someone
   reintroduces the pattern.
"""

from __future__ import annotations

import asyncio
import signal
import threading
import time

import pytest

from tldw_chatbook.Utils import app_shutdown

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_shutdown_state(monkeypatch):
    """A fresh module state per test, and a hard exit that never really exits."""
    monkeypatch.setattr(app_shutdown, "_STATE", app_shutdown._ShutdownState())
    exits: list[int] = []
    monkeypatch.setattr(app_shutdown, "_hard_exit", exits.append)
    # Stand in for the entry point: without this the watchdog refuses to arm.
    app_shutdown.claim_process_exit()
    yield exits
    app_shutdown.disarm_exit_watchdog()


# --- the CPython rule the removed code was built on top of -----------------


def test_daemon_flag_cannot_be_set_on_a_started_thread():
    """Why `thread.daemon = True` in a shutdown path was always a no-op."""
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait, args=(30,), name="probe")
    thread.start()
    try:
        assert thread.daemon is False
        with pytest.raises(RuntimeError, match="daemon"):
            thread.daemon = True
        assert thread.daemon is False, "the flag stayed unchanged, as documented"
    finally:
        stop.set()
        thread.join(timeout=5)


def test_watchdog_thread_is_a_daemon(_isolated_shutdown_state):
    """Daemon-at-construction: the watchdog must not itself delay exit."""
    assert app_shutdown.arm_exit_watchdog(60.0, reason="test") is True
    watchdog = [t for t in threading.enumerate() if t.name == "tldw-exit-watchdog"]
    assert watchdog, "watchdog thread was not started"
    assert all(t.daemon for t in watchdog)


# --- the watchdog ----------------------------------------------------------


def test_watchdog_refuses_to_arm_when_no_entry_point_owns_the_process(monkeypatch):
    """An app under `run_test()` must not be able to kill the pytest process.

    `on_unmount` arms the watchdog, and Textual's test harness unmounts a
    real `TldwCli` inside a runner that then keeps going for thousands more
    tests. This gate is the reason that does not end in `os._exit`.
    """
    monkeypatch.setattr(app_shutdown, "_STATE", app_shutdown._ShutdownState())
    exits: list[int] = []
    monkeypatch.setattr(app_shutdown, "_hard_exit", exits.append)

    assert app_shutdown.owns_process_exit() is False
    assert app_shutdown.arm_exit_watchdog(0.05, reason="unowned") is False
    assert not [t for t in threading.enumerate() if t.name == "tldw-exit-watchdog"]
    time.sleep(0.3)
    assert exits == []


def test_installing_the_handlers_claims_the_process(monkeypatch):
    monkeypatch.setattr(app_shutdown, "_STATE", app_shutdown._ShutdownState())
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        assert app_shutdown.owns_process_exit() is False
        app_shutdown.install_termination_handlers()
        assert app_shutdown.owns_process_exit() is True
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def test_watchdog_hard_exits_only_after_the_grace_period(_isolated_shutdown_state):
    exits = _isolated_shutdown_state
    app_shutdown.arm_exit_watchdog(0.25, reason="test", exit_code=7)
    assert exits == [], "hard exit must not be the first action"
    deadline = time.monotonic() + 5.0
    while not exits and time.monotonic() < deadline:
        time.sleep(0.01)
    assert exits == [7]


def test_watchdog_stands_down_when_disarmed(_isolated_shutdown_state):
    exits = _isolated_shutdown_state
    app_shutdown.arm_exit_watchdog(0.3, reason="test")
    app_shutdown.disarm_exit_watchdog()
    time.sleep(0.6)
    assert exits == []


def test_arming_again_never_relaxes_an_existing_bound(_isolated_shutdown_state):
    exits = _isolated_shutdown_state
    assert app_shutdown.arm_exit_watchdog(0.25, reason="first", exit_code=7) is True
    # A later deadline must be ignored outright.
    assert app_shutdown.arm_exit_watchdog(120.0, reason="later") is False
    deadline = time.monotonic() + 5.0
    while not exits and time.monotonic() < deadline:
        time.sleep(0.01)
    assert exits == [7], "the tighter, earlier bound is the one that held"


def test_arming_tighter_replaces_the_running_watchdog(_isolated_shutdown_state):
    exits = _isolated_shutdown_state
    assert app_shutdown.arm_exit_watchdog(30.0, reason="loose", exit_code=1) is True
    assert app_shutdown.arm_exit_watchdog(0.25, reason="tight", exit_code=9) is True
    deadline = time.monotonic() + 5.0
    while not exits and time.monotonic() < deadline:
        time.sleep(0.01)
    assert exits == [9]
    # The superseded watchdog must have been stood down, not merely ignored:
    # a thread still asleep on the longer deadline is a live timer nobody can
    # cancel any more.
    app_shutdown.disarm_exit_watchdog()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not [t for t in threading.enumerate() if t.name == "tldw-exit-watchdog"]:
            break
        time.sleep(0.01)
    assert not [t for t in threading.enumerate() if t.name == "tldw-exit-watchdog"]


def test_a_laxer_arm_cannot_slip_through_the_thread_start_window(
    _isolated_shutdown_state,
):
    """The monotonic rule must hold while a watchdog is being started.

    Independent review (task-19561): `arm()` published `self._watchdog` under
    the lock but called `thread.start()` outside it, and the guard used to
    read `is_alive()` -- False for a constructed-but-unstarted thread. A
    second arm landing in that window was therefore let through *even with a
    laxer deadline*, relaxing a bound that was already running. Every arm
    site runs on the main thread, and a signal handler re-enters at a
    bytecode boundary, so the window is reachable without extra threads;
    `Thread.start` is patched here only to make it wide enough to hit
    deterministically.
    """
    state = app_shutdown._STATE
    real_start = threading.Thread.start
    in_window = threading.Event()
    release = threading.Event()

    def stalling_start(self):
        if self.name == "tldw-exit-watchdog" and not in_window.is_set():
            in_window.set()
            release.wait(5.0)
        return real_start(self)

    # The same-thread form of the window: a signal handler re-entering
    # `arm()` while the first one is inside `Thread.start()`. An RLock does
    # not stop this, so the guard itself has to.
    assert state.arm(20.0, "first", 1) is True
    assert state.arm(20.0, "reentrant", 143) is False
    state.stand_down()

    lax_result: list[bool] = []
    threading.Thread.start = stalling_start
    try:
        tight = threading.Thread(target=lambda: state.arm(0.5, "tight", 7))
        tight.start()
        assert in_window.wait(5.0), "never reached the thread-start window"
        # A laxer arm attempted from inside the window. Before the fix it saw
        # `is_alive() == False` and was allowed through; now `arm()` is
        # atomic, so it blocks on the lock and is then correctly refused.
        lax = threading.Thread(
            target=lambda: lax_result.append(state.arm(30.0, "lax", 1))
        )
        lax.start()
        time.sleep(0.1)
        release.set()
        lax.join(5.0)
        tight.join(5.0)
        assert lax_result == [False], "a laxer arm replaced a running bound"
        remaining = state._watchdog_deadline - time.monotonic()
        assert remaining < 5.0, f"the 0.5s bound was relaxed to ~{remaining:.1f}s"
    finally:
        release.set()
        threading.Thread.start = real_start


def test_a_stood_down_watchdog_waking_at_its_deadline_does_not_hard_exit(
    _isolated_shutdown_state,
):
    """`stand_down()` racing the deadline must still win.

    `_watch` used to decide purely on `_watchdog_deadline > deadline`, which
    a disarm cannot satisfy: it nulls the deadline. A disarm landing between
    the timeout and the check therefore still ended in `os._exit`.
    """
    exits = _isolated_shutdown_state
    state = app_shutdown._STATE
    state.stand_down()
    state._watchdog_deadline = None
    # A watchdog waking up on an expired deadline with an unset event.
    state._watch(time.monotonic() - 1.0, "stale", 9, threading.Event())
    assert exits == []


def test_a_superseded_watchdog_never_fires_even_on_a_tighter_replacement(
    _isolated_shutdown_state,
):
    """Identity, not deadline arithmetic, is what retires a replaced thread."""
    exits = _isolated_shutdown_state
    state = app_shutdown._STATE
    other = threading.Thread(target=lambda: None, name="tldw-exit-watchdog")
    state._watchdog = other
    # A TIGHTER replacement: the old `current > deadline` test does not fire
    # for this, so only the identity check can retire us.
    state._watchdog_deadline = time.monotonic() - 5.0
    state._watch(time.monotonic() - 1.0, "superseded", 9, threading.Event())
    assert exits == []


# --- configuration ---------------------------------------------------------


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (45.0, 45.0),
        (0.0, app_shutdown._MIN_GRACE_SECONDS),
        (10_000.0, app_shutdown._MAX_GRACE_SECONDS),
        ("not a number", app_shutdown.DEFAULT_SHUTDOWN_GRACE_SECONDS),
        (float("inf"), app_shutdown.DEFAULT_SHUTDOWN_GRACE_SECONDS),
    ],
)
def test_grace_period_is_clamped_and_never_raises(monkeypatch, configured, expected):
    import tldw_chatbook.config as config_mod

    monkeypatch.setattr(
        config_mod, "get_cli_setting", lambda *a, **k: configured, raising=False
    )
    assert app_shutdown.graceful_shutdown_grace_seconds() == expected


def test_grace_period_falls_back_when_config_raises(monkeypatch):
    import tldw_chatbook.config as config_mod

    def _boom(*_a, **_k):
        raise RuntimeError("config is unreadable right now")

    monkeypatch.setattr(config_mod, "get_cli_setting", _boom, raising=False)
    assert (
        app_shutdown.graceful_shutdown_grace_seconds()
        == app_shutdown.DEFAULT_SHUTDOWN_GRACE_SECONDS
    )


# --- the grace period is enforced against healthy work ---------------------
#
# Independent review found a HIGH regression at a 20 s default: a clean quit
# (`app.exit()`, no signal) with one ordinary `run_worker(..., thread=True)`
# holding an open `BEGIN IMMEDIATE` died at 20.1 s with rc 1 and the
# transaction abandoned, where the merge base waited 28.8 s and committed. The
# cliff was exactly the grace period, and Textual thread workers cannot be
# interrupted, so this is any long ingest/export running when the user quits.
#
# These tests run the real shape at 1/20th scale so they finish in seconds.

#: Wall-clock divisor. The absolute numbers below are the real ones scaled by
#: this, so the test costs ~2 s instead of ~2 minutes while pinning the same
#: ordering.
_TIME_SCALE = 0.05

#: The default this fix replaced. Kept as a literal on purpose: the point of
#: the test is that going back to it re-breaks the case below.
_SUPERSEDED_DEFAULT_GRACE = 20.0

#: The reviewer's measured background job -- the one that sits between the old
#: default and the new one.
_LONG_JOB_SECONDS = 30.0


def _commit_after(db_path, seconds: float, committed: threading.Event) -> None:
    """A stand-in for a thread worker holding an open write transaction.

    Non-daemon and uninterruptible, exactly like the default-executor thread
    a `run_worker(..., thread=True)` job runs on: nothing shutdown can do
    will make this finish sooner.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS t (v TEXT)")
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO t (v) VALUES ('stmt1')")
        time.sleep(seconds)
        conn.execute("INSERT INTO t (v) VALUES ('stmt2')")
        conn.commit()
        committed.set()
    finally:
        conn.close()


def _rows(db_path) -> list[str]:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        return [r[0] for r in conn.execute("SELECT v FROM t")]
    except Exception:
        return []
    finally:
        conn.close()


def test_the_default_grace_outlives_a_long_background_job():
    """The constant itself, stated as the requirement it has to meet.

    This is the fast guard: dropping `DEFAULT_SHUTDOWN_GRACE_SECONDS` back to
    20 s fails here immediately, without waiting for the timing test below.
    """
    assert app_shutdown.DEFAULT_SHUTDOWN_GRACE_SECONDS > _LONG_JOB_SECONDS, (
        "a background job of this length must be able to finish its write "
        "during a clean quit; see the module docstring for the measurement"
    )
    assert (
        app_shutdown.DEFAULT_SHUTDOWN_GRACE_SECONDS <= app_shutdown._MAX_GRACE_SECONDS
    )


def test_a_long_job_commits_within_the_default_grace(
    tmp_path, _isolated_shutdown_state
):
    """At the default grace, a 30 s-class job still lands its transaction."""
    exits = _isolated_shutdown_state
    db_path = str(tmp_path / "job.db")
    committed = threading.Event()

    worker = threading.Thread(
        target=_commit_after,
        args=(db_path, _LONG_JOB_SECONDS * _TIME_SCALE, committed),
        name="probe-thread-worker",
    )
    worker.start()
    try:
        app_shutdown.arm_exit_watchdog(
            app_shutdown.DEFAULT_SHUTDOWN_GRACE_SECONDS * _TIME_SCALE,
            reason="clean quit",
        )
        assert committed.wait(30.0), "the job never finished"
        # The assertion that matters: the watchdog had not fired by the time
        # the write landed. Ordering, not wall clock -- a loaded machine
        # slows both sides.
        assert exits == [], "the watchdog killed a healthy job mid-transaction"
        assert _rows(db_path) == ["stmt1", "stmt2"]
    finally:
        worker.join(30.0)


def test_the_superseded_20s_default_would_have_abandoned_that_write(
    tmp_path, _isolated_shutdown_state
):
    """The red half: this is what a 20 s grace does to the same job.

    Kept as a live test rather than a comment so the regression stays
    visible -- if someone lowers the default back, the test above goes red
    and this one explains why.
    """
    exits = _isolated_shutdown_state
    db_path = str(tmp_path / "job.db")
    committed = threading.Event()

    worker = threading.Thread(
        target=_commit_after,
        args=(db_path, _LONG_JOB_SECONDS * _TIME_SCALE, committed),
        name="probe-thread-worker",
    )
    worker.start()
    try:
        app_shutdown.arm_exit_watchdog(
            _SUPERSEDED_DEFAULT_GRACE * _TIME_SCALE, reason="clean quit"
        )
        deadline = time.monotonic() + 30.0
        while not exits and time.monotonic() < deadline:
            time.sleep(0.01)
        assert exits, "the watchdog should have fired at the shorter grace"
        # The process would have died here. Nothing is committed yet, and the
        # worker cannot be interrupted to make it so.
        assert not committed.is_set()
        assert _rows(db_path) == [], "the transaction was still open"
    finally:
        worker.join(30.0)


# --- signal handling -------------------------------------------------------


class _FakeApp:
    def __init__(self) -> None:
        self.exited = 0

    def exit(self) -> None:
        self.exited += 1


def test_first_signal_asks_the_app_to_exit_instead_of_hard_exiting(
    _isolated_shutdown_state,
):
    exits = _isolated_shutdown_state

    async def _scenario():
        app = _FakeApp()
        app_shutdown.register_running_app(app)
        app_shutdown._handle_termination_signal(signal.SIGTERM, None)
        # Delivered through the loop, so it lands on the next tick.
        assert app.exited == 0
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return app

    app = asyncio.run(_scenario())
    assert app.exited == 1, "SIGTERM ran the ordinary shutdown path"
    assert exits == [], "no hard exit on the first signal"
    assert app_shutdown.termination_requested() is True


def test_a_signal_arms_a_bound_even_when_config_is_unreadable(
    _isolated_shutdown_state, monkeypatch
):
    """The handler must not depend on config to get *a* bound in place.

    Reading `[general] shutdown_grace_seconds` imports and locks the config
    module and can do file I/O; "the escape hatch does not exist yet" is a
    bad state to discover that from. The handler arms an unconditional
    backstop first.
    """
    import tldw_chatbook.config as config_mod

    def _boom(*_a, **_k):
        raise RuntimeError("config is unreadable right now")

    monkeypatch.setattr(config_mod, "get_cli_setting", _boom, raising=False)

    with pytest.raises(SystemExit):
        app_shutdown._handle_termination_signal(signal.SIGTERM, None)

    assert app_shutdown._STATE._watchdog_deadline is not None, "no bound was armed"
    assert [t for t in threading.enumerate() if t.name == "tldw-exit-watchdog"]


def test_a_signal_honours_a_configured_grace_larger_than_the_default(
    _isolated_shutdown_state, monkeypatch
):
    """The backstop must not silently clamp the user's own, larger value.

    This is why the backstop is `_MAX_GRACE_SECONDS` and not the default:
    arming the default first would refuse a configured 250s as "laxer" and
    bound the user at 120s -- the abandoned-write direction the grace period
    exists to avoid.
    """
    import tldw_chatbook.config as config_mod

    configured = 250.0
    assert configured > app_shutdown.DEFAULT_SHUTDOWN_GRACE_SECONDS
    monkeypatch.setattr(
        config_mod, "get_cli_setting", lambda *a, **k: configured, raising=False
    )

    before = time.monotonic()
    with pytest.raises(SystemExit):
        app_shutdown._handle_termination_signal(signal.SIGTERM, None)

    remaining = app_shutdown._STATE._watchdog_deadline - before
    assert remaining == pytest.approx(configured, abs=5.0), (
        f"configured {configured}s was not honoured; bound is ~{remaining:.0f}s"
    )


def test_second_signal_escalates_to_a_hard_exit(_isolated_shutdown_state):
    exits = _isolated_shutdown_state

    async def _scenario():
        app = _FakeApp()
        app_shutdown.register_running_app(app)
        app_shutdown._handle_termination_signal(signal.SIGTERM, None)
        app_shutdown._handle_termination_signal(signal.SIGTERM, None)

    asyncio.run(_scenario())
    assert exits == [143], "128 + SIGTERM"


def test_signal_without_a_running_app_unwinds_rather_than_amputating(
    _isolated_shutdown_state,
):
    exits = _isolated_shutdown_state
    with pytest.raises(SystemExit) as excinfo:
        app_shutdown._handle_termination_signal(signal.SIGTERM, None)
    assert excinfo.value.code == 143
    assert exits == [], "SystemExit unwinds through finally blocks; os._exit does not"


def test_unregistering_removes_only_the_matching_app(_isolated_shutdown_state):
    async def _scenario():
        first, second = _FakeApp(), _FakeApp()
        app_shutdown.register_running_app(first)
        app_shutdown.unregister_running_app(second)
        assert app_shutdown._STATE.app_and_loop()[0] is first
        app_shutdown.unregister_running_app(first)
        assert app_shutdown._STATE.app_and_loop()[0] is None

    asyncio.run(_scenario())


@pytest.mark.asyncio
async def test_mounting_the_real_app_under_test_arms_no_watchdog(monkeypatch):
    """The end-to-end version of the gate, through the real `on_unmount`.

    `TldwCli.on_unmount` calls `arm_exit_watchdog`. Under `run_test()` that
    must produce nothing: no watchdog thread, and no `_hard_exit` scheduled
    against the pytest process a few seconds later.
    """
    from tldw_chatbook.app import TldwCli

    monkeypatch.setattr(app_shutdown, "_STATE", app_shutdown._ShutdownState())
    exits: list[int] = []
    monkeypatch.setattr(app_shutdown, "_hard_exit", exits.append)

    app = TldwCli()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app_shutdown._STATE.app_and_loop()[0] is app

    assert app_shutdown.owns_process_exit() is False
    assert not [t for t in threading.enumerate() if t.name == "tldw-exit-watchdog"], (
        "an app under test must never arm the process-exit watchdog"
    )
    assert exits == []
    assert app_shutdown._STATE.app_and_loop()[0] is None, "unmount unregistered it"


def test_install_termination_handlers_is_idempotent(_isolated_shutdown_state):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        app_shutdown.install_termination_handlers()
        assert signal.getsignal(signal.SIGTERM) is (
            app_shutdown._handle_termination_signal
        )
        assert signal.getsignal(signal.SIGINT) is (
            app_shutdown._handle_termination_signal
        )
        # A second call must not reinstall (both entry points call it).
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        app_shutdown.install_termination_handlers()
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


# --- a failed installation must not disable the mechanism forever ----------
#
# Qodo review of PR #1972. `install_termination_handlers` used to call
# `claim_process_exit()` and latch `_handlers_installed` BEFORE attempting
# `signal.signal`, and it swallows installation errors. One failure therefore
# produced the worst available pair of outcomes at once: no signal handlers,
# ever (every later call short-circuited on the latch), plus a live watchdog
# that could still hard-exit the process, because the claim had gone through.


def _failing_signal(*_args, **_kwargs):
    raise ValueError("signal only works in main thread of the main interpreter")


def test_a_failed_install_claims_nothing_and_arms_nothing(monkeypatch):
    """The embedded case: no handlers, no claim, no watchdog. Inert."""
    monkeypatch.setattr(app_shutdown, "_STATE", app_shutdown._ShutdownState())
    exits: list[int] = []
    monkeypatch.setattr(app_shutdown, "_hard_exit", exits.append)
    monkeypatch.setattr(app_shutdown.signal, "signal", _failing_signal)

    app_shutdown.install_termination_handlers()

    assert app_shutdown.owns_process_exit() is False, (
        "an install that installed nothing must not claim the process's exit"
    )
    assert app_shutdown.arm_exit_watchdog(0.05, reason="unowned") is False
    assert not [t for t in threading.enumerate() if t.name == "tldw-exit-watchdog"]
    time.sleep(0.3)
    assert exits == [], "a watchdog was armed for handlers that do not exist"


def test_a_failed_install_stays_retryable(monkeypatch):
    """Failing once must not be what stops the second entry point working."""
    monkeypatch.setattr(app_shutdown, "_STATE", app_shutdown._ShutdownState())
    monkeypatch.setattr(app_shutdown, "_hard_exit", lambda code: None)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    real_signal = signal.signal

    monkeypatch.setattr(app_shutdown.signal, "signal", _failing_signal)
    try:
        app_shutdown.install_termination_handlers()
        assert app_shutdown._STATE.handlers_installed is False, (
            "a failed install latched, so no later call can ever succeed"
        )

        # Now from somewhere it *can* install -- the main thread, the real API.
        monkeypatch.setattr(app_shutdown.signal, "signal", real_signal)
        app_shutdown.install_termination_handlers()

        assert signal.getsignal(signal.SIGTERM) is (
            app_shutdown._handle_termination_signal
        )
        assert signal.getsignal(signal.SIGINT) is (
            app_shutdown._handle_termination_signal
        )
        assert app_shutdown.owns_process_exit() is True, (
            "a successful install owns the exit it is now able to request"
        )
        assert app_shutdown._STATE.handlers_installed is True
    finally:
        for sig, handler in previous.items():
            real_signal(sig, handler)


def test_a_partial_install_claims_the_exit_but_stays_retryable(monkeypatch):
    """One handler live is enough to need a bound, not enough to stop trying.

    A live SIGTERM handler asks the app to exit gracefully, and that ask has
    to be bounded -- so the claim is right. But SIGINT is still missing, so
    latching would make the gap permanent.
    """
    monkeypatch.setattr(app_shutdown, "_STATE", app_shutdown._ShutdownState())
    monkeypatch.setattr(app_shutdown, "_hard_exit", lambda code: None)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    real_signal = signal.signal

    def half_failing(sig, handler):
        if sig == signal.SIGINT:
            raise OSError("cannot install this one")
        return real_signal(sig, handler)

    monkeypatch.setattr(app_shutdown.signal, "signal", half_failing)
    try:
        app_shutdown.install_termination_handlers()

        assert signal.getsignal(signal.SIGTERM) is (
            app_shutdown._handle_termination_signal
        )
        assert app_shutdown.owns_process_exit() is True
        assert app_shutdown._STATE.handlers_installed is False, (
            "the signal that failed must still be retryable"
        )

        monkeypatch.setattr(app_shutdown.signal, "signal", real_signal)
        app_shutdown.install_termination_handlers()
        assert signal.getsignal(signal.SIGINT) is (
            app_shutdown._handle_termination_signal
        )
        assert app_shutdown._STATE.handlers_installed is True
    finally:
        for sig, handler in previous.items():
            real_signal(sig, handler)
        app_shutdown.disarm_exit_watchdog()
