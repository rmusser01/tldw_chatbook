"""TASK-910 fix round 1: concurrent skill-install confirm rounds must not
clobber each other.

Mirrors ``Tests/Chat/test_skill_script_concurrent_confirms.py`` (task-581's
identical fix for skill-script), but for the INSTALL bridge: TASK-910
converted ``request_skill_install_confirm`` from a single global
``_pending_skill_install_event``/``_pending_skill_install_decision`` pair to
the per-round registry ``_pending_skill_install_rounds`` (keyed by a fresh
``request_id``) specifically so two DIFFERENT sessions can each raise their
own install confirm concurrently without clobbering each other -- exactly
the scenario TASK-910's parking makes newly reachable (a background
session's install confirm no longer denies-on-switch; it can now sit
alongside the viewed session's own pending confirm). Pre-fix, arming a
second round while a first was still pending would have overwritten the
single slot's event/decision, leaving BOTH worker threads blocked until
their full timeout with no way to resolve either correctly.
"""

from __future__ import annotations

import threading
import time

import pytest

from tldw_chatbook.Chat.console_chat_controller import (
    _MCP_APPROVAL_POLL_SECONDS,
    ConsoleChatController,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleRunMarker
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class FakeApp:
    """`call_from_thread` stand-in: invokes the callback immediately."""

    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def controller():
    """A controller with a fake UI wired and two DISTINCT sessions ready.

    ``session_a`` stays the ACTIVE/viewed session throughout (its own round
    mounts immediately); ``session_b`` is a background session (its round
    parks) -- the exact mounted-plus-parked mix TASK-910's parking design
    must keep independent.
    """
    store = ConsoleChatStore()
    ctrl = ConsoleChatController(store=store, provider_gateway=object())
    ctrl.app = FakeApp()
    ctrl.set_pending_skill_install = lambda payload: None
    ctrl.skill_install_confirm_timeout_seconds = lambda: 30.0
    ctrl.session_a = store.create_session(title="A").id
    ctrl.session_b = store.create_session(title="B").id
    store.switch_session(ctrl.session_a)  # A stays viewed; B's round parks
    return ctrl


def _arm(ctrl, url, session_id, results, key):
    def worker():
        results[key] = ctrl.request_skill_install_confirm(url, session_id=session_id)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def test_two_concurrent_rounds_for_different_sessions_each_get_their_own_decision(
    controller,
):
    """AC#1: no cross-talk -- two DIFFERENT sessions' rounds each resolve to
    what the user chose for THAT round."""
    results = {}
    t1 = _arm(controller, "https://x/one", controller.session_a, results, "one")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 1)
    id1 = controller.pending_skill_install_ids()[0]

    t2 = _arm(controller, "https://x/two", controller.session_b, results, "two")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 2), (
        "arming a second round (a different session) must not evict the first"
    )
    id2 = [i for i in controller.pending_skill_install_ids() if i != id1][0]

    controller.resolve_pending_skill_install(True, request_id=id2)
    t2.join(timeout=5)
    controller.resolve_pending_skill_install(False, request_id=id1)
    t1.join(timeout=5)

    assert results["two"] is True
    assert results["one"] is False


def test_a_decision_cannot_resolve_the_other_round(controller):
    """AC#2: the per-round request_id stays authoritative -- a bogus id
    releases neither round while both are live."""
    results = {}
    t1 = _arm(controller, "https://x/one", controller.session_a, results, "one")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 1)

    t2 = _arm(controller, "https://x/two", controller.session_b, results, "two")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 2)

    # Resolving with a bogus id must release neither round.
    controller.resolve_pending_skill_install(True, request_id="not-a-real-id")
    time.sleep(0.2)
    assert "one" not in results and "two" not in results
    assert len(controller.pending_skill_install_ids()) == 2, (
        "a stale/unknown id must leave both rounds live"
    )

    for rid in list(controller.pending_skill_install_ids()):
        controller.resolve_pending_skill_install(True, request_id=rid)
    t1.join(timeout=5)
    t2.join(timeout=5)


def test_teardown_of_one_round_leaves_the_other_armed(controller):
    """AC#3: finishing round A must not clear round B's pending state (nor
    B's own fleet needs-approval badge)."""
    results = {}
    t1 = _arm(controller, "https://x/one", controller.session_a, results, "one")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 1)
    id1 = controller.pending_skill_install_ids()[0]

    t2 = _arm(controller, "https://x/two", controller.session_b, results, "two")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 2)
    id2 = [i for i in controller.pending_skill_install_ids() if i != id1][0]

    controller.resolve_pending_skill_install(True, request_id=id1)
    t1.join(timeout=5)

    assert controller.pending_skill_install_ids() == [id2], (
        "round A's teardown must not evict round B"
    )
    # B's badge (parked, background) must still be up -- A's teardown only
    # ever targets A's own session state.
    assert controller.session_b in controller._pending_approvals
    controller.resolve_pending_skill_install(False, request_id=id2)
    t2.join(timeout=5)
    assert results["two"] is False


def test_two_rounds_for_the_same_session_keep_badge_and_payload_until_both_resolve(
    controller,
):
    """TASK-1050 (Defect A/B): two install-confirm rounds for the SAME
    session (unlike every test above, which uses two DIFFERENT sessions)
    -- `_parked_skill_install_payloads` is keyed by session id alone, so
    arming the second round overwrites the first's retained payload under
    that key. Resolving the EARLIER round first must not clear the badge
    (a sibling round is still outstanding) nor discard the NEWER round's
    still-armed payload; only resolving the LAST one does either."""
    results = {}
    t1 = _arm(controller, "https://x/one", controller.session_a, results, "one")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 1)
    id1 = controller.pending_skill_install_ids()[0]
    assert (
        controller.run_marker_for(controller.session_a)
        is ConsoleRunMarker.NEEDS_APPROVAL
    )

    t2 = _arm(controller, "https://x/two", controller.session_a, results, "two")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 2)
    id2 = [i for i in controller.pending_skill_install_ids() if i != id1][0]
    # Round 2 overwrote round 1's stored payload under the same session key.
    assert (
        controller._parked_skill_install_payloads[controller.session_a]["request_id"]
        == id2
    )

    # Round 1 (the EARLIER round) resolves first -- must not evict round
    # 2's still-armed payload nor clear the badge.
    controller.resolve_pending_skill_install(False, request_id=id1)
    t1.join(timeout=5)
    assert results["one"] is False
    assert (
        controller.run_marker_for(controller.session_a)
        is ConsoleRunMarker.NEEDS_APPROVAL
    )
    assert controller.session_a in controller._pending_approvals
    assert (
        controller._parked_skill_install_payloads[controller.session_a]["request_id"]
        == id2
    )

    # Round 2 (the LAST remaining round) resolves -- now everything clears.
    controller.resolve_pending_skill_install(True, request_id=id2)
    t2.join(timeout=5)
    assert results["two"] is True
    assert controller.run_marker_for(controller.session_a) is ConsoleRunMarker.NONE
    assert controller.session_a not in controller._pending_approvals
    assert controller.session_a not in controller._parked_skill_install_payloads


def test_two_rounds_for_the_same_session_resolving_the_newer_one_first_leaves_the_slot_populated(
    controller,
):
    """TASK-1050 fix round 1 (review): reverse-ordering counterpart to the
    sibling test above (mirrors `test_console_mcp_approval.py`'s identical
    MCP-bridge test). `_parked_skill_install_payloads` is a SINGLE
    per-session slot holding whichever round's payload was LAST WRITTEN,
    so resolving the NEWER (newest-armed) round FIRST -- the natural live
    ordering, since arming a round re-mounts its card, which typically
    gets decided before an already-waiting sibling does -- must not pop
    the slot while the OLDER round is still outstanding: the badge must
    stay up and the slot must still hold a payload (remount still works,
    even though it is round 2's own now-stale payload rather than round
    1's -- the accepted single-slot scope). Only resolving the older,
    now-last round clears both."""
    results = {}
    t1 = _arm(controller, "https://x/one", controller.session_a, results, "one")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 1)
    id1 = controller.pending_skill_install_ids()[0]

    t2 = _arm(controller, "https://x/two", controller.session_a, results, "two")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 2)
    id2 = [i for i in controller.pending_skill_install_ids() if i != id1][0]

    # Round 2 (the NEWER round) resolves FIRST -- round 1 is still
    # outstanding, so the badge must stay up and the slot must still hold
    # a payload.
    controller.resolve_pending_skill_install(True, request_id=id2)
    t2.join(timeout=5)
    assert results["two"] is True
    assert (
        controller.run_marker_for(controller.session_a)
        is ConsoleRunMarker.NEEDS_APPROVAL
    )
    assert controller.session_a in controller._pending_approvals
    assert controller.session_a in controller._parked_skill_install_payloads, (
        "the parked slot must still hold a payload -- popping it here "
        "would strand the still-armed older round unresolvable on the "
        "next switch-away/back"
    )

    # Round 1 (the OLDER round, now the LAST one armed) resolves -- only
    # now do both the badge and the parked slot clear.
    controller.resolve_pending_skill_install(False, request_id=id1)
    t1.join(timeout=5)
    assert results["one"] is False
    assert controller.run_marker_for(controller.session_a) is ConsoleRunMarker.NONE
    assert controller.session_a not in controller._pending_approvals
    assert controller.session_a not in controller._parked_skill_install_payloads


def test_stale_request_id_with_both_rounds_live_resolves_neither(controller):
    """Security-critical, mirrors `resolve_pending_skill_script`'s stale-id
    hazard: a resolve carrying a PRIOR/unrelated round's id must not
    authorize either round while both are still armed."""
    results = {}
    t1 = _arm(controller, "https://x/one", controller.session_a, results, "one")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 1)
    id1 = controller.pending_skill_install_ids()[0]

    t2 = _arm(controller, "https://x/two", controller.session_b, results, "two")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 2)

    controller.resolve_pending_skill_install(True, request_id="totally-unknown")
    time.sleep(0.2)
    assert t1.is_alive() and t2.is_alive(), (
        "an unknown request_id must not resolve either still-armed round"
    )
    assert len(controller.pending_skill_install_ids()) == 2

    controller.resolve_pending_skill_install(True, request_id=id1)
    t1.join(timeout=5)
    for rid in list(controller.pending_skill_install_ids()):
        controller.resolve_pending_skill_install(False, request_id=rid)
    t2.join(timeout=5)
    assert results["one"] is True
    assert results["two"] is False


def test_shutdown_denies_every_armed_round_with_real_session_ids(controller):
    """Fix round 2 (review): mirrors `test_skill_script_concurrent_confirms
    .test_shutdown_denies_every_armed_round`, but for install and with two
    DIFFERENT REAL session ids throughout (never `session_id=None`).

    `_is_session_cancelled` checks ONLY the round's own
    `_active_cancel_events` entry when `session_id is not None` -- it never
    reads the bare `_shutdown_requested` flag in that branch (see that
    method's own docstring: "when session_id is known, check ONLY that
    session's own _active_cancel_events entry -- never the shared flag").
    `shutdown()`'s actual body (see its docstring/source) reaches a
    real-session round by calling `_signal_stop(session_id=...)` for every
    session it finds in `_active_stream_tasks` -- reproduced directly here
    (rather than driving the full async `shutdown()` coroutine, which also
    cancels/awaits real `asyncio.Task` objects unrelated to this check)
    since that per-session fanout is the ONLY mechanism a real-session
    confirm bridge ever observes teardown through. Pre-populates each
    session's `_active_cancel_events` entry first, mirroring what
    `_run_agent_reply` already does at a real run's start (see
    `test_own_session_cancel_event_denies_the_round` in
    `test_console_skill_install_confirm.py` for the identical single-round
    setup) -- `_signal_stop` itself is a no-op for a session with no
    registered event yet.
    """
    controller._active_cancel_events[controller.session_a] = threading.Event()
    controller._active_cancel_events[controller.session_b] = threading.Event()

    results = {}
    t1 = _arm(controller, "https://x/one", controller.session_a, results, "one")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 1)
    t2 = _arm(controller, "https://x/two", controller.session_b, results, "two")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 2)

    # Mirrors `shutdown()`'s own body: the global flag first, then the
    # per-session cancel-event fanout for every live session.
    controller._shutdown_requested.set()
    controller._signal_stop(session_id=controller.session_a)
    controller._signal_stop(session_id=controller.session_b)

    t1.join(timeout=5)
    t2.join(timeout=5)
    assert results["one"] is False
    assert results["two"] is False
    assert controller.pending_skill_install_ids() == []


def test_bare_shutdown_flag_alone_denies_a_real_session_round_within_one_poll_interval(
    controller,
):
    """TASK-1052 (was Fix round 2's `test_bare_shutdown_flag_alone_does_not_
    deny_a_real_session_round`, pinning the GAP as evidence -- see git
    history for that version and its own docstring for the pre-fix
    reasoning). Contract, now closed: the bare `_shutdown_requested` flag
    alone -- WITHOUT the per-session `_signal_stop` fanout `shutdown()`
    normally performs -- DOES deny a round armed with a real `session_id`,
    within one `_MCP_APPROVAL_POLL_SECONDS` poll interval.

    `shutdown()`'s per-session `_signal_stop` fanout only reaches sessions
    already present in its `_active_stream_tasks` snapshot at the moment it
    runs (the "shutdown-snapshot race": TASK-1052) -- a round armed for a
    session before that session is registered there was previously
    invisible to that fanout and fell back to its own confirm/approval
    timeout (up to ~120s in production) to fail closed. Since real process
    teardown is global by definition, `_is_session_cancelled`'s real-
    `session_id` branch now ALSO ORs in `_shutdown_requested` directly (in
    addition to that session's own `_active_cancel_events` entry), so every
    armed round observes teardown within one poll interval regardless of
    registration timing -- not just the `session_id=None` legacy fallback,
    which already read the flag (see the sibling `test_shutdown_denies_
    every_armed_round_with_real_session_ids` above, which drives the real
    per-session fanout; this test isolates the bare-flag-alone case that
    fanout doesn't reach).

    Uses a timeout far longer than the poll interval so an early
    resolution can only be attributed to the shutdown signal, never the
    round's own deadline -- still fails CLOSED either way, never open.
    """
    controller.skill_install_confirm_timeout_seconds = lambda: 30.0
    results = {}
    t1 = _arm(controller, "https://x/one", controller.session_a, results, "one")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 1)

    controller._shutdown_requested.set()  # global flag only -- no per-session fanout
    t1.join(timeout=_MCP_APPROVAL_POLL_SECONDS + 3.0)
    assert not t1.is_alive(), (
        "a real-session round with no matching _active_cancel_events entry "
        "did not observe the bare _shutdown_requested flag within one poll "
        "interval -- if this now fails, _is_session_cancelled's "
        "real-session branch stopped ORing in the shared flag; that is the "
        "TASK-1052 regression this test guards against"
    )
    assert results["one"] is False  # still fails closed, never auto-approved
    assert controller.pending_skill_install_ids() == []  # round's own accounting cleaned up


def test_shutdown_flag_alone_denies_both_unregistered_sessions_rounds_and_cleans_accounting(
    controller,
):
    """TASK-1052 x TASK-1050 interplay: two rounds armed for DIFFERENT
    sessions, NEITHER registered in `_active_cancel_events` (so
    `shutdown()`'s own per-session `_signal_stop` fanout could never reach
    either one even if driven for real -- see `test_shutdown_denies_every_
    armed_round_with_real_session_ids` above, which pre-registers both
    sessions' cancel events specifically so that fanout DOES reach them;
    this test deliberately skips that setup). Setting the bare
    `_shutdown_requested` flag alone must still deny BOTH rounds within one
    poll interval, and each round's own `finally` -- TASK-1050's
    round-keyed `discard_pending_round` plus its guarded
    `_parked_skill_install_payloads` pop -- must still run cleanly for
    both: no stale pending-approval accounting or retained payload left
    behind for either session, and no cross-talk between the two (each
    session's own entries clear independently).
    """
    results = {}
    t1 = _arm(controller, "https://x/one", controller.session_a, results, "one")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 1)
    t2 = _arm(controller, "https://x/two", controller.session_b, results, "two")
    assert _wait_until(lambda: len(controller.pending_skill_install_ids()) == 2)

    assert controller.session_a not in controller._active_cancel_events
    assert controller.session_b not in controller._active_cancel_events

    controller._shutdown_requested.set()  # global flag only -- no per-session fanout
    t1.join(timeout=_MCP_APPROVAL_POLL_SECONDS + 3.0)
    t2.join(timeout=_MCP_APPROVAL_POLL_SECONDS + 3.0)

    assert not t1.is_alive() and not t2.is_alive(), (
        "both real-session rounds, neither registered in "
        "_active_cancel_events, must observe the bare _shutdown_requested "
        "flag within one poll interval"
    )
    assert results["one"] is False
    assert results["two"] is False
    assert controller.pending_skill_install_ids() == []
    assert controller.session_a not in controller._pending_approvals
    assert controller.session_b not in controller._pending_approvals
    assert controller.session_a not in controller._parked_skill_install_payloads
    assert controller.session_b not in controller._parked_skill_install_payloads
