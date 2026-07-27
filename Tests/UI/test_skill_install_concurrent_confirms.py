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

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class _FakeApp:
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
    ctrl.app = _FakeApp()
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
