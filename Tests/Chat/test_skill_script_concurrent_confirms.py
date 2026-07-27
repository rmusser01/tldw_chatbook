"""task-581: concurrent skill-script confirm rounds must not clobber each other.

The confirm bridge held a SINGLE pending slot, so if two rounds were ever armed
at once the second overwrote the first's event/decision and both worker threads
blocked until their 120s deadline. It fails closed and is not reachable while
the agent loop dispatches tool calls one at a time on one worker thread — but it
becomes reachable the moment anything runs tool calls concurrently, or a second
agent run overlaps the first.
"""

import threading
import time

import pytest

from .test_console_skill_script_confirm import make_controller as make_controller  # noqa: F401


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def controller(make_controller):
    """A controller with a fake UI already wired (see the sibling module)."""
    return make_controller()


def _arm(ctrl, skill_name, results, key):
    def worker():
        results[key] = ctrl.request_skill_script_confirm({"skill_name": skill_name})

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def test_two_concurrent_rounds_each_get_their_own_decision(controller):
    """AC#1: no cross-talk — each round resolves to what the user chose for it."""
    results = {}
    t1 = _arm(controller, "skill-one", results, "one")
    assert _wait_until(lambda: len(controller.pending_skill_script_ids()) == 1)
    id1 = controller.pending_skill_script_ids()[0]

    t2 = _arm(controller, "skill-two", results, "two")
    assert _wait_until(lambda: len(controller.pending_skill_script_ids()) == 2), (
        "arming a second round must not evict the first"
    )
    id2 = [i for i in controller.pending_skill_script_ids() if i != id1][0]

    controller.resolve_pending_skill_script(True, False, request_id=id2)
    t2.join(timeout=5)
    controller.resolve_pending_skill_script(False, False, request_id=id1)
    t1.join(timeout=5)

    assert results["two"] == {"allow": True, "remember": False}
    assert results["one"] == {"allow": False, "remember": False}


def test_a_decision_cannot_resolve_the_other_round(controller):
    """AC#2: the per-round request_id stays authoritative."""
    results = {}
    t1 = _arm(controller, "skill-one", results, "one")
    assert _wait_until(lambda: len(controller.pending_skill_script_ids()) == 1)

    t2 = _arm(controller, "skill-two", results, "two")
    assert _wait_until(lambda: len(controller.pending_skill_script_ids()) == 2)

    # Resolving with a bogus id must release neither round.
    controller.resolve_pending_skill_script(True, True, request_id="not-a-real-id")
    time.sleep(0.2)
    assert "one" not in results and "two" not in results

    for rid in list(controller.pending_skill_script_ids()):
        controller.resolve_pending_skill_script(False, False, request_id=rid)
    t1.join(timeout=5)
    t2.join(timeout=5)


def test_teardown_of_one_round_leaves_the_other_armed(controller):
    """AC#3: finishing round A must not clear round B's pending state."""
    results = {}
    t1 = _arm(controller, "skill-one", results, "one")
    assert _wait_until(lambda: len(controller.pending_skill_script_ids()) == 1)
    id1 = controller.pending_skill_script_ids()[0]

    t2 = _arm(controller, "skill-two", results, "two")
    assert _wait_until(lambda: len(controller.pending_skill_script_ids()) == 2)
    id2 = [i for i in controller.pending_skill_script_ids() if i != id1][0]

    controller.resolve_pending_skill_script(True, False, request_id=id1)
    t1.join(timeout=5)

    assert controller.pending_skill_script_ids() == [id2], (
        "round A's teardown must not evict round B"
    )
    controller.resolve_pending_skill_script(False, False, request_id=id2)
    t2.join(timeout=5)
    assert results["two"]["allow"] is False


def test_shutdown_denies_every_armed_round(controller):
    """TASK-910: `_deny_pending_skill_script_on_context_change` was removed
    (a plain context change/conversation switch no longer denies -- see
    ``Tests/Chat/test_console_skill_script_confirm.py``'s park tests).
    Real process teardown (`_shutdown_requested`) still releases every
    armed round at once, not just the newest."""
    results = {}
    t1 = _arm(controller, "skill-one", results, "one")
    assert _wait_until(lambda: len(controller.pending_skill_script_ids()) == 1)
    t2 = _arm(controller, "skill-two", results, "two")
    assert _wait_until(lambda: len(controller.pending_skill_script_ids()) == 2)

    controller._shutdown_requested.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert results["one"]["allow"] is False
    assert results["two"]["allow"] is False
    assert controller.pending_skill_script_ids() == []
