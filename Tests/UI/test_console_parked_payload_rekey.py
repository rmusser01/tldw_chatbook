"""PR0: concurrent same-session interrupt rounds must not clobber each other.

Mirrors ``Tests/UI/test_skill_install_concurrent_confirms.py`` (TASK-910) but
targets the half that task left unfixed: the RETAINED PAYLOAD each bridge
re-derives its mounted card from is keyed by ``session_id``, not ``round_id``,
so arming a second round for the same session overwrites the first's payload.
The code names this itself in ``request_mcp_approvals``' ``finally`` block:
"per-round payload storage is a larger change out of scope here".
"""

from __future__ import annotations

import threading
import time

import pytest

from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class FakeApp:
    """``call_from_thread`` stand-in: invokes the callback immediately."""

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
    """A controller with a fake UI wired and one ACTIVE session.

    ``mounted`` records every payload the approval card was told to show,
    including the ``None`` clears, so a test can assert on what the user
    would actually be looking at after each transition.
    """
    store = ConsoleChatStore()
    ctrl = ConsoleChatController(store=store, provider_gateway=object())
    ctrl.app = FakeApp()
    ctrl.mounted = []
    ctrl.set_pending_approval = ctrl.mounted.append
    ctrl.mcp_approval_timeout_seconds = lambda: 30.0
    ctrl.session_a = store.create_session(title="A").id
    store.switch_session(ctrl.session_a)
    return ctrl


def _call(name):
    return MCPPendingCall(
        llm_name=name,
        server_key="agent:builtin",
        tool_name=name,
        server_label="builtin",
        arguments={},
        reason="ask",
    )


def _arm(ctrl, name, session_id, results, key):
    def worker():
        results[key] = ctrl.request_mcp_approvals(
            [_call(name)], session_id=session_id
        )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def _round_ids(ctrl):
    return list(ctrl._pending_approval_rounds)


def _mounted_round(ctrl):
    """The round id of the card currently shown, or None if cleared.

    The approvals payload names its id `round_id`; both skill bridges name
    theirs `request_id`. Every payload already carries one of the two, so
    no production payload needs a duplicate field for this helper.
    """
    payload = ctrl.mounted[-1] if ctrl.mounted else None
    if not payload:
        return None
    return payload.get("round_id") or payload.get("request_id")


def test_arming_a_second_same_session_round_does_not_evict_the_first_card(
    controller,
):
    """The head round keeps the card; a later sibling waits its turn."""
    results = {}
    first = _arm(controller, "alpha", controller.session_a, results, "alpha")
    assert _wait_until(lambda: len(_round_ids(controller)) == 1)
    round_1 = _round_ids(controller)[0]
    assert _wait_until(lambda: _mounted_round(controller) == round_1)

    second = _arm(controller, "beta", controller.session_a, results, "beta")
    assert _wait_until(lambda: len(_round_ids(controller)) == 2)
    time.sleep(0.1)  # let any errant mount land before asserting

    assert _mounted_round(controller) == round_1, (
        "arming a second same-session round must not evict the first's card"
    )

    for round_id in _round_ids(controller):
        controller.resolve_pending_approval({"alpha": "approve_once", "beta": "approve_once"}, round_id=round_id)
    first.join(timeout=5)
    second.join(timeout=5)


def test_the_queued_round_mounts_when_the_head_resolves(controller):
    """FIFO: resolving the head promotes the next same-session round."""
    results = {}
    first = _arm(controller, "alpha", controller.session_a, results, "alpha")
    assert _wait_until(lambda: len(_round_ids(controller)) == 1)
    round_1 = _round_ids(controller)[0]

    second = _arm(controller, "beta", controller.session_a, results, "beta")
    assert _wait_until(lambda: len(_round_ids(controller)) == 2)
    round_2 = [r for r in _round_ids(controller) if r != round_1][0]

    # Pre-condition: the head still owns the card. Without this, the test
    # cannot tell FIFO promotion from the arm-time clobber it exists to
    # catch -- round_2 would already be mounted and the post-assert would
    # pass for the wrong reason, on both sides of the fix.
    time.sleep(0.1)  # let any errant mount land before asserting
    assert _mounted_round(controller) == round_1

    controller.resolve_pending_approval({"alpha": "approve_once"}, round_id=round_1)
    first.join(timeout=5)

    assert _wait_until(lambda: _mounted_round(controller) == round_2), (
        "the queued round must mount once the head resolves"
    )

    controller.resolve_pending_approval({"beta": "approve_once"}, round_id=round_2)
    second.join(timeout=5)


def test_last_round_teardown_clears_the_card(controller):
    """With no rounds left for the session, the card clears."""
    results = {}
    only = _arm(controller, "alpha", controller.session_a, results, "alpha")
    assert _wait_until(lambda: len(_round_ids(controller)) == 1)
    round_1 = _round_ids(controller)[0]

    controller.resolve_pending_approval({"alpha": "approve_once"}, round_id=round_1)
    only.join(timeout=5)

    assert _wait_until(lambda: _mounted_round(controller) is None), (
        "the card must clear once the session has no armed rounds left"
    )


def _arm_install(ctrl, url, session_id, results, key):
    def worker():
        results[key] = ctrl.request_skill_install_confirm(url, session_id=session_id)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


@pytest.fixture
def install_controller():
    store = ConsoleChatStore()
    ctrl = ConsoleChatController(store=store, provider_gateway=object())
    ctrl.app = FakeApp()
    ctrl.mounted = []
    ctrl.set_pending_skill_install = ctrl.mounted.append
    ctrl.skill_install_confirm_timeout_seconds = lambda: 30.0
    ctrl.session_a = store.create_session(title="A").id
    store.switch_session(ctrl.session_a)
    return ctrl


def test_install_second_same_session_round_does_not_evict_the_first(
    install_controller,
):
    ctrl = install_controller
    results = {}
    first = _arm_install(ctrl, "https://x/one", ctrl.session_a, results, "one")
    assert _wait_until(lambda: len(ctrl.pending_skill_install_ids()) == 1)
    round_1 = ctrl.pending_skill_install_ids()[0]
    assert _wait_until(lambda: _mounted_round(ctrl) == round_1)

    second = _arm_install(ctrl, "https://x/two", ctrl.session_a, results, "two")
    assert _wait_until(lambda: len(ctrl.pending_skill_install_ids()) == 2)
    time.sleep(0.1)

    assert _mounted_round(ctrl) == round_1, (
        "a second same-session install confirm must not evict the first's card"
    )

    ctrl.resolve_pending_skill_install(True, request_id=round_1)
    first.join(timeout=5)
    round_2 = ctrl.pending_skill_install_ids()[0]
    assert _wait_until(lambda: _mounted_round(ctrl) == round_2)

    ctrl.resolve_pending_skill_install(True, request_id=round_2)
    second.join(timeout=5)
