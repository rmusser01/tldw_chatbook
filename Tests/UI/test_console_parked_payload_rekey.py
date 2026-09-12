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
    ctrl._test_workers = []
    ctrl._test_worker_errors = []
    try:
        yield ctrl
    finally:
        _finish_controller_workers(ctrl)


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
        try:
            results[key] = ctrl.request_mcp_approvals(
                [_call(name)], session_id=session_id
            )
        except BaseException as exc:  # noqa: BLE001 -- fixture reports worker failures
            ctrl._test_worker_errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    ctrl._test_workers.append(thread)
    thread.start()
    return thread


def _finish_controller_workers(ctrl):
    """Cancel every fixture-owned round and prove its waiter has exited."""
    ctrl.begin_shutdown()
    for thread in ctrl._test_workers:
        thread.join(timeout=5.0)
    alive = [thread.name for thread in ctrl._test_workers if thread.is_alive()]
    assert not alive, f"controller fixture leaked worker threads: {alive}"
    assert not ctrl._test_worker_errors, (
        f"controller fixture workers raised: {ctrl._test_worker_errors!r}"
    )


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
        controller.resolve_pending_approval(
            {"alpha": "approve_once", "beta": "approve_once"}, round_id=round_id
        )
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
        try:
            results[key] = ctrl.request_skill_install_confirm(
                url, session_id=session_id
            )
        except BaseException as exc:  # noqa: BLE001 -- fixture reports worker failures
            ctrl._test_worker_errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    ctrl._test_workers.append(thread)
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
    ctrl._test_workers = []
    ctrl._test_worker_errors = []
    try:
        yield ctrl
    finally:
        _finish_controller_workers(ctrl)


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


def _arm_script(ctrl, script, session_id, results, key):
    def worker():
        try:
            results[key] = ctrl.request_skill_script_confirm(
                {"skill": "demo", "script": script}, session_id=session_id
            )
        except BaseException as exc:  # noqa: BLE001 -- fixture reports worker failures
            ctrl._test_worker_errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    ctrl._test_workers.append(thread)
    thread.start()
    return thread


@pytest.fixture
def script_controller():
    store = ConsoleChatStore()
    ctrl = ConsoleChatController(store=store, provider_gateway=object())
    ctrl.app = FakeApp()
    ctrl.mounted = []
    ctrl.set_pending_skill_script = ctrl.mounted.append
    ctrl.skill_script_confirm_timeout_seconds = lambda: 30.0
    ctrl.session_a = store.create_session(title="A").id
    store.switch_session(ctrl.session_a)
    ctrl._test_workers = []
    ctrl._test_worker_errors = []
    try:
        yield ctrl
    finally:
        _finish_controller_workers(ctrl)


def test_script_second_same_session_round_does_not_evict_the_first(
    script_controller,
):
    ctrl = script_controller
    results = {}
    first = _arm_script(ctrl, "echo one", ctrl.session_a, results, "one")
    assert _wait_until(lambda: len(ctrl.pending_skill_script_ids()) == 1)
    round_1 = ctrl.pending_skill_script_ids()[0]
    assert _wait_until(lambda: _mounted_round(ctrl) == round_1)

    second = _arm_script(ctrl, "echo two", ctrl.session_a, results, "two")
    assert _wait_until(lambda: len(ctrl.pending_skill_script_ids()) == 2)
    time.sleep(0.1)

    assert _mounted_round(ctrl) == round_1, (
        "a second same-session script confirm must not evict the first's card"
    )

    # resolve_pending_skill_script takes (allow, remember, request_id).
    ctrl.resolve_pending_skill_script(True, False, round_1)
    first.join(timeout=5)

    round_2 = ctrl.pending_skill_script_ids()[0]
    assert _wait_until(lambda: _mounted_round(ctrl) == round_2), (
        "the queued script confirm must mount once the head resolves"
    )

    ctrl.resolve_pending_skill_script(True, False, round_2)
    second.join(timeout=5)


def test_bridges_do_not_share_a_head(controller):
    """Each bridge keeps its own FIFO head for the same session.

    An approval round and a skill-install round armed for one session are
    independent surfaces -- neither may evict or promote the other.
    """
    ctrl = controller
    approvals_mounted = ctrl.mounted
    install_mounted = []
    ctrl.set_pending_skill_install = install_mounted.append
    ctrl.skill_install_confirm_timeout_seconds = lambda: 30.0
    # Exercise the host's documented setter-based, per-kind projection seam.
    ctrl._publish_pending_decision = None

    results = {}
    approval = _arm(ctrl, "alpha", ctrl.session_a, results, "alpha")
    assert _wait_until(lambda: len(_round_ids(ctrl)) == 1)
    approval_round = _round_ids(ctrl)[0]
    assert _wait_until(lambda: _mounted_round(ctrl) == approval_round)

    install = _arm_install(ctrl, "https://x/one", ctrl.session_a, results, "one")
    assert _wait_until(lambda: len(ctrl.pending_skill_install_ids()) == 1)
    install_round = ctrl.pending_skill_install_ids()[0]
    assert _wait_until(lambda: install_round in ctrl._parked_skill_install_payloads)

    # The approvals payload names its id `round_id`; the install payload
    # names its own `request_id`. Neither bridge renames the other's.
    assert _wait_until(
        lambda: (
            bool(install_mounted)
            and install_mounted[-1] is not None
            and install_mounted[-1].get("request_id") == install_round
        )
    )
    ctrl.resolve_pending_approval({"alpha": "approve_once"}, round_id=approval_round)
    approval.join(timeout=5)

    assert _wait_until(lambda: approvals_mounted[-1] is None)
    assert install_mounted[-1] is not None, (
        "resolving an approval round must not clear the install card"
    )

    ctrl.resolve_pending_skill_install(True, request_id=install_round)
    install.join(timeout=5)


def test_queued_round_spends_timeout_only_while_answerable(controller):
    """Queued and hidden time preserve a round's answerable-time budget."""
    now = [100.0]
    controller.decision_monotonic_clock = lambda: now[0]
    results = {}
    first = _arm(controller, "alpha", controller.session_a, results, "alpha")
    assert _wait_until(lambda: len(_round_ids(controller)) == 1)
    round_1 = _round_ids(controller)[0]
    assert _wait_until(lambda: _mounted_round(controller) == round_1)
    assert controller.set_answerable_decision(controller.session_a, round_1)

    second = _arm(controller, "beta", controller.session_a, results, "beta")
    assert _wait_until(lambda: len(_round_ids(controller)) == 2)
    round_2 = next(r for r in _round_ids(controller) if r != round_1)
    assert _wait_until(lambda: round_2 in controller._parked_approval_payloads)
    assert (
        controller.pending_decision_projection(controller.session_a).decision_id
        == round_1
    )

    now[0] += 10.0
    with controller._approval_state_lock:
        queued_state = controller._pending_approval_rounds[round_2]
        assert queued_state["remaining_active_seconds"] == 30.0
        assert queued_state["active_since"] is None

    controller.resolve_pending_approval({"alpha": "approve_once"}, round_id=round_1)
    first.join(timeout=5)
    assert _wait_until(lambda: _mounted_round(controller) == round_2)
    promoted = controller.pending_decision_projection(controller.session_a)
    assert promoted is not None
    assert promoted.decision_id == round_2
    assert promoted.remaining_active_seconds == 30.0
    assert promoted.payload["timeout_seconds"] == 30.0

    assert controller.set_answerable_decision(controller.session_a, round_2)
    now[0] += 7.0
    answerable = controller.pending_decision_projection(controller.session_a)
    assert answerable is not None
    assert answerable.remaining_active_seconds == 23.0
    assert answerable.payload["timeout_seconds"] == 23.0

    controller.on_console_view_visibility_changed(False)
    assert controller.set_answerable_decision(controller.session_a, None)
    now[0] += 11.0
    hidden = controller.pending_decision_projection(controller.session_a)
    assert hidden is not None
    assert hidden.remaining_active_seconds == 23.0
    assert hidden.payload["timeout_seconds"] == 23.0
    with controller._approval_state_lock:
        stored = controller._parked_approval_payloads[round_2]
    assert stored["timeout_seconds"] == 30.0, (
        "projecting remaining answerable time must not rewrite the retained payload"
    )

    controller.resolve_pending_approval({"beta": "approve_once"}, round_id=round_2)
    second.join(timeout=5)


def test_legacy_round_teardown_clears_its_card_after_a_session_switch(controller):
    """Qodo PR #1836 finding 2: a legacy ``session_id=None`` round mounts
    unconditionally, but its teardown re-derived for the session that was
    active AT ARM. If the active session changed in between, the re-derive
    guard rejected the stale id and the legacy card stayed mounted
    indefinitely -- where the deleted pre-PR0 guard cleared it
    unconditionally. Teardown of a legacy round must re-derive for
    whichever session is active WHEN THE CALLBACK RUNS.
    """
    results = {}
    legacy = _arm(controller, "legacy", None, results, "legacy")
    assert _wait_until(lambda: len(_round_ids(controller)) == 1)
    round_1 = _round_ids(controller)[0]
    assert _wait_until(lambda: _mounted_round(controller) == round_1)

    # Direct store switch: the end-state of the documented race, where the
    # legacy mount's call_from_thread lands AFTER the switch already ran
    # its own re-derive -- active session B, legacy card still on screen.
    session_b = controller.store.create_session(title="B").id
    controller.store.switch_session(session_b)

    controller.resolve_pending_approval({"legacy": "approve_once"}, round_id=round_1)
    legacy.join(timeout=5)

    assert _wait_until(lambda: controller.mounted[-1] is None), (
        "a legacy round's teardown must clear its card even after a "
        "session switch -- the pre-PR0 unconditional clear's job"
    )
