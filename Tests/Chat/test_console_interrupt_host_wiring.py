"""task-31384: the controller's legacy round attributes are aliases of the host.

Eleven test files and ``ChatScreen._current_park_round_ids`` read the
historical per-kind names; the host must be the single owner behind them.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_interrupt_rounds import KIND_SETTER_ATTRS


def test_every_kind_has_a_setter_attribute_on_the_controller():
    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    for attr in KIND_SETTER_ATTRS.values():
        assert hasattr(controller, attr), attr


def test_legacy_registry_payload_and_lock_names_alias_the_host():
    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    host = controller._interrupt_host
    for kind, registry, payloads in [
        ("approval", "_pending_approval_rounds", "_parked_approval_payloads"),
        ("skill_install", "_pending_skill_install_rounds", "_parked_skill_install_payloads"),
        ("skill_script", "_pending_skill_script_rounds", "_parked_skill_script_payloads"),
        ("worktree_merge", "_pending_worktree_merge_rounds", "_parked_worktree_merge_payloads"),
        ("question", "_pending_question_rounds", "_parked_question_payloads"),
    ]:
        assert getattr(controller, registry) is host.registries[kind], registry
        assert getattr(controller, payloads) is host.payloads[kind], payloads
    for lock in (
        "_approval_state_lock",
        "_pending_skill_install_lock",
        "_pending_skill_script_lock",
        "_pending_worktree_merge_lock",
        "_pending_question_lock",
    ):
        assert getattr(controller, lock) is host.lock, lock


def test_store_based_helpers_work_over_any_dict_under_the_one_lock():
    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    payload = {"round_id": "r1", "session_id": "s1"}
    assert controller._park_round_payload(controller._parked_question_payloads, "r1", payload) is True
    assert controller._head_round_payload(controller._parked_question_payloads, "s1") is payload
    assert controller._head_round_payload(controller._parked_approval_payloads, "s1") is None
    controller._unpark_round_payload(controller._parked_question_payloads, "r1")
    assert controller._head_round_payload(controller._parked_question_payloads, "s1") is None
    # A dict the host does not own (test doubles and the skill bridges'
    # own remount calls pass one) is handled the same way.
    foreign: dict = {}
    assert controller._park_round_payload(foreign, "x", {"round_id": "x", "session_id": "s9"}) is True
    assert controller._session_round_payloads(foreign, "s9")[0]["round_id"] == "x"


def test_approvals_register_the_permission_summary_as_the_after_remount_hook():
    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    hook = controller._interrupt_host.after_remount["approval"]
    assert hook.__func__ is ConsoleChatController._maybe_fire_permission_summary
    assert set(controller._interrupt_host.after_remount) == {"approval"}


@pytest.mark.parametrize("kind", ["approval", "skill_install", "skill_script"])
def test_host_orders_custodied_decisions_without_reentering_its_lock(kind):
    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    session = controller.store.ensure_session()
    controller.app = SimpleNamespace(call_from_thread=lambda fn, *args: fn(*args))
    host = controller._interrupt_host
    state = {"event": threading.Event(), "session_id": session.id}
    state["event"].set()
    payload = {"session_id": session.id, "timeout_seconds": 5.0}
    observed = []

    def observe():
        projection = controller.pending_decision_projection(session.id)
        observed.append(projection)
        assert controller.set_answerable_decision(session.id, "exact-round")
        assert controller.set_answerable_decision(session.id, None)

    errors = []

    def run():
        try:
            host.run_round(
                kind, "exact-round", payload, state,
                session_id=session.id, owning_session_id=session.id,
                deadline=None, is_parked=False, before_wait=observe,
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(2)
    assert not worker.is_alive(), "decision mutation reentered the host lock"
    assert errors == []
    assert observed[0].decision_id == "exact-round"
    assert host.registries[kind] == {}
    assert host.payloads[kind] == {}


def test_finishing_approval_remains_projected_after_its_waiter_exits():
    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    session = controller.store.ensure_session()
    controller._parked_approval_payloads["finished-wait"] = {
        "session_id": session.id, "phase": "finishing", "calls": [],
        "_decision_type": "approval", "_decision_id": "finished-wait",
        "_decision_order": 1,
    }
    projection = controller.pending_decision_projection(session.id)
    assert projection is not None
    assert projection.payload["phase"] == "finishing"
    assert not controller.set_answerable_decision(session.id, "finished-wait")


@pytest.mark.parametrize("kind", ["skill_install", "skill_script"])
def test_hidden_skill_notice_uses_one_host_lock_and_retries_failed_delivery(kind):
    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    notices = []

    def notify(message, **_kwargs):
        notices.append(message)
        if len(notices) == 1:
            raise RuntimeError("notification unavailable")

    controller.app = SimpleNamespace(notify=notify)
    controller._interrupt_host.registries[kind]["exact"] = {
        "session_id": "owner",
        "decision_type": kind,
        "decision_id": "exact",
    }

    def announce():
        controller._announce_hidden_decision(kind, "wrong-owner", "exact")
        controller._announce_hidden_decision(kind, "owner", "exact")
        controller._announce_hidden_decision(kind, "owner", "exact")
        controller._announce_hidden_decision(kind, "owner", "exact")

    worker = threading.Thread(target=announce, daemon=True)
    worker.start()
    worker.join(2)
    assert not worker.is_alive(), "hidden notice reentered the shared host lock"
    assert len(notices) == 2
    assert controller._announced_pending_decision_ids == {"exact"}
    controller._forget_hidden_decision("exact")
    controller._interrupt_host.registries[kind]["exact"]["settled"] = True
    controller._announce_hidden_decision(kind, "owner", "exact")
    assert len(notices) == 2


def test_stable_router_does_not_suppress_detached_interrupt_bell():
    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    bells = []
    controller.app = SimpleNamespace(
        call_from_thread=lambda fn, *args: fn(*args),
        is_headless=False, bell=lambda: bells.append("bell"),
    )
    controller.set_pending_decision = lambda projection: False
    controller._interrupt_bell_enabled = lambda: True
    controller.on_pending_rounds_changed(1, "question", True)
    assert bells == ["bell"]


def test_cached_hidden_view_cannot_start_an_answerable_clock():
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    session = controller.store.ensure_session()
    app = SimpleNamespace(screen=object())
    runtime = ConsoleRuntime(app)
    runtime.set_chat_store(controller.store)
    runtime.set_chat_controller(controller)
    projections = []
    view = SimpleNamespace(
        app=app,
        console_view_hooks=lambda: {"set_pending_decision": lambda item: projections.append(item) or True},
    )
    runtime.view = runtime._reconciled_view = view
    state = {"event": threading.Event(), "session_id": session.id}
    controller._pending_approval_rounds["r1"] = state
    controller._publish_pending_decision(
        round_state=state, payload={"session_id": session.id},
        decision_type="approval", decision_id="r1", timeout_seconds=5,
        retained_store=controller._parked_approval_payloads,
    )
    assert not controller.project_pending_decision_for_active_session()
    assert state["active_since"] is None
    assert projections == []
    app.screen = view
    assert controller.project_pending_decision_for_active_session()
    assert state["active_since"] is not None
    assert len(projections) == 1
