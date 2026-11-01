"""Exact command targets use native custody and conversation-owned mailboxes."""

from types import SimpleNamespace

import pytest

from Tests.Chat import test_console_received_intent_custody as custody_controls
from Tests.Chat.test_console_agent_bridge_steering import (
    _bridge_with_fleet,
    _live_child,
)
from tldw_chatbook.Chat.console_remote_models import ConsoleHostIdentity, ConsoleTarget

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]
intent_case = custody_controls.intent_case


def _commands(case):
    from tldw_chatbook.Chat.console_remote_commands import ConsoleRemoteCommands

    runtime, _store, session, _request = case
    host = ConsoleHostIdentity(server_epoch="epoch", console_id="console")
    target = ConsoleTarget(
        **host.model_dump(),
        session_id=session.id,
        incarnation_id=session.incarnation_id,
    )
    return ConsoleRemoteCommands(runtime, host), target


async def test_exact_background_controls_never_switch_desktop(intent_case, tmp_path):
    from Tests.Agents.test_primary_redirect import _service

    commands, target = _commands(intent_case)
    runtime, store, session, _request = intent_case
    controller = runtime._chat_controller
    session.persisted_conversation_id = "conv-1"
    desktop = store.create_session(session_id="desktop-b", ephemeral=True)
    store.switch_session(desktop.id)
    bridge, _coordinator = _bridge_with_fleet(tmp_path)
    controller._agent_bridge = bridge
    service = _service(
        on_primary_steer_ready=lambda hook: controller._active_steer_hooks.__setitem__(
            session.id, hook
        ),
        on_primary_redirect_ready=lambda hook, _abort: (
            controller._active_redirect_hooks.__setitem__(session.id, hook)
        ),
    )
    drain = service._register_primary_mailbox("run-a")
    bridge._fleet_services["conv-1"] = service
    bridge._live_primary_runs["conv-1"] = "run-a"
    target = target.model_copy(update={"run_id": "run-a"})
    assert (await commands.execute("steer", target, {"text": "guide"}, "c1"))[
        "state"
    ] == "queued"
    assert (await commands.execute("redirect", target, {"text": "correct"}, "c2"))[
        "state"
    ] == "queued"
    stale = target.model_copy(update={"run_id": "old-run"})
    assert (await commands.execute("steer", stale, {"text": "wrong"}, "c3"))[
        "state"
    ] == "refused"
    assert service._primary_redirect_pending("run-a")
    assert drain() == [("user", "guide"), ("redirect", "correct")]
    store.switch_session(session.id)
    assert controller.steer_active_run("desktop guide") is None
    assert controller.redirect_active_run("desktop correction") is None
    assert drain() == [("user", "desktop guide"), ("redirect", "desktop correction")]
    store.switch_session(desktop.id)
    service._unregister_primary_mailbox("run-a")
    assert (await commands.execute("steer", target, {"text": "late"}, "c4"))[
        "state"
    ] == "refused"
    assert store.active_session_id == desktop.id
    bridge._fleet_services.clear()


@pytest.mark.parametrize(
    "mutation",
    [{"incarnation_id": "old"}, {"server_epoch": "old"}, {"session_id": "missing"}],
)
async def test_stale_target_cannot_touch_native_controls(intent_case, mutation):
    commands, target = _commands(intent_case)
    result = await commands.execute(
        "stop", target.model_copy(update=mutation), {}, "c1"
    )
    assert result["state"] == "refused"
    assert intent_case[1].session_draft(target.session_id) == "captured original draft"


async def test_untargeted_stop_and_worker_redirect_refuse(intent_case):
    commands, target = _commands(intent_case)
    assert (await commands.execute("stop", target, {}, "c1"))["state"] == "refused"
    worker = target.model_copy(update={"handle_id": "worker", "run_id": "child"})
    assert (await commands.execute("redirect", worker, {"text": "no"}, "c2"))[
        "state"
    ] == "refused"


async def test_subagent_guidance_uses_exact_mailbox_and_reports_queued(
    intent_case, tmp_path
):
    commands, target = _commands(intent_case)
    runtime, _store, session, _request = intent_case
    bridge, coordinator = _bridge_with_fleet(tmp_path)
    runtime._chat_controller._agent_bridge = bridge
    session.persisted_conversation_id = "conv-1"
    handle = _live_child(coordinator, run_id="child-run")
    worker = target.model_copy(update={"handle_id": handle, "run_id": "child-run"})
    result = await commands.execute(
        "subagent_guidance", worker, {"text": "focus"}, "c1"
    )
    assert result["state"] == "queued"
    assert coordinator.get(handle).queued_steering == 1
    assert coordinator.drain_steering(handle) == [("user", "focus")]
    assert coordinator.get(handle).queued_steering == 0
    stale = worker.model_copy(update={"run_id": "old-child"})
    assert (
        await commands.execute("subagent_guidance", stale, {"text": "wrong"}, "c2")
    )["state"] == "refused"
    coordinator.finish(handle, status="done")
    assert (
        await commands.execute("subagent_guidance", worker, {"text": "late"}, "c3")
    )["state"] == "refused"


@pytest.mark.parametrize(
    "revisions", [{}, {"settings_revision": True}, {"settings_revision": -1}]
)
async def test_send_refuses_incomplete_or_invalid_revision_map(intent_case, revisions):
    commands, target = _commands(intent_case)
    result = await commands.execute(
        "send", target, {"text": "phone", "expected_revisions": revisions}, "c1"
    )
    assert result["state"] == "refused"
    assert not intent_case[0].has_received_intents(target.session_id)


async def test_exact_received_stop_cancels_only_original_turn(intent_case):
    commands, target = _commands(intent_case)
    runtime, store, _session, _request = intent_case
    intent = custody_controls._intent(intent_case)
    runtime.accept_received_intent(intent)
    record = runtime._turn_custody[intent.turn_id]
    desktop = store.create_session(session_id="untouched-desktop", ephemeral=True)
    store.set_session_draft(desktop.id, "desktop next")
    wrong = target.model_copy(update={"turn_id": "old-turn"})
    assert (await commands.execute("stop", wrong, {}, "stop-wrong"))[
        "state"
    ] == "refused"
    assert not record.task.cancelling()
    exact = target.model_copy(update={"turn_id": intent.turn_id})
    assert (await commands.execute("stop", exact, {}, "stop-exact"))[
        "state"
    ] == "accepted"
    assert record.task.cancelling()
    assert store.active_session_id == desktop.id
    assert store.session_draft(desktop.id) == "desktop next"


async def test_finished_primary_memo_cannot_stop_next_preparation(
    intent_case, tmp_path
):
    import threading

    commands, target = _commands(intent_case)
    runtime, store, session, _request = intent_case
    bridge, _coordinator = _bridge_with_fleet(tmp_path)
    runtime._chat_controller._agent_bridge = bridge
    session.persisted_conversation_id = "conv-1"
    bridge._live_primary_runs["conv-1"] = "finished-primary"
    service = SimpleNamespace(
        _primary_steering_lock=threading.Lock(), _primary_mailboxes={}
    )
    bridge._fleet_services["conv-1"] = service
    intent = custody_controls._intent(intent_case, turn_id="next-preparation")
    runtime.accept_received_intent(intent)
    record = runtime._turn_custody[intent.turn_id]
    old = target.model_copy(update={"run_id": "finished-primary"})
    assert (await commands.execute("stop", old, {}, "stale-stop"))["state"] == "refused"
    assert not record.task.cancelling()
    assert store.received_turn_for_session(session.id) is record.received_claim
    bridge._fleet_services.clear()


async def test_subagent_stop_checks_exact_run_and_uses_native_owner(
    intent_case, tmp_path
):
    import threading

    from Tests.Agents.test_primary_redirect import _service

    commands, target = _commands(intent_case)
    runtime, _store, session, _request = intent_case
    bridge, coordinator = _bridge_with_fleet(tmp_path)
    runtime._chat_controller._agent_bridge = bridge
    session.persisted_conversation_id = "conv-1"
    handle_id = _live_child(coordinator, run_id="child-run")
    service = _service()
    service._fleet = coordinator
    cancelled = threading.Event()
    service._fleet_cancels[handle_id] = cancelled
    revoked = []
    service._revoke_approvals = revoked.append
    bridge._fleet_services["conv-1"] = service
    worker = target.model_copy(update={"handle_id": handle_id, "run_id": "child-run"})
    stale = worker.model_copy(update={"run_id": "old-child"})
    assert (await commands.execute("subagent_stop", stale, {}, "stop-old"))[
        "state"
    ] == "refused"
    assert not cancelled.is_set()
    assert (await commands.execute("subagent_stop", worker, {}, "stop-exact"))[
        "state"
    ] == "accepted"
    assert cancelled.is_set() and revoked == ["child-run"]
    assert coordinator.get(handle_id).status == "running"
    coordinator.finish(handle_id, "done")
    assert (await commands.execute("subagent_stop", worker, {}, "stop-late"))[
        "state"
    ] == "refused"
    bridge._fleet_services.clear()
