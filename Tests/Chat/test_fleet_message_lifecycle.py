"""Real native ownership gates preserve navigation and invalidate closed sessions."""

from __future__ import annotations

import pytest

from tldw_chatbook.Agents.fleet_messages import MessageError
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.fixture
def setup(tmp_path):
    store = ConsoleChatStore()
    bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(tmp_path / "runs.db", client_id="test"),
        store=store,
        provider_gateway=object(),
    )
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_bridge=bridge
    )
    return store, bridge, controller


def sender_for(bridge, conversation):
    store = bridge._store
    session_id = store.active_session_id
    if session_id is None:
        session_id = store.create_session(session_id=conversation).id
    with store.progress_owner_scope(
        session_id, message_store=bridge.message_store
    ) as owner:
        fleet = bridge._conversation_fleet_coordinator(
            conversation, progress_owner_id=owner
        )
    handle = fleet.reserve(task="task", agent="child")
    fleet.attach_run(handle.handle_id, "child-run")
    return fleet, fleet.bind_progress_sender(
        handle.handle_id, parent_run_id="primary", chain_id=None
    )


def test_inspection_is_noncreating_and_reopen_has_new_authority(setup):
    store, bridge, controller = setup
    assert bridge._message_store is None
    assert bridge.progress_snapshot("missing") == ()
    assert bridge.progress_counts() == {}
    assert bridge.discard_progress("missing", ["id"]) == 0
    assert bridge._fleet_coordinators == {}
    assert bridge._message_store is None
    controller.new_session()
    session = store.active_session_id
    conversation = controller._agent_conversation_id(session)
    old_fleet, old_sender = sender_for(bridge, conversation)
    owner = store.progress_owner_id(session)
    old_sender.send("before navigation")
    controller.new_session()
    controller.switch_session(session)
    assert len(bridge.progress_snapshot(owner)) == 1
    reader = old_fleet.message_inbox.reader("primary", chain_id=None, automatic=False)
    cancellation = []

    def cancel(conversation_id):
        assert bridge.progress_snapshot(owner) == ()
        with pytest.raises(MessageError):
            old_sender.send("late callback during cancellation")
        for handle in old_fleet.snapshot():
            old_fleet.finish(handle.handle_id, "cancelled")
        cancellation.append(conversation_id)
        return 0

    bridge.cancel_all_subagents = cancel
    ticket = controller.begin_session_close(session, expected_revision=controller.lifecycle_impact(session_id=session).revision)
    controller.finalize_session_close(ticket)
    assert controller.release_session_close_fences(ticket)
    assert cancellation == [conversation]
    store.create_session(session_id=session)
    new_owner = store.progress_owner_id(session)
    new_fleet, new_sender = sender_for(bridge, conversation)
    assert new_fleet is not old_fleet
    new_sender.send("replacement")
    with pytest.raises(MessageError):
        reader.collect()
    with pytest.raises(MessageError):
        old_sender.send("late")
    assert [m.body for m in bridge.progress_snapshot(new_owner)] == ["replacement"]


@pytest.mark.asyncio
async def test_detach_inert_disposal_and_replacement_close_before_shutdown(setup):
    store, bridge, controller = setup
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    runtime.set_agent_bridge(bridge)
    _, sender = sender_for(bridge, "c")
    sender.send("pending")
    runtime.detach_view(None)
    assert bridge.progress_counts() == {"c": 1}
    calls = []

    async def shutdown():
        assert bridge.progress_counts() == {}
        with pytest.raises(MessageError):
            sender.send("late")
        calls.append("shutdown")

    controller.shutdown = shutdown
    await runtime.dispose()
    assert calls == ["shutdown"]


def test_bridge_replacement_revokes_old_progress(setup):
    _, bridge, _ = setup
    runtime = ConsoleRuntime(app=None)
    runtime.set_agent_bridge(bridge)
    _, sender = sender_for(bridge, "c")
    sender.send("pending")
    runtime.set_agent_bridge(None)
    assert bridge.progress_counts() == {}
    with pytest.raises(MessageError):
        sender.send("late")


@pytest.mark.parametrize("existing", [False, True])
def test_real_bridge_toolless_disclosure_does_not_create_an_inbox(
    tmp_path, monkeypatch, existing
):
    from dataclasses import replace

    from Tests.Chat.test_console_agent_bridge import _bridge, _run
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat import console_agent_bridge as module

    bridge, _, store, session, assistant = _bridge(tmp_path, [["done"]])
    bridge._registry = ToolCatalogRegistry()
    bridge._allowed_tools = ()
    budget = replace(module.console_run_budget(), max_subagents=0)
    monkeypatch.setattr(module, "console_run_budget", lambda: budget)
    if existing:
        _, sender = sender_for(bridge, "conv-1")
        sender.send("waiting")
    assert _run(bridge, store, session, assistant).status == "done"
    assert (
        bridge.message_store.get_inbox(store.progress_owner_id(session.id)) is not None
    ) is existing
    assert bool(bridge._fleet_coordinators) is existing
    prompt = bridge._gateway.messages_seen[0][0]["content"]
    assert ("read_agent_messages" in prompt) is existing
    if existing:
        from tldw_chatbook.Agents.fleet_message_tools import READ_INSTRUCTIONS

        assert prompt.count(READ_INSTRUCTIONS) == 1


def test_existing_progress_remains_readable_when_fleet_is_disabled(
    tmp_path, monkeypatch
):
    from Tests.Agents.conftest import pin_agent_settings
    from Tests.Agents.test_agent_service import fence
    from Tests.Chat.test_console_agent_bridge import _bridge, _run

    bridge, _, store, session, assistant = _bridge(
        tmp_path,
        [[fence("spawn_subagent", {"task": "inline"})], ["child done"], ["done"]],
    )
    _, sender = sender_for(bridge, "conv-1")
    sender.send("waiting")
    pin_agent_settings(monkeypatch, max_live_subagents=1)
    assert _run(bridge, store, session, assistant).status == "done"
    prompt = bridge._gateway.messages_seen[0][0]["content"]
    assert '"name": "read_agent_messages"' in prompt
    assert '"name": "wait_agents"' not in prompt
    assert '"name": "send_to_agent"' not in prompt
    assert '"name": "check_agents"' not in prompt
    assert bridge._gateway.calls == 3
    child_prompt = bridge._gateway.messages_seen[1][0]["content"]
    assert "report_to_supervisor" not in child_prompt
    assert "child done" in bridge._gateway.messages_seen[2][-1]["content"]
    assert len(bridge._fleet_coordinators["conv-1"].snapshot()) == 1


def test_live_and_restored_rail_markers_are_body_free_with_capture_off(
    tmp_path, monkeypatch
):
    import json

    from Tests.Agents.conftest import pin_agent_settings
    from Tests.Chat.test_console_agent_bridge import (
        _bridge,
        _native_calls,
        _native_resolution,
        _run,
    )
    from tldw_chatbook.Agents import agent_service, run_log
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    pin_agent_settings(monkeypatch, run_log_enabled=False)
    monkeypatch.setattr(run_log, "_setting", agent_service._setting)
    bridge, db, store, session, assistant = _bridge(
        tmp_path, [[_native_calls("read_agent_messages", {})], ["done"]]
    )
    _, sender = sender_for(bridge, "conv-1")
    secret = "PRIVATE-RAIL-REPORT-BODY"
    sender.send(secret)
    assert (
        _run(bridge, store, session, assistant, resolution=_native_resolution()).status
        == "done"
    )
    result_row = bridge._gateway.messages_seen[1][-1]
    assert result_row["role"] == "tool"
    assert json.loads(result_row["content"])["messages"][0]["body"] == secret
    assert secret not in repr(bridge.live_snapshot("conv-1"))
    assert secret not in repr(bridge.historical_snapshot("conv-1"))
    live_markers = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert live_markers
    assert secret not in repr(live_markers)
    restored_markers = bridge.resume_marker_messages("conv-1")
    assert restored_markers
    assert secret not in repr(restored_markers)
    assert secret not in json.dumps([row["steps"] for row in db.list_runs("conv-1")])


def test_replaced_unused_bridge_cannot_revoke_replacement(tmp_path):
    from Tests.Chat.test_console_agent_bridge import _bridge, _run

    old, _, store, session, assistant = _bridge(tmp_path, [["done"]])
    runtime = ConsoleRuntime(app=None)
    runtime.set_agent_bridge(old)
    assert old._message_store is None
    new = ConsoleAgentBridge(
        agent_runs_db=old._db, store=store, provider_gateway=object()
    )
    runtime.set_agent_bridge(new)
    _, sender = sender_for(new, "conv-1")
    sender.send("current report")
    with pytest.raises(MessageError):
        _run(old, store, session, assistant)
    assert old._message_store is None
    assert old.progress_counts() == {}
    assert old.progress_snapshot("missing") == ()
    sender.send("still current")


def test_lazy_progress_allocation_cannot_cross_close(setup):
    import threading

    _, bridge, _ = setup
    entered = threading.Event()
    failures = []

    def allocate():
        entered.set()
        try:
            bridge.message_store
        except MessageError as exc:
            failures.append(exc)

    with bridge._message_store_lock:
        worker = threading.Thread(target=allocate)
        worker.start()
        assert entered.wait(2)
        bridge.close_all_progress()
    worker.join(2)
    assert not worker.is_alive()
    assert len(failures) == 1
    assert bridge._message_store is None
    assert bridge._session_progress_inbox("missing") is None
