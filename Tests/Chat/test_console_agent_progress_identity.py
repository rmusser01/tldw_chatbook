"""Real submitted progress survives temporary-session persistence transitions."""

import asyncio
import threading

import pytest

from Tests.Agents.conftest import pin_agent_settings
from Tests.Chat.test_console_agent_swap import _controller, _fence
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _setup(tmp_path, monkeypatch, *, temporary=True):
    pin_agent_settings(monkeypatch, max_live_subagents=3, run_log_enabled=False)
    controller, store, runs = _controller(
        tmp_path,
        [
            [_fence("spawn_subagent", {"task": "report finding"})],
            [_fence("wait_agents", {})],
            ["done"],
            [_fence("read_agent_messages", {})],
            ["later done"],
        ],
        child_scripts=[
            [_fence("report_to_supervisor", {"message": "original finding"})],
            ["child done"],
        ],
    )
    db = CharactersRAGDB(str(tmp_path / "chat.db"), "progress-identity")
    store.persistence = ChatPersistenceService(db)
    session = store.create_session(
        ephemeral=temporary,
        project_instruction_state=ProjectInstructionControlState.legacy_disabled(),
    )
    return controller, store, runs, db, session


def _reader_results(runs, conversation_id):
    return [
        step
        for run in runs.list_runs(conversation_id)
        for step in runs.get_run(run["id"])["steps"]
        if step.get("tool_name") == "read_agent_messages"
        and step.get("kind") == "tool_result"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("temporary", [False, True])
async def test_actual_submit_report_save_then_submitted_reader(
    tmp_path, monkeypatch, temporary
):
    controller, store, runs, db, session = _setup(
        tmp_path, monkeypatch, temporary=temporary
    )
    try:
        assert (await controller.submit_draft("Delegate and retain progress")).accepted
        bridge = controller._agent_bridge
        owner = next(iter(bridge.message_store.pending_counts()))
        inbox = bridge.message_store.get_inbox(owner)
        original = inbox.snapshot()
        assert len(original) == 1
        if temporary:
            assert await asyncio.to_thread(store.promote_ephemeral_session, session.id)
        assert (await controller.submit_draft("Collect earlier progress")).accepted
        results = _reader_results(runs, session.persisted_conversation_id)
        assert len(results) == 1
        assert '"collected_count":1' in results[0]["result"].replace(" ", "")
        assert inbox.snapshot() == ()
        controller.close_session(session.id)
        assert bridge.message_store.pending_counts() == {}
    finally:
        await controller.shutdown()
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["rollback", "cancel_await", "close_during_save"])
async def test_progress_owner_during_threaded_save(tmp_path, monkeypatch, outcome):
    controller, store, _runs, db, session = _setup(tmp_path, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original_persist = store.persist_message_if_needed

    def parked_write(message_id):
        assert session.persisted_conversation_id
        entered.set()
        assert release.wait(10)
        if outcome == "rollback":
            raise RuntimeError("injected Save failure")
        return original_persist(message_id)

    def save():
        try:
            return store.promote_ephemeral_session(session.id)
        finally:
            finished.set()

    save_task = None
    try:
        assert (await controller.submit_draft("Delegate and retain progress")).accepted
        bridge = controller._agent_bridge
        owner = next(iter(bridge.message_store.pending_counts()))
        inbox = bridge.message_store.get_inbox(owner)
        original = inbox.snapshot()
        monkeypatch.setattr(store, "persist_message_if_needed", parked_write)
        save_task = asyncio.create_task(asyncio.to_thread(save))
        assert await asyncio.to_thread(entered.wait, 10)
        # Save has assigned its transient durable ID but has not committed.
        assert store.progress_owner_id(session.id) == owner
        assert inbox.snapshot() == original
        if outcome == "close_during_save":
            controller.close_session(session.id)
            assert not bridge.message_store.pending_counts()
        if outcome == "cancel_await":
            save_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await save_task
        release.set()
        if outcome == "rollback":
            with pytest.raises(RuntimeError, match="injected Save failure"):
                await save_task
            assert session.ephemeral and session.persisted_conversation_id is None
        elif outcome == "close_during_save":
            # The existing persistence operation can fail after native close;
            # either result must leave progress closed and unresurrected.
            await asyncio.gather(save_task, return_exceptions=True)
        else:
            assert await asyncio.to_thread(finished.wait, 10)
            assert not session.ephemeral and session.persisted_conversation_id
        if outcome != "close_during_save":
            assert store.progress_owner_id(session.id) == owner
            assert inbox.snapshot() == original
            controller.close_session(session.id)
        assert not bridge.message_store.pending_counts()
    finally:
        release.set()
        if save_task is not None:
            await asyncio.gather(save_task, return_exceptions=True)
        await controller.shutdown()
        db.close_connection()


@pytest.mark.asyncio
async def test_saved_native_sibling_shares_owner_until_last_close(
    tmp_path, monkeypatch
):
    controller, store, runs, db, session = _setup(tmp_path, monkeypatch)
    try:
        assert (await controller.submit_draft("Delegate and retain progress")).accepted
        owner = store.progress_owner_id(session.id)
        bridge = controller._agent_bridge
        inbox = bridge.message_store.get_inbox(owner)
        saved = await asyncio.to_thread(store.promote_ephemeral_session, session.id)
        sibling = store.restore_persisted_session(
            title="Same saved chat",
            workspace_id=None,
            persisted_conversation_id=saved,
            all_nodes=[],
        )
        assert sibling.id != session.id
        assert store.progress_owner_id(sibling.id) == owner
        controller.close_session(session.id)
        assert bridge.message_store.get_inbox(owner) is inbox
        assert bridge.progress_counts() == {sibling.id: 1}
        assert (await controller.submit_draft("Collect earlier progress")).accepted
        assert '"collected_count":1' in _reader_results(runs, saved)[0]["result"]
        controller.close_session(sibling.id)
        assert bridge.message_store.get_inbox(owner) is None
    finally:
        await controller.shutdown()
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["native_state", "runtime"])
async def test_replacement_revokes_old_progress_without_stale_teardown(
    tmp_path, monkeypatch, replacement
):
    from tldw_chatbook.Agents.fleet_messages import MessageError, MessageStore

    controller, store, _runs, db, session = _setup(tmp_path, monkeypatch)
    try:
        assert (await controller.submit_draft("Delegate and retain progress")).accepted
        bridge = controller._agent_bridge
        owner = store.progress_owner_id(session.id)
        inbox = bridge.message_store.get_inbox(owner)
        if replacement == "native_state":
            store.restore_state(sessions=[session], active_session_id=session.id)
            assert store.progress_owner_id(session.id) != owner
        else:
            successor = MessageStore()
            store.register_progress_message_store(successor)
            with store.progress_owner_scope(
                session.id, message_store=successor
            ) as current:
                replacement_inbox = successor.open_inbox(current)
            bridge.close_progress(session.id)
            bridge.close_all_progress()
            assert successor.get_inbox(owner) is replacement_inbox
            assert store.progress_owner_id(session.id) == owner
        with pytest.raises(MessageError, match="unavailable"):
            inbox.snapshot()
        assert bridge.message_store.pending_counts() == {}
        controller.close_session(session.id)
    finally:
        await controller.shutdown()
        db.close_connection()


@pytest.mark.asyncio
async def test_live_child_and_current_reader_keep_exact_owner_across_save(
    tmp_path, monkeypatch
):
    from Tests.Chat.test_console_agent_swap import SUBAGENT_PROMPT_PREFIX

    controller, store, runs, db, session = _setup(tmp_path, monkeypatch)
    gateway = controller._agent_bridge._gateway
    gateway._scripts = [
        [_fence("spawn_subagent", {"task": "report twice"})],
        [_fence("wait_agents", {})],
        [_fence("read_agent_messages", {})],
        ["done"],
    ]
    gateway._child_scripts = [
        [_fence("report_to_supervisor", {"message": "before Save"})],
        [_fence("report_to_supervisor", {"message": "after Save"})],
        ["child done"],
    ]
    entered = threading.Event()
    release = threading.Event()
    received = []
    original_stream = gateway.stream_chat

    async def gated_stream(resolution, messages, **kwargs):
        child = str(messages[0].get("content", "")).startswith(SUBAGENT_PROMPT_PREFIX)
        if child and gateway.child_calls == 1:
            entered.set()
            assert await asyncio.to_thread(release.wait, 10)
        if not child:
            received.append(str(messages))
        async for chunk in original_stream(resolution, messages, **kwargs):
            yield chunk

    monkeypatch.setattr(gateway, "stream_chat", gated_stream)
    task = asyncio.create_task(controller.submit_draft("Delegate and collect"))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        bridge = controller._agent_bridge
        owner = store.progress_owner_id(session.id)
        inbox = bridge.message_store.get_inbox(owner)
        original = inbox.snapshot()[0]
        assert await asyncio.to_thread(store.promote_ephemeral_session, session.id)
        assert store.progress_owner_id(session.id) == owner
        assert inbox.snapshot()[0] == original
        release.set()
        assert (await task).accepted
        assert inbox.snapshot() == ()
        assert '"collected_count":2' in _reader_results(runs, session.id)[0]["result"]
        assert "before Save" in received[-1] and "after Save" in received[-1]
        assert original.message_id in received[-1]
        assert original.identity.run_id in received[-1]
        controller.close_session(session.id)
        assert not bridge.message_store.pending_counts()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await controller.shutdown()
        db.close_connection()


@pytest.mark.asyncio
async def test_setup_and_last_close_serialize_without_reopening(tmp_path, monkeypatch):
    from tldw_chatbook.Agents.fleet_messages import MessageError

    controller, store, _runs, db, session = _setup(tmp_path, monkeypatch)
    binding_entered = threading.Event()
    release_binding = threading.Event()
    close_entered = threading.Event()
    order = []
    try:
        assert (await controller.submit_draft("Delegate and retain progress")).accepted
        bridge = controller._agent_bridge
        owner = store.progress_owner_id(session.id)
        inbox = bridge.message_store.get_inbox(owner)

        def setup():
            with store.progress_owner_scope(
                session.id, message_store=bridge.message_store
            ) as current:
                assert current == owner
                assert bridge.message_store.open_inbox(current) is inbox
                binding_entered.set()
                assert release_binding.wait(10)
                assert len(inbox.snapshot()) == 1
                order.append("binding_finished")

        def close():
            close_entered.set()
            store.close_session(session.id)
            order.append("closed")

        setup_task = asyncio.create_task(asyncio.to_thread(setup))
        assert await asyncio.to_thread(binding_entered.wait, 10)
        close_task = asyncio.create_task(asyncio.to_thread(close))
        assert await asyncio.to_thread(close_entered.wait, 10)
        release_binding.set()
        await asyncio.gather(setup_task, close_task)
        assert order == ["binding_finished", "closed"]
        with store.progress_owner_scope(
            session.id, message_store=bridge.message_store
        ) as current:
            assert current is None
        assert bridge._session_progress_inbox(session.id) is None
        assert bridge.message_store.get_inbox(owner) is None
        assert bridge.message_store.pending_counts() == {}
        with pytest.raises(MessageError):
            inbox.snapshot()
    finally:
        release_binding.set()
        await controller.shutdown()
        db.close_connection()


def test_simultaneous_saved_sibling_publication_shares_one_owner():
    from concurrent.futures import ThreadPoolExecutor

    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore()
    start = threading.Barrier(2)

    def restore():
        start.wait(10)
        return store.restore_persisted_session(
            title="Saved",
            workspace_id=None,
            persisted_conversation_id="shared-durable",
            all_nodes=[],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = [
            future.result(10) for future in [pool.submit(restore), pool.submit(restore)]
        ]
    assert first.id != second.id
    assert store.progress_owner_id(first.id) == store.progress_owner_id(second.id)
    assert len(set(store.progress_owner_ids().values())) == 1


@pytest.mark.asyncio
async def test_saved_progress_preview_is_noncreating_and_matches_later_reader(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from Tests.Chat.test_console_agent_project_instructions import _candidate
    from tldw_chatbook.Chat import console_agent_bridge as bridge_module

    controller, store, runs, db, session = _setup(tmp_path, monkeypatch)
    try:
        assert (await controller.submit_draft("Delegate and retain progress")).accepted
        bridge = controller._agent_bridge
        owner = store.progress_owner_id(session.id)
        inbox = bridge.message_store.get_inbox(owner)
        before = inbox.snapshot()
        assert await asyncio.to_thread(store.promote_ephemeral_session, session.id)
        gateway = bridge._gateway
        calls_before = gateway.calls
        with monkeypatch.context() as preview_patch:
            budget = replace(bridge_module.console_run_budget(), max_subagents=0)
            preview_patch.setattr(bridge_module, "console_run_budget", lambda: budget)
            resolution = await gateway.resolve_for_send(None)
            preview = await asyncio.to_thread(
                bridge.build_project_instruction_preview_request,
                candidate=_candidate(tmp_path),
                session_id=session.id,
                resolution=resolution,
                fallback_model="test-model",
                session_system_prompt="",
                agent_messages=[{"role": "user", "content": "preview"}],
            )
            assert preview is not None
            assert "read_agent_messages" in str(preview[0])
            assert "original finding" not in str(preview[0])
        assert gateway.calls == calls_before
        assert inbox.snapshot() == before
        assert bridge.message_store.get_inbox(owner) is inbox
        assert (await controller.submit_draft("Collect earlier progress")).accepted
        assert (
            '"collected_count":1'
            in _reader_results(runs, session.persisted_conversation_id)[0]["result"]
        )
    finally:
        await controller.shutdown()
        db.close_connection()


def test_delayed_run_setup_cannot_bind_reused_native_id(tmp_path, monkeypatch):
    from Tests.Chat.test_console_agent_bridge import _bridge, _run
    from tldw_chatbook.Agents.fleet_messages import MessageError
    from tldw_chatbook.Chat import console_agent_bridge as bridge_module

    bridge, _, store, session, assistant = _bridge(tmp_path, [["done"]])
    owner = store.progress_owner_id(session.id)
    original_plan = bridge_module.build_console_first_request_plan

    def replace_during_setup(**kwargs):
        plan = original_plan(**kwargs)
        store.close_session(session.id)
        store.create_session(session_id=session.id)
        assert store.progress_owner_id(session.id) != owner
        return plan

    monkeypatch.setattr(
        bridge_module, "build_console_first_request_plan", replace_during_setup
    )
    with pytest.raises(MessageError, match="unavailable"):
        _run(bridge, store, session, assistant)
    assert bridge.message_store.pending_counts() == {}
    assert bridge.message_store.get_inbox(store.progress_owner_id(session.id)) is None
    assert bridge._gateway.calls == 0


@pytest.mark.asyncio
async def test_queued_submission_cannot_bind_replaced_native_state(
    tmp_path, monkeypatch
):
    controller, store, _runs, db, session = _setup(tmp_path, monkeypatch)
    bridge = controller._agent_bridge
    owner = store.progress_owner_id(session.id)
    entered, release = threading.Event(), threading.Event()
    original_reply = bridge.run_reply

    def queued_reply(**kwargs):
        entered.set()
        assert release.wait(10)
        return original_reply(**kwargs)

    monkeypatch.setattr(bridge, "run_reply", queued_reply)
    task = asyncio.create_task(controller.submit_draft("Queued delegation"))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        store.restore_state(sessions=[session], active_session_id=session.id)
        replacement = store.progress_owner_id(session.id)
        assert replacement != owner
        release.set()
        await task
        assert bridge._gateway.calls == 0
        assert bridge.message_store.get_inbox(replacement) is None
        assert bridge.message_store.pending_counts() == {}
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await controller.shutdown()
        db.close_connection()
