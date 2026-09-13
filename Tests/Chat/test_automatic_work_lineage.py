"""Accepted manual work and descendant wakes keep explicit immutable lineage."""

import asyncio
import sqlite3
import threading

import pytest

from Tests.Chat.test_child_run_scope_ordering import _survivor_bridge
from Tests.Chat.test_console_agent_bridge import _fence, _join_fleet_threads, _run
from Tests.Chat.test_console_agent_swap import (
    _controller,
    _disable_project_instructions_for_legacy_agent_swap_tests,  # noqa: F401
)
from Tests.Chat.test_console_fleet_wake import (
    _controller_rig,
    _drain,
    _settle,
    _survivor,
    _terminal_subagent_run,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_enabled", [True, False])
async def test_accepted_manual_turns_establish_distinct_chains_for_both_paths(
    tmp_path, agent_enabled
):
    controller, store, db = _controller(
        tmp_path, [["first"], ["second"]], enabled=agent_enabled
    )
    first = await controller.submit_draft("first request")
    second = await controller.submit_draft("second request")
    assert first.accepted and second.accepted
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT id FROM automatic_work_chains ORDER BY created_at"
        ).fetchall()
    assert len(rows) == 2
    assert rows[0]["id"] != rows[1]["id"]
    for row in rows:
        assert db.automatic_work.snapshot(row["id"]).used["generation"] == 0
    if agent_enabled:
        assert {
            run["work_chain_id"] for run in db.list_runs(store.ensure_session().persisted_conversation_id or store.ensure_session().id)
        } == {row["id"] for row in rows}
    db.close()


@pytest.mark.asyncio
async def test_refused_manual_send_creates_no_allowance(tmp_path):
    chacha, _, db, _, session, gateway, _, controller = _controller_rig(tmp_path)
    gateway.ready = False
    try:
        result = await controller.submit_draft("refused", session_id=session.id)
        assert not result.accepted
        with db.connection() as conn:
            assert (
                conn.execute("SELECT COUNT(*) FROM automatic_work_chains").fetchone()[0]
                == 0
            )
    finally:
        controller._disposed = True
        db.close()
        chacha.close()


@pytest.mark.asyncio
async def test_failed_chain_write_prevents_dispatch_and_clears_stream_ownership(
    tmp_path,
):
    controller, store, db = _controller(tmp_path, [["must not dispatch"]])
    with db.transaction() as conn:
        conn.execute(
            "CREATE TRIGGER deny_chain BEFORE INSERT ON automatic_work_chains BEGIN SELECT RAISE(ABORT, 'write refused'); END"
        )
    try:
        result = await controller.submit_draft("save me")
        # Durable acceptance precedes allowance creation; retain the accepted
        # input for recovery while refusing provider dispatch.
        assert result.accepted
        assert controller.provider_gateway.calls == 0
        assert controller._active_stream_tasks == {}
        assert controller._active_assistant_message_ids == {}
        assert controller.run_state_for(store.ensure_session().id).is_send_allowed
    finally:
        db.close()


@pytest.mark.asyncio
async def test_chain_creation_waits_for_sqlite_without_blocking_console(
    tmp_path, monkeypatch
):
    controller, _, db = _controller(tmp_path, [["reply"]], enabled=False)
    entered = threading.Event()
    create_chain = db.automatic_work.create_chain

    def observed_create(*args, **kwargs):
        entered.set()
        return create_chain(*args, **kwargs)

    monkeypatch.setattr(db.automatic_work, "create_chain", observed_create)
    blocker = sqlite3.connect(db.db_path_str, check_same_thread=False)
    blocker.execute("BEGIN IMMEDIATE")
    # A watchdog releases a regression that blocks the event loop itself.
    watchdog_fired = threading.Event()

    def unblock():
        watchdog_fired.set()
        blocker.rollback()

    watchdog = threading.Timer(3, unblock)
    watchdog.start()
    send = asyncio.create_task(controller.submit_draft("hello"))
    try:
        assert await _settle(entered.is_set)
        await asyncio.sleep(0.03)
        assert not send.done()
        assert not watchdog_fired.is_set()
        assert controller.provider_gateway.calls == 0
        blocker.rollback()
        assert (await asyncio.wait_for(send, 5)).accepted
    finally:
        watchdog.cancel()
        watchdog.join()
        blocker.rollback()
        blocker.close()
        if not send.done():
            send.cancel()
        await asyncio.gather(send, return_exceptions=True)
        db.close()


@pytest.mark.asyncio
async def test_manual_chain_uses_persisted_conversation_identity(tmp_path):
    from Tests.Chat.test_console_agent_swap import _Gateway
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    chacha = CharactersRAGDB(tmp_path / "conversations.sqlite", client_id="test")
    db = AgentRunsDB(tmp_path / "runs.sqlite")
    store = ConsoleChatStore(persistence=ChatPersistenceService(chacha))
    gateway = _Gateway([["reply"]])
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    try:
        assert (await controller.submit_draft("hello")).accepted
        session = store.ensure_session()
        conversation_id = controller._agent_conversation_id(session.id)
        assert conversation_id != session.id
        assert chacha.get_conversation_by_id(conversation_id) is not None
        run = db.list_runs(conversation_id, agent_kind="primary")[0]
        assert (
            db.automatic_work.snapshot(run["work_chain_id"]).conversation_id
            == conversation_id
        )
    finally:
        db.close()
        chacha.close_connection()


def test_child_keeps_its_spawning_chain_after_a_new_primary_turn(tmp_path):
    gate, gateway, db, store, session, aid, bridge = _survivor_bridge(
        tmp_path,
        parent_script=[
            [_fence("spawn_subagent", {"task": "old child"})],
            ["first parent"],
            ["new parent"],
        ],
        needed=1,
    )
    first_chain = db.automatic_work.create_chain(session.id, root_submission_id="first")
    second_chain = db.automatic_work.create_chain(
        session.id, root_submission_id="second"
    )
    try:
        assert (
            _run(
                bridge,
                store,
                session,
                aid,
                conversation_id=session.id,
                work_chain_id=first_chain,
            ).status
            == "done"
        )
        assert gateway.entered_event.wait(5)
        assert (
            _run(
                bridge,
                store,
                session,
                aid,
                conversation_id=session.id,
                work_chain_id=second_chain,
            ).status
            == "done"
        )
    finally:
        gate.set()
        _join_fleet_threads()
    children = db.list_runs(session.id, agent_kind="subagent")
    assert len(children) == 1
    assert children[0]["work_chain_id"] == first_chain
    assert db.get_run(children[0]["parent_run_id"])["work_chain_id"] == first_chain
    assert {
        row["work_chain_id"] for row in db.list_runs(session.id, agent_kind="primary")
    } == {first_chain, second_chain}
    db.close()


@pytest.mark.asyncio
async def test_plain_wakes_keep_separate_result_lineage_and_no_fresh_allowance(
    tmp_path,
):
    chacha, _, db, _, session, gateway, _, controller = _controller_rig(tmp_path)
    seen = []
    real_stream = gateway.stream_chat

    async def stream(*args, signals=None, **kwargs):
        seen.append(signals.automatic_work_chain_id)
        async for chunk in real_stream(*args, signals=signals, **kwargs):
            yield chunk

    gateway.stream_chat = stream
    run_ids = []
    chains = []
    for name in ("one", "two"):
        chain_id = db.automatic_work.create_chain(session.id, root_submission_id=name)
        chains.append(chain_id)
        _, child = _terminal_subagent_run(
            db, session.id, result=f"result {name}", work_chain_id=chain_id
        )
        run_ids.append(child)
    wake = controller.fleet_wake
    try:
        wake.on_fleet_drained(
            _drain(
                session.id,
                *[_survivor(run_id, session_id=session.id) for run_id in run_ids],
            )
        )
        assert await _settle(lambda: len(gateway.payloads) == 2)
        assert await _settle(lambda: not wake.has_pending(session.id))
        assert seen == chains
        assert "result one" in gateway.payloads[0][-1]["content"]
        assert "result two" not in gateway.payloads[0][-1]["content"]
        assert "result two" in gateway.payloads[1][-1]["content"]
        with db.connection() as conn:
            assert (
                conn.execute("SELECT COUNT(*) FROM automatic_work_chains").fetchone()[0]
                == 2
            )
    finally:
        controller._disposed = True
        db.close()
        chacha.close()


@pytest.mark.asyncio
async def test_prior_wake_token_cannot_authorize_a_later_delivery(tmp_path):
    chacha, _, db, _, session, gateway, _, controller = _controller_rig(tmp_path)
    tokens = []
    submit = controller.submit_draft

    async def capture(*args, wake_authorization=None, **kwargs):
        tokens.append(wake_authorization)
        return await submit(*args, wake_authorization=wake_authorization, **kwargs)

    controller.submit_draft = capture
    gate = asyncio.Event()
    wake = controller.fleet_wake
    try:
        _, first = _terminal_subagent_run(db, session.id)
        wake.on_fleet_drained(
            _drain(session.id, _survivor(first, session_id=session.id))
        )
        assert await _settle(lambda: tokens and not wake.has_pending(session.id))
        gateway.stream_gate = gate
        _, second = _terminal_subagent_run(db, session.id)
        wake.on_fleet_drained(
            _drain(session.id, _survivor(second, session_id=session.id))
        )
        assert await _settle(lambda: len(tokens) == 2)
        assert not wake.authorizes(tokens[0], session.id)
        assert wake.authorizes(tokens[1], session.id)
    finally:
        gate.set()
        await _settle(lambda: not wake.has_pending(session.id))
        controller._disposed = True
        db.close()
        chacha.close()
