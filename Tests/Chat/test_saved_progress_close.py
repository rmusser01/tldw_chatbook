"""Native close retires both causal sources after explicit Save."""

import asyncio

import pytest

from Tests.Chat.test_automatic_wake_budget import close_rig, queue_result, result_for
from Tests.Chat.test_console_fleet_wake import _settle
from Tests.Chat.test_progress_wakes import progress_rig, reporter
from Tests.private_profile import private_profile_test
from tldw_chatbook.Agents.fleet_messages import MessageError
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


@pytest.mark.asyncio
@private_profile_test
async def test_saved_native_close_fences_old_completion_and_new_saved_progress(
    tmp_path, request, monkeypatch
):
    rig = list(progress_rig(tmp_path))
    rig[4] = rig[3].create_session(ephemeral=True)
    rig = tuple(rig)
    entered, release = asyncio.Event(), asyncio.Event()
    controller, bridge, native = rig[7], rig[6], rig[3]
    original_substitution = controller._apply_skill_substitution

    async def gate_substitution(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original_substitution(*args, **kwargs)

    monkeypatch.setattr(controller, "_apply_skill_substitution", gate_substitution)
    fenced, cancelled, drained = [], [], []
    original_fence = bridge.fence_fleet
    original_cancel = bridge.cancel_all_subagents
    original_drain = bridge.await_fleet_terminal

    def fence(conversation_id, *, generation):
        fenced.append(conversation_id)
        return original_fence(conversation_id, generation=generation)

    def cancel(conversation_id):
        cancelled.append(conversation_id)
        return original_cancel(conversation_id)

    async def drain(conversation_id):
        drained.append(conversation_id)
        return await original_drain(conversation_id)

    monkeypatch.setattr(bridge, "fence_fleet", fence)
    monkeypatch.setattr(bridge, "cancel_all_subagents", cancel)
    monkeypatch.setattr(bridge, "await_fleet_terminal", drain)
    try:
        old_sender, _, old_chain = reporter(rig)
        saved_id = native.promote_ephemeral_session(rig[4].id)
        assert saved_id != rig[4].id
        completed = result_for(rig, old_chain)
        queue_result(rig, completed)
        assert await _settle(entered.is_set)
        authorization = controller.fleet_wake._active[rig[4].id].authorization
        assert authorization.progress_owner_key is None
        saved_sender, _, _ = reporter(rig, conversation_id=saved_id)
        saved_report = saved_sender.send("Retain saved report after native close")
        runtime = ConsoleRuntime(app=None)
        runtime.set_chat_store(native)
        runtime.set_agent_bridge(bridge)
        runtime.set_chat_controller(controller)
        revision = controller.lifecycle_impact(session_id=rig[4].id).revision
        await runtime.close_session(
            rig[4].id, expected_revision=revision, timeout_seconds=3
        )
        expected = {rig[4].id, saved_id}
        assert set(fenced) == expected
        assert set(cancelled) == expected
        assert set(drained) == expected
        assert rig[4].id in controller.fleet_wake._conversation_fences
        assert native.progress_owner_id(rig[4].id) is None
        for sender in (old_sender, saved_sender):
            with pytest.raises(MessageError, match="unavailable"):
                await asyncio.to_thread(sender.send, "late report")
        release.set()
        queue_result(rig, result_for(rig, old_chain))
        await asyncio.sleep(0.4)
        assert rig[5].payloads == []
        assert not controller.fleet_wake.has_pending(rig[4].id)
        assert not controller.fleet_wake.delivering_session_ids()
        assert [
            row[0]
            for row in rig[0]
            .get_connection()
            .execute(
                "SELECT message_id FROM fleet_progress_messages WHERE conversation_id=?",
                (saved_id,),
            )
        ] == [saved_report]
    finally:
        release.set()
        await close_rig(rig)


@private_profile_test
async def test_progress_prepare_cancelled_before_start_settles(tmp_path, request):
    import asyncio
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import asyncio
        from concurrent.futures import Future
        from contextlib import contextmanager
        from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

        class Executor:
            def submit(self, callback, *args, **kwargs):
                receipt = Future()
                receipt.cancel()
                return receipt

        class Native:
            persistence = None
            _progress_message_store = object()
            _stream_persistence_executor = Executor()

            @contextmanager
            def progress_owner_scope(self, session_id):
                yield "exact-owner"

            def prepare_progress_inbox(self, *args, **kwargs):
                raise AssertionError("cancelled preparation must never start")

        original = asyncio.create_task
        def cancel_physical(coroutine, **kwargs):
            task = original(coroutine, **kwargs)
            if coroutine.cr_code.co_name == "to_thread":
                task.cancel()
            return task

        async def main():
            asyncio.create_task = cancel_physical
            try:
                await ConsoleChatStore.prepare_progress_inbox_owned(Native(), "native")
            except asyncio.CancelledError:
                print("CANCELLED-BEFORE-START-SETTLED", flush=True)

        asyncio.run(main())
    """)
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    assert "CANCELLED-BEFORE-START-SETTLED" in result.stdout


@pytest.mark.parametrize("borrowed", [False, True])
@private_profile_test
async def test_progress_prepare_retires_only_its_new_worker_db_cache(
    tmp_path, request, borrowed
):
    from time import monotonic

    from tldw_chatbook.Agents.fleet_messages import MessageStore
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "prepare-cache.sqlite", "progress-test")
    persistence = ChatPersistenceService(db)
    saved_id = persistence.create_conversation(conversation_title="Saved")
    native = ConsoleChatStore(persistence=persistence)
    native.register_progress_message_store(MessageStore())
    session = native.restore_persisted_session(
        title="Saved",
        workspace_id=None,
        persisted_conversation_id=saved_id,
        all_nodes=[],
        prepare_progress=False,
    )
    executor = native._stream_persistence_executor
    participant = db._maintenance_participant
    main_connection = db.get_connection()
    try:
        existing = executor.submit(db.get_connection).result(5) if borrowed else None
        assert await native.prepare_progress_inbox_owned(session.id) is not None
        cached = executor.submit(lambda: getattr(db._local, "conn", None)).result(5)
        if borrowed:
            assert cached is existing
            assert (
                executor.submit(
                    lambda: existing.execute("SELECT 1").fetchone()[0]
                ).result(5)
                == 1
            )
            # Its prior caller owns retirement of this borrowed worker cache.
            executor.submit(db.close_connection).result(5)
        else:
            assert cached is None, (
                "finite progress preparation retained its new DB cache"
            )
            assert set(participant.connections) == {main_connection}
        assert main_connection.execute("SELECT 1").fetchone()[0] == 1
        await asyncio.to_thread(native.end_app_runtime)
        db.close()
        assert not participant.connections
        participant.close_admission()
        try:
            assert participant.drain(monotonic() + 0.05)
        finally:
            participant.resume()
    finally:
        if not native._stream_persistence_executor_closed:
            executor.submit(db.close_connection).result(5)
            await asyncio.to_thread(native.end_app_runtime)
        db.close()
