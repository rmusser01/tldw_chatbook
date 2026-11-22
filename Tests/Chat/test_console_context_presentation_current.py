"""Current memory skipping and exact finite context-display owners."""

import asyncio
import sys
import threading

import pytest

from Tests.Chat.test_console_memory_presence import _reset
from Tests.UI.test_console_shared_context_policy import _screen

pytestmark = pytest.mark.bootstrap_profile


def _memory_owner(store, controller):
    session = store.sessions()[0]
    row = store.messages_for_session(session.id)[0]
    controller._context_repository.insert_memory_selection(
        _reset(session.persisted_conversation_id, row.persisted_message_id)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("memory", [False, True])
async def test_display_preserves_actual_no_memory_lineage_skip(tmp_path, memory):
    database, store, controller, _screen_owner, tasks = _screen(tmp_path)
    if memory:
        _memory_owner(store, controller)
    code = controller._durable_context_snapshots.__func__.__code__
    reads = []
    main = threading.get_ident()
    previous, previous_thread = sys.getprofile(), threading.getprofile()

    def observe(frame, event, _argument):
        if event == "call" and frame.f_code is code:
            reads.append(threading.get_ident())

    threading.setprofile_all_threads(observe)
    try:
        shown = await controller.context_control_presentation_inputs("session-1")
        assert len(reads) == int(memory)
        assert all(thread != main for thread in reads)
        assert shown == controller.context_control_inputs("session-1")
        assert not controller._preparation_reads
    finally:
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous)
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("all_tasks", [False, True])
async def test_context_display_retains_exact_native_handle_through_cancel(tmp_path, all_tasks):
    database, store, controller, _screen_owner, tasks = _screen(tmp_path)
    _memory_owner(store, controller)
    code = store.persistence.get_message_versions.__func__.__code__
    entered, release, retired = (threading.Event() for _ in range(3))
    previous, previous_thread = sys.getprofile(), threading.getprofile()
    existing = asyncio.all_tasks()
    baseline = database._connection_quiescence.connection_count()

    def observe(frame, event, _argument):
        if frame.f_code is code and event == "return":
            # The original version query has acquired its real worker handle;
            # the enclosing finite callback still owns it until this returns.
            entered.set()
            assert release.wait(5)
            retired.set()

    threading.setprofile_all_threads(observe)
    task = asyncio.create_task(controller.context_control_presentation_inputs("session-1"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert controller._preparation_reads
        assert database._connection_quiescence.connection_count() == baseline + 1
        for _ in range(3):
            selected = asyncio.all_tasks() - existing if all_tasks else [task]
            for pending in selected:
                pending.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not retired.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert retired.is_set() and not controller._preparation_reads
        assert database._connection_quiescence.connection_count() == baseline
    finally:
        release.set()
        await asyncio.gather(task, *tasks, return_exceptions=True)
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous)
        database.close()


@pytest.mark.asyncio
async def test_session_close_drains_context_display_and_refuses_a_new_read(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Chat import console_preparation_reads
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    database, store, controller, _screen_owner, tasks = _screen(tmp_path)
    _memory_owner(store, controller)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    session_id = "session-1"
    code = store.persistence.get_message_versions.__func__.__code__
    entered, drain_entered = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    previous, previous_thread = sys.getprofile(), threading.getprofile()
    original_drain = console_preparation_reads.drain_preparation_reads

    async def observe_drain(reads, selected_session=None):
        if reads is runtime._preparation_reads and selected_session == session_id:
            drain_entered.set()
        return await original_drain(reads, selected_session)

    def hold_read(frame, event, _argument):
        if frame.f_code is code and event == "return":
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(20)

    monkeypatch.setattr(
        console_preparation_reads, "drain_preparation_reads", observe_drain
    )
    threading.setprofile_all_threads(hold_read)
    presentation = asyncio.create_task(
        controller.context_control_presentation_inputs(session_id)
    )
    close = None
    try:
        async with asyncio.timeout(15):
            await entered.wait()
            (read,) = controller._preparation_reads
            assert read in runtime._preparation_reads
            close = asyncio.create_task(
                runtime.close_session(
                    session_id,
                    expected_revision=controller.lifecycle_impact(
                        session_id=session_id
                    ).revision,
                    timeout_seconds=0.01,
                )
            )
            await drain_entered.wait()
            assert not close.done() and not read.retired.done()
            assert any(session.id == session_id for session in store.sessions())
            with pytest.raises(
                RuntimeError, match="Console context display owner changed"
            ):
                await controller.context_control_presentation_inputs(session_id)
            assert controller._preparation_reads == {read}
            release.set()
            await close
            with pytest.raises(asyncio.CancelledError):
                await presentation
        assert all(session.id != session_id for session in store.sessions())
        assert not controller._preparation_reads and not runtime._preparation_reads
    finally:
        release.set()
        await asyncio.gather(presentation, *tasks, return_exceptions=True)
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous)
        await runtime.dispose()
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["snapshot", "attempt_snapshot"])
async def test_session_close_drains_a_prompt_mixin_read_and_refuses_a_new_one(
    tmp_path, monkeypatch, kind
):
    """TASK-42342: a controller-owned read the bound runtime never observed."""
    from Tests.Chat.test_console_first_send_atomicity import _controller
    from tldw_chatbook.Backup_Recovery import runtime_producer_lifetime
    from tldw_chatbook.Chat import console_preparation_reads
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_prompt_mixin_service import (
        ConsolePromptMixinService,
    )
    from tldw_chatbook.Chat.console_prompt_mixins import (
        MixinPreferences,
        MixinSelection,
        MixinSnapshotUnavailableError,
        MixinSourceChangedError,
        MixinSourceRef,
    )
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.Prompt_Management.prompt_scope_service import (
        LocalPromptService,
        PromptScopeService,
    )

    database, store, controller, _gateway = _controller(tmp_path)
    prompts = PromptsDatabase(tmp_path / "prompts.db", client_id="task-42342")
    scope = PromptScopeService(LocalPromptService(prompts), None)
    # ConsoleChatController.prompt_mixin_service()'s composition, minus an app.
    service = ConsolePromptMixinService(
        store=store,
        prompt_scope_service=scope,
        persistence=store.persistence,
        preparation_owner=controller,
    )
    session_id = "session-1"
    saved = await scope.save_prompt(
        mode="local",
        name="Held",
        artifact_type="mixin",
        prompt_format="structured",
        prompt_schema_version=3,
        prompt_definition={
            "kind": "text_mixin",
            "schema_version": 3,
            "body": "literal",
            "placement": "before",
        },
    )
    source = MixinSourceRef(
        scope.mixin_source_identity(mode="local")[0], "local", saved["uuid"]
    )
    service.replace_preferences(
        session_id,
        expected_revision=0,
        value=MixinPreferences(
            entries=(MixinSelection(source, True, "Held", "before"),)
        ),
    )
    store.append_message(
        session_id, role=ConsoleMessageRole.USER, content="hello", persist=True
    )
    reply = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="hi", persist=True
    )
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    entered, drain_entered = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    calls, registered, held = [], [], []
    if kind == "snapshot":
        target, name = prompts, "fetch_mixin_snapshot"
        refusal = MixinSourceChangedError

        def read_mixins():
            revision = service.preferences(session_id).revision
            return service.capture(session_id, expected_revision=revision)

    else:
        target, name = ChatPersistenceService, "get_conversation_mixin_attempt"
        refusal = MixinSnapshotUnavailableError

        def read_mixins():
            return service.retry_snapshot(session_id, assistant_message_id=reply.id)

    original = getattr(target, name)

    def hold_first(*args, **kwargs):
        # The native callback's own database read, on its worker thread.
        calls.append(threading.get_ident())
        if len(calls) == 1:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(20)
        return original(*args, **kwargs)

    original_observe = runtime_producer_lifetime._observe_locked

    def record_registration(read, observer):
        if observer is controller._preparation_reads and read.session_id == session_id:
            registered.append(read)
        return original_observe(read, observer)

    original_drain = console_preparation_reads.drain_preparation_reads

    async def observe_drain(reads, selected_session=None):
        if selected_session == session_id and any(read in reads for read in held):
            drain_entered.set()
        return await original_drain(reads, selected_session)

    monkeypatch.setattr(target, name, hold_first)
    monkeypatch.setattr(runtime_producer_lifetime, "_observe_locked", record_registration)
    monkeypatch.setattr(
        console_preparation_reads, "drain_preparation_reads", observe_drain
    )
    issuing = asyncio.create_task(read_mixins())
    close = None
    try:
        async with asyncio.timeout(15):
            await entered.wait()
            (read,) = controller._preparation_reads
            # The runtime bound before this read existed, so it never saw it.
            assert read not in runtime._preparation_reads
            held.append(read)
            close = asyncio.create_task(
                runtime.close_session(
                    session_id,
                    expected_revision=controller.lifecycle_impact(
                        session_id=session_id
                    ).revision,
                    timeout_seconds=0.01,
                )
            )
            draining = asyncio.create_task(drain_entered.wait())
            await asyncio.wait({close, draining}, return_when=asyncio.FIRST_COMPLETED)
            draining.cancel()
            assert not close.done(), close.cancelled() or close.exception()
            assert issuing.cancelling() and not read.retired.done()
            assert any(session.id == session_id for session in store.sessions())
            with pytest.raises(refusal):
                await read_mixins()
            assert registered == [read] and len(calls) == 1
            release.set()
            await close
            with pytest.raises(asyncio.CancelledError):
                await issuing
        assert all(session.id != session_id for session in store.sessions())
        assert not controller._preparation_reads and not runtime._preparation_reads
    finally:
        release.set()
        await asyncio.gather(issuing, return_exceptions=True)
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        await runtime.dispose()
        prompts.close_connection()
        database.close()


@pytest.mark.asyncio
async def test_custom_context_reader_keeps_issuing_thread_and_live_result(
    tmp_path, monkeypatch
):
    database, _store, controller, _screen_owner, tasks = _screen(tmp_path)
    calls = []
    result = object()

    def custom(session_id):
        calls.append((threading.get_ident(), session_id))
        return result

    monkeypatch.setattr(controller, "context_control_inputs", custom)
    try:
        assert await controller.context_control_presentation_inputs("session-1") is result
        assert calls == [(threading.get_ident(), "session-1")]
        assert not controller._preparation_reads
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["warm", "queued"])
@pytest.mark.parametrize("closed", ["controller", "screen", "runtime"])
async def test_no_config_display_never_calls_custom_reader_after_closure(
    tmp_path, monkeypatch, entry, closed
):
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleContextReadSnapshot,
    )

    database, _store, controller, screen, tasks = _screen(tmp_path)
    runtime = ConsoleRuntime(app=None)
    screen._console_runtime_fallback = runtime
    screen.app_instance.app_config = {}
    snapshot = ConsoleContextReadSnapshot.for_screen(screen, max_age=1)
    assert snapshot.config_projection is None
    result = controller.context_control_inputs("session-1")
    calls = []

    def custom(session_id):
        calls.append((threading.get_ident(), session_id))
        return result

    monkeypatch.setattr(controller, "context_control_inputs", custom)
    locked = False
    try:
        assert await snapshot.warm(controller, "session-1")
        assert calls == [(threading.get_ident(), "session-1")]
        snapshot.at -= 2
        if entry == "queued":
            await snapshot.lock.acquire()
            locked = True
            assert snapshot.inputs(controller, "session-1") is result
            assert len(tasks) == 1
            await asyncio.sleep(0)
        if closed == "controller":
            await controller.shutdown()
            assert controller._disposed
        elif closed == "runtime":
            await runtime.dispose()
            assert runtime._disposed
        else:
            screen._closing = True
        if locked:
            snapshot.lock.release()
            locked = False
            await asyncio.gather(*tasks)
        else:
            assert not await snapshot.warm(controller, "session-1")
        assert calls == [(threading.get_ident(), "session-1")]
        # Direct explicit action readers retain their existing callable ABI.
        assert controller.context_control_inputs("session-1") is result
        assert len(calls) == 2
    finally:
        if locked:
            snapshot.lock.release()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runtime.dispose()
        await controller.shutdown()
        database.close()
