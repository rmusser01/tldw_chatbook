"""Original attention reads remain finite across acknowledgement and cancellation."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio
import sys
import threading
from types import SimpleNamespace
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

network_guard.install()
real_profile_guard.install()

async def until(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(.005)

async def exercise():
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.conversation_local_marks_service import ConversationLocalMarksService
    from tldw_chatbook.Chat import console_preparation_reads as reads

    database = config.get_chachanotes_db_lazy()
    marks = ConversationLocalMarksService(database)
    conversation = database.add_conversation({'title': 'attention control'})
    receipt = '11111111-1111-4111-8111-111111111111'
    pair = (conversation, receipt)
    main = threading.get_ident()
    target = ConversationLocalMarksService.list_console_unseen_marks.__code__
    tool_id = 4
    sys.monitoring.use_tool_id(tool_id, 'attention-native-control')
    try:
        for mode in ('publish', 'ack', 'cancel', 'source', 'authority', 'reentrant'):
            marks.set_mark(conversation, marks.console_unseen_mark_type(receipt))
            updates = []
            def publish(value):
                updates.append(value)
                if mode == 'reentrant' and len(updates) == 1:
                    runtime.request_console_attention_refresh(force_projection=True)
            app = SimpleNamespace(conversation_local_marks_service=marks,
                                  set_console_attention_projection=publish)
            runtime = ConsoleRuntime(app)
            entered, release = threading.Event(), threading.Event()
            held = []
            def hold(code, offset, result):
                if entered.is_set():
                    return
                assert threading.get_ident() != main, 'attention query ran on UI loop'
                binding = reads._source_binding.get()
                assert binding is not None, 'attention query has no finite read owner'
                held.append(binding.read)
                entered.set()
                assert release.wait(10), 'attention watchdog expired'
            sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_RETURN, hold)
            sys.monitoring.set_local_events(tool_id, target, sys.monitoring.events.PY_RETURN)
            try:
                runtime.request_console_attention_refresh()
                await until(entered.is_set)
                task = runtime._attention_refresh_task
                assert held[0] in runtime._preparation_reads
                # A held read must not keep the operation lock away from the UI.
                assert runtime._attention_operation_lock.acquire(blocking=False)
                runtime._attention_operation_lock.release()
                if mode == 'publish':
                    for _ in range(20):
                        runtime.request_console_attention_refresh()
                        assert runtime._attention_refresh_task is task
                elif mode == 'ack':
                    assert runtime.acknowledge_rendered_terminal_receipts((pair,)) == (receipt,)
                elif mode == 'cancel':
                    task.cancel()
                    await asyncio.sleep(.01)
                    task.cancel()
                    await asyncio.sleep(.01)
                    assert not task.done() and held[0] in runtime._preparation_reads
                elif mode == 'source':
                    app.conversation_local_marks_service = ConversationLocalMarksService(database)
                elif mode == 'authority':
                    runtime.authority_token = 'replaced'
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                assert not runtime._preparation_reads
                if mode in ('publish', 'reentrant'):
                    assert runtime.console_needs_attention and updates[-1] is True
                    if mode == 'reentrant':
                        assert updates == [True, True]
                elif mode == 'ack':
                    assert not runtime.console_needs_attention
                    assert runtime._last_known_terminal_marks == ()
                else:
                    assert not updates and runtime._last_known_terminal_marks is None
                assert database.registered_connection_count() <= 1
            finally:
                release.set()
                sys.monitoring.set_local_events(tool_id, target, 0)
                sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_RETURN, None)
            marks.acknowledge_console_unseen(*pair)
        # A custom service retains its synchronous caller and result contract.
        calls = []
        custom = SimpleNamespace(list_console_unseen_marks=lambda: calls.append(threading.get_ident()) or ())
        runtime = ConsoleRuntime(SimpleNamespace(conversation_local_marks_service=custom))
        assert runtime.request_console_attention_refresh() is False
        assert calls == [main]
        assert runtime._attention_refresh_task is None
        # Leaf transaction callbacks are customizable too: do not move them.
        from tldw_chatbook.DB.ChaChaNotes_DB import TransactionContextManager
        for name in ('_enter_transaction', '_exit_transaction'):
            original = getattr(TransactionContextManager, name)
            calls = []
            def callback(self, *args):
                calls.append(threading.get_ident())
                return original(self, *args)
            setattr(TransactionContextManager, name, callback)
            try:
                runtime = ConsoleRuntime(SimpleNamespace(conversation_local_marks_service=marks))
                assert runtime.request_console_attention_refresh() is False
                assert calls == [main]
                assert runtime._attention_refresh_task is None
            finally:
                setattr(TransactionContextManager, name, original)
        # Disposal fences a queued coroutine before it issues a native read.
        runtime = ConsoleRuntime(SimpleNamespace(conversation_local_marks_service=marks))
        runtime.request_console_attention_refresh()
        task = runtime._attention_refresh_task
        await runtime.dispose()
        await task
        assert not runtime._preparation_reads and runtime._last_known_terminal_marks is None
    finally:
        sys.monitoring.free_tool_id(tool_id)
        database.close_connection()
    assert not network_guard.blocked_attempts()
    assert not real_profile_guard.take_violations()
    print('retired and reopened')

with user_fixture_default_owner():
    asyncio.run(exercise())
"""


def test_native_attention_read_ownership_and_stale_results(tmp_path):
    launch = (
        "from Tests.Performance.test_console_attention_owned import _SCRIPT; "
        "exec(compile(_SCRIPT, '<attention-control>', 'exec'))"
    )
    _run(tmp_path, "attention", "owned", script=launch, timeout=120)
