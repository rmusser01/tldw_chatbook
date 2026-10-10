"""Native helper controls; original fleet effects need their separate witness."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


pytestmark = pytest.mark.bootstrap_profile


_SCRIPT = r"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import sqlite3
import sys
import threading
from types import SimpleNamespace

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, outcome = sys.argv[1:]
assert route == 'prepared_close_owner'


def closed(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError as error:
        assert 'closed' in str(error).lower()
        return True
    return False


def main():
    root = Path(os.environ['XDG_DATA_HOME']).absolute()
    selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    selector.write_text('[general]\nusers_name="qualifier"\n[paths]\ndata_dir="'
                        + root.as_posix() + '"\n', encoding='utf-8')
    selector.chmod(0o600)
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from Tests.UI import _prepared_close_owned_resources as module

    paths = tuple(Path(item.__file__).absolute() for item in (
        module, sys.modules[ConsoleRuntime.__module__], sys.modules[AgentRunsDB.__module__]
    ))
    hashes = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)
    anchors = (ConsoleRuntime.dispose, ConsoleRuntime.ensure_activity_receipt_service,
               ConsoleRuntime._canvas_enabled, AgentRunsDB.get_run,
               module.PreparedCloseOwnedResources.request_connection_scope)
    codes = tuple(function.__code__ for function in anchors)
    namespaces = tuple(function.__globals__ for function in anchors)

    async def run():
        directory = config.get_user_data_dir() / 'prepared-close-control'
        directory.mkdir(mode=0o700)
        characters = CharactersRAGDB(directory / 'chats.sqlite', client_id='prepared-control')
        app = SimpleNamespace(chachanotes_db=characters, app_config={},
                              conversation_local_marks_service=None)
        runtime = ConsoleRuntime(app)
        app.console_runtime = runtime
        watcher = runtime._canvas_policy_watch_task
        assert watcher is not None
        resources = None
        outside = None
        observed = []
        worker_done = threading.Event()
        entered = threading.Event()
        release = threading.Event()
        invalid = []
        worker_thread = None
        owner_thread = threading.current_thread()
        tool = None
        held = False
        result_snapshot = None

        if outcome == 'outside_initial':
            outside = AgentRunsDB(directory.parent / 'borrowed-agent-runs.sqlite')
            outside_connection = outside._held_connection()
            runtime._agent_runs_db = outside
            try:
                module.PreparedCloseOwnedResources(app, directory, characters)
            except RuntimeError as error:
                assert str(error) == 'prepared_close_runtime_not_fresh_owned'
            else:
                raise AssertionError('borrowed prebuilt runtime was claimed')
            assert not runtime._disposed and not closed(outside_connection)
            await runtime.dispose()  # Test caller owns this outside cleanup.
            outside.close()
            characters.close_connection()
            return

        resources = module.PreparedCloseOwnedResources(app, directory, characters)
        assert runtime._agent_runs_db is None, 'the original route must remain lazy'
        assert runtime.ensure_activity_receipt_service() is not None
        runs = runtime._agent_runs_db
        assert type(runs) is AgentRunsDB
        assert Path(runs.db_path).absolute() == directory / 'agent_runs.db'
        resources.adopt_runtime_runs()
        creator_connection = runs._held_connection()
        run_id = runs.create_run(conversation_id='prepared-conversation', agent_kind='primary')
        get_code = AgentRunsDB.get_run.__code__
        policy_code = ConsoleRuntime._canvas_enabled.__code__

        def line(code, lineno):
            nonlocal held
            frame = sys._getframe(1)
            if code is get_code:
                if frame.f_locals.get('self') is not runs or 'conn' not in frame.f_locals:
                    return
                connection = frame.f_locals['conn']
                if observed:
                    return
                assert threading.current_thread() is worker_thread
                participant = runs._maintenance_participant
                with storage._lock:
                    lease = participant.connections[connection]
                    assert lease in storage._live_leases and lease.resource_thread is worker_thread
                observed.append((connection, lease, participant))
                if outcome == 'cancel_request':
                    held = True
                    entered.set()
                    if not release.wait(10):
                        invalid.append('held_request_release_timeout')
            elif (outcome == 'cancel_dispose' and code is policy_code
                  and frame.f_locals.get('self') is runtime
                  and threading.current_thread() is not owner_thread and not held):
                held = True
                entered.set()
                if not release.wait(10):
                    invalid.append('held_policy_release_timeout')

        monitor = sys.monitoring
        tool = next(slot for slot in range(6) if monitor.get_tool(slot) is None)
        monitor.use_tool_id(tool, 'prepared-close-native-owner-controls')
        monitor.register_callback(tool, monitor.events.LINE, line)
        monitor.set_local_events(tool, get_code, monitor.events.LINE)
        if outcome == 'cancel_dispose':
            monitor.set_local_events(tool, policy_code, monitor.events.LINE)
        assert monitor.get_events(tool) == 0

        async def wait_entered():
            deadline = asyncio.get_running_loop().time() + 10
            while not entered.is_set():
                assert asyncio.get_running_loop().time() < deadline, 'original held effect not reached'
                await asyncio.sleep(.01)

        def worker():
            nonlocal worker_thread, result_snapshot
            worker_thread = threading.current_thread()
            borrowed = None
            try:
                if outcome == 'borrowed_worker':
                    borrowed = runs._held_connection()
                    borrowed.execute('BEGIN')
                with resources.request_connection_scope(runs):
                    row = runs.get_run(run_id)
                    assert row['id'] == run_id
                connection, lease, participant = observed[0]
                with storage._lock:
                    live = lease in storage._live_leases
                    registered = connection in participant.connections
                result_snapshot = (closed(connection), live, registered)
                if borrowed is not None:
                    assert runs._held_connection() is borrowed
                    assert sqlite3.Connection.in_transaction.__get__(borrowed)
            finally:
                if borrowed is not None and not closed(borrowed):
                    borrowed.rollback()
                runs.close()  # Test-owned caller cleanup on the same worker.
                worker_done.set()

        try:
            if outcome == 'cancel_dispose':
                await wait_entered()
                caller = asyncio.create_task(resources.dispose_runtime())
                await asyncio.sleep(0)
                assert resources.dispose_task is not None
                caller.cancel()
                await asyncio.sleep(0)
                assert not resources.dispose_task.done()
                assert not closed(creator_connection), 'creator close raced the retained policy read'
                release.set()
                try:
                    await caller
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError('awaiting disposal caller was not cancelled')
            else:
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix='prepared-close-owned') as executor:
                    native = executor.submit(worker)
                    if outcome == 'cancel_request':
                        waiter_entered = asyncio.Event()
                        async def awaiting_request():
                            waiter_entered.set()
                            return await asyncio.wrap_future(native)
                        waiter = asyncio.create_task(awaiting_request())
                        await waiter_entered.wait()
                        await wait_entered()
                        waiter.cancel()
                        try:
                            await waiter
                        except asyncio.CancelledError:
                            pass
                        else:
                            raise AssertionError('request awaiting task was not cancelled')
                        assert native.running() and not native.done()
                        assert not closed(observed[0][0])
                        with storage._lock:
                            assert observed[0][1] in storage._live_leases
                        release.set()
                    await asyncio.wait_for(asyncio.wrap_future(native), 10)
                assert worker_done.is_set() and not worker_thread.is_alive()
                assert result_snapshot == ((False, True, True) if outcome == 'borrowed_worker'
                                           else (True, False, False)), result_snapshot
                await resources.dispose_runtime()
            assert not invalid, invalid
            assert resources.runtime_terminal and runtime._disposed
            assert watcher.done()
            assert runtime._canvas_policy_read_task is None
            assert runtime.ensure_activity_receipt_service() is None
            assert closed(creator_connection)
            resources.close_creators()
            resources.close_creators()  # Successful terminal cleanup is idempotent.
            from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
            try:
                runs.get_run(run_id)
            except RecoveryRequired:
                pass
            else:
                raise AssertionError('terminal exact owned database reopened')
            assert closed(creator_connection)
        finally:
            release.set()
            monitor.set_local_events(tool, get_code, 0)
            if outcome == 'cancel_dispose':
                monitor.set_local_events(tool, policy_code, 0)
            monitor.register_callback(tool, monitor.events.LINE, None)
            monitor.free_tool_id(tool)
            # The exact runtime remains retained and can retry its original disposal.
            if resources.dispose_task is not None:
                await asyncio.shield(resources.dispose_task)
            else:
                await resources.dispose_runtime()
            resources.close_creators()
        assert monitor.get_tool(tool) is None

    asyncio.run(run())
    assert tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths) == hashes
    assert tuple(function.__code__ for function in anchors) == codes
    assert all(function.__globals__ is namespace for function, namespace in zip(anchors, namespaces))
    print('retired and reopened')


with user_fixture_default_owner():
    main()
"""


@pytest.mark.parametrize(
    "outcome",
    [
        "new_worker",
        "borrowed_worker",
        "outside_initial",
        "cancel_request",
        "cancel_dispose",
    ],
)
def test_prepared_close_owner_native_control(tmp_path, outcome):
    _run(tmp_path, "prepared_close_owner", outcome, script=_SCRIPT)
