"""Original Agent caller and exact post-use native fixture retirement controls."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
from types import CodeType, SimpleNamespace

from Tests import network_guard, real_profile_guard
network_guard.install()
real_profile_guard.install()
from Tests.windows_private_fixture_runner import user_fixture_default_owner  # noqa: E402 - guards precede other imports.

route, outcome = sys.argv[1:]
assert route == 'original_agent_fixture'
assert outcome in {'mounted', 'mounted_red', 'replaced_runtime', 'sync_memoized', 'sync_absent',
                   'borrowed_database', 'borrowed_runtime', 'foreign_retry',
                   'body_error', 'repeated_cancel'}

def control():
    root = Path(os.environ['XDG_DATA_HOME']).absolute().parent
    selected = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    selected.write_text('[general]\nusers_name="agent-fixture"\n[paths]\ndata_dir="'
                        + (root / 'data').as_posix() + '"\n', encoding='utf-8')
    selected.chmod(0o600)
    os.environ['TLDW_TEST_CONFIG_ROOT'] = str(root)
    from loguru import logger
    logger.remove()
    import pytest
    import Tests.conftest  # noqa: F401 - original isolated profile and fixture setup.
    from Tests.UI import test_console_agent_controller as helpers
    from Tests.UI._agent_fixture_owners import AgentFixtureOwners, finish_owner_retirement
    from Tests.UI import app_factory
    from tldw_chatbook import app as app_module
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Backup_Recovery import participants, storage_admission as storage
    from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver

    facts = {'outcome': outcome, 'original_module_sha256': hashlib.sha256(Path(helpers.__file__).read_bytes()).hexdigest(),
             'original_assertions_called': False, 'guards_and_global_drain_unchanged': True}
    monkeypatch = pytest.MonkeyPatch()
    dependent = helpers._real_fleet_recovery_database.__wrapped__(
        monkeypatch, root / 'data', SimpleNamespace(module=helpers))
    next(dependent)
    owner = None
    other = None
    borrowed_app = None
    release = threading.Event()
    foreign = None
    monitor = None

    def closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError as error:
            return 'closed' in str(error).lower()
        return False

    def retired(database, connection):
        participant = participants._repository_participant(database)
        with storage._lock:
            return (closed(connection) and getattr(database._thread_local, 'conn', None) is None
                    and not participant.connections
                    and not any(getattr(lease, 'resource_participant', None) is participant for lease in storage._live_leases))

    async def initialize():
        result = AgentFixtureOwners(helpers)
        result.install_births()
        return result

    async def exercise():
        nonlocal owner, other, foreign, monitor, borrowed_app
        if outcome == 'borrowed_runtime':
            borrowed_app = helpers._build_test_app()
            borrowed_runtime = borrowed_app.console_runtime
            stock = app_module.ConsoleRuntime
            calls = []
            def borrowed_constructor(app):
                calls.append(app)
                return borrowed_runtime
            app_module.ConsoleRuntime = borrowed_constructor
            try:
                owner = await initialize()
                replacement_app = helpers._build_test_app()
                assert replacement_app.console_runtime is borrowed_runtime
                assert len(calls) == 1
                await owner.retire()
                assert not owner.apps and not borrowed_runtime._disposed
                assert borrowed_runtime.ensure_chat_store() is not None
                facts['borrowed_runtime_preserved'] = True
            finally:
                app_module.ConsoleRuntime = stock
                await ConsoleRuntime.dispose(borrowed_runtime)
            return
        if outcome == 'borrowed_database':
            other = AgentRunsDB(root / 'data' / 'caller-owned.sqlite', client_id='caller')
            other_connection = other._held_connection()
            other_connection.execute('BEGIN')
            stock = helpers.AgentRunsDB
            calls = []
            def borrowed_constructor(*args, **kwargs):
                calls.append((args, kwargs))
                return other
            helpers.AgentRunsDB = borrowed_constructor
            try:
                owner = await initialize()
                bridge = helpers._bridge_over(root / 'data' / 'ignored.sqlite')
                assert bridge._db is other and len(calls) == 1
                await owner.retire()
                assert not owner.databases
                assert other._thread_local.conn is other_connection
                assert other_connection.in_transaction and other_connection.execute('SELECT 1').fetchone()[0] == 1
                facts['borrowed_transaction_preserved'] = True
            finally:
                helpers.AgentRunsDB = stock
                other_connection.rollback()
                other.close()
            return
        if outcome == 'replaced_runtime':
            borrowed_app = helpers._build_test_app()
        owner = await initialize()
        if outcome in {'mounted', 'mounted_red', 'repeated_cancel', 'replaced_runtime'}:
            # This is the exact installed original caller, including all actions,
            # waits, bounds and assertions; the module fixture is not invoked here.
            original = helpers.test_persisted_run_state_reaches_the_mounted_agent_rail_statics
            pin = OriginalStorageUnitObserver({}, False, lambda _name: None)
            pin._pin(original)
            await original(root / 'data')
            facts['original_assertions_called'] = True
            assert len(owner.apps) == 1 and len(owner.hosts) == 1 and len(owner.databases) == 1
            app, runtime, birth_loop = owner.apps[0]
            database = owner.databases[0][0]
            connection = database._thread_local.conn
            participant = participants._repository_participant(database)
            lease = participant.connections[connection]
            before = {'native_live': not closed(connection), 'creator_cache_exact': database._thread_local.conn is connection,
                      'lease_live': lease in storage._live_leases, 'runtime_live': not runtime._disposed,
                      'host_tasks_captured': len(owner.workers)}
            assert before['native_live'] and before['creator_cache_exact'] and before['lease_live']
            assert birth_loop is asyncio.get_running_loop()
            watcher = runtime._canvas_policy_watch_task
            assert watcher is not None
            facts['after_unchanged_original_use'] = before
            was_retired = retired(database, connection)
            if outcome == 'replaced_runtime':
                app.console_runtime = borrowed_app.console_runtime
            if outcome == 'repeated_cancel':
                lock_entered, callback_entered = threading.Event(), threading.Event()
                def hold_lock():
                    with runtime._activity_receipts_lock:
                        lock_entered.set()
                        assert release.wait(10)
                foreign = threading.Thread(target=hold_lock)
                foreign.start()
                assert await asyncio.to_thread(lock_entered.wait, 5)
                monitor = OriginalStorageUnitObserver({}, False, lambda _name: None)
                monitor._pin(ConsoleRuntime.dispose)
                nested = next(code for code in ConsoleRuntime.dispose.__code__.co_consts
                              if type(code) is CodeType and code.co_name == 'receipt_database_after_creation')
                def entered(code, offset):
                    frame = sys._getframe(1)
                    assert frame.f_code is nested and frame.f_locals['self'] is runtime
                    callback_entered.set()
                    del frame
                for tool in range(5, 0, -1):
                    if tool == sys.monitoring.DEBUGGER_ID:
                        continue
                    try:
                        sys.monitoring.use_tool_id(tool, 'original-fixture-dispose-custody')
                    except ValueError:
                        continue
                    monitor.tool = tool
                    break
                assert monitor.tool is not None
                event = sys.monitoring.events.PY_START
                assert sys.monitoring.register_callback(monitor.tool, event, entered) is None
                monitor.registered[event] = entered
                monitor.codes[nested] = 'original_dispose_callback'
                sys.monitoring.set_local_events(monitor.tool, nested, event)
                monitor.active = monitor.installed = True
                waiter = asyncio.Task(owner.retire())
                try:
                    assert await asyncio.to_thread(callback_entered.wait, 5)
                    for _ in range(3):
                        waiter.cancel()
                        await asyncio.sleep(0)
                        assert not waiter.done() and not owner.retirement_task.done()
                        assert not closed(connection) and lease in storage._live_leases
                    facts['original_callback_held_through_three_cancellations'] = True
                finally:
                    release.set()
                try:
                    await waiter
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError('original cancellation was lost')
                foreign.join(5)
                assert not foreign.is_alive()
                receipt = monitor.close()
                assert receipt['original_source_current'] and receipt['hooks_retired_before_inactive']
            else:
                await owner.retire()
            assert retired(database, connection)
            assert runtime._disposed and watcher.done()
            assert runtime._canvas_policy_watch_task is None and runtime._canvas_policy_read_task is None
            assert all(task.done() for _worker, _node, _work, task, _host in owner.workers)
            facts['after_local_owned_retirement'] = True
            if outcome == 'replaced_runtime':
                assert app.console_runtime is borrowed_app.console_runtime
                assert not borrowed_app.console_runtime._disposed
                assert borrowed_app.console_runtime.ensure_chat_store() is not None
                facts['replaced_runtime_remains_usable'] = True
                await ConsoleRuntime.dispose(borrowed_app.console_runtime)
            assert pin.close()['original_source_current']
            if outcome == 'mounted_red':
                assert was_retired, 'RED: original successful mounted caller retained its exact creator connection'
            return
        bridge = helpers._bridge_over(root / 'data' / 'manual-owned.sqlite')
        database, connection = bridge._db, bridge._db._thread_local.conn
        assert len(owner.databases) == 1
        if outcome == 'body_error':
            primary = ValueError('original body error control')
            with database.transaction():
                assert not await finish_owner_retirement(owner, primary)
                assert primary.__notes__ == ['Agent fixture cleanup could not prove owned retirement']
                assert not closed(connection) and connection.in_transaction
                assert connection.execute('SELECT 1').fetchone()[0] == 1
            facts['original_error_primary_and_active_transaction_preserved'] = True
            await owner.retire()
        elif outcome == 'foreign_retry':
            ready = threading.Event()
            foreign_handles = []
            def hold_foreign_cache():
                current = database._held_connection()
                foreign_handles.append(current)
                ready.set()
                try:
                    assert release.wait(10)
                finally:
                    database.close()
            foreign = threading.Thread(target=hold_foreign_cache)
            foreign.start()
            assert await asyncio.to_thread(ready.wait, 5)
            try:
                await owner.retire()
            except RuntimeError as error:
                assert str(error) == 'agent_fixture_database_still_active'
            else:
                raise AssertionError('foreign live cache was closed')
            assert not closed(connection) and not closed(foreign_handles[0])
            facts['foreign_refusal_preserved_both_native_handles'] = True
            release.set()
            foreign.join(5)
            assert not foreign.is_alive() and closed(foreign_handles[0])
            await owner.retire()
        assert retired(database, connection)
        facts['exact_creator_retired'] = True

    try:
        if outcome in {'sync_memoized', 'sync_absent'}:
            with asyncio.Runner() as runner:
                owner = runner.run(initialize())
                name = ('test_agent_bridge_is_built_from_the_sibling_run_store_and_memoized'
                        if outcome == 'sync_memoized' else 'test_agent_bridge_is_absent_without_a_durable_run_store')
                getattr(helpers, name)(root / 'data')
                facts['original_assertions_called'] = True
                assert len(owner.apps) == 1
                app, runtime, birth_loop = owner.apps[0]
                assert birth_loop is None and runtime._canvas_policy_watch_task is None
                captured = owner.runtime_databases[id(runtime)]
                connection = runtime._agent_runs_db._thread_local.conn
                assert not closed(connection)
                assert len(owner.notes_databases) == 1
                notes = owner.notes_databases[0]
                explicit_notes, notes_connection = notes[0], notes[5]
                assert explicit_notes is app.chachanotes_db and not closed(notes_connection)
                assert notes[6].is_registered(notes_connection)
                fleet_notes = dependent.gi_frame.f_locals['owned_resources'][0][0]
                assert fleet_notes is not explicit_notes
                fleet_connection = fleet_notes._local.conn
                assert not closed(fleet_connection)
                runner.run(owner.retire())
                assert runtime._disposed and retired(captured[0], connection)
                facts['original_sync_runtime_and_sibling_retired'] = True
                assert closed(notes_connection) and explicit_notes._local.conn is None
                assert not notes[6].is_registered(notes_connection) and not notes[3].connections
                assert not any(getattr(lease, 'resource_participant', None) is notes[3] for lease in storage._live_leases)
                assert not closed(fleet_connection) and fleet_connection.execute('SELECT 1').fetchone()[0] == 1
                facts['explicit_sync_attachment_retired_before_fleet_teardown'] = True
                facts['distinct_fleet_attachment_still_usable'] = True
        else:
            asyncio.run(exercise())
        assert owner.source_receipt['original_source_current']
        assert owner.source_receipt['hooks_retired_before_inactive']
        assert sys.monitoring.get_tool(owner.tool) is None
    finally:
        release.set()
        if foreign is not None:
            foreign.join(5)
            assert not foreign.is_alive()
        if monitor is not None and monitor.active:
            monitor.close()
        if owner is not None and owner.active:
            owner.close()
        try:
            next(dependent)
        except StopIteration:
            pass
        monkeypatch.undo()
        app_factory.drain_active_service_patches()
        app_factory.drain_created_dirs()
    facts_prefix = os.environ.get('TLDW_AGENT_FIXTURE_FACTS_PREFIX')
    if facts_prefix:
        facts_path = Path(facts_prefix + '.agent-fixture-' + outcome + '.json')
        with facts_path.open('x', encoding='utf-8') as output:
            output.write(json.dumps(facts, indent=2, sort_keys=True) + '\n')
    print(json.dumps(facts, sort_keys=True))
    print('retired and reopened')

with user_fixture_default_owner():
    control()
"""


@pytest.mark.parametrize(
    "outcome",
    [
        "mounted",
        "replaced_runtime",
        "sync_memoized",
        "sync_absent",
        "borrowed_database",
        "borrowed_runtime",
        "foreign_retry",
        "body_error",
        "repeated_cancel",
    ],
)
def test_original_agent_fixture_physically_retires_exact_owners(tmp_path, outcome):
    _run(tmp_path, "original_agent_fixture", outcome, script=_SCRIPT)
