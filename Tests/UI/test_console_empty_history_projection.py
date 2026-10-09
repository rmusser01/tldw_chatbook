"""Original full Agent rail preserves one owned empty-conversation history read."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""

import asyncio
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
from types import CodeType, SimpleNamespace

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger  # noqa: E402 - original network/profile guards run first.
logger.remove()
route, outcome = sys.argv[1:]
assert route == 'historical_empty_rail'
selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
root = Path(os.environ['XDG_DATA_HOME']).absolute()
selector.write_text('[general]\nusers_name="qualifier"\n[paths]\ndata_dir="'
                    + root.as_posix() + '"\n', encoding='utf-8')
selector.chmod(0o600)


def shape(code):
    return (
        code.co_name, code.co_qualname, code.co_firstlineno,
        code.co_code, code.co_exceptiontable, code.co_stacksize,
        code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
        code.co_nlocals, code.co_flags, code.co_names, code.co_varnames,
        code.co_freevars, code.co_cellvars,
        tuple(shape(value) if isinstance(value, CodeType) else value
              for value in code.co_consts),
    )


def nested(code, qualname):
    if code.co_qualname == qualname:
        return code
    for value in code.co_consts:
        if isinstance(value, CodeType):
            found = nested(value, qualname)
            if found is not None:
                return found
    return None


def closed(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError as error:
        assert 'closed' in str(error).lower()
        return True
    return False


async def wait_flag(event):
    deadline = asyncio.get_running_loop().time() + 10
    while not event.is_set():
        assert asyncio.get_running_loop().time() < deadline, 'original callback stage not reached'
        await asyncio.sleep(.01)


def main():
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook import config
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB import base_db
    from tldw_chatbook.Chat import console_agent_bridge as bridge_module
    from tldw_chatbook.UI.Console_Modules import agent as agent_module
    from Tests.UI import test_console_refresh_read_batching as helpers
    from tldw_chatbook.UI.Console_Modules import character as character_module
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession

    functions = (
        (agent_module, agent_module.ConsoleAgentController, '_load_historical_presentation'),
        (agent_module, agent_module.ConsoleAgentController, '_presentation_historical_snapshot'),
        (agent_module, agent_module.ConsoleAgentController, '_historical_presentation_key'),
        (agent_module, agent_module.ConsoleAgentController, '_console_agent_section_payload'),
        (agent_module, agent_module.ConsoleAgentController, '_console_agent_section_lines'),
        (agent_module, agent_module.ConsoleAgentController, '_console_agent_fleet_rows'),
        (agent_module, agent_module.ConsoleAgentController, '_console_agent_fleet_section_state'),
        (character_module, character_module.ConsoleCharacterController, '_current_console_rail_conversation_id'),
        (bridge_module, bridge_module.ConsoleAgentBridge, '_derive_historical_snapshot'),
        (base_db, base_db, 'run_owned_db_call'),
        (storage, storage, '_shutdown'),
        (storage, storage._Hold, '_run'),
        (helpers, helpers, '_agent'),
        (helpers, helpers, '_historical_bridge'),
    )
    anchors, sources = [], {}
    for module, owner, name in functions:
        function = inspect.getattr_static(owner, name)
        path = Path(module.__file__).absolute()
        assert path == Path(module.__spec__.origin).absolute()
        assert path.is_relative_to(Path.cwd().resolve())
        source = path.read_bytes()
        compiled = compile(source, str(path), 'exec', dont_inherit=True)
        assert shape(nested(compiled, function.__qualname__)) == shape(function.__code__)
        assert function.__globals__ is module.__dict__
        closure = function.__closure__
        type_params = function.__type_params__
        if module is base_db and name == 'run_owned_db_call':
            definition = next(node for node in ast.parse(source).body
                              if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
            assert tuple(node.name for node in definition.type_params) == ('_CallParameters', '_CallResult')
            assert tuple(value.__name__ for value in type_params) == ('_CallParameters', '_CallResult')
            assert function.__code__.co_freevars == ('_CallResult',)
            assert closure is not None and len(closure) == 1
            assert closure[0].cell_contents is type_params[1]
        else:
            assert closure is None and type_params == ()
        cells = tuple(cell.cell_contents for cell in closure or ())
        anchors.append((module, owner, name, function, function.__code__, function.__globals__,
                        function.__defaults__, function.__kwdefaults__, module.__spec__, module.__loader__,
                        closure, cells, type_params))
        sources[path] = source
    # Native ownership/guard modules stay installed byte for byte; no replacement
    # callback, getter, transaction, admission predicate or configuration guard.
    for name in ('tldw_chatbook.DB.AgentRuns_DB',
                 'tldw_chatbook.Backup_Recovery.participants',
                 'tldw_chatbook.Backup_Recovery.storage_admission'):
        module = sys.modules[name]
        sources[Path(module.__file__).absolute()] = Path(module.__file__).read_bytes()

    loader = agent_module.ConsoleAgentController._load_historical_presentation
    derive = bridge_module.ConsoleAgentBridge._derive_historical_snapshot
    load_code, derive_code = loader.__code__, derive.__code__
    tree = ast.parse(sources[Path(bridge_module.__file__).absolute()])
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'ConsoleAgentBridge')
    body = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name == '_derive_historical_snapshot')
    after_query = next(node.lineno for node in body.body if isinstance(node, ast.If)
                       and isinstance(node.test, ast.UnaryOp)
                       and isinstance(node.test.operand, ast.Name)
                       and node.test.operand.id == 'primary_records')
    load_tree = ast.parse(sources[Path(agent_module.__file__).absolute()])
    load_class = next(node for node in load_tree.body if isinstance(node, ast.ClassDef)
                      and node.name == 'ConsoleAgentController')
    load_body = next(node for node in load_class.body if isinstance(node, ast.AsyncFunctionDef)
                     and node.name == '_load_historical_presentation')
    shield_lines = sorted({node.lineno for node in ast.walk(load_body)
                           if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute)
                           and isinstance(node.func.value, ast.Name)
                           and node.func.value.id == 'asyncio' and node.func.attr == 'shield'})
    assert len(shield_lines) == 2
    drain_line = shield_lines[1]

    # tldw import enrolls this fresh interpreter until its original atexit path.
    # Capture that actual process-owned lease before creating any test database.
    with storage._lock:
        startup_entries = tuple(storage._startups.items())
        assert len(startup_entries) == 1
        assert set(storage._live_leases) == {lease for _, lease in startup_entries}
        startup_holds, startup_facts = [], []
        for startup_key, lease in startup_entries:
            assert startup_key[0] == os.getpid() and type(lease) is storage.StorageLease
            assert lease.resource_path is None and lease.resource_thread is None
            assert lease.resource_policy is None and not lease.resource_close_failed
            issued_key = lease._key
            hold = None if issued_key is None else storage._holds.get(issued_key)
            if issued_key is not None:
                assert issued_key[0] == os.getpid() and hold is not None
                assert type(hold) is storage._Hold and hold.ready.is_set()
                assert hold.error is None and not hold.stop.is_set() and hold.thread.is_alive()
                assert hold.count == 1 and hold.names
                assert hold.authority._observed_groups.get(hold.names)
                startup_holds.append((lease, issued_key, hold, hold.thread))
            startup_facts.append(dict(process_pid_matches=True, exact_registered_lease=True,
                                      key_none=issued_key is None, native_hold_issued=hold is not None,
                                      native_ready=False if hold is None else hold.ready.is_set(),
                                      namespace_count=0 if hold is None else len(hold.names),
                                      hold_thread_identity=None if hold is None else id(hold.thread),
                                      bootstrap_root_sha256=hashlib.sha256(os.fsencode(startup_key[1])).hexdigest()))
    hold_code = storage._Hold._run.__code__
    startup_returns = []
    startup_shutdown = None

    database = AgentRunsDB(config.get_user_data_dir() / 'historical-empty.sqlite',
                           client_id='historical-empty')
    bridge, child = helpers._historical_bridge(database)
    replacement_db = AgentRunsDB(config.get_user_data_dir() / 'historical-other.sqlite',
                                 client_id='historical-other')
    replacement_bridge, _replacement_child = helpers._historical_bridge(replacement_db)
    database.close()
    replacement_db.close()  # Creator seed caches retire before any observed worker.
    agent, tasks = helpers._agent(bridge)
    scope = {'session': ConsoleChatSession(id='empty-original')}
    original_owner = scope['session']
    screen = agent._screen
    screen._console_chat_store = SimpleNamespace(
        active_session_id=original_owner.id, sessions=lambda: (scope['session'],))
    async def no_ui_sync():
        return None
    character = character_module.ConsoleCharacterController(
        app_config_accessor=lambda: {},
        chat_store_accessor=lambda: screen._console_chat_store,
        active_native_session_accessor=lambda: scope['session'],
        current_conversation_id_accessor=lambda: None,
        character_db_accessor=lambda: None,
        ensure_chat_store=lambda: screen._console_chat_store,
        provider_readiness_config_accessor=lambda: {},
        default_session_settings=lambda: None,
        swap_session_character=lambda *args, **kwargs: False,
        sync_temporary_chip=lambda: None,
        sync_native_chat_ui=no_ui_sync,
        notify=lambda *args, **kwargs: None,
        actor_scope_accessor=lambda: None,
        manual_reaction_key=lambda actor: None,
        resolve_visual_identity=lambda *args: None,
        resolve_historical_visual_identity=lambda *args: None,
        ensure_console_image_view=lambda: (None, None),
        console_image_default_mode=lambda: None,
        is_mounted=lambda: False,
        render_character_avatar=lambda *args: None,
    )
    # Actual original UI accessor returns None for this real unpersisted session.
    agent._current_rail_conversation_id = character._current_console_rail_conversation_id
    agent._chat_controller_accessor = lambda: None
    agent._current_rail_state_accessor = lambda: SimpleNamespace(agent_open=False)
    assert character._current_console_rail_conversation_id() is None
    assert agent._current_console_rail_conversation_id() is None
    entered, release, returned, drain_entered = (threading.Event() for _ in range(4))
    held, inner_tasks, invalid, observed_states = [], [], [], []
    read_resources, read_returns = [], []

    def observe_line(code, line):
        frame = sys._getframe(1)
        if code is load_code and frame.f_locals.get('self') is agent:
            if line == drain_line:
                drain_entered.set()
            return
        if code is not derive_code or line != after_query:
            return
        observed_bridge = frame.f_locals.get('self')
        observed_database = frame.f_locals.get('database')
        if observed_bridge not in (bridge, replacement_bridge) or observed_database not in (database, replacement_db):
            return
        try:
            assert frame.f_code is derive_code
            assert (observed_bridge, observed_database) in ((bridge, database), (replacement_bridge, replacement_db))
            connection = getattr(observed_database._thread_local, 'conn', None)
            thread = threading.current_thread()
            assert isinstance(connection, sqlite3.Connection) and not closed(connection)
            participant = observed_database._maintenance_participant
            operation = getattr(storage._operation_local, 'operation', None)
            with storage._lock:
                lease = participant.connections.get(connection)
                assert lease is not None and lease in storage._live_leases
                assert lease.resource_thread is thread
                assert lease.resource_participant is participant
                assert operation in storage._operations and operation.participant is participant
            resource = dict(frame=frame, connection=connection, thread=thread,
                            participant=participant, lease=lease, operation=operation,
                            database=observed_database, conversation_id=frame.f_locals['conversation_id'])
            read_resources.append(resource)
            if not held:
                assert observed_database is database and resource['conversation_id'] == ''
                assert frame.f_locals['primary_records'] == []
                held.append(resource)
                entered.set()
                if not release.wait(10):
                    invalid.append('original_callback_release_timeout')
        except BaseException as error:
            invalid.append(type(error).__name__)
            entered.set()

    def observe_yield(code, offset, value):
        frame = sys._getframe(1)
        if code is not load_code or frame.f_locals.get('self') is not agent:
            return
        worker = frame.f_locals.get('worker')
        if worker is None or any(worker is item for item in inner_tasks):
            return
        if len(inner_tasks) >= 4:
            invalid.append('inner_task_capacity')
            return
        assert type(worker) is asyncio.Task
        inner_tasks.append(worker)

    def observe_return(code, offset, value):
        frame = sys._getframe(1)
        if code is hold_code:
            assert frame.f_code is hold_code and frame.f_globals is storage.__dict__
            for lease, issued_key, hold, owner_thread in startup_holds:
                if frame.f_locals.get('self') is hold:
                    assert threading.current_thread() is owner_thread and hold.stop.is_set()
                    assert hold.error is None
                    startup_returns.append(hold)
            return
        if code is derive_code and any(frame is item['frame'] for item in read_resources):
            read_returns.append(frame)
            if frame is held[0]['frame']:
                returned.set()

    tool = next(slot for slot in range(6) if sys.monitoring.get_tool(slot) is None)
    monitor = sys.monitoring
    monitor.use_tool_id(tool, 'historical-read-repeated-cancel')
    callbacks = ((monitor.events.LINE, observe_line),
                 (monitor.events.PY_YIELD, observe_yield),
                 (monitor.events.PY_RETURN, observe_return))
    for event, callback in callbacks:
        assert monitor.register_callback(tool, event, callback) is None
    masks = {derive_code: monitor.events.LINE | monitor.events.PY_RETURN,
             load_code: monitor.events.LINE | monitor.events.PY_YIELD,
             hold_code: monitor.events.PY_RETURN}
    for code, mask in masks.items():
        assert monitor.get_local_events(tool, code) == 0
        monitor.set_local_events(tool, code, mask)
    assert monitor.get_events(tool) == 0
    violations = []
    post_retirement = None

    def render():
        payload = agent._console_agent_section_payload()
        assert isinstance(payload, tuple) and len(payload) == 9
        if character._current_console_rail_conversation_id() is None:
            assert payload[2].rows == () and not payload[8]
        return payload

    def history_tasks():
        result = []
        for task in tasks:
            assert type(task) is asyncio.Task
            coroutine = task.get_coro()
            if coroutine.cr_code is load_code:
                result.append(task)
        return result

    def initial_state():
        actual = history_tasks()
        assert len(actual) == 1
        frame = actual[0].get_coro().cr_frame
        assert frame is not None and frame.f_code is load_code
        assert frame.f_globals is agent_module.__dict__ and frame.f_locals['self'] is agent
        state = frame.f_locals['state']
        assert isinstance(state, dict) and state['pending']
        assert state['key'][0] is bridge and state['key'][1] is database
        assert state['key'][7] == ''
        return actual[0], state

    async def settle_known():
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 10
        while any(not task.done() for task in tasks) or any(not task.done() for task in inner_tasks):
            assert loop.time() < deadline, 'known original historical tasks did not retire'
            await asyncio.sleep(.01)
        results = await asyncio.gather(*tasks, *inner_tasks, return_exceptions=True)
        assert not [value for value in results if isinstance(value, BaseException)
                    and not isinstance(value, asyncio.CancelledError)]

    def check_retired():
        assert read_resources and len(read_returns) == len(read_resources) and not invalid
        facts = []
        for item in read_resources:
            connection, participant = item['connection'], item['participant']
            with storage._lock:
                live = item['lease'] in storage._live_leases
                registered = connection in participant.connections
                operation_live = item['operation'] in storage._operations
            facts.append(dict(closed=closed(connection), lease_live=live,
                              registered=registered, operation_live=operation_live))
            assert closed(connection) and not live and not registered and not operation_live
        return facts

    async def exercise(executor):
        nonlocal post_retirement
        loop = asyncio.get_running_loop()
        loop.set_default_executor(executor)
        original = state = None
        try:
            if outcome == 'publication_only':
                assert agent._console_agent_section_lines() == ('Agent: idle', '', '')
            else:
                assert render()[0] == 'Agent: idle'
            original, state = initial_state()
            observed_states.append(dict(stage='first_original_render',
                                        state_same=agent._console_historical_read is state,
                                        workers=len(history_tasks())))
            await wait_flag(entered)
            assert held and not invalid and inner_tasks
            assert held[0]['thread'] is not threading.current_thread()
            assert not returned.is_set() and not closed(held[0]['connection'])
            with storage._lock:
                assert held[0]['lease'] in storage._live_leases
                assert held[0]['operation'] in storage._operations

            if outcome in ('full_payload_reuse', 'publication_only'):
                if agent._console_historical_read is not state:
                    violations.append('full_rail_cleared_matching_empty_pending_state')
                if outcome == 'full_payload_reuse':
                    render()
                    if len(history_tasks()) != 1 or agent._console_historical_read is not state:
                        violations.append('same_key_full_rail_rearmed_held_history_reader')
                release.set()
                await settle_known()
                if agent._console_historical_read is not state or state['value'] is None:
                    violations.append('normalized_empty_history_not_published')
                before = len(history_tasks())
                render()
                render()
                if len(history_tasks()) != before or agent._console_historical_read is not state:
                    violations.append('full_rail_did_not_reuse_completed_empty_history')
            elif outcome == 'cancel_double':
                for number in (1, 2):
                    assert original.cancel()
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    assert drain_entered.is_set()
                    with storage._lock:
                        assert held[0]['lease'] in storage._live_leases
                        assert held[0]['operation'] in storage._operations
                    assert not returned.is_set() and not closed(held[0]['connection'])
                    render()
                    observed_states.append(dict(stage='cancel', number=number,
                                                outer_done=original.done(), state_same=agent._console_historical_read is state,
                                                pending=state['pending'], workers=len(history_tasks())))
                    if original.done() or not state['pending'] or agent._console_historical_read is not state:
                        violations.append('empty_pending_owner_released_before_cancelled_native_retirement')
                    if len(history_tasks()) != 1:
                        violations.append('empty_rail_rearmed_during_cancellation_drain')
                release.set()
                await settle_known()
                assert original.cancelled() and not state['pending']
                assert state['value'] is None
                if not violations:
                    assert agent._console_historical_read is None
                    render()
                    await settle_known()
                    assert agent._console_historical_read is not state
                    assert agent._console_historical_read['value'] is not None
            else:
                before_key = state['key']
                if outcome == 'new_conversation':
                    original_owner.persisted_conversation_id = 'conv'
                    assert character._current_console_rail_conversation_id() == 'conv'
                    render()
                elif outcome == 'source_generation':
                    before_identity = config.current_config_identity()
                    assert config.save_setting_to_cli_config('general', 'users_name', 'generation-change')
                    assert config.current_config_identity() != before_identity
                    assert agent._historical_presentation_key(bridge, '') != before_key
                    assert agent._console_agent_fleet_rows() == ()
                    assert agent._console_historical_read is None
                    render()
                elif outcome == 'foreign_store':
                    scope['session'] = ConsoleChatSession(id='empty-replacement')
                    screen._console_chat_store = SimpleNamespace(
                        active_session_id=scope['session'].id, sessions=lambda: (scope['session'],))
                    assert agent._historical_presentation_key(bridge, '') != before_key
                    assert agent._console_agent_fleet_rows() == ()
                    assert agent._console_historical_read is None
                    render()
                elif outcome == 'foreign_bridge':
                    screen._console_runtime().agent_bridge = replacement_bridge
                    assert agent._console_agent_fleet_rows() == ()
                    assert agent._console_historical_read is None
                    render()
                else:
                    raise AssertionError('unknown route')
                current = agent._console_historical_read
                if current is None or current is state:
                    violations.append('new_matching_history_scope_not_retained')
                release.set()
                await settle_known()
                assert state['value'] is None and not state['pending'], 'old owner published after change'
                if current is None or agent._console_historical_read is not current or current['value'] is None:
                    violations.append('new_owner_empty_history_not_published')
                if outcome == 'new_conversation' and current is not None and current['value'] is not None:
                    assert current['value'].subagents[0].run_id == child
                assert agent._console_historical_read is not state
        finally:
            release.set()
            await settle_known()
            assert returned.is_set() and not invalid
            assert all(item['thread'].is_alive() for item in read_resources)
            post_retirement = check_retired()
            assert len(read_resources) == len(history_tasks())
        assert bridge._historical_cache == {} and replacement_bridge._historical_cache == {}

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='historical-original') as executor:
            asyncio.run(exercise(executor))
    finally:
        release.set()
        failures = []
        try:
            database.close()
            replacement_db.close()  # Creator caches only, after known worker retirement.
            with storage._lock:
                assert not storage._operations and not storage._raw_operations
                assert not storage._pending_acquisitions and not storage._retiring_holds
                assert len(storage._startups) == len(startup_entries)
                assert all(storage._startups.get(key) is lease for key, lease in startup_entries)
                assert set(storage._live_leases) == {lease for _, lease in startup_entries}
                for lease, issued_key, hold, owner_thread in startup_holds:
                    assert lease._key == issued_key and storage._holds.get(issued_key) is hold
                    assert hold.count == 1 and hold.ready.is_set() and not hold.stop.is_set()
                    assert hold.error is None and owner_thread is hold.thread and owner_thread.is_alive()
            # The exact remaining owners are this fresh interpreter's original
            # startup entries. Invoke only its unchanged supported atexit path.
            storage._shutdown()
            with storage._lock:
                assert not storage._startups
                assert all(lease not in storage._live_leases and lease._key is None
                           for _, lease in startup_entries)
                for lease, issued_key, hold, owner_thread in startup_holds:
                    assert issued_key not in storage._holds and hold not in storage._retiring_holds
                    assert hold.stop.is_set() and hold.error is None and not owner_thread.is_alive()
                    assert hold in startup_returns
                assert not storage._holds
            startup_shutdown = dict(exact_startup_entries_retired=len(startup_entries),
                                    native_hold_returns=len(startup_returns),
                                    native_holds_issued=len(startup_holds),
                                    native_hold_threads_retired=True,
                                    original_shutdown_invoked=True,
                                    ordinary_SQL_owners_already_physically_retired=True)
        finally:
            try:
                assert monitor.get_events(tool) == 0
                for code, mask in masks.items():
                    assert monitor.get_local_events(tool, code) == mask
                    monitor.set_local_events(tool, code, 0)
                for event, callback in callbacks:
                    assert monitor.register_callback(tool, event, None) is callback
            except BaseException as error:
                failures.append(type(error).__name__)
            finally:
                monitor.free_tool_id(tool)
        assert not failures and monitor.get_tool(tool) is None
    for module, owner, name, function, code, namespace, defaults, kwdefaults, spec, loader_, closure, cells, type_params in anchors:
        assert sys.modules[module.__name__] is module and module.__spec__ is spec
        assert module.__loader__ is loader_ and Path(spec.origin).absolute() == Path(module.__file__).absolute()
        assert inspect.getattr_static(owner, name) is function
        assert function.__code__ is code and function.__globals__ is namespace
        assert function.__defaults__ is defaults and function.__kwdefaults__ is kwdefaults
        assert function.__closure__ is closure and function.__type_params__ is type_params
        assert all(cell.cell_contents is value for cell, value in zip(closure or (), cells, strict=True))
    assert all(path.read_bytes() == source for path, source in sources.items())
    with storage._lock:
        final_census = dict(ordinary=len(storage._live_leases), pending=len(storage._pending_acquisitions),
                            operations=len(storage._operations), raw=len(storage._raw_operations),
                            retiring=len(storage._retiring_holds))
    assert not any(final_census.values()), final_census
    receipt = dict(outcome=outcome, held_stock_callback=True, source_current=True,
                   final_census=final_census, original_query_callbacks=len(read_resources),
                   startup_baseline=startup_facts, startup_shutdown=startup_shutdown,
                   empty_first_query=True, mounted_dom_or_performance_claim=False,
                   guards_replaced=False, global_events=0, hooks_retired=True,
                   repeated_cancel_states=observed_states, post_retirement=post_retirement,
                   violation_reasons=violations,
                   source_hashes={str(path): hashlib.sha256(source).hexdigest()
                                  for path, source in sources.items()})
    (selector.parent.parent / 'historical-empty-rail-receipt.json').write_text(
        json.dumps(receipt, indent=2), encoding='utf-8')
    assert not violations, violations
    print('retired and reopened')


with user_fixture_default_owner():
    main()

"""


@pytest.mark.parametrize(
    "outcome",
    [
        "full_payload_reuse",
        "publication_only",
        "new_conversation",
        "source_generation",
        "foreign_store",
        "foreign_bridge",
        "cancel_double",
    ],
)
def test_empty_history_preserves_full_rail_owned_scope(tmp_path, outcome):
    _run(tmp_path, "historical_empty_rail", outcome, script=_SCRIPT)
