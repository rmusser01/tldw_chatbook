"""Separate no-retention and captured-actor original seed custody controls."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import ast
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from types import FunctionType

from Tests import network_guard, real_profile_guard
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, outcome = sys.argv[1:]
assert route == 'original_seed_creator'
assert outcome in {'unretained', 'captured', 'borrowed', 'foreign'}
from Tests.windows_private_fixture_runner import user_fixture_default_owner

def control():
    selected = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    root = Path(os.environ['XDG_DATA_HOME']).absolute()
    selected.write_text('[general]\nusers_name="seed-control"\n[paths]\ndata_dir="'
                        + root.as_posix() + '"\n', encoding='utf-8')
    selected.chmod(0o600)

    from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
    from Tests.UI import test_console_agent_controller as helpers
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery import participants
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB import private_sqlite

    seed = helpers._seed_done_primary_with_subagents
    assert type(seed) is FunctionType and seed.__globals__ is vars(helpers)
    source = Path(helpers.__file__).absolute()
    assert source == Path(helpers.__spec__.origin).absolute()
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    original_seed = seed, seed.__code__, seed.__globals__
    source_verifier = OriginalStorageUnitObserver({}, False, lambda _name: None)
    source_verifier._pin(seed)  # Source only, never installed and no actor references.
    assert helpers.AgentRunsDB is AgentRunsDB

    def counts():
        with storage._lock:
            return {'ordinary': len(storage._live_leases - set(storage._startups.values())),
                    'pending': len(storage._pending_acquisitions),
                    'core': len(storage._operations), 'raw': len(storage._raw_operations),
                    'retiring': len(storage._retiring_holds)}

    def current():
        assert helpers._seed_done_primary_with_subagents is original_seed[0]
        assert seed.__code__ is original_seed[1] and seed.__globals__ is original_seed[2]
        assert helpers.AgentRunsDB is AgentRunsDB
        assert source.read_bytes() == source_bytes

    def closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError as error:
            assert 'closed' in str(error).lower()
            return True
        return False

    baseline = counts()
    assert not any(baseline.values()), baseline
    path = root / 'original-seeded-runs.sqlite'
    facts = {'route': outcome, 'source_hash': source_hash, 'baseline': baseline,
             'no_App_instance_or_worker': True, 'no_original_body_or_guard_replacement': True}
    pause = None
    captured = []
    foreign = None
    foreign_connection = None
    monitor = None

    try:
        if outcome == 'unretained':
            # No observer, weakrefs, actor lists or census object snapshots are used
            # before this point. Original helper return is deliberately discarded.
            seed(path, tasks=('original seed',))
            gc.collect()
            gc.collect()
            current()
            after = counts()
            pause = storage._begin_local_pause()
            drained = pause.drain(time.monotonic() + 1)
            facts.update(after_original_return_and_gc=after, original_global_drain=drained,
                         observer_installed=False, actor_references_retained=False)
        else:
            if outcome in {'borrowed', 'foreign'}:
                foreign = AgentRunsDB(root / 'caller-owned-runs.sqlite', client_id='caller')
                foreign_connection = foreign._held_connection()
                if outcome == 'borrowed':
                    foreign_connection.execute('BEGIN')
                foreign_participant = participants._repository_participant(foreign)
                foreign_lease = foreign_participant.connections[foreign_connection]
            # Separate child: this intentional actor capture must not be cited as
            # the no-retention leak witness.
            monitor = OriginalStorageUnitObserver({}, False, lambda _name: None)
            code = monitor._pin(seed)
            close_function = inspect.getattr_static(AgentRunsDB, 'close')
            close_code = monitor._pin(close_function)
            monitor.slots.extend(((helpers, '_seed_done_primary_with_subagents', seed),
                                  (AgentRunsDB, 'close', close_function)))
            main = threading.current_thread()
            invalid = []
            seed_returned = []

            def started(actual_code, offset):
                if actual_code is not close_code:
                    return
                frame = parent = owner = connection = participant = lease = None
                try:
                    frame = sys._getframe(1)
                    assert frame.f_code is close_code
                    assert frame.f_globals is close_function.__globals__
                    assert threading.current_thread() is main
                    parent = frame.f_back
                    assert parent is not None and parent.f_code is code
                    assert parent.f_globals is seed.__globals__
                    owner = frame.f_locals['self']
                    assert parent.f_locals['db'] is owner
                    assert type(owner) is AgentRunsDB and owner.db_path == path
                    connection = owner._thread_local.conn
                    assert isinstance(connection, sqlite3.Connection) and not closed(connection)
                    assert not sqlite3.Connection.in_transaction.__get__(connection)
                    participant = participants._repository_participant(owner)
                    lease = participant.connections[connection]
                    assert private_sqlite._ordinary_connections[connection] is lease
                    assert lease.resource_participant is participant and lease.resource_path == path
                    assert lease.resource_thread is main and lease in storage._live_leases
                    assert not any(op.participant is participant for op in storage._operations)
                    assert not captured
                    captured.append((owner, connection, participant, lease))
                except BaseException as error:
                    invalid.append('close_start:' + type(error).__name__)
                finally:
                    del frame, parent, owner, connection, participant, lease

            def returned(actual_code, offset, value):
                if actual_code is not code:
                    return
                frame = owner = connection = participant = lease = None
                try:
                    frame = sys._getframe(1)
                    assert frame.f_code is code and frame.f_globals is seed.__globals__
                    assert threading.current_thread() is main and len(captured) == 1
                    owner, connection, participant, lease = captured[0]
                    assert frame.f_locals['db'] is owner
                    assert closed(connection) and owner._thread_local.conn is None
                    assert connection not in participant.connections
                    assert connection not in private_sqlite._ordinary_connections
                    assert lease not in storage._live_leases
                    seed_returned.append(True)
                except BaseException as error:
                    invalid.append('seed_return:' + type(error).__name__)
                finally:
                    del frame, owner, connection, participant, lease, value

            for tool in range(5, 0, -1):
                if tool == sys.monitoring.DEBUGGER_ID:
                    continue
                try:
                    sys.monitoring.use_tool_id(tool, 'original-seed-actor-proof')
                except ValueError:
                    continue
                monitor.tool = tool
                break
            assert monitor.tool is not None and sys.monitoring.get_events(monitor.tool) == 0
            for event, callback in ((sys.monitoring.events.PY_START, started),
                                    (sys.monitoring.events.PY_RETURN, returned)):
                assert sys.monitoring.register_callback(monitor.tool, event, callback) is None
                monitor.registered[event] = callback
            monitor.codes[code] = 'seed_return_physical_proof'
            monitor.codes[close_code] = 'original_creator_close_start'
            assert sys.monitoring.get_local_events(monitor.tool, code) == 0
            assert sys.monitoring.get_local_events(monitor.tool, close_code) == 0
            sys.monitoring.set_local_events(monitor.tool, code, sys.monitoring.events.PY_RETURN)
            sys.monitoring.set_local_events(monitor.tool, close_code, sys.monitoring.events.PY_START)
            monitor.active = monitor.installed = True
            seed(path, tasks=('original seed',))
            observer = monitor.close()
            monitor = None
            assert observer['complete'] and observer['original_source_current']
            assert observer['global_events'] == 0 and observer['hooks_retired_before_inactive']
            assert not observer['invalid'] and not invalid and len(captured) == 1
            assert seed_returned == [True]
            owner, connection, participant, lease = captured[0]
            assert closed(connection) and owner._thread_local.conn is None
            assert connection not in participant.connections
            assert connection not in private_sqlite._ordinary_connections
            assert lease not in storage._live_leases
            if foreign is not None:
                assert foreign._thread_local.conn is foreign_connection
                assert foreign_participant.connections[foreign_connection] is foreign_lease
                assert foreign_lease in storage._live_leases and not closed(foreign_connection)
                assert sqlite3.Connection.in_transaction.__get__(foreign_connection) is (outcome == 'borrowed')
                assert foreign_connection.execute('SELECT 1').fetchone()[0] == 1
                facts['caller_owned_native_preserved'] = True
                foreign_connection.rollback()
                AgentRunsDB.close(foreign)
                assert closed(foreign_connection) and foreign_lease not in storage._live_leases
            current()
            pause = storage._begin_local_pause()
            drained = pause.drain(time.monotonic() + 1)
            facts.update(after_original_creator_finally=counts(), original_global_drain=drained,
                         observer=observer, captured_exact_native_closed=True,
                         original_seed_return_physically_closed=True,
                         original_close_start_live_actor=True,
                         actor_capture_separate_from_unretained_case=True)
    finally:
        if pause is not None:
            pause.resume()
        if monitor is not None and monitor.tool is not None:
            monitor.close()
        current()
        facts['source_current_final'] = True
        receipt = root.parent / ('original-seed-' + outcome + '.json')
        receipt.write_text(json.dumps(facts, sort_keys=True), encoding='utf-8')

    if outcome == 'unretained':
        # Sole causal oracle, after original native/source/GC/pause observations.
        assert facts['after_original_return_and_gc'] == baseline, facts
        assert facts['original_global_drain'], facts
    else:
        assert facts['after_original_creator_finally'] == baseline, facts
        assert facts['original_global_drain'], facts
    print('retired and reopened')

with user_fixture_default_owner():
    control()
"""


@pytest.mark.parametrize("case", ["unretained", "captured", "borrowed", "foreign"])
def test_original_seed_creator_retirement(tmp_path, case):
    _run(tmp_path, "original_seed_creator", case, script=_SCRIPT)
