"""Evidence draft: guarded getter bodies must retain exact stock provenance.

Native execution is coordinated by the parent; this module is not yet qualified.
"""

import json

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


_SCRIPT = r"""
import hashlib, json, os, sqlite3, sys, threading
from pathlib import Path
from types import FunctionType
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, outcome = sys.argv[1:]


def main():
    root = Path(os.environ['XDG_DATA_HOME']).absolute()
    selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    selector.write_text('[general]\nusers_name="qualifier"\n[paths]\ndata_dir="'
                        + root.as_posix() + '"\n', encoding='utf-8')
    selector.chmod(0o600)
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import bootstrap, config_participants as life
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Utils import windows_files

    wrapper = config.get_user_data_dir
    body = wrapper.__wrapped__
    wrapper_code = wrapper.__code__
    wrapper_globals = wrapper.__globals__
    closure = wrapper.__closure__
    body_code = body.__code__
    body_globals = body.__globals__
    body_defaults = body.__defaults__
    body_kwdefaults = body.__kwdefaults__
    assert type(wrapper) is FunctionType and type(body) is FunctionType
    assert wrapper_globals is life.__dict__ and body_globals is config.__dict__
    assert closure is not None and not body_code.co_freevars
    cells = dict(zip(wrapper_code.co_freevars, closure))
    assert cells['function'].cell_contents is body
    assert cells['wrapped'].cell_contents is wrapper
    original_body = FunctionType(body_code, body_globals, body.__name__, body_defaults)
    original_body.__kwdefaults__ = body_kwdefaults

    # Establish the actual registered profile before the only intentional fault.
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    database = AgentRunsDB(wrapper() / 'agent_runs.db',
                           client_id='guarded-body-native', reconcile_on_init=False)
    with database.connection() as connection:
        assert connection.execute('SELECT COUNT(*) FROM sqlite_master WHERE type = ?',
                                  ('table',)).fetchone()[0] > 0
    database.close()
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        pass
    else:
        raise AssertionError('seeded SQLite connection remains physically open')

    from tldw_chatbook.Utils import sensitive_paths as sensitive
    assert sensitive._RAW_INPUTS_MEMO is None, 'fixture requires a natural cold memo'
    modules = (config, life, raw, storage, sensitive, windows_files)
    repository = Path.cwd().resolve()
    files = {module.__name__: Path(module.__file__).absolute() for module in modules}
    assert all(path.is_relative_to(repository) for path in files.values())
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in files.items()}
    protected = (wrapper, config._load_settings_guarded,
                 config._load_settings_uncached, life.operation, raw._check,
                 storage.acquire_storage, windows_files._Native.open_handle)
    protected_codes = tuple(function.__code__ for function in protected)
    protected_globals = tuple(function.__globals__ for function in protected)
    first_builder = sensitive._sensitive_single_file_paths
    first_builder_code = first_builder.__code__
    wrapper_entries = []
    custom_entries = []
    config.__dict__['_sensitive_guarded_test_original'] = original_body
    config.__dict__['_sensitive_guarded_test_wrapper_entries'] = wrapper_entries
    config.__dict__['_sensitive_guarded_test_custom_entries'] = custom_entries
    exec('def _sensitive_guarded_test_replacement():\n'
         '    _sensitive_guarded_test_custom_entries.append(\n'
         '        _sensitive_guarded_test_wrapper_entries[-1])\n'
         '    return _sensitive_guarded_test_original()\n', config.__dict__)
    replacement_code = config.__dict__['_sensitive_guarded_test_replacement'].__code__
    assert replacement_code.co_freevars == body_code.co_freevars == ()
    changed = []
    barrier_preconditions = {}

    def originals_current():
        assert config.get_user_data_dir is wrapper
        assert wrapper.__wrapped__ is body
        assert wrapper.__code__ is wrapper_code and wrapper.__globals__ is wrapper_globals
        assert wrapper.__closure__ is closure
        assert cells['function'].cell_contents is body
        assert cells['wrapped'].cell_contents is wrapper
        assert body.__globals__ is body_globals
        assert body.__defaults__ is body_defaults and body.__kwdefaults__ is body_kwdefaults
        assert all(function.__code__ is code and function.__globals__ is defining
                   for function, code, defining in
                   zip(protected, protected_codes, protected_globals))
        assert sensitive._sensitive_single_file_paths is first_builder
        assert first_builder.__code__ is first_builder_code

    def mutate():
        originals_current()
        assert body.__code__ is body_code and not changed
        body.__code__ = replacement_code
        assert body.__code__ is replacement_code
        changed.append(True)

    def returned(frame, event, arg):
        if event == 'return' and frame.f_code is first_builder_code and not changed:
            active = getattr(raw._local, 'operation', None)
            with storage._lock:
                state = raw._states.get(active)
                barrier_preconditions.update({
                    'actual_original_builder_return': frame.f_code is first_builder_code,
                    'actual_registered_stock_operation': active in storage._raw_operations,
                    'actual_operation_state_live': state is not None and state.active,
                    'same_config_source': state is not None and state.source is config,
                    'same_native_thread': state is not None
                        and state.thread is threading.current_thread(),
                    'same_process': state is not None and state.pid == os.getpid(),
                    'raw_memo_naturally_cold': sensitive._RAW_INPUTS_MEMO is None,
                    'original_guarded_reader_body': body.__code__ is body_code,
                })
            assert all(barrier_preconditions.values()), barrier_preconditions
            mutate()
        return None

    def selected(frame, event, arg):
        if event != 'call':
            return None
        if frame.f_code is wrapper_code and frame.f_locals.get('function') is body:
            # At the original wrapper's CALL, before it enters its own guard:
            # previous_operation therefore observes only a surrounding scope.
            previous_operation = getattr(raw._local, 'operation', None)
            with storage._lock:
                previous_state = raw._states.get(previous_operation)
                stock_already_active = previous_operation is not None
                if stock_already_active:
                    assert previous_state is not None and previous_state.active
                    assert previous_state.source is config
                    assert previous_state.thread is threading.current_thread()
                    assert previous_operation in storage._raw_operations
            wrapper_entries.append({
                'surrounding_stock_operation_already_active': stock_already_active,
                'actual_original_wrapper': frame.f_locals.get('wrapped') is wrapper,
                'actual_original_body_object': frame.f_locals.get('function') is body,
            })
        if route == 'mid_build' and frame.f_code is first_builder_code:
            frame.f_trace_lines = False
            frame.f_trace_opcodes = False
            return returned
        return None

    assert sys.gettrace() is None and threading.gettrace() is None
    refused = False
    context = None
    memo_published = False
    failure = None
    if route == 'before_qualification':
        mutate()
    sys.settrace(selected)
    try:
        try:
            context = sensitive.resolve_sensitive_context()
        except bootstrap.RecoveryRequired:
            refused = True
        except BaseException as error:
            failure = error
        memo_published = sensitive._RAW_INPUTS_MEMO is not None
    finally:
        assert sys.gettrace() is selected
        sys.settrace(None)
        body.__code__ = body_code
        originals_current()
        assert body.__code__ is body_code
        with storage._lock:
            census = {
                'ordinary': len(storage._live_leases - set(storage._startups.values())),
                'pending': len(storage._pending_acquisitions),
                'operations': len(storage._operations), 'raw': len(storage._raw_operations),
                'raw_states': len(raw._states), 'retiring': len(storage._retiring_holds),
            }
        assert all(value == 0 for value in census.values()), census
        assert not network_guard.blocked_attempts()
        assert not real_profile_guard.take_violations()
        assert all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name]
                   for name, path in files.items())
    receipt = {
        'route': route, 'fault_injected': changed == [True],
        'original_wrapper_body_closure_restored': True,
        'original_guard_bodies_unchanged': True, 'trace_restored': sys.gettrace() is None,
        'seeded_sqlite_physically_closed': True, 'source_hashes': hashes,
        'barrier_preconditions': barrier_preconditions, 'final_census': census,
        'custom_entries': custom_entries, 'custom_body_executions': len(custom_entries),
        'refused': refused, 'memo_published': memo_published,
        'db_paths': len(context.db_paths) if context is not None else None,
        'unexpected_error': type(failure).__name__ if failure is not None else None,
    }
    (root.parent / 'sensitive-guarded-body-receipt.json').write_text(
        json.dumps(receipt, sort_keys=True), encoding='utf-8')
    print(json.dumps(receipt, sort_keys=True), flush=True)
    if failure is not None:
        raise failure
    assert changed == [True], 'original accepted boundary was not reached'
    assert all(entry['actual_original_wrapper'] and entry['actual_original_body_object']
               for entry in custom_entries)
    if route == 'before_qualification':
        assert not refused and context is not None and len(context.db_paths) == 13
        assert custom_entries, 'the legitimate custom original body was never called'
        assert not any(entry['surrounding_stock_operation_already_active']
                       for entry in custom_entries), (
            'custom wrapped getter body ran inside the added stock config scope')
    else:
        assert barrier_preconditions and all(barrier_preconditions.values())
        assert refused, 'accepted cold stock build failed to refuse wrapped-body drift'
        assert not custom_entries, 'changed wrapped body ran before source refusal'
        assert not memo_published, 'wrapped-body drift published the raw memo'
    print('retired and reopened')


with user_fixture_default_owner():
    main()
"""


@pytest.mark.parametrize("route", ["before_qualification", "mid_build"])
def test_guarded_getter_body_changes_keep_custom_and_refusal_contracts(tmp_path, route):
    _run(tmp_path, route, "guarded_body_change", script=_SCRIPT, timeout=90)
    receipt = json.loads(
        (tmp_path / "sensitive-guarded-body-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["fault_injected"]
    assert receipt["original_guard_bodies_unchanged"]
    assert receipt["original_wrapper_body_closure_restored"]
    assert receipt["seeded_sqlite_physically_closed"]
    assert receipt["trace_restored"] and not any(receipt["final_census"].values())