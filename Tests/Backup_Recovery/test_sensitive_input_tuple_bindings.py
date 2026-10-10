"""Definition-time accessor names distinguish stock and custom cold readers.

Evidence-only draft. This module is not installed or native-qualified.
"""

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
    from tldw_chatbook.Backup_Recovery import raw_participants as raw, storage_admission as storage
    from tldw_chatbook.Utils import windows_files
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    database = AgentRunsDB(config.get_user_data_dir() / 'agent_runs.db',
                           client_id='tuple-binding-native', reconcile_on_init=False)
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
    original_names = sensitive._DB_PATH_ACCESSOR_NAMES
    expected_names = (
        'get_chachanotes_db_path', 'get_prompts_db_path', 'get_media_db_path',
        'get_library_collections_db_path', 'get_library_ingest_jobs_db_path',
        'get_workspaces_db_path', 'get_subscriptions_db_path',
        'get_notifications_db_path', 'get_research_db_path', 'get_writing_db_path',
        'get_scheduled_tasks_db_path', 'get_evals_db_path', 'get_rag_indexing_db_path',
    )
    assert type(original_names) is tuple and original_names == expected_names
    original_accessors = tuple(config.__dict__[name] for name in original_names)
    assert all(type(reader) is FunctionType for reader in original_accessors)
    original_codes = tuple(reader.__code__ for reader in original_accessors)
    original_prompts = config.get_prompts_db_path
    original_builders = tuple(sensitive.__dict__[name] for name in (
        '_sensitive_single_file_paths', '_sensitive_skill_trust_dir',
        '_sensitive_db_paths', '_direct_child_rule_container_dirs',
    ))
    builder_codes = tuple(builder.__code__ for builder in original_builders)
    protected = (config.get_user_data_dir, config._load_settings_guarded,
                 config._load_settings_uncached, life.operation, raw._check,
                 storage.acquire_storage, windows_files._Native.open_handle)
    protected_codes = tuple(function.__code__ for function in protected)
    repository = Path.cwd().resolve()
    modules = (config, life, raw, storage, sensitive, windows_files)
    files = {module.__name__: Path(module.__file__).absolute() for module in modules}
    assert all(path.is_relative_to(repository) for path in files.values())
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    custom_name = '_sensitive_tuple_test_extra_db_path'
    assert custom_name not in config.__dict__
    entries = []
    actors = {}
    changed = []
    barrier = {}
    native_calls = []

    def custom_extra():
        actor = threading.current_thread()
        actors[id(actor)] = actor
        entries.append(getattr(raw._local, 'operation', None))
        # An actual existing accessor supplies the path. Duplicate identity keeps
        # every original literal protected path requirement meaningful.
        return original_prompts()

    config.__dict__[custom_name] = custom_extra

    def unchanged_readers():
        assert all(config.__dict__[name] is reader
                   and reader.__code__ is code
                   for name, reader, code in zip(original_names, original_accessors, original_codes))
        assert all(sensitive.__dict__[name] is reader and reader.__code__ is code
                   for name, reader, code in zip((
                       '_sensitive_single_file_paths', '_sensitive_skill_trust_dir',
                       '_sensitive_db_paths', '_direct_child_rule_container_dirs',
                   ), original_builders, builder_codes))

    def mutate():
        unchanged_readers()
        assert sensitive._DB_PATH_ACCESSOR_NAMES is original_names
        replacement = original_names + (custom_name,)
        assert type(replacement) is tuple and replacement is not original_names
        sensitive._DB_PATH_ACCESSOR_NAMES = replacement
        changed.append(True)

    if route == 'preinstalled':
        mutate()

    first_builder_code = original_builders[0].__code__
    native_code = windows_files._Native.open_handle.__code__

    def returned(frame, event, arg):
        if event == 'return' and frame.f_code is first_builder_code and not changed:
            active = getattr(raw._local, 'operation', None)
            assert active is not None, 'original builder did not run in the issued stock config lifetime'
            actor = threading.current_thread()
            with storage._lock:
                state = raw._states[active]
                assert state.active and active in storage._raw_operations
                assert state.source is config and state.thread is actor
                assert state.pid == os.getpid() and not state.uncertain
                assert state.participant is not None and state.leases
                assert all(lease in storage._live_leases
                           and hold is not None and storage._holds.get(lease._key) is hold
                           and hold.ready.is_set() and not hold.stop.is_set() and hold.error is None
                           for lease, hold in zip(state.leases, state.holds))
                assert len(state.leases) == len(state.holds)
                barrier.update(issued=True, actual_source=True, native_holds=len(state.holds))
            actors[id(actor)] = actor
            mutate()
        return None

    def selected(frame, event, arg):
        if event != 'call':
            return None
        if frame.f_code is native_code:
            native_calls.append(True)
        if route != 'mid_build' or frame.f_code is not first_builder_code:
            return None
        frame.f_trace_lines = False
        frame.f_trace_opcodes = False
        return returned

    def census():
        with storage._lock:
            startup = set(storage._startups.values())
            result = {
                'ordinary': len(storage._live_leases - startup),
                'pending': len(storage._pending_acquisitions),
                'operations': len(storage._operations), 'raw': len(storage._raw_operations),
                'raw_states': len(raw._states), 'retiring': len(storage._retiring_holds),
            }
        assert all(value == 0 for value in result.values()), result
        return result

    census()
    assert sys.gettrace() is None and threading.gettrace() is None
    threading.settrace_all_threads(selected)
    refused = None
    context = None
    try:
        try:
            context = sensitive.resolve_sensitive_context()
        except bootstrap.RecoveryRequired as error:
            refused = type(error).__name__
    finally:
        threading.settrace_all_threads(None)
        sys.settrace(None)
        sensitive._DB_PATH_ACCESSOR_NAMES = original_names
        del config.__dict__[custom_name]
        database.close()
        final_census = census()
        unchanged_readers()
        current_protected = (config.get_user_data_dir, config._load_settings_guarded,
                             config._load_settings_uncached, life.operation, raw._check,
                             storage.acquire_storage, windows_files._Native.open_handle)
        assert all(current is original and current.__code__ is code
                   for current, original, code in zip(current_protected, protected, protected_codes))
        assert sensitive._DB_PATH_ACCESSOR_NAMES is original_names
        assert all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name]
                   for name, path in files.items())
        assert sys.gettrace() is None and threading.gettrace() is None
        assert not network_guard.blocked_attempts()
        assert not real_profile_guard.take_violations()

    assert changed, 'actual tuple mutation was not reached'
    if os.name == 'nt':
        assert native_calls, 'original native metadata boundary was not observed'
    if route == 'mid_build':
        assert barrier.get('issued') and barrier.get('actual_source')
    receipt = {'route': route, 'outcome': outcome, 'native_entries': len(native_calls),
               'actual_issued_barrier': barrier, 'custom_calls': len(entries),
               'custom_inside_added_scope': sum(item is not None for item in entries),
               'refused': refused, 'memo_published': sensitive._RAW_INPUTS_MEMO is not None,
               'final_census': final_census, 'source_hashes': hashes,
               'original_readers_and_codes_unchanged': True,
               'original_tuple_restored': True, 'guard_violations': 0,
               'network_attempts': 0, 'startup_owners_until_process_exit': len(storage._startups)}
    (selector.parent.parent / 'sensitive-tuple-receipt.json').write_text(
        json.dumps(receipt, sort_keys=True), encoding='utf-8')
    print(json.dumps(receipt, sort_keys=True))
    if route == 'preinstalled':
        assert refused is None, 'preinstalled custom name tuple changed the original additive route'
        assert context is not None
        user = root / 'qualifier'
        assert set(context.db_paths) == {
            user / 'tldw_chatbook_ChaChaNotes.db', user / 'tldw_chatbook_prompts.db',
            user / 'tldw_chatbook_media_v2.db', user / 'tldw_chatbook_library_collections.db',
            user / 'tldw_chatbook_library_ingest_jobs.db', user / 'tldw_chatbook_workspaces.db',
            user / 'tldw_chatbook_subscriptions.db', user / 'tldw_chatbook_notifications.db',
            user / 'tldw_chatbook_research.db', user / 'tldw_chatbook_writing.db',
            user / 'tldw_chatbook_scheduled_tasks.db', user / 'evals.db', user / 'rag_indexing.db',
        }
        assert sensitive.is_sensitive_path(user / 'mcp_permissions.json', context=context)
        assert sensitive.is_sensitive_path(user / 'tldw_chatbook_prompts.db', context=context)
        assert not sensitive.is_sensitive_path(root / 'ordinary.txt', context=context)
        assert len(entries) == 1 and entries[0] is None, (
            'preinstalled custom name tuple acquired the added stock config scope')
    else:
        assert refused is not None, 'qualified build accepted a changed DB-name tuple'
        assert not entries, 'newly named custom callback ran before tuple drift refusal'
        assert sensitive._RAW_INPUTS_MEMO is None, 'changed DB-name tuple published a raw memo'
    print('retired and reopened')


with user_fixture_default_owner():
    main()
"""


@pytest.mark.parametrize("route", ["preinstalled", "mid_build"])
def test_db_name_tuple_preserves_custom_route_and_refuses_qualified_drift(tmp_path, route):
    _run(tmp_path, route, "tuple_change", script=_SCRIPT)
