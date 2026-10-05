"""Cold raw sensitive inputs share one fresh config lifetime, never authority."""

import json

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


_SCRIPT = r"""
import hashlib, inspect, json, os, sqlite3, sys, threading
from pathlib import Path
from types import MethodType
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
    extra = '\n[database]\nscheduled_tasks_db_path="relative/scheduled.db"\n' if route == 'cwd' else ''
    selector.write_text('[general]\nusers_name="qualifier"\n[paths]\ndata_dir="'
                        + root.as_posix() + '"\n' + extra, encoding='utf-8')
    selector.chmod(0o600)
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import bootstrap, config_participants as life
    from tldw_chatbook.Backup_Recovery import raw_participants as raw, storage_admission as storage
    from tldw_chatbook.Utils import windows_files
    original_prompts = config.get_prompts_db_path
    custom_entries = []

    def custom_prompts():
        custom_entries.append(getattr(raw._local, 'operation', None))
        return original_prompts()

    if route == 'custom' and outcome == 'before_import':
        assert 'tldw_chatbook.Utils.sensitive_paths' not in sys.modules
        config.get_prompts_db_path = custom_prompts

    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    database = AgentRunsDB(config.get_user_data_dir() / 'agent_runs.db',
                           client_id='sensitive-bundle-native', reconcile_on_init=False)
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
    assert sensitive._RAW_INPUTS_MEMO is None, 'fixture must have a natural cold memo'
    repository = Path.cwd().resolve()
    modules = (config, life, raw, storage, sensitive, windows_files)
    files = {module.__name__: Path(module.__file__).absolute() for module in modules}
    assert all(path.is_relative_to(repository) for path in files.values())
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    protected = (config.get_user_data_dir, config._load_settings_guarded,
                 config._load_settings_uncached, life.operation, raw._check,
                 storage.acquire_storage, windows_files._Native.open_handle)
    original_builder = sensitive._sensitive_db_paths

    if route == 'custom':
        if outcome == 'after_import':
            config.get_prompts_db_path = custom_prompts
        elif outcome == 'bound':
            class Receiver:
                def select(self):
                    return custom_prompts()
            receiver = Receiver()
            config.get_prompts_db_path = receiver.select
            assert type(config.get_prompts_db_path) is MethodType
        elif outcome == 'proxy':
            class Proxy:
                __wrapped__ = original_prompts
                __code__ = original_prompts.__code__
                __globals__ = original_prompts.__globals__
                def __call__(self):
                    return custom_prompts()
                def __eq__(self, other):
                    return True
            config.get_prompts_db_path = Proxy()
        elif outcome == 'helper':
            def builder():
                custom_entries.append(getattr(raw._local, 'operation', None))
                return original_builder()
            sensitive._sensitive_db_paths = builder
    if route == 'failure':
        def failing_prompts():
            custom_entries.append(getattr(raw._local, 'operation', None))
            raise ValueError('controlled accessor failure')
        config.get_prompts_db_path = failing_prompts

    custom_entries.clear()
    native_code = windows_files._Native.open_handle.__code__
    user_code = inspect.unwrap(config.get_user_data_dir).__code__
    check_code = raw._check.__code__
    first_builder_code = sensitive._sensitive_single_file_paths.__code__
    same_key_code = sensitive._same_key.__code__
    inputs_code = sensitive._raw_inputs.__code__
    phase = 'cold'
    counts = {}
    getter_operations = {}
    actor_objects = {}
    changed = []
    pauses = []
    cwd = os.getcwd()
    previous_rag = os.environ.get('RAG_PERSIST_DIR')
    alternate = root / 'alternate'
    alternate.mkdir(mode=0o700)
    alternate_selector = selector.with_name('other-config.toml')
    alternate_selector.write_text('[general]\nusers_name="other"\n', encoding='utf-8')
    alternate_selector.chmod(0o600)
    lock = threading.Lock()

    def census():
        with storage._lock:
            startup = set(storage._startups.values())
            result = {
                'ordinary': len(storage._live_leases - startup),
                'pending': len(storage._pending_acquisitions),
                'operations': len(storage._operations),
                'raw': len(storage._raw_operations),
                'raw_states': len(raw._states),
                'retiring': len(storage._retiring_holds),
            }
        assert all(value == 0 for value in result.values()), result
        return result

    def assert_context(context):
        user = root / 'qualifier'
        assert context.user_data_dir == user.resolve()
        assert set(context.files) == {
            selector.resolve(), user / 'mcp_permissions.json',
            user / 'local_mcp_store.json', user / 'mcp_execution_log.jsonl',
        }
        assert set(context.db_paths) == {
            user / 'tldw_chatbook_ChaChaNotes.db',
            user / 'tldw_chatbook_prompts.db',
            user / 'tldw_chatbook_media_v2.db',
            user / 'tldw_chatbook_library_collections.db',
            user / 'tldw_chatbook_library_ingest_jobs.db',
            user / 'tldw_chatbook_workspaces.db',
            user / 'tldw_chatbook_subscriptions.db',
            user / 'tldw_chatbook_notifications.db',
            user / 'tldw_chatbook_research.db',
            user / 'tldw_chatbook_writing.db',
            user / 'tldw_chatbook_scheduled_tasks.db',
            user / 'evals.db',
            user / 'rag_indexing.db',
        }
        assert user / 'skills' / 'trust' in context.dirs
        assert set(context.direct_child_denied_dirs) == {
            user.resolve(), selector.parent.resolve(), user / 'chromadb', user / 'rag_profiles',
        }
        assert sensitive.is_sensitive_path(user / 'mcp_permissions.json', context=context)
        assert sensitive.is_sensitive_path(user / 'tldw_chatbook_prompts.db', context=context)
        assert not sensitive.is_sensitive_path(root / 'ordinary.txt', context=context)

    def mutate():
        changed.append(True)
        if route == 'source':
            os.environ['TLDW_CONFIG_PATH'] = str(alternate_selector)
        elif route in {'pause', 'final_pause'}:
            pauses.append(storage._begin_local_pause())
        elif route == 'callback':
            config.get_prompts_db_path = custom_prompts
        elif route == 'helper':
            def changed_builder():
                custom_entries.append(getattr(raw._local, 'operation', None))
                return original_builder()
            sensitive._sensitive_db_paths = changed_builder
        elif route == 'environment':
            os.environ['RAG_PERSIST_DIR'] = str(alternate)
        elif route == 'cwd':
            os.chdir(alternate)

    def returned(frame, event, arg):
        if event != 'return':
            return returned
        if frame.f_code is user_code:
            active = getattr(raw._local, 'operation', None)
            if active is not None:
                state = raw._states[active]
                assert state.source is config
                assert state.thread is threading.current_thread()
                getter_operations.setdefault(phase, set()).add(active)
        elif not changed:
            if route == 'final_pause' and frame.f_code is same_key_code:
                caller = frame.f_back
                if (arg is True and caller.f_code is inputs_code
                        and 'inputs' in caller.f_locals):
                    active = getattr(raw._local, 'operation', None)
                    assert active is not None, 'final key check ran outside the finite config scope'
                    state = raw._states[active]
                    assert state.active and state.source is config
                    assert state.thread is threading.current_thread()
                    mutate()
            elif route in {'source', 'pause', 'callback', 'helper', 'environment', 'cwd'}:
                if frame.f_code is first_builder_code:
                    mutate()
        return None

    def selected(frame, event, arg):
        if event != 'call':
            return None
        label = ('native' if frame.f_code is native_code else
                 'user_dir' if frame.f_code is user_code else
                 'raw_check' if frame.f_code is check_code else None)
        if label is not None:
            frame.f_trace_lines = False
            frame.f_trace_opcodes = False
            with lock:
                counter = counts.setdefault(phase, {})
                counter[label] = counter.get(label, 0) + 1
                actor = threading.current_thread()
                actor_objects[id(actor)] = actor
        if frame.f_code not in {user_code, first_builder_code, same_key_code}:
            return None
        frame.f_trace_lines = False
        frame.f_trace_opcodes = False
        return returned

    census()
    assert sys.gettrace() is None and threading.gettrace() is None
    threading.settrace_all_threads(selected)
    refused = None
    try:
        try:
            context = sensitive.resolve_sensitive_context()
        except bootstrap.RecoveryRequired as error:
            refused = type(error).__name__
        if route in {'source', 'pause', 'callback', 'helper', 'final_pause'}:
            assert changed, 'original builder/final key barrier was not reached'
            assert sensitive._RAW_INPUTS_MEMO is None, 'failed or drifted build published a raw memo'
            assert refused is not None, 'a standard build accepted actual source/callback/pause drift'
            if route in {'callback', 'helper'}:
                assert not custom_entries, 'mid-build replacement ran inside the standard route'
        elif route in {'environment', 'cwd'}:
            assert changed
            assert refused is None, 'ordinary environment/cwd drift changed the additive contract'
            assert sensitive._RAW_INPUTS_MEMO is None, 'changed environment/cwd was memoized'
            fresh = sensitive.resolve_sensitive_context()
            assert sensitive._RAW_INPUTS_MEMO is not None
            assert len(fresh.db_paths) == 13
            if route == 'environment':
                assert alternate.resolve() in fresh.direct_child_denied_dirs
                assert root / 'qualifier' / 'chromadb' not in fresh.direct_child_denied_dirs
            else:
                assert alternate / 'relative' / 'scheduled.db' in fresh.db_paths
                assert Path(cwd) / 'relative' / 'scheduled.db' not in fresh.db_paths
        elif route == 'failure':
            assert refused is None
            assert len(context.db_paths) == 12
            assert sensitive._RAW_INPUTS_MEMO is None
            sensitive.resolve_sensitive_context()
            assert len(custom_entries) == 2 and all(item is None for item in custom_entries)
            assert sensitive._RAW_INPUTS_MEMO is None
        else:
            assert refused is None
            assert_context(context)
            assert sensitive._RAW_INPUTS_MEMO is not None
            if route == 'cold':
                assert counts['cold']['user_dir'] == 20, counts
                assert len(getter_operations['cold']) <= 2, 'cold getters independently reacquired config'
                if os.name == 'nt':
                    assert counts['cold']['native'] <= 5000, counts
            elif route == 'custom':
                assert len(custom_entries) == 1
                assert custom_entries[0] is None, 'custom callback acquired the added standard scope'
            elif route == 'warm':
                census()
                phase = 'warm'
                warm = sensitive.resolve_sensitive_context()
                assert warm == context
                assert counts['warm']['user_dir'] == 1 and counts['warm']['raw_check'] == 2, counts
                assert len(getter_operations['warm']) == 1
    finally:
        threading.settrace_all_threads(None)
        sys.settrace(None)
        for pause in pauses:
            pause.resume()
        os.environ['TLDW_CONFIG_PATH'] = str(selector)
        if previous_rag is None:
            os.environ.pop('RAG_PERSIST_DIR', None)
        else:
            os.environ['RAG_PERSIST_DIR'] = previous_rag
        os.chdir(cwd)
        database.close()
        final_census = census()
    assert sys.gettrace() is None and threading.gettrace() is None
    current_protected = (config.get_user_data_dir, config._load_settings_guarded,
                         config._load_settings_uncached, life.operation, raw._check,
                         storage.acquire_storage, windows_files._Native.open_handle)
    assert all(original is current for original, current in zip(protected, current_protected))
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name]
               for name, path in files.items())
    assert not network_guard.blocked_attempts()
    assert not real_profile_guard.take_violations()
    receipt = {'route': route, 'outcome': outcome,
               'native_counts': counts, 'finite_resources_retired': True,
               'final_census': final_census,
               'startup_owners_until_process_exit': len(storage._startups),
               'source_hashes': hashes}
    (selector.parent.parent / 'sensitive-bundle-receipt.json').write_text(
        json.dumps(receipt, sort_keys=True), encoding='utf-8')
    print(json.dumps(receipt, sort_keys=True))
    print('retired and reopened')


with user_fixture_default_owner():
    main()
"""


def _run_bundle(tmp_path, route, outcome):
    _run(tmp_path, route, outcome, script=_SCRIPT)
    receipt = json.loads(
        (tmp_path / "sensitive-bundle-receipt.json").read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            {
                name: receipt[name]
                for name in (
                    "route",
                    "outcome",
                    "native_counts",
                    "final_census",
                    "startup_owners_until_process_exit",
                )
            },
            sort_keys=True,
        )
    )


def test_cold_sensitive_input_build_retains_all_getters_with_one_finite_config_scope(
    tmp_path,
):
    _run_bundle(tmp_path, "cold", "success")


def test_warm_sensitive_input_hit_retains_original_one_getter_two_checks(tmp_path):
    _run_bundle(tmp_path, "warm", "success")


@pytest.mark.parametrize(
    "outcome", ["before_import", "after_import", "bound", "proxy", "helper"]
)
def test_custom_sensitive_input_callbacks_keep_original_unscoped_contract(
    tmp_path, outcome
):
    _run_bundle(tmp_path, "custom", outcome)


@pytest.mark.parametrize(
    "route", ["source", "pause", "callback", "helper", "final_pause"]
)
def test_cold_standard_input_drift_or_failed_scope_exit_does_not_publish_memo(
    tmp_path, route
):
    _run_bundle(tmp_path, route, "refused")


@pytest.mark.parametrize("route", ["environment", "cwd"])
def test_changed_raw_input_environment_or_cwd_is_never_memoized(tmp_path, route):
    _run_bundle(tmp_path, route, "changed")


def test_accessor_failure_remains_additive_and_is_retried_without_memo(tmp_path):
    _run_bundle(tmp_path, "failure", "preserved")
