"""Original function identity must not qualify a replaced reader body."""

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
    from tldw_chatbook.Backup_Recovery import bootstrap, raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    original = config.get_prompts_db_path
    original_code = original.__code__
    assert type(original) is FunctionType and not original_code.co_freevars
    original_copy = FunctionType(original_code, original.__globals__,
                                 original.__name__, original.__defaults__)
    original_copy.__kwdefaults__ = original.__kwdefaults__
    entries = []
    config.__dict__['_sensitive_body_test_original'] = original_copy
    config.__dict__['_sensitive_body_test_entries'] = entries
    exec('def _sensitive_body_test_replacement():\n'
         '    from tldw_chatbook.Backup_Recovery import raw_participants as raw\n'
         '    _sensitive_body_test_entries.append(getattr(raw._local, "operation", None))\n'
         '    return _sensitive_body_test_original()\n', config.__dict__)
    replacement_code = config.__dict__['_sensitive_body_test_replacement'].__code__
    changed = []

    def mutate():
        assert config.get_prompts_db_path is original
        original.__code__ = replacement_code
        changed.append(True)

    if route == 'before_import':
        assert 'tldw_chatbook.Utils.sensitive_paths' not in sys.modules
        mutate()

    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    database = AgentRunsDB(config.get_user_data_dir() / 'agent_runs.db',
                           client_id='body-binding-native', reconcile_on_init=False)
    with database.connection() as connection:
        connection.execute('SELECT COUNT(*) FROM sqlite_master').fetchone()
    database.close()
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        pass
    else:
        raise AssertionError('seeded SQLite connection remains physically open')
    from tldw_chatbook.Utils import sensitive_paths as sensitive
    assert sensitive._RAW_INPUTS_MEMO is None
    modules = (config, raw, storage, sensitive)
    files = {module.__name__: Path(module.__file__).absolute() for module in modules}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    entries.clear()
    first_builder_code = sensitive._sensitive_single_file_paths.__code__

    def returned(frame, event, arg):
        if event == 'return' and frame.f_code is first_builder_code and not changed:
            mutate()
        return None

    def selected(frame, event, arg):
        if event == 'call' and frame.f_code is first_builder_code:
            frame.f_trace_lines = False
            frame.f_trace_opcodes = False
            return returned
        return None

    assert sys.gettrace() is None and threading.gettrace() is None
    if route == 'mid_build':
        threading.settrace_all_threads(selected)
    refused = False
    try:
        try:
            context = sensitive.resolve_sensitive_context()
        except bootstrap.RecoveryRequired:
            refused = True
        assert changed, 'original builder return was not reached'
        if route == 'before_import':
            assert not refused
            assert len(context.db_paths) == 13
            assert entries and all(active is None for active in entries), (
                'a custom same-identity reader body ran inside the added stock scope')
        else:
            assert refused, 'a qualified build accepted a changed reader body'
            assert not entries, 'changed reader body ran before refusal'
            assert sensitive._RAW_INPUTS_MEMO is None
    finally:
        threading.settrace_all_threads(None)
        sys.settrace(None)
        original.__code__ = original_code
        assert config.get_prompts_db_path is original
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
    print('retired and reopened')


with user_fixture_default_owner():
    main()
"""


@pytest.mark.parametrize("route", ["before_import", "mid_build"])
def test_same_function_reader_body_changes_keep_custom_and_refusal_contracts(
    tmp_path, route
):
    _run(tmp_path, route, "body_change", script=_SCRIPT)
