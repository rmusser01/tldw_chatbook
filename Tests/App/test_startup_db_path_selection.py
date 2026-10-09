"""Native finite startup selection and exact publication controls."""

import ast
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import hashlib, json, os, sys, threading
from pathlib import Path
from types import FunctionType, MethodType
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, outcome = sys.argv[1:]


def main():
    data = Path(os.environ['XDG_DATA_HOME']).absolute()
    selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    custom_root = data / 'explicit'
    custom_root.mkdir(mode=0o700)
    custom_paths = {
        'chachanotes_db_path': custom_root / 'notes.db',
        'media_db_path': custom_root / 'media.db',
        'prompts_db_path': custom_root / 'prompts.db',
    }
    document = '[general]\nusers_name="finite-paths"\n[paths]\ndata_dir="' + data.as_posix() + '"\n'
    if route in {'configured', 'partial_configured'}:
        selected_custom = custom_paths if route == 'configured' else {'media_db_path': custom_paths['media_db_path']}
        document += '\n[database]\n' + ''.join(name + '="' + path.as_posix() + '"\n'
                                                for name, path in selected_custom.items())
    selector.write_text(document, encoding='utf-8')
    selector.chmod(0o600)
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import config_participants as life
    from tldw_chatbook.Backup_Recovery import raw_participants as raw, storage_admission as storage
    from tldw_chatbook.Utils import windows_files
    config.get_user_data_dir()
    original_prompts = config.get_prompts_db_path
    original_prompts_code = original_prompts.__code__
    original_prompts_kwdefaults = original_prompts.__kwdefaults__
    original_prompts_kwitems = dict(original_prompts_kwdefaults)
    custom_entries = []
    changed_body_calls = []
    changes = []
    pauses = []
    refused = None
    actual = None
    previous_selector = os.environ['TLDW_CONFIG_PATH']
    alternate_selector = selector.with_name('alternate.toml')
    alternate_selector.write_text('[general]\nusers_name="alternate"\n', encoding='utf-8')
    alternate_selector.chmod(0o600)

    def changed_prompts_body(*, ignore_override=False):
        return Path('changed-prompts.db')


    def custom_prompts():
        custom_entries.append(getattr(raw._local, 'operation', None))
        return original_prompts()

    if route == 'custom_before':
        config.get_prompts_db_path = custom_prompts
    from tldw_chatbook import app_service_wiring as wiring
    if route == 'custom_after':
        wiring.get_prompts_db_path = custom_prompts
    if route == 'custom_default_before':
        original_prompts.__kwdefaults__['ignore_override'] = True
    if route == 'custom_borrowed':
        class Borrowed:
            def read(self):
                custom_entries.append(getattr(raw._local, 'operation', None))
                return original_prompts()
        borrowed = Borrowed()
        wiring.get_prompts_db_path = borrowed.read
    function = wiring.ServiceWiringMixin._build_chatbook_db_paths
    receiver = object.__new__(wiring.ServiceWiringMixin)
    bound = function.__get__(receiver, type(receiver))
    assert type(function) is FunctionType and function.__globals__ is wiring.__dict__
    assert type(bound) is MethodType and bound.__func__ is function and bound.__self__ is receiver
    originals = (config.get_chachanotes_db_path, config.get_media_db_path, original_prompts)
    assert all(type(callback) is FunctionType and callback.__globals__ is config.__dict__
               for callback in originals)
    user_body = config.get_user_data_dir._config_guarded_body[0]
    assert config.get_user_data_dir.__wrapped__ is user_body
    assert user_body.__globals__ is config.__dict__
    codes = {callback.__code__: name for callback, name in zip(originals, ('notes', 'media', 'prompts'))}
    user_code, check_code = user_body.__code__, raw._check.__code__
    native_code = windows_files._Native.open_handle.__code__
    operation_body = life.operation.__wrapped__
    operation_code = operation_body.__code__
    operation_body_kwdefaults = operation_body.__kwdefaults__
    operation_body_kwitems = dict(operation_body_kwdefaults)
    notes_code = originals[0].__code__
    changed_body_code = changed_prompts_body.__code__
    selected = set(codes) | {user_code, check_code, native_code, changed_body_code, operation_code}
    modules = (config, life, raw, storage, windows_files, wiring)
    source = {module.__name__: Path(module.__file__).absolute() for module in modules}
    assert all(path.is_relative_to(Path.cwd().resolve()) for path in source.values())
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source.items()}
    protected = (config.get_user_data_dir, user_body, config._load_settings_guarded,
                 config._load_settings_uncached, life.operation, raw._check,
                 storage.acquire_storage, windows_files._Native.open_handle, function, operation_body)
    protected_records = tuple((callback, callback.__code__, callback.__globals__) for callback in protected)
    actor = threading.current_thread()
    current = 'none'
    entries = {}
    operations = {}
    source_metadata = []
    monitoring = sys.monitoring
    tool = next(value for value in (5, 4, 3, 2, 1, 0) if monitoring.get_tool(value) is None)
    events = monitoring.events.PY_START
    masks = {code: events for code in selected}
    masks[notes_code] |= monitoring.events.PY_RETURN
    masks[operation_code] = monitoring.events.PY_RETURN
    active = True

    def on_start(code, offset):
        assert active and threading.current_thread() is actor
        count = entries.setdefault(current, {'native': 0, 'checks': 0, 'getters': {'notes': 0, 'media': 0, 'prompts': 0}, 'user_bodies': 0})
        if code is changed_body_code:
            changed_body_calls.append(True)
        elif code is native_code:
            count['native'] += 1
        elif code is check_code:
            count['checks'] += 1
        elif code in codes:
            count['getters'][codes[code]] += 1
        elif code is user_code:
            count['user_bodies'] += 1
            operation = getattr(raw._local, 'operation', None)
            with storage._lock:
                state = raw._states.get(operation)
                assert operation in storage._raw_operations and state is not None
                assert state.source is config and state.thread is actor and state.pid == os.getpid()
                assert state.participant in raw._participants
                assert state.leases and all(lease in storage._live_leases for lease in state.leases)
                source_metadata.append((operation, state, actor))
            operations.setdefault(current, []).append(operation)

    def mutate():
        count = entries['production']
        assert count['native'] > 0 if os.name == 'nt' else count['checks'] > 0
        assert count['user_bodies'] > 0 and source_metadata
        changes.append(True)
        if route == 'callback':
            wiring.get_prompts_db_path = custom_prompts
        elif route == 'body':
            original_prompts.__code__ = changed_body_code
        elif route == 'source':
            os.environ['TLDW_CONFIG_PATH'] = str(alternate_selector)
        elif route in {'pause', 'final_pause'}:
            pauses.append(storage._begin_local_pause())

    def on_return(code, offset, value):
        assert active and threading.current_thread() is actor
        if current != 'production' or changes:
            return
        if route in {'callback', 'body', 'source', 'pause'} and code is notes_code:
            mutate()
        elif route == 'final_pause' and code is operation_code:
            if (entries.get('production', {}).get('user_bodies', 0) == 3
                    and getattr(raw._local, 'operation', None) is None):
                with storage._lock:
                    assert not storage._raw_operations and not raw._states
                mutate()

    monitoring.use_tool_id(tool, 'finite-startup-path-control')
    monitoring.register_callback(tool, monitoring.events.PY_START, on_start)
    monitoring.register_callback(tool, monitoring.events.PY_RETURN, on_return)
    try:
        for code, mask in masks.items():
            monitoring.set_local_events(tool, code, mask)
        assert monitoring.get_events(tool) == 0
        if route == 'mechanism':
            current = 'independent_original_getters'
            baseline = {name: str(callback()) for name, callback in zip(('ChaChaNotes', 'Media', 'Prompts'), originals)}
        if route == 'custom_operation_body_before':
            operation_body.__code__ = operation_code.replace(co_name='custom_operation')
        elif route == 'custom_operation_defaults_before':
            operation_body_kwdefaults['wait_for_locks'] = False
        current = 'production'
        try:
            actual = bound()
        except life.bootstrap.RecoveryRequired as error:
            refused = str(error)
        if route == 'mechanism':
            assert actual == baseline

            current = 'supported_finite_hypothesis'
            with life.operation(config) as admitted:
                before = life.checked_config_identity(config, admitted)
                grouped = bound()
                after = life.checked_config_identity(config, admitted)
                assert before == after
            assert actual == grouped
    finally:
        assert monitoring.get_tool(tool) == 'finite-startup-path-control'
        assert monitoring.get_events(tool) == 0
        for code, mask in masks.items():
            assert monitoring.get_local_events(tool, code) == mask
            monitoring.set_local_events(tool, code, 0)
        assert monitoring.register_callback(tool, monitoring.events.PY_START, None) is on_start
        assert monitoring.register_callback(tool, monitoring.events.PY_RETURN, None) is on_return
        assert all(monitoring.get_local_events(tool, code) == 0 for code in selected)
        monitoring.free_tool_id(tool)
        active = False
        wiring.get_prompts_db_path = original_prompts
        config.get_prompts_db_path = original_prompts
        operation_body.__code__ = operation_code
        operation_body.__kwdefaults__ = operation_body_kwdefaults
        operation_body_kwdefaults.clear()
        operation_body_kwdefaults.update(operation_body_kwitems)
        original_prompts.__code__ = original_prompts_code
        original_prompts.__kwdefaults__ = original_prompts_kwdefaults
        original_prompts_kwdefaults.clear()
        original_prompts_kwdefaults.update(original_prompts_kwitems)
        os.environ['TLDW_CONFIG_PATH'] = previous_selector
        for pause in pauses:
            pause.resume()
    assert monitoring.get_tool(tool) is None
    if route == 'custom_before':
        config.get_prompts_db_path = original_prompts
    if route == 'custom_after':
        wiring.get_prompts_db_path = original_prompts
    assert all(callback.__code__ is code and callback.__globals__ is defining
               for callback, code, defining in protected_records)
    assert wiring.ServiceWiringMixin._build_chatbook_db_paths is function
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name] for name, path in source.items())
    with storage._lock:
        startup = set(storage._startups.values())
        census = {
            'ordinary': len(storage._live_leases - startup),
            'pending': len(storage._pending_acquisitions),
            'operations': len(storage._operations),
            'raw': len(storage._raw_operations),
            'raw_states': len(raw._states),
            'retiring': len(storage._retiring_holds),
        }
        assert all(value == 0 for value in census.values()), census
        assert all(operation not in raw._states and operation not in storage._raw_operations
                   for operation, state, thread in source_metadata)
    expected = {
        'ChaChaNotes': str(custom_paths['chachanotes_db_path'] if route == 'configured' else data / 'finite-paths' / config.profile_paths.database_leaf('chachanotes_db_path')),
        'Media': str(custom_paths['media_db_path'] if route in {'configured', 'partial_configured'} else data / 'finite-paths' / config.profile_paths.database_leaf('media_db_path')),
        'Prompts': str(custom_paths['prompts_db_path'] if route == 'configured' else data / 'finite-paths' / config.profile_paths.database_leaf('prompts_db_path')),
    }
    if route not in {'callback', 'body', 'source', 'pause', 'final_pause'}:
        assert actual == expected
    receipt = {'route': route, 'entries': entries,
               'distinct_raw_source_cohorts': {phase: len(set(owners)) for phase, owners in operations.items()},
               'all_original_getter_bodies_observed': entries['production']['getters'] == {'notes': 1, 'media': 1, 'prompts': 1},
               'native_actor_source_metadata_qualified': True, 'final_census': census,
               'monitoring_global_zero': True, 'monitoring_owned_and_restored': True,
               'actual_original_function': True, 'App_constructed': False,
               'source_unchanged': True, 'protected_code_bindings_unchanged': True,
               'custom_callback_ambient_operations': [operation is None for operation in custom_entries],
               'mutation_boundary_reached': bool(changes), 'refusal_reason': refused,
               'changed_body_calls': len(changed_body_calls), 'returned_dictionary': actual is not None,
               'network_attempts': len(network_guard.blocked_attempts()),
               'real_profile_guard_refusals': len(real_profile_guard.take_violations())}
    (data.parent / 'startup-db-path-selection.json').write_text(json.dumps(receipt, sort_keys=True), encoding='utf-8')
    print(json.dumps(receipt))
    assert receipt['network_attempts'] == receipt['real_profile_guard_refusals'] == 0
    if route == 'mechanism':
        original = entries['independent_original_getters']
        production = entries['production']
        grouped = entries['supported_finite_hypothesis']
        assert original['getters'] == production['getters'] == grouped['getters'] == {'notes': 1, 'media': 1, 'prompts': 1}
        assert original['user_bodies'] == production['user_bodies'] == grouped['user_bodies'] == 3
        assert original['checks'] > 0 and production['checks'] > 0 and grouped['checks'] > 0
        assert len(set(operations['independent_original_getters'])) == 3
        assert len(set(operations['production'])) == len(set(operations['supported_finite_hypothesis'])) == 1
        if os.name == 'nt':
            assert production['native'] < original['native'], receipt
    elif route == 'stock':
        assert entries['production']['getters'] == {'notes': 1, 'media': 1, 'prompts': 1}
        assert entries['production']['user_bodies'] == 3
        assert len(set(operations['production'])) == 1, receipt
    elif route in {'callback', 'body', 'source', 'pause', 'final_pause'}:
        assert changes, 'actual original native-backed mutation boundary was not reached'
        assert refused is not None and actual is None, receipt
        assert not custom_entries and not changed_body_calls, receipt
    elif route.startswith('custom'):
        if route not in {'custom_default_before', 'custom_operation_body_before', 'custom_operation_defaults_before'}:
            assert custom_entries and all(operation is None for operation in custom_entries), receipt
        assert len(set(operations['production'])) == 3
    elif route == 'configured':
        assert entries['production']['getters'] == {'notes': 1, 'media': 1, 'prompts': 1}
        assert entries['production']['user_bodies'] == 0
        assert entries['production']['native'] == 0, receipt
    elif route == 'partial_configured':
        assert entries['production']['getters'] == {'notes': 1, 'media': 1, 'prompts': 1}
        assert entries['production']['user_bodies'] == 2
        assert len(set(operations['production'])) == 2
    print('retired and reopened')


with user_fixture_default_owner():
    main()
"""


@pytest.mark.parametrize(
    "route",
    [
        "mechanism",
        "stock",
        "custom_before",
        "custom_after",
        "custom_borrowed",
        "custom_default_before",
        "configured",
        "partial_configured",
        "callback",
        "body",
        "source",
        "pause",
        "final_pause",
        "custom_operation_body_before",
        "custom_operation_defaults_before",
    ],
)
def test_actual_startup_db_path_selection(tmp_path, route, record_property):
    try:
        _run(tmp_path, route, "success", script=_SCRIPT)
    finally:
        receipt = Path(tmp_path) / "startup-db-path-selection.json"
        if receipt.exists():
            record_property(
                "native_path_selection", receipt.read_text(encoding="utf-8")
            )


def test_embedded_startup_path_control_is_syntax_valid():
    ast.parse(_SCRIPT)
    assert "setprofile" not in _SCRIPT and "settrace" not in _SCRIPT
    assert "ThreadPoolExecutor" not in _SCRIPT and "TldwCli(" not in _SCRIPT
