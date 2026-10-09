"""Real private config and exact-origin factory writer controls; no reader fake."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio, ast, collections, copy, inspect, os, sys, threading, tomllib
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, outcome = sys.argv[1:]
assert route == 'factory_lifecycle'
assert outcome in {'stock', 'repeat', 'edited', 'old', 'preserve', 'override', 'override_none',
                   'template_identity', 'library_shape', 'library_identity', 'future_table',
                   'snapshot_race', 'full_factory_override', 'retarget_stock', 'retarget_stock_reload', 'same_a_reload', 'source_tuple'}
selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
if outcome == 'old':
    selector.write_text('[general]\nusers_name="old-factory"\n[library.rail_state]\nlifecycle="starter"\n', encoding='utf-8')
    selector.chmod(0o600)


async def run():
    from tldw_chatbook import config
    from Tests.UI import app_factory as factory
    from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
    before = selector.read_bytes()
    generation = config._CONFIG_GENERATION
    template, defaults, library, shape, creation = config._FRESH_LIBRARY_CREATION_SOURCE
    original_library = copy.deepcopy(library)
    original_source_tuple = config._FRESH_LIBRARY_CREATION_SOURCE
    original_selector_env = os.environ['TLDW_CONFIG_PATH']
    target = selector
    counts = collections.Counter()
    observer = OriginalStorageUnitObserver(counts, lambda: True, lambda unit: counts.update((unit,)))
    writer = config.apply_settings_mutation_to_cli_config
    code = observer._pin(writer)
    observer.codes[code] = 'original_canonical_writer'
    observer.slots.append((config, 'apply_settings_mutation_to_cli_config', writer))
    for tool in range(5, 0, -1):
        if tool == observer.monitor.DEBUGGER_ID: continue
        try: observer.monitor.use_tool_id(tool, 'factory-origin-original-writer')
        except ValueError: continue
        observer.tool = tool
        break
    assert observer.tool is not None
    callback_start, callback_return = observer._start, observer._return
    race_seen = False
    race_code, race_line = None, None
    if outcome == 'snapshot_race':
        function = factory._prepare_returning_factory_library_lifecycle
        race_code = observer._pin(function)
        observer.slots.append((factory, '_prepare_returning_factory_library_lifecycle', function))
        module = sys.modules[function.__module__]
        tree = ast.parse(Path(module.__file__).read_bytes())
        body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function.__name__)
        assignment = next(node for node in ast.walk(body) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'result' for target in node.targets))
        race_line = assignment.lineno

    def external_edit(code, lineno):
        nonlocal race_seen
        if code is race_code and lineno == race_line and not race_seen:
            race_seen = True
            selector.write_bytes(before + b'\n# externally edited after snapshot\n')

    try:
        for event, callback in ((observer.monitor.events.PY_START, callback_start), (observer.monitor.events.PY_RETURN, callback_return)):
            previous = observer.monitor.register_callback(observer.tool, event, callback)
            observer.registered[event] = callback
            if previous is not None:
                observer.monitor.register_callback(observer.tool, event, previous)
                observer.registered.pop(event)
                raise RuntimeError('factory_origin_callback_borrowed')
        observer.active = observer.installed = True
        observer.monitor.set_local_events(observer.tool, code, observer.monitor.events.PY_START | observer.monitor.events.PY_RETURN)
        if race_code is not None:
            previous = observer.monitor.register_callback(observer.tool, observer.monitor.events.LINE, external_edit)
            assert previous is None
            observer.registered[observer.monitor.events.LINE] = external_edit
            observer.codes[race_code] = 'source_only_factory_precondition'
            observer.monitor.set_local_events(observer.tool, race_code, observer.monitor.events.LINE)
        if outcome == 'edited':
            selector.write_bytes(before + b'\n# user edited the stock document\n')
        elif outcome == 'template_identity':
            config.CONFIG_TOML_CONTENT = (template + ' ')[:-1]
            assert config.CONFIG_TOML_CONTENT == template and config.CONFIG_TOML_CONTENT is not template
        elif outcome == 'library_shape':
            library['ingest_directory_scan_limit'] += 1
        elif outcome == 'library_identity':
            defaults['library'] = copy.deepcopy(library)
        elif outcome == 'future_table':
            library['rail_state'] = {'lifecycle': 'starter'}
        elif outcome == 'source_tuple':
            config._FRESH_LIBRARY_CREATION_SOURCE = tuple(list(original_source_tuple))
            assert config._FRESH_LIBRARY_CREATION_SOURCE is not original_source_tuple
        elif outcome in {'retarget_stock', 'retarget_stock_reload'}:
            target = selector.parent / 'preexisting-stock-config.toml'
            target.write_bytes(before)
            target.chmod(0o600)
            os.environ['TLDW_CONFIG_PATH'] = str(target)
            assert config.first_profile_created_this_session()
            if outcome == 'retarget_stock_reload':
                from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
                try:
                    config.load_cli_config_and_ensure_existence(force_reload=True)
                except RecoveryRequired as error:
                    assert type(error) is RecoveryRequired
                    assert error.args == ('raw_source_selection_changed',)
                else:
                    raise AssertionError('original raw source selection did not refuse')
        elif outcome == 'same_a_reload':
            config.load_cli_config_and_ensure_existence(force_reload=True)
            assert config.first_profile_created_this_session()
            assert config._CONFIG_CACHE_SOURCE == selector
            assert config._CONFIG_CACHE.get('_first_run') is not True

        overrides = {'library': {'rail_state': {'lifecycle': None if outcome == 'override_none' else 'starter'}}} if outcome in {'override', 'override_none', 'full_factory_override'} else None
        explicit = factory._has_explicit_library_lifecycle_override(overrides)
        if outcome == 'full_factory_override':
            app = factory._build_test_app(config_overrides=overrides)
            try:
                assert app.app_config['library']['rail_state']['lifecycle'] == 'starter'
            finally:
                await app.console_runtime.dispose()
                factory.drain_active_service_patches()
                factory.drain_created_dirs()
            prepared = False
        else:
            prepared = factory._prepare_returning_factory_library_lifecycle(
                preserve_profile_admission=outcome == 'preserve', explicit_override=explicit)
            if outcome == 'repeat':
                assert prepared and not factory._prepare_returning_factory_library_lifecycle(preserve_profile_admission=False, explicit_override=False)
    finally:
        config.CONFIG_TOML_CONTENT = template
        config._FRESH_LIBRARY_CREATION_SOURCE = original_source_tuple
        os.environ['TLDW_CONFIG_PATH'] = original_selector_env
        defaults['library'] = library
        library.clear()
        library.update(original_library)
        receipt = observer.close()
    assert receipt['complete'] and receipt['original_source_current'] and not receipt['invalid'], receipt
    assert receipt['global_events'] == 0 and receipt['hooks_retired_before_inactive']
    assert observer.monitor.get_tool(observer.tool) is None
    assert all(observer.monitor.get_local_events(observer.tool, selected) == 0 for selected in observer.codes)
    expected_write = outcome in {'stock', 'repeat'}
    assert prepared is expected_write
    assert counts['original_canonical_writer'] == int(expected_write or outcome == 'snapshot_race')
    assert config._CONFIG_GENERATION == generation + int(expected_write)
    physical = target.read_bytes()
    if expected_write:
        assert tomllib.loads(physical.decode('utf-8'))['library']['rail_state']['lifecycle'] == 'expanded'
        assert 'lifecycle' not in config.DEFAULT_CONFIG_FROM_TOML['library'].get('rail_state', {})
    elif outcome == 'snapshot_race':
        assert race_seen and physical == before + b'\n# externally edited after snapshot\n'
    elif outcome == 'edited':
        assert physical == before + b'\n# user edited the stock document\n'
    else:
        assert physical == before
    assert not network_guard.blocked_attempts() and not real_profile_guard.take_violations()
    print('retired and reopened')


with user_fixture_default_owner():
    asyncio.run(run())
"""


@pytest.mark.parametrize(
    "outcome",
    (
        "stock",
        "repeat",
        "edited",
        "old",
        "preserve",
        "override",
        "override_none",
        "template_identity",
        "library_shape",
        "library_identity",
        "future_table",
        "snapshot_race",
        "full_factory_override",
        "retarget_stock",
        "retarget_stock_reload",
        "same_a_reload",
        "source_tuple",
    ),
)
def test_factory_returning_fiction_writes_only_exact_untouched_creation_document(
    tmp_path, outcome
):
    _run(tmp_path, "factory_lifecycle", outcome, script=_SCRIPT, timeout=90)
