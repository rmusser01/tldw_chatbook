"""Actual private-profile controls; Root owns all Native executions."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio, inspect, json, os, sys, threading
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, outcome = sys.argv[1:]
assert route == 'compact_models'
assert outcome in {'normal', 'mapping', 'body', 'choice', 'recancel', 'remove', 'queued_reader', 'queued_body', 'custom_reader', 'custom_body', 'malformed', 'missing_providers', 'missing_defaults', 'dispose', 'closing'}
selected = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
assert 'tldw_chatbook.config' not in sys.modules and 'tldw_chatbook.app' not in sys.modules
selected.write_text('[general]\nusers_name="compact-owner"\n[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n[api_settings.openai]\napi_key="compact-fixture-not-a-real-key"\n[providers]\nopenai=["compact-first","compact-second"]\n[chat_defaults]\nprovider="openai"\nmodel="compact-second"\n', encoding='utf-8')
selected.chmod(0o600)


def external_editor():
    original = selected.read_bytes()
    temporary = selected.with_name(selected.name + '.compact-owner-edit')
    try:
        with temporary.open('xb') as output:
            output.write(original + b'\n# compact owned native reader\n')
        temporary.chmod(0o600)
        os.replace(temporary, selected)
    finally:
        if temporary.exists(): temporary.unlink()


def custom_body_read(*args, **kwargs):
    import sys, threading
    frame = sys._getframe(1)
    _compact_fixture_all_calls.append(threading.current_thread() is threading.main_thread())
    if frame.f_locals.get('self') is _compact_fixture_widget:
        _compact_fixture_hits.append(threading.current_thread() is threading.main_thread())
    return {'custom': ['custom-first']}


async def run():
    from tldw_chatbook import config
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Widgets import compact_model_bar as module
    from textual.widgets import Select, Input
    from Tests.Performance._compact_setup_ownership_controls import CompactSetupControlGate
    app = TldwCli()
    runtime = app.console_runtime
    original_mapping = app.app_config
    original_reader = module.get_cli_providers_and_models
    body = config._load_settings_guarded.__wrapped__
    original_body_code = body.__code__
    original_reader_code = original_reader.__code__
    original_capsule = module._CONFIG_SOURCE
    gate = None
    widget = None
    remove_job = None
    dispose_job = None
    custom_hits = []
    custom_all_calls = []
    facts = {}
    receipt = None

    def custom_reader():
        frame = sys._getframe(1)
        if frame.f_locals.get('self') is widget:
            custom_hits.append(threading.current_thread() is threading.main_thread())
        return {'custom': ['custom-first']}

    async def action(observer):
        nonlocal remove_job, dispose_job
        request = observer.entry_request if observer.queued else observer.worker_request
        assert request is not None and request.widget is widget
        assert request.task is not None and not request.task.done()
        read = observer.owned_reads[id(request)]
        assert not read._producer.done() and not read.retired.done()
        assert read in request.reads and read in runtime._preparation_reads
        if not observer.queued:
            assert observer.operation in observer.raw._states
            assert all(lease in observer.storage._live_leases for lease in observer.leases)
        if outcome == 'mapping':
            app.app_config = dict(original_mapping)
        elif outcome in {'body', 'queued_body'}:
            body.__code__ = custom_body_read.__code__
        elif outcome == 'queued_reader':
            module.get_cli_providers_and_models = custom_reader
        elif outcome == 'choice':
            widget.query_one('#compact-api-provider', Select).value = 'openai'
            widget.query_one('#compact-api-model', Select).value = 'compact-first'
            widget.query_one('#compact-temperature', Input).value = '0.9'
        elif outcome == 'recancel':
            mount_task = observer.mount_task
            assert mount_task is request.task
            assert type(mount_task) is asyncio.Task and not mount_task.done()
            mount_task.cancel()
            await asyncio.sleep(0)
            mount_task.cancel()
            await asyncio.sleep(0)
            assert not mount_task.done() and not request.task.done()
            assert widget._compact_setup_request is request
            assert observer.operation in observer.raw._states
            assert all(lease in observer.storage._live_leases for lease in observer.leases)
            facts['repeated_cancel_retains_actual_callback_and_native_scope'] = True
        elif outcome == 'remove':
            # Original synchronous prune marks retirement before its queued
            # Prune message can reach the pump awaiting this held callback.
            removal = app._prune(widget, parent=widget.parent)
            assert widget._pruning and not widget._closing and not widget._closed
            assert widget.parent is request.parent
            async def remove():
                await removal
            remove_job = asyncio.create_task(remove())
            await asyncio.sleep(0)
            assert not request.task.done()
            facts['actual_remove_started_while_original_callback_alive'] = True
        elif outcome == 'closing':
            dispose_job = asyncio.create_task(app.on_shutdown_request())
            await asyncio.sleep(0)
            assert app._shutting_down and not runtime._disposed
            assert not read._producer.done() and not read.retired.done()
            facts['original_shutdown_admitted_before_Runtime_disposal'] = True
        elif outcome == 'dispose':
            dispose_job = asyncio.create_task(runtime.dispose())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert runtime._disposed and not dispose_job.done()
            read = observer.owned_reads[id(request)]
            assert read in runtime._preparation_reads and not read.retired.done()
            facts['Runtime_disposal_waits_for_original_native_reader'] = True
        elif outcome not in {'normal', 'missing_providers', 'missing_defaults'}:
            raise AssertionError('unexpected_control_route')
        facts['actual_callback_held_at_intervention'] = not request.task.done()

    try:
        async with app.run_test(size=(140, 42)) as pilot:
            while not getattr(app, '_initial_screen_pushed', False):
                await asyncio.sleep(.01)
            screen = app.screen
            assert type(screen).__name__ == 'ChatScreen'
            old = screen.query_one(module.CompactModelBar)
            parent = old.parent
            assert parent is not None
            # The real startup latch precedes completion of its owned Task.
            # Retire only these exact existing producers; no cache/guard reset.
            loop = asyncio.get_running_loop()
            startup = app._initial_screen_setup_task
            assert type(startup) is asyncio.Task and startup.get_loop() is loop
            await asyncio.wait_for(asyncio.shield(startup), timeout=10)
            assert startup.done() and not startup.cancelled()
            prior_request = getattr(old, '_compact_setup_request', None)
            prior_task = None
            if prior_request is not None:
                from tldw_chatbook.Widgets import compact_model_bar as compact_source
                assert type(prior_request) is compact_source._CompactSetup
                assert prior_request.widget is old and prior_request.app is app
                assert prior_request.loop is loop
                prior_task = prior_request.task
                if prior_task is not None:
                    assert type(prior_task) is asyncio.Task and prior_task.get_loop() is loop
                    from tldw_chatbook.Chat.console_preparation_reads import drain_preparation_reads
                    async def retire_original_bar():
                        assert not await drain_preparation_reads(prior_request.reads)
                        assert prior_request.finished
                    await asyncio.wait_for(retire_original_bar(), timeout=10)
            from tldw_chatbook import config as config_source
            prearm_facts = {
                'exact_initial_owned_Task_retired': startup.done(),
                'original_bar_issued_request_present': prior_request is not None,
                'original_bar_issued_Task_present': prior_task is not None,
                'original_bar_issued_Task_retired': prior_request is None or (prior_request.finished and not prior_request.reads),
                'app_mapping_is_original_current_cache': app.app_config is config_source._SETTINGS_CACHE,
                'mapping_or_cache_overridden': False,
                'not_a_generic_idle_or_permission_proof': True,
            }
            # This test owns a model-reader lifecycle intervention, not the
            # separate first-use skill-trust workload. Finish that exact
            # original dependency through its public async preparation seam.
            trust = await asyncio.wait_for(app.ensure_local_skill_trust_service(), timeout=10)
            assert trust is app._local_skill_trust_service and trust is not None
            prearm_facts['original_skill_trust_dependency_prepared'] = True
            await old.remove()
            if outcome in {'missing_providers', 'missing_defaults'}:
                app.app_config = dict(original_mapping)
                app.app_config.pop('providers' if outcome == 'missing_providers' else 'chat_defaults', None)
            widget = module.CompactModelBar(app, id='compact-owned-control')
            assert not custom_body_read.__code__.co_freevars
            assert not body.__code__.co_freevars and not original_reader.__code__.co_freevars
            assert '_compact_fixture_widget' not in config.__dict__ and '_compact_fixture_hits' not in config.__dict__
            config._compact_fixture_widget = widget
            config._compact_fixture_hits = custom_hits
            assert '_compact_fixture_all_calls' not in config.__dict__
            config._compact_fixture_all_calls = custom_all_calls
            if outcome == 'custom_reader':
                module.get_cli_providers_and_models = custom_reader
            elif outcome == 'custom_body':
                # Preserve the same stock public function object, change only
                # its installed body before optional selection.
                original_reader.__code__ = custom_body_read.__code__
            elif outcome == 'malformed':
                module._CONFIG_SOURCE = ()
            else:
                gate = CompactSetupControlGate(widget, asyncio.get_running_loop(), selected, action, queued=outcome.startswith('queued'))
                gate.install()  # Source/native preparation precedes real edit.
            try:
                await parent.mount(widget)
                if remove_job is not None:
                    await remove_job
                if dispose_job is not None:
                    await dispose_job
                if gate is not None:
                    assert gate.entered.is_set() and not gate.invalid, (gate.entered.is_set(), tuple(gate.invalid))
                    assert gate.action_task is not None and gate.action_task.done()
                    assert facts.get('actual_callback_held_at_intervention') is True
                    request = gate.entry_request if gate.queued else gate.worker_request
                    assert request is not None and gate._physically_retired(request)
                    facts['exact_original_callback_producer_and_retirement_done_before_checks'] = True
                    read = gate.owned_reads[id(request)]
                    if outcome in {'mapping', 'body', 'queued_reader', 'queued_body', 'closing'}:
                        assert not read._producer.cancelled()
                        assert type(read._producer.exception()) is module._CompactSetupChanged
                        facts['actual_private_owner_source_refusal'] = True
                    if outcome == 'closing':
                        late = module.CompactModelBar(app)
                        try:
                            module._capture_setup(late)
                        except module._CompactSetupChanged:
                            facts['late_stock_capture_refuses_instead_of_direct_fallback'] = True
                        else:
                            raise AssertionError('closing_stock_capture_was_treated_as_custom')
                    if outcome in {'body', 'queued_body'}:
                        assert not custom_all_calls
                        facts['replacement_body_never_executed'] = True
                    elif outcome not in {'mapping', 'queued_reader', 'recancel', 'remove', 'dispose', 'closing'}:
                        assert read._producer.exception() is None
                    if gate.queued:
                        assert gate.actual_entry_facts and gate.reader_started == 0
                        assert gate.scope_facts is None and not gate.body_returned
                        assert not custom_hits
                    else:
                        assert gate.body_returned and gate.scope_facts
                    if outcome in {'mapping', 'body', 'choice', 'recancel', 'remove', 'queued_reader', 'queued_body', 'dispose', 'closing'}:
                        assert gate.populate_started == gate.host_sync_started == 0
                        facts['no_retiring_or_changed_default_publication_or_host_sync'] = True
                    if outcome in {'mapping', 'body', 'queued_reader', 'queued_body'}:
                        assert widget.query_one('#compact-api-provider', Select).value is Select.NULL
                        assert widget.query_one('#compact-api-model', Select).value is Select.NULL
                    elif outcome == 'choice':
                        assert widget.query_one('#compact-api-provider', Select).value == 'openai'
                        assert widget.query_one('#compact-api-model', Select).value == 'compact-first'
                        assert widget.query_one('#compact-temperature', Input).value == '0.9'
                    elif outcome in {'normal', 'missing_providers'}:
                        assert widget.query_one('#compact-api-provider', Select).value == 'openai'
                        assert widget.query_one('#compact-api-model', Select).value == 'compact-second'
                    elif outcome == 'missing_defaults':
                        assert widget.query_one('#compact-api-provider', Select).value is Select.NULL
                        values = [value for _, value in widget.query_one('#compact-api-provider', Select)._options if isinstance(value, str)]
                        assert 'openai' in values
                else:
                    assert widget._compact_setup_request is None
                    values = [value for _, value in widget.query_one('#compact-api-provider', Select)._options if isinstance(value, str)]
                    if outcome in {'custom_reader', 'custom_body'}:
                        assert values == ['custom'] and custom_hits == [True, True]
                        assert widget.query_one('#compact-api-provider', Select).value is Select.NULL
                    else:
                        assert 'openai' in values
                        assert widget.query_one('#compact-api-provider', Select).value == 'openai'
                    facts['original_direct_custom_or_malformed_ABI'] = True
            finally:
                app.app_config = original_mapping
                body.__code__ = original_body_code
                original_reader.__code__ = original_reader_code
                module.get_cli_providers_and_models = original_reader
                module._CONFIG_SOURCE = original_capsule
                assert config._compact_fixture_widget is widget and config._compact_fixture_hits is custom_hits
                assert config._compact_fixture_all_calls is custom_all_calls
                del config._compact_fixture_widget, config._compact_fixture_hits, config._compact_fixture_all_calls
                if gate is not None:
                    gate.release.set()
        assert runtime._disposed and app.console_runtime is None
        facts['normal_App_survival_and_original_Runtime_disposal'] = True
    finally:
        if gate is not None:
            receipt = gate.close()
        else:
            receipt = {'original_direct_compatibility_route': True}
        receipt['outcome'] = outcome
        receipt['facts'] = facts
        receipt['fixture_prearm'] = prearm_facts if 'prearm_facts' in locals() else None
        receipt['normal_original_runtime_disposal'] = runtime._disposed
        receipt['network_refusals'] = len(network_guard.blocked_attempts())
        receipt['profile_refusals'] = len(real_profile_guard.take_violations())
        (selected.parent.parent / 'compact-setup-ownership.json').write_text(json.dumps(receipt), encoding='utf-8')
    assert facts['normal_App_survival_and_original_Runtime_disposal']
    assert receipt['network_refusals'] == receipt['profile_refusals'] == 0
    if gate is not None:
        assert receipt['complete'] and receipt['original_source_current'] and not receipt['invalid'], receipt
        assert receipt['global_events'] == 0 and receipt['hooks_retired_before_inactive']
        assert receipt['controller_retired'] and receipt['control_UI_Task_retired']
        assert receipt['captured_mount_Task_retired'] and receipt['queued_callback_Task_retired']
        if not gate.queued:
            assert receipt['original_body_normal_RETURN'] and receipt['original_raw_scope_and_leases_retired']
    print('retired and reopened')


with user_fixture_default_owner():
    asyncio.run(asyncio.wait_for(run(), timeout=240))
"""


@pytest.mark.parametrize(
    "outcome",
    (
        "normal",
        "mapping",
        "body",
        "choice",
        "recancel",
        "remove",
        "queued_reader",
        "queued_body",
        "custom_reader",
        "custom_body",
        "malformed",
        "missing_providers",
        "missing_defaults",
        "dispose",
        "closing",
    ),
)
def test_compact_setup_retains_owner_source_choices_and_native_callback(
    tmp_path, outcome
):
    _run(tmp_path, "compact_models", outcome, script=_SCRIPT, timeout=240)
