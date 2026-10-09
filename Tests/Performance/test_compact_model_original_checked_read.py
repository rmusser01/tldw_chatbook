"""Actual original CompactModelBar config read; private native profile baseline."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio, inspect, json, os, sys
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, target = sys.argv[1:]
assert route == 'compact_models' and target in {'compose', 'mount'}
selected = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
assert 'tldw_chatbook.config' not in sys.modules and 'tldw_chatbook.app' not in sys.modules
selected.write_text('[general]\nusers_name="compact-read"\n[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n[api_settings.openai]\napi_key="compact-fixture-not-a-real-key"\n[chat_defaults]\nprovider="openai"\nmodel="gpt-4o-mini"\n', encoding='utf-8')
selected.chmod(0o600)


def external_editor():
    original = selected.read_bytes()
    temporary = selected.with_name(selected.name + '.compact-compose-edit')
    try:
        with temporary.open('xb') as output:
            output.write(original + b'\n# compact compose external editor\n')
        temporary.chmod(0o600)
        os.replace(temporary, selected)
    finally:
        if temporary.exists(): temporary.unlink()


async def run():
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Widgets.compact_model_bar import CompactModelBar
    from Tests.Performance._compact_original_config_gate import OriginalCompactModelReadGate
    app = TldwCli()
    runtime = app.console_runtime
    gate = None
    receipt = None
    try:
        async with app.run_test(size=(140, 42)) as pilot:
            while not getattr(app, '_initial_screen_pushed', False):
                await asyncio.sleep(.01)
            screen = app.screen
            assert type(screen).__name__ == 'ChatScreen'
            old = screen.query_one(CompactModelBar)
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
            await old.remove()
            widget = CompactModelBar(app, id='compact-native-read-control')
            gate = OriginalCompactModelReadGate(widget, asyncio.get_running_loop(), selected, target)
            gate.install()  # All source preparation before the genuine external edit.
            try:
                await parent.mount(widget)
                assert gate.entered.is_set() and gate.body_returned
                assert gate.scope_facts is not None and not gate.invalid
                assert widget.app_instance is app and widget.parent is parent
            finally:
                gate.release.set()
        assert runtime._disposed and app.console_runtime is None
    finally:
        if gate is not None:
            receipt = gate.close()
            receipt['normal_original_runtime_disposal'] = runtime._disposed
            receipt['fixture_prearm'] = prearm_facts if 'prearm_facts' in locals() else None
            receipt['network_refusals'] = len(network_guard.blocked_attempts())
            receipt['profile_refusals'] = len(real_profile_guard.take_violations())
            (selected.parent.parent / 'compact-original-read.json').write_text(json.dumps(receipt), encoding='utf-8')
    assert receipt['complete'] and receipt['original_source_current'] and not receipt['invalid'], receipt
    assert receipt['global_events'] == 0 and receipt['hooks_retired_before_inactive']
    assert receipt['controller_retired'] and receipt['original_body_normal_RETURN']
    assert receipt['original_raw_scope_and_leases_retired'] and receipt['normal_original_runtime_disposal']
    assert receipt['network_refusals'] == receipt['profile_refusals'] == 0
    assert receipt['actual_loop_progress_while_original_checked_read_held'], 'Original compact model settings read blocked its actual shared loop'
    print('retired and reopened')


with user_fixture_default_owner():
    asyncio.run(asyncio.wait_for(run(), timeout=240))
"""


@pytest.mark.parametrize("target", ("compose", "mount"))
def test_original_compact_model_checked_read_keeps_shared_loop_responsive(
    tmp_path, target
):
    _run(tmp_path, "compact_models", target, script=_SCRIPT, timeout=240)
