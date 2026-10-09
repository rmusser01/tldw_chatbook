"""Original cold Console composition must not block its actual running loop."""

import json
import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio
import json
import os
from pathlib import Path
import sys

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
root = Path(os.environ['XDG_DATA_HOME']).absolute()
selector.write_text('[general]\nusers_name="startup-census"\n'
                    '[first_run]\nsetup_completed=true\n'
                    '[_first_run]\nsetup_completed=true\n'
                    '[splash_screen]\nenabled=false\n'
                    '[api_settings.openai]\n'
                    'api_key="sk-census-000000000000000000000000000000000000"\n', encoding='utf-8')
selector.chmod(0o600)
repo = Path.cwd().absolute()
assert 'tldw_chatbook.app' not in sys.modules
assert 'tldw_chatbook.DB.AgentRuns_DB' not in sys.modules
from Tests.Performance._startup_receipt_native_gate import StartupReceiptNativeGate, physical_open

async def run():
    gate = StartupReceiptNativeGate(repo, asyncio.get_running_loop())
    gate.start()
    app = runtime = None
    try:
        from tldw_chatbook.app import TldwCli
        app = TldwCli()
        runtime = app.console_runtime
        assert type(runtime) is sys.modules['tldw_chatbook.Chat.console_runtime'].ConsoleRuntime
        async with app.run_test(size=(140, 42)) as pilot:
            # run_test may yield while the original initial-screen task awaits
            # finite receipt I/O. The enclosing original 240s deadline still owns
            # this prerequisite and every subsequent original action/assertion.
            while not getattr(app, '_initial_screen_pushed', False):
                await asyncio.sleep(.01)
            screen = app.screen
            assert type(screen).__name__ == 'ChatScreen'
            composer = screen._console_composer_or_none()
            assert composer is not None
            before = composer.draft_text()
            composer.focus()
            caret = composer._cursor_index
            await pilot.press('k')
            assert composer.draft_text() == before[:caret] + 'k' + before[caret:]
    finally:
        receipt = gate.stop()
        receipt['held_connection_physically_closed_after_original_shutdown'] = (
            gate.held_connection is not None and not physical_open(gate.held_connection)
        )
        receipt['normal_runtime_disposal'] = bool(runtime is not None and runtime._disposed
            and (app.console_runtime is None or app.console_runtime is runtime))
        receipt['network_attempts'] = len(network_guard.blocked_attempts())
        receipt['profile_refusals'] = len(real_profile_guard.take_violations())
        (selector.parent.parent / 'cold-receipt-native.json').write_text(json.dumps(receipt), encoding='utf-8')
    assert receipt['original_source_current'] and not receipt['invalid'] and receipt['overflow'] == 0, receipt
    assert receipt['global_events_zero'] and receipt['local_masks_retired'] and receipt['tool_retired'], receipt
    assert receipt['callbacks_owned'] and receipt['local_masks_owned'], receipt
    assert receipt['hold'] is not None and receipt['hold']['physically_open_at_hold'], receipt
    assert receipt['normal_runtime_disposal'] and receipt['held_connection_physically_closed_after_original_shutdown'], receipt
    assert receipt['exact_held_native_and_lease_retired'], receipt
    assert receipt['network_attempts'] == receipt['profile_refusals'] == 0, receipt
    assert receipt['ui_progress_while_original_native_held'], 'Original receipt-schema SQL initialization blocked the actual startup loop'

with user_fixture_default_owner():
    asyncio.run(asyncio.wait_for(run(), timeout=240))
print('retired and reopened')
"""


def test_original_cold_console_receipt_sql_keeps_its_actual_loop_responsive(tmp_path):
    _run(tmp_path, "cold_receipts", "ownership", script=_SCRIPT, timeout=240)
    receipt = json.loads(
        (tmp_path / "cold-receipt-native.json").read_text(encoding="utf-8")
    )
    assert receipt["ui_progress_while_original_native_held"]
