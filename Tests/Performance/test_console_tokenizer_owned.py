"""Original tokenizer admission/loading runs off the mounted Console loop."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio
import os
from pathlib import Path
import sys
import threading
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

network_guard.install()
real_profile_guard.install()
os.environ['PYTHON_KEYRING_BACKEND'] = 'keyring.backends.null.Keyring'

async def until(predicate):
    async with asyncio.timeout(60):
        while not predicate():
            await asyncio.sleep(.01)

async def exercise():
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.Utils import custom_tokenizers as custom
    from tldw_chatbook.Chat import console_preparation_reads as reads

    app = TldwCli()
    assert custom._tokenizer_manager is None
    entered, release = threading.Event(), threading.Event()
    held = []
    main = threading.get_ident()
    # This original method starts AFTER __init__'s original acquire_storage
    # has exited successfully. No native check/callback is replaced.
    target = custom.CustomTokenizerManager._load_mappings.__code__
    def hold(code, offset):
        if entered.is_set():
            return
        assert threading.get_ident() != main, 'cold tokenizer loaded on UI loop'
        binding = reads._source_binding.get()
        assert binding is not None, 'unowned native read'
        held.append(binding.read)
        entered.set()
        assert release.wait(15), 'native hold watchdog expired'
    tool_id = 4
    sys.monitoring.use_tool_id(tool_id, 'tokenizer-native-control')
    sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_START, hold)
    sys.monitoring.set_local_events(tool_id, target, sys.monitoring.events.PY_START)
    try:
        async with app.run_test(size=(170, 48)) as pilot:
            await until(lambda: getattr(app, '_ui_ready', False))
            app.post_message(NavigateToScreen('chat'))
            await until(entered.is_set)
            assert held[0] in app.console_runtime._preparation_reads
            await asyncio.sleep(.02)
            release.set()
            await until(lambda: held[0] not in app.console_runtime._preparation_reads)
            await until(lambda: type(app.screen).__name__ == 'ChatScreen')
            projection = app.screen._console_readiness_config_projection
            await until(lambda: projection._tokenizer_prepared is not None)
            assert custom._tokenizer_manager is not None
            composer = app.screen.query_one('#console-native-composer')
            composer.focus_draft_from(None)
            await pilot.press('a')
            await pilot.pause(.3)
            assert composer.draft_text() == 'a'
    finally:
        release.set()
        sys.monitoring.set_local_events(tool_id, target, 0)
        sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_START, None)
        sys.monitoring.free_tool_id(tool_id)
    assert not network_guard.blocked_attempts()
    assert not real_profile_guard.take_violations()
    print('retired and reopened')

with user_fixture_default_owner():
    Path(os.environ['TLDW_CONFIG_PATH']).write_text(
        '[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n'
        '[chat_defaults]\nprovider="llama_cpp"\nmodel="local-model"\n'
        '[api_settings.llama_cpp]\napi_url="http://127.0.0.1:9099"\nmodel="local-model"\n',
        encoding='utf-8',
    )
    asyncio.run(exercise())
"""


def test_original_tokenizer_admission_is_off_loop_and_owned(tmp_path):
    launch = (
        "from Tests.Performance.test_console_tokenizer_owned import _SCRIPT; "
        "exec(compile(_SCRIPT, '<tokenizer-control>', 'exec'))"
    )
    _run(tmp_path, "tokenizer", "owned", script=launch, timeout=150)
