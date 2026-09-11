"""Actual Console provider derivation shares only one synchronous config read."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_DERIVE = r"""
import asyncio
import os
from pathlib import Path
import sys
from Tests.network_guard import install, blocked_attempts
install()
for name in ('sounddevice', 'pyaudio'):sys.modules[name] = None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
selector = Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_text('[general]\nusers_name="test"\ndefault_tab="settings"\n'
    '[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n'
    '[chat_defaults]\nprovider="llama_cpp"\n'
    '[api_settings.llama_cpp]\nmodel="configured-before"\n')
selector.chmod(0o600)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Screens import chat_screen

async def main():
    app = TldwCli()
    async with app.run_test(size=(100,36)):
        screen = chat_screen.ChatScreen(app)
        loads = []
        original = chat_screen.load_settings
        def counted(*args, **kwargs):
            loads.append(True)
            return original(*args, **kwargs)
        chat_screen.load_settings = counted
        try:
            selection = screen._build_console_provider_selection()
            assert selection.provider == 'llama_cpp', selection
            assert selection.configured_model == 'configured-before', selection
            assert len(loads) == 1, len(loads)
            assert screen._console_derivation_memo is None
            if sys.argv[2] == 'freshness':
                from tldw_chatbook import config
                assert config.save_settings_to_cli_config(
                    {'api_settings.llama_cpp': {'model': 'configured-after'}})
                loads.clear()
                changed = screen._build_console_provider_selection()
                assert changed.configured_model == 'configured-after', changed
                assert len(loads) == 1, len(loads)
            elif sys.argv[2] == 'sessions':
                from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
                store = screen._ensure_console_chat_store()
                active = store.active_session_id
                first = store.create_session(settings=ConsoleSessionSettings(
                    provider='llama_cpp', model='first-user-model', source='user'),
                    activate=False)
                second = store.create_session(settings=ConsoleSessionSettings(
                    provider='llama_cpp', model='second-user-model', source='user'),
                    activate=False)
                loads.clear()
                with screen._console_derivation_scope():
                    one = screen._build_console_provider_selection(first.id)
                    two = screen._build_console_provider_selection(second.id)
                    assert screen._build_console_provider_selection(first.id) is one
                    assert one.explicit_model == 'first-user-model', one
                    assert two.explicit_model == 'second-user-model', two
                    assert len(loads) == 1, len(loads)
                assert store.active_session_id == active
            elif sys.argv[2] == 'exception':
                try:
                    screen._build_console_provider_selection('absent-session')
                except KeyError:
                    pass
                else:
                    raise AssertionError('missing session did not refuse')
                assert screen._console_derivation_memo is None
                loads.clear()
                with screen._console_derivation_scope():
                    outer = screen._console_derivation_memo
                    first = screen._build_console_provider_selection()
                    try:
                        screen._build_console_provider_selection('absent-session')
                    except KeyError:
                        pass
                    else:
                        raise AssertionError('nested missing session did not refuse')
                    assert screen._console_derivation_memo is outer
                    assert screen._build_console_provider_selection() is first
                    assert len(loads) == 1, len(loads)
                assert screen._console_derivation_memo is None
                loads.clear()
                screen._build_console_provider_selection()
                assert len(loads) == 1, len(loads)
        finally:
            chat_screen.load_settings = original
asyncio.run(main())
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("case", ["once", "freshness", "sessions", "exception"])
def test_actual_provider_derivation_scope(tmp_path, case):
    _run(tmp_path, "derive", case, script=_DERIVE)
