"""One guidance update presents one fresh native readiness result."""

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_GUIDANCE = r'''
import asyncio
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from Tests.network_guard import install, blocked_attempts
install()
for name in ('sounddevice', 'pyaudio'):
    sys.modules[name] = None
for name in ('OPENAI_API_KEY', 'TLDW_CONSOLE_LLAMA_CPP_BASE_URL'):
    os.environ.pop(name, None)
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
selector = Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_text('[general]\nusers_name="test"\ndefault_tab="settings"\n'
    '[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n'
    '[chat_defaults]\nprovider="openai"\nmodel="gpt-4o"\n'
    '[api_settings.openai]\napi_key=""\n')
selector.chmod(0o600)
def phase(label):
    print('guidance_phase=' + label, flush=True)
    try:
        with (Path.home() / 'guidance-phases.log').open('a', encoding='utf-8') as output:
            output.write(label + '\n')
    except OSError:
        return  # Observation must not change the native operation's outcome.
phase('import_begin')
from tldw_chatbook import config
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.UI.Screens import chat_screen
phase('import_complete')

def check_display_pairs(screen):
    derives, actions, observed = [], [], []
    original = screen._active_console_settings_readiness_uncached
    original_action = screen._console_provider_recovery_action
    def derive():
        pair = original()
        derives.append(pair)
        return pair
    def action(**kwargs):
        actions.append(True)
        return original_action(**kwargs)
    screen._active_console_settings_readiness_uncached = derive
    screen._console_provider_recovery_action = action
    def inspect_pair(status, setup):
        control = screen._build_console_control_state(None)
        derives.clear()
        actions.clear()
        workbench = screen._build_console_workbench_state(control)
        observed.append((len(derives), len(actions)))
        assert next(m for m in workbench.modes if m.id == 'provider').status == status
        assert next(a for a in workbench.actions if a.id == 'send').disabled
        assert workbench.recovery is None
        derives.clear()
        actions.clear()
        inspector = screen._build_console_inspector_state(None)
        observed.append((len(derives), len(actions)))
        rows = {row.label: row.value for row in inspector.rows}
        assert ('Setup' in rows) == setup
        assert ('Next action' in rows) == setup
        if setup:
            assert rows['Next action'] == 'Set up provider'
            assert rows['Blocked impact'] == 'Send is blocked until setup is finished.'
        assert screen._console_derivation_memo is None

    inspect_pair('blocked', True)
    assert config.save_settings_to_cli_config({
        'api_settings.openai': {'api_key': 'synthetic-guidance-test-value'}})
    inspect_pair('ready', False)
    run_active = screen._console_run_active
    screen._console_run_active = lambda: True
    try:
        inspect_pair('ready', False)
    finally:
        screen._console_run_active = run_active

    # A real save after display copy is derived must still affect the later
    # send guard. The display result cannot become sending authority.
    composer = screen._console_composer_or_none
    def draft():
        assert config.save_settings_to_cli_config({
            'api_settings.openai': {'api_key': ''}})
        return 'draft after credential removal'
    control = screen._build_console_control_state(None)
    screen._console_composer_or_none = lambda: SimpleNamespace(draft_text=draft)
    derives.clear()
    actions.clear()
    try:
        workbench = screen._build_console_workbench_state(control)
        observed.append((len(derives), len(actions)))
        assert derives[0][1].native_send_supported
        assert derives[-1][1].recovery_action == 'configure_credential'
        assert next(a for a in workbench.actions if a.id == 'send').disabled
    finally:
        screen._console_composer_or_none = composer

    # Keep acquisition/projection errors and recover through a fresh build.
    for attribute in ('_active_console_settings_readiness',
                      '_console_provider_blocker_copy',
                      '_console_provider_recovery_action'):
        method = getattr(screen, attribute)
        failure = RuntimeError('display pair observation failure')
        def fail(**kwargs):
            method(**kwargs)
            raise failure
        setattr(screen, attribute, fail)
        try:
            screen._build_console_inspector_state(None)
        except RuntimeError as error:
            assert error is failure
        else:
            raise AssertionError('inspector failure was swallowed')
        finally:
            setattr(screen, attribute, method)
        assert screen._console_derivation_memo is None
        derives.clear()
        assert any(r.label == 'Setup' for r in screen._build_console_inspector_state(None).rows)
        assert derives
    assert observed == [(1, 0), (2, 1), (1, 0), (2, 0),
                        (1, 0), (2, 0), (2, 0)], observed

async def main():
    phase('construct_begin')
    app = TldwCli()
    phase('construct_complete')
    with app._context():
        try:
            phase('context_ready')
            screen = chat_screen.ChatScreen(app)
            store = screen._ensure_console_chat_store()
            assert store.active_session_id is None
            if sys.argv[2] == 'display_pairs':
                check_display_pairs(screen)
                phase('assertions_complete')
                return
            derives, events, projections, loads = [], [], [], []
            original = screen._active_console_settings_readiness_uncached
            original_load = chat_screen.load_settings
            def derive():
                pair = original()
                derives.append(pair)
                events.append('readiness')
                return pair
            def load(*args, **kwargs):
                loads.append(True)
                return original_load(*args, **kwargs)
            screen._active_console_settings_readiness_uncached = derive
            chat_screen.load_settings = load
            # Observe the two UI consumers without mounting another Console or
            # starting discovery. Session/config/readiness and all projections run.
            def consume(consumer, card, **kwargs):
                assert screen._console_derivation_memo is None
                events.append(consumer)
                projections.append((consumer, card, kwargs))
            screen.query_one = lambda *args: SimpleNamespace(
                sync_inline_guidance=lambda card, **kwargs: consume('surface', card, **kwargs))
            screen._sync_console_setup_modal = lambda card, **kwargs: consume('modal', card, **kwargs)
            first_send = screen._console_first_send_completed
            messages = screen._message._active_console_transcript_has_messages
            def first():
                events.append('first_send')
                return first_send()
            def has_messages():
                events.append('messages')
                return messages()
            screen._console_first_send_completed = first
            screen._message._active_console_transcript_has_messages = has_messages
            counts = []
            def sync():
                derives.clear()
                events.clear()
                projections.clear()
                loads.clear()
                screen._sync_console_transcript_guidance()
                counts.append(len(derives))
                assert loads, 'full native-backed config acquisition must still run'
                assert events[-4:] == ['first_send', 'messages', 'surface', 'modal'], events
                assert projections[0][1] == projections[1][1]
                assert screen._console_derivation_memo is None
                return projections[0][1], derives[-1]
            try:
                card, (settings, readiness) = sync()
                assert card.mode == 'card'
                assert readiness.recovery_action == 'configure_credential'
                assert settings.provider == 'openai'
                session = store.ensure_session()
                assert session.settings.provider == 'openai'
                assert projections[0][2]['provider_action_label'] == 'Set up provider'

                # The real Settings write/readback must converge this still-unused
                # blocked session before the next pass projects any guidance.
                assert config.save_settings_to_cli_config({
                    'chat_defaults': {'provider': 'llama_cpp', 'model': 'local-model'},
                    'api_settings.llama_cpp': {
                        'api_url': 'http://127.0.0.1:9099', 'model': 'local-model'}})
                assert config.load_settings()['chat_defaults']['provider'] == 'llama_cpp'
                card, (settings, readiness) = sync()
                assert settings.provider == 'llama_cpp'
                assert settings.model == 'local-model'
                assert readiness.native_send_supported
                assert card.mode != 'card'
                assert session.settings.provider == 'llama_cpp'
                assert session.settings == session.canonical_settings_baseline
                assert not session.has_user_work
                assert projections[0][2]['provider_action_label'] == ''

                # Explicit settings remain explicit even when configured defaults
                # are sendable. Subsequent config saves must refresh every helper.
                chosen = ConsoleSessionSettings(provider='openai', model='gpt-4o', source='user')
                store.replace_session_settings(session.id, chosen)
                card, (_, readiness) = sync()
                assert card.mode == 'card'
                assert readiness.recovery_action == 'configure_credential'
                assert session.settings == chosen
                assert config.save_settings_to_cli_config({
                    'api_settings.openai': {'api_key': 'synthetic-guidance-test-value'}})
                derives.clear()
                assert screen._console_provider_blocker_copy() == ''
                assert screen._console_provider_recovery_action()[1] == 'hidden'
                assert screen._build_console_setup_card_state().mode != 'card'
                assert len(derives) == 3, 'independent helpers must each acquire fresh state'
                card, (_, readiness) = sync()
                assert readiness.native_send_supported
                assert card.mode != 'card'
                assert session.settings == chosen
                assert counts == [1, 1, 1, 1], counts
                # Preserve the original exception and do not retain a readiness
                # result when either acquisition or a later projection fails.
                for attribute in ('_active_console_settings_readiness',
                                  '_console_first_send_completed'):
                    method = getattr(screen, attribute)
                    failure = RuntimeError('guidance observation failure')
                    def fail():
                        method()
                        raise failure
                    setattr(screen, attribute, fail)
                    projections.clear()
                    try:
                        screen._sync_console_transcript_guidance()
                    except RuntimeError as error:
                        assert error is failure
                    else:
                        raise AssertionError('guidance failure was swallowed')
                    finally:
                        setattr(screen, attribute, method)
                    assert not projections
                    assert screen._console_derivation_memo is None
                    card, (_, readiness) = sync()
                    assert readiness.native_send_supported
                    assert counts[-1] == 1
            finally:
                chat_screen.load_settings = original_load
            phase('assertions_complete')
        finally:
            await app._shutdown()
            if app._exception is not None:
                raise app._exception
asyncio.run(main())
phase('shutdown_complete')
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
'''


def test_native_guidance_is_coherent_and_refreshes_after_real_settings_save(tmp_path):
    """Repeated derives cannot triple the native work or retain stale guidance."""
    _run(tmp_path, "guidance", "freshness", script=_GUIDANCE)


def test_native_display_pairs_preserve_outputs_and_later_fresh_send_guard(tmp_path):
    """Unused/repeated display reads must not replace a later send check."""
    _run(tmp_path, "guidance", "display_pairs", script=_GUIDANCE)
