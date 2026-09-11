"""Canonical F9 composition uses actual services in a fresh selected process."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


@pytest.fixture(autouse=True)
def _disable_model_catalog_refresh():
    """The child installs its network guard before importing the actual app."""


_ENTRY = r"""
import asyncio, os, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
for name in ('sounddevice', 'pyaudio'): sys.modules[name] = None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
selector = Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_text('[general]\nusers_name="default_user"\ndefault_tab="settings"\n[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n')
selector.chmod(0o600)
from tldw_chatbook.app import TldwCli, SettingsProvider
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService, default_control_root
async def main():
    app = TldwCli()
    app.app_config['_first_run'] = False
    app.app_config.setdefault('first_run', {})['setup_completed'] = True
    assert getattr(app, '_recovery_service', None) is None
    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press('f9')
        for _ in range(100):
            if isinstance(app.screen, SettingsScreen): break
            await pilot.pause(.05)
        assert isinstance(app.screen, SettingsScreen), type(app.screen)
        button = app.screen.query_one('#settings-backup-restore')
        button.scroll_visible(immediate=True)
        await pilot.pause()
        assert await pilot.click('#settings-backup-restore'), (button.region, button.visible)
        await pilot.pause()
        assert isinstance(app.screen, BackupRestoreScreen), type(app.screen)
        service = app.recovery_service
        assert isinstance(service, RecoveryService)
        assert app.screen.service is service
        assert app.screen.config_paths == (selector,)
        assert service.control_root == default_control_root()
        await pilot.press('escape')
        assert isinstance(app.screen, SettingsScreen)
        provider = SettingsProvider(app.screen)
        assert any('Backup & Restore' in str(hit.text) for hit in [item async for item in provider.discover()])
        provider.handle_setting('backup_restore')
        await pilot.pause()
        assert app.screen.service is service
    assert service._closed
    assert not blocked_attempts(), blocked_attempts()
    print('retired and reopened')
asyncio.run(main())
"""


def test_canonical_f9_entry_reuses_app_service_and_shutdown_closes_it(tmp_path):
    _run(tmp_path, "entry", "normal", script=_ENTRY, timeout=120)


_BACKUP = (
    _ENTRY.split("async def main():", 1)[0]
    + r"""
from textual.widgets import Input, Button, Static
from tldw_chatbook.Backup_Recovery import archive_reader, storage_admission
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from threading import Event
async def main():
    app = TldwCli()
    app.app_config['_first_run'] = False
    app.app_config.setdefault('first_run', {})['setup_completed'] = True
    destination = Path.home() / 'mounted.tldw-backup.zip'
    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press('f9')
        await pilot.pause()
        assert await pilot.click('#settings-backup-restore')
        await pilot.pause()
        await pilot.click('#backup-open-create')
        screen = app.screen
        screen.query_one('#backup-destination', Input).value = str(destination)
        await pilot.pause()
        await pilot.click('#backup-review')
        async with asyncio.timeout(30):
            while screen.query_one('#backup-create', Button).disabled:
                await asyncio.sleep(.05)
                message = str(screen.query_one('#backup-message', Static).render())
                assert 'failed' not in message.lower(), message
        assert 'Complete coverage' in str(screen.query_one('#backup-coverage', Static).render())
        print('checkpoint: review complete', flush=True)
        await pilot.click('#backup-create')
        service = app.recovery_service
        assert service.current(), str(screen.query_one('#backup-message', Static).render())
        operation = service.current()['operation_id']
        print('checkpoint: accepted', dict(service.current()), flush=True)
        await pilot.press('escape')
        assert isinstance(app.screen, SettingsScreen)
        async with asyncio.timeout(75):
            while service.status(operation)['state'] == 'running':
                await asyncio.sleep(.05)
        result = service.status(operation)
        print('checkpoint: terminal', dict(result), flush=True)
        assert result['state'] == 'succeeded', dict(result)
        assert result['result']['archive_verified'] and result['result']['path'] == str(destination)
        assert storage_admission._pause is None
        app.chachanotes_db.add_note('After mounted backup', 'Ordinary native writer resumed.')
        await pilot.click('#settings-backup-restore')
        await pilot.pause()
        assert str(destination) in str(app.screen.query_one('#backup-status', Static).render())
        verified = await asyncio.to_thread(archive_reader.acquire, destination, password=None,
            work_root=Path.home() / 'mounted-readback', limits=ArchiveLimits(), cancel=Event())
        assert archive_reader.verify_sealed(verified).consistency == 'coherent'
    assert not blocked_attempts(), blocked_attempts()
    print('retired and reopened')
asyncio.run(main())
"""
)


@pytest.fixture
def retained_child_output(tmp_path, monkeypatch):
    """Keep the primary native error even if production shutdown also fails."""
    import subprocess

    run = subprocess.run

    def retained_output(*args, **kwargs):
        result = run(*args, **kwargs)
        (tmp_path / "child-output.log").write_text(result.stdout + "\n" + result.stderr)
        return result

    monkeypatch.setattr(subprocess, "run", retained_output)


def test_actual_mounted_backup_publishes_verified_archive_after_navigation(
    tmp_path, retained_child_output
):
    _run(tmp_path, "backup", "normal", script=_BACKUP, timeout=120)


_ISOLATED = (
    _ENTRY.split("async def main():", 1)[0]
    + r"""
from textual.widgets import Input, Button, Static
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog
def config_manifest(doc):
    doc['owners'][0]['owner_id'] = 'config'
    doc['files'][0].update(owner_id='config', logical_id='profile:profile:config', relative_path='config.toml')
    doc['dependency_groups'][0]['members'] = ['profile:profile:config']
archive = sealed(Path.home(), mutate=config_manifest, data=b'[general]\nusers_name="original"\n')
parent = Path.home() / 'chosen-destinations'
parent.mkdir(mode=0o700)
async def main():
    app = TldwCli()
    app.app_config['_first_run'] = False
    app.app_config.setdefault('first_run', {})['setup_completed'] = True
    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press('f9')
        await pilot.pause()
        await pilot.click('#settings-backup-restore')
        await pilot.pause()
        await pilot.click('#backup-open-inspect')
        screen = app.screen
        screen.query_one('#backup-source', Input).value = str(archive.path)
        await pilot.click('#backup-inspect')
        async with asyncio.timeout(15):
            while not screen.query_one('#backup-restore-form').display:
                await asyncio.sleep(.03)
        for index, slot in enumerate(screen._inspection_summary['destination_slots']):
            screen.query_one(f'#backup-root-{index}', Input).value = str(parent / ('data' if slot['kind'] == 'data_root' else 'config'))
        screen.query_one('#backup-profile-name-0', Input).value = 'Recovered UI profile'
        review = screen.query_one('#backup-review-restore', Button)
        review.focus()
        await pilot.pause()
        await pilot.press('enter')
        async with asyncio.timeout(20):
            while screen.query_one('#backup-start-restore', Button).disabled:
                await asyncio.sleep(.03)
                text = str(screen.query_one('#backup-restore-preview', Static).render())
                assert 'refused' not in text, text
        assert 'Restore:' in str(screen.query_one('#backup-restore-preview', Static).render())
        print('checkpoint: isolated plan reviewed', flush=True)
        screen.query_one('#backup-start-restore', Button).focus()
        await pilot.press('enter')
        service = app.recovery_service
        operation = service.current()['operation_id']
        async with asyncio.timeout(35):
            while service.status(operation)['state'] == 'running':
                await asyncio.sleep(.03)
        state = service.status(operation)
        print('checkpoint: isolated terminal', dict(state), flush=True)
        assert state['state'] == 'succeeded' and state['result']['restoration_validated'], dict(state)
        await pilot.pause()
        async with asyncio.timeout(5):
            while 'Restoration validated' not in str(screen.query_one('#backup-status', Static).render()):
                await asyncio.sleep(.03)
        assert 'Restoration validated' in str(screen.query_one('#backup-status', Static).render())
        config_path, data_path = ProfileCatalog(service.control_root).resolve(state['result']['profile_id'])
        assert config_path == parent / 'config' / 'config.toml'
        assert data_path == parent / 'data'
        import tomllib
        assert tomllib.loads(config_path.read_text())['general']['users_name'] == 'Recovered UI profile'
        assert not state['result'].get('opened', False)
    assert not blocked_attempts(), blocked_attempts()
    print('retired and reopened')
asyncio.run(main())
"""
)


def test_mounted_isolated_restore_uses_reviewed_local_destination_slots(tmp_path, retained_child_output):
    _run(tmp_path, "isolated", "normal", script=_ISOLATED, timeout=120)
