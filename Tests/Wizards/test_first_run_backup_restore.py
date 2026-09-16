"""First-run entry uses the canonical view without committing setup choices."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, os, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
selector = Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_text('[general]\nusers_name="default_user"\n')
selector.chmod(0o600)
from textual.app import App
from textual.screen import Screen
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import WelcomeStep
from tldw_chatbook.UI.Wizards.first_run_recovery_dialog import SetupRecoveryDialog
class Welcome(Screen):
    def compose(self):
        yield WelcomeStep()
class Host(App):
    def __init__(self):
        super().__init__()
        self.service = RecoveryService(Path.home() / 'control')
        self.results = []
    def on_mount(self):
        self.push_screen(Welcome() if sys.argv[1] == 'welcome' else SetupRecoveryDialog(), self.results.append)
    def action_backup_restore(self):
        self.push_screen(BackupRestoreScreen(self.service, config_paths=(selector,)))
async def main():
    app = Host()
    try:
        async with app.run_test(size=(100, 36)) as pilot:
            original = app.screen
            before = selector.read_bytes()
            button = original.query_one('#setup-backup-restore')
            button.scroll_visible(immediate=True)
            button.focus()
            await pilot.press('enter')
            await pilot.pause()
            assert isinstance(app.screen, BackupRestoreScreen), type(app.screen)
            assert app.screen.service is app.service
            assert app.service.current() is None
            await pilot.press('escape')
            assert app.screen is original
            assert not app.results
            assert selector.read_bytes() == before
        assert not blocked_attempts(), blocked_attempts()
    finally:
        app.service.close()
    print('retired and reopened')
asyncio.run(main())
"""


@pytest.mark.parametrize("surface", ["welcome", "interrupted"])
def test_restore_entry_returns_to_same_first_run_surface(tmp_path, surface):
    _run(tmp_path, surface, "normal", script=_SCRIPT)


def test_actual_app_whole_setup_wizard_opens_canonical_recovery(tmp_path):
    from Tests.ProductionApp.test_backup_restore_composition import _ENTRY

    script = (
        _ENTRY.split("async def main():", 1)[0]
        + r"""
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import FirstRunSetupWizard
async def main():
    app = TldwCli()
    app.app_config['_first_run'] = False
    app.app_config.setdefault('first_run', {})['setup_completed'] = True
    async with app.run_test(size=(100, 36)) as pilot:
        wizard = FirstRunSetupWizard(app, rerun=True)
        app.push_screen(wizard)
        await pilot.pause()
        button = wizard.query_one('#setup-backup-restore')
        button.scroll_visible(immediate=True)
        button.focus()
        before = selector.read_bytes()
        await pilot.press('enter')
        await pilot.pause()
        assert isinstance(app.screen, BackupRestoreScreen), type(app.screen)
        assert app.screen.service is app.recovery_service
        assert app.recovery_service.current() is None
        await pilot.press('escape')
        assert app.screen is wizard
        assert selector.read_bytes() == before
    assert app.recovery_service._closed
    assert not blocked_attempts(), blocked_attempts()
    print('retired and reopened')
asyncio.run(main())
"""
    )
    _run(tmp_path, "whole-wizard", "normal", script=script, timeout=120)
