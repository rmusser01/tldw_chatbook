"""Settings > Privacy & Security > Encryption card (TASK-34100.4, protect-summary-03).

Drives the real card against a real temp config file: each action is gated by
the password dialog, runs in a worker, and reports inline; the file on disk is
the evidence.
"""

from __future__ import annotations

import asyncio
import tomllib

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

import tldw_chatbook.app  # noqa: F401 -- bind the app to the session profile at
# collection; Tests/UI/conftest.py's autouse fixture patches TldwCli and would
# otherwise import it under this test's per-test profile.
from Tests.Backup_Recovery.config_test_support import install_config_source
from tldw_chatbook.UI.Screens import settings_encryption as card_module
from tldw_chatbook.UI.Screens.settings_encryption import EncryptionSettingsCard
from tldw_chatbook.Widgets.password_dialog import PasswordDialog

PLAINTEXT_KEY = "sk-proj-settings-card-plaintext-key"
PASSWORD_A = "settings-card-pw-a"
PASSWORD_B = "settings-card-pw-b"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    path = tmp_path / "profile" / "config.toml"
    path.parent.mkdir(mode=0o700, parents=True)
    path.write_text(f'[api_settings.openai]\napi_key = "{PLAINTEXT_KEY}"\n')
    path.chmod(0o600)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    module = install_config_source(monkeypatch)
    monkeypatch.setattr(module, "DEFAULT_CONFIG_PATH", tmp_path / "decoy.toml")
    module.clear_encryption_password()
    module.card_test_path = path
    yield module
    module.clear_encryption_password()


class _Host(App):
    def __init__(self, *, enabled: bool, unlocked: bool) -> None:
        super().__init__()
        self._card_args = {"enabled": enabled, "unlocked": unlocked}

    def compose(self) -> ComposeResult:
        yield EncryptionSettingsCard(**self._card_args)


def _text(app: App, selector: str) -> str:
    return str(app.query_one(selector, Static).render())


async def _settle(pilot, app: App, predicate, timeout: float = 30.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await pilot.pause(0.05)


def _disabled(app: App, button_id: str) -> bool:
    return app.query_one(f"#{button_id}", Button).disabled


async def _submit_dialog(pilot, app: App, **fields: str) -> None:
    await _settle(pilot, app, lambda: isinstance(app.screen, PasswordDialog))
    dialog = app.screen
    for field_id, value in fields.items():
        dialog.query_one(f"#{field_id}", Input).value = value
    dialog.query_one("#submit-button", Button).press()
    await _settle(pilot, app, lambda: not isinstance(app.screen, PasswordDialog))


def _result_settled(app: App) -> bool:
    text = _text(app, "#settings-encryption-result")
    return bool(text) and not text.endswith("…")


@pytest.mark.asyncio
async def test_encrypt_change_and_turn_off_from_the_card(cfg) -> None:
    path = cfg.card_test_path
    app = _Host(enabled=False, unlocked=False)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        assert "Off" in _text(app, "#settings-encryption-state")
        assert not _disabled(app, card_module.ENCRYPT_BUTTON_ID)
        assert _disabled(app, card_module.CHANGE_BUTTON_ID)
        assert _disabled(app, card_module.DISABLE_BUTTON_ID)

        # Encrypt keys: password-gated by the setup dialog, applied in a worker.
        app.query_one(f"#{card_module.ENCRYPT_BUTTON_ID}", Button).press()
        await _submit_dialog(
            pilot, app, **{"password-input": PASSWORD_A, "confirm-input": PASSWORD_A}
        )
        await _settle(pilot, app, lambda: _result_settled(app))
        assert _text(app, "#settings-encryption-result") == card_module.RESULT_ENABLED
        assert "On —" in _text(app, "#settings-encryption-state")
        stored = tomllib.loads(path.read_text())
        assert stored["encryption"]["enabled"] is True
        assert stored["api_settings"]["openai"]["api_key"].startswith("enc:")
        assert _disabled(app, card_module.ENCRYPT_BUTTON_ID)
        assert not _disabled(app, card_module.CHANGE_BUTTON_ID)

        # Change password with the WRONG current password: refused inline.
        before = path.read_bytes()
        app.query_one(f"#{card_module.CHANGE_BUTTON_ID}", Button).press()
        await _submit_dialog(
            pilot,
            app,
            **{
                "current-password-input": "not-the-password",
                "password-input": PASSWORD_B,
                "confirm-input": PASSWORD_B,
            },
        )
        await _settle(pilot, app, lambda: _result_settled(app))
        assert (
            _text(app, "#settings-encryption-result")
            == card_module.RESULT_WRONG_PASSWORD
        )
        assert path.read_bytes() == before

        # Change password with the right current password.
        app.query_one(f"#{card_module.CHANGE_BUTTON_ID}", Button).press()
        await _submit_dialog(
            pilot,
            app,
            **{
                "current-password-input": PASSWORD_A,
                "password-input": PASSWORD_B,
                "confirm-input": PASSWORD_B,
            },
        )
        await _settle(
            pilot,
            app,
            lambda: _text(app, "#settings-encryption-result")
            == card_module.RESULT_CHANGED,
        )
        assert cfg.verify_config_encryption_password(PASSWORD_B) is True
        assert cfg.verify_config_encryption_password(PASSWORD_A) is False

        # Turn off: asks for the current master password.
        app.query_one(f"#{card_module.DISABLE_BUTTON_ID}", Button).press()
        await _submit_dialog(pilot, app, **{"password-input": PASSWORD_B})
        await _settle(
            pilot,
            app,
            lambda: _text(app, "#settings-encryption-result")
            == card_module.RESULT_DISABLED,
        )
        assert "Off" in _text(app, "#settings-encryption-state")
        restored = tomllib.loads(path.read_text())
        assert "encryption" not in restored
        assert restored["api_settings"]["openai"]["api_key"] == PLAINTEXT_KEY
        assert not _disabled(app, card_module.ENCRYPT_BUTTON_ID)
        assert _disabled(app, card_module.DISABLE_BUTTON_ID)


@pytest.mark.asyncio
async def test_card_corrects_a_stale_compose_state_from_the_file(cfg) -> None:
    assert cfg.enable_config_encryption(PASSWORD_A) is True
    cfg.clear_encryption_password()
    # Composed as "off" from a stale in-memory view; the file says on+locked.
    app = _Host(enabled=False, unlocked=False)
    async with app.run_test(size=(120, 40)) as pilot:
        await _settle(
            pilot,
            app,
            lambda: "locked" in _text(app, "#settings-encryption-state"),
        )
        assert _disabled(app, card_module.ENCRYPT_BUTTON_ID)
        assert not _disabled(app, card_module.DISABLE_BUTTON_ID)


@pytest.mark.asyncio
async def test_cancelling_the_dialog_changes_nothing(cfg) -> None:
    before = cfg.card_test_path.read_bytes()
    app = _Host(enabled=False, unlocked=False)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        app.query_one(f"#{card_module.ENCRYPT_BUTTON_ID}", Button).press()
        await _settle(pilot, app, lambda: isinstance(app.screen, PasswordDialog))
        await pilot.press("escape")
        await _settle(pilot, app, lambda: not isinstance(app.screen, PasswordDialog))
        await pilot.pause(0.2)
        assert _text(app, "#settings-encryption-result") == ""
    assert cfg.card_test_path.read_bytes() == before


def test_enable_dialog_warns_about_comment_loss() -> None:
    assert card_module.COMMENT_LOSS_WARNING in card_module.ENABLE_DIALOG_MESSAGE
