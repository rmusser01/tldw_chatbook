"""Settings > Privacy & Security > Encryption card (TASK-34100.4).

The only reachable home for the key-encryption lifecycle after setup: a state
line plus three password-gated actions -- Encrypt keys, Change password and
Turn off encryption. The setup wizard's Protect step promised "you can enable
this later in Settings > Privacy & Security", but that category only showed a
read-only row, and the old controls lived in the nav-unreachable (and
deprecated) Tools & Settings window (protect-summary-03). The flows here are
ported from that window, with two corrections: Change password asks for the
current and the new password in one dialog, and every action runs off the UI
thread (each one derives scrypt keys) and reports its outcome inline.

Config functions are looked up on ``tldw_chatbook.config`` at call time, not
imported at module load, so the card always talks to the installed config
source.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.password_dialog import (
    COMMENT_LOSS_WARNING,
    PasswordChange,
    PasswordDialog,
)

ENCRYPTION_CARD_ID = "settings-encryption-card"
ENCRYPT_BUTTON_ID = "settings-encryption-enable"
CHANGE_BUTTON_ID = "settings-encryption-change"
DISABLE_BUTTON_ID = "settings-encryption-disable"

STATE_ON = "On — you'll enter your master password when chatbook starts."
STATE_OFF = "Off — API keys are stored as plain text in config.toml."
STATE_LOCKED = (
    "On, but locked — this session started without the master password, so "
    "saved API keys read as missing. Relaunch chatbook to unlock them."
)

ENABLE_DIALOG_MESSAGE = (
    "Create a master password to encrypt the API keys saved in config.toml. "
    "At least 8 characters. You'll enter it each time chatbook starts. If you "
    "forget it, you can reset the saved keys at startup and re-enter them; "
    "chats, notes and documents are not affected.\n" + COMMENT_LOSS_WARNING
)
DISABLE_DIALOG_MESSAGE = (
    "Enter your master password to turn encryption off. Your API keys will be "
    "saved as plain text in config.toml."
)

RESULT_ENABLED = (
    "Done: saved API keys are encrypted. You'll enter your master password "
    "when chatbook starts."
)
RESULT_ALREADY_ON = "Encryption is already on. Use Change password instead."
RESULT_ENABLE_FAILED = "Encrypting failed; config.toml was left as it was."
#: Encryption is off but a value is still `enc:` ciphertext under an earlier
#: password; encrypting over it would strand it again, so enable refuses.
#: ``{where}`` is the screen that can edit every named value (review round 2,
#: R2-F2): Providers & Models edits only `api_settings.<provider>.api_key`;
#: Advanced Config edits any value.
RESULT_ENABLE_STRANDED = (
    "Encrypting failed: {path} is still encrypted with an earlier password "
    "and can't be read. Re-enter or clear it in {where}, then try again. "
    "config.toml was left as it was."
)
WHERE_PROVIDERS = "Settings > Providers & Models"
WHERE_ADVANCED_CONFIG = "Settings > Advanced Config"
#: The change failed AND putting the old file back failed: the file may hold
#: the new document. The state line shows what the file holds now.
RESULT_FILE_CHANGED = (
    "Something went wrong and undoing it failed, so config.toml may have "
    "changed. The state above is what the file holds now; see the log."
)
RESULT_WRONG_PASSWORD = "That password didn't match. Nothing was changed."
RESULT_CHANGED = "Done: master password changed. Use the new one at next start."
RESULT_CHANGE_FAILED = "Changing the password failed; nothing was changed."
RESULT_DISABLED = (
    "Done: encryption is off. API keys are stored as plain text in config.toml."
)
RESULT_DISABLE_FAILED = "Turning encryption off failed; nothing was changed."
RESULT_ERROR = "Something went wrong; nothing was changed. See the log."

_BUSY_TEXT = {
    "enable": "Encrypting saved keys…",
    "change": "Changing the master password…",
    "disable": "Turning encryption off…",
}


def _config():
    """The installed config module (resolved per call, never cached)."""
    import tldw_chatbook.config as config_module

    return config_module


@dataclass(frozen=True)
class EncryptionActionOutcome:
    """What one card action reports back to the UI thread."""

    message: str
    succeeded: bool
    enabled: bool
    unlocked: bool


def encryption_state_text(*, enabled: bool, unlocked: bool) -> str:
    """The card's state line for a given on/off and locked/unlocked state."""
    if not enabled:
        return STATE_OFF
    return STATE_ON if unlocked else STATE_LOCKED


def _current_state(assumed_enabled: bool) -> tuple[bool, bool]:
    """(enabled, unlocked) from the file and the session.

    ``assumed_enabled`` stands in when the file cannot be read.
    """
    config = _config()
    enabled = config.config_encryption_enabled_on_disk()
    return (
        assumed_enabled if enabled is None else enabled,
        config.get_encryption_password() is not None,
    )


def _provider_key_path(path: str) -> bool:
    """Whether Providers & Models edits this dotted config path."""
    parts = path.split(".")
    return len(parts) == 3 and parts[0] == "api_settings" and parts[2] == "api_key"


def _enable_refused_message(config) -> str:
    """Why an enable on an encryption-off file was refused, in plain words."""
    stranded = config.encrypted_value_paths_on_disk()
    if not stranded:
        return RESULT_ENABLE_FAILED
    path = stranded[0]
    if len(stranded) > 1:
        path += f" (and {len(stranded) - 1} more)"
    where = (
        WHERE_PROVIDERS
        if all(_provider_key_path(item) for item in stranded)
        else WHERE_ADVANCED_CONFIG
    )
    return RESULT_ENABLE_STRANDED.format(path=path, where=where)


def run_enable(password: str) -> EncryptionActionOutcome:
    """Encrypt the saved keys (refused when encryption is already on).

    Runs scrypt; call it off the UI thread.

    Args:
        password: The new master password the user typed (already confirmed
            and length-checked by the setup dialog).

    Returns:
        The outcome to show: ``succeeded`` is True when the keys are now
        encrypted (or already were with this password); ``message`` explains
        a refusal -- encryption already on, a value still encrypted under an
        earlier password, or a failed rollback; ``enabled`` and ``unlocked``
        are re-read from the file and the session afterwards.
    """
    config = _config()
    was_enabled = config.config_encryption_enabled_on_disk()
    succeeded = bool(config.enable_config_encryption(password))
    enabled, unlocked = _current_state(assumed_enabled=succeeded)
    if succeeded:
        message = RESULT_ENABLED
    elif not enabled:
        message = _enable_refused_message(config)
    elif was_enabled is False:
        # Off before, on now, and still refused: the rollback failed.
        message = RESULT_FILE_CHANGED
    else:
        message = RESULT_ALREADY_ON
    return EncryptionActionOutcome(message, succeeded, enabled, unlocked)


def run_change(change: PasswordChange) -> EncryptionActionOutcome:
    """Re-encrypt the saved keys under a new master password.

    Runs scrypt; call it off the UI thread.

    Args:
        change: The current and new master passwords from the change
            dialog. The current one is checked against the saved verifier
            before anything is written.

    Returns:
        The outcome to show: ``succeeded`` is True when the file now answers
        to ``change.new``; ``message`` says whether the current password was
        wrong, the change was refused with nothing written, or a failed
        rollback may have left the new document; ``enabled`` and
        ``unlocked`` are re-read afterwards.
    """
    config = _config()
    if not config.verify_config_encryption_password(change.current):
        enabled, unlocked = _current_state(assumed_enabled=True)
        return EncryptionActionOutcome(RESULT_WRONG_PASSWORD, False, enabled, unlocked)
    succeeded = bool(config.change_encryption_password(change.current, change.new))
    enabled, unlocked = _current_state(assumed_enabled=True)
    if succeeded:
        message = RESULT_CHANGED
    elif change.new != change.current and config.verify_config_encryption_password(
        change.new
    ):
        # Refused, yet the file answers to the NEW password: the rollback failed.
        message = RESULT_FILE_CHANGED
    else:
        message = RESULT_CHANGE_FAILED
    return EncryptionActionOutcome(message, succeeded, enabled, unlocked)


def run_disable(password: str) -> EncryptionActionOutcome:
    """Decrypt the saved keys and turn encryption off.

    Runs scrypt; call it off the UI thread.

    Args:
        password: The master password, checked against the saved verifier
            before anything is written.

    Returns:
        The outcome to show: ``succeeded`` is True when the file is now
        plain text with ``[encryption]`` removed; ``message`` says whether
        the password was wrong, the change was refused with nothing
        written, or a failed rollback left the file decrypted; ``enabled``
        and ``unlocked`` are re-read afterwards.
    """
    config = _config()
    if not config.verify_config_encryption_password(password):
        enabled, unlocked = _current_state(assumed_enabled=True)
        return EncryptionActionOutcome(RESULT_WRONG_PASSWORD, False, enabled, unlocked)
    succeeded = bool(config.disable_config_encryption(password))
    enabled, unlocked = _current_state(assumed_enabled=not succeeded)
    if succeeded:
        message = RESULT_DISABLED
    elif not enabled:
        # Refused, yet the file is now decrypted: the rollback failed.
        message = RESULT_FILE_CHANGED
    else:
        message = RESULT_DISABLE_FAILED
    return EncryptionActionOutcome(message, succeeded, enabled, unlocked)


class EncryptionSettingsCard(Vertical):
    """State line, three password-gated actions and an inline result."""

    def __init__(self, *, enabled: bool, unlocked: bool, **kwargs: Any) -> None:
        """Build the card from the state known at compose time.

        Args:
            enabled: Whether encryption is on (in-memory view; the card
                re-reads the file once mounted).
            unlocked: Whether this session holds the master password.
            **kwargs: Passed to ``Vertical`` (id, classes).
        """
        kwargs.setdefault("id", ENCRYPTION_CARD_ID)
        kwargs.setdefault("classes", "settings-focus-card")
        super().__init__(**kwargs)
        self._enabled = enabled
        self._unlocked = unlocked
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Static("Encryption", classes="destination-section")
        yield Static(
            "Config encryption: "
            + encryption_state_text(enabled=self._enabled, unlocked=self._unlocked),
            id="settings-encryption-state",
            classes="settings-status-row",
            markup=False,
        )
        with Horizontal(
            id="settings-encryption-actions", classes="settings-action-row"
        ):
            yield Button(
                "Encrypt keys…",
                id=ENCRYPT_BUTTON_ID,
                tooltip="Set a master password and encrypt the API keys in config.toml.",
            )
            yield Button(
                "Change password…",
                id=CHANGE_BUTTON_ID,
                tooltip="Re-encrypt the saved keys under a new master password.",
            )
            yield Button(
                "Turn off encryption…",
                id=DISABLE_BUTTON_ID,
                tooltip="Decrypt the saved keys and store them as plain text.",
            )
        yield Static(
            "",
            id="settings-encryption-result",
            classes="settings-status-row",
            markup=False,
        )

    def on_mount(self) -> None:
        self._sync_controls()
        # The compose-time state comes from the in-memory config; confirm it
        # from the file off the UI thread.
        self.run_worker(
            self._read_state_worker,
            thread=True,
            group="settings-encryption-state",
            exit_on_error=False,
        )

    def _read_state_worker(self) -> None:
        enabled, unlocked = _current_state(assumed_enabled=self._enabled)
        self.app.call_from_thread(self._apply_state, enabled, unlocked)

    def _apply_state(self, enabled: bool, unlocked: bool) -> None:
        if not self.is_attached or self._busy:
            return
        self._enabled = enabled
        self._unlocked = unlocked
        self._sync_controls()

    def _sync_controls(self) -> None:
        self.query_one("#settings-encryption-state", Static).update(
            "Config encryption: "
            + encryption_state_text(enabled=self._enabled, unlocked=self._unlocked)
        )
        self.query_one(f"#{ENCRYPT_BUTTON_ID}", Button).disabled = (
            self._busy or self._enabled
        )
        for button_id in (CHANGE_BUTTON_ID, DISABLE_BUTTON_ID):
            self.query_one(f"#{button_id}", Button).disabled = (
                self._busy or not self._enabled
            )

    def _set_result(self, message: str) -> None:
        self.query_one("#settings-encryption-result", Static).update(message)

    @on(Button.Pressed, f"#{ENCRYPT_BUTTON_ID}")
    def _encrypt_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.app.push_screen(
            PasswordDialog(
                mode="setup",
                title="Encrypt saved API keys",
                message=ENABLE_DIALOG_MESSAGE,
            ),
            self._on_enable_password,
        )

    @on(Button.Pressed, f"#{CHANGE_BUTTON_ID}")
    def _change_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.app.push_screen(PasswordDialog(mode="change"), self._on_change_passwords)

    @on(Button.Pressed, f"#{DISABLE_BUTTON_ID}")
    def _disable_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.app.push_screen(
            PasswordDialog(
                mode="unlock",
                title="Turn off encryption",
                message=DISABLE_DIALOG_MESSAGE,
            ),
            self._on_disable_password,
        )

    def _on_enable_password(self, password: object) -> None:
        if isinstance(password, str) and password:
            self._start("enable", lambda: run_enable(password))

    def _on_change_passwords(self, change: object) -> None:
        if isinstance(change, PasswordChange):
            self._start("change", lambda: run_change(change))

    def _on_disable_password(self, password: object) -> None:
        if isinstance(password, str) and password:
            self._start("disable", lambda: run_disable(password))

    def _start(self, action: str, job: Callable[[], EncryptionActionOutcome]) -> None:
        if self._busy or not self.is_attached:
            return
        self._busy = True
        self._sync_controls()
        self._set_result(_BUSY_TEXT[action])
        # Not exclusive: an exclusive worker would CANCEL a running change
        # (lessons-textual); the busy flag already refuses a second start.
        self.run_worker(
            lambda: self._run_job(action, job),
            thread=True,
            group="settings-encryption-action",
            exit_on_error=False,
        )

    def _run_job(self, action: str, job: Callable[[], EncryptionActionOutcome]) -> None:
        try:
            outcome = job()
        except Exception as error:
            logger.error(
                "Settings encryption action failed (action={}, error_type={}).",
                action,
                type(error).__name__,
            )
            outcome = EncryptionActionOutcome(
                RESULT_ERROR, False, self._enabled, self._unlocked
            )
        self.app.call_from_thread(self._finish, outcome)

    def _finish(self, outcome: EncryptionActionOutcome) -> None:
        self._busy = False
        if not self.is_attached:
            return
        self._enabled = outcome.enabled
        self._unlocked = outcome.unlocked
        self._sync_controls()
        self._set_result(outcome.message)
