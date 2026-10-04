"""The first-run wizard's Protect keys step.

Moved whole out of ``FirstRunSetupWizard.py`` (TASK-34100.1), so its ``@on``
handlers and workers stay registered on the class and fixes to this step have
room under the size ratchet. ``FirstRunSetupWizard`` still exports the class.
Patch this module, not the wizard, to replace what the step calls.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
)

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import (
    Button,
    Static,
)

from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import SetupStep
from tldw_chatbook.UI.Wizards.first_run_step_guard import run_wizard_worker


class ProtectKeysStep(SetupStep):
    """Offer config encryption for any keys entered this run.

    Encryption goes only through the existing mechanism: PasswordDialog
    (setup mode) collects the password, enable_config_encryption(password)
    does the actual rewrite under the config RLock. This step never rolls
    its own crypto.
    """

    def __init__(self, wizard=None, config=None, *, enable_encryption=None, **kwargs):
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._enable_encryption = enable_encryption
        self.encryption_enabled = False

    def compose_step(self) -> ComposeResult:
        with Vertical(classes="setup-protect"):
            yield Static("Protect your keys", classes="setup-title")
            yield Static(
                "Encrypt the API keys in your config file with a password. "
                "You'll be asked for this password each time chatbook starts. "
                "Skip to leave keys as plain text (you can enable this later "
                "in Settings ▸ Privacy & Security).",
                classes="setup-subtitle",
            )
            yield Button(
                "Set a password", id="setup-protect-set-password", variant="primary"
            )
            yield Static("", id="setup-protect-status", classes="setup-probe-status")

    def on_show(self) -> None:
        """Render the nothing-to-do state while no key exists (UAT N-6).

        TASK-21148: Protect is always on the track now (a stable step
        total beats a shorter one), so a keyless run reaches this step —
        say why there is nothing to do instead of offering a password for
        keys that don't exist.
        """
        super().on_show()
        key_entered = bool(getattr(self.wizard, "key_entered", False))
        stored = False
        try:
            stored = wizard_state.stored_plaintext_key_present(
                getattr(self.wizard.app_instance, "app_config", {}) or {}
            )
        except Exception:
            logger.debug("Protect stored-key probe skipped", exc_info=True)
        has_keys = key_entered or stored
        try:
            button = self.query_one("#setup-protect-set-password", Button)
            status = self.query_one("#setup-protect-status", Static)
        except NoMatches:
            return
        button.display = has_keys
        if not has_keys and not self.encryption_enabled:
            status.update(
                "No API keys saved yet — nothing to protect. This step "
                "matters once a key is stored; Next continues."
            )
        elif has_keys and not self.encryption_enabled:
            status.update("")

    @on(Button.Pressed, "#setup-protect-set-password")
    def _on_set_password(self) -> None:
        from tldw_chatbook.Widgets.password_dialog import PasswordDialog

        # Mirrors the only other setup-mode caller,
        # Tools_Settings_Window.py's _setup_encryption (~line 7309):
        #   PasswordDialog(mode="setup", on_submit=lambda p: None,
        #                  on_cancel=lambda: None)
        # That caller does not override title/message -- it relies on
        # PasswordDialog's own mode="setup" defaults ("Setup Master
        # Password" / "Create a master password to encrypt your API keys
        # and sensitive configuration data."). Its on_submit/on_cancel are
        # no-ops (the real work happens after dismiss, same as here), so
        # they add nothing; this uses the push_screen(dialog, callback)
        # idiom already established in this module (see
        # FirstRunSetupWizard.action_cancel's ConfirmationDialog) instead of
        # that caller's await/wait_for_dismiss=True style -- both dispatch
        # through the same ModalScreen.dismiss(password), so the two forms
        # are behaviorally identical here.
        dialog = PasswordDialog(mode="setup")
        self.app.push_screen(dialog, self._on_password_result)

    def _on_password_result(self, password: str | None) -> None:
        if not password:
            return
        # Deviation from the task brief: the brief's pseudocode runs this
        # worker in group "setup-wizard-advance", but that group name is
        # SetupWizardContainer's OWN commit-on-Next / finalize worker
        # (handle_next, _skip_entirely, _finalize all use it, each
        # exclusive=True). Reusing it here would let this step's worker
        # collide with the container's -- exclusive=True workers in the same
        # group cancel/replace each other, so a password-apply in flight
        # could be cancelled by a Next click, or vice versa. A dedicated
        # group avoids that; the actual serialization guarantee against
        # concurrent config writes is enable_config_encryption's own config
        # RLock, not the worker group name.
        run_wizard_worker(
            self,
            self._apply_password_worker(password),
            exclusive=True,
            group="setup-protect-encrypt",
        )

    async def _apply_password_worker(self, password: str) -> None:
        ok = await self.apply_password(password)
        # TASK-32892: the wizard can be dismissed or advanced while the
        # await above is in flight, and a post-await `query_one` raising
        # out of a worker whose `exit_on_error` defaults to True exits the
        # whole app mid-setup. Recheck, and the launch sites pass
        # exit_on_error=False for everything this recheck cannot see.
        #
        # Qodo review of PR #2799: `is_attached`, NOT `is_mounted`.
        # Textual 8.2.8 sets `_is_mounted = True` once and never clears it
        # (message_pump.py:612 is its only assignment after __init__), so a
        # removed widget still reports `is_mounted is True` -- the check this
        # comment describes was inert. `is_attached` walks `_parent` to the
        # DOM root and goes False the moment the node is removed.
        if not self.is_attached:
            return
        status = self.query_one("#setup-protect-status", Static)
        if ok:
            status.update("✓ Encryption enabled.")
        else:
            self.show_step_error(
                "Enabling encryption failed — your keys are unchanged (plain text)."
            )

    async def apply_password(self, password: str) -> bool:
        import asyncio

        enable = self._enable_encryption
        if enable is None:
            from tldw_chatbook.config import enable_config_encryption

            enable = enable_config_encryption
        ok = bool(
            await asyncio.get_running_loop().run_in_executor(None, enable, password)
        )
        self.encryption_enabled = ok
        return ok

    def get_step_data(self) -> Dict[str, Any]:
        return {"encryption_enabled": self.encryption_enabled}
