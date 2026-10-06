"""The first-run wizard's Protect keys step.

Moved whole out of ``FirstRunSetupWizard.py`` (TASK-34100.1), so its ``@on``
handlers and workers stay registered on the class and fixes to this step have
room under the size ratchet. ``FirstRunSetupWizard`` still exports the class.
Patch this module, not the wizard, to replace what the step calls.

TASK-34100.4 (protect-summary-04, -06, -07) made the step honest. What it
shows comes from the config FILE, read off the UI thread every time the step
is shown (``first_run_protect_state``): encrypted, a plaintext key saved
(named per provider, with the file path), a typed key that was never saved,
keys only from environment variables, or nothing at all. Encrypting goes
through the Settings Encryption card's own ``run_enable``, so a refused enable
reports why instead of "your keys are unchanged (plain text)" beside "✓
Encryption enabled.". After success the button becomes "Change password…" and
focus moves to Next, so Enter continues instead of reopening the dialog.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import (
    Any,
    Dict,
    Optional,
)

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import (
    Button,
    Static,
)

from tldw_chatbook.UI.Wizards import first_run_protect_state as protect_state
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import SetupStep
from tldw_chatbook.UI.Wizards.first_run_step_guard import run_wizard_worker

PRIMARY_BUTTON_ID = "setup-protect-set-password"
KEEP_PLAIN_BUTTON_ID = "setup-protect-keep-plain"
BACK_TO_MODEL_BUTTON_ID = "setup-protect-back-to-model"


def _load_persisted_config() -> tuple[Mapping[str, Any], str]:
    """Read config.toml back the way Summary does, plus its path.

    Returns:
        The merged config (decrypted with the session password, if any) and
        the config file path as text ("" when it cannot be resolved).
    """
    from tldw_chatbook.config import (
        get_cli_config_path,
        load_cli_config_and_ensure_existence,
    )

    config = load_cli_config_and_ensure_existence(force_reload=True)
    try:
        path_text = str(get_cli_config_path())
    except Exception as error:
        # error_type only: the exception text may carry a path.
        logger.debug(
            "Protect could not resolve the config path (error_type={}).",
            type(error).__name__,
        )
        path_text = ""
    return config, path_text


def _run_enable(password: str):
    """Encrypt the saved keys through the Settings card's own action."""
    from tldw_chatbook.UI.Screens.settings_encryption import run_enable

    return run_enable(password)


def _run_change(change):
    """Change the master password through the Settings card's own action."""
    from tldw_chatbook.UI.Screens.settings_encryption import run_change

    return run_change(change)


class ProtectKeysStep(SetupStep):
    """Offer config encryption for the keys saved in config.toml.

    Encryption goes only through the existing mechanism: PasswordDialog
    collects the password, and the Settings Encryption card's ``run_enable``
    / ``run_change`` (``enable_config_encryption`` /
    ``change_encryption_password``) do the rewrite under the config lock.
    This step never rolls its own crypto.
    """

    def __init__(
        self,
        wizard=None,
        config=None,
        *,
        enable_encryption: Optional[Callable[[str], bool]] = None,
        load_config: Optional[Callable[[], tuple[Mapping[str, Any], str]]] = None,
        environ: Optional[Mapping[str, str]] = None,
        **kwargs,
    ):
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._enable_encryption = enable_encryption
        self._load_config = load_config
        self._environ = environ
        self.encryption_enabled = False
        self._state: protect_state.ProtectState | None = None
        self._config_path = ""
        self._state_generation = 0
        self._busy = False
        self._last_outcome = None

    def compose_step(self) -> ComposeResult:
        with Vertical(classes="setup-protect"):
            yield Static(
                "Protect your keys", id="setup-protect-title", classes="setup-title"
            )
            yield Static(
                "Checking what config.toml holds…",
                id="setup-protect-body",
                classes="setup-subtitle",
                markup=False,
            )
            yield Static(
                "", id="setup-protect-path", classes="setup-subtitle", markup=False
            )
            # Reuses the Provider step's button-row rule (three-row height,
            # spaced buttons) rather than adding boot-parsed CSS bytes.
            with Horizontal(classes="setup-provider-connection-actions"):
                yield Button(
                    protect_state.ENCRYPT_LABEL, id=PRIMARY_BUTTON_ID, variant="primary"
                )
                yield Button(protect_state.KEEP_PLAIN_LABEL, id=KEEP_PLAIN_BUTTON_ID)
                yield Button(
                    protect_state.BACK_TO_MODEL_LABEL, id=BACK_TO_MODEL_BUTTON_ID
                )
            yield Static("", id="setup-protect-status", classes="setup-probe-status")
            yield Static(
                "", id="setup-protect-later", classes="setup-subtitle", markup=False
            )

    def on_mount(self) -> None:
        # Nothing is offered until the file has been read.
        for button in self.query(Button):
            button.display = False

    def on_show(self) -> None:
        """Re-derive the state from the config file each time the step shows.

        TASK-34100.4: the old gate (``key_entered`` or a plaintext key in the
        in-memory config, which has no ``[encryption]`` table) offered a
        second password over encrypted keys and a password for keys that
        were never saved.
        """
        super().on_show()
        self.reload_state()

    def reload_state(self) -> None:
        """Read the persisted config off the UI thread and re-render."""
        self._state_generation += 1
        run_wizard_worker(
            self,
            self._load_state_worker(self._state_generation),
            exclusive=True,
            group="setup-protect-state",
        )

    async def _load_state_worker(self, generation: int) -> None:
        import asyncio

        load = self._load_config or _load_persisted_config
        try:
            config, path_text = await asyncio.get_running_loop().run_in_executor(
                None, load
            )
        except Exception as error:
            # The in-memory view has no [encryption] table, so it can only
            # tell plaintext from nothing; say that rather than guess.
            logger.warning(
                "Protect could not read config.toml (error_type={}).",
                type(error).__name__,
            )
            config = getattr(
                getattr(self.wizard, "app_instance", None), "app_config", {}
            )
            path_text = ""
        if not self.is_attached or generation != self._state_generation:
            return
        environ = self._environ if self._environ is not None else os.environ
        state = protect_state.read_protect_state(
            config if isinstance(config, Mapping) else {},
            environ,
            typed_provider_key=protect_state.typed_provider_key(
                getattr(self.wizard, "wizard_data", None)
            ),
        )
        self._config_path = path_text
        self.render_state(state)
        self._settle_focus()

    def render_state(self, state: protect_state.ProtectState) -> None:
        """Show one Protect state's copy and buttons.

        Args:
            state: The state derived from the persisted config.
        """
        self._state = state
        self.encryption_enabled = state.kind == protect_state.PROTECT_ENCRYPTED
        copy = protect_state.protect_copy(state, config_path=self._config_path)
        try:
            self.query_one("#setup-protect-title", Static).update(copy.title)
            self.query_one("#setup-protect-body", Static).update(copy.body)
            path = self.query_one("#setup-protect-path", Static)
            path.update(copy.path_line)
            path.display = bool(copy.path_line)
            self.query_one("#setup-protect-later", Static).update(copy.later_line)
            primary = self.query_one(f"#{PRIMARY_BUTTON_ID}", Button)
            primary.label = copy.primary_label or protect_state.ENCRYPT_LABEL
            primary.display = bool(copy.primary_label)
            primary.disabled = self._busy
            self.query_one(
                f"#{KEEP_PLAIN_BUTTON_ID}", Button
            ).display = copy.show_keep_plain
            self.query_one(
                f"#{BACK_TO_MODEL_BUTTON_ID}", Button
            ).display = copy.show_back_to_model
        except NoMatches:
            return

    # -- focus ---------------------------------------------------------------
    def _next_button(self) -> Button | None:
        try:
            return self.screen.query_one("#wizard-next", Button)
        except Exception:
            return None

    def preferred_focus(self) -> Optional[Widget]:
        """The state's own control, else Next so Enter continues.

        Returns:
            The encrypt button for a plaintext key, "Back to Model" for an
            unsaved key, otherwise the wizard's Next button (encrypted,
            env-only and nothing-to-protect states have no step action that
            Enter should repeat). None while the file is still being read.
        """
        state = self._state
        if state is None:
            return None
        try:
            if state.kind == protect_state.PROTECT_PLAINTEXT:
                return self.query_one(f"#{PRIMARY_BUTTON_ID}", Button)
            if state.kind == protect_state.PROTECT_UNSAVED:
                return self.query_one(f"#{BACK_TO_MODEL_BUTTON_ID}", Button)
        except NoMatches:
            return None
        return self._next_button()

    def _settle_focus(self) -> None:
        """Move focus to the state's control once the file has been read.

        The step-change focus fix runs before this read finishes, so it can
        only land on Next or a control from the previous render. Focus moves
        only from there -- never out of a dialog or off Back / Exit setup.
        """
        if (
            not (self.is_attached and self.display)
            or self.app.screen is not self.screen
        ):
            return
        target = self.preferred_focus()
        if target is None or not target.display or target.disabled:
            return
        focused = self.app.focused
        movable = (
            focused is None
            or not focused.is_attached
            or not focused.display
            or focused is self._next_button()
            or self in focused.ancestors_with_self
        )
        if movable and focused is not target:
            target.focus()

    def _focus_next(self) -> None:
        """After an action succeeds, Next takes focus so Enter continues."""
        if (
            not (self.is_attached and self.display)
            or self.app.screen is not self.screen
        ):
            return
        next_button = self._next_button()
        if next_button is not None and next_button.display and not next_button.disabled:
            next_button.focus()

    # -- actions -------------------------------------------------------------
    @on(Button.Pressed, f"#{PRIMARY_BUTTON_ID}")
    def _on_set_password(self) -> None:
        from tldw_chatbook.Widgets.password_dialog import PasswordDialog

        if self._busy:
            return
        if self._state is not None and self._state.kind == (
            protect_state.PROTECT_ENCRYPTED
        ):
            # Already encrypted: never a second enable (which used to
            # rewrite the verifier over keys still under the first password).
            self.app.push_screen(PasswordDialog(mode="change"), self._on_change_result)
            return
        from tldw_chatbook.UI.Screens.settings_encryption import ENABLE_DIALOG_MESSAGE

        dialog = PasswordDialog(
            mode="setup",
            title="Encrypt saved API keys",
            message=ENABLE_DIALOG_MESSAGE,
        )
        self.app.push_screen(dialog, self._on_password_result)

    @on(Button.Pressed, f"#{KEEP_PLAIN_BUTTON_ID}")
    def _on_keep_plain(self) -> None:
        """'Keep as plain text' continues exactly like Next."""
        advance = getattr(self.wizard, "advance_programmatically", None)
        if callable(advance):
            advance()

    @on(Button.Pressed, f"#{BACK_TO_MODEL_BUTTON_ID}")
    def _on_back_to_model(self) -> None:
        """Return to Model, where choosing a model saves the typed key."""
        index_for = getattr(self.wizard, "_step_index_for_id", None)
        show = getattr(self.wizard, "show_step", None)
        index = index_for(wizard_state.STEP_MODEL) if callable(index_for) else None
        if index is not None and callable(show):
            show(index)

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

    def _on_change_result(self, change: object) -> None:
        from tldw_chatbook.Widgets.password_dialog import PasswordChange

        if not isinstance(change, PasswordChange):
            return
        run_wizard_worker(
            self,
            self._apply_change_worker(change),
            exclusive=True,
            group="setup-protect-encrypt",
        )

    def _set_status(self, text: str) -> None:
        """Update the outcome line, if the step is still built."""
        status = self.query("#setup-protect-status")
        if status:
            status.first(Static).update(text)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        # A newer render supersedes any file read still in flight.
        self._state_generation += 1
        try:
            self.query_one(f"#{PRIMARY_BUTTON_ID}", Button).disabled = busy
        except NoMatches:
            pass

    async def _apply_password_worker(self, password: str) -> None:
        if self.is_attached:
            self._set_busy(True)
        try:
            ok = await self.apply_password(password)
        finally:
            if self.is_attached:
                self._set_busy(False)
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
        if ok:
            self.render_state(
                protect_state.ProtectState(protect_state.PROTECT_ENCRYPTED)
            )
            self._set_status(protect_state.ENABLED_STATUS)
            self._focus_next()
            return
        # TASK-34100.4 AC #1: no stale ✓ and no claim about the file that the
        # outcome did not establish; then show what the file really holds.
        self._set_status("")
        outcome = self._last_outcome
        self.show_step_error(getattr(outcome, "message", "") or "Encrypting failed.")
        self.reload_state()

    async def _apply_change_worker(self, change) -> None:
        import asyncio

        self._set_busy(True)
        try:
            outcome = await asyncio.get_running_loop().run_in_executor(
                None, _run_change, change
            )
        except Exception as error:
            logger.error(
                "Setup password change failed (error_type={}).", type(error).__name__
            )
            outcome = None
        finally:
            if self.is_attached:
                self._set_busy(False)
        if not self.is_attached:
            return
        if outcome is not None and outcome.succeeded:
            self._set_status(protect_state.CHANGED_STATUS)
            self._focus_next()
            return
        self._set_status("")
        self.show_step_error(
            getattr(outcome, "message", "")
            or "Changing the password failed; nothing was changed."
        )
        self.reload_state()

    async def apply_password(self, password: str) -> bool:
        """Encrypt the saved keys with ``password`` off the UI thread.

        Args:
            password: The new master password from the setup dialog.

        Returns:
            True when the keys are now encrypted under ``password``.
        """
        import asyncio

        from tldw_chatbook.UI.Screens.settings_encryption import (
            RESULT_ENABLE_FAILED,
            RESULT_ENABLED,
            EncryptionActionOutcome,
        )

        enable = self._enable_encryption
        loop = asyncio.get_running_loop()
        if enable is None:
            outcome = await loop.run_in_executor(None, _run_enable, password)
        else:
            ok = bool(await loop.run_in_executor(None, enable, password))
            outcome = EncryptionActionOutcome(
                RESULT_ENABLED if ok else RESULT_ENABLE_FAILED, ok, ok, ok
            )
        self._last_outcome = outcome
        self.encryption_enabled = bool(outcome.enabled)
        return bool(outcome.succeeded)

    def get_step_data(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"encryption_enabled": self.encryption_enabled}
        state = getattr(self, "_state", None)
        if state is not None:
            # Read by the tracker ('–' when there was nothing to protect);
            # the setup checkpoint keeps only encryption_enabled.
            data["protect_state"] = state.kind
        return data
