"""The first-run Protect step tells the truth about config.toml (TASK-34100.4).

The 2026-10-02 review (protect-summary-04, -06, -07) found the step:

* decided "keys or no keys" from a key TYPED this run or a plaintext key in
  the in-memory config, which has no ``[encryption]`` table -- so an encrypted
  config was offered a second password, a keyless or env-key user read the
  encryption pitch beside "No API keys saved yet", and a key typed but never
  saved (Model skipped) was offered a password;
* kept focus on "Set a password" after success, so Enter (the footer's
  "next") reopened the dialog, and a second, different password printed
  "✓ Encryption enabled." beside a false "your keys are unchanged (plain
  text)";
* named a "Skip" control that does not exist and never said where the key
  is stored;
* left the user stuck when Exit setup could not save its checkpoint;
* ticked ✓ on the tracker when there was nothing to protect.

These drive the real wizard. Config reads and writes go through the config
module's own functions, patched here, so the same tests run against the
pre-fix step (RED) and the fixed one.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

import tldw_chatbook.config as config_module
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_MODEL,
    STEP_PROTECT,
    STEP_PROVIDER,
    STEP_SUMMARY,
    TRACK_QUICK,
)
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    FirstRunSetupWizard,
    ProtectKeysStep,
    SetupWizardContainer,
)
from tldw_chatbook.Widgets.password_dialog import PasswordChange, PasswordDialog

pytestmark = pytest.mark.bootstrap_profile

CARD = "Settings ▸ Privacy & Security ▸ Encryption"
CONFIG_PATH = Path("/profiles/test-user/.config/tldw_cli/config.toml")


class _HostApp(App):
    def __init__(self, wizard: FirstRunSetupWizard):
        super().__init__()
        self._wizard = wizard
        self.wizard_result = "UNSET"

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.push_screen(self._wizard, self._capture)

    def _capture(self, result) -> None:
        self.wizard_result = result


class _Disk:
    """The config file as the patched config functions see it."""

    def __init__(self, document: dict):
        self.document = document
        self.enable_calls: list[str] = []
        self.change_calls: list[tuple[str, str]] = []

    @property
    def encrypted(self) -> bool:
        return bool(self.document.get("encryption", {}).get("enabled"))


def _install(monkeypatch, document: dict, *, enable_result: bool = True) -> _Disk:
    disk = _Disk(document)

    def _enable(password: str) -> bool:
        disk.enable_calls.append(password)
        if disk.encrypted:
            return False
        if enable_result:
            disk.document = {**disk.document, "encryption": {"enabled": True}}
        return enable_result

    def _change(old: str, new: str) -> bool:
        disk.change_calls.append((old, new))
        return True

    monkeypatch.setattr(
        config_module,
        "load_cli_config_and_ensure_existence",
        lambda force_reload=False: disk.document,
    )
    monkeypatch.setattr(config_module, "get_cli_config_path", lambda: CONFIG_PATH)
    monkeypatch.setattr(config_module, "enable_config_encryption", _enable)
    monkeypatch.setattr(config_module, "change_encryption_password", _change)
    monkeypatch.setattr(
        config_module, "config_encryption_enabled_on_disk", lambda: disk.encrypted
    )
    monkeypatch.setattr(
        config_module, "verify_config_encryption_password", lambda _pw: True
    )
    monkeypatch.setattr(
        config_module,
        "get_encryption_password",
        lambda: "session-pw" if disk.encrypted else None,
    )
    # Next and Exit setup save a checkpoint; keep it off the real profile.
    monkeypatch.setattr(
        config_module, "save_settings_to_cli_config", lambda *a, **k: True
    )
    return disk


def _wizard(in_memory: dict | None = None) -> FirstRunSetupWizard:
    app_instance = MagicMock()
    app_instance.app_config = in_memory or {}
    return FirstRunSetupWizard(app_instance)


async def _until(pilot, predicate, *, what: str, timeout: float = 4.0) -> None:
    waited = 0.0
    while waited < timeout:
        if predicate():
            return
        await pilot.pause(0.05)
        waited += 0.05
    raise AssertionError(f"timed out waiting for {what}")


async def _at_summary(pilot, app, container) -> None:
    """Wait for Summary and its read-back, so teardown never races it."""
    await _until(
        pilot, lambda: _current_step_id(container) == STEP_SUMMARY, what="Summary"
    )
    await _until(
        pilot,
        lambda: (
            not any(
                worker.is_running and worker.group == "setup-summary-load"
                for worker in app.workers
            )
        ),
        what="the Summary read-back",
    )


def _step_text(step: ProtectKeysStep) -> str:
    return "\n".join(
        str(widget.render()) for widget in step.query(Static) if widget.display
    )


def _shown_buttons(step: ProtectKeysStep) -> dict[str, str]:
    return {
        str(button.label): button.id
        for button in step.query(Button)
        if button.display and button.id and button.id.startswith("setup-protect")
    }


async def _open_protect(pilot, wizard, *, provider_data: dict | None = None):
    container = wizard.query_one(SetupWizardContainer)
    container.select_track(TRACK_QUICK)
    if provider_data is not None:
        container.wizard_data[STEP_PROVIDER] = provider_data
        container.key_entered = True
    index = container._step_index_for_id(STEP_PROTECT)
    container.show_step(index)
    step = container.steps[index]
    assert isinstance(step, ProtectKeysStep)
    await pilot.pause(0.2)
    return container, step


def _current_step_id(container) -> str:
    return container.steps[container.current_step].config.id


def _confirm_label(app, wizard) -> str:
    """The open dialog's confirm-button label, or "" when none is open."""
    if app.screen is wizard:
        return ""
    buttons = app.screen.query("#confirm-button")
    return str(buttons.first(Button).label) if buttons else ""


# --- AC #7: content from persisted config -------------------------------------


@pytest.mark.asyncio
async def test_keyless_profile_reads_nothing_to_protect_without_the_pitch(
    monkeypatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _install(monkeypatch, {"api_settings": {"llama_cpp": {"api_url": "http://x"}}})
    wizard = _wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        container, step = await _open_protect(pilot, wizard)
        await _until(
            pilot, lambda: "Nothing to protect" in _step_text(step), what="state"
        )
        text = _step_text(step)

        assert "Encrypt the API keys" not in text  # the pitch
        assert "Skip" not in text
        assert "No API key is saved in config.toml" in text
        assert CARD in text
        assert _shown_buttons(step) == {}

        # The tracker shows '–', not ✓, for a step that had nothing to do.
        await pilot.press("ctrl+n")
        await _at_summary(pilot, app, container)
        box = wizard.query_one(f"#setup-progress-{STEP_PROTECT} .step-number", Static)
        assert str(box.render()) == "–"


@pytest.mark.asyncio
async def test_env_only_keys_are_named_as_not_stored(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-only-openai-key-sentinel")
    _install(
        monkeypatch,
        {"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}},
    )
    wizard = _wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        _container, step = await _open_protect(pilot, wizard)
        await _until(
            pilot, lambda: "environment variables" in _step_text(step), what="state"
        )
        text = _step_text(step)

        assert "OPENAI_API_KEY" in text
        assert "config.toml holds no key" in text
        assert "sk-env-only" not in text
        assert _shown_buttons(step) == {}


@pytest.mark.asyncio
async def test_typed_but_unsaved_key_is_called_out_with_back_to_model(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _install(monkeypatch, {"api_settings": {"anthropic": {"api_key": ""}}})
    wizard = _wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        container, step = await _open_protect(
            pilot,
            wizard,
            provider_data={
                "provider_key": "anthropic",
                "provider_value": "anthropic",
                "entered_key": True,
            },
        )
        await _until(pilot, lambda: "isn't saved" in _step_text(step), what="state")

        assert "Anthropic key you typed" in _step_text(step)
        assert _shown_buttons(step) == {"Back to Model": "setup-protect-back-to-model"}

        step.query_one("#setup-protect-back-to-model", Button).press()
        await _until(
            pilot,
            lambda: _current_step_id(container) == STEP_MODEL,
            what="the Model step",
        )


# --- AC #8: a stored plaintext key gets the honest offer ----------------------


@pytest.mark.asyncio
async def test_stored_plaintext_keys_name_each_provider_and_the_file(monkeypatch):
    _install(
        monkeypatch,
        {
            "api_settings": {
                "openai": {"api_key": "sk-plain-openai-sentinel"},
                "anthropic": {"api_key": "sk-ant-plain-sentinel"},
                "google": {"api_key": "<API_KEY_HERE>"},
            }
        },
    )
    wizard = _wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        container, step = await _open_protect(pilot, wizard)
        await _until(pilot, lambda: "plain text" in _step_text(step), what="state")
        text = _step_text(step)

        assert "OpenAI and Anthropic keys are saved as plain text" in text
        assert "Google" not in text  # a template placeholder is not a key
        assert str(CONFIG_PATH) in text
        assert "Skip" not in text
        assert CARD in text
        assert "sk-plain" not in text
        assert _shown_buttons(step) == {
            "Encrypt with a password…": "setup-protect-set-password",
            "Keep as plain text": "setup-protect-keep-plain",
        }

        step.query_one("#setup-protect-keep-plain", Button).press()
        await _at_summary(pilot, app, container)


# --- AC #2 + #1: success relabels and hands Enter to Next ------------------------


@pytest.mark.asyncio
async def test_enter_after_encrypting_advances_instead_of_reopening(monkeypatch):
    disk = _install(
        monkeypatch, {"api_settings": {"openai": {"api_key": "sk-plain-openai-x"}}}
    )
    wizard = _wizard({"api_settings": {"openai": {"api_key": "sk-plain-openai-x"}}})
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        container, step = await _open_protect(pilot, wizard)
        primary = step.query_one("#setup-protect-set-password", Button)
        await _until(pilot, lambda: primary.display, what="the encrypt button")
        await _until(pilot, lambda: app.focused is primary, what="focus on it")

        await pilot.press("enter")
        await _until(
            pilot, lambda: isinstance(app.screen, PasswordDialog), what="the dialog"
        )
        app.screen.dismiss("first-master-pw")
        await _until(
            pilot,
            lambda: "Encryption enabled" in _step_text(step),
            what="the success line",
        )

        assert disk.enable_calls == ["first-master-pw"]
        assert str(primary.label) == "Change password…"
        assert "unchanged (plain text)" not in _step_text(step)
        await _until(
            pilot,
            lambda: app.focused is wizard.query_one("#wizard-next", Button),
            what="focus on Next",
        )

        await pilot.press("enter")
        await _at_summary(pilot, app, container)
        assert not isinstance(app.screen, PasswordDialog)
        assert disk.enable_calls == ["first-master-pw"]


@pytest.mark.asyncio
async def test_after_encrypting_the_button_changes_the_password_never_re_enables(
    monkeypatch,
):
    disk = _install(
        monkeypatch, {"api_settings": {"openai": {"api_key": "sk-plain-openai-x"}}}
    )
    wizard = _wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        _container, step = await _open_protect(pilot, wizard)
        primary = step.query_one("#setup-protect-set-password", Button)
        await _until(pilot, lambda: primary.display, what="the encrypt button")
        step._on_password_result("first-master-pw")
        await _until(
            pilot,
            lambda: "Encryption enabled" in _step_text(step),
            what="the success line",
        )

        primary.press()
        await _until(
            pilot, lambda: isinstance(app.screen, PasswordDialog), what="the dialog"
        )
        assert app.screen.mode == "change"
        app.screen.dismiss(PasswordChange("first-master-pw", "second-master-pw"))
        await _until(
            pilot,
            lambda: "Master password changed" in _step_text(step),
            what="the change line",
        )

        assert disk.enable_calls == ["first-master-pw"]
        assert disk.change_calls == [("first-master-pw", "second-master-pw")]


@pytest.mark.asyncio
async def test_a_refused_enable_never_claims_plain_text_beside_a_tick(monkeypatch):
    # The file was encrypted under another password between the read and the
    # click: enable refuses, and the step must say so -- never "✓ Encryption
    # enabled." next to "your keys are unchanged (plain text)".
    disk = _install(
        monkeypatch, {"api_settings": {"openai": {"api_key": "sk-plain-openai-x"}}}
    )
    wizard = _wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        _container, step = await _open_protect(pilot, wizard)
        primary = step.query_one("#setup-protect-set-password", Button)
        await _until(pilot, lambda: primary.display, what="the encrypt button")
        disk.document = {**disk.document, "encryption": {"enabled": True}}

        step._on_password_result("second-master-pw")
        pinned = wizard.query_one("#setup-step-error-pinned", Static)
        await _until(
            pilot, lambda: str(pinned.render()).strip() != "", what="the refusal"
        )
        await _until(
            pilot, lambda: str(primary.label) == "Change password…", what="re-read"
        )
        shown = _step_text(step) + "\n" + str(pinned.render())

        assert "unchanged (plain text)" not in shown
        assert "Encryption enabled" not in shown
        assert "already on" in shown


# --- AC #2: an already-encrypted config -----------------------------------------


@pytest.mark.asyncio
async def test_rerun_on_an_encrypted_config_offers_change_password(monkeypatch):
    # The in-memory config holds the decrypted key and no [encryption] table,
    # exactly like the running app's; only the file says encryption is on.
    disk = _install(
        monkeypatch,
        {
            "encryption": {"enabled": True},
            "api_settings": {"openai": {"api_key": "sk-decrypted-openai-x"}},
        },
    )
    wizard = _wizard({"api_settings": {"openai": {"api_key": "sk-decrypted-openai-x"}}})
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        _container, step = await _open_protect(pilot, wizard)
        primary = step.query_one("#setup-protect-set-password", Button)
        await _until(
            pilot, lambda: str(primary.label) == "Change password…", what="state"
        )
        text = _step_text(step)

        assert "Your saved API keys are encrypted" in text
        assert CARD in text
        assert _shown_buttons(step) == {
            "Change password…": "setup-protect-set-password"
        }
        await _until(
            pilot,
            lambda: app.focused is wizard.query_one("#wizard-next", Button),
            what="focus on Next",
        )

        primary.press()
        await _until(
            pilot, lambda: isinstance(app.screen, PasswordDialog), what="the dialog"
        )
        assert app.screen.mode == "change"
        assert disk.enable_calls == []


# --- AC #2: a failing checkpoint never traps the user ---------------------------


@pytest.mark.asyncio
async def test_exit_setup_offers_exit_without_saving_when_the_save_fails(
    monkeypatch,
):
    _install(monkeypatch, {})
    monkeypatch.setattr(
        config_module, "save_settings_to_cli_config", lambda *a, **k: False
    )
    wizard = _wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        await _open_protect(pilot, wizard)

        await pilot.press("escape")
        await _until(
            pilot,
            lambda: _confirm_label(app, wizard) == "Exit setup",
            what="the Exit setup dialog",
        )
        await pilot.pause(0.6)  # past the dialog's double-Escape grace
        app.screen.query_one("#confirm-button", Button).press()
        await _until(
            pilot,
            lambda: "Exit without saving progress" in _confirm_label(app, wizard),
            what="the exit-without-saving offer",
        )
        app.screen.query_one("#confirm-button", Button).press()
        await _until(
            pilot, lambda: app.wizard_result != "UNSET", what="the wizard to close"
        )

        assert app.wizard_result is None
