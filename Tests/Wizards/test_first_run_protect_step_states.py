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
import tldw_chatbook.UI.Wizards.first_run_summary_step as summary_step_module
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
    # Summary's speech-model check resolves the data dir through the config
    # cache, which the patched loader above never fills; since TASK-36478 each
    # bootstrap_profile test starts with that cache dropped, so the check
    # raised RecoveryRequired. No speech model is part of these tests.
    monkeypatch.setattr(
        summary_step_module,
        "managed_model_artifact_root",
        lambda: CONFIG_PATH.parent / "no-managed-models",
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


def _pinned_error(wizard) -> str:
    """The wizard's pinned error strip, or "" when it is hidden."""
    strip = wizard.query_one("#setup-step-error-pinned", Static)
    return "" if strip.has_class("hidden") else str(strip.render()).strip()


def _progress_box(wizard, step_id: str) -> str:
    """The tracker glyph for ``step_id``, or "" while the tracker recomposes.

    ``SetupWizardProgress.set_items`` rebuilds the tracker with
    ``refresh(recompose=True)``, so a read that lands between the old boxes'
    removal and the new ones' mount finds no box (seen under load).
    """
    boxes = wizard.query(f"#setup-progress-{step_id} .step-number")
    return str(boxes.first(Static).render()) if boxes else ""


async def _until_progress_box(pilot, wizard, step_id: str, glyph: str) -> None:
    """Wait for the recomposed tracker to show ``glyph`` for ``step_id``."""
    await _until(
        pilot,
        lambda: _progress_box(wizard, step_id) == glyph,
        what=f"the {step_id} tracker box to read {glyph!r}",
    )


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
        await _until_progress_box(pilot, wizard, STEP_PROTECT, "–")


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
        # Review round 1 (R1-F4): the footer says "Enter / Ctrl+N next", so
        # Enter must not go backwards; Back to Model is one Tab away.
        next_button = wizard.query_one("#wizard-next", Button)
        await _until(pilot, lambda: app.focused is next_button, what="focus on Next")
        await pilot.pause(0.3)  # no late re-focus onto Back to Model
        assert app.focused is next_button

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
        # Review round 1 (R1-F7): kept as plain text (which Summary flags) is
        # not the ✓ that reads "your keys are protected".
        await _until_progress_box(pilot, wizard, STEP_PROTECT, "–")


@pytest.mark.asyncio
async def test_a_long_config_path_is_shown_whole(monkeypatch):
    # Review round 1 (F5): middle-truncating the path dropped the very
    # directories that say where the file is ('…/ho…config.toml').
    long_path = Path(
        "/Volumes/External Drive/profiles/research-workstation/"
        "custom-xdg-config-home/tldw_cli/profiles/work/config.toml"
    )
    _install(monkeypatch, {"api_settings": {"openai": {"api_key": "sk-plain-x"}}})
    monkeypatch.setattr(config_module, "get_cli_config_path", lambda: long_path)
    wizard = _wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause(0.3)
        _container, step = await _open_protect(pilot, wizard)
        path = step.query_one("#setup-protect-path", Static)
        await _until(pilot, lambda: path.display, what="the path line")
        shown = str(path.render())

        assert "…" not in shown
        assert "".join(shown.removeprefix("Config file:").split()) == "".join(
            str(long_path).split()
        )
        for line in shown.splitlines()[1:-1]:
            assert line.endswith("/")  # broken only at separators


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
    # TE-9: the running app's in-memory view holds the same key, so the
    # pre-fix step reaches the behaviour under test, not a precondition.
    wizard = _wizard({"api_settings": {"openai": {"api_key": "sk-plain-openai-x"}}})
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        container, step = await _open_protect(pilot, wizard)
        primary = step.query_one("#setup-protect-set-password", Button)
        await _until(pilot, lambda: primary.display, what="the encrypt button")
        step._on_password_result("first-master-pw")
        await _until(
            pilot,
            lambda: "Encryption enabled" in _step_text(step),
            what="the success line",
        )

        # The user moves to "Change password…" and presses Enter, so the
        # dialog returns focus to that button when it closes (TE-5: a
        # programmatic press left focus on Next, and the guard below could
        # not fail).
        primary.focus()
        await _until(pilot, lambda: app.focused is primary, what="focus on Change")
        await pilot.press("enter")
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
        # TE-5: after a change too, Next takes focus so Enter continues
        # instead of reopening the change dialog.
        await _until(
            pilot,
            lambda: app.focused is wizard.query_one("#wizard-next", Button),
            what="focus on Next",
        )
        await pilot.press("enter")
        await _at_summary(pilot, app, container)
        assert not isinstance(app.screen, PasswordDialog)


@pytest.mark.asyncio
async def test_a_refused_enable_never_claims_plain_text_beside_a_tick(monkeypatch):
    # The file was encrypted under another password between the read and the
    # click: enable refuses, and the step must say so -- never "✓ Encryption
    # enabled." next to "your keys are unchanged (plain text)".
    disk = _install(
        monkeypatch, {"api_settings": {"openai": {"api_key": "sk-plain-openai-x"}}}
    )
    wizard = _wizard({"api_settings": {"openai": {"api_key": "sk-plain-openai-x"}}})
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


@pytest.mark.asyncio
async def test_a_failed_enable_then_a_successful_one_leaves_no_stale_failure(
    monkeypatch,
):
    # Review round 1 (R1-F2/F1): the pinned strip kept "Encrypting failed…"
    # beside the later "✓ Encryption enabled." -- the contradiction AC #1
    # exists to remove.
    disk = _install(
        monkeypatch,
        {"api_settings": {"openai": {"api_key": "sk-plain-openai-x"}}},
        enable_result=False,
    )
    wizard = _wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        _container, step = await _open_protect(pilot, wizard)
        primary = step.query_one("#setup-protect-set-password", Button)
        await _until(pilot, lambda: primary.display, what="the encrypt button")

        step._on_password_result("first-master-pw")
        await _until(pilot, lambda: _pinned_error(wizard) != "", what="the failure")
        await _until(pilot, lambda: not step._busy, what="the attempt to settle")

        def _enable_now_works(password: str) -> bool:
            disk.enable_calls.append(password)
            disk.document = {**disk.document, "encryption": {"enabled": True}}
            return True

        monkeypatch.setattr(
            config_module, "enable_config_encryption", _enable_now_works
        )
        step._on_password_result("first-master-pw")
        await _until(
            pilot,
            lambda: "Encryption enabled" in _step_text(step),
            what="the success line",
        )

        assert _pinned_error(wizard) == ""
        assert disk.enable_calls == ["first-master-pw", "first-master-pw"]


@pytest.mark.asyncio
async def test_a_wrong_current_password_then_the_right_one_leaves_no_stale_failure(
    monkeypatch,
):
    disk = _install(
        monkeypatch,
        {
            "encryption": {"enabled": True},
            "api_settings": {"openai": {"api_key": "sk-decrypted-openai-x"}},
        },
    )
    monkeypatch.setattr(
        config_module,
        "verify_config_encryption_password",
        lambda password: password == "right-master-pw",
    )
    wizard = _wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        _container, step = await _open_protect(pilot, wizard)
        primary = step.query_one("#setup-protect-set-password", Button)
        await _until(
            pilot, lambda: str(primary.label) == "Change password…", what="state"
        )

        step._on_change_result(PasswordChange("wrong-master-pw", "next-master-pw"))
        await _until(
            pilot, lambda: "didn't match" in _pinned_error(wizard), what="the refusal"
        )
        await _until(pilot, lambda: not step._busy, what="the attempt to settle")
        step._on_change_result(PasswordChange("right-master-pw", "next-master-pw"))
        await _until(
            pilot,
            lambda: "Master password changed" in _step_text(step),
            what="the change line",
        )

        assert _pinned_error(wizard) == ""
        assert disk.change_calls == [("right-master-pw", "next-master-pw")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("on_disk", "title"),
    [(None, "Couldn't read config.toml"), (True, "Your keys are protected")],
    ids=["unknown", "encrypted"],
)
async def test_an_unreadable_config_never_falls_back_to_plain_text(
    monkeypatch, on_disk, title
):
    # Review round 1 (R1-F5/TE-2): the fallback was the in-memory view, which
    # has no [encryption] table and holds decrypted keys -- so an encrypted,
    # unlocked config read "saved as plain text" with an Encrypt offer.
    _install(monkeypatch, {"encryption": {"enabled": True}})

    def _unreadable(force_reload=False):
        raise OSError("config.toml is unreadable")

    monkeypatch.setattr(
        config_module, "load_cli_config_and_ensure_existence", _unreadable
    )
    monkeypatch.setattr(
        config_module, "config_encryption_enabled_on_disk", lambda: on_disk
    )
    wizard = _wizard({"api_settings": {"openai": {"api_key": "sk-decrypted-openai-x"}}})
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        _container, step = await _open_protect(pilot, wizard)
        await _until(pilot, lambda: title in _step_text(step), what="the state")
        text = _step_text(step)

        assert "plain text" not in text
        assert "Encrypt with a password…" not in _shown_buttons(step)


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


@pytest.mark.asyncio
async def test_keep_going_after_a_failed_exit_save_stays_on_the_step(monkeypatch):
    # TE-10: the other branch of the exit-without-saving offer.
    _install(monkeypatch, {})
    monkeypatch.setattr(
        config_module, "save_settings_to_cli_config", lambda *a, **k: False
    )
    wizard = _wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        container, _step = await _open_protect(pilot, wizard)

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
        app.screen.query_one("#cancel-button", Button).press()
        await _until(pilot, lambda: app.screen is wizard, what="the wizard again")
        await pilot.pause(0.2)

        assert app.wizard_result == "UNSET"
        assert _current_step_id(container) == STEP_PROTECT


def _save_raises(*_args, **_kwargs):
    # What an immutable config.toml really does (live, `chflags uchg`): the
    # config writer raises RecoveryRequired instead of returning False.
    raise RuntimeError("config.toml could not be replaced")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing_save",
    [lambda *a, **k: False, _save_raises],
    ids=["returns-false", "raises"],
)
async def test_skip_setup_offers_exit_without_saving_when_completion_cannot_save(
    monkeypatch, failing_save
):
    # Review round 1 (R1-F6): on Welcome, Esc is "Skip setup", which saves
    # completion. When that save kept failing the only way out was Ctrl+Q.
    # Live, the save raised, and the step guard showed "Something went
    # wrong on this step" with no way out either.
    _install(monkeypatch, {})
    monkeypatch.setattr(config_module, "save_settings_to_cli_config", failing_save)
    wizard = _wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        container = wizard.query_one(SetupWizardContainer)
        assert _current_step_id(container) == "welcome"

        await pilot.press("escape")
        await _until(
            pilot,
            lambda: _confirm_label(app, wizard) == "Skip setup",
            what="the Skip setup dialog",
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


# --- the Provider step's record of a typed key ----------------------------------


@pytest.mark.asyncio
async def test_the_typed_key_record_follows_the_provider_committed_last():
    # Review round 1 (R1-F3): `entered_key` stayed True once set, while
    # `provider_key` followed the newest choice. Type an OpenAI key, Next,
    # Back, switch to keyless llama.cpp: Protect then said "The llama.cpp key
    # you typed isn't saved" and sent the user to Model.
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from textual.widgets import Input

    from Tests.Wizards.test_first_run_setup_wizard import _provider_step, _StepHost
    from tldw_chatbook.UI.Wizards.first_run_protect_state import typed_provider_key

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        step.query_one("#setup-provider-api-key", Input).value = "sk-typed-openai"
        ok, error = await step.commit()
        assert ok, error
        assert typed_provider_key({STEP_PROVIDER: step.get_step_data()}) == "openai"

        step.select_provider("llama_cpp")
        await pilot.pause()
        ok, error = await step.commit()
        assert ok, error

        assert typed_provider_key({STEP_PROVIDER: step.get_step_data()}) == ""
