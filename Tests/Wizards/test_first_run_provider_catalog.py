"""TASK-33621.14: the first-run Provider step survives every provider it lists.

Review finding G3-01 (2026-09-29): on Quick setup ▸ Provider the fourth Down
arrow landed on the first Cloud row and raised ``ValueError('Provider is not
supported.')`` from ``provider_setup_persistence._ownership_for`` -- the
wizard listed the whole handler catalog, but setup persistence owned only a
hand-kept subset of it. The step went blank, the keyboard went dead, and Next
(or Back then Next) quit the app. A session resumed from "Continue setup?"
also had Back disabled on the Provider step.

These cases drive the real ``FirstRunSetupWizard`` through real key presses.
The per-provider case is parametrised over the Settings picker's provider set
(``settings_provider_catalog()``, which ``SettingsScreen`` reads), so a
provider preset added after this review gets its own case the day it lands in
the catalog.
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_session_settings import settings_provider_catalog
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    SETUP_DRAFT_VERSION,
    STEP_PROVIDER,
    STEP_WELCOME,
    TRACK_QUICK,
    SetupDraft,
)
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    FirstRunSetupWizard,
    ProviderChoiceList,
    ProviderStep,
    SetupWizardContainer,
    _SettlingGuardedConfirmationDialog,
)

_SIZE = (235, 52)

# Selecting a provider reads the atomic config snapshot, which goes through
# the config-participant admission: keep the collection-time profile, the
# per-node opt-in Tests/conftest.py documents (TASK-32873).
pytestmark = pytest.mark.bootstrap_profile


def _settings_picker_provider_keys() -> tuple[str, ...]:
    """The Settings picker's provider set."""
    return tuple(entry.readiness_key for entry in settings_provider_catalog())


class _WizardHost(App):
    # The real app stylesheet: the nav bar's geometry (and so every click on
    # Back / Next / Exit setup) only holds with it loaded.
    CSS_PATH = str(
        Path(__file__).resolve().parents[2] / "tldw_chatbook/css/tldw_cli_modular.tcss"
    )

    def __init__(self, wizard: FirstRunSetupWizard, *, keep_alive: bool = False):
        super().__init__()
        self._wizard = wizard
        self.wizard_results: list[object] = []
        if keep_alive:
            # The production app's policy (``app_lifecycle._handle_exception``):
            # outside headless runs a raising handler keeps the screen alive.
            self._keep_screen_alive_on_handler_error = True

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.push_screen(self._wizard, self.wizard_results.append)


def _app_instance() -> MagicMock:
    from tldw_chatbook.config import load_settings

    app_instance = MagicMock()
    # The real loaded settings, as TldwCli holds them: an ``api_settings``
    # table is what routes a provider choice through setup ownership (an
    # empty dict short-circuits that lookup and hid the crash).
    app_instance.app_config = copy.deepcopy(load_settings())
    assert isinstance(app_instance.app_config.get("api_settings"), dict)
    # No catalog scope service: selecting a provider never reaches the network.
    app_instance.llm_provider_catalog_scope_service = None
    return app_instance


def _resumed_on_provider() -> FirstRunSetupWizard:
    """A wizard resumed from "Continue setup?" onto Quick setup ▸ Provider."""
    draft = SetupDraft(
        version=SETUP_DRAFT_VERSION,
        track=TRACK_QUICK,
        active_step_id=STEP_PROVIDER,
        values={STEP_WELCOME: {"track": TRACK_QUICK}},
    )
    return FirstRunSetupWizard(_app_instance(), resume_draft=draft)


def _current_step(wizard: FirstRunSetupWizard):
    container = wizard.query_one(SetupWizardContainer)
    return container, container.steps[container.current_step]


async def _wait_for_step(pilot, wizard, step_id: str) -> None:
    for _ in range(100):
        _container, step = _current_step(wizard)
        if step.config is not None and step.config.id == step_id and step.display:
            await pilot.pause()
            return
        await pilot.pause(0.02)
    raise AssertionError(f"the wizard never showed the {step_id!r} step")


def _provider_list(step: ProviderStep) -> ProviderChoiceList:
    return step.query_one("#setup-provider-choice", ProviderChoiceList)


def _listed_provider_keys(step: ProviderStep) -> list[str]:
    choices = _provider_list(step)
    keys = []
    for index in range(choices.option_count):
        key = getattr(choices.get_option_at_index(index), "provider_key", None)
        if key is not None:
            keys.append(key)
    return keys


async def _arrow_to(pilot, choices: ProviderChoiceList, provider: str) -> None:
    """Land on ``provider`` with one real arrow press from its neighbour row.

    The neighbour is parked programmatically (before any user interaction,
    so parking selects nothing); the last move onto the target row is a real
    key press, which is the path that raised before the fix.
    """
    rows = [
        (index, getattr(choices.get_option_at_index(index), "provider_key", None))
        for index in range(choices.option_count)
    ]
    enabled = [index for index, key in rows if key is not None]
    target = next(index for index, key in rows if key == provider)
    position = enabled.index(target)
    if position > 0:
        choices.highlighted = enabled[position - 1]
        key = "down"
    else:
        choices.highlighted = enabled[position + 1]
        key = "up"
    await pilot.pause()
    await pilot.press(key)
    await pilot.pause()
    assert choices.highlighted == target, f"{key} did not reach {provider!r}"


def _pinned_error(wizard: FirstRunSetupWizard) -> str:
    strip = wizard.query_one("#setup-step-error-pinned", Static)
    if strip.has_class("hidden"):
        return ""
    return str(strip.render())


def _assert_step_is_live(app, wizard, step: ProviderStep) -> None:
    """The step still owns the page and the keyboard (no blank, no dead focus)."""
    _container, current = _current_step(wizard)
    assert current is step
    assert step.is_attached and step.display, "the Provider step was torn down"
    assert _provider_list(step).is_attached
    focused = app.focused
    assert focused is not None and focused.is_attached, "keyboard focus was lost"


@pytest.mark.asyncio
async def test_wizard_provider_list_is_the_settings_picker_set():
    """AC#2: one provider set -- no execution-only key, no missing preset."""
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, step = _current_step(wizard)
        listed = _listed_provider_keys(step)

    assert len(listed) == len(set(listed)), "a provider is listed twice"
    assert set(listed) == set(_settings_picker_provider_keys())
    assert "custom_hosted" not in listed
    # ... and the picker really is what Settings ▸ Providers & Models shows.
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    settings_entries = SettingsScreen._provider_catalog_entries(None)
    assert settings_entries == settings_provider_catalog()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", _settings_picker_provider_keys())
async def test_every_listed_provider_can_be_highlighted_and_selected(provider):
    """AC#1/AC#5: arrow onto the row, select it; the step stays whole."""
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, step = _current_step(wizard)
        choices = _provider_list(step)
        choices.focus()
        await pilot.pause()

        await _arrow_to(pilot, choices, provider)
        await pilot.press("space")
        await pilot.pause()

        assert choices.highlighted_option.provider_key == provider
        assert step.selected_provider_key == provider
        _assert_step_is_live(app, wizard, step)
        assert app.focused is choices
        # The body shows the provider's controls, not a blank page.
        assert step.query_one("#setup-provider-auth-toggle").display
        assert _pinned_error(wizard) == ""

        # The keyboard still drives the step: Tab leaves the list.
        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is not choices
        assert app.focused is not None and app.focused.is_attached


def _fail_provider_secret_lookup(monkeypatch) -> None:
    """Make selecting a provider raise, the way an unowned provider did."""
    from tldw_chatbook.UI.Wizards import first_run_setup_state

    def _unsupported(*_args, **_kwargs):
        raise ValueError("Provider is not supported.")

    monkeypatch.setattr(
        first_run_setup_state, "read_provider_secret_presence", _unsupported
    )


@pytest.mark.asyncio
async def test_a_raising_step_handler_keeps_the_step_and_the_keyboard(monkeypatch):
    """AC#3: a caught step error leaves content, an error state, and keys."""
    _fail_provider_secret_lookup(monkeypatch)
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard, keep_alive=True)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, step = _current_step(wizard)
        choices = _provider_list(step)
        choices.focus()
        await pilot.pause()

        await pilot.press("down")
        await pilot.pause()

        assert app.is_running and app._exception is None
        _assert_step_is_live(app, wizard, step)
        assert app.focused is choices
        assert "went wrong" in _pinned_error(wizard)

        # Tab still moves focus.
        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is not choices and app.focused.is_attached

        # Esc still asks before leaving setup ...
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, _SettlingGuardedConfirmationDialog)
        app.screen._escape_grace_seconds = 0.0  # the double-press guard elapsed
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is wizard
        assert app.wizard_results == []

        # ... Ctrl+B still goes back, even on a resumed session ...
        await pilot.press("ctrl+b")
        await _wait_for_step(pilot, wizard, STEP_WELCOME)
        assert app.is_running

        # ... and re-entering the step (pre-fix: NoMatches, app exit) is safe.
        await pilot.press("ctrl+n")
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        assert app.is_running and app._exception is None
        _assert_step_is_live(app, wizard, step)


@pytest.mark.asyncio
async def test_resumed_provider_step_keeps_back_and_exit_available():
    """AC#4: a session resumed from "Continue setup?" can still go Back."""
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)

        assert wizard.query_one("#wizard-back", Button).disabled is False
        assert wizard.query_one("#wizard-cancel", Button).disabled is False

        await pilot.click("#wizard-back")
        await _wait_for_step(pilot, wizard, STEP_WELCOME)


@pytest.mark.asyncio
async def test_next_after_a_step_error_never_exits_the_app(monkeypatch):
    """AC#4: Next from a step whose handler raised stays inside the wizard."""
    _fail_provider_secret_lookup(monkeypatch)
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard, keep_alive=True)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, step = _current_step(wizard)
        _provider_list(step).focus()
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()

        await pilot.press("ctrl+n")
        await pilot.pause(0.3)

        assert app.is_running and app._exception is None
        assert app.screen is wizard
        assert _pinned_error(wizard)
        assert wizard.query_one("#wizard-back", Button).disabled is False
        assert wizard.query_one("#wizard-cancel", Button).disabled is False


@pytest.mark.asyncio
async def test_a_raising_step_commit_is_reported_not_fatal(monkeypatch):
    """AC#4: Next's commit worker must not take the app down with it."""

    async def _raising_commit(self):
        raise RuntimeError("secret-bearing commit detail")

    monkeypatch.setattr(ProviderStep, "commit", _raising_commit)
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard, keep_alive=True)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)

        await pilot.click("#wizard-next")
        await pilot.pause(0.3)

        assert app.is_running and app._exception is None
        _container, step = _current_step(wizard)
        assert step.config is not None and step.config.id == STEP_PROVIDER
        message = _pinned_error(wizard)
        assert "went wrong" in message
        assert "secret-bearing" not in message
        assert wizard.query_one("#wizard-back", Button).disabled is False
        assert wizard.query_one("#wizard-next", Button).disabled is False
        assert wizard.query_one("#wizard-cancel", Button).disabled is False


def _fresh_wizard() -> FirstRunSetupWizard:
    return FirstRunSetupWizard(_app_instance())


@pytest.mark.asyncio
@private_profile_test
async def test_fresh_quick_setup_arrows_into_cloud_then_back_and_next_stay_open(
    request,
):
    """The finding's repro, fresh profile: Enter, Down past Popular, Back, Next.

    Pre-fix the first Cloud row blanked the step, and re-entering it quit the
    app. Runs in a private profile because Next from Welcome writes the
    setup checkpoint to that profile's config file.
    """
    wizard = _fresh_wizard()
    app = _WizardHost(wizard)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_WELCOME)
        await pilot.press("enter")  # Quick setup is preselected
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, step = _current_step(wizard)
        choices = _provider_list(step)
        assert app.focused is choices

        for _ in range(4):  # OpenAI -> Anthropic -> Ollama -> llama.cpp -> Cloud
            await pilot.press("down")
            await pilot.pause()
        first_cloud = choices.highlighted_option.provider_key
        assert first_cloud not in {"openai", "anthropic", "ollama", "llama_cpp"}
        assert step.selected_provider_key == first_cloud
        _assert_step_is_live(app, wizard, step)
        assert _pinned_error(wizard) == ""

        await pilot.press("ctrl+b")
        await _wait_for_step(pilot, wizard, STEP_WELCOME)
        await pilot.press("ctrl+n")
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        _assert_step_is_live(app, wizard, step)

        # Next with no key: the step explains, the app stays.
        await pilot.press("ctrl+n")
        await pilot.pause(0.3)
        assert app.is_running and app._exception is None
        _container, current = _current_step(wizard)
        assert current is step
        assert "API key required" in _pinned_error(wizard)
