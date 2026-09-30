"""TASK-33621.14: the first-run Provider step survives every provider it lists.

Review finding G3-01 (2026-09-29): on Quick setup ▸ Provider the fourth Down
arrow landed on the first Cloud row and raised ``ValueError('Provider is not
supported.')`` from ``provider_setup_persistence._ownership_for`` -- the
wizard listed the whole ``chat_api_call`` handler catalog, but setup
persistence owned only a hand-kept subset of it. The step went blank, the
keyboard went dead, and Next (or Back then Next) quit the app. A session
resumed from "Continue setup?" also had Back disabled on the Provider step.

TASK-33510 since made setup own every engine preset; the one handler key it
still cannot own is execution-only ``custom_hosted``, which the wizard listed
under Local. These cases drive the real ``FirstRunSetupWizard`` through real
key presses. The per-provider case is parametrised over every provider the
app can run (the live handler catalog plus the Settings picker's set), so a
preset added after this review gets its own case the day it lands, and the
pre-fix list (``custom_hosted`` included) fails it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from loguru import logger
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_provider_support import (
    supported_console_provider_catalog,
)
from tldw_chatbook.Chat.console_session_settings import settings_provider_catalog
from tldw_chatbook.UI.Wizards import first_run_step_guard as step_guard
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    SETUP_DRAFT_VERSION,
    STEP_MODEL,
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


def _candidate_provider_keys() -> tuple[str, ...]:
    """Every provider the wizard could list: runnable handlers plus Settings'.

    Enumerated live at collection, so a preset registered later gets a case,
    and so does a handler key the wizard must NOT list (``custom_hosted``):
    the pre-fix Provider step listed the whole handler catalog.
    """
    handlers = {entry.readiness_key for entry in supported_console_provider_catalog()}
    return tuple(sorted(handlers | set(_settings_picker_provider_keys())))


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

    settings_entries = SettingsScreen(_app_instance())._provider_catalog_entries()
    assert settings_entries == settings_provider_catalog()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", _candidate_provider_keys())
async def test_every_listed_provider_can_be_highlighted_and_selected(provider):
    """AC#1/AC#5: arrow onto the row, select it; the step stays whole.

    A runnable provider the wizard does not list must be one the Settings
    picker withholds too (AC#2). The headless host re-raises a handler error,
    so a listed row whose selection raises fails its own case.
    """
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, step = _current_step(wizard)
        if provider not in _listed_provider_keys(step):
            assert provider not in _settings_picker_provider_keys()
            return
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


@pytest.mark.asyncio
async def test_arrowing_through_every_listed_row_keeps_the_step_live():
    """AC#1: the finding's own gesture, over the whole list, in one session.

    Down from the first row to the last, one real key press at a time, the
    way a new user browses. Each row must select, keep the step and the
    keyboard, and show no error line.
    """
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, step = _current_step(wizard)
        listed = _listed_provider_keys(step)
        choices = _provider_list(step)
        choices.focus()
        await pilot.pause()
        await _arrow_to(pilot, choices, listed[0])

        visited = [choices.highlighted_option.provider_key]
        for expected in listed[1:]:
            await pilot.press("down")
            await pilot.pause()
            assert choices.highlighted_option.provider_key == expected
            assert step.selected_provider_key == expected
            assert _pinned_error(wizard) == "", expected
            visited.append(expected)
        _assert_step_is_live(app, wizard, step)
        assert app.focused is choices

    assert visited == listed
    assert len(listed) == len(_settings_picker_provider_keys())


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
        assert "Couldn't switch to" in _pinned_error(wizard)

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
    """AC#4: a session resumed from "Continue setup?" can still go Back.

    ``BaseWizard``'s nav bar refreshes Back only when ``current_step`` or
    ``can_go_forward`` changes, so setup's ``update_progress`` must set
    ``can_go_back`` first. It set it after, and a resume that jumped straight
    to Provider showed a disabled Back.
    """
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


@pytest.mark.asyncio
@private_profile_test
async def test_a_failed_provider_switch_keeps_the_typed_key_with_its_provider(
    request, monkeypatch
):
    """A contained switch error must not hand OpenAI's typed key to Anthropic.

    Review finding (major): ``select_provider`` recorded the new provider
    before its fallible reads and before it swapped the key field. A failure
    in between left Anthropic selected with the key typed for OpenAI still in
    the field; the error guard kept the step alive, and Next staged that key
    as Anthropic's credential (model discovery would then send it to
    Anthropic). The switch now reads everything that can fail before any
    state moves, so a failed switch leaves OpenAI whole. Runs in a private
    profile because Next writes the setup checkpoint.
    """
    from tldw_chatbook.UI.Wizards import first_run_setup_state

    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    real_presence = first_run_setup_state.read_provider_secret_presence

    def _fail_for_anthropic(*args, provider_key, **kwargs):
        if provider_key == "anthropic":
            raise ValueError("Provider is not supported.")
        return real_presence(*args, provider_key=provider_key, **kwargs)

    monkeypatch.setattr(
        first_run_setup_state, "read_provider_secret_presence", _fail_for_anthropic
    )
    typed = "sk-typed-for-openai-REVIEWPROBE-0001"
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard, keep_alive=True)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, step = _current_step(wizard)
        choices = _provider_list(step)
        choices.focus()
        await pilot.pause()
        await _arrow_to(pilot, choices, "openai")
        assert step.selected_provider_key == "openai"
        key_input = step.query_one("#setup-provider-api-key", Input)
        assert key_input.display, "the OpenAI key field is hidden"
        key_input.focus()
        await pilot.pause()
        await pilot.press(*typed)
        await pilot.pause()
        assert key_input.value == typed

        choices.focus()
        await pilot.pause()
        await pilot.press("down")  # onto Anthropic: its switch raises
        await pilot.pause()
        assert choices.highlighted_option.provider_key == "anthropic"
        # The failed switch moved nothing: OpenAI is still selected, with its key,
        # and the line says so, since the highlight sits on Anthropic.
        message = _pinned_error(wizard)
        assert "Couldn't switch to Anthropic" in message
        assert "OpenAI is still selected" in message
        assert step.selected_provider_key == "openai"
        assert key_input.value == typed

        await pilot.press("ctrl+n")
        await pilot.pause(0.5)
        assert app.is_running and app._exception is None
        staged = step.wizard._staged_provider_draft  # the container stages it
        # Control: the typed key did reach the staging boundary ...
        assert staged is not None
        # ... and only as the credential of the provider it was typed for.
        assert staged.provider == "openai"
        value = first_run_setup_state._credential_value_for_boundary(staged.credential)
        assert value == typed


@pytest.mark.asyncio
async def test_a_provider_pick_that_succeeds_clears_the_failed_switch_line(
    monkeypatch,
):
    """Review (minor): the error line goes once the user recovers.

    After a failed switch, the next pick that works must take the "couldn't
    switch" line down (it described a failure the step no longer has), and
    only that line: a message another part of the step wrote stays.
    """
    from tldw_chatbook.UI.Wizards import first_run_setup_state

    real_presence = first_run_setup_state.read_provider_secret_presence

    def _fail_for_anthropic(*args, provider_key, **kwargs):
        if provider_key == "anthropic":
            raise ValueError("Provider is not supported.")
        return real_presence(*args, provider_key=provider_key, **kwargs)

    monkeypatch.setattr(
        first_run_setup_state, "read_provider_secret_presence", _fail_for_anthropic
    )
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard, keep_alive=True)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, step = _current_step(wizard)
        choices = _provider_list(step)
        choices.focus()
        await pilot.pause()
        await _arrow_to(pilot, choices, "openai")
        listed = _listed_provider_keys(step)
        after = listed[listed.index("anthropic") + 1]

        await pilot.press("down")  # onto Anthropic: its switch raises
        await pilot.pause()
        assert choices.highlighted_option.provider_key == "anthropic"
        assert "Couldn't switch to Anthropic" in _pinned_error(wizard)
        assert step.selected_provider_key == "openai"

        await pilot.press("down")  # the next row works
        await pilot.pause()
        assert step.selected_provider_key == after
        assert _pinned_error(wizard) == ""

        # Only the guard's own line goes: a line written over it stays.
        await pilot.press("up")  # Anthropic again: fails
        await pilot.pause()
        assert "Couldn't switch to Anthropic" in _pinned_error(wizard)
        step.show_step_error("A refused Next says why here.")
        await pilot.press("up")  # OpenAI works
        await pilot.pause()
        assert step.selected_provider_key == "openai"
        assert _pinned_error(wizard) == "A refused Next says why here."
        assert app.is_running and app._exception is None


@pytest.mark.asyncio
async def test_a_next_that_fails_after_changing_step_does_not_claim_it_stayed(
    monkeypatch,
):
    """Review (nit): "setup stayed here" only when setup did stay.

    Next from Welcome commits it, then shows Provider. When Provider's
    ``on_show`` raises, the wizard is already on Provider: the line there
    must not say setup stayed, and the log names the step Next committed.
    """
    records: list[dict] = []
    sink = logger.add(
        lambda message: records.append(message.record),
        level="ERROR",
        filter=lambda record: "First-run setup error contained" in record["message"],
    )
    calls: list[str] = []
    real_on_show = ProviderStep.on_show

    def _raise_once(self):  # no event parameter: Textual's Show also calls it
        calls.append("on_show")
        if len(calls) == 1:
            raise RuntimeError("step show failure")
        real_on_show(self)

    wizard = _resumed_on_provider()
    app = _WizardHost(wizard, keep_alive=True)
    try:
        async with app.run_test(size=_SIZE) as pilot:
            await _wait_for_step(pilot, wizard, STEP_PROVIDER)
            await pilot.press("ctrl+b")
            await _wait_for_step(pilot, wizard, STEP_WELCOME)
            monkeypatch.setattr(ProviderStep, "on_show", _raise_once)

            await pilot.press("ctrl+n")
            await _wait_for_step(pilot, wizard, STEP_PROVIDER)
            await pilot.pause(0.3)

            assert calls, "Provider was never shown"
            assert app.is_running and app._exception is None
            message = _pinned_error(wizard)
            assert "went wrong" in message
            assert "stayed here" not in message
            assert wizard.query_one("#wizard-back", Button).disabled is False
    finally:
        logger.remove(sink)

    assert len(records) == 1, [record["message"] for record in records]
    assert "category=advance" in records[0]["message"]
    assert f"step={STEP_WELCOME}," in records[0]["message"]


@pytest.mark.asyncio
async def test_a_raising_wizard_container_handler_keeps_the_wizard(monkeypatch):
    """AC#3: the guard covers the wizard container's own handlers, not only steps.

    Back is handled by ``SetupWizardContainer`` itself. Unguarded, a raise
    there stops the container's message loop and the whole wizard body goes.
    """
    calls = []
    real_previous = SetupWizardContainer._previous_active_index

    def _raise_once(self, absolute_index):
        calls.append(absolute_index)
        if len(calls) == 1:
            raise RuntimeError("container handler failure")
        return real_previous(self, absolute_index)

    monkeypatch.setattr(SetupWizardContainer, "_previous_active_index", _raise_once)
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard, keep_alive=True)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        container, step = _current_step(wizard)

        await pilot.click("#wizard-back")
        await pilot.pause()

        assert len(calls) == 1
        assert app.is_running and app._exception is None
        assert container.is_attached and container.display
        _assert_step_is_live(app, wizard, step)
        assert "went wrong" in _pinned_error(wizard)

        await pilot.pause(0.3)  # a Button ignores clicks during its press effect
        await pilot.click("#wizard-back")
        await _wait_for_step(pilot, wizard, STEP_WELCOME)
        assert len(calls) == 2


def _raise_on_first_call(monkeypatch, owner: type, name: str) -> list[str]:
    """Make ``owner.name`` raise the first time it runs, then behave."""
    calls: list[str] = []
    real = getattr(owner, name)

    def _patched(self, *args, **kwargs):
        calls.append(name)
        if len(calls) == 1:
            raise RuntimeError("keyboard action failure")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(owner, name, _patched)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "owner", "name"),
    [
        ("ctrl+b", SetupWizardContainer, "_previous_active_index"),
        ("ctrl+n", ProviderStep, "confirm_before_advance"),
        ("escape", SetupWizardContainer, "hold_provider_save_settlement"),
    ],
)
async def test_a_raising_keyboard_navigation_action_keeps_the_wizard(
    monkeypatch, key, owner, name
):
    """AC#3/AC#4, review (major): a key binding is contained like a click.

    Ctrl+B, Ctrl+N and Esc are the wizard container's key bindings. Textual
    runs a binding's action in the App's own message loop, not the
    container's, so the container's handler guard never saw it, and the app
    refuses to keep its own loop alive: the failure a click on Back survives
    (the case above) quit the app from the keyboard.
    """
    calls = _raise_on_first_call(monkeypatch, owner, name)
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard, keep_alive=True)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        container, step = _current_step(wizard)
        _provider_list(step).focus()
        await pilot.pause()

        await pilot.press(key)
        await pilot.pause()

        assert calls == [name]
        assert app.is_running and app._exception is None
        assert app.screen is wizard
        _assert_step_is_live(app, wizard, step)
        assert "went wrong" in _pinned_error(wizard)
        assert container._advancing is False
        for button in ("#wizard-back", "#wizard-next", "#wizard-cancel"):
            assert wizard.query_one(button, Button).disabled is False, button

        # The same key works once the failure has passed.
        await pilot.press(key)
        await pilot.pause()
        assert len(calls) >= 2  # Esc's container and screen both ask
        if key == "ctrl+b":
            await _wait_for_step(pilot, wizard, STEP_WELCOME)
        elif key == "escape":
            assert isinstance(app.screen, _SettlingGuardedConfirmationDialog)
        assert app.is_running and app._exception is None


def test_every_wizard_container_key_binding_runs_a_contained_action():
    """Review (major): each key the container binds reaches a guarded action.

    Pins the structural half of the case above: a binding added later, or
    one whose action still lives in the unguarded ``BaseWizard`` class, would
    quit the app from the keyboard again.
    """
    actions: set[str] = set()
    for cls in SetupWizardContainer.__mro__:
        for binding in vars(cls).get("BINDINGS", ()):
            actions.add(getattr(binding, "action", None) or binding[1])
    assert {"next", "back", "cancel"} <= actions
    for action in sorted(actions):
        method = getattr(SetupWizardContainer, f"action_{action}")
        assert getattr(method, "_first_run_contained_action", False), action


@pytest.mark.asyncio
async def test_back_into_a_step_whose_show_raises_keeps_the_keyboard(monkeypatch):
    """AC#3/AC#4, review (major): re-entering a step that fails to show.

    Ctrl+B runs ``show_step``, whose ``on_show`` for the step being entered
    is where the finding's ``NoMatches`` came from. A failure there leaves
    the new step half shown: the nav bar still counts the old step, and focus
    sits in the step just hidden, which Textual then drops, so no binding
    resolves. The wizard must land on the entered step with its error line,
    a nav bar that matches it, and a keyboard that still works.
    """
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import WelcomeStep

    calls: list[str] = []
    real_on_show = WelcomeStep.on_show

    def _raise_once(self):  # no event parameter: Textual's Show also calls it
        calls.append("on_show")
        if len(calls) == 1:
            raise RuntimeError("step show failure")
        real_on_show(self)

    wizard = _resumed_on_provider()
    app = _WizardHost(wizard, keep_alive=True)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = _current_step(wizard)
        _provider_list(provider).focus()
        await pilot.pause()
        monkeypatch.setattr(WelcomeStep, "on_show", _raise_once)

        await pilot.press("ctrl+b")
        await _wait_for_step(pilot, wizard, STEP_WELCOME)
        await pilot.pause(0.2)

        assert calls, "Welcome was never shown"  # Textual's Show event re-runs it
        assert app.is_running and app._exception is None
        _container, welcome = _current_step(wizard)
        assert welcome.display and not provider.display
        message = _pinned_error(wizard)
        assert "went wrong" in message and "Back" not in message
        # The nav bar follows the step that is on screen: Welcome has no Back.
        assert wizard.query_one("#wizard-back", Button).disabled
        focused = app.focused
        assert focused is not None and focused.is_attached
        assert welcome in focused.ancestors_with_self or (
            focused in wizard.query("#wizard-next, #wizard-cancel")
        ), f"focus stayed in a hidden step: {focused!r}"

        # Esc resolves (it needs a live focus chain) and asks before leaving.
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, _SettlingGuardedConfirmationDialog)
        assert app.is_running and app._exception is None


@pytest.mark.asyncio
async def test_a_contained_error_that_dropped_focus_puts_the_keyboard_back(
    monkeypatch,
):
    """AC#3: an error that left nothing focused re-anchors focus on the step.

    With no focused widget, Ctrl+B / Ctrl+N (bound on the wizard container)
    have no focus chain to resolve through, and the keyboard is dead.
    """
    from tldw_chatbook.UI.Wizards import first_run_setup_state

    hosts: list[App] = []

    def _drop_focus_then_raise(*_args, **_kwargs):
        hosts[0].screen.set_focus(None)
        raise ValueError("Provider is not supported.")

    monkeypatch.setattr(
        first_run_setup_state, "read_provider_secret_presence", _drop_focus_then_raise
    )
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard, keep_alive=True)
    hosts.append(app)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, step = _current_step(wizard)
        choices = _provider_list(step)
        choices.focus()
        await pilot.pause()

        await pilot.press("down")
        await pilot.pause()

        assert app.is_running and app._exception is None
        assert app.focused is choices, "focus was not re-anchored on the step"
        await pilot.press("ctrl+b")
        await _wait_for_step(pilot, wizard, STEP_WELCOME)


def _raised(error: Exception) -> Exception:
    """Return ``error`` with a real traceback, as a handler would raise it."""
    try:
        raise error
    except Exception as caught:
        return caught


@pytest.mark.asyncio
async def test_a_contained_error_is_attributed_to_the_step_that_raised():
    """Review (minor): log the raising step; pin copy only on the shown step.

    A hidden step's error must not put "something went wrong on this step"
    over a healthy step, and a handler that fails on every tick (a 250 ms
    interval) must not log an ERROR each time.
    """
    records: list[dict] = []
    sink = logger.add(
        lambda message: records.append(message.record),
        level="DEBUG",
        filter=lambda record: "First-run setup error contained" in record["message"],
    )
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard, keep_alive=True)
    try:
        async with app.run_test(size=_SIZE) as pilot:
            await _wait_for_step(pilot, wizard, STEP_PROVIDER)
            container, provider = _current_step(wizard)
            welcome = next(
                step
                for step in container.steps
                if step.config is not None and step.config.id == STEP_WELCOME
            )
            assert not welcome.display

            step_guard.report_contained_error(
                welcome, "handler", _raised(ValueError("hidden"))
            )
            assert _pinned_error(wizard) == ""

            repeated = _raised(ValueError("shown"))
            for _ in range(3):
                step_guard.report_contained_error(provider, "handler", repeated)
            assert "went wrong" in _pinned_error(wizard)
            await pilot.pause()
    finally:
        logger.remove(sink)

    errors = [record for record in records if record["level"].name == "ERROR"]
    assert len(errors) == 2, [record["message"] for record in records]
    assert f"step={STEP_WELCOME}," in errors[0]["message"]
    assert f"step={STEP_PROVIDER}," in errors[1]["message"]
    assert all("hidden" not in r["message"] for r in records)
    assert all("shown" not in r["message"] for r in records)


@pytest.mark.asyncio
async def test_error_copy_offers_back_only_where_back_works():
    """Review (nit): Welcome has no Back, so its error copy must not offer it."""
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard, keep_alive=True)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = _current_step(wizard)
        step_guard.report_contained_error(provider, "handler", _raised(KeyError()))
        assert "Back" in _pinned_error(wizard)

        await pilot.press("ctrl+b")
        await _wait_for_step(pilot, wizard, STEP_WELCOME)
        _container, welcome = _current_step(wizard)
        assert wizard.query_one("#wizard-back", Button).disabled
        for category in ("handler", "advance"):
            step_guard.report_contained_error(welcome, category, _raised(KeyError()))
            message = _pinned_error(wizard)
            assert "went wrong" in message and "Esc" in message
            assert "Back" not in message


def _fresh_wizard() -> FirstRunSetupWizard:
    return FirstRunSetupWizard(_app_instance())


@pytest.mark.asyncio
@private_profile_test
async def test_fresh_quick_setup_arrows_into_cloud_then_back_and_next_stay_open(
    request,
):
    """The finding's repro, fresh profile: Enter, Down x4, Back, Next, then on.

    Pre-fix the fourth Down (the first Cloud row) blanked the step, and
    re-entering it quit the app. TASK-33510 made that row ownable, so after
    Back, Next and a keyless Next the walk goes on to the last row: the
    pre-fix list still carried Custom Hosted under Local. Runs in a private
    profile because Next from Welcome writes the setup checkpoint to that
    profile's config file.
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
        listed = _listed_provider_keys(step)

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

        # Then on through every remaining row, one press per row.
        choices.focus()
        await pilot.pause()
        assert choices.highlighted_option.provider_key == listed[4] == first_cloud
        for expected in listed[5:]:
            await pilot.press("down")
            await pilot.pause()
            assert step.selected_provider_key == expected
            assert "went wrong" not in _pinned_error(wizard), expected
        _assert_step_is_live(app, wizard, step)
        assert app.is_running and app._exception is None


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["byteplus", "deepinfra"])
@private_profile_test
async def test_a_newly_owned_hosted_preset_saves_through_the_wizard(
    provider, request, monkeypatch
):
    """Review (major): a preset setup newly owns saves end to end from first run.

    TASK-33510 made setup own every engine preset. BytePlus (``/api/v3``) and
    DeepInfra (``/v1/openai``) are the two whose documented base URL a save
    used to rewrite or refuse, so an earlier cut of this fix withheld them from
    the list. Pick the row with the keyboard, type a key, Next, type a model,
    Next: the Model step's commit writes the private profile's real config
    file, and the shipped base URL must come back unchanged.
    """
    from tldw_chatbook.config import load_cli_config_and_ensure_existence
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY

    record = RECORDS_BY_KEY[provider]
    assert record.default_base_url and record.api_key_env_var
    monkeypatch.delenv(record.api_key_env_var, raising=False)
    typed_key = f"sk-{provider}-wizard-save-0001"
    model = f"{provider}-wizard-model"
    wizard = _resumed_on_provider()
    app = _WizardHost(wizard)
    async with app.run_test(size=_SIZE) as pilot:
        await _wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, step = _current_step(wizard)
        choices = _provider_list(step)
        choices.focus()
        await pilot.pause()
        await _arrow_to(pilot, choices, provider)
        assert step.selected_provider_key == provider
        key_input = step.query_one("#setup-provider-api-key", Input)
        assert key_input.display, "the key field is hidden"
        key_input.focus()
        await pilot.pause()
        await pilot.press(*typed_key)
        await pilot.pause()

        await pilot.press("ctrl+n")
        await _wait_for_step(pilot, wizard, STEP_MODEL)
        _container, model_step = _current_step(wizard)
        custom = model_step.query_one("#setup-model-custom", Input)
        custom.focus()
        await pilot.pause()
        await pilot.press(*model)
        await pilot.pause()
        await pilot.press("ctrl+n")
        for _ in range(150):
            if _current_step(wizard)[1] is not model_step:
                break
            await pilot.pause(0.05)
        assert _current_step(wizard)[1] is not model_step, _pinned_error(wizard)
        assert app.is_running and app._exception is None

    saved = load_cli_config_and_ensure_existence(force_reload=True)
    table = saved["api_settings"][provider]
    assert table["api_key"] == typed_key
    assert table["model"] == model
    assert table["api_base_url"] == record.default_base_url
