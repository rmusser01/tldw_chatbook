"""Pilot tests for the first-run setup wizard skeleton."""

from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    FirstRunSetupWizard,
    SetupWizardContainer,
)
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_PROVIDER,
    STEP_RAG,
    STEP_SUMMARY,
    TRACK_FULL,
    TRACK_QUICK,
)


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


def _make_wizard(**kwargs) -> FirstRunSetupWizard:
    app_instance = MagicMock()
    app_instance.app_config = {}
    wizard = FirstRunSetupWizard(app_instance, **kwargs)
    return wizard


@pytest.mark.asyncio
async def test_welcome_track_choice_activates_quick_steps():
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        assert STEP_PROVIDER in container.active_ids
        assert STEP_RAG not in container.active_ids
        assert container.active_ids[-1] == STEP_SUMMARY


@pytest.mark.asyncio
async def test_welcome_full_track_activates_all_non_conditional_steps():
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_FULL)
        assert STEP_RAG in container.active_ids


@pytest.mark.asyncio
async def test_escape_asks_for_confirmation_instead_of_dismissing():
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("escape")
        await pilot.pause()
        # The wizard must still be open (confirm dialog on top), not dismissed.
        assert app.wizard_result == "UNSET"


@pytest.mark.asyncio
async def test_next_button_click_drives_quick_track_to_completion():
    """Regression test for a real Textual double-dispatch trap.

    Textual's @on-decorated handlers are collected across the WHOLE MRO
    (textual.message_pump.MessagePump._get_dispatch_methods), so both
    WizardContainer.handle_next (base) and SetupWizardContainer.handle_next
    (override) fire on a single Button.Pressed("#wizard-next"). Without
    event.prevent_default() in the override, the base handler flat-advances
    current_step by one BEFORE the override's own worker runs — silently
    breaking track selection (select_track() on the Welcome step never
    actually applies) and skipping/duplicating steps. This test drives the
    real click path (not container.select_track() directly) so a regression
    of that suppression would fail it.
    """
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        await pilot.click("#setup-track-quick")
        await pilot.pause(0.1)

        seen_step_ids = []
        for _ in range(10):
            if app.wizard_result != "UNSET":
                break
            await pilot.click("#wizard-next")
            await pilot.pause(0.2)
            step = container.steps[container.current_step]
            seen_step_ids.append(step.config.id if step.config else None)

        assert app.wizard_result == {"completed": True, "exit_route": None}
        # Exactly the quick-track subset, each step visited once, in order.
        assert seen_step_ids == ["provider", "model", "summary", "summary"]
        assert set(container.wizard_data.keys()) == {
            "welcome",
            "provider",
            "model",
            "summary",
        }
