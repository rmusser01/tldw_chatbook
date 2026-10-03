"""TASK-34100.1 AC#3: first-run background work never exits the app.

Every first-run worker starts through ``first_run_step_guard.run_wizard_worker``
(or ``@wizard_work``), which passes ``exit_on_error=False`` and reports an
escaped error on the pinned status strip. These tests run a real wizard in a
headless app that has NOT opted into the production keep-alive policy, which
is exactly where a worker started with Textual's default used to take the
whole app down.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static
from textual.worker import WorkerState

from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    FirstRunSetupWizard,
    SetupWizardContainer,
    WelcomeStep,
)

pytestmark = pytest.mark.bootstrap_profile


class _Host(App):
    def __init__(self, wizard: FirstRunSetupWizard) -> None:
        super().__init__()
        self._wizard = wizard

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.push_screen(self._wizard)


def _wizard() -> FirstRunSetupWizard:
    app_instance = MagicMock()
    app_instance.app_config = {}
    return FirstRunSetupWizard(app_instance)


async def _settle(pilot, app) -> None:
    for _ in range(40):
        await pilot.pause(0.05)
        if not any(worker.is_running for worker in app.workers):
            return


def _strip(wizard: FirstRunSetupWizard) -> Static:
    return wizard.query_one("#setup-step-error-pinned", Static)


@pytest.mark.asyncio
async def test_a_next_whose_commit_raises_keeps_the_app_and_explains_it(
    monkeypatch,
):
    async def _boom(self):
        raise RuntimeError("commit exploded")

    monkeypatch.setattr(WelcomeStep, "commit", _boom)
    wizard = _wizard()
    app = _Host(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)

        await pilot.press("ctrl+n")
        await _settle(pilot, app)

        assert app.is_running, "a raising Next took the app down"
        assert isinstance(container.steps[container.current_step], WelcomeStep)
        strip = _strip(wizard)
        assert not strip.has_class("hidden")
        assert "setup stayed here" in str(strip.content)
        assert container._advancing is False
        assert wizard.query_one("#wizard-next", Button).disabled is False


@pytest.mark.asyncio
async def test_a_raising_async_wizard_worker_reports_on_the_pinned_strip():
    from tldw_chatbook.UI.Wizards.first_run_step_guard import run_wizard_worker

    wizard = _wizard()
    app = _Host(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)

        async def _fails() -> None:
            await asyncio.sleep(0)
            raise ValueError("worker exploded")

        worker = run_wizard_worker(container, _fails(), group="test-async")
        await _settle(pilot, app)

        assert app.is_running
        assert worker.state is WorkerState.ERROR
        strip = _strip(wizard)
        assert not strip.has_class("hidden")
        assert "Something went wrong" in str(strip.content)
        assert "worker exploded" not in str(strip.content)


@pytest.mark.asyncio
async def test_a_raising_thread_wizard_worker_reports_on_the_pinned_strip():
    from tldw_chatbook.UI.Wizards.first_run_step_guard import run_wizard_worker

    wizard = _wizard()
    app = _Host(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        step = container.steps[container.current_step]

        def _fails_in_a_thread() -> None:
            raise OSError("disk exploded")

        worker = run_wizard_worker(
            step, _fails_in_a_thread, group="test-thread", thread=True
        )
        await _settle(pilot, app)

        assert app.is_running
        assert worker.state is WorkerState.ERROR
        strip = _strip(wizard)
        assert not strip.has_class("hidden")
        assert "Something went wrong" in str(strip.content)


@pytest.mark.asyncio
async def test_a_hidden_steps_failing_worker_does_not_flag_the_step_on_screen():
    """The strip belongs to the step on screen; a background step only logs."""
    from tldw_chatbook.UI.Wizards.first_run_step_guard import run_wizard_worker

    wizard = _wizard()
    app = _Host(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        hidden = next(
            step
            for index, step in enumerate(container.steps)
            if index != container.current_step
        )

        async def _fails() -> None:
            raise ValueError("background step exploded")

        worker = run_wizard_worker(hidden, _fails(), group="test-hidden")
        await _settle(pilot, app)

        assert app.is_running
        assert worker.state is WorkerState.ERROR
        assert _strip(wizard).has_class("hidden")
