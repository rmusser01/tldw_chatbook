"""TASK-34100.1 AC#3: first-run background work never exits the app.

Every first-run worker starts through ``first_run_step_guard.run_wizard_worker``
(or ``@wizard_work``), which passes ``exit_on_error=False`` and reports an
escaped error on the pinned status strip. These tests run a real wizard in a
headless app that has NOT opted into the production keep-alive policy, which
is exactly where a worker started with Textual's default used to take the
whole app down.

The app stays up, but the test run still hears about it (review round 1):
in a headless run an escaped worker error is recorded in ``App._exception``,
the slot ``run_test`` re-raises from, as the old exit did. These tests expect
their error, so each takes it back with ``_take_recorded_error``.
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


def _take_recorded_error(app: App) -> BaseException | None:
    """Return the error recorded for the test run, and clear it.

    ``run_test`` re-raises ``App._exception`` when the app shuts down; a test
    that expects its error takes it back first.
    """
    recorded, app._exception = app._exception, None
    return recorded


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
        assert isinstance(_take_recorded_error(app), RuntimeError)
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
        assert isinstance(_take_recorded_error(app), ValueError)
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
        assert isinstance(_take_recorded_error(app), OSError)
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
        assert isinstance(_take_recorded_error(app), ValueError)
        assert worker.state is WorkerState.ERROR
        assert _strip(wizard).has_class("hidden")


@pytest.mark.asyncio
@pytest.mark.parametrize("kept_alive", [False, True])
async def test_a_headless_run_still_hears_about_a_worker_error(kept_alive: bool):
    """The first error is recorded where ``run_test`` re-raises it.

    Review round 1: with ``exit_on_error=False`` everywhere, a worker bug
    only logged and showed on the strip, so a test that never looked at the
    strip passed. Workers used to exit the app whatever its keep-alive
    choice, so the error is recorded with or without it.
    """
    from tldw_chatbook.UI.Wizards.first_run_step_guard import run_wizard_worker

    wizard = _wizard()
    app = _Host(wizard)
    app._keep_screen_alive_on_handler_error = kept_alive

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)

        async def _fails() -> None:
            raise KeyError("first")

        async def _fails_again() -> None:
            raise LookupError("second")

        run_wizard_worker(container, _fails(), group="test-signal-1")
        await _settle(pilot, app)
        run_wizard_worker(container, _fails_again(), group="test-signal-2")
        await _settle(pilot, app)

        assert app.is_running
        recorded = _take_recorded_error(app)
        assert isinstance(recorded, KeyError), "the first error is the one kept"


@pytest.mark.asyncio
async def test_a_kept_alive_run_contains_a_raising_next_without_recording_it(
    monkeypatch,
):
    """A Next that raises keeps the handler policy: kept alive means contained.

    Before TASK-34100.1 a raising Next re-raised only when the app did not
    keep its UI alive; the provider-catalog tests rely on that opt-in.
    """

    async def _boom(self):
        raise RuntimeError("commit exploded")

    monkeypatch.setattr(WelcomeStep, "commit", _boom)
    wizard = _wizard()
    app = _Host(wizard)
    app._keep_screen_alive_on_handler_error = True

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("ctrl+n")
        await _settle(pilot, app)

        assert app.is_running and app._exception is None
        assert "setup stayed here" in str(_strip(wizard).content)


def test_a_production_app_never_records_a_worker_error() -> None:
    """Only a headless (test) run records; a real app has nothing to re-raise."""
    from types import SimpleNamespace

    from tldw_chatbook.UI.Wizards import first_run_step_guard as guard

    app = SimpleNamespace(
        is_headless=False, _exception=None, _keep_screen_alive_on_handler_error=True
    )
    guard._record_for_test_run(SimpleNamespace(app=app), ValueError("x"))

    assert app._exception is None


@pytest.mark.asyncio
async def test_off_loop_work_runs_on_another_thread_and_returns_its_result():
    """``off_loop`` keeps blocking setup off the UI loop (review round 1)."""
    import threading

    from tldw_chatbook.UI.Wizards.first_run_step_guard import off_loop

    async def work(value: int) -> tuple[int, threading.Thread]:
        await asyncio.sleep(0)
        return value, threading.current_thread()

    async def fails() -> None:
        raise ValueError("off-loop failure")

    value, thread = await off_loop(work)(7)

    assert value == 7
    assert thread is not threading.current_thread()
    with pytest.raises(ValueError, match="off-loop failure"):
        await off_loop(fails)()


@pytest.mark.asyncio
async def test_cancelling_off_loop_work_cancels_it_on_its_thread():
    """Leaving Provider cancels the scan; its thread must not run on for long."""
    import threading

    from tldw_chatbook.UI.Wizards.first_run_step_guard import off_loop

    started = threading.Event()
    finished = threading.Event()

    async def slow() -> None:
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            finished.set()

    task = asyncio.ensure_future(off_loop(slow)())
    assert await asyncio.to_thread(started.wait, 5.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await asyncio.to_thread(finished.wait, 3.0), "the work kept running"


@pytest.mark.asyncio
async def test_a_localhost_scan_that_lands_after_its_list_is_gone_is_dropped(
    monkeypatch,
):
    """The scan's result is dropped once the step's widgets are gone.

    Review round 1: the scan now finishes on its own thread, so it can land
    while the app tears the step down (children first, the step itself
    later). Rendering into the missing detection list raised ``NoMatches``.
    """
    import threading

    release = threading.Event()

    async def slow_scan(*_args, **_kwargs):
        release.wait(5)  # blocks only the scan's own thread
        return ()

    monkeypatch.setattr(
        "tldw_chatbook.Chat.local_server_discovery.discover_local_servers", slow_scan
    )
    wizard = _wizard()
    app = _Host(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        provider_index = container._step_index_for_id("provider")
        container.show_step(provider_index)
        await pilot.pause(0.2)
        provider = container.steps[provider_index]
        assert provider._local_discovery_state == "in_progress"

        await provider.query_one("#setup-provider-detection-results").remove()
        release.set()
        for _ in range(60):
            if provider._local_discovery_state != "in_progress":
                break
            await pilot.pause(0.05)

        assert provider._local_discovery_state == "complete"
        assert app._exception is None, "a late scan result raised into a torn-down step"


def test_the_recovery_prompts_worker_never_exits_the_app() -> None:
    """Review round 2: Resume / Start over runs in app.py, not UI/Wizards/.

    It started with Textual's default ``exit_on_error=True``, so an error
    reading the draft or building the wizard quit the whole app at relaunch.
    """
    from tldw_chatbook.app import TldwCli

    fake = MagicMock()
    for result in ("resume", "start_over"):
        fake.run_worker.reset_mock()
        TldwCli._handle_first_run_recovery_result(fake, result)
        work = fake.run_worker.call_args.args[0]
        work.close()  # the coroutine is not run here
        assert fake.run_worker.call_args.kwargs["exit_on_error"] is False


@pytest.mark.asyncio
async def test_a_recovery_failure_after_its_save_reprompts_instead_of_raising(
    monkeypatch,
) -> None:
    from functools import partial
    from types import SimpleNamespace

    from tldw_chatbook.app import TldwCli

    class _Broken:
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("the wizard could not be built")

    monkeypatch.setattr(
        "tldw_chatbook.UI.Wizards.FirstRunSetupWizard.FirstRunSetupWizard", _Broken
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    fake = SimpleNamespace(app_config={"first_run": {}})
    fake._mirror_first_run_setup_mutation = MagicMock()
    fake._handle_first_run_wizard_result = MagicMock()
    fake.push_screen = MagicMock()
    fake.notify = MagicMock()
    fake._schedule_first_run_recovery_retry = MagicMock()
    fake._apply_first_run_recovery_result = partial(
        TldwCli._apply_first_run_recovery_result, fake
    )
    fake.run_worker = MagicMock()

    TldwCli._handle_first_run_recovery_result(fake, "start_over")
    await fake.run_worker.call_args.args[0]  # the worker's body, run here

    fake.push_screen.assert_not_called()
    assert fake.notify.call_args.kwargs.get("severity") == "error"
    fake._schedule_first_run_recovery_retry.assert_called_once()
