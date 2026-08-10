"""Focused tests for the asynchronous, queue-aware application quit guard."""

from __future__ import annotations

import asyncio
import threading

import pytest

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.console_chat_models import ConsoleLifecycleImpact
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController


class _DispatchHarness:
    action_quit = TldwCli.action_quit

    def __init__(self, *, fail_start: bool = False) -> None:
        self._quit_in_progress = False
        self.fail_start = fail_start
        self.work: list[tuple[object, dict]] = []

    async def _confirm_and_quit(self) -> None:
        return None

    def run_worker(self, awaitable, **kwargs):
        if self.fail_start:
            raise RuntimeError("worker unavailable")
        self.work.append((awaitable, kwargs))
        return object()


def test_action_quit_dispatches_one_exclusive_worker_for_repeated_requests():
    app = _DispatchHarness()

    app.action_quit()
    app.action_quit()

    assert app._quit_in_progress is True
    assert len(app.work) == 1
    awaitable, kwargs = app.work[0]
    assert kwargs == {
        "group": "application-quit",
        "exclusive": True,
        "exit_on_error": False,
    }
    awaitable.close()


def test_action_quit_clears_guard_when_worker_cannot_start():
    app = _DispatchHarness(fail_start=True)

    app.action_quit()

    assert app._quit_in_progress is False
    assert app.work == []


class _ConfirmationScreen:
    def __init__(self, *, decision=True, error: Exception | None = None) -> None:
        self.decision = decision
        self.error = error
        self.calls: list[str] = []

    async def confirm_quit(self):
        self.calls.append("confirm")
        if self.error is not None:
            raise self.error
        return self.decision

    def prepare_for_quit(self) -> None:
        self.calls.append("prepare")


class _ConfirmationHarness:
    _confirm_and_quit = TldwCli._confirm_and_quit

    def __init__(self, screen: _ConfirmationScreen) -> None:
        self.screen = screen
        self._quit_in_progress = True
        self._shutting_down = False
        self.cleanup_calls = 0
        self.notifications: list[tuple[str, str]] = []

    async def _run_approved_quit_cleanup(self) -> None:
        self.cleanup_calls += 1

    def notify(self, message: str, *, severity: str) -> None:
        self.notifications.append((message, severity))


@pytest.mark.asyncio
async def test_quit_stay_preserves_screen_state_and_clears_reentrancy_guard():
    screen = _ConfirmationScreen(decision=False)
    app = _ConfirmationHarness(screen)

    await app._confirm_and_quit()

    assert screen.calls == ["confirm"]
    assert app._quit_in_progress is False
    assert app._shutting_down is False
    assert app.cleanup_calls == 0


@pytest.mark.asyncio
async def test_quit_confirmation_error_fails_closed_and_preserves_state():
    screen = _ConfirmationScreen(error=RuntimeError("dialog failed"))
    app = _ConfirmationHarness(screen)

    await app._confirm_and_quit()

    assert screen.calls == ["confirm"]
    assert app._quit_in_progress is False
    assert app._shutting_down is False
    assert app.cleanup_calls == 0
    assert app.notifications == [
        ("Couldn't confirm quitting; staying in Chatbook.", "warning")
    ]


class _Timer:
    def __init__(self, events: list[tuple[str, int]]) -> None:
        self.events = events

    def stop(self) -> None:
        self.events.append(("timer", threading.get_ident()))


class _ApprovedQuitHarness:
    _confirm_and_quit = TldwCli._confirm_and_quit
    _run_approved_quit_cleanup = TldwCli._run_approved_quit_cleanup

    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []
        self.loop_thread = threading.get_ident()
        self._quit_in_progress = True
        self._shutting_down = False
        self._media_cleanup_timer = _Timer(self.events)
        self.screen = self

    async def confirm_quit(self) -> bool:
        self.events.append(("confirm", threading.get_ident()))
        return True

    def prepare_for_quit(self) -> None:
        self.events.append(("prepare", threading.get_ident()))

    async def _cleanup_audio_for_quit(self) -> None:
        self.events.append(("audio", threading.get_ident()))

    def _run_blocking_quit_persistence(self) -> None:
        self.events.append(("persistence", threading.get_ident()))

    def exit(self) -> None:
        self.events.append(("exit", threading.get_ident()))


@pytest.mark.asyncio
async def test_approved_quit_tombstones_then_cleans_once_without_blocking_loop():
    app = _ApprovedQuitHarness()

    await app._confirm_and_quit()

    assert [name for name, _thread in app.events] == [
        "confirm",
        "prepare",
        "audio",
        "timer",
        "persistence",
        "exit",
    ]
    event_threads = dict(app.events)
    assert event_threads["prepare"] == app.loop_thread
    assert event_threads["audio"] == app.loop_thread
    assert event_threads["timer"] == app.loop_thread
    assert event_threads["persistence"] != app.loop_thread
    assert event_threads["exit"] == app.loop_thread
    assert app._shutting_down is True
    assert app._quit_in_progress is True


@pytest.mark.asyncio
async def test_approved_quit_still_exits_when_background_persistence_raises():
    app = _ApprovedQuitHarness()

    def _fail_persistence() -> None:
        raise RuntimeError("disk unavailable")

    app._run_blocking_quit_persistence = _fail_persistence
    await app._run_approved_quit_cleanup()

    assert [name for name, _thread in app.events] == ["audio", "timer", "exit"]


@pytest.mark.asyncio
async def test_blocking_quit_persistence_does_not_stall_the_app_loop():
    app = _ApprovedQuitHarness()
    persistence_started = threading.Event()
    persistence_release = threading.Event()
    loop_progressed = False

    def _block_persistence() -> None:
        persistence_started.set()
        persistence_release.wait(timeout=5)

    async def _observe_loop_progress() -> None:
        nonlocal loop_progressed
        while not persistence_started.is_set():
            await asyncio.sleep(0)
        loop_progressed = True
        persistence_release.set()

    app._run_blocking_quit_persistence = _block_persistence
    await asyncio.gather(
        app._run_approved_quit_cleanup(),
        _observe_loop_progress(),
    )

    assert loop_progressed is True
    assert app.events[-1][0] == "exit"


class _ImpactSequenceController:
    def __init__(self, impacts: list[ConsoleLifecycleImpact]) -> None:
        self.impacts = impacts
        self.calls = 0

    def lifecycle_impact(self) -> ConsoleLifecycleImpact:
        impact = self.impacts[min(self.calls, len(self.impacts) - 1)]
        self.calls += 1
        return impact


class _FleetConfirmationHarness:
    _confirm_fleet_loss = ConsoleSessionController._confirm_fleet_loss

    def __init__(self, decisions: list[bool]) -> None:
        self.app_instance = self
        self.decisions = decisions
        self.dialogs = []
        self.notifications: list[tuple[str, str]] = []

    async def _await_confirmation(self, dialog) -> bool:
        self.dialogs.append(dialog)
        return self.decisions.pop(0)

    def notify(self, message: str, *, severity: str) -> None:
        self.notifications.append((message, severity))


@pytest.mark.asyncio
async def test_changed_fleet_impact_requires_updated_confirmation():
    first = ConsoleLifecycleImpact(1, 1, 0, 0)
    updated = ConsoleLifecycleImpact(2, 0, 1, 3)
    controller = _ImpactSequenceController([first, updated, updated, updated])
    harness = _FleetConfirmationHarness([True, True])

    assert await harness._confirm_fleet_loss(controller, quitting=True) is True

    assert len(harness.dialogs) == 2
    assert "Live agent runs: 1" in harness.dialogs[0].message
    assert "Sessions with queued prompts: 1" in harness.dialogs[1].message
    assert "Unsent queued prompts: 3" in harness.dialogs[1].message
    assert harness.notifications == [
        ("Console activity changed; review the updated impact.", "warning")
    ]


@pytest.mark.asyncio
async def test_fleet_confirmation_stay_preserves_the_observed_impact():
    impact = ConsoleLifecycleImpact(7, 0, 1, 2)
    controller = _ImpactSequenceController([impact])
    harness = _FleetConfirmationHarness([False])

    assert await harness._confirm_fleet_loss(controller, quitting=False) is False

    assert controller.calls == 1
    assert len(harness.dialogs) == 1
    assert "Live agent runs: 0" in harness.dialogs[0].message
    assert "Sessions with queued prompts: 1" in harness.dialogs[0].message
    assert "Unsent queued prompts: 2" in harness.dialogs[0].message
