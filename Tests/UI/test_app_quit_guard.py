"""Focused tests for the asynchronous, queue-aware application quit guard."""

from __future__ import annotations

import asyncio
import threading

import pytest

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleLifecycleImpact,
    ConsoleLifecycleRevisionChanged,
)
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
    _confirm_console_runtime_quit = TldwCli._confirm_console_runtime_quit

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

    def _close_boot_worker_gate(self, _reason: str) -> None:
        return None


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
    _confirm_console_runtime_quit = TldwCli._confirm_console_runtime_quit
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

    def _close_boot_worker_gate(self, _reason: str) -> None:
        return None


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


class _QuitRuntime:
    def __init__(
        self,
        controller: _ImpactSequenceController,
        events: list[str],
        *,
        fail_fence: bool = False,
    ) -> None:
        self.chat_controller = controller
        self.events = events
        self.disposed = False
        self.fail_fence = fail_fence
        self.voice_promotion_owner = _QuitOwner()

    def begin_dispose(
        self,
        *,
        expected_revision: int | None = None,
        voice_promotion_permit=None,
    ) -> None:
        assert voice_promotion_permit is self.voice_promotion_owner.permit
        if self.fail_fence:
            raise RuntimeError("fence unavailable")
        if (
            expected_revision is not None
            and self.chat_controller.lifecycle_impact().revision != expected_revision
        ):
            raise ConsoleLifecycleRevisionChanged(
                "Console activity changed during shutdown."
            )
        self.disposed = True
        self.events.append("fence")


class _QuitOwner:
    def __init__(self) -> None:
        self.token = object()
        self.permit = object()
        self.calls: list[tuple[str, object | None]] = []

    def begin_quit(self):
        self.calls.append(("begin", None))
        return self.token

    async def wait_for_quiescence(self, token, timeout: float) -> bool:
        assert token is self.token
        assert timeout > 0
        self.calls.append(("wait", token))
        return True

    def seal_quiescent(self, token):
        assert token is self.token
        self.calls.append(("seal", token))
        return self.permit

    def abort_quit(self, token_or_permit) -> None:
        self.calls.append(("abort", token_or_permit))


class _AppLevelQuitHarness:
    _confirm_and_quit = TldwCli._confirm_and_quit
    _confirm_console_runtime_quit = TldwCli._confirm_console_runtime_quit

    def __init__(
        self,
        screen: _ConfirmationScreen,
        impacts: list[ConsoleLifecycleImpact],
        decisions: list[bool],
        *,
        fail_fence: bool = False,
    ) -> None:
        self.screen = screen
        self._quit_in_progress = True
        self._shutting_down = False
        self.events: list[str] = []
        self.dialogs = []
        self.decisions = decisions
        self.notifications: list[tuple[str, str]] = []
        self.console_runtime = _QuitRuntime(
            _ImpactSequenceController(impacts),
            self.events,
            fail_fence=fail_fence,
        )

    async def _await_console_quit_confirmation(self, dialog) -> bool:
        self.dialogs.append(dialog)
        return self.decisions.pop(0)

    async def _run_approved_quit_cleanup(self) -> None:
        assert self.console_runtime.disposed is True
        self.events.append("cleanup")

    def notify(self, message: str, *, severity: str) -> None:
        self.notifications.append((message, severity))

    def _close_boot_worker_gate(self, _reason: str) -> None:
        return None


@pytest.mark.asyncio
async def test_non_console_quit_confirms_console_once_then_fences_before_cleanup():
    screen = _ConfirmationScreen()
    impact = ConsoleLifecycleImpact(8, 1, 0, 0, 2)
    app = _AppLevelQuitHarness(screen, [impact, impact], [True])

    await app._confirm_and_quit()

    assert screen.calls == ["confirm", "prepare"]
    assert len(app.dialogs) == 1
    assert "Live agent runs: 1" in app.dialogs[0].message
    assert "Delegated agents: 2" in app.dialogs[0].message
    assert app.events == ["fence", "cleanup"]
    assert app._shutting_down is True
    assert app.notifications == []
    assert app.console_runtime.voice_promotion_owner.calls == [
        ("begin", None),
        ("wait", app.console_runtime.voice_promotion_owner.token),
        ("seal", app.console_runtime.voice_promotion_owner.token),
    ]


@pytest.mark.asyncio
async def test_prepare_for_quit_failure_exact_token_aborts_promotion_fence():
    class _FailingPrepareScreen(_ConfirmationScreen):
        def prepare_for_quit(self) -> None:
            self.calls.append("prepare")
            raise RuntimeError("preparation failed")

    impact = ConsoleLifecycleImpact(1, 0, 0, 0, 0)
    app = _AppLevelQuitHarness(_FailingPrepareScreen(), [impact], [])

    await app._confirm_and_quit()

    owner = app.console_runtime.voice_promotion_owner
    assert owner.calls == [
        ("begin", None),
        ("wait", owner.token),
        ("seal", owner.token),
        ("abort", owner.permit),
    ]
    assert app.console_runtime.disposed is False


@pytest.mark.asyncio
async def test_console_fence_failure_stays_in_app_without_starting_cleanup():
    impact = ConsoleLifecycleImpact(8, 1, 0, 0, 0)
    app = _AppLevelQuitHarness(
        _ConfirmationScreen(),
        [impact, impact],
        [True],
        fail_fence=True,
    )

    await app._confirm_and_quit()

    assert app.events == []
    assert app._quit_in_progress is False
    assert app._shutting_down is False
    assert app.notifications == [
        ("Couldn't prepare a safe shutdown; staying in Chatbook.", "warning")
    ]
    owner = app.console_runtime.voice_promotion_owner
    assert owner.calls[-1] == ("abort", owner.permit)


@pytest.mark.asyncio
async def test_quit_worker_cancellation_exact_token_aborts_promotion_fence():
    class _BlockingPrepareScreen(_ConfirmationScreen):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def prepare_for_quit(self) -> None:
            self.calls.append("prepare")
            self.started.set()
            await self.release.wait()

    screen = _BlockingPrepareScreen()
    impact = ConsoleLifecycleImpact(1, 0, 0, 0, 0)
    app = _AppLevelQuitHarness(screen, [impact], [])
    worker = asyncio.create_task(app._confirm_and_quit())
    await screen.started.wait()

    worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await worker

    owner = app.console_runtime.voice_promotion_owner
    assert owner.calls[-1] == ("abort", owner.permit)
    assert app.console_runtime.disposed is False
    assert app._quit_in_progress is False

    admitted: list[asyncio.Task] = []

    def run_worker(coroutine, **_kwargs):
        task = asyncio.create_task(coroutine)
        admitted.append(task)
        return task

    app.run_worker = run_worker
    screen.release.set()
    TldwCli.action_quit(app)

    assert len(admitted) == 1
    await admitted[0]
    assert app.console_runtime.disposed is True
    assert app._quit_in_progress is True


@pytest.mark.asyncio
async def test_app_quit_reconfirms_when_console_revision_changes():
    first = ConsoleLifecycleImpact(11, 1, 0, 0, 1)
    updated = ConsoleLifecycleImpact(12, 0, 1, 3, 2)
    app = _AppLevelQuitHarness(
        _ConfirmationScreen(),
        [first, updated, updated, updated],
        [True, True],
    )

    assert await app._confirm_console_runtime_quit() is True

    assert len(app.dialogs) == 2
    assert "Delegated agents: 1" in app.dialogs[0].message
    assert "Delegated agents: 2" in app.dialogs[1].message
    assert app.notifications == [
        ("Console activity changed; review the updated impact.", "warning")
    ]


@pytest.mark.asyncio
async def test_app_quit_reconfirms_when_revision_changes_at_the_fence():
    first = ConsoleLifecycleImpact(21, 1, 0, 0, 0)
    updated = ConsoleLifecycleImpact(22, 1, 0, 0, 1)
    app = _AppLevelQuitHarness(
        _ConfirmationScreen(),
        [first, first, updated, updated, updated],
        [True, True],
    )

    await app._confirm_and_quit()

    assert len(app.dialogs) == 2
    assert "Delegated agents: 0" in app.dialogs[0].message
    assert "Delegated agents: 1" in app.dialogs[1].message
    assert app.events == ["fence", "cleanup"]
    assert app.notifications == [
        ("Console activity changed; review the updated impact.", "warning")
    ]


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
