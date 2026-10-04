"""A ``/generate-video`` send never parks the pump that delivered it.

TASK-33622.16. Found live during TASK-33622.15 and reproduced on dev: after
Enter sent ``/generate-video`` to a paid backend, the "Generate video?" cost
confirm ignored Escape, F1 and Ctrl+Q, and so did the storage choice after it,
while a mouse click on Send left them working.

The mechanism these tests pin (each fails on dev for it, on a bounded poll):

* Enter schedules the visible send with ``app.call_later``. Textual's
  ``MessagePump.on_callback`` AWAITS a coroutine callback on the pump it was
  posted to -- the APP pump, which reads every key and click. A slash command
  ran inline in that send (``_dispatch_console_command``), and
  ``/generate-video`` awaits its cost confirm, the whole paid generation and
  the storage choice. The app pump was parked under all of it, so the confirm
  could never receive the Escape that would have ended it.
* The Send button awaits the same send on the Console's own pump. The app
  pump stayed free (why a click "worked"), but the Console pump was parked for
  the whole generation -- and the Stop button that cancels a paid run is
  handled there.

Input after the send is delivered the way the terminal driver does (``_key``,
shared with TASK-33621.28's ``test_console_hook_review_send_freeze``), never
through Pilot, whose idle wait never returns while a pump is parked. Every
test releases what it opened in a ``finally`` so a red run fails instead of
hanging the suite.

The paid backend is a stand-in at the Console's generation seam
(``run_video_generation``): these tests are about which pump waits, not about
MiniMax. The live run with the real backend is recorded in the task.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import threading
from types import SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_hook_review_send_freeze import (
    _key,
    _pump_runs,
    _task_factory,
    _until,
)
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from tldw_chatbook.Chat.console_generate_video import PendingVideoArtifact
from tldw_chatbook.UI.Console_Modules import video as video_module
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel
from tldw_chatbook.Video_Generation import adapter_registry
from tldw_chatbook.Video_Generation.exceptions import VideoGenerationError
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Widgets.cancel_confirmation_dialog import CancelConfirmationDialog
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Console.console_video_capacity_modal import (
    ConsoleVideoCapacityModal,
)

pytestmark = pytest.mark.bootstrap_profile

DRAFT = "/generate-video a paper boat drifting on a pond"
COST_CONFIRM = "Generate video?"
ROUTES = ["enter", "send-button"]


class _PaidBackend:
    """Stand-in for a paid backend at the Console's generation seam.

    Blocks like a real remote poll until released, and honours the
    cooperative cancel event the way the MiniMax adapter does.
    """

    def __init__(self, outcome=None) -> None:
        self.calls: list[dict] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self._outcome = outcome

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        cancel = kwargs.get("cancel_event")
        while not self.release.wait(0.02):
            if cancel is not None and cancel.is_set():
                self.cancelled.set()
                raise VideoGenerationError("cancelled by user")
        if self._outcome is None:
            raise VideoGenerationError("released by the test")
        return self._outcome(kwargs)


class _Registry:
    @staticmethod
    def resolve_backend(backend):
        return backend


def _pending_artifact(kwargs) -> PendingVideoArtifact:
    stream = tempfile.TemporaryFile(mode="w+b")
    stream.write(b"generated-video")
    stream.seek(0)
    return PendingVideoArtifact(
        metadata=VideoGenerationMetadata(
            name="paper-boat", prompt="a paper boat", backend="minimax"
        ),
        message_id=kwargs["message_id"],
        slug="paper-boat",
        extension="mp4",
        size_bytes=300 * 1024 * 1024,
        max_bytes=256 * 1024 * 1024,
        reason="over_capacity",
        stream=stream,
    )


def _configure_paid_backend(monkeypatch, console, backend: _PaidBackend) -> None:
    monkeypatch.setattr(
        video_module,
        "get_video_generation_config",
        lambda: SimpleNamespace(default_backend="minimax", confirm_cost_estimate=True),
    )
    monkeypatch.setattr(adapter_registry, "get_registry", lambda: _Registry())
    monkeypatch.setattr(video_module, "run_video_generation", backend)
    console._video._console_video_store = object()


def _send(route: str, app, console) -> None:
    if route == "send-button":
        console.query_one("#console-send-message", Button).press()
    else:
        _key(app, "enter", "\r")


async def _load_draft(app, console, pilot) -> None:
    composer = console._console_composer_or_none()
    composer.load_draft(DRAFT)
    composer.focus()
    await pilot.pause()
    assert composer.draft_text() == DRAFT


def _cost_confirm(app):
    screen = app.screen
    if (
        isinstance(screen, CancelConfirmationDialog)
        and screen.is_mounted
        and screen.title == COST_CONFIRM
    ):
        return screen
    return None


async def _wait_for_cost_confirm(app, route: str):
    assert await _until(lambda: _cost_confirm(app) is not None, 10), (
        f"{route}: the cost confirm never opened"
    )
    await asyncio.sleep(0.1)
    return _cost_confirm(app)


async def _assert_pumps_run(app, console, when: str) -> None:
    assert await _pump_runs(app), (
        f"{when}: the app pump -- the one that reads every key, Escape, F1 "
        "and Ctrl+Q included -- is parked"
    )
    assert await _pump_runs(console), f"{when}: the Console message pump is parked"


async def _release(app, backend: _PaidBackend) -> None:
    """Let a red run tear down instead of hanging: release the stand-in
    backend's thread, then answer, top down, every prompt nothing reached --
    each confirm as its safe "no", the storage choice as Discard. Their
    results are futures the waiting worker holds, so no parked pump has to
    flush them, and the pump the send parked comes free."""
    backend.release.set()
    for _ in range(6):
        await asyncio.sleep(0.2)
        top = app.screen
        if isinstance(top, ConfirmationDialog):
            top.dismiss(False)
        elif isinstance(top, ConsoleVideoCapacityModal):
            top.dismiss("discard")
        else:
            break
    await asyncio.sleep(0.3)


def _video_idle(console) -> bool:
    return not console._video._console_videogen_inflight_sessions()


def _dialogs_titled(app, title: str) -> list[ConfirmationDialog]:
    return [
        screen
        for screen in app.screen_stack
        if isinstance(screen, ConfirmationDialog) and screen.title == title
    ]


@contextlib.asynccontextmanager
async def _console_sending_video(monkeypatch, route: str, backend: _PaidBackend):
    """Run the real ``TldwCli`` on its Console, send ``DRAFT`` by ``route``
    and yield ``(app, console, cleanups)``.

    ``ChatScreen`` pushes the cost confirm on its ``app_instance``, which in
    a ``ConsoleHarness`` is a TldwCli that is not running -- the confirm never
    shows there -- so every test here drives the real app. Its irreversible
    shutdown is replaced by a recorder, as the other quit tests do.
    """
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups: list[bool] = []

    async def _record_cleanup() -> None:
        cleanups.append(True)

    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _record_cleanup)
    async with app.run_test(size=(140, 44)) as pilot:
        assert await _until(
            lambda: (
                type(app.screen).__name__ == "ChatScreen"
                and bool(app.screen.query("#console-native-composer"))
            ),
            15,
        ), "the Console composer never mounted"
        await pilot.pause(0.2)
        console = app.screen
        _configure_paid_backend(monkeypatch, console, backend)
        await _load_draft(app, console, pilot)
        _send(route, app, console)
        try:
            yield app, console, cleanups
        finally:
            await _release(app, backend)


async def _confirm_generate(app, route: str) -> None:
    """Answer the cost confirm with a synthetic click its own pump handles,
    so a test of a later leg reaches it even on dev, where the keyboard could
    not answer the confirm at all."""
    confirm = await _wait_for_cost_confirm(app, route)
    confirm.query_one("#confirm-button", Button).press()


async def _wait_for_storage_choice(app) -> ConsoleVideoCapacityModal:
    assert await _until(
        lambda: (
            isinstance(app.screen, ConsoleVideoCapacityModal) and app.screen.is_mounted
        ),
        10,
    ), "the storage choice never opened"
    await asyncio.sleep(0.1)
    return app.screen


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("factory", ["lazy", "eager"])
async def test_the_cost_confirm_answers_its_keys_and_escape_starts_nothing(
    route, factory, monkeypatch
):
    """AC#1/#3: while "Generate video?" is up, both pumps run, F1 is consumed
    (the modal owns the keyboard, so it opens nothing over it), and Escape
    cancels with no generation started and the draft kept. ``eager`` runs it
    under the task factory the real app installs, where the hand-off's worker
    starts inside the send itself (see ``_task_factory``)."""
    backend = _PaidBackend()
    with _task_factory(factory):
        await _escape_at_the_cost_confirm(route, backend, monkeypatch)


async def _escape_at_the_cost_confirm(route, backend, monkeypatch) -> None:
    async with _console_sending_video(monkeypatch, route, backend) as (
        app,
        console,
        _cleanups,
    ):
        confirm = await _wait_for_cost_confirm(app, route)
        await _assert_pumps_run(app, console, f"{route}, cost confirm open")
        _key(app, "f1")
        await _assert_pumps_run(app, console, f"{route}, F1 under the confirm")
        assert app.screen is confirm, "F1 covered the cost confirm"
        _key(app, "escape")
        assert await _until(lambda: app.screen is console, 5), (
            f"{route}: Escape never reached the cost confirm"
        )
        await _assert_pumps_run(app, console, f"{route}, after Escape")
        assert backend.calls == [], "Escape at the cost confirm started a generation"
        composer = console._console_composer_or_none()
        assert composer.draft_text() == DRAFT, "a cancelled confirm lost the draft"
        assert _video_idle(console)


@pytest.mark.parametrize("route", ROUTES)
async def test_ctrl_q_under_the_cost_confirm_reaches_the_quit_flow(route, monkeypatch):
    """AC#1: Ctrl+Q under the cost confirm starts the app's own quit flow.
    Nothing has been generated, so quitting there loses no video: the flow
    may ask about other Console work ("Quit Chatbook?") or run straight
    through to the (recorded) shutdown -- either proves it was answered."""
    backend = _PaidBackend()
    async with _console_sending_video(monkeypatch, route, backend) as (
        app,
        _console,
        cleanups,
    ):
        await _wait_for_cost_confirm(app, route)
        _key(app, "ctrl+q")
        assert await _until(
            lambda: bool(cleanups) or bool(_dialogs_titled(app, "Quit Chatbook?")),
            10,
        ), f"{route}: Ctrl+Q under the cost confirm never reached the quit flow"
        assert backend.calls == []


@pytest.mark.parametrize("route", ROUTES)
async def test_a_paid_generation_in_flight_answers_keys_help_and_stop(
    route, monkeypatch
):
    """AC#2/#3: once the generation runs, both pumps keep running: Escape is
    consumed, F1 opens and closes help, and the Stop button reaches the
    backend's cancel event (dev parked the Console pump that handles it, on
    the Send route too)."""
    backend = _PaidBackend()
    async with _console_sending_video(monkeypatch, route, backend) as (
        app,
        console,
        _cleanups,
    ):
        await _confirm_generate(app, route)
        assert await _until(backend.started.is_set, 10), (
            f"{route}: Generate never started the generation"
        )
        await _assert_pumps_run(app, console, f"{route}, generation in flight")
        _key(app, "escape")
        await _assert_pumps_run(app, console, f"{route}, Escape in flight")
        _key(app, "f1")
        assert await _until(lambda: isinstance(app.screen, WorkbenchHelpPanel), 5), (
            f"{route}: F1 never opened help during the generation"
        )
        _key(app, "f1")
        assert await _until(lambda: app.screen is console, 5), (
            f"{route}: F1 never closed help again"
        )
        stop = console.query_one("#console-stop-generation", Button)
        assert await _until(lambda: stop.display and not stop.disabled, 5), (
            f"{route}: Stop is not offered during the generation"
        )
        stop.press()
        assert await _until(backend.cancelled.is_set, 5), (
            f"{route}: Stop never reached the paid backend's cancel event"
        )
        assert await _until(lambda: _video_idle(console), 5), (
            f"{route}: the stopped generation never settled"
        )
        await _assert_pumps_run(app, console, f"{route}, after Stop")


async def test_ctrl_q_during_a_paid_generation_reaches_the_quit_flow(monkeypatch):
    """AC#2: Ctrl+Q while an Enter-sent generation runs reaches the quit
    flow. No video exists yet, so there is nothing to ask about here."""
    backend = _PaidBackend()
    async with _console_sending_video(monkeypatch, "enter", backend) as (
        app,
        _console,
        cleanups,
    ):
        await _confirm_generate(app, "enter")
        assert await _until(backend.started.is_set, 10)
        _key(app, "ctrl+q")
        assert await _until(
            lambda: bool(cleanups) or bool(_dialogs_titled(app, "Quit Chatbook?")),
            10,
        ), "Ctrl+Q during the Enter-sent generation never reached the quit flow"


async def test_the_storage_choice_after_an_enter_send_answers_its_keys(monkeypatch):
    """AC#2: the storage choice an Enter send leads to answers F1 (consumed)
    and Escape (its own "Discard generated video?" guard), and Discard
    settles the send."""
    backend = _PaidBackend(outcome=_pending_artifact)
    backend.release.set()
    async with _console_sending_video(monkeypatch, "enter", backend) as (
        app,
        console,
        _cleanups,
    ):
        await _confirm_generate(app, "enter")
        choice = await _wait_for_storage_choice(app)
        await _assert_pumps_run(app, console, "storage choice open")
        _key(app, "f1")
        await _assert_pumps_run(app, console, "F1 under the storage choice")
        assert app.screen is choice
        _key(app, "escape")
        assert await _until(
            lambda: bool(_dialogs_titled(app, "Discard generated video?")), 5
        ), "Escape never reached the storage choice"
        _dialogs_titled(app, "Discard generated video?")[0].query_one(
            "#confirm-button", Button
        ).press()
        assert await _until(lambda: app.screen is console, 5)
        assert await _until(lambda: _video_idle(console), 5), (
            "the discarded video's send never settled"
        )
        await _assert_pumps_run(app, console, "after Discard")


async def test_ctrl_q_under_an_enter_sent_storage_choice_asks_before_discarding(
    monkeypatch,
):
    """AC#2: Ctrl+Q over the storage choice an Enter send led to still asks
    "Discard generated video and quit?"; Stay returns to the choice with the
    video's fate undecided."""
    backend = _PaidBackend(outcome=_pending_artifact)
    backend.release.set()
    async with _console_sending_video(monkeypatch, "enter", backend) as (
        app,
        console,
        cleanups,
    ):
        await _confirm_generate(app, "enter")
        choice = await _wait_for_storage_choice(app)
        _key(app, "ctrl+q")
        assert await _until(
            lambda: (
                bool(_dialogs_titled(app, "Discard generated video and quit?"))
                or bool(cleanups)
            ),
            10,
        ), "Ctrl+Q over the Enter-sent storage choice was never answered"
        assert cleanups == [], "quit straight past the generated video"
        _key(app, "escape")
        assert await _until(lambda: app.screen is choice, 5), (
            "Stay never returned to the storage choice"
        )
        assert await _until(lambda: app._quit_in_progress is False, 5)
        assert cleanups == []
        assert console._video._pending_console_video_artifacts(), (
            "Stay decided the video's fate"
        )


# -- the hand-off's own contract ---------------------------------------------


async def test_a_repeat_press_of_the_same_draft_opens_one_cost_confirm(monkeypatch):
    """The hand-off lets a second press arrive while the first command runs
    (a parked pump used to queue it until the draft was gone). A fast second
    click on Send for the same unmodified draft must not stack a second paid
    confirm."""
    backend = _PaidBackend()
    async with _console_sending_video(monkeypatch, "enter", backend) as (
        app,
        console,
        _cleanups,
    ):
        confirm = await _wait_for_cost_confirm(app, "enter")
        _send("send-button", app, console)
        await asyncio.sleep(0.5)
        assert _dialogs_titled(app, COST_CONFIRM) == [confirm], (
            "a repeat press of the same draft stacked a second cost confirm"
        )
        _key(app, "escape")
        assert await _until(lambda: app.screen is console, 5)
        assert backend.calls == []


async def test_another_command_runs_while_a_generation_is_in_flight(monkeypatch):
    """Only a repeat of the same draft waits: a different command sent while
    the generation runs runs at once -- ``/help`` leaves the generation going,
    and ``/stop`` (sent with the Send button: a bare ``/stop`` has the command
    popup open, where Enter accepts the completion) stops it. On dev both
    queued behind the parked pump until the generation had resolved."""
    backend = _PaidBackend()
    async with _console_sending_video(monkeypatch, "enter", backend) as (
        app,
        console,
        _cleanups,
    ):
        helped: list[str] = []

        async def _record_help(parse) -> None:
            helped.append(parse.name)

        monkeypatch.setattr(console, "_console_command_help", _record_help)
        await _confirm_generate(app, "enter")
        assert await _until(backend.started.is_set, 10)
        composer = console._console_composer_or_none()
        assert await _until(lambda: composer.draft_text() == "", 5), (
            "the generation never took its draft"
        )
        # With an argument: a bare "/help" has the command popup open, and
        # Enter there accepts the completion instead of sending.
        composer.load_draft("/help generate-video")
        composer.focus()
        await asyncio.sleep(0.2)
        _key(app, "enter", "\r")
        assert await _until(lambda: helped == ["help"], 5), (
            "/help never ran while the generation was in flight"
        )
        assert not backend.cancelled.is_set() and not _video_idle(console)
        composer.load_draft("/stop")
        await asyncio.sleep(0.2)
        _send("send-button", app, console)
        assert await _until(backend.cancelled.is_set, 5), (
            "/stop never reached the paid backend's cancel event"
        )
        assert await _until(lambda: _video_idle(console), 5)


async def test_a_command_whose_modal_wait_is_cancelled_ends_quietly(monkeypatch):
    """Live finding: Ctrl+Q under the cost confirm quit the app, and the
    shutdown cancelled the worker holding the confirm, so ``Worker.wait``
    raised ``WorkerCancelled`` through the command into its own worker -- an
    "unhandled_exception" on the way out. The command is over, not broken:
    it ends quietly, starts nothing, keeps the draft and frees its repeat
    key. Cancelling that one waiting worker reproduces it without quitting."""
    from textual.worker import WorkerState

    backend = _PaidBackend()
    async with _console_sending_video(monkeypatch, "enter", backend) as (
        app,
        console,
        _cleanups,
    ):
        await _wait_for_cost_confirm(app, "enter")
        waits = [
            worker
            for worker in console.workers
            if worker.state == WorkerState.RUNNING
            and "push_screen_wait" in getattr(worker._work, "__qualname__", "")
        ]
        assert len(waits) == 1, f"expected the confirm's one waiting worker: {waits}"
        waits[0].cancel()
        assert await _until(lambda: _video_idle(console) and app.screen is not None, 5)
        await asyncio.sleep(0.5)
        assert app._exception is None, f"the command surfaced {app._exception!r}"
        assert app.is_running
        assert backend.calls == []
        composer = console._console_composer_or_none()
        assert composer.draft_text() == DRAFT
        # The repeat key is free again: the same draft opens a fresh confirm.
        for screen in _dialogs_titled(app, COST_CONFIRM):
            screen.dismiss(False)
        await asyncio.sleep(0.3)
        _send("send-button", app, console)
        assert await _until(lambda: len(_dialogs_titled(app, COST_CONFIRM)) == 1, 5)
