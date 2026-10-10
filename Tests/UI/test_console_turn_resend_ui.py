"""TASK-33661: the Resend action in the transcript and the real Console.

The transcript tests mount a bare ``ConsoleTranscript``; the pilot tests drive
the real ``ChatScreen`` through a failed stream and a refused send, then click
(or press ``r`` on) Resend. One pilot pins the transcript poll those sends rely
on, which once stopped before a slow runner's turn had started.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from textual.app import ComposeResult
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_native_chat_flow import (
    BlockedGateway,
    FailThenRecoverGateway,
    _build_console_send_test_app,
    _ReadyResolutionGateway,
    _select_llamacpp_console,
    _visible_text,
    _wait_for_text,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
    ConsoleReadinessConfigProjection,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_transcript import (
    _ACTION_TOOLTIPS,
    ConsoleTranscript,
)

# The pilot tests mount the real ChatScreen (lessons-testing-evidence.md).
pytestmark = pytest.mark.bootstrap_profile

USER = ConsoleMessageRole.USER
ASSISTANT = ConsoleMessageRole.ASSISTANT


def _broken_turn() -> list[ConsoleChatMessage]:
    return [
        ConsoleChatMessage(
            role=USER, content="hello", id="u1", persisted_message_id="p-u1"
        ),
        ConsoleChatMessage(
            role=ASSISTANT,
            content="",
            id="a1",
            status="failed",
            persisted_message_id="p-a1",
            assistant_generation_state="failed",
        ),
    ]


class _BrokenTurnHarness(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.pressed: list[str] = []

    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(_broken_turn())
        yield transcript

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.pressed.append(event.button.id or "")


@pytest.mark.asyncio
async def test_broken_last_user_row_offers_resend_in_the_regenerate_slot():
    app = _BrokenTurnHarness()

    async with app.run_test(size=(211, 44)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.select_message("u1")
        await _wait_for_selector(app, pilot, "#console-message-action-resend-u1")
        resend = app.query_one("#console-message-action-resend-u1", Button)
        labels = [
            str(button.label)
            for button in app.query_one("#console-message-actions-u1").query(Button)
        ]
        regenerate = app.query("#console-message-action-regenerate-u1")
        continue_ = app.query("#console-message-action-continue-u1")
        guide = transcript._action_guide(transcript._message_by_id("u1"))

    assert labels == ["Copy", "Edit", "Fork", "Resend", "More…"]
    assert len(" ".join(labels)) <= 48
    assert not resend.disabled
    assert resend.tooltip == _ACTION_TOOLTIPS["resend"]
    assert not regenerate and not continue_
    assert "r Resend" in guide


@pytest.mark.asyncio
async def test_resend_is_hidden_while_a_run_is_live():
    app = _BrokenTurnHarness()

    async with app.run_test(size=(211, 44)) as pilot:
        app.screen._current_console_run_status_value = lambda: "validating"
        transcript = app.query_one(ConsoleTranscript)
        transcript.select_message("u1")
        await _wait_for_selector(app, pilot, "#console-message-action-regenerate-u1")
        resend = app.query("#console-message-action-resend-u1")

    assert not resend


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message_id", "pressed"),
    [
        ("u1", "console-message-action-resend-u1"),
        ("a1", "console-message-action-retry-a1"),
    ],
)
async def test_r_presses_the_rows_resend_or_retry(message_id, pressed):
    app = _BrokenTurnHarness()

    async with app.run_test(size=(211, 44)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.focus()
        transcript.select_message(message_id)
        await _wait_for_selector(app, pilot, f"#{pressed}")
        await pilot.press("r")
        await pilot.pause(0.2)

    assert app.pressed == [pressed]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "persisted_id", "timer_starts"),
    [("failed", None, 0), ("complete", "p-u1", 1)],
)
async def test_resend_worker_leaves_a_refused_echo_sync_timer_to_the_send_path(
    monkeypatch, status, persisted_id, timer_starts
):
    """Live check, 2026-10-01: starting the poll before the normal send path's
    own awaits let a tick stop it while the session still read blocked, so
    the resent turn ran unpolled and the transcript froze on "Generating…".
    A controller re-run starts it, exactly like Retry."""
    from types import SimpleNamespace

    from Tests.UI.console_controller_stubs import stub_message_controller
    from tldw_chatbook.Chat import console_turn_resend as resend_module
    from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult

    starts: list[None] = []

    async def synced():
        return None

    async def resent(*_args, **_kwargs):
        return ConsoleSubmitResult(True, False)

    message = ConsoleChatMessage(
        role=USER,
        content="hello",
        id="u1",
        status=status,
        persisted_message_id=persisted_id,
    )
    controller = SimpleNamespace(store=SimpleNamespace(get_message=lambda _id: message))
    monkeypatch.setattr(resend_module, "resend_turn", resent)
    owner = stub_message_controller(
        SimpleNamespace(),
        app_instance=SimpleNamespace(notify=lambda *_a, **_k: None),
        start_console_transcript_sync_timer=lambda: starts.append(None),
        sync_native_console_chat_ui=synced,
    )

    await owner._resend_console_turn(controller, "u1")

    assert len(starts) == timer_starts


@pytest.mark.asyncio
@pytest.mark.parametrize("in_flight", [True, False])
async def test_a_second_resend_never_cancels_the_one_in_flight(in_flight):
    """Review I2: a second press during a refused echo's send-path await used
    to start an exclusive worker that cancelled the first mid-flight (echo
    already deleted, recovery consumed), losing the text."""
    from types import SimpleNamespace

    from Tests.UI.console_controller_stubs import stub_message_controller
    from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage

    echo = ConsoleChatMessage(role=USER, content="hello", id="u1", status="failed")
    store = SimpleNamespace(get_message=lambda _id: echo, active_session_id="s1")
    controller = SimpleNamespace(store=store, send_refusal_copy=lambda _sid: None)
    running = SimpleNamespace(
        name="console-resend", group="console-run-s1", is_finished=not in_flight
    )
    started: list[dict] = []
    notices: list[str] = []

    def run_worker(work, **kwargs):
        work.close()
        started.append(kwargs)

    owner = stub_message_controller(
        SimpleNamespace(
            run_worker=run_worker,
            workers=[running],
            _console_message_presentation=lambda message: SimpleNamespace(
                content=message.content
            ),
        ),
        app_instance=SimpleNamespace(
            notify=lambda text, **_kwargs: notices.append(text)
        ),
        chat_store_accessor=lambda: store,
        ensure_console_chat_controller=lambda: controller,
        generation_refusal_copy=lambda controller, session_id: (
            controller.send_refusal_copy(session_id)
        ),
    )
    button = SimpleNamespace(
        id="console-message-action-resend-u1",
        console_action_id="resend",
        console_message_id="u1",
    )

    assert await owner.handle_console_message_action(
        SimpleNamespace(button=button, stop=lambda: None)
    )

    if in_flight:
        assert started == []
        assert notices == ["Resend is already in progress."]
    else:
        assert [kwargs["group"] for kwargs in started] == ["console-run-s1"]


# --- the real Console --------------------------------------------------------


def _console_app(gateway):
    app = _build_console_send_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    return ConsoleHarness(app)


async def _await_selected_console_readiness(console, pilot, *, deadline):
    """Await the current readiness owner, then its original checked paint."""
    import inspect
    from types import CoroutineType, MethodType

    bindings = (
        (ChatScreen, "_sync_console_control_bar"),
        (ConsoleReadinessConfigProjection, "_refresh"),
    )
    originals = tuple(
        (
            owner,
            name,
            vars(owner)[name],
            vars(owner)[name].__code__,
            vars(owner)[name].__globals__,
            vars(owner)[name].__defaults__,
        )
        for owner, name in bindings
    )
    control, refresh_code = originals[0][2], originals[1][3]

    def owns_original(coroutine):
        for _depth in range(64):
            if type(coroutine) is not CoroutineType:
                return False
            frame = coroutine.cr_frame
            if frame is not None:
                receiver = frame.f_locals.get("self")
                if (
                    coroutine.cr_code is refresh_code
                    and type(receiver) is ConsoleReadinessConfigProjection
                    and receiver
                    is getattr(console, "_console_readiness_config_projection", None)
                    and receiver.screen is console
                    and frame.f_globals is originals[1][4]
                ):
                    return True
            coroutine = coroutine.cr_await
        return False

    while True:
        remaining = deadline - time.monotonic()
        assert (
            remaining > 0
        ), "Selected Console readiness exceeded original startup budget"
        assert all(
            vars(owner).get(name) is function
            and function.__code__ is code
            and function.__globals__ is namespace
            and function.__defaults__ is defaults
            for owner, name, function, code, namespace, defaults in originals
        ), "Original Console readiness source changed"
        assert inspect.getattr_static(console, "_sync_console_control_bar") is control
        pending = any(
            worker.node is console and owns_original(worker._work)
            for worker in console.workers
        ) or any(
            owns_original(task.get_coro())
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
        )
        projection = getattr(console, "_console_readiness_config_projection", None)
        projection_pending = (
            type(projection) is ConsoleReadinessConfigProjection
            and projection.screen is console
            and projection.pending
        )
        if not pending and not projection_pending:
            callback = console._sync_console_control_bar
            assert (
                type(callback) is MethodType
                and callback.__self__ is console
                and callback.__func__ is control
            )
            if callback() is True:
                assert (
                    time.monotonic() < deadline
                ), "Selected Console readiness exceeded original startup budget"
                return
        await asyncio.sleep(min(0.01, remaining))


async def _select_ready_llamacpp_console(console, pilot):
    """Share the original composer's two-second setup budget with publication."""
    deadline = time.monotonic() + 2.0
    await _wait_for_selector(console, pilot, "#console-native-composer")
    _select_llamacpp_console(console)
    await _await_selected_console_readiness(console, pilot, deadline=deadline)


def _session_rows(console) -> list[ConsoleChatMessage]:
    store = console._ensure_console_chat_store()
    return store.messages_for_session(store.active_session_id)


@pytest.mark.asyncio
async def test_console_resend_click_re_runs_a_failed_turn_in_place():
    host = _console_app(FailThenRecoverGateway())

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _select_ready_llamacpp_console(console, pilot)
        console.query_one("#console-native-composer", ConsoleComposerBar).load_draft(
            "hello"
        )
        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "llama.cpp stream failed")
        user, failed = (
            row for row in _session_rows(console) if row.role in {USER, ASSISTANT}
        )
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(user.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-resend-{user.id}"
        )

        assert await pilot.click(f"#console-message-action-resend-{user.id}")
        await _wait_for_text(console, pilot, "recovered")
        rows = _session_rows(console)

    assert [(row.role, row.status, row.content) for row in rows] == [
        (USER, "complete", "hello"),
        (ASSISTANT, "complete", "recovered"),
    ]
    assert [row.id for row in rows] == [user.id, failed.id]


@pytest.mark.asyncio
async def test_console_poll_outlives_a_turn_the_controller_has_not_started(
    monkeypatch,
):
    """The fast-lane failure, made deterministic: on a slow runner the 0.2s
    transcript poll ticked while the runtime held the accepted turn but the
    controller had not started it, saw an idle run, stopped, and never
    rendered the turn.

    Poll decisions are counted from the moment the stalled start is entered,
    not from the Send press. A Send reaches custody on its own schedule (the
    hook snapshot and the turn authority are read off the UI pump,
    TASK-33620.15): 0.3-0.7 s on a dev machine, longer on a loaded runner. A
    fixed 0.6 s pause after the press asserted before the poll had even been
    started (UI Fast Lane, TASK-33628.5.1; red on dev too under
    ``taskpolicy -b``).
    """
    host = _console_app(FailThenRecoverGateway())

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _select_ready_llamacpp_console(console, pilot)
        controller = console._ensure_console_chat_controller()
        entered = asyncio.Event()
        started = asyncio.Event()
        submit_draft = controller.submit_draft

        async def slow_start(*args, **kwargs):
            entered.set()
            await started.wait()
            return await submit_draft(*args, **kwargs)

        decisions: list[bool] = []
        poll_needed = console._console_transcript_poll_needed

        def counted_poll_needed() -> bool:
            needed = poll_needed()
            decisions.append(needed)
            return needed

        monkeypatch.setattr(controller, "submit_draft", slow_start)
        monkeypatch.setattr(
            console, "_console_transcript_poll_needed", counted_poll_needed
        )
        console.query_one("#console-native-composer", ConsoleComposerBar).load_draft(
            "hello"
        )
        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(entered.wait(), 10)
        assert console._console_runtime().has_custodied_turns()
        held = len(decisions)
        for _ in range(100):  # three poll decisions while the start is held
            if (
                len(decisions) >= held + 3
                or console._console_transcript_sync_timer is None
            ):
                break
            await pilot.pause(0.05)
        assert decisions[held:][:3] == [True, True, True]
        assert console._console_transcript_sync_timer is not None
        started.set()
        await _wait_for_text(console, pilot, "llama.cpp stream failed")


class _BlockThenStreamGateway(_ReadyResolutionGateway):
    def __init__(self) -> None:
        self.blocked = True

    async def resolve_for_send(self, selection):
        if self.blocked:
            return await BlockedGateway().resolve_for_send(selection)
        return await super().resolve_for_send(selection)

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        yield "answered"


@pytest.mark.asyncio
async def test_console_r_resends_a_refused_echo_as_one_message():
    gateway = _BlockThenStreamGateway()
    host = _console_app(gateway)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _select_ready_llamacpp_console(console, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")
        console.query_one("#console-send-message", Button).press()
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        # Control-bar refusal text can precede the echo's transcript publication.
        await _wait_for_text(transcript, pilot, "llama.cpp unavailable")
        echo = next(row for row in _session_rows(console) if row.role is USER)
        assert echo.status == "failed"
        session_id = console._ensure_console_chat_store().active_session_id
        assert console._console_runtime().recoveries_for_session(session_id)
        gateway.blocked = False

        transcript.focus()
        transcript.select_message(echo.id)
        await _wait_for_selector(
            console, pilot, f"#console-message-action-resend-{echo.id}"
        )
        await pilot.press("r")
        await _wait_for_text(console, pilot, "answered")
        rows = _session_rows(console)
        recoveries = console._console_runtime().recoveries_for_session(session_id)
        draft = composer.draft_text()

    assert [(row.role, row.status, row.content) for row in rows] == [
        (USER, "complete", "hello"),
        (ASSISTANT, "complete", "answered"),
    ]
    assert recoveries == ()
    assert draft == ""


class _HeldRecoveryGateway(_ReadyResolutionGateway):
    def __init__(self):
        self.calls = 0
        self.streaming = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            yield "original partial"
            raise RuntimeError("llama.cpp stream failed")
        yield "probe incremental reply"
        self.streaming.set()
        await self.release.wait()
        yield " finished"


@pytest.mark.asyncio
@private_profile_test
async def test_resend_poll_survives_initial_hook_admission_read(request, monkeypatch):
    gateway = _HeldRecoveryGateway()
    host = _console_app(gateway)
    hook_entered = asyncio.Event()
    hook_release = asyncio.Event()
    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _select_ready_llamacpp_console(console, pilot)
        console.query_one("#console-native-composer", ConsoleComposerBar).load_draft(
            "hello"
        )
        send = console.query_one("#console-send-message", Button)
        assert send.display and not send.disabled
        send.press()
        await _wait_for_text(console, pilot, "llama.cpp stream failed")
        await pilot.pause(0.4)
        user = next(row for row in _session_rows(console) if row.role is USER)
        assert user.persisted_message_id is not None
        controller = console._ensure_console_chat_controller()
        original_admission = controller.hook_admission_reason
        reads = 0

        async def held_admission():
            nonlocal reads
            reads += 1
            if reads == 1:
                hook_entered.set()
                await hook_release.wait()
            return await original_admission()

        monkeypatch.setattr(controller, "hook_admission_reason", held_admission)
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(user.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-resend-{user.id}"
        )
        try:
            assert await pilot.click(f"#console-message-action-resend-{user.id}")
            await asyncio.wait_for(hook_entered.wait(), 5)
            assert any(
                worker.name == "console-resend" and not worker.is_finished
                for worker in console.workers
            )
            timer = console._console_transcript_sync_timer
            if timer is not None:
                await timer._callback()
            hook_release.set()
            await asyncio.wait_for(gateway.streaming.wait(), 5)
            await pilot.pause(0.65)
            assert any(
                "probe incremental reply" in row.content
                for row in _session_rows(console)
            )
            assert "probe incremental reply" in _visible_text(console)
        finally:
            hook_release.set()
            gateway.release.set()
            await pilot.pause(0.25)


@pytest.mark.asyncio
@private_profile_test
async def test_transcript_resend_retains_failed_rows_until_connection_is_ready(request):
    from datetime import UTC, datetime

    from tldw_chatbook.Chat.console_session_settings import console_send_connection
    from tldw_chatbook.Chat.provider_test_evidence import (
        ProviderTestEvidence,
        provider_connection_evidence,
    )

    gateway = FailThenRecoverGateway()
    host = _console_app(gateway)
    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _select_ready_llamacpp_console(console, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")
        send = console.query_one("#console-send-message", Button)
        assert send.display and not send.disabled
        send.press()
        await _wait_for_text(console, pilot, "llama.cpp stream failed")
        user = next(row for row in _session_rows(console) if row.role is USER)
        settings = console._active_console_settings_readiness_uncached()[0]
        identity = console_send_connection(
            settings,
            app_config=console._provider_readiness_app_config(),
        )
        assert identity is not None
        owner = provider_connection_evidence(host)
        assert owner.publish(
            ProviderTestEvidence(
                identity,
                "unreachable",
                (),
                "connection_refused",
                observed_at=datetime.now(UTC),
            ),
            order=1,
        )
        console._poll_console_credential_readiness()
        await console._sync_native_console_chat_ui()
        assert (
            console._active_console_settings_readiness()[1].blocker
            == "endpoint_unreachable"
        )
        assert console.query_one("#console-send-message", Button).disabled
        composer.load_draft("")
        console._ensure_console_chat_store().set_session_draft(
            console._ensure_console_chat_store().active_session_id, ""
        )
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(user.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-resend-{user.id}"
        )
        before = tuple(
            (row.id, row.status, row.content) for row in _session_rows(console)
        )
        assert await pilot.click(f"#console-message-action-resend-{user.id}")
        await pilot.pause(0.3)
        assert gateway.calls == 1
        assert (
            tuple((row.id, row.status, row.content) for row in _session_rows(console))
            == before
        )
        assert owner.publish(
            ProviderTestEvidence(
                identity,
                "reachable",
                (settings.model,),
                observed_at=datetime.now(UTC),
            ),
            order=2,
        )
        console._poll_console_credential_readiness()
        await console._sync_native_console_chat_ui()
        assert console._active_console_settings_readiness()[1].blocker is None
        assert await pilot.click(f"#console-message-action-resend-{user.id}")
        await _wait_for_text(console, pilot, "recovered")
        assert gateway.calls == 2
        assert (
            next(row for row in _session_rows(console) if row.role is USER).id
            == user.id
        )
