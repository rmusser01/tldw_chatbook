"""TASK-33661: the Resend action in the transcript and the real Console.

The transcript tests mount a bare ``ConsoleTranscript``; the pilot tests drive
the real ``ChatScreen`` through a failed stream and a refused send, then click
(or press ``r`` on) Resend. One pilot pins the transcript poll those sends rely
on, which once stopped before a slow runner's turn had started.
"""

from __future__ import annotations

import asyncio
import sys
import threading
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
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
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


def _session_rows(console) -> list[ConsoleChatMessage]:
    store = console._ensure_console_chat_store()
    return store.messages_for_session(store.active_session_id)


def _unstarted_send_stage(console) -> str:
    """Name how far a pressed Send got without reaching ``submit_draft``."""
    store = console._ensure_console_chat_store()
    acknowledgement = getattr(console, "_console_send_ack", None)
    send = console.query_one("#console-send-message", Button)
    return (
        "the Send never reached the stalled submit_draft: "
        f"in runtime custody={console._console_runtime().has_custodied_turns()}, "
        "acknowledged="
        f"{acknowledgement is not None and acknowledgement.active_for(store.active_session_id)}, "
        f"Send={str(send.label)!r} disabled={send.disabled} display={send.display} "
        f"tooltip={send.tooltip!r}, "
        f"notices={[notice.message for notice in console.app._notifications]!r}"
    )


@pytest.mark.asyncio
async def test_console_resend_click_re_runs_a_failed_turn_in_place():
    host = _console_app(FailThenRecoverGateway())

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
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


#: How long the holder thread keeps the config locks. The helper is called as
#: soon as the hold is taken, so this only has to outlast the gap between the
#: two.
_CONFIG_LOCK_HOLD_S = 1.0


async def _hold_config_locks() -> tuple[
    threading.Thread, threading.Event, threading.Event
]:
    """Hold both config locks on another thread for ``_CONFIG_LOCK_HOLD_S``.

    Waits for the hold off the event loop: a holder of the locks that needs
    the loop to release them (a Console worker, say) must not wait on this.

    Returns:
        The holder thread, an event set once its hold has ended, and an event
        that ends the hold early.
    """
    held, ended, release = (threading.Event() for _ in range(3))

    def hold() -> None:
        # Same locks, same order as run_console_config_sync takes them.
        with (
            config_module._settings_rebuild_lock(),
            config_module._config_file_lock(),
        ):
            held.set()
            release.wait(_CONFIG_LOCK_HOLD_S)
            ended.set()

    holder = threading.Thread(target=hold, name="config-lock-holder", daemon=True)
    holder.start()
    assert await asyncio.to_thread(held.wait, 10), (
        "the holder never took the config locks"
    )
    return holder, ended, release


@pytest.mark.asyncio
async def test_select_llamacpp_console_enables_send_while_another_thread_holds_the_config_locks():
    """Callers of ``_select_llamacpp_console`` load a draft and press Send at
    once, so the helper must render the selection before it returns, even
    while another thread holds the config locks.

    In the CI failures TASK-34789 lists (TASK-35660), a control-bar sync that
    could not take the locks deferred to a 0.2 s retry. Send kept its
    mount-time block, and ``Button.press()`` on a disabled button posts
    nothing, so the test timed out waiting for a send that was never made.
    Locally the Console's post-mount hook-permission read holds the locks on a
    worker thread; which holder overlapped on the runner is not known.

    This pins the outcome callers rely on, not how the helper reaches it. The
    helper's readiness warm-up currently waits for the hold to end (about 1 s
    here) before its sync, so removing only its lock wrap would still pass;
    the helper as it was before TASK-34789 fails.
    """
    host = _console_app(FailThenRecoverGateway())

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        send = console.query_one("#console-send-message", Button)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        # Not ``send.disabled``: with an empty draft Send is disabled whether or
        # not a block exists, and without the block this test passes vacuously.
        assert composer._send_blocked, (
            f"no mount-time block for the helper to lift: {send.tooltip!r}"
        )
        for _ in range(3):
            holder, hold_ended, release = await _hold_config_locks()
            if not hold_ended.is_set():
                break
            holder.join(10)  # A stalled loop outlived the hold: take it again.
        else:
            raise AssertionError("the config-lock hold kept lapsing before the helper")
        try:
            await _select_llamacpp_console(console)
            composer.load_draft("hello")
            assert not send.disabled, f"Send is disabled: {send.tooltip!r}"
        finally:
            release.set()
            holder.join(10)


@pytest.mark.asyncio
async def test_select_llamacpp_console_renders_after_its_readiness_proof_goes_stale(
    monkeypatch,
):
    """The helper still renders when its loop resumes it late.

    Under ``taskpolicy -b`` the helper's sync deferred and the helper failed
    with a bare ``assert False`` (TASK-35660). Its sync renders from the
    readiness proof that its warm-up read, and that proof is good for one
    second. This replays one order that makes the sync defer: the warm-up
    returns current, the loop resumes the helper 1.1 s later, and a display
    has started the next refresh in between.

    Send alone cannot show that the helper rendered: during the stall the
    product's own delayed control-bar replay may lift the mount-time block.
    So the test records the helper's own syncs and requires a deferral
    followed by a render.
    """
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleReadinessConfigProjection,
    )

    host = _console_app(FailThenRecoverGateway())

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        send = console.query_one("#console-send-message", Button)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        # The block itself, not ``send.disabled``: see the lock-hold test above.
        assert composer._send_blocked, (
            f"no mount-time block for the helper to lift: {send.tooltip!r}"
        )
        warm = ConsoleReadinessConfigProjection.warm
        late: list[bool] = []
        # The helper runs in this task; the context snapshot warms from workers.
        helper_task = asyncio.current_task()

        async def warm_then_resume_late(projection) -> bool:
            current = await warm(projection)
            if current and not late and asyncio.current_task() is helper_task:
                # A display that refreshed during a stall leaves the proof
                # fresh and starts nothing, so stall past its second again.
                for _ in range(3):
                    await asyncio.sleep(1.1)  # past the proof's one second
                    projection.run(lambda: None)  # a display starts the next refresh
                    if projection.pending:
                        break
                late.append(projection.pending)
            return current

        real_sync = console._sync_console_control_bar
        helper_syncs: list[bool] = []

        def recording_sync(*rail_state) -> bool:
            rendered = real_sync(*rail_state)
            if asyncio.current_task() is helper_task:
                helper_syncs.append(bool(rendered))
            return rendered

        monkeypatch.setattr(
            ConsoleReadinessConfigProjection, "warm", warm_then_resume_late
        )
        monkeypatch.setattr(console, "_sync_console_control_bar", recording_sync)
        await _select_llamacpp_console(console)
        syncs = list(helper_syncs)  # only the helper's, not the draft load's
        composer.load_draft("hello")
        assert late == [True], (
            "the warm-up never returned current, or no refresh started"
        )
        # The stale proof deferred the helper's first sync; it must not return
        # until one of its own syncs renders.
        assert syncs and not syncs[0] and syncs[-1], syncs
        assert not send.disabled, f"Send is disabled: {send.tooltip!r}"


#: Bound for the poll pilot's waits after its press. Under ``taskpolicy -b`` on
#: a quiet host the send reached the stalled start in about 4 s. With a 5 s
#: precapture delay on a host at load average ~100 it took up to ~30 s, and
#: one poll tick (a full paint) took several seconds, so bounds of 30 s and
#: 20 s failed there, with the shared helper before and after TASK-35660's
#: change alike.
_LOADED_HOST_WAIT_S = 60.0


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

    A press made while the provider switch's deferred control-bar sync still
    held Send disabled was dropped, and the wait for the stalled start timed
    out with no send begun (UI Fast Lane, TASK-35660). The shared helper now
    renders before it returns (TASK-34789), and the press asserts Send is
    enabled so a dropped press fails here, naming Send's blocked reason.
    Both waits after the press are bounded by ``_LOADED_HOST_WAIT_S`` and end
    as soon as their condition holds, and the first names the stage the send
    reached if it times out.
    """
    host = _console_app(FailThenRecoverGateway())

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
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
        send = console.query_one("#console-send-message", Button)
        # press() on a disabled or undisplayed Send posts nothing (TASK-35660).
        assert not send.disabled and send.display, (
            f"Send would drop the press: disabled={send.disabled} "
            f"display={send.display} tooltip={send.tooltip!r}"
        )
        send.press()
        try:
            await asyncio.wait_for(entered.wait(), _LOADED_HOST_WAIT_S)
        except TimeoutError:
            raise AssertionError(_unstarted_send_stage(console)) from None
        assert console._console_runtime().has_custodied_turns()
        held = len(decisions)
        # Three poll decisions while the start is held.
        deadline = time.monotonic() + _LOADED_HOST_WAIT_S
        while (
            len(decisions) < held + 3
            and console._console_transcript_sync_timer is not None
            and time.monotonic() < deadline
        ):
            await pilot.pause(0.05)
        assert decisions[held:][:3] == [True, True, True]
        assert console._console_transcript_sync_timer is not None
        started.set()
        await _wait_for_text(console, pilot, "llama.cpp stream failed")


@pytest.mark.asyncio
async def test_console_poll_paints_again_after_work_that_ended_during_a_paint(
    monkeypatch,
):
    """A turn that ends while a poll tick is painting committed its last rows
    after that tick read them, and once a send's echo has replaced its
    "Sending…" row nothing repaints a finished turn by itself.

    Measured on a loaded machine (TASK-34789, the test above, 1 run in 86):
    one tick's paint took 0.448 s (0.346 s of it the transcript), the turn
    ended inside it, the tick then found no live work and stopped the poll,
    and the Console stayed at "Run: Validating provider." with the failed
    reply never shown.
    """
    host = _console_app(FailThenRecoverGateway())

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console._ensure_console_chat_controller()
        live = True
        poll_paints = []

        async def paint() -> None:
            nonlocal live
            if sys._getframe(1).f_code.co_name != "sync_poll_display_ui":
                return  # only the poll's own paints are counted
            poll_paints.append(live)
            live = False  # the work ends while the first one is under way

        monkeypatch.setattr(console, "_console_transcript_poll_needed", lambda: live)
        monkeypatch.setattr(console, "_sync_native_console_chat_ui", paint)
        console._start_console_transcript_sync_timer()
        for _ in range(100):
            if console._console_transcript_sync_timer is None:
                break
            await pilot.pause(0.05)
        stopped = console._console_transcript_sync_timer is None

    assert stopped
    # The paint during which the work ended, then the poll's own repaint.
    assert poll_paints[:2] == [True, False]


@pytest.mark.asyncio
async def test_console_poll_paints_again_after_work_that_started_and_ended_during_a_paint(
    monkeypatch,
):
    """The poll's last tick began with no live work, and a send was admitted
    and its turn ended while that tick painted (review of #3091). The send's
    request for the poll was a no-op, the timer still existing; the tick found
    no live work and stopped, and the turn's last rows were never painted."""
    host = _console_app(FailThenRecoverGateway())

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console._ensure_console_chat_controller()
        live = False
        poll_paints = []

        async def paint() -> None:
            nonlocal live
            if sys._getframe(1).f_code.co_name != "sync_poll_display_ui":
                return  # only the poll's own paints are counted
            poll_paints.append(not poll_paints)
            if len(poll_paints) == 1:
                live = True  # a send is admitted...
                console._start_console_transcript_sync_timer()  # ...asks...
                await asyncio.sleep(0)
                live = False  # ...and its turn ends, all inside this paint

        monkeypatch.setattr(console, "_console_transcript_poll_needed", lambda: live)
        monkeypatch.setattr(console, "_sync_native_console_chat_ui", paint)
        console._start_console_transcript_sync_timer()
        for _ in range(100):
            if console._console_transcript_sync_timer is None:
                break
            await pilot.pause(0.05)
        stopped = console._console_transcript_sync_timer is None

    assert stopped
    assert poll_paints[:2] == [True, False]


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
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
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
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
        console.query_one("#console-native-composer", ConsoleComposerBar).load_draft(
            "hello"
        )
        console.query_one("#console-send-message", Button).press()
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
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")
        console.query_one("#console-send-message", Button).press()
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
