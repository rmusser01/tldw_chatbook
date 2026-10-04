"""Enter acknowledges a Console send before its admission work (TASK-33620.5).

Measured on dev 7d155170dc (live, 160x45, Anthropic haiku through the review
proxy): after Enter nothing changed for 0.46-0.95 s, then the composer cleared
to "Send disabled: type a message" under an idle "Ready" header, and the user
row, tab dot, Run chip and running header only appeared 1.7-3.1 s after Enter.
Two causes: Enter's send ran its synchronous admission (``_build_console_turn_
execution_context``) on the UI pump before anything was painted, and the row
and status surfaces only reached the screen through the 0.2 s whole-screen
sync tick.

These tests drive the real ChatScreen send path (driver-delivered Enter, the
app's eager task factory, a real in-memory ChaChaNotes store and runtime) and
read what the compositor actually painted. The send's admission gate (the
hook-permission snapshot, the dispatch's first awaited step) and provider
validation are each held open, so "painted before admission" is an ordering
fact in one recorded sequence, never a wall-clock race.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from textual import events

from Tests.UI.test_console_native_chat_flow import (
    _ReadyResolutionGateway,
    _build_console_send_test_app,
    _select_llamacpp_console,
    _wait_for_selector,
)
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.Console_Modules.prompt_queue import turn_recovery_label
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

pytestmark = pytest.mark.bootstrap_profile

DRAFT = "acknowledge this send"
REPLY = "acknowledged reply"
REFUSAL = "Hook admission refused for this test."


@dataclass(frozen=True)
class Paint:
    """The send-relevant facts of one painted frame."""

    user_row: bool
    draft_in_composer: bool
    sending: bool
    header_idle: bool
    tab_running: bool
    run_chip: bool
    empty_state: bool
    empty_draft_reason: bool


def _painted_lines(host) -> list[str]:
    return [strip.text for strip in host.screen._compositor.render_strips()]


def paint_state(host) -> Paint:
    """Classify the frame the compositor would put on the terminal now."""
    lines = _painted_lines(host)
    composer = [line for line in lines if "Composer" in line and "Menu" in line]
    others = [line for line in lines if line not in composer]
    header = next((line for line in lines if "Console — Chat" in line), "")
    tabs = [line for line in lines if "✕" in line]
    text = "\n".join(lines)
    return Paint(
        user_row=any(DRAFT in line for line in others),
        draft_in_composer=any(DRAFT in line for line in composer),
        sending="Sending" in text,
        header_idle="Ready" in header,
        tab_running=any("●" in line for line in tabs),
        run_chip="Run:" in text,
        empty_state="No messages yet." in text or "type a message to begin" in text,
        empty_draft_reason="Send disabled: type a message" in text,
    )


class Timeline:
    """One ordered record of painted frames and send-path events."""

    def __init__(self, host) -> None:
        self.host = host
        self.entries: list[tuple[str, object]] = []
        self._display = None

    def install(self) -> None:
        original = self.host._display

        def display(screen, renderable) -> None:
            original(screen, renderable)
            if renderable is not None:
                self.entries.append(("paint", paint_state(self.host)))

        self._display = original
        self.host._display = display

    def uninstall(self) -> None:
        if self._display is not None:
            self.host._display = self._display

    def mark(self, name: str) -> None:
        self.entries.append(("event", name))

    def index(self, name: str) -> int:
        return next(
            i
            for i, (kind, value) in enumerate(self.entries)
            if kind == "event" and value == name
        )

    def paints_between(self, start: str, end: str) -> list[tuple[int, Paint]]:
        lo, hi = self.index(start), self.index(end)
        return [
            (i, value)
            for i, (kind, value) in enumerate(self.entries)
            if kind == "paint" and lo < i < hi
        ]


class HeldGateway(_ReadyResolutionGateway):
    """Holds provider validation open and counts provider stream calls."""

    def __init__(self, timeline: Timeline) -> None:
        self.timeline = timeline
        self.validation_started = asyncio.Event()
        self.validation_release = asyncio.Event()
        self.stream_calls = 0

    def cached_context_window(self, settings):
        from tldw_chatbook.Utils.token_counter import resolve_context_window

        return resolve_context_window(settings.provider, settings.model or "")

    async def resolve_context_window(self, settings):
        return self.cached_context_window(settings)

    async def resolve_for_send(self, selection):
        self.timeline.mark("validation")
        self.validation_started.set()
        await self.validation_release.wait()
        return await super().resolve_for_send(selection)

    async def stream_chat(self, resolution, messages, **kwargs):
        self.stream_calls += 1
        self.timeline.mark("provider")
        yield REPLY


class HeldAdmission:
    """Holds the Send's hook-permission snapshot: the dispatch's first await.

    It runs in ``asyncio.to_thread``, so the pumps stay free to paint while
    the send cannot reach admission.
    """

    def __init__(self, console) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.armed = True
        real = console._hooks._permissions
        gate = self

        class _Owner:
            def __init__(self, owner) -> None:
                self._owner = owner

            def snapshot(self):
                if gate.armed:
                    gate.armed = False
                    gate.entered.set()
                    gate.release.wait(timeout=30)
                return self._owner.snapshot()

            def __getattr__(self, name):
                return getattr(self._owner, name)

        console._hooks._permissions = lambda: _Owner(real())


def mark_admission(console, timeline: Timeline) -> None:
    """Record when the synchronous runtime admission starts."""
    real = console._session._build_console_turn_execution_context

    def recording_build(session_id):
        timeline.mark("admission")
        return real(session_id)

    console._session._build_console_turn_execution_context = recording_build


def press(host, key: str, char: str | None = None) -> None:
    """Deliver a key the way the terminal driver does (no Pilot idle wait)."""
    event = events.Key(key, char)
    event.set_sender(host)
    host._driver.send_message(event)


async def until(predicate, *, timeout: float = 15.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for {predicate}")
        await asyncio.sleep(0.01)


@contextlib.contextmanager
def eager_tasks():
    """``App.run_async`` installs the eager task factory; ``run_test`` does not."""
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        yield
    finally:
        loop.set_task_factory(previous)


async def ready_console(host, pilot, gateway):
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    _select_llamacpp_console(console)
    await pilot.pause(0.3)
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.focus()
    composer.load_draft(DRAFT)
    await pilot.pause()
    assert paint_state(host).draft_in_composer
    return console, composer


def build() -> tuple:
    """A send-ready Console harness whose gateway holds validation."""
    app = _build_console_send_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ConsoleHarness(app)
    timeline = Timeline(host)
    gateway = HeldGateway(timeline)
    app.console_provider_gateway_factory = lambda: gateway
    return host, gateway, timeline


@pytest.mark.asyncio
async def test_enter_paints_a_sending_row_before_admission_and_validation():
    """AC#1/#2/#5: the acknowledgement frame precedes admission and validation."""
    host, gateway, timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            gate = HeldAdmission(console)
            mark_admission(console, timeline)
            timeline.install()
            try:
                timeline.mark("enter")
                press(host, "enter", "\r")
                await until(gate.entered.is_set)
                # The app pump is now awaiting the held send, as it does in
                # production during admission: no Pilot idle wait here. What
                # the compositor holds IS the last painted frame.
                held = paint_state(host)
                assert held.user_row and held.sending, held
                assert held.tab_running and held.run_chip, held
                assert not held.header_idle, held
                assert not held.empty_state and not held.empty_draft_reason, held
                assert not gateway.validation_started.is_set()
                assert gateway.stream_calls == 0

                gate.release.set()
                await until(gateway.validation_started.is_set)
                validating = paint_state(host)
                assert validating.user_row and validating.run_chip, validating
                assert not validating.header_idle, validating
                assert not validating.empty_draft_reason, validating
                assert not validating.empty_state, validating
                assert gateway.stream_calls == 0
            finally:
                gate.release.set()
                gateway.validation_release.set()
            await until(lambda: gateway.stream_calls == 1)
            await until(lambda: REPLY in "\n".join(_painted_lines(host)))
            timeline.uninstall()

    enter = timeline.index("enter")
    admission = timeline.index("admission")
    acknowledged = next(
        i
        for i, (kind, value) in enumerate(timeline.entries)
        if kind == "paint"
        and i > enter
        and value.user_row
        and value.sending
        and not value.header_idle
    )
    assert enter < acknowledged < admission < timeline.index("validation")
    assert timeline.index("validation") < timeline.index("provider")
    pending = timeline.paints_between("enter", "provider")
    # AC#6 in the harness: never an empty composer under an idle header with
    # no user row; AC#2: never the empty-draft reason or the empty state.
    assert not [
        i
        for i, paint in pending
        if not paint.draft_in_composer and not paint.user_row and paint.header_idle
    ]
    assert not [i for i, paint in pending if paint.empty_draft_reason]
    assert not [i for i, paint in pending if i > acknowledged and paint.empty_state]


@pytest.mark.asyncio
async def test_refused_admission_rolls_the_sending_row_back_and_keeps_the_draft():
    """AC#3: a runtime refusal removes the pending row and leaves the draft."""
    host, gateway, _timeline = build()
    notices: list[str] = []
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            host.app_instance.notify = lambda message, **_k: notices.append(
                str(message)
            )
            gate = HeldAdmission(console)
            runtime = console._console_runtime()

            def refuse(*_args, **_kwargs):
                raise RuntimeError("Console session is closed.")

            runtime.accept_turn = refuse
            try:
                press(host, "enter", "\r")
                await until(gate.entered.is_set)
                held = paint_state(host)
                assert held.user_row and held.sending, held
            finally:
                gate.release.set()
            await until(lambda: "Console session is closed." in notices)
            await until(lambda: not paint_state(host).sending)
            await pilot.pause()
            settled = paint_state(host)
            assert not settled.user_row and not settled.sending, settled
            assert settled.draft_in_composer, settled
            assert settled.header_idle and not settled.run_chip, settled
            assert composer.draft_text() == DRAFT
            assert gateway.stream_calls == 0


@pytest.mark.asyncio
async def test_turn_refused_before_its_echo_releases_the_row_and_offers_recovery():
    """AC#3: custody that ends before the store echo drops the pending row."""
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            controller = console._ensure_console_chat_controller()
            controller.hook_admission_reason = AsyncMock(return_value=REFUSAL)
            gate = HeldAdmission(console)
            try:
                press(host, "enter", "\r")
                await until(gate.entered.is_set)
                held = paint_state(host)
                assert held.user_row and held.sending, held
            finally:
                gate.release.set()
            await until(lambda: not console._console_runtime().has_custodied_turns())
            await until(lambda: not paint_state(host).sending)
            await pilot.pause()
            settled = paint_state(host)
            assert not settled.user_row and not settled.sending, settled
            assert not settled.run_chip and settled.header_idle, settled
            # Dev's TASK-33621.2 shelf states the refusal: "Not sent: <reason>".
            shelf = turn_recovery_label(REFUSAL)
            assert shelf.startswith("Not sent: Hook admission refused"), shelf
            assert shelf in "\n".join(_painted_lines(host))
            assert gateway.stream_calls == 0


@pytest.mark.asyncio
async def test_second_enter_during_admission_never_starts_a_second_provider_call():
    """AC#4: the first send is visibly in flight; a repeat is refused, not sent."""
    host, gateway, _timeline = build()
    notices: list[str] = []
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            host.app_instance.notify = lambda message, **_k: notices.append(
                str(message)
            )
            gate = HeldAdmission(console)
            try:
                press(host, "enter", "\r")
                await until(gate.entered.is_set)
                assert paint_state(host).user_row, paint_state(host)
                assert paint_state(host).sending, paint_state(host)
                # Queued behind the held send, like a key typed during the
                # real synchronous admission.
                press(host, "enter", "\r")
                await asyncio.sleep(0.2)
                gate.release.set()
                await until(gateway.validation_started.is_set)
                await until(lambda: composer.draft_text() == "")
                composer.load_draft(DRAFT)
                await pilot.pause()
                press(host, "enter", "\r")
                await until(lambda: len(notices) >= 1)
                await pilot.pause()
                in_flight = paint_state(host)
                assert in_flight.user_row and not in_flight.header_idle, in_flight
                assert in_flight.draft_in_composer, in_flight
            finally:
                gate.release.set()
                gateway.validation_release.set()
            await until(lambda: REPLY in "\n".join(_painted_lines(host)))
            await pilot.pause(0.3)
            assert gateway.stream_calls == 1
            assert composer.draft_text() == DRAFT


@pytest.mark.asyncio
async def test_unchanged_control_bar_sync_does_no_style_work():
    """A whole-screen tick must not re-style the control bar it did not change.

    ``ConsoleControlBar._set_recovery_height`` removed and re-added its
    height class on every 0.2 s tick; each class change synchronously
    re-applies CSS to the bar and every descendant (20-100 ms per tick in the
    harness profile, the largest single cost of a settled tick).
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        console, _composer = await ready_console(host, pilot, gateway)
        bar = console.query_one("#console-control-bar")
        console._console_auto_speak.sync_controls()
        await pilot.pause()
        restyled: list[object] = []
        original = host.update_styles

        def update_styles(node, animate: bool = True) -> None:
            if node is bar or bar in node.ancestors_with_self:
                restyled.append(node)
            original(node, animate=animate)

        host.update_styles = update_styles
        try:
            console._console_auto_speak.sync_controls()
            console._sync_console_control_bar()
        finally:
            host.update_styles = original
        assert restyled == []


# -- the acknowledgement's own rules (no app) ---------------------------------


def _ack():
    from tldw_chatbook.UI.Console_Modules.send_acknowledgement import (
        ConsoleSendAcknowledgement,
    )

    released: list[int] = []
    return ConsoleSendAcknowledgement(lambda: released.append(1)), released


def _message(role, message_id: str):
    from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage

    return ConsoleChatMessage(role=role, content="x", id=message_id)


def test_acknowledgement_row_stands_in_until_the_store_echo_lands():
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole as Role

    ack, released = _ack()
    old = _message(Role.USER, "old-user")
    token = ack.begin("s1", DRAFT, ["old-user"])
    assert token is not None
    assert ack.begin("s1", "second", ["old-user"]) is None  # one per session
    other = ack.begin("s2", "another tab", [])  # never blocked by s1's turn
    assert other is not None and ack.active_for("s2")
    ack.release(other)
    released.clear()

    pending = ack.project("s1", [old])
    assert [row.id for row in pending][0] == "old-user"
    assert pending[-1].content == DRAFT and pending[-1].status == "pending"
    assert ack.project("other-session", [old]) == [old]
    assert ack.active_for("s1") and not ack.active_for("other-session")
    assert ack.run_copy("s1") == "Sending…"

    echo = _message(Role.USER, "echo")
    assert ack.project("s1", [old, echo]) == [old, echo]
    assert not ack.active_for("s1")
    assert released == []  # the projection that saw the echo renders it


def test_acknowledgement_released_when_custody_ends_without_an_echo():
    ack, released = _ack()
    token = ack.begin("s1", DRAFT, [])
    callback = ack.custody_callback("s1")
    assert ack.custody_callback("other-session") is None
    ack.mark_admitted("s1")
    ack.dispatch_finished(token)  # admitted: the turn still owns the row
    assert ack.active_for("s1") and released == []
    callback(False)  # e.g. durable commit failed, turn refused
    assert not ack.active_for("s1") and released == [1]
    callback(False)
    assert released == [1]  # a stale token never releases a newer send


def test_acknowledgement_released_when_dispatch_admits_no_turn():
    ack, released = _ack()
    token = ack.begin("s1", DRAFT, [])
    ack.dispatch_finished(token)
    assert not ack.active_for("s1") and released == [1]
    newer = ack.begin("s1", DRAFT, [])
    ack.release(token)
    assert ack.active_for("s1")
    ack.release(newer)
    assert released == [1, 1]


def test_acknowledgement_marks_its_tab_running_but_keeps_approval():
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunMarker as Marker

    ack, _released = _ack()
    assert ack.overlay_run_markers({"s1": Marker.NONE}) == {"s1": Marker.NONE}
    ack.begin("s1", DRAFT, [])
    assert ack.overlay_run_markers({"s1": Marker.NONE}) == {"s1": Marker.RUNNING}
    assert ack.overlay_run_markers({"s1": Marker.NEEDS_APPROVAL}) == {
        "s1": Marker.NEEDS_APPROVAL
    }
    assert ack.overlay_run_markers(None) is None


def test_sending_presentation_replaces_the_empty_draft_reason():
    """AC#2 at the derivation: Sending refuses with the queue reason."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleControllerActivity
    from tldw_chatbook.Chat.console_display_state import (
        QUEUE_REASON_PREPARING,
        SEND_LABEL_SENDING,
        build_console_disabled_reason,
    )
    from tldw_chatbook.Chat.console_prompt_queue import ConsolePromptQueueRegistry
    from tldw_chatbook.UI.Console_Modules.prompt_queue import (
        derive_prompt_queue_presentation,
    )

    def activity(*, occupies: bool) -> ConsoleControllerActivity:
        return ConsoleControllerActivity(
            session_id="s1",
            occupies_slot=occupies,
            preparing_before_acceptance=occupies,
            accepted_live_turn=False,
            needs_approval=False,
            queued_count=0,
            queue_paused=False,
            terminal_notification_eligible=False,
        )

    empty = ConsolePromptQueueRegistry().snapshot("s1")
    idle = derive_prompt_queue_presentation(empty, activity(occupies=False))
    assert idle.send_label == "Send" and idle.send_enabled
    sending = derive_prompt_queue_presentation(
        empty, activity(occupies=False), sending=True
    )
    assert sending.send_label == SEND_LABEL_SENDING
    assert sending.send_enabled is False
    assert sending.send_tooltip == QUEUE_REASON_PREPARING
    validating = derive_prompt_queue_presentation(
        empty, activity(occupies=True), sending=True
    )
    assert validating.send_label == "Preparing..."
    reason = build_console_disabled_reason(
        action_id="send",
        has_draft=False,
        send_blocked=True,
        send_label=sending.send_label,
        queue_blocked_reason=sending.send_tooltip,
    )
    assert reason == QUEUE_REASON_PREPARING
