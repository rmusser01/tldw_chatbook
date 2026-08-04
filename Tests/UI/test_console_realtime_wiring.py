"""Console realtime (V4) hands-free loop -- screen wiring (task 5).

Tasks 1-4 are pure/headless and covered by their own suites
(`Tests/Chat/test_console_realtime_loop.py`, `Tests/Audio/
test_realtime_mic_tap.py`, `Tests/LLM_Calls/test_openai_realtime_session.py`).
This module covers the `ChatScreen` wiring that composes them into the
feature: the engine fork, the connect/seed/ready sequence, transcript
continuity in both directions, barge-in, the loud viable fallback, the
reconnect-once path and exit teardown. Harness mirrors
`Tests/UI/test_console_hands_free_wiring.py` exactly (`_build_test_app`,
`ConsoleHarness`, `_mounted_console`, fake injection via app attributes).

See `.superpowers/sdd/2026-08-04-realtime-voice-engine/task-5-brief.md`.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from Tests.UI.test_console_dictation_streaming import (
    FakeDictationService,
    _install_streaming_session,
    _patch_availability,
)
from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module

_ASYNC_SETTLE_TIMEOUT = 10.0


async def _wait_for(condition, pilot, *, timeout: float = _ASYNC_SETTLE_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause(0.02)
    raise AssertionError(f"condition never became true: {condition!r}")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRealtimeSession:
    """`RealtimeSession`-shaped double with test-driven callbacks.

    Records every outbound call in `calls` (ordered, so teardown ordering
    can be pinned across the tap/session/sink trio) and exposes `fire_*`
    helpers that trampoline the session's callbacks back onto the app's
    event loop -- the same thread affinity a real session's receive task
    gives them.
    """

    def __init__(
        self,
        config,
        callbacks,
        *,
        connect_error: Exception | None = None,
        connect_hangs: bool = False,
        order: list[str] | None = None,
    ) -> None:
        self.config = config
        self.callbacks = callbacks
        self._connect_error = connect_error
        self._connect_hangs = connect_hangs
        self._order = order if order is not None else []
        self.calls: list[str] = []
        self.seeds: list[tuple[list[tuple[str, str]], str | None]] = []
        self.text_items: list[tuple[str, bool]] = []
        self.cancels: list[int] = []
        self.audio_frames: list[bytes] = []
        self.connected = False
        self.closed = False
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- RealtimeSession protocol -----------------------------------------

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.calls.append("connect")
        if self._connect_error is not None:
            raise self._connect_error
        if self._connect_hangs:
            await asyncio.sleep(3600)
        self.connected = True

    def append_audio(self, frames: bytes) -> None:
        self.audio_frames.append(frames)

    def send_seed(self, items, instructions) -> None:
        self.calls.append("send_seed")
        self.seeds.append(([tuple(item) for item in items], instructions))

    def send_text_item(self, text: str, *, request_response: bool) -> None:
        self.calls.append("send_text_item")
        self.text_items.append((text, request_response))

    def cancel_response(self, played_ms: int) -> None:
        self.calls.append("cancel_response")
        self.cancels.append(played_ms)

    async def close(self) -> None:
        self.calls.append("close")
        self._order.append("session.close")
        self.closed = True

    # -- test drivers ------------------------------------------------------

    def _fire(self, name: str, *args: Any) -> None:
        callback = getattr(self.callbacks, name)
        assert callback is not None, f"wiring never registered {name}"
        loop = self._loop or asyncio.get_event_loop()
        loop.call_soon_threadsafe(callback, *args)

    def fire_ready(self) -> None:
        self._fire("on_ready")

    def fire_turn_committed(self) -> None:
        self._fire("on_turn_committed")

    def fire_input_transcript(self, text: str) -> None:
        self._fire("on_input_transcript", text)

    def fire_reply_started(self, item_id: str = "item-1") -> None:
        self._fire("on_reply_started", item_id)

    def fire_output_transcript_delta(self, text: str) -> None:
        self._fire("on_output_transcript_delta", text)

    def fire_audio_delta(self, pcm: bytes) -> None:
        self._fire("on_audio_delta", pcm)

    def fire_first_audio(self) -> None:
        self._fire("on_first_audio")

    def fire_reply_done(self) -> None:
        self._fire("on_reply_done")

    def fire_speech_started(self) -> None:
        self._fire("on_speech_started")

    def fire_usage(self, payload: dict) -> None:
        self._fire("on_usage", payload)

    def fire_closed(self, reason: str = "connection lost") -> None:
        self._fire("on_closed", reason)


class FakeRecorder:
    """Stands in for `AudioRecordingService` inside the REAL `RealtimeMicTap`.

    Keeping the real tap in the loop is deliberate: the first-words
    buffering guarantee under test is the tap's, and a fake tap would
    prove nothing about the wiring actually calling `mark_ready()`.
    """

    def __init__(self, order: list[str] | None = None, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._order = order if order is not None else []
        self.callback = None
        self.start_calls = 0
        self.stop_calls = 0

    def start_recording(self, callback) -> bool:
        self.callback = callback
        self.start_calls += 1
        return True

    def stop_recording(self) -> None:
        self.stop_calls += 1
        self._order.append("tap.stop")

    def push(self, frame: bytes) -> None:
        assert self.callback is not None, "recorder was never started"
        self.callback(frame)


class FakeSink:
    """`StreamingPcmSink`-shaped double exposing the surface `pump` uses."""

    def __init__(self, order: list[str] | None = None) -> None:
        self._order = order if order is not None else []
        self.opened: tuple[int, int] | None = None
        self.fed: list[bytes] = []
        self.state = "idle"
        self.terminal_reason = None
        self.fail_reason = None
        self.bytes_per_second = 48000
        self.buffered_seconds = 0.0

    def open(self, sample_rate: int, channels: int = 1) -> None:
        self.opened = (sample_rate, channels)
        self.state = "open"

    def feed(self, pcm: bytes) -> bool:
        self.fed.append(pcm)
        return True

    def close(self) -> None:
        if self.terminal_reason is None:
            self.state = "closed"
            self.terminal_reason = "drained"
            self._order.append("sink.close")

    def stop(self) -> None:
        if self.terminal_reason is None:
            self.state = "stopped"
            self.terminal_reason = "stopped"
            self._order.append("sink.stop")


# ---------------------------------------------------------------------------
# Harness helpers
# ---------------------------------------------------------------------------


class _RealtimeRig:
    """Everything a mounted realtime test needs to observe the wiring."""

    def __init__(self) -> None:
        self.sessions: list[FakeRealtimeSession] = []
        self.recorders: list[FakeRecorder] = []
        self.sinks: list[FakeSink] = []
        self.order: list[str] = []
        self.connect_error: Exception | None = None
        self.connect_hangs = False

    @property
    def session(self) -> FakeRealtimeSession:
        assert self.sessions, "no realtime session was ever built"
        return self.sessions[-1]

    @property
    def recorder(self) -> FakeRecorder:
        assert self.recorders, "the mic tap never built a recorder"
        return self.recorders[-1]

    @property
    def sink(self) -> FakeSink:
        assert self.sinks, "no audio sink was ever opened"
        return self.sinks[-1]


def _install_realtime_fakes(app) -> _RealtimeRig:
    rig = _RealtimeRig()

    def _session_factory(config, callbacks):
        session = FakeRealtimeSession(
            config,
            callbacks,
            connect_error=rig.connect_error,
            connect_hangs=rig.connect_hangs,
            order=rig.order,
        )
        rig.sessions.append(session)
        return session

    def _recorder_factory(**kwargs):
        recorder = FakeRecorder(order=rig.order, **kwargs)
        rig.recorders.append(recorder)
        return recorder

    def _sink_factory():
        sink = FakeSink(order=rig.order)
        rig.sinks.append(sink)
        return sink

    app.console_realtime_session_factory = _session_factory
    app.console_realtime_recorder_factory = _recorder_factory
    app.console_realtime_sink_factory = _sink_factory
    return rig


def _patch_realtime_config(
    monkeypatch,
    *,
    engine: str = "realtime",
    enabled: bool = True,
    provider: str = "openai",
    acoustic: bool = False,
    idle_timeout_seconds: float = 300.0,
) -> None:
    """Pin every config reader the engine fork consults."""
    monkeypatch.setattr(
        chat_screen_module, "resolve_handsfree_engine", lambda: engine
    )
    monkeypatch.setattr(chat_screen_module, "realtime_enabled", lambda: enabled)
    monkeypatch.setattr(chat_screen_module, "realtime_provider", lambda: provider)
    monkeypatch.setattr(chat_screen_module, "realtime_model", lambda: "gpt-realtime")
    monkeypatch.setattr(chat_screen_module, "realtime_voice", lambda: "marin")
    monkeypatch.setattr(
        chat_screen_module,
        "realtime_idle_timeout_seconds",
        lambda: idle_timeout_seconds,
    )
    monkeypatch.setattr(
        chat_screen_module, "acoustic_barge_in_enabled", lambda: acoustic
    )


def _capture_notifications(console) -> list[tuple[str, dict]]:
    notifications: list[tuple[str, dict]] = []
    console.app_instance.notify = lambda message, **kwargs: notifications.append(
        (message, kwargs)
    )
    return notifications


async def _enter_live_realtime(console, pilot, rig) -> FakeRealtimeSession:
    """Enter the loop, let it connect, and drive it to `live`."""
    console.action_toggle_console_hands_free()
    await _wait_for(lambda: bool(rig.sessions), pilot)
    await _wait_for(lambda: rig.session.connected, pilot)
    rig.session.fire_ready()
    await _wait_for(
        lambda: console._console_realtime is not None
        and console._console_realtime.controller.state == "live",
        pilot,
    )
    return rig.session


def _messages(console):
    store = console._ensure_console_chat_store()
    return store.messages_for_session(store.active_session_id)


# ---------------------------------------------------------------------------
# Rule 1: engine fork honesty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_with_realtime_disabled_leaves_the_pipeline_path_untouched(
    monkeypatch,
):
    """`resolve_handsfree_engine() == "pipeline"` must reach the V3 loop
    unchanged -- same session object, same store tap, no realtime state."""
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _patch_realtime_config(monkeypatch, engine="pipeline", enabled=False)
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console.action_toggle_console_hands_free()
        await pilot.pause()

        assert console._console_hands_free is not None
        assert console._console_realtime is None
        assert console._console_hands_free_store_tap_installed is True


@pytest.mark.asyncio
async def test_forced_realtime_while_unconfigured_refuses_with_a_toast(monkeypatch):
    """`handsfree_engine = "realtime"` with `[realtime] enabled = false`
    resolves to `"realtime"` on purpose (the reader never silently
    downgrades an explicit choice) -- so the wiring, not the reader, owes
    the user the refusal."""
    _patch_realtime_config(monkeypatch, engine="realtime", enabled=False)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        notifications = _capture_notifications(console)

        console.action_toggle_console_hands_free()
        await pilot.pause()

        assert console._console_realtime is None
        assert console._console_hands_free is None
        assert rig.sessions == []
        assert any(
            "realtime" in message.lower() and kwargs.get("severity") == "warning"
            for message, kwargs in notifications
        ), notifications


@pytest.mark.asyncio
async def test_unsupported_realtime_provider_refuses_with_a_toast(monkeypatch):
    """`[realtime] provider` is NOT validated by the config readers, so the
    engine fork is the only place a non-openai value can be refused."""
    _patch_realtime_config(monkeypatch, provider="anthropic")
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        notifications = _capture_notifications(console)

        console.action_toggle_console_hands_free()
        await pilot.pause()

        assert console._console_realtime is None
        assert rig.sessions == []
        assert any("anthropic" in message for message, _kwargs in notifications), (
            notifications
        )


@pytest.mark.asyncio
async def test_realtime_enabled_enters_the_realtime_loop_and_paints_connecting(
    monkeypatch,
):
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    rig.connect_hangs = True  # stay in `connecting` for the assertion
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(host, pilot)

        console.action_toggle_console_hands_free()
        await _wait_for(lambda: console._console_realtime is not None, pilot)

        assert console._console_hands_free is None
        assert console._console_realtime.controller.state == "connecting"
        await _wait_for(lambda: "connecting" in _visible_text(console), pilot)


# ---------------------------------------------------------------------------
# Rule 2: the V3 store tap is a pipeline-only mechanism
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_realtime_path_never_installs_the_v3_store_tap(monkeypatch):
    """The tap exists to observe a V3 provider run's store writes. The
    realtime engine writes those rows itself, so installing it would tax
    every message for nothing -- and would feed the V3 sequencer."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        await _enter_live_realtime(console, pilot, rig)

        assert console._console_hands_free_store_tap_installed is False


# ---------------------------------------------------------------------------
# Rule 3: connect -> ready -> seed -> live
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_ready_live_chip_sequence(monkeypatch):
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(host, pilot)

        console.action_toggle_console_hands_free()
        await _wait_for(lambda: bool(rig.sessions), pilot)
        assert console._console_realtime.controller.state == "connecting"
        # The mic tap opens immediately, BEFORE the handshake completes --
        # that is what makes the user's first words survive the connect.
        assert rig.recorders, "the mic tap was not started before the handshake"

        rig.session.fire_ready()
        await _wait_for(
            lambda: console._console_realtime.controller.state == "live", pilot
        )
        await _wait_for(lambda: "listening" in _visible_text(console), pilot)


@pytest.mark.asyncio
async def test_seed_sends_recent_turns_and_the_system_prompt_as_instructions(
    monkeypatch,
):
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        store.set_session_system_prompt(session_id, "You are terse.")
        for index in range(25):
            store.append_message(
                session_id,
                role=ConsoleMessageRole.USER
                if index % 2 == 0
                else ConsoleMessageRole.ASSISTANT,
                content=f"turn {index}",
            )

        await _enter_live_realtime(console, pilot, rig)

        items, instructions = rig.session.seeds[0]
        assert instructions == "You are terse."
        # Budget: the newest 20 turns only, still in transcript order.
        assert len(items) == 20
        assert items[0] == ("assistant", "turn 5")
        assert items[-1] == ("user", "turn 24")


@pytest.mark.asyncio
async def test_seed_char_budget_drops_the_oldest_turns(monkeypatch):
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        for index in range(6):
            store.append_message(
                session_id,
                role=ConsoleMessageRole.USER,
                content=f"{index}" * 3000,
            )

        await _enter_live_realtime(console, pilot, rig)

        # Six 3000-char turns: only the newest two fit under the 8000-char
        # ceiling (a third would be 9000), and they stay in transcript order.
        items, _instructions = rig.session.seeds[0]
        assert [text[:1] for _role, text in items] == ["4", "5"]
        assert sum(len(text) for _role, text in items) <= 8000


@pytest.mark.asyncio
async def test_first_words_buffer_until_ready_then_flush_in_order(monkeypatch):
    """Frames captured while the handshake is still in flight must reach
    the session AFTER `mark_ready()`, in arrival order -- the whole reason
    the tap starts before the connect completes."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console.action_toggle_console_hands_free()
        await _wait_for(lambda: bool(rig.recorders), pilot)

        rig.recorder.push(b"aa")
        rig.recorder.push(b"bb")

        await _wait_for(lambda: bool(rig.sessions) and rig.session.connected, pilot)
        # Connected, but the handshake has not been acknowledged yet: the
        # tap must still be holding those first frames.
        assert rig.session.audio_frames == []
        rig.session.fire_ready()
        await _wait_for(
            lambda: console._console_realtime.controller.state == "live", pilot
        )
        rig.recorder.push(b"cc")

        assert rig.session.audio_frames == [b"aa", b"bb", b"cc"]


# ---------------------------------------------------------------------------
# Rule 5: continuity out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_turn_commit_creates_the_user_row_before_the_reply_row(monkeypatch):
    """The user row is created at commit time -- not when the transcript
    finally arrives -- so it can never be ordered after the reply it
    prompted."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_turn_committed()
        await _wait_for(lambda: len(_messages(console)) == 1, pilot)
        session.fire_reply_started("item-1")
        await _wait_for(lambda: len(_messages(console)) == 2, pilot)
        # The transcript lands LAST, and still fills the row created first.
        session.fire_input_transcript("what is the weather")
        await _wait_for(
            lambda: _messages(console)[0].content == "what is the weather", pilot
        )

        rows = _messages(console)
        assert [row.role for row in rows] == [
            ConsoleMessageRole.USER,
            ConsoleMessageRole.ASSISTANT,
        ]


@pytest.mark.asyncio
async def test_assistant_transcript_streams_into_the_reply_row(monkeypatch):
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_turn_committed()
        session.fire_reply_started("item-1")
        session.fire_output_transcript_delta("Hello ")
        session.fire_output_transcript_delta("there.")
        session.fire_reply_done()
        await _wait_for(
            lambda: any(
                row.role is ConsoleMessageRole.ASSISTANT
                and row.content == "Hello there."
                for row in _messages(console)
            ),
            pilot,
        )

        assistant = [
            row for row in _messages(console) if row.role is ConsoleMessageRole.ASSISTANT
        ][0]
        assert assistant.status == "complete"


@pytest.mark.asyncio
async def test_barge_in_marks_the_reply_row_as_interrupted(monkeypatch):
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_turn_committed()
        session.fire_reply_started("item-1")
        session.fire_output_transcript_delta("Half a sen")
        await _wait_for(
            lambda: any(
                row.role is ConsoleMessageRole.ASSISTANT and row.content
                for row in _messages(console)
            ),
            pilot,
        )

        console._console_realtime.controller.on_keypress()
        await pilot.pause()

        assistant = [
            row for row in _messages(console) if row.role is ConsoleMessageRole.ASSISTANT
        ][0]
        assert assistant.content.endswith("interrupted"), assistant.content


@pytest.mark.asyncio
async def test_usage_is_attached_to_the_assistant_row(monkeypatch):
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)
        store = console._ensure_console_chat_store()

        session.fire_turn_committed()
        session.fire_reply_started("item-1")
        session.fire_output_transcript_delta("Hi.")
        await _wait_for(
            lambda: console._console_realtime.assistant_row_id is not None, pilot
        )
        row_id = console._console_realtime.assistant_row_id
        session.fire_usage({"input_tokens": 12, "output_tokens": 7})
        await _wait_for(lambda: store.get_message(row_id).usage is not None, pilot)

        usage = store.get_message(row_id).usage
        assert usage.output == 7


# ---------------------------------------------------------------------------
# Rules 6 + 7: audio out, barge-in, mic gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keypress_barge_in_stops_the_sink_and_cancels_with_played_ms(
    monkeypatch,
):
    """One second of 24 kHz mono PCM16 is 48000 bytes -> 1000 ms."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_turn_committed()
        session.fire_reply_started("item-1")
        session.fire_audio_delta(b"\x00" * 48000)
        session.fire_first_audio()
        await _wait_for(
            lambda: console._console_realtime.controller.state == "speaking", pilot
        )
        assert rig.sink.opened == (24000, 1)
        # Default (keyboard-only) barge-in mode gates the mic while a reply
        # is outstanding.
        await _wait_for(lambda: console._console_realtime.mic_gated is True, pilot)

        console._console_realtime.controller.on_keypress()
        await pilot.pause()

        assert rig.sink.terminal_reason == "stopped"
        assert session.cancels == [1000]
        assert console._console_realtime.controller.state == "live"
        assert console._console_realtime.mic_gated is False


@pytest.mark.asyncio
async def test_acoustic_mode_never_gates_the_mic(monkeypatch):
    _patch_realtime_config(monkeypatch, acoustic=True)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_turn_committed()
        session.fire_reply_started("item-1")
        session.fire_audio_delta(b"\x00" * 4800)
        session.fire_first_audio()
        await _wait_for(
            lambda: console._console_realtime.controller.state == "speaking", pilot
        )

        assert console._console_realtime.mic_gated is False
        rig.recorder.push(b"live frame")
        assert b"live frame" in session.audio_frames

        # Server-side VAD barges in, in this mode only.
        session.fire_speech_started()
        await _wait_for(
            lambda: console._console_realtime.controller.state == "live", pilot
        )
        assert session.cancels == [100]


# ---------------------------------------------------------------------------
# Rule 4: the loud viable fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_failure_falls_back_to_the_pipeline_loop_loudly(monkeypatch):
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    rig.connect_error = RuntimeError("handshake rejected")
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        notifications = _capture_notifications(console)

        console.action_toggle_console_hands_free()
        await _wait_for(lambda: console._console_hands_free is not None, pilot)

        assert console._console_realtime is None
        assert any(
            "handshake rejected" in message for message, _kwargs in notifications
        ), notifications
        # The tap is released when the realtime attempt is abandoned.
        assert rig.recorder.stop_calls == 1


@pytest.mark.asyncio
async def test_connect_failure_without_a_viable_pipeline_names_both_reasons(
    monkeypatch,
):
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    rig.connect_error = RuntimeError("handshake rejected")
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        notifications = _capture_notifications(console)
        # VAD-degraded: the pipeline loop cannot auto-send, which is exactly
        # the "not viable" case the fallback must refuse rather than paper
        # over.
        console._console_hands_free_vad_degraded = True

        console.action_toggle_console_hands_free()
        await _wait_for(lambda: bool(notifications), pilot)
        await pilot.pause(0.2)

        assert console._console_realtime is None
        assert console._console_hands_free is None
        joined = " ".join(message for message, _kwargs in notifications)
        assert "handshake rejected" in joined, joined
        assert "voice-activity" in joined or "auto-send" in joined, joined


@pytest.mark.asyncio
async def test_connect_timeout_is_bounded_and_falls_back(monkeypatch):
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _patch_realtime_config(monkeypatch)
    monkeypatch.setattr(
        chat_screen_module, "CONSOLE_REALTIME_CONNECT_TIMEOUT_SECONDS", 0.05
    )
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    rig.connect_hangs = True
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        notifications = _capture_notifications(console)

        console.action_toggle_console_hands_free()
        await _wait_for(lambda: console._console_hands_free is not None, pilot)

        assert console._console_realtime is None
        assert any("timed out" in message for message, _kwargs in notifications), (
            notifications
        )


# ---------------------------------------------------------------------------
# Rules 8 + 9: reasoned toasts and reconnect-once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transport_drop_reconnects_once_and_reseeds(monkeypatch):
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        notifications = _capture_notifications(console)
        session = await _enter_live_realtime(console, pilot, rig)
        store = console._ensure_console_chat_store()
        store.append_message(
            store.active_session_id,
            role=ConsoleMessageRole.USER,
            content="remember this",
        )

        session.fire_closed("connection lost")
        await _wait_for(lambda: len(rig.sessions) == 2, pilot)
        assert console._console_realtime.controller.state == "reconnecting"
        assert any("reconnect" in message.lower() for message, _kw in notifications)

        await _wait_for(lambda: rig.sessions[1].connected, pilot)
        rig.sessions[1].fire_ready()
        await _wait_for(
            lambda: console._console_realtime.controller.state == "live", pilot
        )

        items, _instructions = rig.sessions[1].seeds[0]
        assert ("user", "remember this") in items

        # A SECOND drop within the same loop entry gives up outright.
        rig.sessions[1].fire_closed("connection lost")
        await _wait_for(lambda: console._console_realtime is None, pilot)
        assert any(
            "connection lost" in message for message, _kwargs in notifications
        ), notifications


@pytest.mark.asyncio
async def test_idle_timeout_exits_with_a_reasoned_toast(monkeypatch):
    _patch_realtime_config(monkeypatch, idle_timeout_seconds=120.0)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        notifications = _capture_notifications(console)
        await _enter_live_realtime(console, pilot, rig)

        # Drive the controller's injected clock rather than waiting out a
        # real two-minute ceiling.
        controller = console._console_realtime.controller
        now = time.monotonic()
        controller.tick(now)
        controller.tick(now + 121.0)
        await pilot.pause()

        assert console._console_realtime is None
        assert any(
            "idle for 2 minutes" in message for message, _kwargs in notifications
        ), notifications


# ---------------------------------------------------------------------------
# Rules 10 + 11: adopted capture and exit teardown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adopted_capture_sends_its_transcript_as_a_text_turn(monkeypatch):
    """Spoken "Console, hands free." mid-capture: the open pipeline capture
    is stopped and transcribed through the existing V2 path, and the
    transcript becomes the realtime session's first turn."""
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )

        await pilot.click("#console-dictation")
        await _wait_for(
            lambda: console._console_dictation_state == "recording", pilot
        )
        service.emit_final("what is the capital of france")
        await pilot.pause()

        console.action_toggle_console_hands_free()
        await _wait_for(lambda: bool(rig.sessions), pilot)
        await _wait_for(lambda: rig.session.connected, pilot)
        rig.session.fire_ready()

        await _wait_for(lambda: bool(rig.session.text_items), pilot)
        assert rig.session.text_items == [
            ("what is the capital of france", True)
        ]
        # The adopted transcript became the turn itself, not a stray draft.
        assert composer.draft_text.strip() == ""
        assert any(
            row.role is ConsoleMessageRole.USER
            and row.content == "what is the capital of france"
            for row in _messages(console)
        )


@pytest.mark.asyncio
async def test_exit_tears_down_tap_then_session_then_sink(monkeypatch):
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_turn_committed()
        session.fire_reply_started("item-1")
        session.fire_audio_delta(b"\x00" * 480)
        await _wait_for(lambda: bool(rig.sinks), pilot)

        console._console_realtime.controller.on_exit_request()
        await _wait_for(lambda: session.closed, pilot)

        assert console._console_realtime is None
        assert rig.recorder.stop_calls == 1
        assert rig.order[0] == "tap.stop"
        assert rig.order.index("session.close") < rig.order.index("sink.stop")


@pytest.mark.asyncio
async def test_unmount_abandons_the_realtime_loop(monkeypatch):
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)
        await console.on_unmount()

        assert console._console_realtime is None
        assert rig.recorder.stop_calls == 1
        assert session.closed is True
