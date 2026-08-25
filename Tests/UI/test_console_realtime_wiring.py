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
import logging
import math
import threading
import time
from contextlib import contextmanager
from typing import Any

import pytest
from loguru import logger

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
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
from tldw_chatbook.UI.Console_Modules import hands_free as hands_free_module
from tldw_chatbook.UI.Console_Modules import realtime as realtime_module
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

_ASYNC_SETTLE_TIMEOUT = 10.0


async def _wait_for(
    condition, pilot, *, timeout: float = _ASYNC_SETTLE_TIMEOUT
) -> None:
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
        # The REAL session fires `on_closed` from its recv loop's `finally`,
        # so a deliberate close produces one too (final review M1). Firing
        # it here is what puts the wiring's attempt-staleness guard under
        # test at all -- without it, a reconnect never delivered the old
        # session's close and the guard could be deleted with every test
        # still green.
        self._fire("on_closed", "closed by client")

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

    def fire_transcription_usage(self, payload: dict) -> None:
        self._fire("on_transcription_usage", payload)

    def fire_closed(self, reason: str = "connection lost") -> None:
        self._fire("on_closed", reason)

    def fire_error(self, exc: Exception) -> None:
        self._fire("on_error", exc)


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
        self.stop_thread_ident: int | None = None

    def start_recording(self, callback) -> bool:
        self.callback = callback
        self.start_calls += 1
        return True

    def stop_recording(self) -> None:
        self.stop_calls += 1
        #: Which thread the (blocking, recorder-joining) stop ran on --
        #: pinned by the teardown test, since running it on the UI thread
        #: freezes the app for as long as the join takes.
        self.stop_thread_ident = threading.get_ident()
        self._order.append("tap.stop")

    def push(self, frame: bytes) -> None:
        assert self.callback is not None, "recorder was never started"
        self.callback(frame)


class FakeSink:
    """`StreamingPcmSink`-shaped double exposing the surface `pump` uses.

    `close()` starts DRAINING and stays non-terminal until
    `finish_playback()` -- exactly like the real sink, whose drain ends
    when the device callback has actually played the buffered audio out.
    That gap is the whole subject of the generation-done vs playback-done
    distinction, so a fake that collapsed it (terminal on close) could
    never test it. `buffered_seconds` is deliberately generous so `pump`'s
    own drain-wait deadline cannot expire mid-test on a loaded machine.
    """

    def __init__(self, order: list[str] | None = None) -> None:
        self._order = order if order is not None else []
        self.opened: tuple[int, int] | None = None
        self.fed: list[bytes] = []
        self.state = "idle"
        self.terminal_reason = None
        self.fail_reason = None
        self.bytes_per_second = 48000
        self.buffered_seconds = 30.0
        self.settled = 0

    def open(self, sample_rate: int, channels: int = 1) -> None:
        self.opened = (sample_rate, channels)
        self.state = "open"

    def feed(self, pcm: bytes) -> bool:
        self.fed.append(pcm)
        return True

    def close(self) -> None:
        if self.terminal_reason is None and self.state != "draining":
            self.state = "draining"
            self._order.append("sink.close")

    def finish_playback(self) -> None:
        """The device finished playing everything buffered."""
        if self.terminal_reason is None:
            self.state = "closed"
            self.terminal_reason = "drained"

    def stop(self) -> None:
        if self.terminal_reason is None:
            self.state = "stopped"
            self.terminal_reason = "stopped"
            self._order.append("sink.stop")

    def settle(self, timeout: float = 5.0) -> bool:
        self.settled += 1
        return True


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
    api_key: str = "test-realtime-key",
) -> None:
    """Pin every config reader the engine fork consults.

    `api_key` included: the wiring refuses to dispatch a connect without
    one (there is nothing to authenticate with), so every mounted test
    needs a configured key even though the injected session never uses it.

    `resolve_handsfree_engine`/`realtime_enabled` are pinned on `hands_free_
    module`, not `realtime_module`: the engine fork that reads them moved
    to `ConsoleHandsFreeController` (wave-2 console decomposition, task 1),
    and it holds its own separate copy of this import -- pinning only
    `realtime_module`'s copy would silently stop
    governing the fork's actual behaviour. `acoustic_barge_in_enabled` is
    pinned on BOTH modules: the realtime engine's own `_enter_console_
    realtime_loop` reads `realtime_module`'s copy, and a pipeline
    fallback (`_console_realtime_fallback_to_pipeline` -> `Console
    HandsFreeController._enter_console_hands_free_pipeline_loop`) reads
    `hands_free_module`'s separate copy -- a test exercising the fallback
    path needs both to agree.
    """
    monkeypatch.setattr(realtime_module, "get_api_key", lambda _name: api_key)
    monkeypatch.setattr(hands_free_module, "resolve_handsfree_engine", lambda: engine)
    monkeypatch.setattr(hands_free_module, "realtime_enabled", lambda: enabled)
    monkeypatch.setattr(realtime_module, "realtime_provider", lambda: provider)
    monkeypatch.setattr(realtime_module, "realtime_model", lambda: "gpt-realtime")
    monkeypatch.setattr(realtime_module, "realtime_voice", lambda: "marin")
    monkeypatch.setattr(
        realtime_module,
        "realtime_idle_timeout_seconds",
        lambda: idle_timeout_seconds,
    )
    monkeypatch.setattr(realtime_module, "acoustic_barge_in_enabled", lambda: acoustic)
    monkeypatch.setattr(
        hands_free_module, "acoustic_barge_in_enabled", lambda: acoustic
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
        lambda: (
            console._console_realtime is not None
            and console._console_realtime.controller.state == "live"
        ),
        pilot,
    )
    return rig.session


@pytest.mark.asyncio
async def test_persona_buddy_realtime_fsm_replaces_generation_and_releases_on_exit(
    monkeypatch,
):
    """Mounted realtime callbacks drive Buddy and release on exact loop exit."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    app.persona_buddy_controller = PersonaBuddyController()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console.action_toggle_console_hands_free()
        await _wait_for(lambda: console._console_realtime is not None, pilot)
        assert app.persona_buddy_controller.snapshot().state == "offline"

        await _wait_for(lambda: bool(rig.sessions), pilot)
        session = rig.session
        session.fire_ready()
        await _wait_for(
            lambda: app.persona_buddy_controller.snapshot().state == "listening",
            pilot,
        )
        session.fire_turn_committed()
        await _wait_for(
            lambda: app.persona_buddy_controller.snapshot().state == "thinking",
            pilot,
        )
        session.fire_reply_started()
        session.fire_first_audio()
        await _wait_for(
            lambda: app.persona_buddy_controller.snapshot().state == "speaking",
            pilot,
        )

        generation = console._console_realtime.buddy_generation
        console._realtime._console_realtime_exit_loop(None)
        await _wait_for(
            lambda: app.persona_buddy_controller.snapshot().state == "idle", pilot
        )
        console.action_toggle_console_hands_free()
        await _wait_for(lambda: console._console_realtime is not None, pilot)
        assert console._console_realtime.buddy_generation > generation
        assert app.persona_buddy_controller.snapshot().state == "offline"

    assert app.persona_buddy_controller.snapshot().state == "idle"


@pytest.mark.asyncio
async def test_persona_buddy_realtime_generation_survives_screen_replacement(
    monkeypatch,
):
    """A stale real screen cannot release its successor's same-session loop."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    app.persona_buddy_controller = PersonaBuddyController()
    _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        old_screen = await _mounted_console(host, pilot)
        old_screen.action_toggle_console_hands_free()
        await _wait_for(lambda: old_screen._console_realtime is not None, pilot)
        old_generation = old_screen._console_realtime.buddy_generation

        replacement = chat_screen_module.ChatScreen(app)
        await host.push_screen(replacement)
        await _wait_for(
            lambda: replacement.query("#console-native-composer").first() is not None,
            pilot,
        )
        replacement.action_toggle_console_hands_free()
        await _wait_for(lambda: replacement._console_realtime is not None, pilot)
        replacement_generation = replacement._console_realtime.buddy_generation

        assert replacement_generation > old_generation
        assert app.persona_buddy_controller.snapshot().state == "offline"

        old_screen._realtime._release_console_realtime_state()
        assert app.persona_buddy_controller.snapshot().state == "offline"
        assert (
            replacement._console_runtime().persona_buddy_sink.active_owner_count(
                "voice"
            )
            == 1
        )

        replacement._realtime._console_realtime_exit_loop(None)
        await _wait_for(
            lambda: app.persona_buddy_controller.snapshot().state == "idle", pilot
        )


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
async def test_turn_detection_settings_reach_the_session_on_connect_and_reconnect(
    monkeypatch,
):
    """Gate round 5: the turn-detection knobs are the fix for speech being
    chopped into fragments, so they have to reach the provider on EVERY
    connect -- a reconnect that silently reverted to the default would
    bring the symptom back mid-conversation."""
    _patch_realtime_config(monkeypatch)
    monkeypatch.setattr(
        realtime_module, "realtime_turn_detection", lambda: "server_vad"
    )
    monkeypatch.setattr(realtime_module, "realtime_vad_threshold", lambda: 0.6)
    monkeypatch.setattr(realtime_module, "realtime_vad_silence_ms", lambda: 700)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        first = rig.sessions[0].config
        assert first.turn_detection == "server_vad"
        assert first.vad_threshold == 0.6
        assert first.vad_silence_ms == 700

        session.fire_closed("connection lost")
        await _wait_for(lambda: len(rig.sessions) == 2, pilot)

        second = rig.sessions[1].config
        assert second.turn_detection == "server_vad"
        assert second.vad_threshold == 0.6
        assert second.vad_silence_ms == 700


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
async def test_seed_skips_an_oversized_turn_instead_of_ending_the_seed(monkeypatch):
    """F6: one long newest reply must not ship ZERO history. The budget
    walk skips what cannot fit and keeps taking older turns that can."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        store.append_message(
            session_id, role=ConsoleMessageRole.USER, content="short older turn"
        )
        store.append_message(
            session_id, role=ConsoleMessageRole.ASSISTANT, content="A" * 9000
        )

        await _enter_live_realtime(console, pilot, rig)

        items, _instructions = rig.session.seeds[0]
        assert items == [("user", "short older turn")]


@pytest.mark.asyncio
async def test_seed_strips_the_interrupted_ui_marker(monkeypatch):
    """M4: `⏹ interrupted` is OUR chrome for the human reader. Replaying it
    into the model's context teaches it that the marker is part of how the
    assistant talks."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        store.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content=(
                "Half a sentence" + realtime_module.CONSOLE_REALTIME_INTERRUPTED_MARKER
            ),
        )

        await _enter_live_realtime(console, pilot, rig)

        items, _instructions = rig.session.seeds[0]
        assert items == [("assistant", "Half a sentence")]


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
async def test_a_late_input_transcript_never_overwrites_the_next_turn(monkeypatch):
    """F5: `on_input_transcript` carries no item id, so a transcript that
    arrives after the NEXT turn committed would land in that turn's row.
    A filled row is never overwritten -- the stale text is dropped, loudly
    enough to diagnose."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)
    warnings: list[str] = []
    sink_id = logger.add(lambda message: warnings.append(str(message)), level="WARNING")

    try:
        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)
            session = await _enter_live_realtime(console, pilot, rig)

            session.fire_turn_committed()
            await _wait_for(lambda: len(_messages(console)) == 1, pilot)
            first_row_id = console._console_realtime.user_row_id
            session.fire_turn_committed()
            await _wait_for(lambda: len(_messages(console)) == 2, pilot)
            second_row_id = console._console_realtime.user_row_id

            session.fire_input_transcript("turn two")
            await _wait_for(lambda: _messages(console)[1].content == "turn two", pilot)
            # Turn one's transcript finally arrives -- far too late.
            session.fire_input_transcript("turn one")
            await pilot.pause()
            await pilot.pause()

            store = console._ensure_console_chat_store()
            assert store.get_message(second_row_id).content == "turn two"
            assert store.get_message(first_row_id).content == ""
    finally:
        logger.remove(sink_id)

    assert any(
        "realtime_input_transcript" in message and second_row_id in message
        for message in warnings
    ), warnings


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
            row
            for row in _messages(console)
            if row.role is ConsoleMessageRole.ASSISTANT
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
            row
            for row in _messages(console)
            if row.role is ConsoleMessageRole.ASSISTANT
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


@pytest.mark.asyncio
async def test_realtime_usage_records_cached_input_tokens(monkeypatch):
    """F9: the Realtime API spells the details key `input_token_details`
    (SINGULAR). Unrecognized, every cached token was billed as uncached."""
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
        session.fire_usage(
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_token_details": {"cached_tokens": 80},
            }
        )
        await _wait_for(lambda: store.get_message(row_id).usage is not None, pilot)

        usage = store.get_message(row_id).usage
        assert usage.cache_read == 80
        assert usage.uncached_input == 20
        assert usage.output == 20


@pytest.mark.asyncio
async def test_realtime_usage_records_the_audio_token_split(monkeypatch):
    """task-2363: realtime `response.done` usage splits BOTH input and
    output tokens into text/audio -- live-confirmed, see openai_session.py's
    ground-truth header USAGE section. Previously folded into the plain
    uncached/output buckets with no distinct audio count at all."""
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
        session.fire_usage(
            {
                "total_tokens": 151,
                "input_tokens": 33,
                "output_tokens": 118,
                "input_token_details": {
                    "text_tokens": 15,
                    "audio_tokens": 18,
                    "image_tokens": 0,
                    "cached_tokens": 0,
                    "cached_tokens_details": {
                        "text_tokens": 0,
                        "audio_tokens": 0,
                        "image_tokens": 0,
                    },
                },
                "output_token_details": {"text_tokens": 28, "audio_tokens": 90},
            }
        )
        await _wait_for(lambda: store.get_message(row_id).usage is not None, pilot)

        usage = store.get_message(row_id).usage
        assert usage.uncached_input == 33
        assert usage.output == 118
        assert usage.audio_input == 18
        assert usage.audio_output == 90


@pytest.mark.asyncio
async def test_transcription_usage_attaches_duration_to_the_user_row(monkeypatch):
    """task-2363 / T2-F12: the input-audio transcription's OWN `usage`
    field (`{"type": "duration", "seconds": N}`) is about the USER's spoken
    turn, not the assistant's reply -- it must land on `user_row_id`, not
    `last_reply_row_id` (the target `on_usage` uses)."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)
        store = console._ensure_console_chat_store()

        session.fire_turn_committed()
        await _wait_for(
            lambda: console._console_realtime.user_row_id is not None, pilot
        )
        row_id = console._console_realtime.user_row_id
        session.fire_transcription_usage({"type": "duration", "seconds": 2})
        await _wait_for(lambda: store.get_message(row_id).usage is not None, pilot)

        usage = store.get_message(row_id).usage
        assert usage.transcription_seconds == 2.0


@pytest.mark.asyncio
async def test_a_late_transcription_usage_never_overwrites_the_next_turn(monkeypatch):
    """Mirrors `test_a_late_input_transcript_never_overwrites_the_next_turn`
    for usage: a duration payload that lands after the NEXT turn committed
    (and already got its own usage) must not clobber it."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)
        store = console._ensure_console_chat_store()

        session.fire_turn_committed()
        await _wait_for(lambda: len(_messages(console)) == 1, pilot)
        first_row_id = console._console_realtime.user_row_id
        session.fire_turn_committed()
        await _wait_for(lambda: len(_messages(console)) == 2, pilot)
        second_row_id = console._console_realtime.user_row_id

        session.fire_transcription_usage({"type": "duration", "seconds": 3})
        await _wait_for(
            lambda: store.get_message(second_row_id).usage is not None, pilot
        )
        # Turn one's duration usage finally arrives -- far too late.
        session.fire_transcription_usage({"type": "duration", "seconds": 1})
        await pilot.pause()
        await pilot.pause()

        assert store.get_message(second_row_id).usage.transcription_seconds == 3.0
        assert store.get_message(first_row_id).usage is None


@pytest.mark.asyncio
async def test_a_failed_audio_sink_is_latched_not_retried_per_delta(monkeypatch):
    """F2: a sink that cannot open must fail ONCE per reply. Retrying per
    audio delta means one construction (and one traceback) per ~20 ms of
    speech -- a traceback storm on the UI thread, for a device that is not
    coming back mid-reply."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    builds: list[int] = []

    def _failing_sink_factory():
        builds.append(1)
        raise RuntimeError("no audio device")

    app.console_realtime_sink_factory = _failing_sink_factory
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        notifications = _capture_notifications(console)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_turn_committed()
        session.fire_reply_started("item-1")
        for _ in range(12):
            session.fire_audio_delta(b"\x00" * 480)
        await pilot.pause()
        await pilot.pause()

        assert len(builds) == 1, f"sink construction retried per delta: {len(builds)}"
        assert (
            sum(1 for message, _kwargs in notifications if "audio" in message.lower())
            == 1
        ), notifications

        # A second reply is allowed one fresh attempt -- the device may have
        # come back between replies.
        session.fire_reply_done()
        session.fire_reply_started("item-2")
        session.fire_audio_delta(b"\x00" * 480)
        await pilot.pause()
        assert len(builds) == 2


# ---------------------------------------------------------------------------
# Rules 6 + 7: audio out, barge-in, mic gating
# ---------------------------------------------------------------------------


async def _drive_to_speaking(console, pilot, session, *, audio: bytes) -> None:
    """Take a live loop through commit -> reply -> audio, into `speaking`."""
    session.fire_turn_committed()
    session.fire_reply_started("item-1")
    session.fire_audio_delta(audio)
    session.fire_first_audio()
    await _wait_for(
        lambda: console._console_realtime.controller.state == "speaking", pilot
    )


@pytest.mark.asyncio
async def test_keypress_barge_in_stops_the_sink_and_cancels_with_played_ms(
    monkeypatch,
):
    """One second of 24 kHz mono PCM16 is 48000 bytes -> 1000 ms.

    Fix round 1 (F1): driven through a REAL `pilot.press`, on the ready
    console harness, rather than by calling `controller.on_keypress()`
    directly. The bare-`_build_test_app` harness sits behind the first-run
    setup modal, where `on_key` returns before the hands-free branch is
    ever reached -- so a direct controller call proved the FSM worked while
    leaving the actual key routing (modal gate, focus gate ordering,
    `event.stop`) completely uncovered.
    """
    _patch_realtime_config(monkeypatch)
    app, host = _ready_host()
    rig = _install_realtime_fakes(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        await _drive_to_speaking(console, pilot, session, audio=b"\x00" * 48000)
        assert rig.sink.opened == (24000, 1)
        # Default (keyboard-only) barge-in mode gates the mic while a reply
        # is outstanding -- asserted on the TAP's real behavior (a pushed
        # frame is dropped), not just on the wiring's own bookkeeping.
        await _wait_for(lambda: console._console_realtime.mic_gated is True, pilot)
        rig.recorder.push(b"while speaking")
        assert b"while speaking" not in session.audio_frames

        await pilot.press("x")
        await pilot.pause()

        assert rig.sink.terminal_reason == "stopped"
        assert session.cancels == [1000]
        assert console._console_realtime.controller.state == "live"
        assert console._console_realtime.mic_gated is False
        rig.recorder.push(b"after barge in")
        assert b"after barge in" in session.audio_frames


@pytest.mark.asyncio
async def test_realtime_barge_in_and_esc_work_with_focus_off_the_composer(
    monkeypatch,
):
    """The realtime engine inherits V3's promise: "press any key" / "Esc
    from any point in the loop" must hold with focus on the transcript
    (clicked or scrolled), not only on the composer."""
    _patch_realtime_config(monkeypatch)
    app, host = _ready_host()
    rig = _install_realtime_fakes(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )
        session = await _enter_live_realtime(console, pilot, rig)

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.focus()
        await pilot.pause()
        assert console.app.focused is transcript
        assert console._should_capture_console_input(composer) is False

        await _drive_to_speaking(console, pilot, session, audio=b"\x00" * 4800)
        await pilot.press("x")
        await pilot.pause()
        assert console._console_realtime.controller.state == "live", (
            "barge-in did nothing with focus off the composer"
        )
        assert session.cancels == [100]

        transcript.focus()
        await pilot.pause()
        await pilot.press("escape")
        await _wait_for(lambda: console._console_realtime is None, pilot)


@pytest.mark.asyncio
async def test_esc_exits_the_realtime_loop_and_the_action_gate_follows_it(
    monkeypatch,
):
    """F8: the priority-Esc action and its `check_action` gate must know
    about the realtime loop, not just the V3 one."""
    _patch_realtime_config(monkeypatch)
    app, host = _ready_host()
    rig = _install_realtime_fakes(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        await _enter_live_realtime(console, pilot, rig)

        assert console.check_action("exit_console_hands_free", ()) is True

        await pilot.press("escape")
        await _wait_for(lambda: console._console_realtime is None, pilot)

        assert console.check_action("exit_console_hands_free", ()) is False


@pytest.mark.asyncio
async def test_mic_button_exits_the_loop_and_opens_no_second_microphone(monkeypatch):
    """CRITICAL (final review C1): the mic button must exit the realtime
    loop, exactly like Esc and the toggle.

    Falling through to the ordinary dictation toggle would open a SECOND
    `AudioRecordingService` at 16 kHz alongside the realtime tap, load the
    whole STT stack the realtime engine exists to avoid, and arm the V2
    spoken-command classifier mid-session -- while the realtime session
    kept billing.
    """
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _patch_realtime_config(monkeypatch)
    app, host = _ready_host()
    rig = _install_realtime_fakes(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        await pilot.click("#console-dictation")
        await _wait_for(lambda: console._console_realtime is None, pilot)
        await pilot.pause(0.2)

        assert console._console_dictation_state == "idle"
        assert service.start_calls == 0, "a second capture stack was opened"
        assert len(rig.recorders) == 1, "a second recorder was constructed"
        await _wait_for(lambda: session.closed, pilot)


@pytest.mark.asyncio
async def test_toggle_exits_a_running_realtime_loop(monkeypatch):
    """F8: `ctrl+shift+h` is a toggle for BOTH engines."""
    _patch_realtime_config(monkeypatch)
    app, host = _ready_host()
    rig = _install_realtime_fakes(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        console.action_toggle_console_hands_free()
        await _wait_for(lambda: console._console_realtime is None, pilot)

        assert len(rig.sessions) == 1, "the toggle started a second loop"
        await _wait_for(lambda: session.closed, pilot)


@pytest.mark.asyncio
async def test_exit_restores_the_ordinary_voice_chip(monkeypatch):
    """F8: the realtime chip is borrowed; exiting must give it back rather
    than leaving `realtime · listening` painted over an idle composer."""
    _patch_realtime_config(monkeypatch)
    app, host = _ready_host()
    rig = _install_realtime_fakes(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(host, pilot)
        await _enter_live_realtime(console, pilot, rig)
        await _wait_for(lambda: "realtime" in _visible_text(console), pilot)

        console._console_realtime.controller.on_exit_request()
        await _wait_for(lambda: console._console_realtime is None, pilot)
        await pilot.pause()

        assert "realtime" not in _visible_text(console), _visible_text(console)


@pytest.mark.asyncio
async def test_reply_transcript_is_actually_repainted_mid_reply(monkeypatch):
    """F8: the store write is not the point -- the user seeing it is. Pins
    the repaint cadence with a token no chrome could supply."""
    _patch_realtime_config(monkeypatch)
    app, host = _ready_host()
    rig = _install_realtime_fakes(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_turn_committed()
        session.fire_reply_started("item-1")
        session.fire_output_transcript_delta("ZEBRAFISH")

        # No reply_done: the repaint must happen WHILE the reply streams.
        await _wait_for(lambda: "ZEBRAFISH" in _visible_text(console), pilot)


def _spy_on_reply_done(session_state) -> list[float]:
    """Record every `on_reply_done(now)` the wiring hands the FSM."""
    controller = session_state.controller
    original = controller.on_reply_done
    calls: list[float] = []

    def _spy(now: float) -> None:
        calls.append(now)
        original(now)

    controller.on_reply_done = _spy
    return calls


@pytest.mark.asyncio
async def test_mic_stays_gated_until_the_reply_audio_finishes_playing(monkeypatch):
    """LIVE GATE: in default (speaker-safe) mode the model heard ITSELF and
    replied to its own voice.

    `response.done` means GENERATION finished, and 24 kHz audio generates
    far faster than it plays -- so the sink still held seconds of the
    reply. Handing that straight to `controller.on_reply_done` left
    `speaking` early, ungated the tap into the reply's own audible tail,
    and the provider's server-side VAD committed the model's voice as a
    user turn. `on_reply_done` to the FSM must mean "including playback".
    """
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        await _drive_to_speaking(console, pilot, session, audio=b"\x00" * 4800)
        assert console._console_realtime.mic_gated is True

        # Generation finishes; the audio is still draining.
        session.fire_reply_done()
        await pilot.pause()
        await pilot.pause()

        assert console._console_realtime.controller.state == "speaking"
        assert console._console_realtime.mic_gated is True
        rig.recorder.push(b"the model's own tail")
        assert b"the model's own tail" not in session.audio_frames, (
            "the mic reopened into the reply's audible tail -- the model "
            "will hear itself"
        )

        # Playback actually completes.
        rig.sink.finish_playback()
        await _wait_for(
            lambda: console._console_realtime.controller.state == "live", pilot
        )
        assert console._console_realtime.mic_gated is False
        rig.recorder.push(b"the user's next turn")
        assert b"the user's next turn" in session.audio_frames


@pytest.mark.asyncio
async def test_reply_with_no_audio_completes_immediately(monkeypatch):
    """A reply that produced no audio has no playback to wait for --
    deferring it would hang the loop in `speaking` forever."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_turn_committed()
        session.fire_reply_started("item-1")
        session.fire_output_transcript_delta("text only, no audio")
        await _wait_for(
            lambda: console._console_realtime.controller.state == "thinking", pilot
        )

        session.fire_reply_done()
        await _wait_for(
            lambda: console._console_realtime.controller.state == "live", pilot
        )
        assert rig.sinks == [], "a sink was opened for a reply with no audio"


@pytest.mark.asyncio
async def test_barge_in_mid_drain_fires_no_late_reply_done(monkeypatch):
    """Task 2's semantics, mirrored: a cancelled reply completes nothing.
    The aborted pump's completion must not report a reply the user cut."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)
        calls = _spy_on_reply_done(console._console_realtime)

        await _drive_to_speaking(console, pilot, session, audio=b"\x00" * 4800)
        session.fire_reply_done()  # generation done, audio still draining
        await pilot.pause()
        assert console._console_realtime.controller.state == "speaking"

        console._console_realtime.controller.on_keypress()
        await pilot.pause()
        assert console._console_realtime.controller.state == "live"

        # The aborted pump now unwinds.
        await pilot.pause(0.2)
        assert calls == [], "a barged reply reported itself finished"
        assert console._console_realtime.controller.state == "live"


@pytest.mark.asyncio
async def test_a_stale_playback_completion_never_ends_the_next_reply(monkeypatch):
    """The identity guard, driven directly.

    Today the barge-in latch also covers the only reachable ordering, so
    this is pinned at the seam (the same idiom the V3 suite uses for its
    deferred capture-ended delivery) rather than through a race the fakes
    cannot stage deterministically. Without it, a completion from the
    PREVIOUS reply would report the CURRENT one finished -- ungating the
    mic into a reply that is still speaking, which is the exact defect
    this whole change exists to prevent.
    """
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)
        state = console._console_realtime

        await _drive_to_speaking(console, pilot, session, audio=b"\x00" * 4800)
        stale_token = state.reply_token - 1

        console._realtime._console_realtime_playback_finished(state, stale_token)
        await pilot.pause()

        assert state.controller.state == "speaking"
        assert state.playback_pending is True


@pytest.mark.asyncio
async def test_reply_done_is_stamped_at_playback_end_not_generation_end(monkeypatch):
    """The idle ceiling anchors on this `now`. Stamping it at generation
    end would charge a long drain against the idle budget."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)
        calls = _spy_on_reply_done(console._console_realtime)

        await _drive_to_speaking(console, pilot, session, audio=b"\x00" * 4800)
        session.fire_reply_done()
        await pilot.pause()
        assert calls == []

        drain_ended_after = time.monotonic()
        rig.sink.finish_playback()
        await _wait_for(lambda: bool(calls), pilot)

        assert calls[0] >= drain_ended_after


@pytest.mark.asyncio
async def test_acoustic_mode_never_gates_the_mic(monkeypatch):
    _patch_realtime_config(monkeypatch, acoustic=True)
    app, host = _ready_host()
    rig = _install_realtime_fakes(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        await _drive_to_speaking(console, pilot, session, audio=b"\x00" * 4800)

        assert console._console_realtime.mic_gated is False
        rig.recorder.push(b"live frame")
        assert b"live frame" in session.audio_frames

        # Server-side VAD barges in, in this mode only.
        with _realtime_diagnostics() as capture:
            session.fire_speech_started()
            await _wait_for(
                lambda: console._console_realtime.controller.state == "live", pilot
            )
        assert session.cancels == [100]
        # The OTHER trigger, recorded as itself: a barge report that could
        # not tell VAD from a keypress would be useless in acoustic mode,
        # where both are live.
        assert _fields_for(capture, "realtime_barge")["initiator"] == "speech"


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
async def test_missing_api_key_refuses_before_any_connect_attempt(monkeypatch):
    """There is nothing to authenticate with, so the connect is never
    dispatched -- and the refusal names the real cause instead of quoting
    whatever 401 the provider would have sent back."""
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _patch_realtime_config(monkeypatch, api_key="")
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        notifications = _capture_notifications(console)

        console.action_toggle_console_hands_free()
        await _wait_for(lambda: console._console_hands_free is not None, pilot)

        assert rig.sessions == [], "a connect was attempted with no API key"
        assert console._console_realtime is None
        assert any("API key" in message for message, _kwargs in notifications), (
            notifications
        )


@pytest.mark.asyncio
async def test_double_failure_toast_carries_the_install_remedy(monkeypatch):
    """M7: the pipeline's unavailability has a fix -- an install command --
    and the toast that reports it is the only place the user sees it."""
    _patch_realtime_config(monkeypatch)
    monkeypatch.setattr(
        hands_free_module.console_voice_input,
        "probe",
        lambda: hands_free_module.console_voice_input.Availability(
            ok=False,
            kind="missing-capture",
            reason=hands_free_module.console_voice_input.CAPTURE_REASON,
            remedy=hands_free_module.console_voice_input.CAPTURE_REMEDY,
        ),
    )
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    rig.connect_error = RuntimeError("handshake rejected")
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        notifications = _capture_notifications(console)

        console.action_toggle_console_hands_free()
        await _wait_for(lambda: bool(notifications), pilot)

        joined = " ".join(message for message, _kwargs in notifications)
        assert "handshake rejected" in joined, joined
        assert "pip install" in joined, joined


@pytest.mark.asyncio
async def test_connect_timeout_is_bounded_and_falls_back(monkeypatch):
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _patch_realtime_config(monkeypatch)
    monkeypatch.setattr(
        realtime_module, "CONSOLE_REALTIME_CONNECT_TIMEOUT_SECONDS", 0.05
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


# ---------------------------------------------------------------------------
# Persistent diagnostics
# ---------------------------------------------------------------------------


class _DiagnosticsCapture(logging.Handler):
    """Collect records from the persistent-diagnostics logger namespace."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.asyncio
async def test_realtime_lifecycle_is_persistently_logged(monkeypatch):
    """The persistent log admits only `tldw_chatbook.diagnostics.*`, so the
    owner's stuck-connecting run left ZERO trace to diagnose. Entry and
    connect-failure must both be reconstructable from the log alone."""
    _patch_realtime_config(monkeypatch)
    monkeypatch.setattr(
        hands_free_module.console_voice_input,
        "probe",
        lambda: hands_free_module.console_voice_input.Availability(
            ok=False, kind="missing-capture", reason="No microphone.", remedy=""
        ),
    )
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    rig.connect_error = RuntimeError(_INVALID_KEY_REASON)
    host = ConsoleHarness(app)

    capture = _DiagnosticsCapture()
    diagnostics_logger = logging.getLogger("tldw_chatbook.diagnostics.realtime")
    diagnostics_logger.addHandler(capture)
    previous_level = diagnostics_logger.level
    diagnostics_logger.setLevel(logging.DEBUG)

    try:
        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)
            console.action_toggle_console_hands_free()
            await _wait_for(lambda: console._console_realtime is None, pilot)
    finally:
        diagnostics_logger.removeHandler(capture)
        diagnostics_logger.setLevel(previous_level)

    assert all(
        record.name == "tldw_chatbook.diagnostics.realtime"
        for record in capture.records
    )
    # `event=<name> field=value …` -- the same single-line shape the
    # dictation events already write, and the shape the persistent
    # formatter puts on disk.
    messages = [record.getMessage() for record in capture.records]
    assert any(message.startswith("event=realtime_entry ") for message in messages), (
        messages
    )
    assert any(
        message.startswith("event=realtime_connect_failed ") for message in messages
    ), messages
    assert any(
        "error_category=invalid_credentials" in message for message in messages
    ), messages
    # Never the key, in the one log that is written to disk.
    assert not any(_KEY_FRAGMENT in message for message in messages)


@contextmanager
def _realtime_diagnostics():
    """Capture the persistent-diagnostics records for the realtime logger."""
    capture = _DiagnosticsCapture()
    diagnostics_logger = logging.getLogger("tldw_chatbook.diagnostics.realtime")
    diagnostics_logger.addHandler(capture)
    previous_level = diagnostics_logger.level
    diagnostics_logger.setLevel(logging.DEBUG)
    try:
        yield capture
    finally:
        diagnostics_logger.removeHandler(capture)
        diagnostics_logger.setLevel(previous_level)


def _event_names(capture: "_DiagnosticsCapture") -> list[str]:
    names = []
    for record in capture.records:
        message = record.getMessage()
        assert message.startswith("event="), message
        names.append(message.split(" ", 1)[0].removeprefix("event="))
    return names


def _fields_for(capture: "_DiagnosticsCapture", event: str) -> dict[str, str]:
    for record in capture.records:
        message = record.getMessage()
        if message.startswith(f"event={event} "):
            return dict(
                part.split("=", 1) for part in message.split(" ") if "=" in part
            )
    raise AssertionError(f"{event} never fired: {_event_names(capture)}")


@pytest.mark.asyncio
async def test_a_whole_turn_is_reconstructable_from_the_persistent_log(monkeypatch):
    """Owner gate round 4: a turn that never produced a reply left NO trace
    -- only entry/ready/exit persisted, so the incident could not be
    diagnosed from the log at all. Every turn-level transition now writes
    one line through the same admitted logger."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    with _realtime_diagnostics() as capture:
        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)
            session = await _enter_live_realtime(console, pilot, rig)

            await _drive_to_speaking(console, pilot, session, audio=b"\x00" * 4800)
            session.fire_reply_done()
            await pilot.pause()
            rig.sink.finish_playback()
            await _wait_for(
                lambda: console._console_realtime.controller.state == "live", pilot
            )

    names = _event_names(capture)
    assert all(
        record.name == "tldw_chatbook.diagnostics.realtime"
        for record in capture.records
    )
    turn = [name for name in names if name not in {"realtime_entry", "realtime_ready"}]
    assert turn[:4] == [
        "realtime_turn_committed",
        "realtime_reply_started",
        "realtime_first_audio",
        "realtime_reply_done",
    ], names

    # The generation half deferred; the playback half fired the FSM. That
    # distinction IS the third live-gate defect, and it must be readable
    # from the log without the code in hand.
    done_records = [
        message
        for message in (record.getMessage() for record in capture.records)
        if message.startswith("event=realtime_reply_done ")
    ]
    assert len(done_records) == 2, done_records
    assert (
        "initiator=generation" in done_records[0]
        and "decision=deferred" in (done_records[0])
    )
    assert (
        "initiator=playback" in done_records[1]
        and "decision=fired" in (done_records[1])
    )


@pytest.mark.asyncio
async def test_barge_diagnostics_carry_the_trigger_and_the_state(monkeypatch):
    """The owner's report was "keyboard barge, then no reply". Which input
    barged, from which state, how much audio had played, and whether the
    provider was actually told -- all of it has to be in the log."""
    _patch_realtime_config(monkeypatch)
    app, host = _ready_host()
    rig = _install_realtime_fakes(app)

    with _realtime_diagnostics() as capture:
        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)
            session = await _enter_live_realtime(console, pilot, rig)

            await _drive_to_speaking(console, pilot, session, audio=b"\x00" * 48000)
            await pilot.press("x")
            await _wait_for(
                lambda: console._console_realtime.controller.state == "live", pilot
            )

    barge = _fields_for(capture, "realtime_barge")
    assert barge["initiator"] == "keypress"
    assert barge["phase"] == "speaking"
    assert barge["duration_ms"] == "1000"
    assert "realtime_cancel_sent" in _event_names(capture)


# ---------------------------------------------------------------------------
# Second live-gate defect: a handshake the endpoint ACCEPTS and then rejects
# ---------------------------------------------------------------------------

#: Shaped like OpenAI's real `error` payload for a bad key: the message
#: quotes a fragment of the key itself, which must never reach a toast or
#: a log line.
_KEY_FRAGMENT = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_INVALID_KEY_REASON = (
    f"Incorrect API key provided: {_KEY_FRAGMENT}. You can find your API "
    "key at https://platform.openai.com/account/api-keys (code=invalid_api_key)"
)


def _capture_warnings() -> tuple[list[str], int]:
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), level="DEBUG")
    return records, sink_id


@pytest.mark.asyncio
async def test_close_before_ready_fails_the_connect_instead_of_hanging(monkeypatch):
    """LIVE GATE: OpenAI ACCEPTS the WebSocket upgrade for a bad key and
    only then rejects, via an `error` event plus `close(3000,
    invalid_api_key)`.

    So `connect()` RETURNS -- no raise, no timeout -- and the failure
    arrives as callbacks while the FSM is still `connecting`, where a
    transport-closed input is deliberately ignored. Nothing routed to
    `on_connect_failed`, so the chip sat at `realtime · connecting…`
    forever with no toast: the dead entry the spec forbids.
    """
    _patch_realtime_config(monkeypatch)
    monkeypatch.setattr(
        hands_free_module.console_voice_input,
        "probe",
        lambda: hands_free_module.console_voice_input.Availability(
            ok=False, kind="missing-capture", reason="No microphone.", remedy=""
        ),
    )
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)
    records, sink_id = _capture_warnings()

    try:
        async with host.run_test(size=(160, 48)) as pilot:
            console = await _mounted_console(host, pilot)
            notifications = _capture_notifications(console)

            console.action_toggle_console_hands_free()
            await _wait_for(lambda: bool(rig.sessions), pilot)
            await _wait_for(lambda: rig.session.connected, pilot)
            assert console._console_realtime.controller.state == "connecting"

            rig.session.fire_closed(_INVALID_KEY_REASON)
            await _wait_for(lambda: console._console_realtime is None, pilot)
            await pilot.pause()

            assert "connecting" not in _visible_text(console), _visible_text(console)
            joined = " ".join(message for message, _kw in notifications)
            assert joined, "the dead entry produced no toast at all"
            assert "invalid_api_key" in joined, joined
    finally:
        logger.remove(sink_id)

    # Never the key itself -- not in the toast, not in any log line.
    assert _KEY_FRAGMENT not in " ".join(message for message, _kw in notifications), (
        "a key fragment reached a toast"
    )
    assert not any(_KEY_FRAGMENT in record for record in records), (
        "a key fragment reached the log"
    )


@pytest.mark.asyncio
async def test_error_before_ready_fails_the_connect(monkeypatch):
    """The provider's `error` event arrives BEFORE the close, so routing it
    ends the dead entry a beat sooner -- and with the same reason."""
    _patch_realtime_config(monkeypatch)
    monkeypatch.setattr(
        hands_free_module.console_voice_input,
        "probe",
        lambda: hands_free_module.console_voice_input.Availability(
            ok=False, kind="missing-capture", reason="No microphone.", remedy=""
        ),
    )
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        notifications = _capture_notifications(console)

        console.action_toggle_console_hands_free()
        await _wait_for(lambda: bool(rig.sessions) and rig.session.connected, pilot)

        rig.session.fire_error(RuntimeError(_INVALID_KEY_REASON))
        await _wait_for(lambda: console._console_realtime is None, pilot)

        joined = " ".join(message for message, _kw in notifications)
        assert "invalid_api_key" in joined, joined
        assert _KEY_FRAGMENT not in joined


@pytest.mark.asyncio
async def test_ready_that_never_arrives_times_out_instead_of_hanging(monkeypatch):
    """Belt and braces: `connect()` returned and NOTHING followed. No
    unforeseen no-ready path may hang the entry."""
    _patch_realtime_config(monkeypatch)
    monkeypatch.setattr(realtime_module, "CONSOLE_REALTIME_READY_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(
        hands_free_module.console_voice_input,
        "probe",
        lambda: hands_free_module.console_voice_input.Availability(
            ok=False, kind="missing-capture", reason="No microphone.", remedy=""
        ),
    )
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        notifications = _capture_notifications(console)

        console.action_toggle_console_hands_free()
        await _wait_for(lambda: bool(rig.sessions) and rig.session.connected, pilot)

        await _wait_for(lambda: console._console_realtime is None, pilot)
        joined = " ".join(message for message, _kw in notifications)
        assert "handshake" in joined.lower(), joined


@pytest.mark.asyncio
async def test_auth_failure_during_reconnect_gives_up_instead_of_hanging(monkeypatch):
    """The same close shape in the RECONNECTING window must reach the
    give-up exit -- the reconnect-once allowance is already spent there."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        notifications = _capture_notifications(console)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_closed("connection lost")
        await _wait_for(lambda: len(rig.sessions) == 2, pilot)
        assert console._console_realtime.controller.state == "reconnecting"

        await _wait_for(lambda: rig.sessions[1].connected, pilot)
        rig.sessions[1].fire_closed(_INVALID_KEY_REASON)
        await _wait_for(lambda: console._console_realtime is None, pilot)

        joined = " ".join(message for message, _kw in notifications)
        assert "connection lost" in joined, joined
        assert _KEY_FRAGMENT not in joined


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
        # M6: the docs promise a toast confirming the reconnect landed --
        # "reconnecting…" alone leaves the user unsure it ever finished.
        assert any(
            "reconnected" in message.lower() for message, _kw in notifications
        ), notifications

        # A SECOND drop within the same loop entry gives up outright.
        rig.sessions[1].fire_closed("connection lost")
        await _wait_for(lambda: console._console_realtime is None, pilot)
        assert any(
            "connection lost" in message for message, _kwargs in notifications
        ), notifications


# ---------------------------------------------------------------------------
# task-2360: reconnect must buffer, not drop, mic audio -- the tap's
# entry-time first-words guarantee extended across a mid-loop RECONNECT.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_buffers_mic_audio_and_flushes_to_the_new_session(monkeypatch):
    """Speech captured during the RECONNECTING window (old session gone,
    new one not yet ready) must not be dropped -- the SAME tap (never
    rebuilt across a reconnect) re-buffers it and flushes it, in order,
    to the new session the moment it goes ready, mirroring the entry-time
    first-words guarantee (`test_first_words_buffer_until_ready_then_
    flush_in_order`)."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)
        recorder = rig.recorder  # the tap survives the reconnect unchanged

        session.fire_closed("connection lost")
        await _wait_for(lambda: len(rig.sessions) == 2, pilot)
        assert console._console_realtime.controller.state == "reconnecting"

        # The old session is gone and the new one has not connected yet --
        # exactly the window where frames used to be silently dropped.
        recorder.push(b"during-reconnect-1")
        recorder.push(b"during-reconnect-2")
        assert rig.sessions[0].audio_frames == []
        assert rig.sessions[1].audio_frames == []

        await _wait_for(lambda: rig.sessions[1].connected, pilot)
        rig.sessions[1].fire_ready()
        await _wait_for(
            lambda: console._console_realtime.controller.state == "live", pilot
        )

        assert rig.sessions[1].audio_frames == [
            b"during-reconnect-1",
            b"during-reconnect-2",
        ]

        # Streaming resumes live afterward, unaffected by the buffering.
        recorder.push(b"after-reconnect")
        assert rig.sessions[1].audio_frames[-1] == b"after-reconnect"


@pytest.mark.asyncio
async def test_failed_reconnect_never_forwards_the_buffered_audio(monkeypatch):
    """The other direction: when the reconnect ITSELF fails (the same
    give-up exit `test_auth_failure_during_reconnect_gives_up_instead_of_
    hanging` pins), any audio buffered during that doomed reconnect window
    must never reach the failed session -- and the loop's ordinary
    teardown (which already stops, and thereby discards, the tap's buffer
    -- see `Tests/Audio/test_realtime_mic_tap.py::
    test_stop_after_begin_buffering_discards_the_rebuffered_frames`) is
    all that is needed; no separate discard path exists."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)
        recorder = rig.recorder

        session.fire_closed("connection lost")
        await _wait_for(lambda: len(rig.sessions) == 2, pilot)
        assert console._console_realtime.controller.state == "reconnecting"

        recorder.push(b"buffered-during-doomed-reconnect")

        await _wait_for(lambda: rig.sessions[1].connected, pilot)
        rig.sessions[1].fire_closed(_INVALID_KEY_REASON)  # the reconnect itself fails
        await _wait_for(lambda: console._console_realtime is None, pilot)
        await _wait_for(lambda: recorder.stop_calls == 1, pilot)

        assert rig.sessions[1].audio_frames == []


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
        await _wait_for(lambda: console._console_dictation_state == "recording", pilot)
        service.emit_final("what is the capital of france")
        await pilot.pause()

        console.action_toggle_console_hands_free()
        await _wait_for(lambda: bool(rig.sessions), pilot)
        await _wait_for(lambda: rig.session.connected, pilot)
        rig.session.fire_ready()

        await _wait_for(lambda: bool(rig.session.text_items), pilot)
        assert rig.session.text_items == [("what is the capital of france", True)]
        # The adopted transcript became the turn itself, not a stray draft.
        assert composer.draft_text().strip() == ""
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
        # F3: the tap's stop waits for callback quiescence and then joins
        # the recorder thread -- seconds of frozen UI if run inline.
        assert rig.recorder.stop_thread_ident != threading.get_ident()


@pytest.mark.asyncio
async def test_exit_mid_reply_closes_the_reply_row_as_interrupted(monkeypatch):
    """A loop torn down while the assistant is still talking must not leave
    a `pending` transcript row nothing will ever complete."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_turn_committed()
        session.fire_reply_started("item-1")
        session.fire_output_transcript_delta("Mid sen")
        await _wait_for(
            lambda: any(
                row.role is ConsoleMessageRole.ASSISTANT and row.content
                for row in _messages(console)
            ),
            pilot,
        )

        console._console_realtime.controller.on_exit_request()
        await _wait_for(lambda: console._console_realtime is None, pilot)

        assistant = [
            row
            for row in _messages(console)
            if row.role is ConsoleMessageRole.ASSISTANT
        ][0]
        assert assistant.status == "complete"
        assert assistant.content.endswith("interrupted"), assistant.content


@pytest.mark.asyncio
async def test_unmount_right_after_exit_still_closes_the_session(monkeypatch):
    """F7: exit dispatches the close to a worker. Unmounting before that
    worker has run must not leave the WebSocket open -- there is nothing
    left holding a reference to it by then."""
    _patch_realtime_config(monkeypatch)
    app, host = _ready_host()
    rig = _install_realtime_fakes(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        # No pause between the two: the close worker has not run yet.
        console._console_realtime.controller.on_exit_request()
        await console.on_unmount()

        assert session.closed is True


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


# ---------------------------------------------------------------------------
# task-2364: structured message metadata instead of parsed UI copy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_realtime_rows_carry_engine_provenance(monkeypatch):
    """The V4 spec's engine/provider/model provenance now has a field to
    live in, so it stops riding usage-attach and a visible marker."""
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
        session.fire_output_transcript_delta("spoken answer")
        session.fire_input_transcript("what is the weather")
        await _wait_for(lambda: len(_messages(console)) == 2, pilot)
        await _wait_for(
            lambda: _messages(console)[0].content == "what is the weather", pilot
        )

        user, assistant = _messages(console)
        assert user.metadata is not None
        assert user.metadata.engine == "realtime"
        assert user.metadata.provider == "openai"
        # The user row is attributed to the TRANSCRIPTION model, matching
        # how its usage (spoken-audio duration) is attributed.
        assert (
            user.metadata.model == realtime_module.CONSOLE_REALTIME_TRANSCRIPTION_MODEL
        )
        assert assistant.metadata is not None
        assert assistant.metadata.engine == "realtime"
        assert assistant.metadata.model == "gpt-realtime"


@pytest.mark.asyncio
async def test_barge_in_sets_the_structured_interrupted_flag(monkeypatch):
    """The visible marker stays for the human reader; machine consumers
    read the flag."""
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
            row
            for row in _messages(console)
            if row.role is ConsoleMessageRole.ASSISTANT
        ][0]
        assert assistant.content.endswith("interrupted"), assistant.content
        assert assistant.metadata is not None
        assert assistant.metadata.interrupted is True


@pytest.mark.asyncio
async def test_a_completed_reply_is_not_marked_interrupted(monkeypatch):
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_turn_committed()
        session.fire_reply_started("item-1")
        session.fire_output_transcript_delta("A whole answer.")
        session.fire_reply_done()
        await _wait_for(
            lambda: any(
                row.role is ConsoleMessageRole.ASSISTANT and row.status == "complete"
                for row in _messages(console)
            ),
            pilot,
        )

        assistant = [
            row
            for row in _messages(console)
            if row.role is ConsoleMessageRole.ASSISTANT
        ][0]
        assert assistant.metadata is not None
        assert assistant.metadata.interrupted is False


@pytest.mark.asyncio
async def test_seed_trims_the_marker_from_a_flagged_interrupted_reply(monkeypatch):
    """The ordinary interrupted case: a reply the engine cut short carries
    both the flag and the trailing marker, and the marker never reaches the
    model -- it is our chrome for the human reader."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        store.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content=(
                "Half a sentence" + realtime_module.CONSOLE_REALTIME_INTERRUPTED_MARKER
            ),
            metadata=MessageMetadata(engine="realtime", interrupted=True),
        )

        await _enter_live_realtime(console, pilot, rig)

        items, _instructions = rig.session.seeds[0]
        assert items == [("assistant", "Half a sentence")]


@pytest.mark.asyncio
async def test_seed_keeps_marker_shaped_text_a_user_actually_said(monkeypatch):
    """The marker is trimmed as a SUFFIX, so text merely CONTAINING it
    survives: the old global replace mangled any row whose words held the
    marker string, which is live user text, not chrome."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        store.append_message(
            session_id,
            role=ConsoleMessageRole.USER,
            content="type ⏹ interrupted into the log",
            metadata=MessageMetadata(engine="realtime", transcript_status="final"),
        )

        await _enter_live_realtime(console, pilot, rig)

        items, _instructions = rig.session.seeds[0]
        assert items == [("user", "type ⏹ interrupted into the log")]


@pytest.mark.asyncio
async def test_turn_commit_records_a_pending_transcript_status(monkeypatch):
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_turn_committed()
        await _wait_for(lambda: len(_messages(console)) == 1, pilot)

        user = _messages(console)[0]
        assert user.content == ""
        assert user.metadata is not None
        assert user.metadata.transcript_status == "pending"

        session.fire_input_transcript("what is the weather")
        await _wait_for(lambda: _messages(console)[0].content, pilot)
        assert _messages(console)[0].metadata.transcript_status == "final"


@pytest.mark.asyncio
async def test_an_empty_transcript_records_why_the_row_is_empty(monkeypatch):
    """The strand case: the provider transcribed the turn and it held no
    words. Before the field, the row sat empty forever with nothing saying
    whether the user was silent or the pipeline broke.

    task-2391: the row's CONTENT becomes the explanation, not just its
    metadata -- the store defers persistence for a content-less row, and
    the DB layer refuses to create a message with neither text nor an
    image at all (`CharactersRAGDB.add_message`), so a metadata-only
    "empty" row could never durably exist. Writing a short placeholder as
    real content flushes the same deferred-create path a real transcript
    already uses (`update_message_content`), so this row persists like
    any other -- see `Tests/Chat/test_console_chat_store.py` and
    `Tests/UI/test_console_resume_active_path.py` for the persistence and
    restart-survival proofs (this harness has no durable DB backing)."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_turn_committed()
        await _wait_for(lambda: len(_messages(console)) == 1, pilot)
        session.fire_input_transcript("   ")
        await _wait_for(
            lambda: (
                _messages(console)[0].metadata is not None
                and _messages(console)[0].metadata.transcript_status == "empty"
            ),
            pilot,
        )

        user = _messages(console)[0]
        assert (
            user.content
            == realtime_module.CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER
        )
        assert user.metadata.engine == "realtime"


@pytest.mark.asyncio
async def test_a_second_empty_transcript_does_not_double_mark_the_row(monkeypatch):
    """An empty payload landing twice for the same commit (a duplicate
    provider event, or a race) must not re-write the placeholder -- the
    content is already the placeholder, so `_mark_console_realtime_
    transcript_empty` skips the content write and only (harmlessly,
    idempotently) re-stamps the status."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_turn_committed()
        await _wait_for(lambda: len(_messages(console)) == 1, pilot)
        session.fire_input_transcript("")
        await _wait_for(
            lambda: (
                _messages(console)[0].content
                == realtime_module.CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER
            ),
            pilot,
        )
        session.fire_input_transcript("   ")
        await pilot.pause()
        await pilot.pause()

        user = _messages(console)[0]
        assert (
            user.content
            == realtime_module.CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER
        )
        assert user.metadata.transcript_status == "empty"


@pytest.mark.asyncio
async def test_an_empty_transcript_retries_the_status_after_a_swallowed_metadata_failure(
    monkeypatch,
):
    """Qodo Q4 (task-2391 review): the has-text early return used to treat
    the PLACEHOLDER itself as "already has text" and bail before ever
    reaching the status write on retry. That is reachable because
    `_set_console_realtime_transcript_status` deliberately swallows its own
    exceptions -- so a content-write-succeeded/metadata-write-failed
    partial state left a row whose content is the placeholder but whose
    `transcript_status` never became "empty", permanently: every later
    attempt hit the SAME early return. Such a row is invisible to
    `_is_empty_transcript_row` and would reach a provider as a fabricated
    user turn -- reopening the exact leak the prior fix closed, by a
    different route."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)
        store = console._ensure_console_chat_store()

        session.fire_turn_committed()
        await _wait_for(lambda: len(_messages(console)) == 1, pilot)

        # Simulate the swallowed failure: the content write below succeeds
        # normally, but the very next `set_message_metadata` call (the
        # status write) raises once.
        real_set_message_metadata = store.set_message_metadata
        state = {"raised": False}

        def _flaky_set_message_metadata(message_id, metadata):
            if not state["raised"]:
                state["raised"] = True
                raise RuntimeError("simulated metadata-write failure")
            return real_set_message_metadata(message_id, metadata)

        monkeypatch.setattr(store, "set_message_metadata", _flaky_set_message_metadata)

        session.fire_input_transcript("")
        await _wait_for(
            lambda: (
                _messages(console)[0].content
                == realtime_module.CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER
            ),
            pilot,
        )
        # Partial state reached: content landed, status write was swallowed.
        user = _messages(console)[0]
        assert user.metadata.transcript_status != "empty"

        # A retry (another empty payload for the same commit) must still
        # reach the status write -- not bail because content is non-blank.
        session.fire_input_transcript("")
        await _wait_for(
            lambda: _messages(console)[0].metadata.transcript_status == "empty",
            pilot,
        )

        user = _messages(console)[0]
        assert (
            user.content
            == realtime_module.CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER
        )

        from tldw_chatbook.Chat.console_chat_controller import _is_empty_transcript_row

        assert _is_empty_transcript_row(store.get_message(user.id))


@pytest.mark.asyncio
async def test_seed_excludes_a_row_whose_transcript_came_back_empty(monkeypatch):
    """task-2391 AC3: `transcript_status` needs a real consumer. The reseed
    builder is it -- an "empty" row's content is now the placeholder text,
    not something the user said, and must never be replayed into a
    reconnected session's context as if it were."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        store.append_message(
            session_id,
            role=ConsoleMessageRole.USER,
            content=realtime_module.CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER,
            metadata=MessageMetadata(engine="realtime", transcript_status="empty"),
        )
        store.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content="a real reply",
        )

        await _enter_live_realtime(console, pilot, rig)

        items, _instructions = rig.session.seeds[0]
        assert items == [("assistant", "a real reply")]


@pytest.mark.asyncio
async def test_a_late_empty_transcript_never_restates_a_filled_row(monkeypatch):
    """`on_input_transcript` carries no item id (see F5). An empty payload
    landing after the row was filled must not relabel a good transcript as
    'empty'."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)

        session.fire_turn_committed()
        await _wait_for(lambda: len(_messages(console)) == 1, pilot)
        session.fire_input_transcript("what is the weather")
        await _wait_for(lambda: _messages(console)[0].content, pilot)

        session.fire_input_transcript("")
        await pilot.pause()
        await pilot.pause()

        user = _messages(console)[0]
        assert user.content == "what is the weather"
        assert user.metadata.transcript_status == "final"


@pytest.mark.asyncio
async def test_seed_keeps_marker_text_typed_by_a_user_with_no_metadata(monkeypatch):
    """F1: only realtime rows are ever stamped with metadata, so every TYPED
    Console turn takes the no-metadata path -- permanently, not just until
    pre-v31 rows age out. A global `replace` there ate the marker string out
    of the middle of text a user actually typed."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        store = console._ensure_console_chat_store()
        store.append_message(
            store.active_session_id,
            role=ConsoleMessageRole.USER,
            content="the docs say ⏹ interrupted means cut off",
        )

        await _enter_live_realtime(console, pilot, rig)

        items, _instructions = rig.session.seeds[0]
        assert items == [("user", "the docs say ⏹ interrupted means cut off")]


@pytest.mark.asyncio
async def test_seed_still_trims_a_marker_suffix_on_a_row_without_metadata(monkeypatch):
    """The no-metadata path must keep doing its job for rows the engine
    genuinely cut short before the field existed: the marker is only ever
    APPENDED, so trimming the suffix covers every one of them."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        store = console._ensure_console_chat_store()
        store.append_message(
            store.active_session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content=(
                "Half a sentence" + realtime_module.CONSOLE_REALTIME_INTERRUPTED_MARKER
            ),
        )

        await _enter_live_realtime(console, pilot, rig)

        items, _instructions = rig.session.seeds[0]
        assert items == [("assistant", "Half a sentence")]


@pytest.mark.asyncio
async def test_seed_trims_a_marker_suffix_even_when_the_flag_says_otherwise(
    monkeypatch,
):
    """F2: the marker append and the metadata write are separate calls, each
    independently swallowed on failure. A row carrying the marker but no flag
    must still not seed chrome into the model."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        store = console._ensure_console_chat_store()
        store.append_message(
            store.active_session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content=(
                "Half a sentence" + realtime_module.CONSOLE_REALTIME_INTERRUPTED_MARKER
            ),
            metadata=MessageMetadata(engine="realtime", interrupted=False),
        )

        await _enter_live_realtime(console, pilot, rig)

        items, _instructions = rig.session.seeds[0]
        assert items == [("assistant", "Half a sentence")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seconds",
    [-1, float("nan"), float("inf"), float("-inf")],
    ids=["negative", "nan", "inf", "-inf"],
)
async def test_a_nonsense_transcription_duration_is_sanitized_before_it_persists(
    monkeypatch, seconds
):
    """Qodo Q2: the duration comes off the wire and went straight into
    `ProviderUsage.transcription_seconds` via a bare `float()`, so a
    negative or non-finite value propagated through `plus()` and into the
    JSON written to the database."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)
        store = console._ensure_console_chat_store()

        session.fire_turn_committed()
        await _wait_for(
            lambda: console._console_realtime.user_row_id is not None, pilot
        )
        row_id = console._console_realtime.user_row_id
        session.fire_transcription_usage({"type": "duration", "seconds": seconds})
        await _wait_for(lambda: store.get_message(row_id).usage is not None, pilot)

        usage = store.get_message(row_id).usage
        assert math.isfinite(usage.transcription_seconds)
        assert usage.transcription_seconds == 0.0
        # And the record it produces is still round-trippable: `json.dumps`
        # emits bare NaN/Infinity, which strict JSON readers reject.
        assert "Infinity" not in usage.to_json()
        assert "NaN" not in usage.to_json()


@pytest.mark.asyncio
async def test_a_duration_payload_with_no_seconds_attaches_nothing(monkeypatch):
    """The sanitizer maps unusable values to 0.0, so the ABSENT-key case
    needs its own guard: attaching a 0.0-second record would occupy the
    row's single usage slot and make the real duration -- if it followed --
    look like a late duplicate."""
    _patch_realtime_config(monkeypatch)
    app = _build_test_app()
    rig = _install_realtime_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        session = await _enter_live_realtime(console, pilot, rig)
        store = console._ensure_console_chat_store()

        session.fire_turn_committed()
        await _wait_for(
            lambda: console._console_realtime.user_row_id is not None, pilot
        )
        row_id = console._console_realtime.user_row_id
        session.fire_transcription_usage({"type": "duration"})
        await pilot.pause()
        await pilot.pause()
        assert store.get_message(row_id).usage is None

        # ...and the real duration that follows is still accepted.
        session.fire_transcription_usage({"type": "duration", "seconds": 2})
        await _wait_for(lambda: store.get_message(row_id).usage is not None, pilot)
        assert store.get_message(row_id).usage.transcription_seconds == 2.0
