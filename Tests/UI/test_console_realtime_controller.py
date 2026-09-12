"""Plain-fake characterization of Console realtime orchestration ownership."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.UI.Console_Modules import realtime as realtime_module
from tldw_chatbook.UI.Console_Modules.realtime import (
    ConsoleRealtimeController,
    ConsoleRealtimeSession,
)


class _FSM:
    def __init__(self, *, state: str = "live") -> None:
        self.state = state
        self.mic_gated = False
        self.calls: list[tuple[str, Any]] = []

    def on_exit_request(self) -> None:
        self.calls.append(("exit", None))

    def on_keypress(self) -> None:
        self.calls.append(("keypress", None))

    def on_session_ready(self) -> None:
        self.calls.append(("ready", None))

    def on_turn_committed(self, now: float) -> None:
        self.calls.append(("turn", now))

    def on_reply_started(self) -> None:
        self.calls.append(("reply_started", None))

    def on_first_audio(self) -> None:
        self.calls.append(("first_audio", None))

    def on_reply_done(self, now: float) -> None:
        self.calls.append(("reply_done", now))

    def on_connect_failed(self) -> None:
        self.calls.append(("connect_failed", None))


class _Store:
    def __init__(self) -> None:
        self.active_session_id = "console-1"
        self.events: list[tuple[Any, ...]] = []
        self.rows: dict[str, SimpleNamespace] = {}

    def messages_for_session(self, session_id: str) -> list[Any]:
        self.events.append(("messages", session_id))
        return list(self.rows.values())

    def append_message(
        self,
        session_id: str,
        *,
        role: ConsoleMessageRole,
        content: str,
        persist: bool,
        metadata: Any,
    ) -> SimpleNamespace:
        row_id = f"row-{len(self.rows) + 1}"
        row = SimpleNamespace(
            id=row_id,
            role=role,
            content=content,
            metadata=metadata,
            usage=None,
            status="pending",
        )
        self.rows[row_id] = row
        self.events.append(("append", session_id, role, content, metadata, persist))
        return row

    def get_message(self, row_id: str) -> SimpleNamespace:
        return self.rows[row_id]

    def finalize_deferred_user_message_content(self, row_id: str, text: str) -> None:
        self.rows[row_id].content = text
        self.events.append(("input", row_id, text))

    def set_message_metadata(self, row_id: str, metadata: Any) -> None:
        self.rows[row_id].metadata = metadata
        self.events.append(("metadata", row_id, metadata))

    def append_stream_chunk(self, row_id: str, text: str) -> None:
        self.rows[row_id].content += text
        self.events.append(("delta", row_id, text))

    def mark_message_complete(self, row_id: str) -> None:
        self.rows[row_id].status = "complete"
        self.events.append(("complete", row_id))

    def set_message_usage(self, row_id: str, usage: Any) -> None:
        self.rows[row_id].usage = usage
        self.events.append(("usage", row_id, usage))


class _Sink:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def open(self, sample_rate: int, channels: int) -> None:
        self.events.append(f"sink-open:{sample_rate}:{channels}")

    def stop(self) -> None:
        self.events.append("sink-stop")


class _Tap:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def begin_buffering(self) -> None:
        self.events.append("tap-buffer")

    def mark_ready(self) -> None:
        self.events.append("tap-ready")

    def set_gated(self, value: bool) -> None:
        self.events.append(f"tap-gate:{value}")

    def stop(self) -> None:
        self.events.append("tap-stop")


class _Provider:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def send_seed(self, items: list[tuple[str, str]], instructions: str | None) -> None:
        self.events.append(f"seed:{items!r}:{instructions!r}")

    def cancel_response(self, played_ms: int) -> bool:
        self.events.append(f"cancel:{played_ms}")
        return True

    async def close(self) -> None:
        self.events.append("provider-close")


class _TaskWorker:
    def __init__(self, task: asyncio.Task[Any]) -> None:
        self.task = task

    async def wait(self) -> None:
        await self.task


class _Worker:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def wait(self) -> None:
        self.events.append("worker-wait")


def _controller(
    *,
    store: _Store | None = None,
    runtime: Any = None,
    events: list[str] | None = None,
    provider_factory: Any = None,
    sink_factory: Any = None,
    run_worker: Any = None,
) -> ConsoleRealtimeController:
    store = store or _Store()
    events = events if events is not None else []
    runtime = runtime or SimpleNamespace(
        persona_buddy_sink=SimpleNamespace(
            next_voice_generation=lambda _session_id: 41,
            voice_state=lambda *_args: None,
            release_voice=lambda session_id, generation: events.append(
                f"buddy-release:{session_id}:{generation}"
            ),
        )
    )

    if run_worker is None:

        def run_worker(awaitable: Any, **_kwargs: Any) -> _Worker:
            if hasattr(awaitable, "close"):
                awaitable.close()
            return _Worker(events)

    return ConsoleRealtimeController(
        ensure_session_settings=lambda: SimpleNamespace(system_prompt="system"),
        chat_store_accessor=lambda: store,
        runtime_accessor=lambda: runtime,
        dictation_state_accessor=lambda: "idle",
        request_dictation_stop=lambda: events.append("dictation-stop"),
        pipeline_blocker=lambda: None,
        enter_pipeline_loop=lambda capture_live: events.append(
            f"pipeline:{capture_live}"
        ),
        recorder_factory_accessor=lambda: None,
        provider_session_factory_accessor=lambda: provider_factory,
        sink_factory_accessor=lambda: sink_factory or (lambda: _Sink(events)),
        notify=lambda text, **kwargs: events.append(
            f"notify:{kwargs.get('severity')}:{text}"
        ),
        ui_thread_id_accessor=threading.get_ident,
        event_loop_accessor=lambda: None,
        set_interval=lambda *_args, **_kwargs: SimpleNamespace(stop=lambda: None),
        run_worker=run_worker,
        defer_native_sync=lambda: events.append("sync"),
        repaint_chip=lambda: events.append("repaint"),
        restore_voice_chip=lambda: events.append("restore"),
    )


def _async_worker_runner(tasks: list[asyncio.Task[Any]]):
    def run_worker(awaitable: Any, **kwargs: Any) -> Any:
        if kwargs.get("group") == "console-realtime-audio":
            awaitable.close()
            return _Worker([])
        task = asyncio.create_task(awaitable)
        tasks.append(task)
        return _TaskWorker(task)

    return run_worker


async def _drain_workers(tasks: list[asyncio.Task[Any]]) -> None:
    while pending := [task for task in tasks if not task.done()]:
        await asyncio.wait_for(asyncio.gather(*pending), timeout=2.0)


def _install_buffering_tap(monkeypatch, events: list[str]):
    from tldw_chatbook.Audio import realtime_mic_tap

    class BufferingTap:
        instances: list[Any] = []

        def __init__(self, on_frames: Any, **_kwargs: Any) -> None:
            events.append("tap-factory")
            self.on_frames = on_frames
            self.buffer: list[bytes] = []
            self.instances.append(self)

        def start(self) -> bool:
            events.append("tap-start")
            self.buffer.append(b"first-words")
            return True

        def begin_buffering(self) -> None:
            events.append("tap-buffer")
            self.buffer.append(b"reconnect-words")

        def mark_ready(self) -> None:
            events.append("tap-ready")
            buffered, self.buffer = self.buffer, []
            for frames in buffered:
                self.on_frames(frames)

        def set_gated(self, value: bool) -> None:
            events.append(f"tap-gate:{value}")

        def stop(self) -> None:
            events.append("tap-stop")

    monkeypatch.setattr(realtime_mic_tap, "RealtimeMicTap", BufferingTap)
    return BufferingTap


class _ConnectingProvider(_Provider):
    def __init__(self, events: list[str], callbacks: Any, number: int) -> None:
        super().__init__(events)
        self.callbacks = callbacks
        self.number = number

    async def connect(self) -> None:
        self.events.append(f"provider-{self.number}-connect")
        self.callbacks.on_ready()

    def send_seed(self, items: list[tuple[str, str]], instructions: str | None) -> None:
        self.events.append(f"provider-{self.number}-seed:{items!r}:{instructions!r}")

    def append_audio(self, frames: bytes) -> None:
        self.events.append(f"provider-{self.number}-audio:{frames.decode()}")

    async def close(self) -> None:
        self.events.append(f"provider-{self.number}-close")


class _ProviderFactory:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.instances: list[_ConnectingProvider] = []

    def __call__(self, _config: Any, callbacks: Any) -> _ConnectingProvider:
        number = len(self.instances) + 1
        self.events.append(f"provider-factory:{number}")
        provider = _ConnectingProvider(self.events, callbacks, number)
        self.instances.append(provider)
        return provider


def _session(*, state: str = "live", generation: int = 41) -> ConsoleRealtimeSession:
    return ConsoleRealtimeSession(
        controller=_FSM(state=state),
        console_session_id="console-1",
        idle_timeout_seconds=60.0,
        buddy_generation=generation,
    )


def test_construction_starts_with_empty_owned_state() -> None:
    controller = _controller()

    assert controller.session is None
    assert controller.close_worker is None


def test_enter_installs_session_before_fsm_emits_connecting(monkeypatch) -> None:
    events: list[Any] = []
    controller = _controller(events=events)

    class OrderingFSM(_FSM):
        def __init__(self, emit, **_kwargs: Any) -> None:
            super().__init__(state="idle")
            self.emit = emit

        def enter(self) -> None:
            events.append(("enter-sees", controller.session))
            self.emit(realtime_module.ModeChanged("connecting", None))

    monkeypatch.setattr(realtime_module, "RealtimeLoopController", OrderingFSM)
    monkeypatch.setattr(realtime_module, "realtime_provider", lambda: "openai")
    monkeypatch.setattr(realtime_module, "realtime_model", lambda: "model")
    monkeypatch.setattr(realtime_module, "realtime_idle_timeout_seconds", lambda: 60.0)
    monkeypatch.setattr(realtime_module, "acoustic_barge_in_enabled", lambda: False)
    monkeypatch.setattr(
        controller,
        "_start_console_realtime_tap",
        lambda state: events.append(("tap", state)) or True,
        raising=False,
    )
    monkeypatch.setattr(
        controller,
        "_start_console_realtime_connect",
        lambda state: events.append(("connect", state)),
        raising=False,
    )

    controller._enter_console_realtime_loop(capture_live=False)

    assert events[0][0] == "enter-sees"
    assert events[0][1] is controller.session
    assert [event[0] for event in events if isinstance(event, tuple)] == [
        "enter-sees",
        "tap",
        "connect",
    ]


def test_marshal_accepts_current_and_rejects_stale_session_callback() -> None:
    controller = _controller()
    current = _session()
    stale = _session()
    current.connect_attempt = stale.connect_attempt = 3
    controller.session = current
    calls: list[tuple[ConsoleRealtimeSession, str]] = []

    controller._console_realtime_marshal(
        lambda state, value: calls.append((state, value)), current, 3, "current"
    )
    controller._console_realtime_marshal(
        lambda state, value: calls.append((state, value)), stale, 3, "stale"
    )
    controller._console_realtime_marshal(
        lambda state, value: calls.append((state, value)), current, 2, "old-attempt"
    )

    assert calls == [(current, "current")]


@pytest.mark.asyncio
async def test_initial_connect_preserves_first_words_and_builds_one_sink_per_reply(
    monkeypatch,
) -> None:
    events: list[str] = []
    tasks: list[asyncio.Task[Any]] = []
    provider_factory = _ProviderFactory(events)
    sink_instances: list[_Sink] = []

    def sink_factory() -> _Sink:
        events.append(f"sink-factory:{len(sink_instances) + 1}")
        sink = _Sink(events)
        sink_instances.append(sink)
        return sink

    runtime = SimpleNamespace(
        persona_buddy_sink=SimpleNamespace(
            next_voice_generation=lambda _session_id: 41,
            voice_state=lambda _session_id, _generation, state: events.append(
                f"mode:{state}"
            ),
            release_voice=lambda *_args: None,
        )
    )
    tap_type = _install_buffering_tap(monkeypatch, events)
    controller = _controller(
        events=events,
        runtime=runtime,
        provider_factory=provider_factory,
        sink_factory=sink_factory,
        run_worker=_async_worker_runner(tasks),
    )
    monkeypatch.setattr(realtime_module, "realtime_provider", lambda: "openai")
    monkeypatch.setattr(realtime_module, "realtime_model", lambda: "model")
    monkeypatch.setattr(realtime_module, "realtime_idle_timeout_seconds", lambda: 60.0)
    monkeypatch.setattr(realtime_module, "acoustic_barge_in_enabled", lambda: False)
    monkeypatch.setattr(controller, "_console_realtime_api_key", lambda: "test-key")
    monkeypatch.setattr(
        controller,
        "_console_realtime_seed_items",
        lambda _session_id: [("user", "prior")],
    )
    monkeypatch.setattr(controller, "_console_realtime_instructions", lambda: "system")

    controller._enter_console_realtime_loop(capture_live=False)
    await _drain_workers(tasks)

    session = controller.session
    assert session is not None
    assert len(tap_type.instances) == 1
    assert len(provider_factory.instances) == 1
    assert session.ready is True
    assert sink_instances == []
    assert events.index("tap-start") < events.index("provider-1-connect")
    assert events.index("provider-1-connect") < events.index(
        "provider-1-seed:[('user', 'prior')]:'system'"
    )
    assert events.index("provider-1-seed:[('user', 'prior')]:'system'") < events.index(
        "tap-ready"
    )
    assert events.index("tap-ready") < events.index("provider-1-audio:first-words")
    assert events.index("provider-1-audio:first-words") < events.index("mode:live")

    controller._on_console_realtime_reply_started(session, "reply-1")
    controller._on_console_realtime_audio_delta(session, b"aa")
    controller._on_console_realtime_audio_delta(session, b"bb")
    assert len(sink_instances) == 1
    first_sink = sink_instances[0]
    controller._on_console_realtime_reply_done(session)
    controller._console_realtime_playback_finished(session, session.reply_token)

    controller._on_console_realtime_reply_started(session, "reply-2")
    assert len(sink_instances) == 1
    controller._on_console_realtime_audio_delta(session, b"cc")

    assert len(sink_instances) == 2
    assert session.sink is sink_instances[1]
    assert session.sink is not first_sink
    assert events.count("sink-open:24000:1") == 2


@pytest.mark.asyncio
async def test_one_drop_reuses_tap_replaces_provider_reseeds_then_second_drop_exits(
    monkeypatch,
) -> None:
    events: list[str] = []
    tasks: list[asyncio.Task[Any]] = []
    provider_factory = _ProviderFactory(events)
    tap_type = _install_buffering_tap(monkeypatch, events)
    controller = _controller(
        events=events,
        provider_factory=provider_factory,
        run_worker=_async_worker_runner(tasks),
    )
    monkeypatch.setattr(realtime_module, "realtime_provider", lambda: "openai")
    monkeypatch.setattr(realtime_module, "realtime_model", lambda: "model")
    monkeypatch.setattr(realtime_module, "realtime_idle_timeout_seconds", lambda: 60.0)
    monkeypatch.setattr(realtime_module, "acoustic_barge_in_enabled", lambda: False)
    monkeypatch.setattr(controller, "_console_realtime_api_key", lambda: "test-key")
    monkeypatch.setattr(
        controller,
        "_console_realtime_seed_items",
        lambda _session_id: [("user", "continuity")],
    )
    monkeypatch.setattr(controller, "_console_realtime_instructions", lambda: "system")

    controller._enter_console_realtime_loop(capture_live=False)
    await _drain_workers(tasks)
    session = controller.session
    assert session is not None
    tap = session.tap
    first_provider = provider_factory.instances[0]

    first_provider.callbacks.on_closed("first-drop")
    await _drain_workers(tasks)

    assert controller.session is session
    assert session.tap is tap is tap_type.instances[0]
    assert len(tap_type.instances) == 1
    assert len(provider_factory.instances) == 2
    second_provider = provider_factory.instances[1]
    assert second_provider is not first_provider
    assert session.session is second_provider
    assert session.ready is True
    assert session.connect_attempt == 2
    assert events.count("tap-buffer") == 1
    assert events.count("provider-1-seed:[('user', 'continuity')]:'system'") == 1
    assert events.count("provider-2-seed:[('user', 'continuity')]:'system'") == 1
    assert events.index("tap-buffer") < events.index("provider-factory:2")
    assert events.index(
        "provider-2-seed:[('user', 'continuity')]:'system'"
    ) < events.index("provider-2-audio:reconnect-words")

    second_provider.callbacks.on_closed("second-drop")
    await _drain_workers(tasks)

    assert controller.session is None
    assert len(provider_factory.instances) == 2
    assert events.count("tap-buffer") == 1
    assert "provider-2-close" in events
    assert any(
        event == "notify:warning:Hands-free ended: connection lost" for event in events
    )


def test_reconnect_exhaustion_falls_back_loudly_with_current_capture_state() -> None:
    events: list[str] = []
    controller = _controller(events=events)

    controller._console_realtime_fallback_to_pipeline("refused")

    assert any(
        event.startswith("notify:warning:Realtime voice unavailable")
        for event in events
    )
    assert "pipeline:False" in events


def test_transcript_reply_audio_playback_and_usage_keep_row_metadata_order(
    monkeypatch,
) -> None:
    store = _Store()
    controller = _controller(store=store)
    session = _session(state="thinking")
    controller.session = session
    monkeypatch.setattr(realtime_module, "realtime_model", lambda: "rt-model")

    controller._on_console_realtime_turn_committed(session)
    user_row = session.user_row_id
    controller._on_console_realtime_input_transcript(session, "private words")
    controller._on_console_realtime_reply_started(session, "provider-item")
    assistant_row = session.assistant_row_id
    controller._on_console_realtime_output_transcript_delta(session, "answer")
    controller._on_console_realtime_audio_delta(session, b"\x01\x02")
    audio_queue = session.audio_queue
    reply_token = session.reply_token
    controller._on_console_realtime_first_audio(session)
    controller._on_console_realtime_reply_done(session)
    assert session.generation_done is True
    assert session.playback_pending is True
    assert not any(call[0] == "reply_done" for call in session.controller.calls)
    controller._on_console_realtime_usage(
        session, {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    )
    controller._console_realtime_playback_finished(session, reply_token)

    assert user_row is not None and assistant_row is not None
    assert audio_queue is not None
    assert audio_queue.get_nowait() == b"\x01\x02"
    assert audio_queue.get_nowait() is None
    assert store.rows[user_row].metadata.engine == "realtime"
    assert store.rows[user_row].metadata.transcript_status == "final"
    assert store.rows[assistant_row].metadata.model == "rt-model"
    assert store.rows[assistant_row].metadata.interrupted is False
    assert store.rows[assistant_row].content == "answer"
    assert store.rows[assistant_row].status == "complete"
    assert store.rows[assistant_row].usage.total_tokens == 5
    assistant_append = next(
        index
        for index, event in enumerate(store.events)
        if event[0] == "append" and event[2] is ConsoleMessageRole.ASSISTANT
    )
    assistant_delta = next(
        index
        for index, event in enumerate(store.events)
        if event[0] == "delta" and event[1] == assistant_row
    )
    assistant_metadata = next(
        index
        for index, event in enumerate(store.events)
        if event[0] == "metadata" and event[1] == assistant_row
    )
    assistant_complete = next(
        index
        for index, event in enumerate(store.events)
        if event[0] == "complete" and event[1] == assistant_row
    )
    assistant_usage = next(
        index
        for index, event in enumerate(store.events)
        if event[0] == "usage" and event[1] == assistant_row
    )
    user_input = next(
        index
        for index, event in enumerate(store.events)
        if event[0] == "input" and event[1] == user_row
    )
    assert (
        user_input
        < assistant_append
        < assistant_delta
        < assistant_metadata
        < assistant_complete
        < assistant_usage
    )
    assert [call[0] for call in session.controller.calls] == [
        "turn",
        "reply_started",
        "first_audio",
        "reply_done",
    ]


def test_handle_key_consumes_escape_only_and_reports_other_key_barge() -> None:
    controller = _controller()
    session = _session(state="speaking")
    controller.session = session

    assert controller.handle_key("escape") is True
    assert controller.handle_key("x") is False

    assert session.controller.calls == [("exit", None), ("keypress", None)]
    assert session.barge_trigger == "keypress"


@pytest.mark.asyncio
async def test_active_teardown_releases_exact_buddy_generation_and_resource_order() -> (
    None
):
    events: list[str] = []
    ui_thread_id = threading.get_ident()
    stop_thread_ids: list[int] = []

    class RecordingTap(_Tap):
        def stop(self) -> None:
            stop_thread_ids.append(threading.get_ident())
            self.events.append("tap-stop")

    controller = _controller(events=events)
    session = _session(generation=73)
    session.tap = RecordingTap(events)
    session.session = _Provider(events)
    session.sink = _Sink(events)
    session.audio_queue = asyncio.Queue()
    controller.session = session

    await controller.teardown()

    assert controller.session is None
    assert "buddy-release:console-1:73" in events
    assert len(stop_thread_ids) == 1
    assert stop_thread_ids[0] != ui_thread_id
    assert events.index("tap-stop") < events.index("provider-close")
    assert events.index("provider-close") < events.index("sink-stop")


@pytest.mark.asyncio
async def test_teardown_awaits_retained_close_worker_after_state_was_cleared() -> None:
    events: list[str] = []
    controller = _controller(events=events)
    controller.close_worker = _Worker(events)

    await controller.teardown()

    assert controller.close_worker is None
    assert events == ["worker-wait"]


def test_failure_sanitization_excludes_keys_payload_audio_and_private_transcript(
    monkeypatch,
) -> None:
    events: list[str] = []
    controller = _controller(events=events)
    session = _session(state="connecting")
    session.connect_attempt = 1
    controller.session = session
    persisted: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        controller,
        "_persist_console_realtime_event",
        lambda event, **fields: persisted.append((event, fields)),
    )
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz012345"
    private = "my private transcript"
    raw = RuntimeError(
        f"Incorrect API key provided: {secret}; payload={{'{private}': b'\\x00\\x01'}} "
        "(code=invalid_api_key)"
    )

    controller._console_realtime_connect_failed(session, 1, raw)

    visible = " ".join(events) + repr(persisted) + session.failure_text
    assert secret not in visible
    assert private not in visible
    assert "\\x00" not in visible
    assert "payload" not in visible
    assert session.failure_text == "Incorrect API key provided (invalid_api_key)"
    assert persisted[0][1]["error_category"] == "invalid_credentials"
