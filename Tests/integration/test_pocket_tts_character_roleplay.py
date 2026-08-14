"""Clean-profile PocketTTS character-roleplay acceptance journey.

This test keeps the first-run wizard, character import/handoff, Console send,
chat adapter, PocketTTS adapter, WAV validation, and speech lifecycle real.
Only the two external HTTP services and the final audio-device sink are faked.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.css.query import NoMatches
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Markdown,
    RadioButton,
    Static,
    Switch,
)

from Tests.Character_Chat.test_character_card_lenient_import import (
    _v2_card,
    _write_png_with_trailing_metadata,
)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Audio.streaming_sink import SinkStarted, SinkStopped
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.config import save_settings_to_cli_config
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_MODEL,
    STEP_PROVIDER,
    STEP_SUMMARY,
    STEP_VOICE,
    STEP_WELCOME,
    TRACK_QUICK,
)
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    ModelStep,
    ProviderStep,
    SetupWizardContainer,
    VoiceSetupStep,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

pytestmark = [pytest.mark.asyncio, pytest.mark.allow_network]


@dataclass(frozen=True, slots=True)
class _CapturedRequest:
    method: str
    path: str
    url: str
    headers: dict[str, str]
    body: bytes

    def json(self) -> dict[str, object]:
        value = json.loads(self.body)
        assert isinstance(value, dict)
        return value


class _LoopbackServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16


def _start_server(handler: type[BaseHTTPRequestHandler]):
    server = _LoopbackServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=server.serve_forever,
        name=f"{handler.__name__}-server",
        daemon=True,
    )
    thread.start()
    return server, thread


def _close_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    assert not thread.is_alive(), f"loopback server did not stop: {thread.name}"


def _playable_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 1_600)
    return output.getvalue()


@pytest.fixture
def fake_chat():
    requests: list[_CapturedRequest] = []
    server: _LoopbackServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _capture(self) -> _CapturedRequest:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            request = _CapturedRequest(
                method=self.command,
                path=self.path,
                url=f"http://127.0.0.1:{server.server_port}{self.path}",
                headers={key: value for key, value in self.headers.items()},
                body=body,
            )
            requests.append(request)
            return request

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._capture()
            body = json.dumps(
                {"object": "list", "data": [{"id": "test-chat-model"}]}
            ).encode("utf-8")
            self._send(body, "application/json")

        def do_POST(self) -> None:
            request = self._capture()
            payload = request.json()
            if payload.get("stream") is True:
                body = (
                    b'data: {"id":"chatcmpl-uat","object":"chat.completion.chunk",'
                    b'"choices":[{"index":0,"delta":{"content":"Welcome back."},'
                    b'"finish_reason":null}]}\n\n'
                    b'data: {"id":"chatcmpl-uat","object":"chat.completion.chunk",'
                    b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                    b"data: [DONE]\n\n"
                )
                self._send(body, "text/event-stream")
                return
            body = json.dumps(
                {
                    "id": "chatcmpl-uat",
                    "object": "chat.completion",
                    "model": "test-chat-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Welcome back.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode("utf-8")
            self._send(body, "application/json")

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server, thread = _start_server(Handler)
    endpoint = SimpleNamespace(
        base_url=f"http://127.0.0.1:{server.server_port}",
        requests=requests,
    )
    try:
        yield endpoint
    finally:
        _close_server(server, thread)


@pytest.fixture
def fake_pocket_tts():
    requests: list[_CapturedRequest] = []
    wav_body = _playable_wav()
    server: _LoopbackServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            requests.append(
                _CapturedRequest(
                    method=self.command,
                    path=self.path,
                    url=f"http://127.0.0.1:{server.server_port}{self.path}",
                    headers={key: value for key, value in self.headers.items()},
                    body=body,
                )
            )
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav_body)))
            self.end_headers()
            self.wfile.write(wav_body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server, thread = _start_server(Handler)
    endpoint = SimpleNamespace(
        speech_url=f"http://127.0.0.1:{server.server_port}/v1/audio/speech",
        requests=requests,
    )
    try:
        yield endpoint
    finally:
        _close_server(server, thread)


@pytest.fixture
def character_png(tmp_path: Path) -> Path:
    card = _v2_card(name="Pocket Ann", first_mes="Hello, I am Ann.")
    payload = base64.b64encode(json.dumps(card).encode("utf-8")).decode("ascii")
    return _write_png_with_trailing_metadata(
        tmp_path / "pocket_ann.png",
        {"chara": payload},
    )


class _PlaybackDeviceSink:
    """Audio-device boundary driven by the real WAV sink pump."""

    def __init__(
        self,
        *,
        on_event: Callable[[object], None],
        owner: _PlaybackBoundary,
    ) -> None:
        self.on_event = on_event
        self.owner = owner
        self.opened_with: tuple[int, int] | None = None
        self.state = "idle"
        self.terminal_reason: str | None = None
        self.fail_reason: str | None = None
        self.fed: list[bytes] = []
        self._started = False
        self._lock = threading.Lock()

    def open(self, sample_rate: int, channels: int = 1) -> None:
        with self._lock:
            self.opened_with = (sample_rate, channels)
            self.state = "open"

    def feed(self, pcm: bytes) -> bool:
        emit_started = False
        with self._lock:
            if self.state != "open":
                return False
            self.fed.append(pcm)
            if not self._started:
                self._started = True
                emit_started = True
        if emit_started:
            self.owner.started.set()
            self.on_event(SinkStarted())
        return True

    def close(self) -> None:
        with self._lock:
            if self.state == "open":
                self.state = "draining"

    def stop(self) -> None:
        with self._lock:
            if self.terminal_reason is not None:
                return
            self.state = "stopped"
            self.terminal_reason = "stopped"
        self.on_event(SinkStopped())

    @property
    def bytes_per_second(self) -> int:
        rate, channels = self.opened_with or (16_000, 1)
        return rate * channels * 2

    @property
    def buffered_seconds(self) -> float:
        return 2.0


@dataclass(slots=True)
class _PlaybackBoundary:
    sinks: list[_PlaybackDeviceSink] = field(default_factory=list)
    started: threading.Event = field(default_factory=threading.Event)

    def build(self, *, on_event: Callable[[object], None], **_kwargs: object):
        sink = _PlaybackDeviceSink(on_event=on_event, owner=self)
        self.sinks.append(sink)
        return sink


@dataclass(slots=True)
class _JourneyContext:
    fake_chat: Any
    fake_pocket_tts: Any
    monkeypatch: pytest.MonkeyPatch
    playback: _PlaybackBoundary
    app: Any = None
    pilot: Any = None
    run_context: Any = None
    chat_screen: Any = None
    greeting_message_id: str | None = None
    reply_message_id: str | None = None
    tts_request_baseline: int | None = None
    playback_sink_baseline: int | None = None


_ACTIVE_JOURNEY: _JourneyContext | None = None


def _journey() -> _JourneyContext:
    assert _ACTIVE_JOURNEY is not None, "journey fixture is not active"
    return _ACTIVE_JOURNEY


@pytest.fixture(autouse=True)
async def _journey_context(
    fake_chat,
    fake_pocket_tts,
    monkeypatch: pytest.MonkeyPatch,
):
    global _ACTIVE_JOURNEY

    from tldw_chatbook.Event_Handlers.TTS_Events import tts_events

    playback = _PlaybackBoundary()
    monkeypatch.setattr(tts_events, "sink_available", lambda: True)
    monkeypatch.setattr(tts_events, "StreamingPcmSink", playback.build)
    context = _JourneyContext(
        fake_chat=fake_chat,
        fake_pocket_tts=fake_pocket_tts,
        monkeypatch=monkeypatch,
        playback=playback,
    )
    _ACTIVE_JOURNEY = context
    try:
        yield context
    finally:
        assert context.run_context is None, "clean app context was not closed"
        _ACTIVE_JOURNEY = None


async def _wait_for(
    app,
    condition: Callable[[], Any],
    *,
    timeout: float = 20.0,
    interval: float = 0.05,
):
    context = _journey()
    assert context.app is app
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = condition()
        if result:
            return result
        await context.pilot.pause(interval)
    raise TimeoutError("condition not met within timeout")


async def launch_clean_chatbook():
    context = _journey()
    assert save_settings_to_cli_config(
        {
            "general": {"default_tab": "chat"},
            "splash_screen": {"enabled": False},
            "scheduling": {
                "watchlist_checks_enabled": False,
                "watchlist_checks_shadow": False,
                "briefing_schedules_enabled": False,
            },
            "media_cleanup": {"enabled": False},
            "model_catalog": {"auto_refresh_enabled": False},
            "diagnostics": {"ui_responsiveness_enabled": False},
        }
    )
    app = TldwCli()
    assert app.notes_service is not None, (
        "journey must use production TldwCli service initialization"
    )
    assert app.media_db is not None
    assert app.chachanotes_db is not None
    context.app = app
    context.run_context = app.run_test(size=(180, 55))
    context.pilot = await context.run_context.__aenter__()
    wizard = await _wait_for(
        app,
        lambda: app.screen
        if type(app.screen).__name__ == "FirstRunSetupWizard"
        else None,
        timeout=30.0,
    )
    await context.pilot.pause(0.2)
    await _wait_for(
        app,
        lambda: wizard.query_one(SetupWizardContainer)
        if wizard.query_one(SetupWizardContainer).current_step == 0
        else None,
    )
    return app


async def _close_clean_chatbook(app) -> None:
    context = _journey()
    assert context.app is app
    run_context, context.run_context = context.run_context, None
    if run_context is not None:
        await run_context.__aexit__(None, None, None)


async def complete_quick_setup(
    app,
    *,
    tts_endpoint: str,
    auth: str,
    model: str,
    voice: str,
    response_format: str,
    use_as_default: bool,
) -> None:
    context = _journey()
    assert auth == "none"
    wizard = app.screen
    container = wizard.query_one(SetupWizardContainer)

    welcome_index = container._step_index_for_id(STEP_WELCOME)
    provider_index = container._step_index_for_id(STEP_PROVIDER)
    model_index = container._step_index_for_id(STEP_MODEL)
    voice_index = container._step_index_for_id(STEP_VOICE)
    assert welcome_index is not None
    assert provider_index is not None
    assert model_index is not None
    assert voice_index is not None

    assert container.current_step == welcome_index
    quick_setup = wizard.query_one("#setup-track-quick", RadioButton)
    await context.pilot.click("#setup-track-quick")
    await context.pilot.pause()
    assert quick_setup.value is True
    assert container.track == TRACK_QUICK
    await context.pilot.click("#wizard-next")
    await _wait_for(app, lambda: container.current_step == provider_index)

    provider_step = container.steps[provider_index]
    assert isinstance(provider_step, ProviderStep)
    await _wait_for(
        app,
        lambda: next(iter(provider_step.query("#setup-provider-key-status")), None)
        if container.current_step == provider_index
        else None,
    )
    provider_step.select_provider("llama_cpp")
    await _wait_for(
        app,
        lambda: next(iter(provider_step.query("#setup-provider-endpoint")), None),
    )
    provider_step.query_one("#setup-provider-endpoint", Input).value = (
        context.fake_chat.base_url
    )
    await context.pilot.pause()
    provider_step.query_one("#setup-provider-test", Button).press()

    def provider_tested():
        identity = provider_step._provider_current_draft_identity()
        if identity is None:
            return None
        evidence = provider_step._provider_evidence_store().evidence_for(identity)
        return evidence if evidence is not None and evidence.endpoint == "reachable" else None

    await _wait_for(app, provider_tested)
    assert provider_step.compose_failure is None
    readiness = provider_step._current_provider_readiness()
    assert readiness.ready, readiness
    assert container.steps[provider_index] is provider_step
    await context.pilot.click("#wizard-next")
    await _wait_for(app, lambda: container.current_step == model_index)

    model_step = container.steps[model_index]
    assert isinstance(model_step, ModelStep)

    def discovered_model_button():
        return next(
            (
                button
                for button in model_step.query(RadioButton)
                if getattr(button, "_model_id", None) == "test-chat-model"
            ),
            None,
        )

    model_button = await _wait_for(app, discovered_model_button)
    model_button.value = True
    await context.pilot.pause()
    await context.pilot.click("#wizard-next")
    await _wait_for(app, lambda: container.current_step == voice_index)

    voice_step = container.steps[voice_index]
    assert isinstance(voice_step, VoiceSetupStep)
    voice_step.query_one("#setup-voice-endpoint", Input).value = tts_endpoint
    voice_step.query_one("#setup-voice-model", Input).value = model
    voice_step.query_one("#setup-voice-voice", Input).value = voice
    voice_step.query_one("#setup-voice-format", Input).value = response_format
    voice_step.query_one("#setup-voice-speed", Input).value = "1.0"
    voice_step.query_one("#setup-voice-default", Checkbox).value = use_as_default
    voice_step.query_one("#setup-voice-auth-none", RadioButton).value = True
    await context.pilot.pause()
    voice_step.query_one("#setup-voice-test", Button).press()
    await _wait_for(
        app,
        lambda: "Verified"
        in str(voice_step.query_one("#setup-voice-status", Static).renderable),
    )
    await context.pilot.click("#wizard-next")
    await _wait_for(
        app,
        lambda: next(iter(wizard.query("#setup-exit-home")), None)
        if container.steps[container.current_step].config.id == STEP_SUMMARY
        else None,
    )
    await _wait_for(
        app,
        lambda: wizard.query_one("#setup-summary-rows", Static)
        if str(wizard.query_one("#setup-summary-rows", Static).renderable).strip()
        and str(wizard.query_one("#setup-summary-footer", Static).renderable).startswith(
            "Config file:"
        )
        else None,
    )
    wizard.query_one("#setup-exit-home", Button).press()
    await _wait_for(
        app,
        lambda: type(app.screen).__name__ != "FirstRunSetupWizard",
    )

    assert app.app_config["chat_defaults"]["provider"] == "llama_cpp"
    assert app.app_config["chat_defaults"]["model"] == "test-chat-model"
    assert len(context.fake_pocket_tts.requests) == 1
    context.tts_request_baseline = len(context.fake_pocket_tts.requests)
    context.playback_sink_baseline = len(context.playback.sinks)


async def import_character_and_start_chat(app, character_png: Path) -> None:
    context = _journey()
    app.post_message(NavigateToScreen("personas"))
    personas = await _wait_for(
        app,
        lambda: app.screen
        if type(app.screen).__name__ == "PersonasScreen" and app.screen.is_mounted
        else None,
        timeout=30.0,
    )
    await _wait_for(
        app,
        lambda: next(iter(personas.query(".loading-text")), None)
        if personas.state.active_mode == "characters"
        else None,
    )
    await personas._import_character_from_path(str(character_png))
    await context.pilot.pause(0.3)
    imported = [
        card
        for card in app.chachanotes_db.list_character_cards()
        if card.get("name") == "Pocket Ann"
    ]
    assert len(imported) == 1
    character_id = int(imported[0]["id"])
    start = personas.query_one("#personas-start-chat", Button)
    await _wait_for(app, lambda: not start.disabled)
    start.press()

    chat_screen = await _wait_for(
        app,
        lambda: app.screen
        if type(app.screen).__name__ == "ChatScreen" and app.screen.is_mounted
        else None,
        timeout=30.0,
    )
    context.chat_screen = chat_screen

    def character_handoff_consumed():
        store = chat_screen._ensure_console_chat_store()
        return (
            not app.pending_handoffs.has_pending(HandoffChannel.CHAT)
            and not chat_screen._handoff_consumption_in_progress
            and len(store.sessions()) == 1
            and str(store.sessions()[0].character_id) == str(character_id)
        )

    await _wait_for(app, character_handoff_consumed)
    store = chat_screen._ensure_console_chat_store()
    session = store.sessions()[0]
    assert session.character_name == "Pocket Ann"
    assert session.title == "Chat with Pocket Ann"
    messages = store.messages_for_session(session.id)
    assert len(messages) == 1
    greeting = messages[0]
    assert greeting.role is ConsoleMessageRole.ASSISTANT
    assert greeting.content == "Hello, I am Ann."
    assert greeting.status == "complete"
    context.greeting_message_id = greeting.id

    def greeting_visible():
        try:
            title = chat_screen.query_one("#console-transcript-title", Static)
            speaker = chat_screen.query_one(
                f"#console-message-header-{greeting.id} "
                ".console-transcript-speaker-label",
                Static,
            )
            action = chat_screen.query_one(
                f"#console-message-speech-action-{greeting.id}",
                Button,
            )
        except NoMatches:
            return None
        return title, speaker, action

    title, speaker, greeting_action = await _wait_for(app, greeting_visible)
    greeting_body = chat_screen.query_one(
        f"#console-message-{greeting.id} .console-markdown-body",
        Markdown,
    )
    assert title.visual.plain == "Conversation | Chat with Pocket Ann"
    assert speaker.visual.plain == "Pocket Ann"
    assert greeting_body.source == "Hello, I am Ann."
    assert greeting_action.console_action_id == "speak"
    assert greeting_action.disabled is False
    assert len(context.fake_pocket_tts.requests) == context.tts_request_baseline


async def enable_speak_replies_and_confirm(app) -> None:
    context = _journey()
    chat_screen = context.chat_screen
    toggle = chat_screen.query_one("#console-auto-speak", Switch)
    assert toggle.value is False
    assert toggle.disabled is False
    toggle.action_toggle_switch()

    modal = await _wait_for(
        app,
        lambda: app.screen
        if type(app.screen).__name__ == "AutoSpeakConsentModal"
        else None,
    )
    destination = str(
        modal.query_one("#console-auto-speak-consent-destination", Static).renderable
    )
    expected_destination = context.fake_pocket_tts.speech_url.rsplit(
        "/v1/audio/speech", 1
    )[0]
    assert expected_destination in destination
    modal.query_one("#console-auto-speak-consent-confirm", Button).press()
    await _wait_for(app, lambda: app.screen is chat_screen)

    def enabled():
        store = chat_screen._ensure_console_chat_store()
        session = next(
            item for item in store.sessions() if item.id == store.active_session_id
        )
        return (
            session.speech_preferences.auto_speak is True
            and session.speech_preferences.consent_destination is not None
            and chat_screen._console_auto_speak._modal_open is False
        )

    await _wait_for(app, enabled)


async def send_roleplay_message(app, message: str) -> None:
    context = _journey()
    chat_screen = context.chat_screen
    greeting_message_id = context.greeting_message_id
    assert greeting_message_id is not None
    greeting_action = chat_screen.query_one(
        f"#console-message-speech-action-{greeting_message_id}",
        Button,
    )
    assert greeting_action.console_action_id == "speak"
    assert greeting_action.disabled is False
    assert len(context.fake_pocket_tts.requests) == context.tts_request_baseline
    composer = chat_screen.query_one("#console-native-composer", ConsoleComposerBar)
    composer.load_draft(message)
    await context.pilot.pause()
    chat_screen.query_one("#console-send-message", Button).press()

    def reply_message():
        store = chat_screen._ensure_console_chat_store()
        session_id = store.active_session_id
        if session_id is None:
            return None
        return next(
            (
                item
                for item in store.messages_for_session(session_id)
                if item.role is ConsoleMessageRole.ASSISTANT
                and item.content == "Welcome back."
                and item.status == "complete"
            ),
            None,
        )

    reply = await _wait_for(app, reply_message, timeout=30.0)
    context.reply_message_id = reply.id


async def wait_for_audio_playback(app) -> None:
    context = _journey()
    chat_screen = context.chat_screen
    message_id = context.reply_message_id
    assert message_id is not None
    assert context.tts_request_baseline is not None
    assert context.playback_sink_baseline is not None

    def playing_controls():
        try:
            status = chat_screen.query_one(
                f"#console-message-speech-status-{message_id}", Static
            )
            action = chat_screen.query_one(
                f"#console-message-speech-action-{message_id}", Button
            )
        except NoMatches:
            return None
        return (status, action) if str(status.renderable) == "Playing" else None

    status, action = await _wait_for(app, playing_controls, timeout=30.0)
    assert context.playback.started.is_set()
    assert len(context.fake_pocket_tts.requests) == context.tts_request_baseline + 1
    assert len(context.playback.sinks) == context.playback_sink_baseline + 1
    assert action.console_action_id == "speak-stop"
    assert action.disabled is False
    assert "Playing" in app.export_screenshot()
    sink = context.playback.sinks[-1]
    assert sink.fed
    action.press()

    def stopped_controls():
        return (
            status,
            action,
        ) if (
            str(status.renderable) == "Stopped"
            and action.console_action_id == "speak"
        ) else None

    stopped_status, speak_action = await _wait_for(app, stopped_controls)
    assert str(stopped_status.renderable) == "Stopped"
    assert speak_action.console_action_id == "speak"
    assert speak_action.disabled is False
    assert sink.state == "stopped"
    assert sink.terminal_reason == "stopped"


async def test_clean_profile_character_roleplay_uses_pocket_tts(
    fake_chat,
    fake_pocket_tts,
    character_png,
):
    app = await launch_clean_chatbook()
    try:
        await complete_quick_setup(
            app,
            tts_endpoint=fake_pocket_tts.speech_url,
            auth="none",
            model="pocket-tts",
            voice="alba",
            response_format="wav",
            use_as_default=True,
        )
        await import_character_and_start_chat(app, character_png)
        await enable_speak_replies_and_confirm(app)
        await send_roleplay_message(app, "Hello there")
        await wait_for_audio_playback(app)
        assert len(fake_pocket_tts.requests) == 2
        assert fake_chat.requests[-1].json()["model"] == "test-chat-model"
        assert fake_pocket_tts.requests[-1].json() == {
            "model": "pocket-tts",
            "input": "Welcome back.",
            "voice": "alba",
            "response_format": "wav",
            "speed": 1.0,
        }
        assert "Authorization" not in fake_pocket_tts.requests[-1].headers
    finally:
        await _close_clean_chatbook(app)
