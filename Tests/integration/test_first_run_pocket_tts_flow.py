from __future__ import annotations

import io
import json
import threading
import wave
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, Input, RadioButton

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
    STTSSettingsSaveResult,
)
from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_VOICE, TRACK_QUICK
from tldw_chatbook.UI.Wizards.first_run_voice_step_state import (
    VoiceSetupDraft,
    run_voice_sample,
)
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    FirstRunSetupWizard,
    SetupWizardContainer,
    VoiceSetupStep,
)

pytestmark = pytest.mark.allow_network


@dataclass(frozen=True, slots=True)
class _CapturedRequest:
    path: str
    headers: dict[str, str]
    body: bytes

    def json(self) -> dict[str, object]:
        value = json.loads(self.body)
        assert isinstance(value, dict)
        return value


def _playable_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 160)
    return output.getvalue()


@pytest.fixture
def fake_pocket_tts():
    requests: list[_CapturedRequest] = []
    body = _playable_wav()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            requests.append(
                _CapturedRequest(
                    path=self.path,
                    headers={key: value for key, value in self.headers.items()},
                    body=self.rfile.read(length),
                )
            )
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(
            url=f"http://127.0.0.1:{server.server_port}/v1/audio/speech",
            requests=requests,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _VoiceFlowApp(App):
    def __init__(self, wizard: FirstRunSetupWizard) -> None:
        super().__init__()
        self.wizard = wizard
        self.saved_event: STTSSettingsSaveEvent | None = None

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.push_screen(self.wizard)

    @on(STTSSettingsSaveEvent)
    def handle_voice_save(self, event: STTSSettingsSaveEvent) -> None:
        self.saved_event = event
        assert event.request_id is not None
        event.reply_to.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=event.request_id,
                persisted=True,
                provider_statuses={"openai": "applied"},
                provider_configuration_revisions={"openai": 7},
                provider_runtime_revisions={"openai": 41},
                defaults_activated=True,
                defaults_activation_status="committed",
            )
        )


async def run_quick_voice_setup(endpoint: str):
    app_instance = MagicMock(app_config={})
    wizard = FirstRunSetupWizard(app_instance)
    app = _VoiceFlowApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        index = container._step_index_for_id(STEP_VOICE)
        assert index is not None
        container.show_step(index)
        step = container.steps[index]
        assert isinstance(step, VoiceSetupStep)

        step.query_one("#setup-voice-endpoint", Input).value = endpoint
        step.query_one("#setup-voice-model", Input).value = "pocket-tts"
        step.query_one("#setup-voice-voice", Input).value = "alba"
        step.query_one("#setup-voice-format", Input).value = "wav"
        step.query_one("#setup-voice-speed", Input).value = "1.0"
        step.query_one("#setup-voice-sample", Input).value = "Hello from Chatbook."
        step.query_one("#setup-voice-default", Checkbox).value = True
        step.query_one("#setup-voice-auth-none", RadioButton).value = True
        await pilot.pause()

        step.query_one("#setup-voice-test", Button).press()
        for _ in range(100):
            if "Verified" in str(step.query_one("#setup-voice-status").renderable):
                break
            await pilot.pause(0.02)
        assert "Verified" in str(step.query_one("#setup-voice-status").renderable)

        ok, error = await step.commit()
        assert (ok, error) == (True, "")
        assert app.saved_event is not None
        return SimpleNamespace(default=app.saved_event.preferences)


@pytest.mark.asyncio
async def test_real_pocket_tts_sample_uses_exact_request_and_playable_response(
    fake_pocket_tts,
) -> None:
    result = await run_voice_sample(
        VoiceSetupDraft(
            endpoint=fake_pocket_tts.url,
            authentication_mode="none",
            model_id="pocket-tts",
            voice_id="alba",
            response_format="wav",
            speed=1.0,
            sample_text="Hello from Chatbook.",
            use_as_default=False,
        )
    )

    assert result.playable is True
    request = fake_pocket_tts.requests[-1]
    assert request.path == "/v1/audio/speech"
    assert request.json() == {
        "input": "Hello from Chatbook.",
        "model": "pocket-tts",
        "voice": "alba",
        "response_format": "wav",
        "speed": 1.0,
    }
    assert "Authorization" not in request.headers


@pytest.mark.asyncio
async def test_quick_voice_setup_sends_sample_and_activates_exact_defaults(
    fake_pocket_tts,
) -> None:
    result = await run_quick_voice_setup(fake_pocket_tts.url)

    assert result.default.provider_id == "openai"
    assert result.default.model_id == "pocket-tts"
    assert result.default.voice_id == "alba"
    assert fake_pocket_tts.requests[-1].json()["input"] == "Hello from Chatbook."
    assert "Authorization" not in fake_pocket_tts.requests[-1].headers
