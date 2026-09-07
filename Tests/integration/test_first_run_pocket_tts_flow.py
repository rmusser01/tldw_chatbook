from __future__ import annotations

import io
import json
import threading
import tomllib
import wave
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, Input, RadioButton

from Tests.TTS.adapter_fakes import FakeAdapterFactory, provider_spec
from tldw_chatbook import config as config_module
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSSettingsSaveEvent,
)
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.TTS_Generation import TTSService
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
    def __init__(
        self,
        wizard: FirstRunSetupWizard,
        service: TTSService,
    ) -> None:
        super().__init__()
        self.wizard = wizard
        self.app_config = config_module.settings
        self.stts_handler = STTSEventHandler(self)
        self.stts_handler._stts_service = service

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.push_screen(self.wizard)

    @on(STTSSettingsSaveEvent)
    async def handle_voice_save(self, event: STTSSettingsSaveEvent) -> None:
        await self.stts_handler.handle_settings_save(event)


async def run_quick_voice_setup(endpoint: str, config_path: Path):
    old_defaults = TTSPreferencesSnapshot.from_settings(config_module.settings)
    registry = TTSAdapterRegistry(
        specs=(provider_spec("openai", FakeAdapterFactory("openai"), {}),),
        aliases={},
    )
    service = TTSService(registry, preferences_snapshot=old_defaults)
    app_instance = MagicMock(app_config=config_module.settings)
    wizard = FirstRunSetupWizard(app_instance)
    app = _VoiceFlowApp(wizard, service)

    try:
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

            persisted = tomllib.loads(config_path.read_text(encoding="utf-8"))
            app_tts = persisted["app_tts"]
            active_configuration = await registry.provider_configuration_snapshot(
                "openai"
            )
            return SimpleNamespace(
                persisted_settings=app_tts,
                saved_revision=service.saved_configuration_revision("openai"),
                applied_revision=service.applied_configuration_revision("openai"),
                active_runtime_revision=service.configuration_revision("openai"),
                active_provider_settings=active_configuration.applied_config[
                    "app_config"
                ]["app_tts"],
                effective_defaults=TTSPreferencesSnapshot.from_settings(persisted),
                active_defaults=service.preferences_snapshot(),
            )
    finally:
        await service.close()
        await service.wait_closed()


@pytest.fixture
def isolated_voice_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    config_path = tmp_path / "voice-flow.toml"
    config_path.write_text(
        "[app_tts]\n"
        'default_provider = "openai"\n'
        'default_model_mode = "exact"\n'
        'default_model = "tts-1-hd"\n'
        'default_voice_mode = "exact"\n'
        'default_voice = "shimmer"\n'
        'default_format = "mp3"\n'
        "default_speed = 1.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    initial = tomllib.loads(config_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        config_module,
        "settings",
        {"COMPREHENSIVE_CONFIG_RAW": initial},
    )
    return config_path


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
    isolated_voice_config: Path,
) -> None:
    result = await run_quick_voice_setup(
        fake_pocket_tts.url,
        isolated_voice_config,
    )

    assert result.persisted_settings["OPENAI_BASE_URL"] == fake_pocket_tts.url
    assert result.persisted_settings["OPENAI_AUTH_MODE"] == "none"
    assert result.saved_revision == result.applied_revision
    assert result.active_runtime_revision > 0
    assert result.active_provider_settings["OPENAI_BASE_URL"] == fake_pocket_tts.url
    assert result.active_provider_settings["OPENAI_AUTH_MODE"] == "none"
    assert result.effective_defaults.provider_id == "openai"
    assert result.effective_defaults.model_id == "pocket-tts"
    assert result.effective_defaults.voice_id == "alba"
    assert result.effective_defaults == result.active_defaults
    assert fake_pocket_tts.requests[-1].json()["input"] == "Hello from Chatbook."
    assert "Authorization" not in fake_pocket_tts.requests[-1].headers
