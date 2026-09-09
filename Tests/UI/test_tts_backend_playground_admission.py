"""Mounted provider admission with real catalogs and synthetic complete audio.

Legacy adapters retain their production request/route checks; only the backend
host's audio execution is replaced. audio.cpp retains its real HTTP adapter,
catalog parser, request validation and WAV validation behind a MockTransport.
These tests make no provider-inference or device-playback claim.
"""

import asyncio
import io
import json
import struct
import threading
import wave
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Select, Static, TextArea
from textual.worker import WorkerCancelled

from Tests.UI.speech_playground_fixtures import _resolved, _wait_until
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSPlaygroundGenerateEvent,
)
from tldw_chatbook.TTS import audio_player as audio_player_module
from tldw_chatbook.TTS import pcm_playback
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.adapter_types import TTSProviderDescriptor, TTSProviderSpec
from tldw_chatbook.TTS.adapters.audio_cpp import AudioCppAdapter
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.audio_cpp_supervisor import AudioCppSupervisor
from tldw_chatbook.TTS.audio_player import PlaybackState
from tldw_chatbook.TTS.legacy_bridge import LegacyTTSAdapter
from tldw_chatbook.TTS.legacy_catalogs import legacy_catalog
from tldw_chatbook.TTS.playground_types import STTSGeneratedAudio
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.studio_preferences import (
    StudioTTSPreferencesSnapshot,
    StudioTTSSelectionOverrides,
)
from tldw_chatbook.TTS.TTS_Generation import TTSService
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.stts_playground_catalog import (
    LOADING_SELECT_VALUE,
    SERVER_DEFAULT_VOICE_ID,
    UNAVAILABLE_SELECT_VALUE,
)

# Independent expected provider defaults, not values calculated by the resolver.
DEFAULT_SELECTIONS = {
    "openai": ("tts-1", "alloy", "openai_official_tts-1"),
    "elevenlabs": (
        "eleven_multilingual_v2",
        "21m00Tcm4TlvDq8ikWAM",
        "elevenlabs_eleven_multilingual_v2",
    ),
    "kokoro": ("kokoro", "af_alloy", "local_kokoro_default_onnx"),
    "chatterbox": ("chatterbox", "default", "local_chatterbox_default"),
    "higgs": ("higgs-audio-v2", "professional_female", "local_higgs_v2"),
    "alltalk": ("alltalk", "female_01.wav", "alltalk_default"),
    "audio_cpp": ("native/model", None, None),
}
DEFAULT_OPTIONS = {
    "openai": {},
    "elevenlabs": {
        "stability": 0.5,
        "similarity_boost": 0.8,
        "style": 0.0,
        "use_speaker_boost": True,
    },
    "kokoro": {"use_onnx": True},
    "chatterbox": {
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "temperature": 0.5,
        "num_candidates": 1,
        "validate_with_whisper": False,
    },
    "higgs": {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1},
    "alltalk": {},
    "audio_cpp": {},
}


def _wav_sample():
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(struct.pack("<hh", 2_000, -2_000) * 1_200)
    return stream.getvalue()


def _preferences(provider):
    model, voice, _internal = DEFAULT_SELECTIONS[provider]
    return TTSPreferencesSnapshot(
        provider_id=provider,
        model_mode="first_available" if provider == "audio_cpp" else "exact",
        model_id=None if provider == "audio_cpp" else model,
        voice_mode="server_default" if voice is None else "exact",
        voice_id=voice,
        response_format="wav",
        speed=1.0,
    )


class _AudioHost:
    def __init__(self, audio):
        self.audio = audio
        self.calls = []

    def admitted_outbound_endpoint(self):
        return "https://tts.example.invalid"

    async def generate(self, internal_model_id, request, progress_sink):
        assert request.response_format == "wav"
        self.calls.append((internal_model_id, request))
        yield self.audio[:101]
        yield self.audio[101:]

    async def close(self):
        return None


class _RecordingLegacyAdapter(LegacyTTSAdapter):
    def __init__(self, provider, audio):
        super().__init__(provider, _AudioHost(audio), legacy_catalog(provider))
        self.requests = []

    async def synthesize(self, request, progress_sink=None):
        self.requests.append(request)
        return await super().synthesize(request, progress_sink)


class _ResponseStream(httpx.AsyncByteStream):
    def __init__(self, body):
        self.body = body

    async def __aiter__(self):
        yield self.body


def _json_response(value):
    return httpx.Response(
        200,
        headers={"Content-Type": "application/json"},
        stream=_ResponseStream(json.dumps(value).encode()),
    )


class _RecordingAudioCppAdapter(AudioCppAdapter):
    def __init__(self, audio):
        self.requests = []
        self.http_calls = []

        def respond(request):
            self.http_calls.append(request)
            if request.url.path == "/health":
                return _json_response({"status": "ok", "backend": "cpu", "models": 1})
            if request.url.path == "/v1/models":
                return _json_response(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": "native/model",
                                "object": "model",
                                "owned_by": "engine",
                                "family": "pocket_tts",
                                "task": "tts",
                                "mode": "native",
                            }
                        ],
                    },
                )
            if request.url.path == "/v1/audio/voices":
                return _json_response({"voices": ["native/voice"]})
            assert request.url.path == "/v1/audio/speech"
            assert json.loads(request.content)["response_format"] == "wav"
            return httpx.Response(
                200,
                stream=_ResponseStream(audio),
                headers={"Content-Type": "audio/wav"},
            )

        super().__init__(
            AudioCppConfig.from_mapping(
                {"mode": "external", "base_url": "http://127.0.0.1:18991"}
            ),
            transport=httpx.MockTransport(respond),
        )

    async def synthesize(self, request, progress_sink=None):
        self.requests.append(request)
        return await super().synthesize(request, progress_sink)


class _Host(App):
    def __init__(self, pane):
        super().__init__()
        self.pane = pane
        self.requests = []
        self.notices = []

    def compose(self) -> ComposeResult:
        yield self.pane

    def post_message(self, message):
        if isinstance(message, STTSPlaygroundGenerateEvent):
            self.requests.append(message.request)
            return True
        return super().post_message(message)

    def notify(self, message, *, severity="information", **kwargs):
        self.notices.append((str(message), severity))


@pytest.fixture
def backend_lab_factory(monkeypatch):
    @asynccontextmanager
    async def mount(provider, *, preferences=None, studio=None, wait_ready=True):
        preferences = preferences or _preferences(provider)
        studio = studio or StudioTTSPreferencesSnapshot()
        audio = _wav_sample()
        adapters = {
            provider_id: (
                _RecordingAudioCppAdapter(audio)
                if provider_id == "audio_cpp"
                else _RecordingLegacyAdapter(provider_id, audio)
            )
            for provider_id in DEFAULT_SELECTIONS
        }
        registry = TTSAdapterRegistry(
            specs=tuple(
                TTSProviderSpec(
                    descriptor=TTSProviderDescriptor(
                        provider_id=provider_id,
                        display_name=provider_id,
                        native=provider_id == "audio_cpp",
                    ),
                    factory=lambda _config, adapter=adapter: adapter,
                    initial_config=(
                        {"mode": "external", "base_url": "http://127.0.0.1:18991"}
                        if provider_id == "audio_cpp"
                        else {}
                    ),
                    exclusive_reconfigure=True,
                )
                for provider_id, adapter in adapters.items()
            ),
            aliases={},
        )
        service = TTSService(
            registry,
            preferences_snapshot=preferences,
            studio_preferences_loader=lambda: studio,
            audio_cpp_supervisor=AudioCppSupervisor(source_environment={}),
        )
        monkeypatch.setattr(
            SpeechPlaygroundPane,
            "_tts_service_factory",
            lambda self: _resolved(service),
        )
        monkeypatch.setattr(
            SpeechPlaygroundPane, "_check_higgs_installation", lambda self: None
        )
        pane = SpeechPlaygroundPane(
            provider=provider,
            axis_defaults=SpeechPlaygroundPane._project_axis_defaults(
                studio, preferences
            ),
            studio_preferences=studio,
            global_preferences=preferences,
        )
        app = _Host(pane)
        handler = STTSEventHandler(app)
        handler._stts_service = service
        app._stts_handler = handler
        try:
            async with app.run_test(size=(150, 65)) as pilot:
                await _wait_until(
                    pilot,
                    lambda: (
                        pane.provider == provider
                        and (
                            not pane.query_one("#tts-generate-btn", Button).disabled
                            if wait_ready
                            else provider in pane._catalogs
                        )
                    ),
                )
                yield SimpleNamespace(
                    app=app,
                    pane=pane,
                    pilot=pilot,
                    handler=handler,
                    service=service,
                    adapters=adapters,
                    audio=audio,
                )
        finally:
            await handler.cleanup_tts_resources()
            await service.close()
            await service.wait_closed()
            # The native adapter owns an HTTP client even if no lease used it.
            await adapters["audio_cpp"].close()

    return mount


def _assert_wav(audio):
    with wave.open(io.BytesIO(audio), "rb") as decoded:
        assert decoded.getnchannels() == 1
        assert decoded.getsampwidth() == 2
        assert decoded.getframerate() == 24_000
        assert decoded.getnframes() == 2_400
        assert decoded.readframes(2_400) == struct.pack("<hh", 2_000, -2_000) * 1_200


async def _generate(lab):
    provider = lab.pane.provider
    lab.pane.query_one("#tts-text-input", TextArea).text = "A synthetic audio reply."
    await lab.pilot.pause()
    previous = lab.handler._current_playground_artifact
    count = len(lab.app.requests)
    lab.pane.action_generate_tts()
    assert len(lab.app.requests) == count + 1, lab.app.notices
    await lab.handler.handle_playground_generate(
        STTSPlaygroundGenerateEvent(lab.app.requests[-1])
    )
    await lab.pilot.pause()
    artifact = lab.handler._current_playground_artifact
    assert artifact is not None and artifact is not previous, lab.app.notices
    assert artifact.operation_id == lab.app.requests[-1].operation_id
    assert artifact.provider_id == provider
    assert artifact.audio_format == "wav" and artifact.content_type == "audio/wav"
    assert artifact.path.read_bytes() == lab.audio
    _assert_wav(artifact.path.read_bytes())
    assert not lab.pane.query_one("#tts-generate-btn", Button).disabled
    assert not lab.pane.query_one("#audio-play-btn", Button).disabled
    if previous is not None:
        assert not previous.path.exists()
    return lab.adapters[provider].requests[-1]


def _assert_selection(request, provider):
    model, voice, internal = DEFAULT_SELECTIONS[provider]
    assert (request.provider_id, request.model_id, request.voice) == (
        provider,
        model,
        voice,
    )
    assert request.speed == 1.0 and request.response_format == "wav"
    if internal is not None:
        assert request.options["_legacy_internal_model_id"] == internal


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", DEFAULT_SELECTIONS)
async def test_fresh_provider_controls_admit_repeated_complete_wav_and_match_defaults(
    backend_lab_factory, provider
):
    async with backend_lab_factory(provider) as lab:
        language = lab.pane.query_one("#tts-language-select", Select)
        if provider == "kokoro":
            assert language.value == "" and not language.disabled
        else:
            assert language.disabled
            assert not lab.pane.query_one(
                "#speech-axis-cell-tts-language-select"
            ).display
        if provider == "audio_cpp":
            assert (
                lab.pane.query_one("#tts-voice-select", Select).value
                is SERVER_DEFAULT_VOICE_ID
            )
            assert lab.pane.query_one("#tts-format-select", Select).disabled
            assert lab.pane.query_one("#tts-speed-input", Input).disabled
        for _ in range(2):
            request = await _generate(lab)
            _assert_selection(request, provider)
            options = (
                request.options["_legacy_openai_request"].extra_params or {}
                if provider != "audio_cpp"
                else dict(request.options)
            )
            assert options == DEFAULT_OPTIONS[provider]
        response = await lab.service.synthesize_default(text="An automatic reply.")
        try:
            automatic_audio = b"".join([chunk async for chunk in response.byte_stream])
            assert automatic_audio == lab.audio
            _assert_wav(automatic_audio)
        finally:
            await response.aclose()
        assert len(lab.adapters[provider].requests) == 3
        _assert_selection(lab.adapters[provider].requests[-1], provider)


async def _switch(lab, provider):
    lab.pane.query_one("#tts-provider-select", Select).value = provider
    await _wait_until(
        lab.pilot,
        lambda: (
            lab.pane.provider == provider
            and not lab.pane.query_one("#tts-generate-btn", Button).disabled
            and bool(lab.pane.query("#speech-param-group"))
            and lab.pane.query_one("#speech-param-group").provider == provider
        ),
    )
    await lab.pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", DEFAULT_SELECTIONS)
async def test_provider_round_trip_does_not_carry_foreign_model_voice_or_options(
    backend_lab_factory, provider
):
    async with backend_lab_factory(provider) as lab:
        first = await _generate(lab)
        _assert_selection(first, provider)
        other = "openai" if provider == "audio_cpp" else "audio_cpp"
        await _switch(lab, other)
        _assert_selection(await _generate(lab), other)
        await _switch(lab, provider)
        _assert_selection(await _generate(lab), provider)


@pytest.mark.asyncio
async def test_explicit_kokoro_language_is_scoped_across_native_round_trip(
    backend_lab_factory,
):
    async with backend_lab_factory("kokoro") as lab:
        lab.pane.query_one("#tts-language-select", Select).value = "fr"
        first = await _generate(lab)
        assert first.options["_legacy_openai_request"].extra_params == {
            "use_onnx": True,
            "language": "fr",
        }
        await _switch(lab, "audio_cpp")
        assert dict((await _generate(lab)).options) == {}
        await _switch(lab, "kokoro")
        language = lab.pane.query_one("#tts-language-select", Select)
        assert language.value == "fr"
        assert language.value not in {LOADING_SELECT_VALUE, UNAVAILABLE_SELECT_VALUE}
        assert (await _generate(lab)).options[
            "_legacy_openai_request"
        ].extra_params == {"use_onnx": True, "language": "fr"}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", tuple(DEFAULT_SELECTIONS)[:-1])
async def test_stale_native_global_selection_does_not_block_selected_legacy_provider(
    backend_lab_factory, provider
):
    stale = replace(
        _preferences("audio_cpp"),
        model_mode="exact",
        model_id="removed/model",
        voice_mode="exact",
        voice_id="removed/voice",
    )
    async with backend_lab_factory(
        "audio_cpp", preferences=stale, wait_ready=False
    ) as lab:
        assert lab.pane.query_one("#tts-model-select", Select).value == "removed/model"
        assert lab.pane.query_one("#tts-voice-select", Select).value == "removed/voice"
        assert lab.pane.query_one("#tts-generate-btn", Button).disabled
        assert lab.adapters["audio_cpp"].requests == []
        await _switch(lab, provider)
        for _ in range(2):
            _assert_selection(await _generate(lab), provider)
        assert lab.service.preferences_snapshot() is stale
        assert lab.adapters["audio_cpp"].requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["global", "studio"])
@pytest.mark.parametrize("missing_axis", ["model", "voice"])
async def test_missing_exact_native_selection_is_pinned_before_and_after_provider_trip(
    backend_lab_factory, owner, missing_axis
):
    model = "removed/model" if missing_axis == "model" else "native/model"
    voice = "removed/voice" if missing_axis == "voice" else "native/voice"
    preferences = replace(
        _preferences("audio_cpp"),
        model_mode="exact",
        model_id=model,
        voice_mode="exact",
        voice_id=voice,
    )
    studio = StudioTTSPreferencesSnapshot()
    if owner == "studio":
        preferences = _preferences("openai")
        studio = StudioTTSPreferencesSnapshot(
            selection=StudioTTSSelectionOverrides(
                provider_id="audio_cpp",
                model_mode="exact",
                model_id=model,
                voice_mode="exact",
                voice_id=voice,
                response_format="wav",
            )
        )
    async with backend_lab_factory(
        "audio_cpp", preferences=preferences, studio=studio, wait_ready=False
    ) as lab:
        for visit in range(2):
            if visit:
                await _switch(lab, "openai")
                lab.pane.query_one("#tts-provider-select", Select).value = "audio_cpp"
            await _wait_until(
                lab.pilot,
                lambda: (
                    lab.pane.provider == "audio_cpp"
                    and lab.pane.query_one("#tts-model-select", Select).value == model
                    and lab.pane.query_one("#tts-voice-select", Select).value == voice
                ),
            )
            await lab.pilot.pause()
            assert lab.pane.query_one("#tts-generate-btn", Button).disabled
            lab.pane.query_one(
                "#tts-text-input", TextArea
            ).text = "Must stay unadmitted."
            lab.pane.action_generate_tts()
            assert lab.app.requests == []
            assert lab.adapters["audio_cpp"].requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("voice", ["native/voice", "removed/voice"])
async def test_returning_to_global_provider_keeps_its_voice_when_studio_owns_another(
    backend_lab_factory, voice
):
    preferences = replace(
        _preferences("audio_cpp"),
        model_mode="exact",
        model_id="native/model",
        voice_mode="exact",
        voice_id=voice,
    )
    studio = StudioTTSPreferencesSnapshot(
        selection=StudioTTSSelectionOverrides(provider_id="openai")
    )
    async with backend_lab_factory(
        "openai", preferences=preferences, studio=studio
    ) as lab:
        lab.pane.query_one("#tts-provider-select", Select).value = "audio_cpp"
        await _wait_until(
            lab.pilot,
            lambda: (
                lab.pane.provider == "audio_cpp"
                and "audio_cpp" in lab.pane._catalogs
                and ("audio_cpp", "native/model") in lab.pane._discovered_voices
            ),
        )
        await lab.pilot.pause()
        assert lab.pane.query_one("#tts-voice-select", Select).value == voice
        if voice == "removed/voice":
            assert lab.pane.query_one("#tts-generate-btn", Button).disabled
            assert lab.adapters["audio_cpp"].requests == []
        else:
            assert (await _generate(lab)).voice == voice


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,terminal",
    [
        ("kokoro", "stop"),
        ("kokoro", "finish"),
        ("kokoro", "failure"),
        ("kokoro", "error"),
        ("openai", "unknown"),
    ],
)
async def test_pcm_lab_playback_preserves_raw_export_and_refuses_unknown_shape(
    provider, terminal, backend_lab_factory, monkeypatch, tmp_path
):
    preferences = replace(_preferences(provider), response_format="pcm")
    pcm = struct.pack("<hh", 2000, -2000) * 1200

    class Player:
        def __init__(self):
            self.played = []
            self.bodies = []
            self.state = PlaybackState.IDLE

        async def play(self, path):
            self.played.append(path)
            self.bodies.append(path.read_bytes())
            self.state = PlaybackState.PLAYING
            return terminal != "failure"

        async def stop(self):
            self.state = PlaybackState.IDLE
            return True

        async def get_state(self):
            return self.state

        async def is_playing(self):
            return self.state is PlaybackState.PLAYING

        async def get_position(self):
            return 0

        async def get_duration(self):
            return 0.1

    async with backend_lab_factory(provider, preferences=preferences) as lab:
        host = lab.adapters[provider].host

        async def generate(internal_model_id, request, progress_sink):
            assert request.response_format == "pcm"
            host.calls.append((internal_model_id, request))
            yield pcm[:3]
            yield pcm[3:]

        monkeypatch.setattr(host, "generate", generate)
        player = Player()
        lab.app.audio_player = player
        lab.pane.query_one("#tts-text-input", TextArea).text = "A synthetic PCM reply."
        lab.pane.action_generate_tts()
        await lab.handler.handle_playground_generate(
            STTSPlaygroundGenerateEvent(lab.app.requests[-1])
        )
        await lab.pilot.pause()
        artifact = lab.handler._current_playground_artifact
        assert artifact is not None
        assert artifact.audio_format == "pcm"
        assert artifact.path.suffix == ".pcm"
        assert artifact.path.read_bytes() == pcm
        release = Mock(wraps=lab.handler.release_playground_result)
        monkeypatch.setattr(lab.handler, "release_playground_result", release)

        exported = tmp_path / "exported.pcm"
        lab.pane._handle_audio_export(str(exported))
        assert exported.read_bytes() == pcm
        release.reset_mock()

        for attempt in range(2):
            lab.pane.action_play_audio()
            await lab.app.workers.wait_for_complete()
            if provider == "openai":
                # The fixture's custom origin offers no sample-rate contract.
                assert player.played == []
                assert any("PCM" in message for message, _ in lab.app.notices)
            else:
                assert len(player.played) == attempt + 1
                playback_path = player.played[-1]
                assert playback_path != artifact.path
                assert playback_path.suffix == ".wav"
                _assert_wav(player.bodies[-1])
                if terminal == "stop":
                    await lab.pane._stop_audio_async()
                elif terminal in {"finish", "error"}:
                    player.state = (
                        PlaybackState.ERROR
                        if terminal == "error"
                        else PlaybackState.FINISHED
                    )
                    await _wait_until(
                        lab.pilot, lambda path=playback_path: not path.exists()
                    )
                    assert lab.pane._progress_timer_task.done()
                assert not playback_path.exists()
            assert release.call_count == attempt + 1
            assert lab.handler._playground_file_leases.get(artifact.path, 0) == 0
            assert not lab.pane.query_one("#audio-play-btn", Button).disabled
            assert lab.pane.query_one("#pause-audio-btn", Button).disabled
            assert lab.pane.query_one("#stop-audio-btn", Button).disabled
            if terminal == "error":
                assert str(
                    lab.pane.query_one("#audio-player-status", Static).content
                ) == ("Playback failed")
                assert (
                    lab.app.notices.count(("Playback failed", "error")) == attempt + 1
                )
                assert ("Playback complete", "information") not in lab.app.notices
            assert artifact.path.read_bytes() == pcm
            assert exported.read_bytes() == pcm


@pytest.mark.asyncio
async def test_cancelled_lab_pcm_copy_releases_its_lease_after_the_reader_retires(
    backend_lab_factory, monkeypatch, tmp_path
):
    entered = threading.Event()
    finish = threading.Event()
    copies = []
    real_copy = pcm_playback.create_pcm16_wav_copy

    def held_copy(*args):
        result = real_copy(*args)
        copies.append(result)
        entered.set()
        assert finish.wait(5)
        return result

    monkeypatch.setattr(pcm_playback, "create_pcm16_wav_copy", held_copy)
    async with backend_lab_factory("kokoro") as lab:
        original = tmp_path / "owned.pcm"
        original.write_bytes(struct.pack("<hh", 2000, -2000) * 1200)
        artifact = STTSGeneratedAudio(
            path=original,
            provider_id="kokoro",
            model_id="kokoro",
            voice_id="af_alloy",
            source_text="A synthetic reply.",
            operation_id="pcm-copy-cancel",
            audio_format="pcm",
            content_type="application/octet-stream",
            metadata={"sample_rate": 24000, "channels": 1},
        )
        lab.handler._accept_playground_artifact(artifact)
        lab.pane._generation_complete(artifact)
        release = Mock(wraps=lab.handler.release_playground_result)
        monkeypatch.setattr(lab.handler, "release_playground_result", release)
        lab.app.audio_player = SimpleNamespace(
            get_state=AsyncMock(return_value=PlaybackState.IDLE),
            stop=AsyncMock(return_value=True),
            play=AsyncMock(return_value=True),
        )
        try:
            lab.pane.action_play_audio()
            assert await asyncio.to_thread(entered.wait, 5)
            playback_worker = lab.pane._play_worker_task
            playback_worker.cancel()
            await lab.pilot.pause()
            assert original.exists()
            assert copies[0].exists()
            assert lab.handler._playground_file_leases[original] == 1
            release.assert_not_called()
            finish.set()
            with pytest.raises(WorkerCancelled):
                await playback_worker.wait()
            assert not copies[0].exists()
            assert original.exists()
            assert lab.handler._playground_file_leases.get(original, 0) == 0
            release.assert_called_once_with(artifact.operation_id, original)
            lab.app.audio_player.play.assert_not_called()
        finally:
            finish.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 17, None])
async def test_opus_lab_action_uses_real_player_and_observes_terminal_failure(
    exit_code, backend_lab_factory, monkeypatch, tmp_path
):
    """Exercise application transport; the fake process makes no codec claim."""
    player = audio_player_module.SimpleAudioPlayer()
    player._system = "Darwin"
    monkeypatch.setattr(audio_player_module, "_audio_player_instance", player)
    monkeypatch.setattr(
        audio_player_module.shutil,
        "which",
        lambda name: (
            "/test/ffplay" if name == "ffplay" and exit_code is not None else None
        ),
    )
    finished = threading.Event()
    process = MagicMock()
    process.poll.side_effect = lambda: exit_code if finished.is_set() else None

    def wait(timeout=None):
        assert finished.wait(5 if timeout is None else timeout)
        return exit_code

    process.wait.side_effect = wait
    process.terminate.side_effect = finished.set
    process.kill.side_effect = finished.set
    spawn = Mock(return_value=process)
    monkeypatch.setattr(audio_player_module.subprocess, "Popen", spawn)

    async with backend_lab_factory("kokoro") as lab:
        original = tmp_path / "owned.opus"
        original.write_bytes(b"OggS synthetic device-transport fixture")
        artifact = STTSGeneratedAudio(
            path=original,
            provider_id="kokoro",
            model_id="kokoro",
            voice_id="af_alloy",
            source_text="A synthetic reply.",
            operation_id="opus-transport",
            audio_format="opus",
            content_type="audio/opus",
        )
        lab.handler._accept_playground_artifact(artifact)
        lab.pane._generation_complete(artifact)
        release = Mock(wraps=lab.handler.release_playground_result)
        monkeypatch.setattr(lab.handler, "release_playground_result", release)
        try:
            lab.pane.action_play_audio()
            await lab.app.workers.wait_for_complete()
            assert isinstance(
                lab.app.audio_player, audio_player_module.AsyncAudioPlayer
            )
            assert lab.app.audio_player._player is player
            if exit_code is None:
                spawn.assert_not_called()
                assert ("Failed to start playback", "error") in lab.app.notices
            else:
                assert spawn.call_args.args[0] == [
                    "/test/ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "error",
                    str(original),
                ]
                release.assert_not_called()
                assert lab.handler._playground_file_leases[original] == 1
                finished.set()
                await _wait_until(lab.pilot, lab.pane._progress_timer_task.done)
                expected_state = (
                    PlaybackState.FINISHED if exit_code == 0 else PlaybackState.ERROR
                )
                assert await lab.app.audio_player.get_state() is expected_state
                expected_notice = (
                    ("Playback complete", "information")
                    if exit_code == 0
                    else ("Playback failed", "error")
                )
                assert lab.app.notices.count(expected_notice) == 1
            release.assert_called_once_with(artifact.operation_id, original)
            assert lab.handler._playground_file_leases.get(original, 0) == 0
            assert original.exists()
            assert not lab.pane.query_one("#audio-play-btn", Button).disabled
            assert lab.pane.query_one("#pause-audio-btn", Button).disabled
            assert lab.pane.query_one("#stop-audio-btn", Button).disabled
            assert lab.pane.query_one("#audio-player-transport").has_class("hidden")
            if exit_code != 0:
                assert ("Playback complete", "information") not in lab.app.notices
                assert (
                    str(lab.pane.query_one("#audio-player-status", Static).content)
                    == "Playback failed"
                )
            await lab.pane._stop_audio_async()
            release.assert_called_once_with(artifact.operation_id, original)
        finally:
            finished.set()
