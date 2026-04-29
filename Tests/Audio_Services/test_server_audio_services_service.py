from unittest.mock import Mock

import pytest

from tldw_chatbook.Audio_Services_Interop.server_audio_services_service import ServerAudioServicesService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError
from tldw_chatbook.tldw_api import (
    AudioTokenizerDecodeRequest,
    AudioTokenizerEncodeRequest,
    AudioTranscriptionRequest,
    AudioTranslationRequest,
    OpenAISpeechRequest,
    ReadingExportResponse,
    SpeechChatLLMConfig,
    SpeechChatRequest,
    SubmitAudioJobRequest,
    VoiceEncodeRequest,
)


class FakeAudioClient:
    def __init__(self):
        self.calls = []

    async def get_tts_health(self):
        self.calls.append(("get_tts_health",))
        return {"status": "healthy", "providers": {"kokoro": {"status": "enabled"}}}

    async def get_stt_health(self, **kwargs):
        self.calls.append(("get_stt_health", kwargs))
        return {"status": "healthy", "model": kwargs.get("model")}

    async def list_tts_providers(self):
        self.calls.append(("list_tts_providers",))
        return {"providers": {"kokoro": {"available": True}}, "voices": {"kokoro": [{"id": "af_heart"}]}}

    async def list_tts_voices(self, **kwargs):
        self.calls.append(("list_tts_voices", kwargs))
        return {"kokoro": [{"id": "af_heart", "name": "Heart"}]}

    async def get_audio_streaming_status(self):
        self.calls.append(("get_audio_streaming_status",))
        return {"status": "available", "available_models": ["parakeet-mlx"], "websocket_endpoint": "/api/v1/audio/stream/transcribe", "supported_features": {}}

    async def get_audio_streaming_limits(self):
        self.calls.append(("get_audio_streaming_limits",))
        return {"user_id": "user-1", "tier": "free", "limits": {}, "used_today_minutes": 0.0, "remaining_minutes": None, "active_streams": 0, "can_start_stream": True}

    async def test_audio_streaming(self):
        self.calls.append(("test_audio_streaming",))
        return {"status": "success", "test_passed": True, "message": "ok"}

    async def create_speech_chat(self, request_data):
        self.calls.append(("create_speech_chat", request_data))
        return {"session_id": "speech-session-1", "assistant_text": "Hi", "user_transcript": "Hello", "output_audio": "bXAz", "output_audio_mime_type": "audio/mpeg"}

    async def create_audio_speech(self, request_data):
        self.calls.append(("create_audio_speech", request_data))
        return ReadingExportResponse(content=b"mp3", content_type="audio/mpeg", filename="speech.mp3")

    async def create_audio_speech_job(self, request_data):
        self.calls.append(("create_audio_speech_job", request_data))
        return {"job_id": 42, "status": "queued"}

    async def list_audio_speech_job_artifacts(self, job_id):
        self.calls.append(("list_audio_speech_job_artifacts", job_id))
        return {
            "job_id": job_id,
            "artifacts": [
                {
                    "output_id": 9,
                    "format": "mp3",
                    "type": "tts_audio",
                    "title": "Speech",
                    "download_url": "/download/9",
                }
            ],
        }

    async def submit_audio_job(self, request_data):
        self.calls.append(("submit_audio_job", request_data))
        return {"id": 7, "domain": "audio", "queue": "default", "job_type": "audio_download", "status": "queued"}

    async def get_audio_job(self, job_id):
        self.calls.append(("get_audio_job", job_id))
        return {"id": job_id, "job_type": "audio_download", "status": "queued", "priority": 5, "retry_count": 0, "max_retries": 3}

    async def stream_audio_job_progress(self, job_id, **kwargs):
        self.calls.append(("stream_audio_job_progress", job_id, kwargs))
        yield {"type": "snapshot", "job_id": job_id}
        yield {"type": "complete", "job_id": job_id}

    async def list_tts_history(self, **kwargs):
        self.calls.append(("list_tts_history", kwargs))
        return {"items": [{"id": 3, "created_at": "2026-04-25T12:00:00Z", "has_text": True}], "limit": 50, "offset": 0}

    async def get_tts_history_entry(self, history_id):
        self.calls.append(("get_tts_history_entry", history_id))
        return {"id": history_id, "created_at": "2026-04-25T12:00:00Z", "has_text": True, "text": "Hello"}

    async def create_audio_transcription(self, file_path, request_data=None):
        self.calls.append(("create_audio_transcription", file_path, request_data))
        return {"text": "Transcript", "language": "en"}

    async def create_audio_translation(self, file_path, request_data=None):
        self.calls.append(("create_audio_translation", file_path, request_data))
        return {"text": "Translation", "language": "en"}

    async def encode_audio_tokenizer(self, request_data):
        self.calls.append(("encode_audio_tokenizer", request_data))
        return {"tokens": [1, 2, 3], "token_format": "list", "sample_rate": 24000, "tokenizer_model": "qwen3", "duration_seconds": 1.0}

    async def decode_audio_tokenizer(self, request_data):
        self.calls.append(("decode_audio_tokenizer", request_data))
        return ReadingExportResponse(content=b"RIFF", content_type="audio/wav")

    async def upload_custom_voice(self, file_path, **kwargs):
        self.calls.append(("upload_custom_voice", file_path, kwargs))
        return {"voice_id": "voice-1", "name": kwargs["name"], "provider": kwargs.get("provider")}

    async def encode_custom_voice_reference(self, request_data):
        self.calls.append(("encode_custom_voice_reference", request_data))
        return {"voice_id": request_data.voice_id, "provider": request_data.provider, "cached": False}

    async def list_custom_voices(self):
        self.calls.append(("list_custom_voices",))
        return {"voices": [{"voice_id": "voice-1", "name": "Narrator"}], "count": 1}

    async def get_custom_voice(self, voice_id):
        self.calls.append(("get_custom_voice", voice_id))
        return {"voice_id": voice_id, "name": "Narrator"}

    async def preview_custom_voice(self, voice_id, **kwargs):
        self.calls.append(("preview_custom_voice", voice_id, kwargs))
        return ReadingExportResponse(content=b"mp3", content_type="audio/mpeg", filename=f"preview_{voice_id}.mp3")

    async def delete_custom_voice(self, voice_id):
        self.calls.append(("delete_custom_voice", voice_id))
        return {"message": "Voice deleted successfully", "voice_id": voice_id}


@pytest.mark.asyncio
async def test_server_audio_services_service_routes_core_audio_with_policy():
    client = FakeAudioClient()
    policy = Mock()
    service = ServerAudioServicesService(client=client, policy_enforcer=policy)

    health = await service.get_tts_health()
    stt_health = await service.get_stt_health(model="whisper-1", warm=True)
    providers = await service.list_tts_providers()
    voices = await service.list_tts_voices(provider="kokoro")
    streaming_status = await service.get_audio_streaming_status()
    streaming_limits = await service.get_audio_streaming_limits()
    streaming_test = await service.test_audio_streaming()
    speech_chat = await service.create_speech_chat(
        SpeechChatRequest(
            input_audio="UklGRg==",
            input_audio_format="wav",
            llm_config=SpeechChatLLMConfig(model="gpt-test"),
        )
    )
    speech = await service.create_audio_speech(OpenAISpeechRequest(model="kokoro", input="Hello"))
    speech_job = await service.create_audio_speech_job(OpenAISpeechRequest(model="kokoro", input="Long text"))
    speech_artifacts = await service.list_audio_speech_job_artifacts(42)
    submitted = await service.submit_audio_job(SubmitAudioJobRequest(url="https://example.com/audio.mp3"))
    job = await service.get_audio_job(7)
    progress = [event async for event in service.stream_audio_job_progress(7, after_id=10)]
    history = await service.list_tts_history(provider="kokoro")
    detail = await service.get_tts_history_entry(3)
    transcription = await service.create_audio_transcription("sample.wav", AudioTranscriptionRequest(model="whisper-1"))
    translation = await service.create_audio_translation("sample.wav", AudioTranslationRequest(model="whisper-1"))
    encoded = await service.encode_audio_tokenizer(AudioTokenizerEncodeRequest(audio_base64="UklGRg=="))
    decoded = await service.decode_audio_tokenizer(AudioTokenizerDecodeRequest(tokens=[1, 2, 3]))
    uploaded = await service.upload_custom_voice("voice.wav", name="Narrator", provider="vibevoice")
    voice_encoded = await service.encode_custom_voice_reference(VoiceEncodeRequest(voice_id="voice-1"))
    custom_voices = await service.list_custom_voices()
    custom_voice = await service.get_custom_voice("voice-1")
    preview = await service.preview_custom_voice("voice-1", text="Preview")
    deleted = await service.delete_custom_voice("voice-1")

    assert health["status"] == "healthy"
    assert stt_health["model"] == "whisper-1"
    assert providers["providers"]["kokoro"]["available"] is True
    assert voices["kokoro"][0]["id"] == "af_heart"
    assert streaming_status["status"] == "available"
    assert streaming_limits["can_start_stream"] is True
    assert streaming_test["test_passed"] is True
    assert speech_chat["session_id"] == "speech-session-1"
    assert speech["content"] == b"mp3"
    assert speech_job["job_id"] == 42
    assert speech_artifacts["artifacts"][0]["output_id"] == 9
    assert submitted["id"] == 7
    assert job["id"] == 7
    assert progress[-1]["type"] == "complete"
    assert history["items"][0]["id"] == 3
    assert detail["text"] == "Hello"
    assert transcription["text"] == "Transcript"
    assert translation["text"] == "Translation"
    assert encoded["tokens"] == [1, 2, 3]
    assert decoded["content"] == b"RIFF"
    assert uploaded["voice_id"] == "voice-1"
    assert voice_encoded["voice_id"] == "voice-1"
    assert custom_voices["count"] == 1
    assert custom_voice["voice_id"] == "voice-1"
    assert preview["filename"] == "preview_voice-1.mp3"
    assert deleted["voice_id"] == "voice-1"
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "audio.health.observe.server",
        "audio.health.observe.server",
        "audio.providers.list.server",
        "audio.voices.list.server",
        "audio.streaming.status.server",
        "audio.streaming.detail.server",
        "audio.streaming.launch.server",
        "audio.speech_chat.launch.server",
        "audio.speech.launch.server",
        "audio.speech.launch.server",
        "audio.speech_jobs.detail.server",
        "audio.jobs.create.server",
        "audio.jobs.detail.server",
        "audio.jobs.observe.server",
        "audio.history.list.server",
        "audio.history.detail.server",
        "audio.transcriptions.launch.server",
        "audio.translations.launch.server",
        "audio.tokenizer.launch.server",
        "audio.tokenizer.launch.server",
        "audio.voices.create.server",
        "audio.voices.launch.server",
        "audio.voices.list.server",
        "audio.voices.detail.server",
        "audio.voices.preview.server",
        "audio.voices.delete.server",
    ]


@pytest.mark.asyncio
async def test_server_audio_services_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_auth_required",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeAudioClient()
    service = ServerAudioServicesService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.get_tts_health()

    assert exc.value.reason_code == "server_auth_required"
    assert client.calls == []


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


class ExplodingClientProvider:
    def __init__(self):
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        raise AssertionError("provider should not be used when direct client exists")


class FreshClientProvider:
    def __init__(self):
        self.build_calls = 0
        self.clients = []

    def build_client(self):
        self.build_calls += 1
        client = object()
        self.clients.append(client)
        return client


def test_server_audio_services_service_direct_client_takes_precedence_over_provider():
    client = object()
    provider = ExplodingClientProvider()
    service = ServerAudioServicesService(client=client, client_provider=provider)

    assert service._require_client() is client
    assert provider.build_calls == 0


def test_server_audio_services_service_from_server_context_provider_is_lazy():
    client = object()
    provider = FakeClientProvider(client)
    service = ServerAudioServicesService.from_server_context_provider(provider)

    assert isinstance(service, ServerAudioServicesService)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0
    assert service._require_client() is client
    assert service.client is None
    assert provider.build_calls == 1


def test_server_audio_services_service_re_resolves_provider_without_service_local_client_cache():
    provider = FreshClientProvider()
    service = ServerAudioServicesService.from_server_context_provider(provider)

    first_client = service._require_client()
    second_client = service._require_client()

    assert service.client is None
    assert provider.build_calls == 2
    assert first_client is not second_client
    assert provider.clients == [first_client, second_client]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


def test_server_audio_services_service_from_config_returns_provider_backed_service():
    service = ServerAudioServicesService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerAudioServicesService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client
