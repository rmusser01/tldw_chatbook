import pytest

from tldw_chatbook.Audio_Services_Interop.audio_services_scope_service import AudioServicesScopeService
from tldw_chatbook.Audio_Services_Interop.local_audio_services_service import LocalAudioServicesService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeAudioService:
    def __init__(self, backend):
        self.backend = backend
        self.calls = []

    async def get_tts_health(self):
        self.calls.append(("get_tts_health",))
        return {"status": "healthy"}

    async def list_tts_providers(self):
        self.calls.append(("list_tts_providers",))
        return {"providers": {"kokoro": {"available": True}}, "voices": {"kokoro": [{"id": "af_heart"}]}}

    async def list_tts_voices(self, **kwargs):
        self.calls.append(("list_tts_voices", kwargs))
        return {"kokoro": [{"id": "af_heart"}]}

    async def get_audio_streaming_status(self):
        self.calls.append(("get_audio_streaming_status",))
        return {"status": "available", "available_models": ["parakeet-mlx"], "websocket_endpoint": "/api/v1/audio/stream/transcribe", "supported_features": {}}

    async def get_audio_streaming_limits(self):
        self.calls.append(("get_audio_streaming_limits",))
        return {"user_id": "user-1", "tier": "free", "limits": {}, "used_today_minutes": 0.0, "active_streams": 0, "can_start_stream": True}

    async def test_audio_streaming(self):
        self.calls.append(("test_audio_streaming",))
        return {"status": "success", "test_passed": True, "message": "ok"}

    async def create_speech_chat(self, request_data):
        self.calls.append(("create_speech_chat", request_data))
        return {"session_id": "speech-session-1", "assistant_text": "Hi"}

    async def create_audio_speech_job(self, request_data):
        self.calls.append(("create_audio_speech_job", request_data))
        return {"job_id": 42, "status": "queued"}

    async def create_audio_speech(self, request_data):
        self.calls.append(("create_audio_speech", request_data))
        return {"content": b"mp3", "content_type": "audio/mpeg", "filename": "speech.mp3"}

    async def list_audio_speech_job_artifacts(self, job_id):
        self.calls.append(("list_audio_speech_job_artifacts", job_id))
        return {"job_id": job_id, "artifacts": [{"output_id": 9, "download_url": "/download/9"}]}

    async def submit_audio_job(self, request_data):
        self.calls.append(("submit_audio_job", request_data))
        return {"id": 7, "status": "queued"}

    async def get_audio_job(self, job_id):
        self.calls.append(("get_audio_job", job_id))
        return {"id": job_id, "status": "queued"}

    async def stream_audio_job_progress(self, job_id, **kwargs):
        self.calls.append(("stream_audio_job_progress", job_id, kwargs))
        yield {"type": "snapshot", "job_id": job_id}

    async def list_tts_history(self, **kwargs):
        self.calls.append(("list_tts_history", kwargs))
        return {"items": [{"id": 3, "created_at": "2026-04-25T12:00:00Z", "has_text": True}], "limit": 50, "offset": 0}

    async def get_tts_history_entry(self, history_id):
        self.calls.append(("get_tts_history_entry", history_id))
        return {"id": history_id, "created_at": "2026-04-25T12:00:00Z", "has_text": True}

    async def update_tts_history_favorite(self, history_id, request_data):
        self.calls.append(("update_tts_history_favorite", history_id, request_data))
        return {"id": history_id, "favorite": True}

    async def delete_tts_history_entry(self, history_id):
        self.calls.append(("delete_tts_history_entry", history_id))
        return {"id": history_id, "deleted": True}

    async def create_audio_transcription(self, file_path, request_data=None):
        self.calls.append(("create_audio_transcription", file_path, request_data))
        return {"text": "Transcript"}

    async def create_audio_translation(self, file_path, request_data=None):
        self.calls.append(("create_audio_translation", file_path, request_data))
        return {"text": "Translation"}

    async def encode_audio_tokenizer(self, request_data):
        self.calls.append(("encode_audio_tokenizer", request_data))
        return {"tokens": [1], "token_format": "list"}

    async def decode_audio_tokenizer(self, request_data):
        self.calls.append(("decode_audio_tokenizer", request_data))
        return {"content": b"RIFF", "content_type": "audio/wav"}

    async def upload_custom_voice(self, file_path, **kwargs):
        self.calls.append(("upload_custom_voice", file_path, kwargs))
        return {"voice_id": "voice-1", "name": kwargs["name"]}

    async def encode_custom_voice_reference(self, request_data):
        self.calls.append(("encode_custom_voice_reference", request_data))
        return {"voice_id": "voice-1", "provider": "neutts"}

    async def list_custom_voices(self):
        self.calls.append(("list_custom_voices",))
        return {"voices": [{"voice_id": "voice-1", "name": "Narrator"}], "count": 1}

    async def get_custom_voice(self, voice_id):
        self.calls.append(("get_custom_voice", voice_id))
        return {"voice_id": voice_id, "name": "Narrator"}

    async def preview_custom_voice(self, voice_id, **kwargs):
        self.calls.append(("preview_custom_voice", voice_id, kwargs))
        return {"content": b"mp3", "filename": f"preview_{voice_id}.mp3"}

    async def delete_custom_voice(self, voice_id):
        self.calls.append(("delete_custom_voice", voice_id))
        return {"voice_id": voice_id, "deleted": True}

    async def parse_audiobook_source(self, request_data):
        self.calls.append(("parse_audiobook_source", request_data))
        return {"project_id": "abk_1", "chapters": [{"chapter_id": "ch_1"}]}

    async def create_audiobook_job(self, request_data):
        self.calls.append(("create_audiobook_job", request_data))
        return {"job_id": 77, "project_id": "abk_1", "status": "queued"}

    async def get_audiobook_job_status(self, job_id):
        self.calls.append(("get_audiobook_job_status", job_id))
        return {"job_id": job_id, "project_id": "abk_1", "status": "queued"}

    async def list_audiobook_job_artifacts(self, job_id):
        self.calls.append(("list_audiobook_job_artifacts", job_id))
        return {"project_id": "abk_1", "artifacts": [{"output_id": 11, "download_url": "/download/11"}]}

    async def list_audiobook_projects(self, **kwargs):
        self.calls.append(("list_audiobook_projects", kwargs))
        return {"projects": [{"project_db_id": 12, "project_id": "abk_1", "title": "Book"}]}

    async def get_audiobook_project(self, project_ref):
        self.calls.append(("get_audiobook_project", project_ref))
        return {"project": {"project_db_id": 12, "project_id": project_ref, "title": "Book"}}

    async def list_audiobook_project_chapters(self, project_ref, **kwargs):
        self.calls.append(("list_audiobook_project_chapters", project_ref, kwargs))
        return {"project_id": project_ref, "chapters": [{"id": 1, "chapter_index": 0}]}

    async def list_audiobook_project_artifacts(self, project_ref, **kwargs):
        self.calls.append(("list_audiobook_project_artifacts", project_ref, kwargs))
        return {"project_id": project_ref, "artifacts": [{"output_id": 12, "download_url": "/download/12"}]}

    async def create_audiobook_voice_profile(self, request_data):
        self.calls.append(("create_audiobook_voice_profile", request_data))
        return {"profile_id": "vp_1", "name": "Narrator"}

    async def list_audiobook_voice_profiles(self):
        self.calls.append(("list_audiobook_voice_profiles",))
        return {"profiles": [{"profile_id": "vp_1", "name": "Narrator"}]}

    async def delete_audiobook_voice_profile(self, profile_id):
        self.calls.append(("delete_audiobook_voice_profile", profile_id))
        return {"profile_id": profile_id, "deleted": True}

    async def export_audiobook_subtitles(self, request_data):
        self.calls.append(("export_audiobook_subtitles", request_data))
        return {"content": b"1\nHello", "content_type": "text/plain"}


class WebSocketCapableAudioService(FakeAudioService):
    supports_websocket_streaming = True

    async def stream_audio_transcription_websocket(self, **kwargs):
        self.calls.append(("stream_audio_transcription_websocket", kwargs))
        yield {"type": "partial", "text": "Hello"}


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source=action_id.rsplit(".", 1)[-1],
                authority_owner=action_id.rsplit(".", 1)[-1],
            )


@pytest.mark.asyncio
async def test_audio_services_scope_service_routes_and_normalizes_source_records():
    local = FakeAudioService("local")
    server = FakeAudioService("server")
    policy = FakePolicyEnforcer()
    scope = AudioServicesScopeService(local_service=local, server_service=server, policy_enforcer=policy)

    local_health = await scope.get_tts_health(mode="local")
    providers = await scope.list_tts_providers(mode="server")
    voices = await scope.list_tts_voices(mode="server", provider="kokoro")
    streaming_status = await scope.get_audio_streaming_status(mode="server")
    streaming_limits = await scope.get_audio_streaming_limits(mode="server")
    streaming_test = await scope.test_audio_streaming(mode="server")
    speech_chat = await scope.create_speech_chat(
        mode="server",
        request_data={"input_audio": "UklGRg==", "input_audio_format": "wav", "llm_config": {"model": "gpt-test"}},
    )
    speech_job = await scope.create_audio_speech_job(mode="server", request_data={"model": "kokoro", "input": "Hello"})
    job = await scope.get_audio_job(mode="server", job_id=42)
    history = await scope.list_tts_history(mode="server", provider="kokoro")
    custom_voices = await scope.list_custom_voices(mode="server")
    projects = await scope.list_audiobook_projects(mode="server", limit=10, offset=0)

    assert local_health["record_id"] == "local:audio:health"
    assert providers["record_id"] == "server:audio:providers"
    assert voices["record_id"] == "server:audio:voices"
    assert streaming_status["record_id"] == "server:audio_streaming:status"
    assert streaming_limits["record_id"] == "server:audio_streaming:limits"
    assert streaming_test["record_id"] == "server:audio_streaming:test"
    assert speech_chat["record_id"] == "server:audio_speech_chat:speech-session-1"
    assert speech_job["record_id"] == "server:audio_speech_job:42"
    assert job["record_id"] == "server:audio_job:42"
    assert history["items"][0]["record_id"] == "server:audio_history:3"
    assert custom_voices["voices"][0]["record_id"] == "server:audio_voice:voice-1"
    assert projects["projects"][0]["record_id"] == "server:audiobook_project:abk_1"
    assert local.calls == [("get_tts_health",)]
    assert server.calls == [
        ("list_tts_providers",),
        ("list_tts_voices", {"provider": "kokoro"}),
        ("get_audio_streaming_status",),
        ("get_audio_streaming_limits",),
        ("test_audio_streaming",),
        ("create_speech_chat", {"input_audio": "UklGRg==", "input_audio_format": "wav", "llm_config": {"model": "gpt-test"}}),
        ("create_audio_speech_job", {"model": "kokoro", "input": "Hello"}),
        ("get_audio_job", 42),
        ("list_tts_history", {"provider": "kokoro"}),
        ("list_custom_voices",),
        ("list_audiobook_projects", {"limit": 10, "offset": 0}),
    ]
    assert policy.calls == [
        "audio.health.observe.local",
        "audio.providers.list.server",
        "audio.voices.list.server",
        "audio.streaming.status.server",
        "audio.streaming.detail.server",
        "audio.streaming.launch.server",
        "audio.speech_chat.launch.server",
        "audio.speech.launch.server",
        "audio.jobs.detail.server",
        "audio.history.list.server",
        "audio.voices.list.server",
        "audiobooks.projects.list.server",
    ]


@pytest.mark.asyncio
async def test_audio_services_scope_service_blocks_denied_action_before_dispatch():
    local = FakeAudioService("local")
    scope = AudioServicesScopeService(
        local_service=local,
        server_service=None,
        policy_enforcer=FakePolicyEnforcer("authority_denied"),
    )

    with pytest.raises(PolicyDeniedError):
        await scope.list_tts_providers(mode="local")

    assert local.calls == []


@pytest.mark.asyncio
async def test_audio_services_scope_service_blocks_local_advanced_audio_before_dispatch():
    local = FakeAudioService("local")
    policy = FakePolicyEnforcer()
    scope = AudioServicesScopeService(
        local_service=local,
        server_service=FakeAudioService("server"),
        policy_enforcer=policy,
    )

    with pytest.raises(NotImplementedError, match="Local advanced audio"):
        await scope.create_audio_speech_job(mode="local", request_data={"input": "hello"})
    with pytest.raises(NotImplementedError, match="Local advanced audio"):
        await scope.submit_audio_job(mode="local", request_data={"url": "https://example.com/a.mp3"})
    with pytest.raises(NotImplementedError, match="Local advanced audio"):
        await scope.create_audio_transcription(mode="local", file_path="sample.wav")
    with pytest.raises(NotImplementedError, match="Local advanced audio"):
        await scope.upload_custom_voice(mode="local", file_path="voice.wav", name="Narrator")
    with pytest.raises(NotImplementedError, match="Local advanced audio"):
        await scope.parse_audiobook_source(mode="local", request_data={"source": {"raw_text": "Hello"}})

    assert local.calls == []
    assert policy.calls == [
        "audio.speech.launch.local",
        "audio.jobs.create.local",
        "audio.transcriptions.launch.local",
        "audio.voices.create.local",
        "audiobooks.parse.launch.local",
    ]


def test_audio_services_scope_service_reports_known_unsupported_capabilities():
    scope = AudioServicesScopeService(local_service=None, server_service=None)

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "audio.local_jobs_and_advanced_generation.local",
            "source": "local",
            "supported": False,
            "reason_code": "existing_ui_owned_surface",
            "user_message": "Local immediate TTS generation and TTS history are available through the source-aware seam; local speech jobs, STT/translation, tokenizer, custom voices, and audiobook generation remain in older UI-owned surfaces or are not implemented locally yet.",
            "affected_action_ids": [
                "audio.voices.create.local",
                "audio.voices.delete.local",
                "audio.voices.detail.local",
                "audio.voices.launch.local",
                "audio.voices.preview.local",
                "audio.speech_jobs.detail.local",
                "audio.jobs.create.local",
                "audio.jobs.detail.local",
                "audio.jobs.observe.local",
                "audio.transcriptions.launch.local",
                "audio.translations.launch.local",
                "audio.tokenizer.launch.local",
                "audiobooks.artifacts.list.local",
                "audiobooks.chapters.list.local",
                "audiobooks.jobs.create.local",
                "audiobooks.jobs.detail.local",
                "audiobooks.jobs.observe.local",
                "audiobooks.parse.launch.local",
                "audiobooks.projects.detail.local",
                "audiobooks.projects.list.local",
                "audiobooks.subtitles.export.local",
                "audiobooks.voice_profiles.create.local",
                "audiobooks.voice_profiles.delete.local",
                "audiobooks.voice_profiles.list.local",
            ],
        },
        {
            "operation_id": "audio.streaming_rest.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Audio streaming REST status, limits, test, and speech-chat helpers are active-server owned in this seam.",
            "affected_action_ids": [
                "audio.streaming.status.server",
                "audio.streaming.detail.server",
                "audio.streaming.launch.server",
                "audio.speech_chat.launch.server",
            ],
        },
    ]
    assert server_report == [
        {
            "operation_id": "audio.websocket_streaming.server",
            "source": "server",
            "supported": False,
            "reason_code": "client_adapter_missing",
            "user_message": "The server exposes websocket speech/chat streaming endpoints, but this Chatbook audio adapter only exposes REST status, limits, test, non-streaming speech-chat, and SSE job observation.",
            "affected_action_ids": [],
        },
        {
            "operation_id": "audio.admin_job_controls.server",
            "source": "server",
            "supported": False,
            "reason_code": "out_of_scope_admin_surface",
            "user_message": "Admin audio-job controls are outside Chatbook client parity for this slice.",
            "affected_action_ids": [],
        },
    ]


def test_audio_services_scope_service_omits_websocket_gap_for_capable_adapter():
    scope = AudioServicesScopeService(
        local_service=None,
        server_service=WebSocketCapableAudioService("server"),
    )

    report = scope.list_unsupported_capabilities(mode="server")

    assert [item["operation_id"] for item in report] == [
        "audio.admin_job_controls.server",
    ]


@pytest.mark.asyncio
async def test_audio_services_scope_service_routes_local_speech_and_history(tmp_path):
    async def fake_generator(**kwargs):
        return b"audio"

    local = LocalAudioServicesService(
        tts_audio_generator=fake_generator,
        history_store_path=tmp_path / "audio-history.json",
    )
    policy = FakePolicyEnforcer()
    scope = AudioServicesScopeService(local_service=local, server_service=None, policy_enforcer=policy)

    speech = await scope.create_audio_speech(
        mode="local",
        request_data={"model": "kokoro", "input": "Local speech", "response_format": "wav"},
    )
    history = await scope.list_tts_history(mode="local")
    detail = await scope.get_tts_history_entry(mode="local", history_id=1)
    updated = await scope.update_tts_history_favorite(
        mode="local",
        history_id=1,
        request_data={"favorite": True},
    )
    deleted = await scope.delete_tts_history_entry(mode="local", history_id=1)

    assert speech["record_id"] == "local:audio:speech"
    assert speech["history_id"] == 1
    assert history["items"][0]["record_id"] == "local:audio_history:1"
    assert detail["record_id"] == "local:audio_history:1"
    assert detail["text"] == "Local speech"
    assert updated["favorite"] is True
    assert deleted["deleted"] is True
    assert policy.calls == [
        "audio.speech.launch.local",
        "audio.history.list.local",
        "audio.history.detail.local",
        "audio.history.update.local",
        "audio.history.delete.local",
    ]


@pytest.mark.asyncio
async def test_audio_services_scope_service_exposes_audio_and_audiobook_operations():
    server = FakeAudioService("server")
    policy = FakePolicyEnforcer()
    scope = AudioServicesScopeService(local_service=None, server_service=server, policy_enforcer=policy)

    speech = await scope.create_audio_speech(mode="server", request_data={"model": "kokoro", "input": "Hello"})
    artifacts = await scope.list_audio_speech_job_artifacts(mode="server", job_id=42)
    submitted = await scope.submit_audio_job(mode="server", request_data={"url": "https://example.com/a.mp3"})
    progress = [event async for event in scope.stream_audio_job_progress(mode="server", job_id=7, after_id=1)]
    history_detail = await scope.get_tts_history_entry(mode="server", history_id=3)
    history_update = await scope.update_tts_history_favorite(mode="server", history_id=3, request_data={"favorite": True})
    history_delete = await scope.delete_tts_history_entry(mode="server", history_id=3)
    transcription = await scope.create_audio_transcription(mode="server", file_path="sample.wav", request_data={"model": "whisper-1"})
    translation = await scope.create_audio_translation(mode="server", file_path="sample.wav", request_data={"model": "whisper-1"})
    encoded = await scope.encode_audio_tokenizer(mode="server", request_data={"audio_base64": "UklGRg=="})
    decoded = await scope.decode_audio_tokenizer(mode="server", request_data={"tokens": [1]})
    uploaded = await scope.upload_custom_voice(mode="server", file_path="voice.wav", name="Narrator")
    voice_encoded = await scope.encode_custom_voice_reference(mode="server", request_data={"voice_id": "voice-1"})
    voice_detail = await scope.get_custom_voice(mode="server", voice_id="voice-1")
    voice_preview = await scope.preview_custom_voice(mode="server", voice_id="voice-1", text="Preview")
    voice_delete = await scope.delete_custom_voice(mode="server", voice_id="voice-1")
    parsed = await scope.parse_audiobook_source(mode="server", request_data={"source": {"raw_text": "Hello"}})
    audiobook_job = await scope.create_audiobook_job(mode="server", request_data={"project_title": "Book"})
    audiobook_status = await scope.get_audiobook_job_status(mode="server", job_id=77)
    audiobook_artifacts = await scope.list_audiobook_job_artifacts(mode="server", job_id=77)
    project = await scope.get_audiobook_project(mode="server", project_ref="abk_1")
    chapters = await scope.list_audiobook_project_chapters(mode="server", project_ref="abk_1", limit=25, offset=0)
    project_artifacts = await scope.list_audiobook_project_artifacts(mode="server", project_ref="abk_1", limit=25, offset=0)
    profile = await scope.create_audiobook_voice_profile(mode="server", request_data={"name": "Narrator"})
    profiles = await scope.list_audiobook_voice_profiles(mode="server")
    profile_delete = await scope.delete_audiobook_voice_profile(mode="server", profile_id="vp_1")
    subtitles = await scope.export_audiobook_subtitles(mode="server", request_data={"format": "srt"})

    assert speech["record_id"] == "server:audio:speech"
    assert artifacts["artifacts"][0]["record_id"] == "server:audio_speech_artifact:9"
    assert submitted["record_id"] == "server:audio_job:7"
    assert progress[0]["backend"] == "server"
    assert history_detail["record_id"] == "server:audio_history:3"
    assert history_update["record_id"] == "server:audio_history:3"
    assert history_delete["record_id"] == "server:audio_history:3"
    assert transcription["record_id"] == "server:audio:transcription"
    assert translation["record_id"] == "server:audio:translation"
    assert encoded["record_id"] == "server:audio:tokenizer_encode"
    assert decoded["record_id"] == "server:audio:tokenizer_decode"
    assert uploaded["record_id"] == "server:audio_voice:voice-1"
    assert voice_encoded["record_id"] == "server:audio_voice:voice-1"
    assert voice_detail["record_id"] == "server:audio_voice:voice-1"
    assert voice_preview["record_id"] == "server:audio_voice_preview:voice-1"
    assert voice_delete["record_id"] == "server:audio_voice:voice-1"
    assert parsed["record_id"] == "server:audiobook_parse:abk_1"
    assert audiobook_job["record_id"] == "server:audiobook_job:77"
    assert audiobook_status["record_id"] == "server:audiobook_job:77"
    assert audiobook_artifacts["artifacts"][0]["record_id"] == "server:audiobook_artifact:11"
    assert project["project"]["record_id"] == "server:audiobook_project:abk_1"
    assert chapters["chapters"][0]["record_id"] == "server:audiobook_chapter:1"
    assert project_artifacts["artifacts"][0]["record_id"] == "server:audiobook_artifact:12"
    assert profile["record_id"] == "server:audiobook_voice_profile:vp_1"
    assert profiles["profiles"][0]["record_id"] == "server:audiobook_voice_profile:vp_1"
    assert profile_delete["record_id"] == "server:audiobook_voice_profile:vp_1"
    assert subtitles["record_id"] == "server:audiobook_subtitles:export"
    assert policy.calls == [
        "audio.speech.launch.server",
        "audio.speech_jobs.detail.server",
        "audio.jobs.create.server",
        "audio.jobs.observe.server",
        "audio.history.detail.server",
        "audio.history.update.server",
        "audio.history.delete.server",
        "audio.transcriptions.launch.server",
        "audio.translations.launch.server",
        "audio.tokenizer.launch.server",
        "audio.tokenizer.launch.server",
        "audio.voices.create.server",
        "audio.voices.launch.server",
        "audio.voices.detail.server",
        "audio.voices.preview.server",
        "audio.voices.delete.server",
        "audiobooks.parse.launch.server",
        "audiobooks.jobs.create.server",
        "audiobooks.jobs.detail.server",
        "audiobooks.artifacts.list.server",
        "audiobooks.projects.detail.server",
        "audiobooks.chapters.list.server",
        "audiobooks.artifacts.list.server",
        "audiobooks.voice_profiles.create.server",
        "audiobooks.voice_profiles.list.server",
        "audiobooks.voice_profiles.delete.server",
        "audiobooks.subtitles.export.server",
    ]
