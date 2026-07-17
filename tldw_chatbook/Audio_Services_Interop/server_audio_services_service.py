"""Policy-gated active-server audio/speech/audiobook service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncGenerator, Literal, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
if TYPE_CHECKING:
    from ..tldw_api import (
        AudioTokenizerDecodeRequest,
        AudioTokenizerEncodeRequest,
        AudioTranscriptionRequest,
        AudioTranslationRequest,
        AudiobookJobRequest,
        AudiobookParseRequest,
        OpenAISpeechRequest,
        SpeechChatRequest,
        SubmitAudioJobRequest,
        SubtitleExportRequest,
        TLDWAPIClient,
        TTSHistoryFavoriteUpdate,
        VoiceEncodeRequest,
        VoiceProfileCreateRequest,
    )


class ServerAudioServicesService:
    """Execute stable REST-backed audio operations against the active server."""

    supports_websocket_streaming = False

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        client_provider: Any | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.client = client
        self.client_provider = client_provider
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerAudioServicesService":
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerAudioServicesService":
        return cls(
            client=None,
            client_provider=provider,
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server audio/speech operations.")

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Server audio action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @classmethod
    def _dump(cls, response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="python")
        if isinstance(response, list):
            return [cls._dump(item) for item in response]
        if isinstance(response, dict):
            return dict(response)
        return response

    async def get_tts_health(self) -> dict[str, Any]:
        self._enforce("audio.health.observe.server")
        return self._dump(await self._require_client().get_tts_health())

    async def get_stt_health(self, *, model: str | None = None, warm: bool = False) -> dict[str, Any]:
        self._enforce("audio.health.observe.server")
        return self._dump(await self._require_client().get_stt_health(model=model, warm=warm))

    async def list_tts_providers(self) -> dict[str, Any]:
        self._enforce("audio.providers.list.server")
        return self._dump(await self._require_client().list_tts_providers())

    async def list_tts_voices(self, *, provider: str | None = None) -> dict[str, Any]:
        self._enforce("audio.voices.list.server")
        return self._dump(await self._require_client().list_tts_voices(provider=provider))

    async def get_audio_streaming_status(self) -> dict[str, Any]:
        self._enforce("audio.streaming.status.server")
        return self._dump(await self._require_client().get_audio_streaming_status())

    async def get_audio_streaming_limits(self) -> dict[str, Any]:
        self._enforce("audio.streaming.detail.server")
        return self._dump(await self._require_client().get_audio_streaming_limits())

    async def test_audio_streaming(self) -> dict[str, Any]:
        self._enforce("audio.streaming.launch.server")
        return self._dump(await self._require_client().test_audio_streaming())

    async def create_speech_chat(self, request_data: SpeechChatRequest) -> dict[str, Any]:
        self._enforce("audio.speech_chat.launch.server")
        return self._dump(await self._require_client().create_speech_chat(request_data))

    async def create_audio_speech(self, request_data: OpenAISpeechRequest) -> dict[str, Any]:
        self._enforce("audio.speech.launch.server")
        return self._dump(await self._require_client().create_audio_speech(request_data))

    async def create_audio_speech_job(self, request_data: OpenAISpeechRequest) -> dict[str, Any]:
        self._enforce("audio.speech.launch.server")
        return self._dump(await self._require_client().create_audio_speech_job(request_data))

    async def list_audio_speech_job_artifacts(self, job_id: int) -> dict[str, Any]:
        self._enforce("audio.speech_jobs.detail.server")
        return self._dump(await self._require_client().list_audio_speech_job_artifacts(job_id))

    async def submit_audio_job(self, request_data: SubmitAudioJobRequest) -> dict[str, Any]:
        self._enforce("audio.jobs.create.server")
        return self._dump(await self._require_client().submit_audio_job(request_data))

    async def get_audio_job(self, job_id: int) -> dict[str, Any]:
        self._enforce("audio.jobs.detail.server")
        return self._dump(await self._require_client().get_audio_job(job_id))

    async def stream_audio_job_progress(
        self,
        job_id: int,
        *,
        after_id: int = 0,
    ) -> AsyncGenerator[dict[str, Any], None]:
        self._enforce("audio.jobs.observe.server")
        async for event in self._require_client().stream_audio_job_progress(job_id, after_id=after_id):
            yield dict(event or {})

    async def list_tts_history(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce("audio.history.list.server")
        return self._dump(await self._require_client().list_tts_history(**kwargs))

    async def get_tts_history_entry(self, history_id: int) -> dict[str, Any]:
        self._enforce("audio.history.detail.server")
        return self._dump(await self._require_client().get_tts_history_entry(history_id))

    async def update_tts_history_favorite(
        self,
        history_id: int,
        request_data: TTSHistoryFavoriteUpdate,
    ) -> dict[str, Any]:
        self._enforce("audio.history.update.server")
        return self._dump(await self._require_client().update_tts_history_favorite(history_id, request_data))

    async def delete_tts_history_entry(self, history_id: int) -> dict[str, Any]:
        self._enforce("audio.history.delete.server")
        return self._dump(await self._require_client().delete_tts_history_entry(history_id))

    async def create_audio_transcription(
        self,
        file_path: str,
        request_data: AudioTranscriptionRequest | None = None,
    ) -> dict[str, Any]:
        self._enforce("audio.transcriptions.launch.server")
        return self._dump(await self._require_client().create_audio_transcription(file_path, request_data))

    async def create_audio_translation(
        self,
        file_path: str,
        request_data: AudioTranslationRequest | None = None,
    ) -> dict[str, Any]:
        self._enforce("audio.translations.launch.server")
        return self._dump(await self._require_client().create_audio_translation(file_path, request_data))

    async def encode_audio_tokenizer(self, request_data: AudioTokenizerEncodeRequest) -> dict[str, Any]:
        self._enforce("audio.tokenizer.launch.server")
        return self._dump(await self._require_client().encode_audio_tokenizer(request_data))

    async def encode_audio_tokenizer_file(
        self,
        file_path: str,
        *,
        tokenizer_model: str | None = None,
        token_format: Literal["list", "base64"] = "list",
        sample_rate: int | None = None,
    ) -> dict[str, Any]:
        self._enforce("audio.tokenizer.launch.server")
        return self._dump(
            await self._require_client().encode_audio_tokenizer_file(
                file_path,
                tokenizer_model=tokenizer_model,
                token_format=token_format,
                sample_rate=sample_rate,
            )
        )

    async def decode_audio_tokenizer(self, request_data: AudioTokenizerDecodeRequest) -> dict[str, Any]:
        self._enforce("audio.tokenizer.launch.server")
        return self._dump(await self._require_client().decode_audio_tokenizer(request_data))

    async def upload_custom_voice(
        self,
        file_path: str,
        *,
        name: str,
        description: str | None = None,
        provider: str = "vibevoice",
        reference_text: str | None = None,
    ) -> dict[str, Any]:
        self._enforce("audio.voices.create.server")
        return self._dump(
            await self._require_client().upload_custom_voice(
                file_path,
                name=name,
                description=description,
                provider=provider,
                reference_text=reference_text,
            )
        )

    async def encode_custom_voice_reference(self, request_data: VoiceEncodeRequest) -> dict[str, Any]:
        self._enforce("audio.voices.launch.server")
        return self._dump(await self._require_client().encode_custom_voice_reference(request_data))

    async def list_custom_voices(self) -> dict[str, Any]:
        self._enforce("audio.voices.list.server")
        return self._dump(await self._require_client().list_custom_voices())

    async def get_custom_voice(self, voice_id: str) -> dict[str, Any]:
        self._enforce("audio.voices.detail.server")
        return self._dump(await self._require_client().get_custom_voice(voice_id))

    async def delete_custom_voice(self, voice_id: str) -> dict[str, Any]:
        self._enforce("audio.voices.delete.server")
        return self._dump(await self._require_client().delete_custom_voice(voice_id))

    async def preview_custom_voice(self, voice_id: str, *, text: str = "Hello, this is a preview of your custom voice.") -> dict[str, Any]:
        self._enforce("audio.voices.preview.server")
        return self._dump(await self._require_client().preview_custom_voice(voice_id, text=text))

    async def parse_audiobook_source(self, request_data: AudiobookParseRequest) -> dict[str, Any]:
        self._enforce("audiobooks.parse.launch.server")
        return self._dump(await self._require_client().parse_audiobook_source(request_data))

    async def create_audiobook_job(self, request_data: AudiobookJobRequest) -> dict[str, Any]:
        self._enforce("audiobooks.jobs.create.server")
        return self._dump(await self._require_client().create_audiobook_job(request_data))

    async def get_audiobook_job_status(self, job_id: int) -> dict[str, Any]:
        self._enforce("audiobooks.jobs.detail.server")
        return self._dump(await self._require_client().get_audiobook_job_status(job_id))

    async def list_audiobook_job_artifacts(self, job_id: int) -> dict[str, Any]:
        self._enforce("audiobooks.artifacts.list.server")
        return self._dump(await self._require_client().list_audiobook_job_artifacts(job_id))

    async def list_audiobook_projects(self, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        self._enforce("audiobooks.projects.list.server")
        return self._dump(await self._require_client().list_audiobook_projects(limit=limit, offset=offset))

    async def get_audiobook_project(self, project_ref: str) -> dict[str, Any]:
        self._enforce("audiobooks.projects.detail.server")
        return self._dump(await self._require_client().get_audiobook_project(project_ref))

    async def list_audiobook_project_chapters(
        self,
        project_ref: str,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._enforce("audiobooks.chapters.list.server")
        return self._dump(
            await self._require_client().list_audiobook_project_chapters(
                project_ref,
                limit=limit,
                offset=offset,
            )
        )

    async def list_audiobook_project_artifacts(
        self,
        project_ref: str,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._enforce("audiobooks.artifacts.list.server")
        return self._dump(
            await self._require_client().list_audiobook_project_artifacts(
                project_ref,
                limit=limit,
                offset=offset,
            )
        )

    async def create_audiobook_voice_profile(self, request_data: VoiceProfileCreateRequest) -> dict[str, Any]:
        self._enforce("audiobooks.voice_profiles.create.server")
        return self._dump(await self._require_client().create_audiobook_voice_profile(request_data))

    async def list_audiobook_voice_profiles(self) -> dict[str, Any]:
        self._enforce("audiobooks.voice_profiles.list.server")
        return self._dump(await self._require_client().list_audiobook_voice_profiles())

    async def delete_audiobook_voice_profile(self, profile_id: str) -> dict[str, Any]:
        self._enforce("audiobooks.voice_profiles.delete.server")
        return self._dump(await self._require_client().delete_audiobook_voice_profile(profile_id))

    async def export_audiobook_subtitles(self, request_data: SubtitleExportRequest) -> dict[str, Any]:
        self._enforce("audiobooks.subtitles.export.server")
        return self._dump(await self._require_client().export_audiobook_subtitles(request_data))
