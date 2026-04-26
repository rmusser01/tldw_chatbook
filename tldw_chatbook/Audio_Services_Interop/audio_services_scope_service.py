"""Source-aware routing for audio, speech, and audiobook operations."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, AsyncGenerator


class AudioServicesBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "audio.central_local_generation.local",
        "source": "local",
        "supported": False,
        "reason_code": "existing_ui_owned_surface",
        "user_message": "Local TTS/STT generation remains in Chatbook's existing TTS/STTS event handlers; this source-aware seam exposes local discovery until those call sites are adopted.",
        "affected_action_ids": [
            "audio.voices.create.local",
            "audio.voices.delete.local",
            "audio.voices.detail.local",
            "audio.voices.launch.local",
            "audio.voices.preview.local",
            "audio.speech.launch.local",
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
        "operation_id": "audio.local_history_artifact_scope.local",
        "source": "local",
        "supported": False,
        "reason_code": "sync_semantics_deferred",
        "user_message": "Local audio history/artifact identity is not yet centralized with server history and artifacts; sync/mirroring remains deferred.",
        "affected_action_ids": [
            "audio.history.list.local",
            "audio.history.detail.local",
            "audio.history.update.local",
            "audio.history.delete.local",
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

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "audio.websocket_streaming.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_followup",
        "user_message": "Server websocket speech/chat streaming is not part of this REST-backed audio seam; REST status, limits, test, and non-streaming speech-chat helpers plus SSE job observation remain available.",
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


class AudioServicesScopeService:
    """Route local and server audio services without merging histories or artifacts."""

    def __init__(self, *, local_service: Any = None, server_service: Any = None, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: AudioServicesBackend | str | None) -> AudioServicesBackend:
        if mode is None:
            return AudioServicesBackend.LOCAL
        if isinstance(mode, AudioServicesBackend):
            return mode
        try:
            return AudioServicesBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid audio services backend: {mode}") from exc

    def _service_for_mode(self, mode: AudioServicesBackend) -> Any:
        service = self.local_service if mode == AudioServicesBackend.LOCAL else self.server_service
        if service is None:
            raise ValueError(f"{mode.value.title()} audio services backend is unavailable.")
        return service

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _dump(payload: Any) -> Any:
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="python")
        if isinstance(payload, dict):
            return dict(payload)
        if isinstance(payload, list):
            return [AudioServicesScopeService._dump(item) for item in payload]
        return payload

    @staticmethod
    def _action_id(resource: str, action: str, mode: AudioServicesBackend) -> str:
        return f"{resource}.{action}.{mode.value}"

    async def _call(
        self,
        *,
        mode: AudioServicesBackend | str | None,
        action_id: str,
        method_name: str,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[AudioServicesBackend, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(action_id.rsplit(".", 1)[0] + f".{normalized_mode.value}")
        service = self._service_for_mode(normalized_mode)
        method = getattr(service, method_name, None)
        if not callable(method):
            raise ValueError(f"{method_name} is unavailable for {normalized_mode.value} audio services.")
        return normalized_mode, self._dump(await self._maybe_await(method(**(kwargs or {}))))

    @staticmethod
    def _with_record_id(mode: AudioServicesBackend, kind: str, payload: dict[str, Any], identifier: Any | None = None) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        if identifier is None:
            identifier = record.get("id") or record.get("job_id") or record.get("voice_id") or record.get("profile_id")
        if identifier is None:
            record.setdefault("record_id", f"{mode.value}:audio:{kind}")
        else:
            record.setdefault("record_id", f"{mode.value}:{kind}:{identifier}")
        return record

    @classmethod
    def _normalize_history_list(cls, mode: AudioServicesBackend, payload: dict[str, Any]) -> dict[str, Any]:
        record = cls._with_record_id(mode, "audio_history_list", payload, "list")
        if isinstance(record.get("items"), list):
            record["items"] = [
                cls._with_record_id(mode, "audio_history", item) if isinstance(item, dict) else item
                for item in record["items"]
            ]
        return record

    @classmethod
    def _normalize_voice_list(cls, mode: AudioServicesBackend, payload: dict[str, Any]) -> dict[str, Any]:
        record = cls._with_record_id(mode, "audio_voices", payload, "list")
        if isinstance(record.get("voices"), list):
            record["voices"] = [
                cls._with_record_id(mode, "audio_voice", item) if isinstance(item, dict) else item
                for item in record["voices"]
            ]
        return record

    @classmethod
    def _normalize_audiobook_projects(cls, mode: AudioServicesBackend, payload: dict[str, Any]) -> dict[str, Any]:
        record = cls._with_record_id(mode, "audiobook_projects", payload, "list")
        if isinstance(record.get("projects"), list):
            normalized_projects = []
            for item in record["projects"]:
                if not isinstance(item, dict):
                    normalized_projects.append(item)
                    continue
                project_id = item.get("project_id") or item.get("project_db_id")
                normalized_projects.append(cls._with_record_id(mode, "audiobook_project", item, project_id))
            record["projects"] = normalized_projects
        return record

    @classmethod
    def _normalize_artifacts(
        cls,
        mode: AudioServicesBackend,
        payload: dict[str, Any],
        *,
        list_kind: str,
        artifact_kind: str,
    ) -> dict[str, Any]:
        record = cls._with_record_id(mode, list_kind, payload, "list")
        if isinstance(record.get("artifacts"), list):
            record["artifacts"] = [
                cls._with_record_id(mode, artifact_kind, item, item.get("output_id"))
                if isinstance(item, dict)
                else item
                for item in record["artifacts"]
            ]
        return record

    @classmethod
    def _normalize_audiobook_project_detail(cls, mode: AudioServicesBackend, payload: dict[str, Any]) -> dict[str, Any]:
        record = cls._with_record_id(mode, "audiobook_project_detail", payload, "detail")
        if isinstance(record.get("project"), dict):
            project = dict(record["project"])
            project_id = project.get("project_id") or project.get("project_db_id")
            record["project"] = cls._with_record_id(mode, "audiobook_project", project, project_id)
        return record

    @classmethod
    def _normalize_audiobook_chapters(cls, mode: AudioServicesBackend, payload: dict[str, Any]) -> dict[str, Any]:
        record = cls._with_record_id(mode, "audiobook_chapters", payload, "list")
        if isinstance(record.get("chapters"), list):
            record["chapters"] = [
                cls._with_record_id(mode, "audiobook_chapter", item, item.get("id") or item.get("chapter_id"))
                if isinstance(item, dict)
                else item
                for item in record["chapters"]
            ]
        return record

    @classmethod
    def _normalize_audiobook_voice_profiles(cls, mode: AudioServicesBackend, payload: dict[str, Any]) -> dict[str, Any]:
        record = cls._with_record_id(mode, "audiobook_voice_profiles", payload, "list")
        if isinstance(record.get("profiles"), list):
            record["profiles"] = [
                cls._with_record_id(mode, "audiobook_voice_profile", item, item.get("profile_id"))
                if isinstance(item, dict)
                else item
                for item in record["profiles"]
            ]
        return record

    def list_unsupported_capabilities(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == AudioServicesBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def get_tts_health(self, *, mode: AudioServicesBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.health.observe.local",
            method_name="get_tts_health",
        )
        return self._with_record_id(normalized_mode, "audio", result, "health")

    async def get_stt_health(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        model: str | None = None,
        warm: bool = False,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.health.observe.local",
            method_name="get_stt_health",
            kwargs={"model": model, "warm": warm},
        )
        return self._with_record_id(normalized_mode, "audio", result, "stt_health")

    async def list_tts_providers(self, *, mode: AudioServicesBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.providers.list.local",
            method_name="list_tts_providers",
        )
        return self._with_record_id(normalized_mode, "audio", result, "providers")

    async def list_tts_voices(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.voices.list.local",
            method_name="list_tts_voices",
            kwargs={"provider": provider},
        )
        return self._with_record_id(normalized_mode, "audio", result, "voices")

    async def get_audio_streaming_status(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == AudioServicesBackend.LOCAL:
            raise ValueError("Audio streaming REST helpers are server-only; switch to server mode to use them.")
        self._enforce_policy("audio.streaming.status.server")
        result = await self._maybe_await(self._service_for_mode(normalized_mode).get_audio_streaming_status())
        return self._with_record_id(normalized_mode, "audio_streaming", self._dump(result), "status")

    async def get_audio_streaming_limits(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == AudioServicesBackend.LOCAL:
            raise ValueError("Audio streaming REST helpers are server-only; switch to server mode to use them.")
        self._enforce_policy("audio.streaming.detail.server")
        result = await self._maybe_await(self._service_for_mode(normalized_mode).get_audio_streaming_limits())
        return self._with_record_id(normalized_mode, "audio_streaming", self._dump(result), "limits")

    async def test_audio_streaming(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == AudioServicesBackend.LOCAL:
            raise ValueError("Audio streaming REST helpers are server-only; switch to server mode to use them.")
        self._enforce_policy("audio.streaming.launch.server")
        result = await self._maybe_await(self._service_for_mode(normalized_mode).test_audio_streaming())
        return self._with_record_id(normalized_mode, "audio_streaming", self._dump(result), "test")

    async def create_speech_chat(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == AudioServicesBackend.LOCAL:
            raise ValueError("Audio speech-chat REST helper is server-only; switch to server mode to use it.")
        self._enforce_policy("audio.speech_chat.launch.server")
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).create_speech_chat(request_data)
        )
        payload = self._dump(result)
        identifier = payload.get("session_id") if isinstance(payload, dict) else None
        return self._with_record_id(normalized_mode, "audio_speech_chat", payload, identifier or "session")

    async def create_audio_speech_job(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.speech.launch.local",
            method_name="create_audio_speech_job",
            kwargs={"request_data": request_data},
        )
        return self._with_record_id(normalized_mode, "audio_speech_job", result)

    async def create_audio_speech(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.speech.launch.local",
            method_name="create_audio_speech",
            kwargs={"request_data": request_data},
        )
        return self._with_record_id(normalized_mode, "audio", result, "speech")

    async def list_audio_speech_job_artifacts(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        job_id: int,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.speech_jobs.detail.local",
            method_name="list_audio_speech_job_artifacts",
            kwargs={"job_id": job_id},
        )
        return self._normalize_artifacts(
            normalized_mode,
            result,
            list_kind="audio_speech_artifacts",
            artifact_kind="audio_speech_artifact",
        )

    async def submit_audio_job(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.jobs.create.local",
            method_name="submit_audio_job",
            kwargs={"request_data": request_data},
        )
        return self._with_record_id(normalized_mode, "audio_job", result)

    async def get_audio_job(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        job_id: int,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.jobs.detail.local",
            method_name="get_audio_job",
            kwargs={"job_id": job_id},
        )
        return self._with_record_id(normalized_mode, "audio_job", result, job_id)

    async def stream_audio_job_progress(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        job_id: int,
        after_id: int = 0,
    ) -> AsyncGenerator[dict[str, Any], None]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("audio.jobs", "observe", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        method = getattr(service, "stream_audio_job_progress", None)
        if not callable(method):
            raise ValueError(f"stream_audio_job_progress is unavailable for {normalized_mode.value} audio services.")
        async for event in method(job_id, after_id=after_id):
            record = self._dump(event)
            if isinstance(record, dict):
                record.setdefault("backend", normalized_mode.value)
            yield record

    async def list_tts_history(self, *, mode: AudioServicesBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.history.list.local",
            method_name="list_tts_history",
            kwargs=kwargs,
        )
        return self._normalize_history_list(normalized_mode, result)

    async def get_tts_history_entry(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        history_id: int,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.history.detail.local",
            method_name="get_tts_history_entry",
            kwargs={"history_id": history_id},
        )
        return self._with_record_id(normalized_mode, "audio_history", result, history_id)

    async def update_tts_history_favorite(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        history_id: int,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.history.update.local",
            method_name="update_tts_history_favorite",
            kwargs={"history_id": history_id, "request_data": request_data},
        )
        return self._with_record_id(normalized_mode, "audio_history", result, history_id)

    async def delete_tts_history_entry(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        history_id: int,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.history.delete.local",
            method_name="delete_tts_history_entry",
            kwargs={"history_id": history_id},
        )
        return self._with_record_id(normalized_mode, "audio_history", result, history_id)

    async def create_audio_transcription(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        file_path: str,
        request_data: Any = None,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.transcriptions.launch.local",
            method_name="create_audio_transcription",
            kwargs={"file_path": file_path, "request_data": request_data},
        )
        return self._with_record_id(normalized_mode, "audio", result, "transcription")

    async def create_audio_translation(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        file_path: str,
        request_data: Any = None,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.translations.launch.local",
            method_name="create_audio_translation",
            kwargs={"file_path": file_path, "request_data": request_data},
        )
        return self._with_record_id(normalized_mode, "audio", result, "translation")

    async def encode_audio_tokenizer(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.tokenizer.launch.local",
            method_name="encode_audio_tokenizer",
            kwargs={"request_data": request_data},
        )
        return self._with_record_id(normalized_mode, "audio", result, "tokenizer_encode")

    async def decode_audio_tokenizer(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.tokenizer.launch.local",
            method_name="decode_audio_tokenizer",
            kwargs={"request_data": request_data},
        )
        return self._with_record_id(normalized_mode, "audio", result, "tokenizer_decode")

    async def upload_custom_voice(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        file_path: str,
        name: str,
        description: str | None = None,
        provider: str = "vibevoice",
        reference_text: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.voices.create.local",
            method_name="upload_custom_voice",
            kwargs={
                "file_path": file_path,
                "name": name,
                "description": description,
                "provider": provider,
                "reference_text": reference_text,
            },
        )
        return self._with_record_id(normalized_mode, "audio_voice", result)

    async def encode_custom_voice_reference(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.voices.launch.local",
            method_name="encode_custom_voice_reference",
            kwargs={"request_data": request_data},
        )
        return self._with_record_id(normalized_mode, "audio_voice", result)

    async def list_custom_voices(self, *, mode: AudioServicesBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.voices.list.local",
            method_name="list_custom_voices",
        )
        return self._normalize_voice_list(normalized_mode, result)

    async def get_custom_voice(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        voice_id: str,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.voices.detail.local",
            method_name="get_custom_voice",
            kwargs={"voice_id": voice_id},
        )
        return self._with_record_id(normalized_mode, "audio_voice", result, voice_id)

    async def preview_custom_voice(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        voice_id: str,
        text: str = "Hello, this is a preview of your custom voice.",
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.voices.preview.local",
            method_name="preview_custom_voice",
            kwargs={"voice_id": voice_id, "text": text},
        )
        return self._with_record_id(normalized_mode, "audio_voice_preview", result, voice_id)

    async def delete_custom_voice(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        voice_id: str,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audio.voices.delete.local",
            method_name="delete_custom_voice",
            kwargs={"voice_id": voice_id},
        )
        return self._with_record_id(normalized_mode, "audio_voice", result, voice_id)

    async def parse_audiobook_source(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audiobooks.parse.launch.local",
            method_name="parse_audiobook_source",
            kwargs={"request_data": request_data},
        )
        return self._with_record_id(normalized_mode, "audiobook_parse", result, result.get("project_id"))

    async def create_audiobook_job(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audiobooks.jobs.create.local",
            method_name="create_audiobook_job",
            kwargs={"request_data": request_data},
        )
        return self._with_record_id(normalized_mode, "audiobook_job", result)

    async def get_audiobook_job_status(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        job_id: int,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audiobooks.jobs.detail.local",
            method_name="get_audiobook_job_status",
            kwargs={"job_id": job_id},
        )
        return self._with_record_id(normalized_mode, "audiobook_job", result, job_id)

    async def list_audiobook_job_artifacts(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        job_id: int,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audiobooks.artifacts.list.local",
            method_name="list_audiobook_job_artifacts",
            kwargs={"job_id": job_id},
        )
        return self._normalize_artifacts(
            normalized_mode,
            result,
            list_kind="audiobook_artifacts",
            artifact_kind="audiobook_artifact",
        )

    async def list_audiobook_projects(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audiobooks.projects.list.local",
            method_name="list_audiobook_projects",
            kwargs={"limit": limit, "offset": offset},
        )
        return self._normalize_audiobook_projects(normalized_mode, result)

    async def get_audiobook_project(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        project_ref: str,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audiobooks.projects.detail.local",
            method_name="get_audiobook_project",
            kwargs={"project_ref": project_ref},
        )
        return self._normalize_audiobook_project_detail(normalized_mode, result)

    async def list_audiobook_project_chapters(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        project_ref: str,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audiobooks.chapters.list.local",
            method_name="list_audiobook_project_chapters",
            kwargs={"project_ref": project_ref, "limit": limit, "offset": offset},
        )
        return self._normalize_audiobook_chapters(normalized_mode, result)

    async def list_audiobook_project_artifacts(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        project_ref: str,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audiobooks.artifacts.list.local",
            method_name="list_audiobook_project_artifacts",
            kwargs={"project_ref": project_ref, "limit": limit, "offset": offset},
        )
        return self._normalize_artifacts(
            normalized_mode,
            result,
            list_kind="audiobook_artifacts",
            artifact_kind="audiobook_artifact",
        )

    async def create_audiobook_voice_profile(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audiobooks.voice_profiles.create.local",
            method_name="create_audiobook_voice_profile",
            kwargs={"request_data": request_data},
        )
        return self._with_record_id(normalized_mode, "audiobook_voice_profile", result)

    async def list_audiobook_voice_profiles(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audiobooks.voice_profiles.list.local",
            method_name="list_audiobook_voice_profiles",
        )
        return self._normalize_audiobook_voice_profiles(normalized_mode, result)

    async def delete_audiobook_voice_profile(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        profile_id: str,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audiobooks.voice_profiles.delete.local",
            method_name="delete_audiobook_voice_profile",
            kwargs={"profile_id": profile_id},
        )
        return self._with_record_id(normalized_mode, "audiobook_voice_profile", result, profile_id)

    async def export_audiobook_subtitles(
        self,
        *,
        mode: AudioServicesBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="audiobooks.subtitles.export.local",
            method_name="export_audiobook_subtitles",
            kwargs={"request_data": request_data},
        )
        return self._with_record_id(normalized_mode, "audiobook_subtitles", result, "export")
