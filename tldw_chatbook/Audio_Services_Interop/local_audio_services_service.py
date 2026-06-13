"""Local Chatbook-owned audio adapter."""

from __future__ import annotations

import base64
import inspect
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LocalAudioServicesService:
    """Expose local audio availability and immediate TTS generation."""

    def __init__(
        self,
        *,
        tts_provider_loader: Callable[[], dict[str, Any]] | None = None,
        stt_provider_loader: Callable[[], dict[str, Any]] | None = None,
        voice_catalog_loader: Callable[[], dict[str, Any]] | None = None,
        tts_audio_generator: Callable[..., Any] | None = None,
        history_store_path: str | Path | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.tts_provider_loader = tts_provider_loader or (lambda: {})
        self.stt_provider_loader = stt_provider_loader or (lambda: {})
        self.voice_catalog_loader = voice_catalog_loader or (lambda: {})
        self.tts_audio_generator = tts_audio_generator
        self.history_store_path = Path(history_store_path).expanduser() if history_store_path else None
        self.policy_enforcer = policy_enforcer
        self._history_records: list[dict[str, Any]] = []
        self._next_history_id = 1
        self._load_history()

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)

    @staticmethod
    def _safe_mapping(loader: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            payload = loader()
        except Exception:
            return {}
        return dict(payload or {}) if isinstance(payload, dict) else {}

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _dump_request(request_data: Any) -> dict[str, Any]:
        if hasattr(request_data, "model_dump"):
            return request_data.model_dump(mode="python", exclude_none=True)
        if hasattr(request_data, "dict"):
            return request_data.dict(exclude_none=True)
        if isinstance(request_data, dict):
            return dict(request_data)
        raise TypeError("local_tts_request_must_be_mapping")

    def _load_history(self) -> None:
        if self.history_store_path is None or not self.history_store_path.exists():
            return
        try:
            payload = json.loads(self.history_store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._history_records = []
            self._next_history_id = 1
            return
        records = payload.get("items", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            return
        self._history_records = [dict(item) for item in records if isinstance(item, dict)]
        max_id = max((int(item.get("id", 0) or 0) for item in self._history_records), default=0)
        self._next_history_id = max_id + 1

    def _persist_history(self) -> None:
        if self.history_store_path is None:
            return
        self.history_store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"items": self._history_records}
        temp_path = self.history_store_path.with_suffix(self.history_store_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.history_store_path)

    @staticmethod
    def _tts_provider_for_model(model: str) -> str:
        normalized = str(model or "").strip().lower()
        if normalized in {"kokoro", "chatterbox", "alltalk", "higgs"}:
            return normalized
        if normalized.startswith("elevenlabs"):
            return "elevenlabs"
        return "openai"

    @staticmethod
    def _tts_internal_model_id(model: str) -> str:
        normalized = str(model or "").strip().lower()
        if normalized in {"tts-1", "tts-1-hd"}:
            return f"openai_official_{normalized}"
        if normalized == "kokoro":
            return "local_kokoro_default_onnx"
        if normalized == "chatterbox":
            return "local_chatterbox_default"
        if normalized == "alltalk":
            return "alltalk_default"
        if normalized == "higgs":
            return "local_higgs_default"
        if normalized.startswith("elevenlabs"):
            return f"elevenlabs_{normalized}"
        return normalized

    @staticmethod
    def _tts_content_type(response_format: str) -> str:
        return {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "application/octet-stream",
        }.get(response_format, "application/octet-stream")

    @staticmethod
    def _normalize_response_format(value: Any) -> str:
        response_format = str(value or "mp3").strip().lower()
        valid_formats = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
        if response_format not in valid_formats:
            raise ValueError(f"local_tts_response_format_unsupported:{response_format}")
        return response_format

    @staticmethod
    def _history_summary(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": record["id"],
            "created_at": record["created_at"],
            "provider": record["provider"],
            "model": record["model"],
            "voice": record["voice"],
            "response_format": record["response_format"],
            "filename": record["filename"],
            "content_type": record["content_type"],
            "size_bytes": record["size_bytes"],
            "input_characters": record["input_characters"],
            "text_preview": record["text_preview"],
            "favorite": bool(record.get("favorite", False)),
            "has_text": bool(record.get("text")),
        }

    def _find_history_record(self, history_id: int) -> dict[str, Any]:
        for record in self._history_records:
            if int(record.get("id", 0) or 0) == int(history_id) and not record.get("deleted"):
                return record
        raise ValueError(f"local_tts_history_not_found:{history_id}")

    async def _default_tts_audio_generator(
        self,
        *,
        text: str,
        model: str,
        voice: str,
        response_format: str,
        stream: bool,
        speed: float | None,
    ) -> bytes:
        from tldw_chatbook.TTS import OpenAISpeechRequest, get_tts_service

        app_config = {
            "app_tts": {
                "default_provider": self._tts_provider_for_model(model),
                "default_voice": voice,
                "default_model": model,
                "default_format": response_format,
                "default_speed": speed or 1.0,
            }
        }
        service = await get_tts_service(app_config)
        request = OpenAISpeechRequest(
            model=model,
            input=text,
            voice=voice,
            response_format=response_format,  # type: ignore[arg-type]
            stream=stream,
            speed=speed or 1.0,
        )
        chunks: list[bytes] = []
        async for chunk in service.generate_audio_stream(request, self._tts_internal_model_id(model)):
            chunks.append(bytes(chunk))
        return b"".join(chunks)

    def get_tts_health(self) -> dict[str, Any]:
        self._enforce("audio.health.observe.local")
        providers = self._safe_mapping(self.tts_provider_loader)
        return {
            "status": "available",
            "service": "local_tts",
            "providers": providers,
            "provider_count": len(providers),
        }

    def get_stt_health(self, *, model: str | None = None, warm: bool = False) -> dict[str, Any]:
        del warm
        self._enforce("audio.health.observe.local")
        providers = self._safe_mapping(self.stt_provider_loader)
        return {
            "status": "available",
            "service": "local_stt",
            "model": model,
            "providers": providers,
            "provider_count": len(providers),
        }

    def list_tts_providers(self) -> dict[str, Any]:
        self._enforce("audio.providers.list.local")
        providers = self._safe_mapping(self.tts_provider_loader)
        voices = self._safe_mapping(self.voice_catalog_loader)
        return {
            "providers": providers,
            "voices": voices,
            "timestamp": None,
        }

    def list_tts_voices(self, *, provider: str | None = None) -> dict[str, Any]:
        self._enforce("audio.voices.list.local")
        voices = self._safe_mapping(self.voice_catalog_loader)
        if provider is None:
            return voices
        return {provider: voices.get(provider, [])}

    async def create_audio_speech(self, request_data: Any) -> dict[str, Any]:
        self._enforce("audio.speech.launch.local")
        payload = self._dump_request(request_data)
        text = str(payload.get("input") or "").strip()
        if not text:
            raise ValueError("local_tts_input_required")

        model = str(payload.get("model") or "kokoro").strip() or "kokoro"
        voice = str(payload.get("voice") or "af_heart").strip() or "af_heart"
        response_format = self._normalize_response_format(
            payload.get("download_format") or payload.get("response_format")
        )
        stream = bool(payload.get("stream", True))
        speed = float(payload.get("speed", 1.0) or 1.0)

        generator = self.tts_audio_generator or self._default_tts_audio_generator
        audio = await self._maybe_await(
            generator(
                text=text,
                model=model,
                voice=voice,
                response_format=response_format,
                stream=stream,
                speed=speed,
            )
        )
        if not isinstance(audio, bytes | bytearray):
            raise ValueError("local_tts_generator_must_return_bytes")

        audio_bytes = bytes(audio)
        history_id = self._next_history_id
        self._next_history_id += 1
        filename = str(payload.get("filename") or f"local_speech_{history_id}.{response_format}")
        content_type = self._tts_content_type(response_format)
        record = {
            "id": history_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": self._tts_provider_for_model(model),
            "model": model,
            "voice": voice,
            "response_format": response_format,
            "filename": filename,
            "content_type": content_type,
            "size_bytes": len(audio_bytes),
            "input_characters": len(text),
            "text_preview": text[:160],
            "text": text,
            "favorite": False,
            "content_base64": base64.b64encode(audio_bytes).decode("ascii"),
            "deleted": False,
        }
        self._history_records.append(record)
        self._persist_history()
        return {
            "content": audio_bytes,
            "content_type": content_type,
            "filename": filename,
            "history_id": history_id,
            "provider": record["provider"],
            "model": model,
            "voice": voice,
            "response_format": response_format,
        }

    async def list_tts_history(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce("audio.history.list.local")
        limit = int(kwargs.get("limit", 50) or 50)
        offset = int(kwargs.get("offset", 0) or 0)
        provider = kwargs.get("provider")
        model = kwargs.get("model")
        voice = kwargs.get("voice")
        records = [record for record in self._history_records if not record.get("deleted")]
        if provider:
            records = [record for record in records if record.get("provider") == provider]
        if model:
            records = [record for record in records if record.get("model") == model]
        if voice:
            records = [record for record in records if record.get("voice") == voice]
        records = sorted(records, key=lambda item: int(item.get("id", 0) or 0), reverse=True)
        page = records[offset : offset + limit]
        return {
            "items": [self._history_summary(record) for record in page],
            "limit": limit,
            "offset": offset,
            "total": len(records),
        }

    async def get_tts_history_entry(self, history_id: int) -> dict[str, Any]:
        self._enforce("audio.history.detail.local")
        record = self._find_history_record(history_id)
        detail = self._history_summary(record)
        detail["text"] = record.get("text")
        content_base64 = record.get("content_base64")
        detail["content"] = base64.b64decode(content_base64) if content_base64 else b""
        return detail

    async def update_tts_history_favorite(self, history_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce("audio.history.update.local")
        payload = self._dump_request(request_data)
        record = self._find_history_record(history_id)
        record["favorite"] = bool(payload.get("favorite", False))
        self._persist_history()
        detail = self._history_summary(record)
        detail["text"] = record.get("text")
        return detail

    async def delete_tts_history_entry(self, history_id: int) -> dict[str, Any]:
        self._enforce("audio.history.delete.local")
        record = self._find_history_record(history_id)
        record["deleted"] = True
        self._persist_history()
        return {"id": int(history_id), "deleted": True}
