"""Local Chatbook-owned audio discovery adapter.

Generation/transcription remain in the existing TTS/STTS event handlers until
those call sites are migrated behind the source-aware scope.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class LocalAudioServicesService:
    """Expose local audio availability without coupling it to server history/artifacts."""

    def __init__(
        self,
        *,
        tts_provider_loader: Callable[[], dict[str, Any]] | None = None,
        stt_provider_loader: Callable[[], dict[str, Any]] | None = None,
        voice_catalog_loader: Callable[[], dict[str, Any]] | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.tts_provider_loader = tts_provider_loader or (lambda: {})
        self.stt_provider_loader = stt_provider_loader or (lambda: {})
        self.voice_catalog_loader = voice_catalog_loader or (lambda: {})
        self.policy_enforcer = policy_enforcer

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
