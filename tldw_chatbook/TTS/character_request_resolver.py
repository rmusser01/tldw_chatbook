"""Fail-closed character assignment resolution for trusted Console speech."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias
from uuid import UUID

from tldw_chatbook.TTS.adapter_types import TTSRequest
from tldw_chatbook.TTS.profile_errors import (
    ProfileRepositoryError,
    ProfileServiceError,
    ProfileValidationError,
)
from tldw_chatbook.TTS.profile_service import LoadedCharacterTTSAssignment
from tldw_chatbook.TTS.profile_types import CharacterRef

CharacterTTSResolutionSource: TypeAlias = Literal[
    "global",
    "assigned",
    "explicit_override",
]
CharacterTTSResolutionCode: TypeAlias = Literal[
    "assignment_invalid",
    "authorship_invalid",
    "authority_missing",
    "profile_store_unavailable",
]

_RESOLUTION_COPY: dict[CharacterTTSResolutionCode, str] = {
    "assignment_invalid": (
        "The assigned voice profile could not be used. Repair or remove the "
        "assignment, or use the global voice for this message."
    ),
    "authorship_invalid": "Message authorship could not be verified for speech.",
    "authority_missing": (
        "This character's speech identity is unavailable. Reopen the character "
        "chat, or use the global voice for this message."
    ),
    "profile_store_unavailable": (
        "Character voice profiles are unavailable. Retry, or use the global "
        "voice for this message."
    ),
}
_GLOBAL_OVERRIDE_CODES = frozenset(
    {
        "assignment_invalid",
        "authority_missing",
        "profile_store_unavailable",
    }
)


class _ProfileService(Protocol):
    async def get_assigned_profile(
        self,
        character_ref: CharacterRef,
    ) -> LoadedCharacterTTSAssignment: ...


class CharacterTTSResolutionError(RuntimeError):
    """One bounded assignment-resolution failure safe for direct UI copy."""

    def __init__(self, code: CharacterTTSResolutionCode) -> None:
        if type(code) is not str or code not in _RESOLUTION_COPY:
            raise ValueError("resolution code must be bounded")
        self.code: CharacterTTSResolutionCode = code
        self.allow_global_override = code in _GLOBAL_OVERRIDE_CODES
        super().__init__(_RESOLUTION_COPY[code])


@dataclass(frozen=True, slots=True)
class CharacterTTSRequestResolution:
    """Immutable exact-request or global-path decision."""

    source: CharacterTTSResolutionSource
    request: TTSRequest | None
    repository_generation: int | None
    profile_id: UUID | None
    profile_revision: int | None

    def __post_init__(self) -> None:
        if self.source not in {"global", "assigned", "explicit_override"}:
            raise ValueError("source")
        if self.source == "assigned":
            if (
                type(self.request) is not TTSRequest
                or self.request.provider_id != "audio_cpp"
                or type(self.repository_generation) is not int
                or self.repository_generation < 0
                or type(self.profile_id) is not UUID
                or type(self.profile_revision) is not int
                or self.profile_revision < 1
            ):
                raise ValueError("assigned resolution")
            return
        if any(
            value is not None
            for value in (
                self.request,
                self.repository_generation,
                self.profile_id,
                self.profile_revision,
            )
        ):
            raise ValueError("global resolution")


def _global_resolution(
    source: Literal["global", "explicit_override"],
) -> CharacterTTSRequestResolution:
    return CharacterTTSRequestResolution(
        source=source,
        request=None,
        repository_generation=None,
        profile_id=None,
        profile_revision=None,
    )


class CharacterTTSRequestResolver:
    """Convert validated authorship facts into one immutable TTS selection."""

    def __init__(self, profile_service: _ProfileService | None) -> None:
        self._profile_service = profile_service

    @staticmethod
    def _validate_text(text: object) -> str:
        if type(text) is not str or not text.strip():
            raise CharacterTTSResolutionError("authorship_invalid")
        return text

    async def resolve(
        self,
        *,
        text: str,
        assistant_kind: str | None,
        character_ref: CharacterRef | None,
    ) -> CharacterTTSRequestResolution:
        """Resolve one already-validated trusted message without fallback."""

        validated_text = self._validate_text(text)
        if assistant_kind is not None and type(assistant_kind) is not str:
            raise CharacterTTSResolutionError("authorship_invalid")
        if assistant_kind in {None, "generic", "persona"}:
            if character_ref is not None:
                raise CharacterTTSResolutionError("authorship_invalid")
            return _global_resolution("global")
        if assistant_kind != "character":
            raise CharacterTTSResolutionError("authorship_invalid")
        if character_ref is None:
            raise CharacterTTSResolutionError("authority_missing")
        if type(character_ref) is not CharacterRef:
            raise CharacterTTSResolutionError("authorship_invalid")
        service = self._profile_service
        if service is None:
            raise CharacterTTSResolutionError("profile_store_unavailable")

        try:
            loaded = await service.get_assigned_profile(character_ref)
        except asyncio.CancelledError:
            raise
        except (ProfileRepositoryError, ProfileServiceError, ProfileValidationError):
            raise CharacterTTSResolutionError("profile_store_unavailable") from None
        except Exception:
            raise CharacterTTSResolutionError("profile_store_unavailable") from None

        try:
            if type(loaded) is not LoadedCharacterTTSAssignment:
                raise TypeError
            snapshot = loaded.snapshot
            if snapshot is None:
                return _global_resolution("global")
            if snapshot.assignment.character_ref != character_ref:
                raise ValueError
            profile = snapshot.profile
            request = TTSRequest(
                provider_id=profile.provider_id,
                model_id=profile.model_id,
                text=validated_text,
                voice=profile.voice_id,
                response_format=profile.response_format,
                speed=profile.speed,
                options=profile.options,
            )
            return CharacterTTSRequestResolution(
                source="assigned",
                request=request,
                repository_generation=loaded.repository_generation,
                profile_id=profile.profile_id,
                profile_revision=profile.revision,
            )
        except CharacterTTSResolutionError:
            raise
        except Exception:
            raise CharacterTTSResolutionError("assignment_invalid") from None

    def resolve_explicit_global_override(
        self,
        *,
        text: str,
    ) -> CharacterTTSRequestResolution:
        """Return the handler-authorized one-message global path."""

        self._validate_text(text)
        return _global_resolution("explicit_override")


__all__ = [
    "CharacterTTSRequestResolution",
    "CharacterTTSRequestResolver",
    "CharacterTTSResolutionError",
    "CharacterTTSResolutionSource",
]
