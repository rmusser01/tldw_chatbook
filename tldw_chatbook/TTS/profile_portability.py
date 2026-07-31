"""Sanitized wire codec for local TTS generation-profile portability."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from tldw_chatbook.TTS.profile_errors import ProfileValidationError
from tldw_chatbook.TTS.profile_types import TTSProfileDraft, canonical_json_options

PORTABLE_PROFILE_SCHEMA_VERSION = 1
CHARACTER_CARD_TTS_EXTENSION_KEY = "tldw_chatbook/tts_generation_profile"
_MAX_ATTACHMENT_BYTES = 16 * 1024
_MAX_CONTAINER_LEVELS = 4
_WIRE_FIELDS = frozenset(
    {
        "schema_version",
        "profile_id",
        "name",
        "provider_id",
        "model_id",
        "voice_id",
        "response_format",
        "speed",
        "options",
    }
)


class PortableProfileDecodeStatus(StrEnum):
    """Bounded outcome of decoding one untrusted attachment."""

    VALID = "valid"
    SKIPPED = "skipped"
    INVALID = "invalid"


PortableProfileWarningCode = Literal[
    "unsupported_version",
    "unsupported_provider",
    "invalid_attachment",
]


@dataclass(frozen=True, slots=True)
class PortableTTSProfile:
    """A sanitized profile selection paired with its portable UUID hint."""

    profile_id: UUID
    draft: TTSProfileDraft

    def __post_init__(self) -> None:
        if type(self.profile_id) is not UUID:
            raise ProfileValidationError("profile_id")
        if type(self.draft) is not TTSProfileDraft:
            raise ProfileValidationError("profiles")
        if self.draft.provider_id != "audio_cpp":
            raise ProfileValidationError("audio_cpp")


@dataclass(frozen=True, slots=True)
class PortableProfileDecodeResult:
    """A value-independent decode result safe to surface or log."""

    status: PortableProfileDecodeStatus
    profile: PortableTTSProfile | None
    warning_code: PortableProfileWarningCode | None


def portable_profile_payload(profile: PortableTTSProfile) -> dict[str, Any]:
    """Return the exact version-one sanitized payload for ``profile``."""

    if type(profile) is not PortableTTSProfile:
        raise ProfileValidationError("profiles")
    draft = profile.draft
    options = json.loads(canonical_json_options(draft.options))
    return {
        "schema_version": PORTABLE_PROFILE_SCHEMA_VERSION,
        "profile_id": str(profile.profile_id),
        "name": draft.display_name,
        "provider_id": draft.provider_id,
        "model_id": draft.model_id,
        "voice_id": draft.voice_id,
        "response_format": draft.response_format,
        "speed": draft.speed,
        "options": options,
    }


def portable_profile_json(profile: PortableTTSProfile) -> str:
    """Return deterministic standalone JSON for a sanitized profile."""

    return json.dumps(
        portable_profile_payload(profile),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )


def _invalid_result() -> PortableProfileDecodeResult:
    return PortableProfileDecodeResult(
        status=PortableProfileDecodeStatus.INVALID,
        profile=None,
        warning_code="invalid_attachment",
    )


def _skip_result(
    warning_code: Literal["unsupported_version", "unsupported_provider"],
) -> PortableProfileDecodeResult:
    return PortableProfileDecodeResult(
        status=PortableProfileDecodeStatus.SKIPPED,
        profile=None,
        warning_code=warning_code,
    )


def _validate_json_shape(value: object, level: int, active: set[int]) -> None:
    if value is None or type(value) in (str, int, float, bool):
        return
    if level > _MAX_CONTAINER_LEVELS:
        raise ValueError
    identity = id(value)
    if identity in active:
        raise ValueError
    active.add(identity)
    try:
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError
                _validate_json_shape(item, level + 1, active)
        elif type(value) is list:
            for item in value:
                _validate_json_shape(item, level + 1, active)
        else:
            raise ValueError
    finally:
        active.discard(identity)


def _validate_attachment_bounds(payload: object) -> None:
    if type(payload) is not dict:
        raise ValueError
    _validate_json_shape(payload, 1, set())
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_ATTACHMENT_BYTES:
        raise ValueError


def decode_portable_profile(payload: object) -> PortableProfileDecodeResult:
    """Decode one hostile attachment without echoing attacker-controlled data."""

    try:
        _validate_attachment_bounds(payload)
        assert type(payload) is dict
        schema_version = payload.get("schema_version")
        if type(schema_version) is not int:
            raise ValueError
        if schema_version != PORTABLE_PROFILE_SCHEMA_VERSION:
            return _skip_result("unsupported_version")
        if frozenset(payload) != _WIRE_FIELDS:
            raise ValueError

        profile_id_value = payload["profile_id"]
        if type(profile_id_value) is not str:
            raise ValueError
        profile_id = UUID(profile_id_value)
        draft = TTSProfileDraft(
            display_name=payload["name"],
            provider_id=payload["provider_id"],
            model_id=payload["model_id"],
            voice_id=payload["voice_id"],
            response_format=payload["response_format"],
            speed=payload["speed"],
            options=payload["options"],
        )
        if draft.provider_id != "audio_cpp":
            return _skip_result("unsupported_provider")
        return PortableProfileDecodeResult(
            status=PortableProfileDecodeStatus.VALID,
            profile=PortableTTSProfile(profile_id=profile_id, draft=draft),
            warning_code=None,
        )
    except Exception:
        return _invalid_result()
