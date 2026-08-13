"""Sanitized wire codec for local TTS generation-profile portability."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, TypedDict, cast
from uuid import UUID

from tldw_chatbook.TTS.profile_errors import ProfileValidationError
from tldw_chatbook.TTS.profile_types import (
    PROFILE_PROVIDER_IDS,
    JsonOptionsInput,
    TTSProfileDraft,
    canonical_json_options,
)

PORTABLE_PROFILE_SCHEMA_VERSION = 1
PORTABLE_PROFILE_REFERENCE_OMITTED_SCHEMA_VERSION = 2
CHARACTER_CARD_TTS_EXTENSION_KEY = "tldw_chatbook/tts_generation_profile"
# The cap covers the whole envelope. Portable audio.cpp drafts require empty
# options, and their other validated text fields keep the largest valid UTF-8
# envelope below 3 KiB; the remaining budget rejects hostile future payloads.
_MAX_ATTACHMENT_BYTES = 16 * 1024
_MAX_CONTAINER_LEVELS = 4
_WIRE_V1_FIELDS = frozenset(
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
_WIRE_V2_FIELDS = _WIRE_V1_FIELDS | frozenset({"reference"})


class _WireSelection(TypedDict):
    """Exact typed values extracted from an already bounded wire object."""

    profile_id: str
    name: str
    provider_id: str
    model_id: str
    voice_id: str | None
    response_format: str
    speed: float
    options: JsonOptionsInput


class PortableProfileDecodeStatus(StrEnum):
    """Bounded outcome of decoding one untrusted attachment."""

    VALID = "valid"
    SKIPPED = "skipped"
    REFERENCE_OMITTED = "reference_omitted"
    INVALID = "invalid"


PortableProfileWarningCode = Literal[
    "unsupported_version",
    "unsupported_provider",
    "reference_omitted",
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
        if self.draft.provider_id not in PROFILE_PROVIDER_IDS:
            raise ProfileValidationError("audio_cpp")


@dataclass(frozen=True, slots=True)
class PortableProfileDecodeResult:
    """A value-independent decode result safe to surface or log."""

    status: PortableProfileDecodeStatus
    profile: PortableTTSProfile | None
    warning_code: PortableProfileWarningCode | None


def portable_profile_payload(
    profile: PortableTTSProfile,
    *,
    reference_present: bool = False,
) -> dict[str, Any]:
    """Return the exact sanitized payload for a profile.

    Args:
        profile: Validated local provider selection to make portable.

    Returns:
        The strict version-one payload for a reference-free profile, or the
        exact version-two omission marker when a private reference is present.

    Raises:
        ProfileValidationError: If ``profile`` is not a portable profile
            selection.
    """

    if type(profile) is not PortableTTSProfile:
        raise ProfileValidationError("profiles")
    if type(reference_present) is not bool:
        raise ProfileValidationError("reference_invalid")
    draft = profile.draft
    options = json.loads(canonical_json_options(draft.options))
    payload: dict[str, Any] = {
        "schema_version": (
            PORTABLE_PROFILE_REFERENCE_OMITTED_SCHEMA_VERSION
            if reference_present
            else PORTABLE_PROFILE_SCHEMA_VERSION
        ),
        "profile_id": str(profile.profile_id),
        "name": draft.display_name,
        "provider_id": draft.provider_id,
        "model_id": draft.model_id,
        "voice_id": draft.voice_id,
        "response_format": draft.response_format,
        "speed": draft.speed,
        "options": options,
    }
    if reference_present:
        payload["reference"] = {"status": "omitted"}
    return payload


def portable_profile_json(
    profile: PortableTTSProfile,
    *,
    reference_present: bool = False,
) -> str:
    """Return deterministic standalone JSON for a sanitized profile.

    Args:
        profile: Validated local provider selection to serialize.

    Returns:
        Pretty-printed exact sanitized JSON containing only portable fields.

    Raises:
        ProfileValidationError: If ``profile`` is not a portable profile
            selection.
    """

    return json.dumps(
        portable_profile_payload(profile, reference_present=reference_present),
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


def _reference_omitted_result() -> PortableProfileDecodeResult:
    return PortableProfileDecodeResult(
        status=PortableProfileDecodeStatus.REFERENCE_OMITTED,
        profile=None,
        warning_code="reference_omitted",
    )


def _extract_wire_selection(payload: dict[str, object]) -> _WireSelection:
    """Validate and type one exact wire selection without coercion."""

    profile_id = payload["profile_id"]
    name = payload["name"]
    provider_id = payload["provider_id"]
    model_id = payload["model_id"]
    voice_id = payload["voice_id"]
    response_format = payload["response_format"]
    speed = payload["speed"]
    options = payload["options"]
    if (
        type(profile_id) is not str
        or type(name) is not str
        or type(provider_id) is not str
        or type(model_id) is not str
        or (voice_id is not None and type(voice_id) is not str)
        or type(response_format) is not str
        or type(options) is not dict
    ):
        raise ValueError
    if type(speed) is int:
        normalized_speed = float(speed)
    elif type(speed) is float:
        normalized_speed = speed
    else:
        raise ValueError
    _validate_json_shape(options, 1, set())
    return {
        "profile_id": profile_id,
        "name": name,
        "provider_id": provider_id,
        "model_id": model_id,
        "voice_id": voice_id,
        "response_format": response_format,
        "speed": normalized_speed,
        "options": cast(JsonOptionsInput, options),
    }


def _decode_selection(
    payload: dict[str, object],
    *,
    skip_unsupported_provider: bool,
    require_canonical_profile_id: bool,
) -> PortableTTSProfile | PortableProfileDecodeResult:
    selection = _extract_wire_selection(payload)
    profile_id = UUID(selection["profile_id"])
    if require_canonical_profile_id and str(profile_id) != selection["profile_id"]:
        raise ValueError
    try:
        draft = TTSProfileDraft(
            display_name=selection["name"],
            provider_id=selection["provider_id"],
            model_id=selection["model_id"],
            voice_id=selection["voice_id"],
            response_format=selection["response_format"],
            speed=selection["speed"],
            options=selection["options"],
        )
    except ProfileValidationError as error:
        if skip_unsupported_provider and error.code == "provider_id":
            return _skip_result("unsupported_provider")
        raise
    if draft.provider_id not in PROFILE_PROVIDER_IDS:
        if skip_unsupported_provider:
            return _skip_result("unsupported_provider")
        raise ValueError
    return PortableTTSProfile(profile_id=profile_id, draft=draft)


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
    """Decode one hostile attachment without echoing attacker-controlled data.

    Args:
        payload: Untrusted character-card attachment value.

    Returns:
        A bounded valid, skipped, or invalid result. Invalid input is never
        raised or copied into the result.
    """

    try:
        # Deliberately bound bytes, depth, and cycles before constructing any
        # schema/domain object. Exact wire checks prevent coercion, while
        # TTSProfileDraft remains the centralized semantic validator.
        _validate_attachment_bounds(payload)
        assert type(payload) is dict
        schema_version = payload.get("schema_version")
        if type(schema_version) is not int:
            raise ValueError
        if schema_version not in (
            PORTABLE_PROFILE_SCHEMA_VERSION,
            PORTABLE_PROFILE_REFERENCE_OMITTED_SCHEMA_VERSION,
        ):
            return _skip_result("unsupported_version")
        if schema_version == PORTABLE_PROFILE_SCHEMA_VERSION:
            if frozenset(payload) != _WIRE_V1_FIELDS:
                raise ValueError
            selection = _decode_selection(
                payload,
                skip_unsupported_provider=True,
                require_canonical_profile_id=False,
            )
            if isinstance(selection, PortableProfileDecodeResult):
                return selection
            return PortableProfileDecodeResult(
                status=PortableProfileDecodeStatus.VALID,
                profile=cast(PortableTTSProfile, selection),
                warning_code=None,
            )
        if frozenset(payload) != _WIRE_V2_FIELDS:
            raise ValueError
        if payload["reference"] != {"status": "omitted"}:
            raise ValueError
        selection = _decode_selection(
            payload,
            skip_unsupported_provider=False,
            require_canonical_profile_id=True,
        )
        if isinstance(selection, PortableProfileDecodeResult):
            raise ValueError
        return _reference_omitted_result()
    except Exception:
        return _invalid_result()
