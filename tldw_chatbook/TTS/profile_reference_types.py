"""Private clone-reference domain values and product safety bounds."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Final, Literal, cast
from uuid import UUID

from tldw_chatbook.TTS.profile_errors import ProfileValidationError

MAX_REFERENCE_SOURCE_BYTES = 64 * 1024 * 1024
MAX_REFERENCE_CANONICAL_BYTES = 32 * 1024 * 1024
MAX_REFERENCE_DURATION_MS = 60_000
MAX_REFERENCE_TEXT_CHARACTERS = 4_096
MAX_REFERENCE_TEXT_UTF8_BYTES = 16 * 1024
MAX_REFERENCE_COUNT = 256
MAX_REFERENCE_TOTAL_BYTES = 512 * 1024 * 1024
MIN_REFERENCE_SAMPLE_RATE_HZ = 8_000
MAX_REFERENCE_SAMPLE_RATE_HZ = 96_000
REFERENCE_SAMPLE_ENCODING: Final[Literal["pcm_s16le"]] = "pcm_s16le"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_RECIPE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_ALLOWED_TRANSCRIPT_CONTROLS = frozenset({"\t", "\n"})


def _validation_error(code: str) -> ProfileValidationError:
    return ProfileValidationError(code)


def _validate_uuid(value: object) -> UUID:
    if type(value) is not UUID:
        raise _validation_error("reference_id")
    return value


def _validate_utc_timestamp(value: object, field_name: str) -> datetime:
    if type(value) is not datetime or value.tzinfo is None:
        raise _validation_error(field_name)
    try:
        if value.utcoffset() != timedelta(0):
            raise ValueError
        return value.astimezone(UTC)
    except Exception:
        raise _validation_error(field_name) from None


def _is_unsafe_transcript_character(character: str) -> bool:
    code_point = ord(character)
    category = unicodedata.category(character)
    return (
        category == "Cs"
        or (category == "Cc" and character not in _ALLOWED_TRANSCRIPT_CONTROLS)
        or 0xFDD0 <= code_point <= 0xFDEF
        or code_point & 0xFFFF in (0xFFFE, 0xFFFF)
    )


def _is_unsafe_model_character(character: str) -> bool:
    code_point = ord(character)
    return (
        unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        or 0xFDD0 <= code_point <= 0xFDEF
        or code_point & 0xFFFF in (0xFFFE, 0xFFFF)
    )


def _validate_recipe_model_id(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 256
        or any(_is_unsafe_model_character(character) for character in value)
    ):
        raise _validation_error("model_id")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise _validation_error("model_id") from None
    return value


@dataclass(frozen=True, slots=True, repr=False)
class TTSCloneRecipeRequirement:
    """Exact recipe and model dependency for a portable clone reference."""

    recipe_id: str
    recipe_revision: int
    model_id: str

    def __post_init__(self) -> None:
        if type(self.recipe_id) is not str or not _RECIPE_ID_PATTERN.fullmatch(
            self.recipe_id
        ):
            raise _validation_error("recipe_id")
        if (
            type(self.recipe_revision) is not int
            or not 1 <= self.recipe_revision <= 2_147_483_647
        ):
            raise _validation_error("recipe_revision")
        object.__setattr__(self, "model_id", _validate_recipe_model_id(self.model_id))

    def __repr__(self) -> str:
        return "TTSCloneRecipeRequirement(<private>)"


def validate_reference_text(value: object) -> str:
    """Return a bounded transcript without changing its internal text."""

    if type(value) is not str:
        raise _validation_error("reference_text")
    text = value.strip()
    if (
        not text
        or len(text) > MAX_REFERENCE_TEXT_CHARACTERS
        or any(_is_unsafe_transcript_character(character) for character in text)
    ):
        raise _validation_error("reference_text")
    encoding_error = False
    try:
        encoded = text.encode("utf-8", errors="strict")
    except UnicodeError:
        encoding_error = True
        encoded = b""
    if encoding_error:
        raise _validation_error("reference_text")
    if len(encoded) > MAX_REFERENCE_TEXT_UTF8_BYTES:
        raise _validation_error("reference_text")
    return text


def _validate_digest(value: object) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise _validation_error("reference_invalid")
    return value


def _validate_metadata(
    *,
    byte_length: object,
    duration_ms: object,
    sample_rate_hz: object,
    channels: object,
    sample_encoding: object,
) -> tuple[int, int, int, int, Literal["pcm_s16le"]]:
    if (
        type(byte_length) is not int
        or not 0 < byte_length <= MAX_REFERENCE_CANONICAL_BYTES
    ):
        raise _validation_error("byte_length")
    if type(duration_ms) is not int or not 0 < duration_ms <= MAX_REFERENCE_DURATION_MS:
        raise _validation_error("duration_ms")
    if (
        type(sample_rate_hz) is not int
        or not MIN_REFERENCE_SAMPLE_RATE_HZ
        <= sample_rate_hz
        <= MAX_REFERENCE_SAMPLE_RATE_HZ
    ):
        raise _validation_error("sample_rate_hz")
    if type(channels) is not int or channels not in (1, 2):
        raise _validation_error("channels")
    if type(sample_encoding) is not str or sample_encoding != REFERENCE_SAMPLE_ENCODING:
        raise _validation_error("sample_encoding")
    return (
        byte_length,
        duration_ms,
        sample_rate_hz,
        channels,
        cast(Literal["pcm_s16le"], REFERENCE_SAMPLE_ENCODING),
    )


@dataclass(frozen=True, slots=True)
class TTSCloneReferenceSummary:
    """Metadata-only projection of one profile-owned clone reference."""

    reference_id: UUID
    byte_length: int
    duration_ms: int
    sample_rate_hz: int
    channels: int
    sample_encoding: Literal["pcm_s16le"]
    created_at: datetime
    updated_at: datetime
    recipe_requirement: TTSCloneRecipeRequirement | None = None

    def __post_init__(self) -> None:
        reference_id = _validate_uuid(self.reference_id)
        metadata = _validate_metadata(
            byte_length=self.byte_length,
            duration_ms=self.duration_ms,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            sample_encoding=self.sample_encoding,
        )
        created_at = _validate_utc_timestamp(self.created_at, "created_at")
        updated_at = _validate_utc_timestamp(self.updated_at, "updated_at")
        if created_at > updated_at:
            raise _validation_error("timestamps")
        if (
            self.recipe_requirement is not None
            and type(self.recipe_requirement) is not TTSCloneRecipeRequirement
        ):
            raise _validation_error("reference_invalid")
        object.__setattr__(self, "reference_id", reference_id)
        object.__setattr__(self, "byte_length", metadata[0])
        object.__setattr__(self, "duration_ms", metadata[1])
        object.__setattr__(self, "sample_rate_hz", metadata[2])
        object.__setattr__(self, "channels", metadata[3])
        object.__setattr__(self, "sample_encoding", metadata[4])
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)


@dataclass(frozen=True, slots=True, repr=False)
class CanonicalTTSCloneReference:
    """One source-independent canonical reference ready for persistence."""

    wav_bytes: bytes
    reference_text: str
    sha256: str
    byte_length: int
    duration_ms: int
    sample_rate_hz: int
    channels: int
    sample_encoding: Literal["pcm_s16le"]

    def __post_init__(self) -> None:
        if (
            type(self.wav_bytes) is not bytes
            or not self.wav_bytes
            or len(self.wav_bytes) > MAX_REFERENCE_CANONICAL_BYTES
        ):
            raise _validation_error("reference_invalid")
        text = validate_reference_text(self.reference_text)
        digest = _validate_digest(self.sha256)
        metadata = _validate_metadata(
            byte_length=self.byte_length,
            duration_ms=self.duration_ms,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            sample_encoding=self.sample_encoding,
        )
        if (
            metadata[0] != len(self.wav_bytes)
            or sha256(self.wav_bytes).hexdigest() != digest
        ):
            raise _validation_error("reference_invalid")
        object.__setattr__(self, "reference_text", text)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "byte_length", metadata[0])
        object.__setattr__(self, "duration_ms", metadata[1])
        object.__setattr__(self, "sample_rate_hz", metadata[2])
        object.__setattr__(self, "channels", metadata[3])
        object.__setattr__(self, "sample_encoding", metadata[4])

    def __repr__(self) -> str:
        return "CanonicalTTSCloneReference(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class TTSCloneReference:
    """One exact stored clone reference whose representation is always redacted."""

    summary: TTSCloneReferenceSummary
    reference_text: str
    sha256: str
    wav_bytes: bytes
    recipe_requirement: TTSCloneRecipeRequirement | None = None

    def __post_init__(self) -> None:
        if type(self.summary) is not TTSCloneReferenceSummary:
            raise _validation_error("reference_invalid")
        if (
            self.recipe_requirement is not None
            and type(self.recipe_requirement) is not TTSCloneRecipeRequirement
        ):
            raise _validation_error("reference_invalid")
        if self.recipe_requirement != self.summary.recipe_requirement:
            raise _validation_error("reference_invalid")
        if type(self.wav_bytes) is not bytes or not self.wav_bytes:
            raise _validation_error("reference_invalid")
        text = validate_reference_text(self.reference_text)
        digest = _validate_digest(self.sha256)
        if (
            len(self.wav_bytes) != self.summary.byte_length
            or sha256(self.wav_bytes).hexdigest() != digest
        ):
            raise _validation_error("reference_invalid")
        object.__setattr__(self, "reference_text", text)
        object.__setattr__(self, "sha256", digest)

    def __repr__(self) -> str:
        return "TTSCloneReference(<private>)"


__all__ = [
    "CanonicalTTSCloneReference",
    "MAX_REFERENCE_CANONICAL_BYTES",
    "MAX_REFERENCE_COUNT",
    "MAX_REFERENCE_DURATION_MS",
    "MAX_REFERENCE_SOURCE_BYTES",
    "MAX_REFERENCE_TEXT_CHARACTERS",
    "MAX_REFERENCE_TEXT_UTF8_BYTES",
    "MAX_REFERENCE_TOTAL_BYTES",
    "REFERENCE_SAMPLE_ENCODING",
    "TTSCloneReference",
    "TTSCloneRecipeRequirement",
    "TTSCloneReferenceSummary",
    "validate_reference_text",
]
