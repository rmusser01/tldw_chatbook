"""Tests for private clone-reference domain values and bounds."""

from __future__ import annotations

import pickle
from hashlib import sha256
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from tldw_chatbook.TTS.profile_errors import ProfileValidationError
from tldw_chatbook.TTS.profile_types import TTSGenerationProfile
from tldw_chatbook.TTS.profile_reference_types import (
    MAX_REFERENCE_CANONICAL_BYTES,
    MAX_REFERENCE_COUNT,
    MAX_REFERENCE_DURATION_MS,
    MAX_REFERENCE_SOURCE_BYTES,
    MAX_REFERENCE_TEXT_CHARACTERS,
    MAX_REFERENCE_TEXT_UTF8_BYTES,
    MAX_REFERENCE_TOTAL_BYTES,
    CanonicalTTSCloneReference,
    TTSCloneReference,
    TTSCloneRecipeRequirement,
    TTSCloneReferenceSummary,
    validate_reference_text,
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
REFERENCE_ID = UUID("12345678-1234-4234-8234-123456789abc")
CANONICAL_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"private-wave"
DIGEST = sha256(CANONICAL_WAV).hexdigest()


def _summary(**overrides: object) -> TTSCloneReferenceSummary:
    values: dict[str, object] = {
        "reference_id": REFERENCE_ID,
        "byte_length": len(CANONICAL_WAV),
        "duration_ms": 750,
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_encoding": "pcm_s16le",
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return TTSCloneReferenceSummary(**values)  # type: ignore[arg-type]


def _requirement(**overrides: object) -> TTSCloneRecipeRequirement:
    values: dict[str, object] = {
        "recipe_id": "audio-cpp-0.5.1.supertonic.supertonic_3_orig",
        "recipe_revision": 1,
        "model_id": "supertonic-3",
    }
    values.update(overrides)
    return TTSCloneRecipeRequirement(**values)  # type: ignore[arg-type]


def _canonical(**overrides: object) -> CanonicalTTSCloneReference:
    values: dict[str, object] = {
        "wav_bytes": CANONICAL_WAV,
        "reference_text": "The exact private transcript.",
        "sha256": DIGEST,
        "byte_length": len(CANONICAL_WAV),
        "duration_ms": 750,
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_encoding": "pcm_s16le",
    }
    values.update(overrides)
    return CanonicalTTSCloneReference(**values)  # type: ignore[arg-type]


def test_reference_limits_are_named_finite_product_bounds() -> None:
    assert MAX_REFERENCE_SOURCE_BYTES == 64 * 1024 * 1024
    assert MAX_REFERENCE_CANONICAL_BYTES == 32 * 1024 * 1024
    assert MAX_REFERENCE_DURATION_MS == 60_000
    assert MAX_REFERENCE_TEXT_CHARACTERS == 4_096
    assert MAX_REFERENCE_TEXT_UTF8_BYTES == 16 * 1024
    assert MAX_REFERENCE_COUNT == 256
    assert MAX_REFERENCE_TOTAL_BYTES == 512 * 1024 * 1024


def test_reference_summary_is_immutable_and_pickle_safe() -> None:
    summary = _summary()

    assert pickle.loads(pickle.dumps(summary)) == summary
    with pytest.raises(FrozenInstanceError):
        summary.channels = 2  # type: ignore[misc]


def test_recipe_requirement_is_immutable_pickle_safe_and_permanently_redacted() -> None:
    requirement = _requirement()

    assert pickle.loads(pickle.dumps(requirement)) == requirement
    assert repr(requirement) == "TTSCloneRecipeRequirement(<private>)"
    assert requirement.recipe_id not in repr(requirement)
    assert requirement.model_id not in repr(requirement)
    with pytest.raises(FrozenInstanceError):
        requirement.recipe_revision = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("recipe_id", ""),
        ("recipe_id", ".invalid"),
        ("recipe_id", "UPPER"),
        ("recipe_id", "recipe/id"),
        ("recipe_id", "a" * 129),
        ("recipe_revision", True),
        ("recipe_revision", 0),
        ("recipe_revision", 2_147_483_648),
        ("model_id", ""),
        ("model_id", "x" * 257),
        ("model_id", "model\ncontrol"),
    ],
)
def test_recipe_requirement_rejects_noncanonical_values(
    field: str, value: object
) -> None:
    with pytest.raises(ProfileValidationError):
        _requirement(**{field: value})


def test_reference_provenance_is_absent_or_exactly_shared() -> None:
    requirement = _requirement()
    summary = _summary(recipe_requirement=requirement)
    reference = TTSCloneReference(
        summary=summary,
        recipe_requirement=requirement,
        reference_text="The exact private transcript.",
        sha256=DIGEST,
        wav_bytes=CANONICAL_WAV,
    )

    assert _summary().recipe_requirement is None
    assert summary.recipe_requirement == requirement
    assert reference.recipe_requirement == requirement


@pytest.mark.parametrize(
    ("summary_recipe_revision", "reference_recipe_revision"),
    [
        (1, None),
        (None, 1),
        (1, 2),
    ],
)
def test_reference_rejects_half_present_or_mismatched_provenance(
    summary_recipe_revision: int | None,
    reference_recipe_revision: int | None,
) -> None:
    summary_requirement = (
        None
        if summary_recipe_revision is None
        else _requirement(recipe_revision=summary_recipe_revision)
    )
    reference_requirement = (
        None
        if reference_recipe_revision is None
        else _requirement(recipe_revision=reference_recipe_revision)
    )
    with pytest.raises(
        ProfileValidationError,
        match=r"^TTS profile validation failed: reference_invalid$",
    ):
        TTSCloneReference(
            summary=_summary(recipe_requirement=summary_requirement),
            recipe_requirement=reference_requirement,
            reference_text="The exact private transcript.",
            sha256=DIGEST,
            wav_bytes=CANONICAL_WAV,
        )


def test_profile_rejects_reference_requirement_for_another_model() -> None:
    now = NOW
    summary = _summary(recipe_requirement=_requirement(model_id="different-model"))

    with pytest.raises(
        ProfileValidationError,
        match=r"^TTS profile validation failed: reference_invalid$",
    ):
        TTSGenerationProfile(
            profile_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            display_name="Clone profile",
            normalized_name="clone profile",
            provider_id="audio_cpp",
            model_id="supertonic-3",
            voice_id=None,
            response_format="wav",
            speed=1.0,
            options={},
            revision=1,
            created_at=now,
            updated_at=now,
            reference=summary,
        )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("reference_id", "12345678-1234-4234-8234-123456789abc", "reference_id"),
        ("byte_length", True, "byte_length"),
        ("byte_length", 0, "byte_length"),
        ("duration_ms", 0, "duration_ms"),
        ("duration_ms", MAX_REFERENCE_DURATION_MS + 1, "duration_ms"),
        ("sample_rate_hz", 7_999, "sample_rate_hz"),
        ("sample_rate_hz", 96_001, "sample_rate_hz"),
        ("channels", 0, "channels"),
        ("channels", 3, "channels"),
        ("sample_encoding", "pcm_s24le", "sample_encoding"),
        ("created_at", NOW.replace(tzinfo=None), "created_at"),
        ("updated_at", NOW - timedelta(seconds=1), "timestamps"),
    ],
)
def test_reference_summary_rejects_noncanonical_metadata(
    field: str, value: object, code: str
) -> None:
    with pytest.raises(
        ProfileValidationError,
        match=rf"^TTS profile validation failed: {code}$",
    ):
        _summary(**{field: value})


def test_reference_text_trims_only_outer_whitespace() -> None:
    assert validate_reference_text("  Line one\nLine  two  ") == "Line one\nLine  two"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "contains\x00nul",
        "contains\ud800surrogate",
        "contains\ufdd0noncharacter",
        "x" * (MAX_REFERENCE_TEXT_CHARACTERS + 1),
        "界" * ((MAX_REFERENCE_TEXT_UTF8_BYTES // 3) + 1),
    ],
)
def test_reference_text_rejects_unsafe_or_unbounded_values(value: str) -> None:
    with pytest.raises(
        ProfileValidationError,
        match=r"^TTS profile validation failed: reference_text$",
    ):
        validate_reference_text(value)


def test_reference_text_preserves_format_characters_needed_by_languages() -> None:
    assert validate_reference_text("क्\u200dष") == "क्\u200dष"


def test_canonical_reference_validates_length_digest_and_metadata() -> None:
    canonical = _canonical()

    assert canonical.wav_bytes == CANONICAL_WAV
    assert canonical.byte_length == len(CANONICAL_WAV)
    assert canonical.reference_text == "The exact private transcript."


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("wav_bytes", bytearray(CANONICAL_WAV), "reference_invalid"),
        ("wav_bytes", b"", "reference_invalid"),
        ("byte_length", len(CANONICAL_WAV) + 1, "reference_invalid"),
        ("sha256", "A" * 64, "reference_invalid"),
        ("sha256", "a" * 63, "reference_invalid"),
    ],
)
def test_canonical_reference_rejects_inconsistent_private_payload(
    field: str, value: object, code: str
) -> None:
    with pytest.raises(
        ProfileValidationError,
        match=rf"^TTS profile validation failed: {code}$",
    ):
        _canonical(**{field: value})


def test_private_reference_repr_and_pickle_disclose_no_values() -> None:
    canonical = _canonical()
    reference = TTSCloneReference(
        summary=_summary(),
        reference_text=canonical.reference_text,
        sha256=canonical.sha256,
        wav_bytes=canonical.wav_bytes,
    )

    rendered = repr(reference)
    assert rendered == "TTSCloneReference(<private>)"
    assert canonical.reference_text not in rendered
    assert canonical.wav_bytes.hex() not in rendered
    assert str(reference.summary.reference_id) not in rendered
    assert canonical.sha256 not in rendered
    assert pickle.loads(pickle.dumps(reference)) == reference


def test_private_reference_requires_payload_to_match_summary() -> None:
    with pytest.raises(
        ProfileValidationError,
        match=r"^TTS profile validation failed: reference_invalid$",
    ):
        TTSCloneReference(
            summary=_summary(byte_length=len(CANONICAL_WAV) + 1),
            reference_text="The exact private transcript.",
            sha256=DIGEST,
            wav_bytes=CANONICAL_WAV,
        )
