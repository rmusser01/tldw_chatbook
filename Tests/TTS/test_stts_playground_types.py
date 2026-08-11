from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import UUID

import pytest

import tldw_chatbook.TTS.playground_types as playground_types
from tldw_chatbook.TTS import (
    STTSGeneratedAudio,
    STTSPlaygroundResultProjection,
    STTSPlaygroundRequest,
    TTSRequestedSelectionSnapshot,
)
from tldw_chatbook.TTS.effective_settings import (
    TTSSelectionOverrides,
    TTSStudioDraftSelection,
)
from tldw_chatbook.TTS.studio_preferences import StudioTTSPreferencesSnapshot
from tldw_chatbook.TTS.profile_reference_types import CanonicalTTSCloneReference


def _canonical_reference() -> CanonicalTTSCloneReference:
    wav = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"PRIVATE_REFERENCE_WAV"
    return CanonicalTTSCloneReference(
        wav_bytes=wav,
        reference_text="PRIVATE REFERENCE TRANSCRIPT",
        sha256=sha256(wav).hexdigest(),
        byte_length=len(wav),
        duration_ms=750,
        sample_rate_hz=24_000,
        channels=1,
        sample_encoding="pcm_s16le",
    )


def _studio_clone_request(
    clone_audition: object,
    *,
    provider_id: str = "audio_cpp",
) -> STTSPlaygroundRequest:
    preferences = StudioTTSPreferencesSnapshot(revision=4)
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id=provider_id,
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=4,
    )
    return STTSPlaygroundRequest(
        operation_id="clone-op",
        provider_id=provider_id,
        model_id="clone-model",
        text="hello",
        voice_id=None,
        response_format="wav",
        studio_draft=draft,
        studio_preferences=preferences,
        clone_audition=clone_audition,
    )


def test_playground_clone_snapshot_is_path_free_immutable_and_redacted() -> None:
    snapshot_type = getattr(playground_types, "STTSPlaygroundCloneSnapshot", None)
    assert snapshot_type is not None
    canonical = _canonical_reference()
    snapshot = snapshot_type(
        draft_revision=3,
        canonical_reference=canonical,
    )

    assert snapshot.canonical_reference is canonical
    assert {field.name for field in fields(snapshot)} == {
        "draft_revision",
        "canonical_reference",
    }
    assert not hasattr(snapshot, "source_path")
    rendered = repr(snapshot)
    assert "PRIVATE_REFERENCE_WAV" not in rendered
    assert "PRIVATE REFERENCE TRANSCRIPT" not in rendered
    assert canonical.sha256 not in rendered
    with pytest.raises(FrozenInstanceError):
        snapshot.draft_revision = 4  # type: ignore[misc]


@pytest.mark.parametrize("draft_revision", (True, -1))
def test_playground_clone_snapshot_requires_exact_nonnegative_revision(
    draft_revision: object,
) -> None:
    snapshot_type = getattr(playground_types, "STTSPlaygroundCloneSnapshot", None)
    assert snapshot_type is not None
    with pytest.raises((TypeError, ValueError)):
        snapshot_type(
            draft_revision=draft_revision,  # type: ignore[arg-type]
            canonical_reference=_canonical_reference(),
        )


def test_playground_request_accepts_clone_only_for_complete_audio_cpp_studio() -> None:
    snapshot_type = getattr(playground_types, "STTSPlaygroundCloneSnapshot", None)
    assert snapshot_type is not None
    snapshot = snapshot_type(
        draft_revision=3,
        canonical_reference=_canonical_reference(),
    )

    request = _studio_clone_request(snapshot)

    assert request.clone_audition is snapshot
    with pytest.raises(ValueError, match="audio.cpp Studio"):
        _studio_clone_request(snapshot, provider_id="openai")
    with pytest.raises(ValueError, match="audio.cpp Studio"):
        STTSPlaygroundRequest(
            operation_id="clone-op",
            provider_id="audio_cpp",
            model_id="clone-model",
            text="hello",
            voice_id=None,
            response_format="wav",
            clone_audition=snapshot,
        )


def test_profile_preview_snapshot_is_path_free_and_request_exclusive() -> None:
    preview_type = getattr(playground_types, "STTSPlaygroundProfilePreview", None)
    assert preview_type is not None
    preview = preview_type(
        profile_id=UUID("11111111-1111-4111-8111-111111111111"),
        repository_generation=7,
        profile_revision=4,
    )
    preferences = StudioTTSPreferencesSnapshot(revision=4)
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=4,
        preview=True,
    )

    request = STTSPlaygroundRequest(
        operation_id="preview-op",
        provider_id="audio_cpp",
        model_id="clone-model",
        text="hello",
        voice_id=None,
        response_format="wav",
        studio_draft=draft,
        studio_preferences=preferences,
        profile_preview=preview,
    )

    assert request.profile_preview is preview
    assert {field.name for field in fields(preview)} == {
        "profile_id",
        "repository_generation",
        "profile_revision",
    }
    assert not hasattr(preview, "reference")
    assert not hasattr(preview, "wav_bytes")
    with pytest.raises(ValueError, match="mutually exclusive"):
        STTSPlaygroundRequest(
            operation_id="preview-op",
            provider_id="audio_cpp",
            model_id="clone-model",
            text="hello",
            voice_id=None,
            response_format="wav",
            studio_draft=draft,
            studio_preferences=preferences,
            profile_preview=preview,
            clone_audition=playground_types.STTSPlaygroundCloneSnapshot(
                draft_revision=3,
                canonical_reference=_canonical_reference(),
            ),
        )


def test_playground_request_is_an_immutable_defensive_snapshot() -> None:
    nested_options: dict[str, Any] = {
        "language": "en",
        "conditioning": {"temperature": 0.2},
    }

    snapshot = STTSPlaygroundRequest(
        operation_id="local-op",
        provider_id="audio_cpp",
        model_id="kokoro",
        text="hello",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options=nested_options,
    )
    nested_options["language"] = "fr"
    nested_options["conditioning"]["temperature"] = 0.9

    assert isinstance(snapshot.options, MappingProxyType)
    assert snapshot.options == {
        "language": "en",
        "conditioning": {"temperature": 0.2},
    }
    with pytest.raises(TypeError):
        snapshot.options["language"] = "es"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        snapshot.model_id = "replacement"  # type: ignore[misc]


def test_studio_request_freezes_matching_draft_and_saved_revision() -> None:
    preferences = StudioTTSPreferencesSnapshot(revision=4)
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="supertonic",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=4,
    )

    snapshot = STTSPlaygroundRequest(
        operation_id="studio-op",
        provider_id="audio_cpp",
        model_id="supertonic",
        text="hello",
        voice_id=None,
        response_format="wav",
        studio_draft=draft,
        studio_preferences=preferences,
    )

    assert snapshot.studio_draft is draft
    assert snapshot.studio_preferences is preferences


def test_studio_request_rejects_partial_or_revision_mismatched_state() -> None:
    preferences = StudioTTSPreferencesSnapshot(revision=2)
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(),
        base_revision=1,
    )
    values = {
        "operation_id": "studio-op",
        "provider_id": "audio_cpp",
        "model_id": "supertonic",
        "text": "hello",
        "voice_id": None,
        "response_format": "wav",
    }

    with pytest.raises(ValueError, match="both draft and saved"):
        STTSPlaygroundRequest(**values, studio_preferences=preferences)
    with pytest.raises(ValueError, match="revisions must match"):
        STTSPlaygroundRequest(
            **values,
            studio_draft=draft,
            studio_preferences=preferences,
        )


def test_generated_audio_retains_provenance_and_actual_format_suffix(
    tmp_path: Path,
) -> None:
    response_metadata = {"delivery": "complete_wav", "sample_rate": 24_000}
    artifact = STTSGeneratedAudio(
        path=tmp_path / "result.legacy-selection",
        provider_id="audio_cpp",
        model_id="kokoro",
        voice_id=None,
        source_text="hello",
        operation_id="local-op",
        audio_format="wav",
        content_type="audio/wav",
        metadata=response_metadata,
    )
    response_metadata["delivery"] = "changed"

    assert artifact.file_suffix == ".wav"
    assert artifact.voice_id is None
    assert artifact.source_text == "hello"
    assert artifact.metadata == {
        "delivery": "complete_wav",
        "sample_rate": 24_000,
    }
    assert isinstance(artifact.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        artifact.metadata["delivery"] = "stream"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        artifact.provider_id = "legacy"  # type: ignore[misc]


def test_requested_selection_is_text_free_and_owns_empty_options() -> None:
    source: dict[str, Any] = {}
    selection = TTSRequestedSelectionSnapshot(
        provider_id="audio_cpp",
        model_id="model",
        voice_id="voice",
        response_format="wav",
        speed=1.0,
        options=source,
        configuration_revision=4,
    )
    source["late"] = "private"

    assert {field.name for field in fields(selection)} == {
        "provider_id",
        "model_id",
        "voice_id",
        "response_format",
        "speed",
        "options",
        "configuration_revision",
    }
    assert not hasattr(selection, "text")
    assert selection.options == {}
    assert isinstance(selection.options, MappingProxyType)
    with pytest.raises(TypeError):
        selection.options["late"] = "value"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        selection.model_id = "replacement"  # type: ignore[misc]


@pytest.mark.parametrize(
    "provider_id",
    sorted(("openai", "elevenlabs", "kokoro", "chatterbox", "higgs", "alltalk")),
)
def test_requested_selection_requires_an_exact_legacy_voice(
    provider_id: str,
) -> None:
    """Provenance must not carry a shape a profile can never hold."""

    with pytest.raises(ValueError):
        TTSRequestedSelectionSnapshot(
            provider_id=provider_id,
            model_id="tts-1",
            voice_id=None,
            response_format="mp3",
            speed=1.0,
            options={},
            configuration_revision=1,
        )


def test_requested_selection_keeps_audio_cpp_server_default_voice() -> None:
    selection = TTSRequestedSelectionSnapshot(
        provider_id="audio_cpp",
        model_id="model",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
        configuration_revision=1,
    )

    assert selection.voice_id is None


def test_generated_audio_requested_selection_is_optional_and_immutable(
    tmp_path: Path,
) -> None:
    selection = TTSRequestedSelectionSnapshot(
        provider_id="audio_cpp",
        model_id="model",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
        configuration_revision=2,
    )
    artifact = STTSGeneratedAudio(
        path=tmp_path / "result.wav",
        provider_id="audio_cpp",
        model_id="model",
        voice_id=None,
        source_text="private text",
        operation_id="operation",
        audio_format="wav",
        content_type="audio/wav",
        requested_selection=selection,
    )
    legacy = STTSGeneratedAudio(
        path=tmp_path / "legacy.wav",
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
        source_text="private text",
        operation_id="legacy-operation",
        audio_format="wav",
        content_type="audio/wav",
    )

    assert artifact.requested_selection is selection
    assert legacy.requested_selection is None


def test_only_native_artifact_provenance_is_profile_save_eligible(
    tmp_path: Path,
) -> None:
    selection = TTSRequestedSelectionSnapshot(
        provider_id="audio_cpp",
        model_id="exact/model",
        voice_id="exact/voice",
        response_format="wav",
        speed=1.0,
        options={},
        configuration_revision=2,
    )
    native = STTSGeneratedAudio(
        path=tmp_path / "native.wav",
        provider_id="audio_cpp",
        model_id="response/model",
        voice_id=None,
        source_text="private text",
        operation_id="native-operation",
        audio_format="wav",
        content_type="audio/wav",
        requested_selection=selection,
    )
    legacy = STTSGeneratedAudio(
        path=tmp_path / "legacy.wav",
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
        source_text="private text",
        operation_id="legacy-operation",
        audio_format="wav",
        content_type="audio/wav",
    )

    assert getattr(native, "profile_save_eligible", False) is True
    assert getattr(legacy, "profile_save_eligible", False) is False


def test_clone_projection_is_save_eligible_from_handler_evidence_flag(
    tmp_path: Path,
) -> None:
    """The UI may offer Save while the handler retains private clone proof."""

    projection = STTSPlaygroundResultProjection(
        path=tmp_path / "clone.wav",
        provider_id="audio_cpp",
        model_id="clone/model",
        voice_id=None,
        operation_id="clone-operation",
        audio_format="wav",
        clone_profile_save_eligible=True,
    )

    assert projection.profile_save_eligible is True


class _PrivateOption:
    def __init__(self, value: str) -> None:
        self.value = value


@pytest.mark.parametrize(
    "updates",
    (
        {"options": {"language": "en"}},
        {"options": {"blob": bytearray(b"raw-private-body")}},
        {"options": {"object": _PrivateOption("private-object-value")}},
        {"options": {1: "non-string-key"}},
        {"options": {"origin": "https://user:password@example.invalid"}},
        {"options": {"credential": "PRIVATE_API_KEY"}},
        {"provider_id": "future_native"},
        {"response_format": "mp3"},
        {"speed": 1.1},
    ),
)
def test_requested_selection_rejects_unreviewed_or_private_contract_values(
    updates: dict[str, object],
) -> None:
    private_values = (
        "raw-private-body",
        "private-object-value",
        "https://user:password@example.invalid",
        "PRIVATE_API_KEY",
    )
    values: dict[str, object] = {
        "provider_id": "audio_cpp",
        "model_id": "model",
        "voice_id": None,
        "response_format": "wav",
        "speed": 1.0,
        "options": {},
        "configuration_revision": 1,
    }
    values.update(updates)

    with pytest.raises((TypeError, ValueError)) as captured:
        TTSRequestedSelectionSnapshot(**values)  # type: ignore[arg-type]

    rendered = f"{captured.value!s} {captured.value!r}"
    for private_value in private_values:
        assert private_value not in rendered


@pytest.mark.parametrize(
    "updates",
    (
        {"provider_id": ""},
        {"provider_id": True},
        {"model_id": ""},
        {"voice_id": 1},
        {"response_format": ""},
        {"speed": True},
        {"speed": float("nan")},
        {"options": []},
        {"configuration_revision": True},
        {"configuration_revision": -1},
    ),
)
def test_requested_selection_rejects_invalid_public_values(
    updates: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "provider_id": "audio_cpp",
        "model_id": "model",
        "voice_id": None,
        "response_format": "wav",
        "speed": 1.0,
        "options": {},
        "configuration_revision": 1,
    }
    values.update(updates)

    with pytest.raises((TypeError, ValueError)):
        TTSRequestedSelectionSnapshot(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("operation_id", ""),
        ("provider_id", " "),
        ("model_id", ""),
        ("response_format", ""),
    ),
)
def test_playground_request_rejects_empty_required_identifiers(
    field_name: str,
    value: str,
) -> None:
    values = {
        "operation_id": "local-op",
        "provider_id": "audio_cpp",
        "model_id": "kokoro",
        "text": "hello",
        "voice_id": None,
        "response_format": "wav",
        "speed": 1.0,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        STTSPlaygroundRequest(**values)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("operation_id", ""),
        ("provider_id", ""),
        ("model_id", " "),
        ("audio_format", ""),
        ("content_type", ""),
    ),
)
def test_generated_audio_rejects_empty_required_identifiers(
    tmp_path: Path,
    field_name: str,
    value: str,
) -> None:
    values = {
        "path": tmp_path / "result.wav",
        "provider_id": "audio_cpp",
        "model_id": "kokoro",
        "voice_id": None,
        "source_text": "hello",
        "operation_id": "local-op",
        "audio_format": "wav",
        "content_type": "audio/wav",
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        STTSGeneratedAudio(**values)
