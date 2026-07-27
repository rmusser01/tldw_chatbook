from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from tldw_chatbook.TTS import (
    STTSGeneratedAudio,
    STTSPlaygroundRequest,
    TTSRequestedSelectionSnapshot,
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


def test_requested_selection_is_text_free_and_deeply_immutable() -> None:
    source: dict[str, Any] = {
        "language": "en",
        "conditioning": {"temperature": 0.2},
    }
    selection = TTSRequestedSelectionSnapshot(
        provider_id="audio_cpp",
        model_id="model",
        voice_id="voice",
        response_format="wav",
        speed=1.0,
        options=source,
        configuration_revision=4,
    )
    source["language"] = "fr"
    source["conditioning"]["temperature"] = 0.9

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
    assert selection.options == {
        "language": "en",
        "conditioning": {"temperature": 0.2},
    }
    assert isinstance(selection.options, MappingProxyType)
    with pytest.raises(TypeError):
        selection.options["language"] = "es"  # type: ignore[index]
    with pytest.raises(TypeError):
        selection.options["conditioning"]["temperature"] = 0.5  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        selection.model_id = "replacement"  # type: ignore[misc]


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
