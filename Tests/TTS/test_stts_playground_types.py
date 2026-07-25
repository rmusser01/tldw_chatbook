from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from tldw_chatbook.TTS import STTSGeneratedAudio, STTSPlaygroundRequest


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
