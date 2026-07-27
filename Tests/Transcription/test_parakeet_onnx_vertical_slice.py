"""Focused coverage for the runnable Parakeet ONNX batch path."""

import sys
import wave
from types import SimpleNamespace

import pytest

from tldw_chatbook.Local_Ingestion import transcription_service as service_module
from tldw_chatbook.Local_Ingestion.transcription_service import (
    TranscriptionError,
    TranscriptionService,
)


def _write_silent_wav(path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\0\0" * 1_600)


def test_parakeet_onnx_transcribes_with_local_v2_int8_model(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "speech.wav"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for filename in (
        "config.json",
        "vocab.txt",
        "encoder-model.int8.onnx",
        "decoder_joint-model.int8.onnx",
    ):
        (model_dir / filename).touch()
    _write_silent_wav(audio_path)

    load_calls = []

    class FakeModel:
        def recognize(self, path):
            assert path == str(audio_path)
            return "A working local transcription."

    def fake_load_model(name, **kwargs):
        load_calls.append((name, kwargs))
        return FakeModel()

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=fake_load_model)
    )

    service = TranscriptionService()
    service.config["parakeet_onnx_model_dir"] = str(model_dir)
    result = service.transcribe(
        str(audio_path),
        provider="parakeet-onnx",
    )

    assert load_calls == [
        (
            "nemo-parakeet-tdt-0.6b-v2",
            {
                "path": str(model_dir),
                "quantization": "int8",
                "providers": ["CPUExecutionProvider"],
                "preprocessor_config": {
                    "use_numpy_preprocessors": True,
                    "max_concurrent_workers": 1,
                },
            },
        )
    ]
    assert result == {
        "text": "A working local transcription.",
        "segments": [
            {
                "start": 0.0,
                "end": 0.1,
                "text": "A working local transcription.",
                "Time_Start": 0.0,
                "Time_End": 0.1,
                "Text": "A working local transcription.",
            }
        ],
        "language": "en",
        "provider": "parakeet-onnx",
        "model": "nemo-parakeet-tdt-0.6b-v2",
    }


def test_parakeet_onnx_rejects_incomplete_model_directory_before_loading(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "speech.wav"
    model_dir = tmp_path / "incomplete-model"
    model_dir.mkdir()
    _write_silent_wav(audio_path)

    def unexpected_load(*args, **kwargs):
        raise AssertionError("load_model must not run for an incomplete local bundle")

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=unexpected_load)
    )

    with pytest.raises(TranscriptionError, match="missing required files"):
        TranscriptionService().transcribe(
            str(audio_path),
            provider="parakeet-onnx",
            model_dir=str(model_dir),
        )
