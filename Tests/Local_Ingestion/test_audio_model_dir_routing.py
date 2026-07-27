"""Tests for routing a per-job Parakeet ONNX model directory."""

from pathlib import Path

from tldw_chatbook.Local_Ingestion.audio_processing import LocalAudioProcessor


def test_audio_processor_passes_model_directory_to_transcription(
    tmp_path: Path, monkeypatch
) -> None:
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"not-read-by-stub")
    calls: list[dict] = []

    processor = LocalAudioProcessor(media_db=None)

    def fake_transcribe(path: str, **kwargs):
        calls.append({"path": path, **kwargs})
        return {"text": "transcript", "segments": []}

    monkeypatch.setattr(processor, "_transcribe_audio", fake_transcribe)

    result = processor.process_audio_files(
        inputs=[str(audio_path)],
        transcription_provider="parakeet-onnx",
        transcription_model="nemo-parakeet-tdt-0.6b-v2",
        transcription_model_dir="/models/parakeet-v2-int8",
        transcription_language="en",
        transcription_precision="int8",
        transcription_local_files_only=True,
        transcription_batch_route_resolved=True,
        perform_chunking=False,
        perform_analysis=False,
    )

    assert result["processed_count"] == 1
    assert calls == [
        {
            "path": str(audio_path),
            "provider": "parakeet-onnx",
            "model": "nemo-parakeet-tdt-0.6b-v2",
            "language": "en",
            "target_lang": None,
            "vad_filter": False,
            "diarize": False,
            "model_dir": "/models/parakeet-v2-int8",
            "compute_type": "int8",
            "local_files_only": True,
            "batch_route_resolved": True,
            "progress_callback": None,
        }
    ]


def test_audio_processor_preserves_legacy_positional_perform_chunking(
    tmp_path: Path, monkeypatch
) -> None:
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"not-read-by-stub")
    calls: list[dict] = []
    processor = LocalAudioProcessor(media_db=None)

    def fake_process(**kwargs):
        calls.append(kwargs)
        return {"status": "Success"}

    monkeypatch.setattr(processor, "_process_single_audio", fake_process)

    result = processor.process_audio_files(
        [str(audio_path)],
        "faster-whisper",
        "base",
        None,
        "en",
        None,
        False,
    )

    assert result["processed_count"] == 1
    assert calls[0]["perform_chunking"] is False
    assert calls[0]["transcription_precision"] is None
    assert calls[0]["transcription_local_files_only"] is False
    assert calls[0]["transcription_batch_route_resolved"] is False
