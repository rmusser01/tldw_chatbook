from __future__ import annotations

import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import tldw_chatbook.app as app_module
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Local_Ingestion import local_file_ingestion
from tldw_chatbook.Local_Ingestion.audio_processing import LocalAudioProcessor
from tldw_chatbook.Local_Ingestion.ingest_parse_worker import run_parse_job
from tldw_chatbook.Local_Ingestion.video_processing import LocalVideoProcessor
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJob
from tldw_chatbook.app import TldwCli
from tldw_chatbook.STT.contracts import (
    ExecutionDevice,
    ProducedCapabilities,
    TimestampGranularity,
    TranscriptionFailureCode,
    TranscriptionProvenance,
    TranscriptionResult,
    TranscriptionTask,
    TranscriptionTimings,
)
from tldw_chatbook.STT.persistence import (
    FailedTranscriptionAttempt,
    dump_failed_transcription_attempt,
    load_failed_transcription_attempt,
)


def _result() -> TranscriptionResult:
    return TranscriptionResult(
        text="hello world",
        segments=(),
        provenance=TranscriptionProvenance(
            schema_version=1,
            attempt_id="attempt-1",
            batch_id=None,
            job_id="ingest-job-1",
            retry_of_attempt_id=None,
            retry_of_job_id=None,
            provider_id="transcribe-cpp",
            model_id="local-gguf:whisper",
            artifact_root=None,
            artifact_dependencies=(),
            precision="native",
            requested_device=ExecutionDevice.AUTO,
            effective_device=ExecutionDevice.CPU,
            requested_language="en",
            effective_language="en",
            detected_language=None,
            task=TranscriptionTask.TRANSCRIBE,
        ),
        produced_capabilities=ProducedCapabilities(
            timestamps=TimestampGranularity.NONE,
            punctuation=False,
            capitalization=False,
            vad=False,
            diarization=False,
        ),
        duration_seconds=0.1,
        timings=TranscriptionTimings(total_seconds=0.1),
    )


def _failed_attempt() -> dict[str, object]:
    return load_failed_transcription_attempt(
        dump_failed_transcription_attempt(
            FailedTranscriptionAttempt(
                attempt_id="attempt-1",
                batch_id=None,
                job_id="ingest-job-1",
                provider_id="transcribe-cpp",
                model_id="local-gguf:whisper",
                artifact_root=None,
                artifact_dependencies=(),
                precision="native",
                requested_device=ExecutionDevice.AUTO,
                effective_device=None,
                requested_language="en",
                effective_language="en",
                detected_language=None,
                task=TranscriptionTask.TRANSCRIBE,
                error_code=TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
            )
        )
    )


def test_audio_processor_uses_direct_runner_and_returns_normalized_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_chatbook.STT import transcribe_cpp

    audio_path = tmp_path / "speech.wav"
    model_path = tmp_path / "model.gguf"
    audio_path.write_bytes(b"not-read-by-fake")
    calls: list[dict[str, object]] = []

    def fake_transcribe_file(**kwargs):
        calls.append(kwargs)
        return _result()

    monkeypatch.setattr(transcribe_cpp, "transcribe_file", fake_transcribe_file)
    processor = LocalAudioProcessor(media_db=None)

    batch = processor.process_audio_files(
        inputs=[str(audio_path)],
        transcription_provider="transcribe-cpp",
        transcription_language="en",
        transcription_context={
            "model_path": str(model_path),
            "attempt_id": "attempt-1",
            "job_id": "ingest-job-1",
        },
        timestamp_option=False,
        perform_chunking=False,
        perform_analysis=False,
    )

    assert calls == [
        {
            "audio_path": audio_path,
            "model_path": model_path,
            "attempt_id": "attempt-1",
            "batch_id": None,
            "job_id": "ingest-job-1",
            "retry_of_attempt_id": None,
            "retry_of_job_id": None,
            "language": "en",
            "timestamps": False,
            "ffmpeg_path": None,
        }
    ]
    row = batch["results"][0]
    assert row["status"] == "Success"
    assert row["content"] == "hello world"
    assert row["transcription_model"] == "local-gguf:whisper"
    assert row["transcription_provenance"]["provider_id"] == "transcribe-cpp"
    assert row["transcription_provenance"]["artifact_root"] is None


def test_audio_parse_payload_preserves_model_and_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "speech.wav"
    source.write_bytes(b"fixture")
    provenance = {"provider_id": "transcribe-cpp"}

    class StubAudioProcessor:
        def __init__(self, _media_db=None) -> None:
            pass

        def process_audio_files(self, **_kwargs):
            return {
                "results": [
                    {
                        "status": "Success",
                        "content": "hello world",
                        "metadata": {"title": "Speech", "author": "Unknown"},
                        "chunks": [],
                        "analysis": "",
                        "transcription_model": "local-gguf:whisper",
                        "transcription_provenance": provenance,
                    }
                ]
            }

    monkeypatch.setattr(local_file_ingestion, "LocalAudioProcessor", StubAudioProcessor)

    payload = local_file_ingestion.parse_local_file_for_ingest(
        str(source),
        {"transcription_provider": "transcribe-cpp"},
    )

    assert payload["transcription_model"] == "local-gguf:whisper"
    assert payload["transcription_provenance"] is provenance


def test_video_processor_preserves_direct_transcription_metadata(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "video-audio.mp3"
    audio_path.write_bytes(b"fixture")
    processor = LocalVideoProcessor(media_db=None)

    class StubAudioProcessor:
        def _process_single_audio(self, **_kwargs):
            return {
                "status": "Success",
                "content": "hello video",
                "segments": [],
                "chunks": [],
                "analysis": "",
                "warnings": [],
                "metadata": {},
                "transcription_model": "local-gguf:whisper",
                "transcription_provenance": {"provider_id": "transcribe-cpp"},
            }

    processor.audio_processor = StubAudioProcessor()

    result = processor.process_videos(
        inputs=[str(audio_path)],
        download_video_flag=False,
    )["results"][0]

    assert result["status"] == "Success"
    assert result["transcription_model"] == "local-gguf:whisper"
    assert result["transcription_provenance"] == {"provider_id": "transcribe-cpp"}


def test_parse_worker_preserves_bounded_stt_failure_and_failed_attempt(
    monkeypatch,
) -> None:
    failed_attempt = _failed_attempt()
    error = local_file_ingestion.DirectLocalSTTIngestError(
        "The selected GGUF cannot be used by transcribe.cpp.",
        error_detail={
            "category": "stt_failure",
            "code": "artifact_incompatible",
            "message": "The selected GGUF cannot be used by transcribe.cpp.",
            "actions": ["choose_another_gguf", "retry_faster_whisper"],
        },
        failed_attempt=failed_attempt,
    )

    def fail(_path: str, _options: dict[str, object]) -> None:
        raise error

    monkeypatch.setattr(local_file_ingestion, "parse_local_file_for_ingest", fail)

    result = run_parse_job("/private/model-owner/audio.wav", {})

    assert result == {
        "ok": False,
        "error": "The selected GGUF cannot be used by transcribe.cpp.",
        "permanent": False,
        "error_detail": {
            "category": "stt_failure",
            "code": "artifact_incompatible",
            "message": "The selected GGUF cannot be used by transcribe.cpp.",
            "actions": ["choose_another_gguf", "retry_faster_whisper"],
        },
        "stt_failure_provenance": failed_attempt,
    }
    assert "/private/model-owner" not in str(result)


def test_manual_library_job_reaches_fake_native_model_and_parent_writer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """One focused production-path proof from Library options to stored row."""
    from tldw_chatbook.STT import transcribe_cpp

    audio_path = tmp_path / "speech.wav"
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * 1_600)
    model_path = tmp_path / "private-model.gguf"
    model_path.write_bytes(b"fixture")
    calls: list[str] = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def run(self, _pcm, **_kwargs):
            calls.append("run")
            return SimpleNamespace(
                text="vertical hello",
                language="en",
                segments=(SimpleNamespace(text="vertical hello", t0_ms=0, t1_ms=100),),
                timings=SimpleNamespace(mel_ms=1.0, encode_ms=2.0, decode_ms=3.0),
            )

    class Model:
        def __init__(self, path: str, *, backend: str):
            assert path == str(model_path)
            assert backend == "auto"
            calls.append("load")
            self.arch = "whisper"
            self.backend = "cpu"
            self.device = SimpleNamespace(kind="cpu")
            self.capabilities = SimpleNamespace(
                native_sample_rate=16_000,
                languages=("en",),
                max_timestamp_kind="segment",
                supports_language_detect=True,
                supports_translate=True,
                supports_streaming=False,
                supports_spec_decode=False,
                max_audio_ms=None,
                translate_target_languages=("en",),
            )

        def session(self):
            return Session()

        def close(self):
            calls.append("close")

    monkeypatch.setitem(
        sys.modules,
        "transcribe_cpp",
        SimpleNamespace(Model=Model, set_log_callback=lambda _callback: None),
    )
    monkeypatch.setattr(
        transcribe_cpp,
        "validate_local_gguf",
        lambda path: SimpleNamespace(
            path=path,
            metadata=SimpleNamespace(architecture="whisper"),
        ),
    )
    monkeypatch.setattr(
        app_module,
        "get_cli_setting",
        lambda key, *args: (
            str(model_path)
            if key == "transcription.transcribe_cpp.model_path"
            else args[0]
            if args
            else None
        ),
    )

    app = object.__new__(TldwCli)
    job = LibraryIngestJob(
        job_id="ingest-job-vertical",
        source_path=str(audio_path),
        ingest_options={
            "audio_video": {
                "transcription_provider": "transcribe-cpp",
                "language": "en",
                "timestamps": True,
            }
        },
    )
    options = app._ingest_job_options(job)

    parsed = run_parse_job(job.source_path, options)

    assert parsed["ok"] is True, parsed
    db = MediaDatabase(":memory:", client_id="transcribe-cpp-vertical")
    media_id, _, _ = local_file_ingestion.persist_parsed_media(parsed["payload"], db)
    row = db.get_media_by_id(media_id)
    provenance = json.loads(row["transcription_provenance_json"])
    assert row["content"] == "vertical hello"
    assert row["transcription_model"] == "local-gguf:whisper"
    assert provenance["provider_id"] == "transcribe-cpp"
    assert provenance["model_id"] == "local-gguf:whisper"
    assert str(model_path) not in repr(provenance)
    assert calls == ["load", "run", "close"]
