"""YouTube input must retain the selected STT through both media processors."""

from __future__ import annotations

import wave

import pytest

import tldw_chatbook.app as app_module
from Tests.private_profile import private_profile_test
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJob
from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
    parse_local_file_for_ingest,
)
from tldw_chatbook.Local_Ingestion.video_processing import LocalVideoProcessor


@pytest.mark.parametrize(
    "selected, expected",
    [
        ("default", "faster-whisper"),
        ("faster-whisper", "faster-whisper"),
        ("parakeet-onnx", "parakeet-onnx"),
        ("transcribe-cpp", "transcribe-cpp"),
    ],
)
@pytest.mark.asyncio
@private_profile_test
def test_youtube_job_reaches_selected_transcription_runner(
    request, tmp_path, monkeypatch, selected, expected
):
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    audio = tmp_path / "download.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 1600)

    downloads = []

    def download(self, **kwargs):
        downloads.append(kwargs["url"])
        return str(audio)

    monkeypatch.setattr(LocalVideoProcessor, "download_video", download)
    monkeypatch.setattr(
        LocalVideoProcessor, "extract_metadata", lambda *args: {"title": "Sample video"}
    )
    observed = []

    def transcribe(path, **kwargs):
        assert path == str(audio)
        observed.append(kwargs["provider"])
        return {
            "text": "The selected speech backend transcribed this video.",
            "segments": [],
        }

    app = object.__new__(app_module.TldwCli)
    job = LibraryIngestJob(
        job_id="youtube-selection",
        source_path=url,
        ingest_options={
            "generic": {"analyze": False, "chunk": False},
            "audio_video": {
                "transcription_provider": selected,
                "language": "en",
                "timestamps": False,
            },
        },
    )
    options = app._ingest_job_options(job)
    payload = parse_local_file_for_ingest(url, options, transcription_runner=transcribe)
    assert downloads == [url]
    assert observed == [expected]
    assert payload["content"] == "The selected speech backend transcribed this video."
