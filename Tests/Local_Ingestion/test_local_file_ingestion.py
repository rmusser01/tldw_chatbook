"""Tests for per-type ingestion option routing in ``local_file_ingestion``.

These tests verify that ``parse_local_file_for_ingest`` forwards the new
per-media-type options (produced by ``app._ingest_job_options``) into the
underlying processor functions with the correct keyword names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from tldw_chatbook.Local_Ingestion import local_file_ingestion
from tldw_chatbook.Local_Ingestion.local_file_ingestion import parse_local_file_for_ingest


def _make_pdf_result(**kwargs) -> Dict[str, Any]:
    return {
        "status": "Success",
        "content": "PDF text",
        "title": "PDF title",
        "author": "PDF author",
        "keywords": [],
        "chunks": [],
        "analysis": "",
        "metadata": {},
        "error": None,
        "warnings": kwargs.get("warnings", []),
    }


def _make_ebook_result(**kwargs) -> Dict[str, Any]:
    return {
        "status": "Success",
        "content": "Ebook text",
        "title": "Ebook title",
        "author": "Ebook author",
        "keywords": [],
        "chunks": [],
        "analysis": "",
        "metadata": {},
        "error": None,
        "warnings": kwargs.get("warnings", []),
    }


def test_quick_ingest_uses_canonical_media_database_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_chatbook import config
    from tldw_chatbook.Local_Ingestion import local_file_ingestion

    expected_path = tmp_path / "runtime" / "media.db"
    captured: dict[str, object] = {}

    class FakeMediaDatabase:
        def __init__(self, db_path, client_id):
            captured["db_path"] = db_path
            captured["client_id"] = client_id

        def close_connection(self):
            captured["closed"] = True

    monkeypatch.setattr(config, "get_media_db_path", lambda: expected_path)
    monkeypatch.setattr(
        local_file_ingestion,
        "MediaDatabase",
        FakeMediaDatabase,
    )
    monkeypatch.setattr(
        local_file_ingestion,
        "ingest_local_file",
        lambda file_path, media_db: {"status": "ok"},
    )

    result = local_file_ingestion.quick_ingest(tmp_path / "document.txt")

    assert result == {"status": "ok"}
    assert captured == {
        "db_path": str(expected_path),
        "client_id": "quick_ingest",
        "closed": True,
    }


def test_pdf_options_are_routed_to_process_pdf(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    calls: list[Dict[str, Any]] = []

    def fake_process_pdf(**kwargs):
        calls.append(kwargs)
        return _make_pdf_result()

    monkeypatch.setattr(
        "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_pdf",
        fake_process_pdf,
    )

    parse_local_file_for_ingest(
        str(source),
        {
            "pdf_engine": "docling",
            "ocr": True,
            "extract_images": False,
            "page_range": None,
        },
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["engine"] == "docling"
    assert call["ocr"] is True
    assert call["extract_images"] is False
    assert call["page_range"] is None


def test_pdf_unimplemented_options_record_warnings(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_process_pdf(engine=None, page_range=None, ocr=None, extract_images=False, **kwargs):
        warnings = []
        if page_range is not None:
            warnings.append(f"page_range={page_range}")
        if extract_images:
            warnings.append("extract_images=True")
        return _make_pdf_result(warnings=warnings)

    monkeypatch.setattr(
        "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_pdf",
        fake_process_pdf,
    )

    payload = parse_local_file_for_ingest(
        str(source),
        {
            "pdf_engine": "pymupdf4llm",
            "ocr": False,
            "extract_images": True,
            "page_range": "1-10",
        },
    )

    assert payload["content"] == "PDF text"
    assert payload["warnings"] == [
        "page_range=1-10",
        "extract_images=True",
    ]


def test_audio_options_are_routed_to_processor(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "recording.mp3"
    source.write_bytes(b"ID3\x00" + b"\x00" * 64)

    calls: list[Dict[str, Any]] = []
    progress_events: list[tuple[str, str, float | None]] = []
    synthetic_data = {"stage": "preparing", "private": object()}
    measured_data = {
        "current_time": 2.0,
        "total_time": 8.0,
        "provider": object(),
    }

    class _StubAudioProcessor:
        def __init__(self, media_db=None):
            self.media_db = media_db

        def process_audio_files(self, **kwargs):
            calls.append(kwargs)
            kwargs["transcription_progress_callback"](
                10.0,
                "Preparing audio for transcription",
                synthetic_data,
            )
            kwargs["transcription_progress_callback"](
                91.0,
                "Transcribing segment 3 of 8",
                measured_data,
            )
            return {
                "results": [
                    {
                        "status": "Success",
                        "content": "Audio transcript",
                        "metadata": {"title": "Audio", "author": "Unknown"},
                        "chunks": [],
                        "analysis": "",
                    }
                ]
            }

    monkeypatch.setattr(
        "tldw_chatbook.Local_Ingestion.local_file_ingestion.LocalAudioProcessor",
        _StubAudioProcessor,
    )

    parse_local_file_for_ingest(
        str(source),
        {
            "transcription_provider": "parakeet-onnx",
            "transcription_model_dir": "/models/parakeet-v2-int8",
            "transcription_model": "nemo-parakeet-tdt-0.6b-v2",
            "language": "en",
            "transcription_precision": "int8",
            "transcription_local_files_only": True,
            "transcription_batch_route_resolved": True,
            "timestamps": False,
            "diarization": True,
        },
        progress_callback=lambda phase, message, percent=None: progress_events.append(
            (phase, message, percent)
        ),
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["transcription_provider"] == "parakeet-onnx"
    assert call["transcription_model_dir"] == "/models/parakeet-v2-int8"
    assert call["transcription_model"] == "nemo-parakeet-tdt-0.6b-v2"
    assert call["transcription_language"] == "en"
    assert call["transcription_precision"] == "int8"
    assert call["transcription_local_files_only"] is True
    assert call["transcription_batch_route_resolved"] is True
    assert call["timestamp_option"] is False
    assert call["diarize"] is True
    assert (
        "transcribing",
        "Preparing audio for transcription",
        None,
    ) in progress_events
    assert (
        "transcribing",
        "Transcribing segment 3 of 8",
        25.0,
    ) in progress_events
    assert all(synthetic_data not in event for event in progress_events)
    assert all(measured_data not in event for event in progress_events)


def test_video_options_are_routed_to_processor(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 64)

    calls: list[Dict[str, Any]] = []
    progress_events: list[tuple[str, str, float | None]] = []
    synthetic_data = {"stage": "uploading", "private": object()}
    measured_data = {
        "chunk": 3,
        "total_chunks": 8,
        "provider": object(),
    }

    class _StubVideoProcessor:
        def __init__(self, media_db=None):
            self.media_db = media_db

        def process_videos(self, **kwargs):
            calls.append(kwargs)
            kwargs["transcription_progress_callback"](
                20.0,
                "Uploading audio for transcription",
                synthetic_data,
            )
            kwargs["transcription_progress_callback"](
                91.0,
                "Transcribing segment 3 of 8",
                measured_data,
            )
            return {
                "results": [
                    {
                        "status": "Success",
                        "content": "Video transcript",
                        "metadata": {"title": "Video", "author": "Unknown"},
                        "chunks": [],
                        "analysis": "",
                    }
                ]
            }

    monkeypatch.setattr(
        "tldw_chatbook.Local_Ingestion.local_file_ingestion.LocalVideoProcessor",
        _StubVideoProcessor,
    )

    parse_local_file_for_ingest(
        str(source),
        {
            "transcription_provider": "faster-whisper",
            "transcription_model_dir": None,
            "transcription_model": "medium",
            "language": "fr",
            "translation_target_language": "en",
            "transcription_precision": "int8",
            "transcription_local_files_only": True,
            "transcription_batch_route_resolved": True,
            "timestamps": True,
            "diarization": False,
        },
        progress_callback=lambda phase, message, percent=None: progress_events.append(
            (phase, message, percent)
        ),
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["transcription_provider"] == "faster-whisper"
    assert call["transcription_model_dir"] is None
    assert call["transcription_model"] == "medium"
    assert call["transcription_language"] == "fr"
    assert call["translation_target_language"] == "en"
    assert call["transcription_precision"] == "int8"
    assert call["transcription_local_files_only"] is True
    assert call["transcription_batch_route_resolved"] is True
    assert call["timestamp_option"] is True
    assert call["diarize"] is False
    assert (
        "transcribing",
        "Uploading audio for transcription",
        None,
    ) in progress_events
    assert (
        "transcribing",
        "Transcribing segment 3 of 8",
        37.5,
    ) in progress_events
    assert all(synthetic_data not in event for event in progress_events)
    assert all(measured_data not in event for event in progress_events)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    (
        ({"current_time": 1.5, "total_time": 6.0}, 25.0),
        ({"chunk": 3, "total_chunks": 8}, 37.5),
        ({"current": 4, "total": 5}, 80.0),
        ({"current": 0, "total": 5}, 0.0),
        ({"current": 5, "total": 5}, 100.0),
        ({"stage": "processing", "private": object()}, None),
        ({"percent": 37.0}, None),
        (None, None),
        ((1, 2), None),
        ({"current": True, "total": 2}, None),
        ({"current": 1, "total": False}, None),
        ({"current": "1", "total": 2}, None),
        ({"current": 1, "total": "2"}, None),
        ({"current": float("nan"), "total": 2}, None),
        ({"current": 1, "total": float("inf")}, None),
        ({"current": 1, "total": 0}, None),
        ({"current": 1, "total": -2}, None),
        ({"current": -1, "total": 2}, None),
        ({"current": 3, "total": 2}, None),
    ),
)
def test_measured_transcription_percent_accepts_only_bounded_allowlisted_ratios(
    metadata: object,
    expected: float | None,
) -> None:
    assert local_file_ingestion._measured_transcription_percent(metadata) == expected


def test_measured_transcription_percent_ignores_hostile_mapping() -> None:
    class _HostileMapping(dict[str, object]):
        def __contains__(self, _key: object) -> bool:
            raise RuntimeError("hostile provider metadata")

    assert local_file_ingestion._measured_transcription_percent(_HostileMapping()) is None


def test_ebook_options_are_routed_to_process_ebook(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "book.epub"
    source.write_bytes(b"PK\x03\x04" + b"\x00" * 64)

    calls: list[Dict[str, Any]] = []

    def fake_process_ebook(**kwargs):
        calls.append(kwargs)
        return _make_ebook_result()

    monkeypatch.setattr(
        "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_ebook",
        fake_process_ebook,
    )

    parse_local_file_for_ingest(
        str(source),
        {
            "extraction_method": "markdown",
            "include_toc": False,
            "split_chapters": True,
        },
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["method"] == "markdown"
    assert call["include_toc"] is False
    assert call["split_chapters"] is True


def test_ebook_split_chapters_false_records_warning(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "book.epub"
    source.write_bytes(b"PK\x03\x04" + b"\x00" * 64)

    def fake_process_ebook(method=None, split_chapters=True, include_toc=True, **kwargs):
        warnings = []
        if not split_chapters:
            warnings.append("split_chapters=False")
        return _make_ebook_result(warnings=warnings)

    monkeypatch.setattr(
        "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_ebook",
        fake_process_ebook,
    )

    payload = parse_local_file_for_ingest(
        str(source),
        {
            "extraction_method": "filtered",
            "include_toc": True,
            "split_chapters": False,
        },
    )

    assert payload["content"] == "Ebook text"
    assert payload["warnings"] == ["split_chapters=False"]


# --- task-3307: image extension mapping --------------------------------------


def test_detect_file_type_maps_image_extensions_task_3307() -> None:
    """The raster formats process_image's PIL loader opens map to 'image'."""
    from tldw_chatbook.Local_Ingestion.local_file_ingestion import detect_file_type

    for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"):
        assert detect_file_type(f"/tmp/picture{ext}") == "image", ext
        assert detect_file_type(f"/tmp/PICTURE{ext.upper()}") == "image", ext


def test_detect_file_type_image_lookalikes_unsupported_task_3307() -> None:
    """svg (vector), ico (icon container), heic/heif (need the absent
    pillow_heif opener) stay unsupported -- and the error copy still names
    the supported set."""
    import pytest as _pytest

    from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
        FileIngestionError,
        detect_file_type,
    )

    for ext in (".svg", ".ico", ".heic", ".heif"):
        with _pytest.raises(FileIngestionError, match="Unsupported file type"):
            detect_file_type(f"/tmp/picture{ext}")


def test_get_supported_extensions_includes_image_task_3307() -> None:
    from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
        get_supported_extensions,
    )

    extensions = get_supported_extensions()
    assert extensions["image"] == [
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
    ]
