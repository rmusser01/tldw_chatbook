"""Tests for per-type ingestion option routing in ``local_file_ingestion``.

These tests verify that ``parse_local_file_for_ingest`` forwards the new
per-media-type options (produced by ``app._ingest_job_options``) into the
underlying processor functions with the correct keyword names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

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

    class _StubAudioProcessor:
        def __init__(self, media_db=None):
            self.media_db = media_db

        def process_audio_files(self, **kwargs):
            calls.append(kwargs)
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
            "transcription_model": "small",
            "language": "es",
            "timestamps": False,
            "diarization": True,
        },
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["transcription_model"] == "small"
    assert call["transcription_language"] == "es"
    assert call["timestamp_option"] is False
    assert call["diarize"] is True


def test_video_options_are_routed_to_processor(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 64)

    calls: list[Dict[str, Any]] = []

    class _StubVideoProcessor:
        def __init__(self, media_db=None):
            self.media_db = media_db

        def process_videos(self, **kwargs):
            calls.append(kwargs)
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
            "transcription_model": "medium",
            "language": "fr",
            "timestamps": True,
            "diarization": False,
        },
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["transcription_model"] == "medium"
    assert call["transcription_language"] == "fr"
    assert call["timestamp_option"] is True
    assert call["diarize"] is False


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
