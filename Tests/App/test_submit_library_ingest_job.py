"""Tests for the app-level Library ingest job submission seam."""

from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.Library.library_ingest_jobs import (
    DEFAULT_CHUNK_SIZE,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
)
from tldw_chatbook.app import TldwCli


def _minimal_app(media_db: Any = None) -> TldwCli:
    """Return a TldwCli instance without running its heavy __init__."""
    app = object.__new__(TldwCli)
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    app.media_db = media_db
    app._top_up_ingest_parse_pool = lambda: None  # type: ignore[method-assign]
    return app


def _make_job(
    *,
    source_path: str = "/tmp/test.txt",
    ingest_options: dict[str, Any] | None = None,
    perform_analysis: bool = False,
    chunk_enabled: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> LibraryIngestJob:
    """Build a minimal LibraryIngestJob for _ingest_job_options tests."""
    return LibraryIngestJob(
        job_id="ingest-job-test",
        source_path=source_path,
        perform_analysis=perform_analysis,
        chunk_enabled=chunk_enabled,
        chunk_size=chunk_size,
        ingest_options=ingest_options or {},
    )


class TestIngestJobOptions:
    """Coverage for TldwCli._ingest_job_options."""

    def test_empty_ingest_options_uses_deprecated_job_fields(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.txt",
            perform_analysis=True,
            chunk_enabled=True,
            chunk_size=1234,
        )
        options = app._ingest_job_options(job)

        assert options["title"] is None
        assert options["author"] is None
        assert options["keywords"] is None
        assert options["perform_analysis"] is True
        assert options["chunk_options"] == {
            "method": "sentences",
            "size": 1234,
            "overlap": 50,
        }

    def test_generic_ingest_options_override_deprecated_fields(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.txt",
            perform_analysis=False,
            chunk_enabled=False,
            chunk_size=DEFAULT_CHUNK_SIZE,
            ingest_options={
                "generic": {
                    "analyze": True,
                    "chunk": True,
                    "chunk_size": 2048,
                    "chunk_overlap": 100,
                }
            },
        )
        options = app._ingest_job_options(job)

        assert options["perform_analysis"] is True
        assert options["chunk_options"] == {
            "method": "sentences",
            "size": 2048,
            "overlap": 100,
        }

    def test_pdf_group_options(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.pdf",
            ingest_options={
                "generic": {"analyze": True},
                "pdf": {
                    "pdf_engine": "docling",
                    "extract_images": True,
                    "enable_ocr": True,
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["perform_analysis"] is True
        assert options["pdf_engine"] == "docling"
        assert options["extract_images"] is True
        assert options["ocr"] is True
        assert options["page_range"] is None

    def test_pdf_group_falls_back_to_canonical_names(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.pdf",
            ingest_options={
                "pdf": {
                    "engine": "pymupdf",
                    "pages": "1-10",
                    "ocr": True,
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["pdf_engine"] == "pymupdf"
        assert options["page_range"] == "1-10"
        assert options["ocr"] is True

    def test_audio_video_group_options(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_model": "small",
                    "language": "es",
                    "timestamps": False,
                    "diarization": True,
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["transcription_model"] == "small"
        assert options["language"] == "es"
        assert options["timestamps"] is False
        assert options["diarization"] is True

    def test_ebook_group_options(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.epub",
            ingest_options={
                "ebook": {
                    "html_converter": "html2text",
                    "extract_toc": False,
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["extraction_method"] == "html2text"
        assert options["include_toc"] is False
        assert options["split_chapters"] is True

    def test_ebook_group_options_canonical_extraction_method(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.epub",
            ingest_options={
                "ebook": {
                    "extraction_method": "markdown",
                    "include_toc": True,
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["extraction_method"] == "markdown"
        assert options["include_toc"] is True

    def test_type_specific_overrides_generic(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.pdf",
            ingest_options={
                "generic": {"analyze": False, "chunk_size": 100},
                "pdf": {"analyze": True, "chunk": True, "chunk_size": 999},
            },
        )
        options = app._ingest_job_options(job)

        assert options["perform_analysis"] is True
        assert options["chunk_options"]["size"] == 999

    def test_disabled_chunking_returns_none_chunk_options(self) -> None:
        app = _minimal_app()
        job = _make_job(source_path="/tmp/test.txt", chunk_enabled=False)
        options = app._ingest_job_options(job)

        assert options["chunk_options"] is None


class TestSubmitLibraryIngestJob:
    """Coverage for TldwCli.submit_library_ingest_job."""

    def test_submit_passes_ingest_options_to_registry(self) -> None:
        app = _minimal_app(media_db="present")
        ingest_options = {"generic": {"analyze": True}, "pdf": {"pdf_engine": "docling"}}
        job = app.submit_library_ingest_job(
            source_path="/tmp/test.pdf",
            ingest_options=ingest_options,
        )

        assert job.ingest_options == ingest_options
        stored = next(
            (j for j in app.library_ingest_jobs.jobs() if j.job_id == job.job_id),
            None,
        )
        assert stored is not None
        assert stored.ingest_options == ingest_options

    def test_submit_defaults_ingest_options_to_empty_dict(self) -> None:
        app = _minimal_app(media_db="present")
        job = app.submit_library_ingest_job(source_path="/tmp/test.txt")

        assert job.ingest_options == {}

    def test_submit_without_media_db_marks_job_failed(self) -> None:
        app = _minimal_app(media_db=None)
        job = app.submit_library_ingest_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        assert job.state.name == "FAILED"
        assert job.error == "Media database is unavailable."
        # ingest_options should still be preserved on the failed job.
        assert job.ingest_options == {"generic": {"analyze": True}}


@pytest.mark.parametrize(
    "source_path,expected_group",
    [
        ("/tmp/test.pdf", "pdf"),
        ("/tmp/test.mp3", "audio_video"),
        ("/tmp/test.epub", "ebook"),
        ("/tmp/test.txt", "generic"),
    ],
)
def test_ingest_job_options_detects_type_group(
    source_path: str, expected_group: str
) -> None:
    app = _minimal_app()
    job = _make_job(source_path=source_path)
    options = app._ingest_job_options(job)

    if expected_group == "pdf":
        assert "pdf_engine" in options
    elif expected_group == "audio_video":
        assert "transcription_model" in options
    elif expected_group == "ebook":
        assert "extraction_method" in options
    else:
        assert "pdf_engine" not in options
        assert "transcription_model" not in options
        assert "extraction_method" not in options
