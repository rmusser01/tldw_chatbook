"""Tests for the app-level Library ingest job submission seam."""

from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.Library.ingest_capabilities import get_capabilities
from tldw_chatbook.Library.library_ingest_jobs import (
    DEFAULT_CHUNK_SIZE,
    IngestJobState,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
)
from tldw_chatbook.Library.library_ingest_state import LibraryIngestFormState
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
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
                    "transcription_provider": "parakeet-onnx",
                    "transcription_model_dir": "/models/parakeet-v2-int8",
                    "transcription_model": "base",
                    "language": "en",
                    "timestamps": False,
                    "diarization": True,
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["transcription_provider"] == "parakeet-onnx"
        assert options["transcription_model_dir"] == "/models/parakeet-v2-int8"
        assert options["transcription_model"] == "nemo-parakeet-tdt-0.6b-v2"
        assert options["language"] == "en"
        assert options["transcription_precision"] == "int8"
        assert options["transcription_local_files_only"] is True
        assert options["timestamps"] is False
        assert options["diarization"] is True

    def test_supported_non_english_parakeet_route_uses_v3(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp4",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "parakeet-onnx",
                    "transcription_model_dir": "/models/parakeet-v3-int8",
                    "language": " DE ",
                },
            },
        )

        options = app._ingest_job_options(job)

        assert options["transcription_provider"] == "parakeet-onnx"
        assert options["transcription_model"] == "nemo-parakeet-tdt-0.6b-v3"
        assert options["transcription_model_dir"] == "/models/parakeet-v3-int8"
        assert options["language"] == "de"
        assert options["transcription_precision"] == "int8"
        assert options["transcription_local_files_only"] is True

    def test_parakeet_onnx_defaults_language_to_english(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "parakeet-onnx",
                },
            },
        )

        options = app._ingest_job_options(job)

        assert options["language"] == "en"
        assert options["transcription_model_dir"] is None

    @pytest.mark.parametrize(
        "provider",
        [None, "default"],
        ids=["absent-provider", "explicit-default"],
    )
    def test_semantic_default_stays_on_faster_whisper_and_drops_stale_model(
        self, provider: str | None
    ) -> None:
        app = _minimal_app()
        audio_options = {
            "transcription_model_dir": "/models/parakeet-v2-int8",
            "transcription_model": "small",
            "language": " FR ",
        }
        if provider is not None:
            audio_options["transcription_provider"] = provider
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={"audio_video": audio_options},
        )

        options = app._ingest_job_options(job)

        assert options["transcription_provider"] == "faster-whisper"
        assert options["transcription_model"] is None
        assert options["transcription_model_dir"] is None
        assert options["language"] == "fr"
        assert options["transcription_precision"] == "int8"
        assert options["transcription_local_files_only"] is True

    def test_faster_whisper_preserves_normalized_translation_target(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "faster-whisper",
                    "transcription_model": "small",
                    "language": " JA ",
                    "target_language": " EN ",
                },
            },
        )

        options = app._ingest_job_options(job)

        assert options["transcription_model"] == "small"
        assert options["language"] == "ja"
        assert options["translation_target_language"] == "en"

    def test_explicit_empty_translation_target_does_not_fall_back_to_alias(
        self,
    ) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "faster-whisper",
                    "language": "ja",
                    "translation_target_language": "",
                    "target_language": "fr",
                },
            },
        )

        options = app._ingest_job_options(job)

        assert options["translation_target_language"] is None

    def test_untouched_audio_form_snapshot_resolves_closed_gate_default(self) -> None:
        provider_field = next(
            field
            for field in get_capabilities("audio_video").fields
            if field.name == "transcription_provider"
        )
        screen = object.__new__(LibraryScreen)
        screen._library_ingest_form = LibraryIngestFormState()
        snapshot = screen._build_ingest_options_snapshot()
        submitted_audio_options = snapshot.get("audio_video", {})

        assert provider_field.default == "default"
        assert submitted_audio_options.get("transcription_provider") not in {
            "parakeet-onnx",
            "faster-whisper",
        }

        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options=snapshot,
        )

        options = app._ingest_job_options(job)

        assert options["transcription_provider"] == "faster-whisper"
        assert options["transcription_model_dir"] is None
        assert options["language"] == "en"
        assert options["transcription_precision"] == "int8"
        assert options["transcription_local_files_only"] is True

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
        ingest_options = {
            "generic": {"analyze": True},
            "pdf": {"pdf_engine": "docling"},
        }
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


@pytest.mark.parametrize(
    ("invalid_audio_options", "error_fragment"),
    [
        (
            {
                "transcription_provider": "parakeet-onnx",
                "language": "auto",
            },
            "Retry with faster-whisper",
        ),
        (
            {
                "transcription_provider": "parakeet-onnx",
                "language": 7,
            },
            "language",
        ),
        (
            {
                "transcription_provider": 0,
                "language": "en",
            },
            "provider",
        ),
        (
            {
                "transcription_provider": False,
                "language": "en",
            },
            "provider",
        ),
        (
            {
                "transcription_provider": "",
                "language": "en",
            },
            "Unsupported batch STT provider",
        ),
        (
            {
                "transcription_provider": "faster-whisper",
                "language": "en",
                "translation_target_language": 0,
            },
            "target_language",
        ),
        (
            {
                "transcription_provider": "faster-whisper",
                "language": "en",
                "target_language": False,
            },
            "target_language",
        ),
    ],
)
def test_invalid_audio_request_allows_next_job_to_dispatch(
    invalid_audio_options: dict[str, Any],
    error_fragment: str,
) -> None:
    app = object.__new__(TldwCli)
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    app._ingest_shutdown = False
    app._ingest_parse_worker_count = lambda: 1  # type: ignore[method-assign]
    app._ingest_heavy_lane_max_workers = lambda: 1  # type: ignore[method-assign]
    app._ingest_parse_pool_generation = 1
    app._ingest_parse_jobs_by_generation = {1: set()}

    invalid = app.library_ingest_jobs.submit(
        source_path="/tmp/invalid.mp3",
        detected_type="audio",
        ingest_options={"audio_video": invalid_audio_options},
    )
    valid = app.library_ingest_jobs.submit(
        source_path="/tmp/valid.mp3",
        detected_type="audio",
        ingest_options={
            "audio_video": {
                "transcription_provider": "faster-whisper",
                "transcription_model": "small",
                "language": "en",
            }
        },
    )

    class _Pool:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, ...]] = []

        def apply_async(self, function, args, callback, error_callback) -> None:
            self.calls.append((function, args, callback, error_callback))

    pool = _Pool()
    pool_creation_calls = 0

    def ensure_pool() -> _Pool:
        nonlocal pool_creation_calls
        pool_creation_calls += 1
        return pool

    app._ensure_ingest_parse_pool = ensure_pool  # type: ignore[method-assign]

    app._top_up_ingest_parse_pool()

    jobs_by_id = {job.job_id: job for job in app.library_ingest_jobs.jobs()}
    invalid_job = jobs_by_id[invalid.job_id]
    valid_job = jobs_by_id[valid.job_id]
    assert invalid_job.state is IngestJobState.FAILED
    assert invalid_job.permanent is False
    assert invalid_job.error is not None
    assert error_fragment in invalid_job.error
    assert "\n" not in invalid_job.error
    assert len(invalid_job.error) <= 200
    assert valid_job.state is IngestJobState.PARSING
    assert pool_creation_calls == 1
    assert len(pool.calls) == 1
    _, (source_path, options), _, _ = pool.calls[0]
    assert source_path == valid.source_path
    assert options["transcription_provider"] == "faster-whisper"
