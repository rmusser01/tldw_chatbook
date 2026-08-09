"""Tests for the app-level Library ingest job submission seam."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import patch

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
import tldw_chatbook.app as app_module


def _minimal_app(media_db: Any = None) -> TldwCli:
    """Return a TldwCli instance without running its heavy __init__."""
    app = object.__new__(TldwCli)
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    app.media_db = media_db
    app._top_up_ingest_parse_pool = lambda: None  # type: ignore[method-assign]
    return app


def _minimal_stt_app() -> TldwCli:
    app = object.__new__(TldwCli)
    app._ingest_shutdown = False
    app._local_stt_executor_lock = threading.RLock()
    app._local_stt_executor = None
    app._local_stt_dispatch_coordinator = None
    app._marshal_local_stt_call = lambda callback: callback()  # type: ignore[method-assign]
    app._top_up_ingest_parse_pool = lambda: None  # type: ignore[method-assign]
    return app


def test_local_stt_accessors_share_one_executor_and_coordinator_without_deadlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _minimal_stt_app()
    created: list[object] = []

    def create_executor() -> object:
        executor = object()
        created.append(executor)
        return executor

    monkeypatch.setattr(app, "_create_local_stt_executor", create_executor)
    barrier = threading.Barrier(8)

    def resolve(index: int) -> tuple[object, object]:
        barrier.wait(timeout=2.0)
        if index % 2:
            coordinator = app._ensure_local_stt_dispatch_coordinator()
            executor = app._ensure_local_stt_executor()
        else:
            executor = app._ensure_local_stt_executor()
            coordinator = app._ensure_local_stt_dispatch_coordinator()
        return executor, coordinator

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(resolve, index) for index in range(8)]
        resolved = [future.result(timeout=2.0) for future in futures]

    executors = {id(executor) for executor, _coordinator in resolved}
    coordinators = {id(coordinator) for _executor, coordinator in resolved}
    assert len(created) == 1
    assert len(executors) == 1
    assert len(coordinators) == 1
    assert resolved[0][1]._executor is resolved[0][0]


def test_console_dictation_factory_injects_app_owned_coordinator() -> None:
    app = _minimal_stt_app()
    coordinator = object()
    app._ensure_local_stt_dispatch_coordinator = lambda: coordinator  # type: ignore[method-assign]

    class _FakeTranscriptionService:
        def __init__(self, *, local_stt_dispatcher: object) -> None:
            self.local_stt_dispatcher = local_stt_dispatcher

    class _FakeLazyService:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    with (
        patch(
            "tldw_chatbook.Audio.dictation_service_lazy.LazyLiveDictationService",
            _FakeLazyService,
        ),
        patch(
            "tldw_chatbook.Local_Ingestion.transcription_service.TranscriptionService",
            _FakeTranscriptionService,
        ),
    ):
        service = app._create_console_dictation_service(
            transcription_provider="parakeet-onnx",
            language="en",
        )

    assert service.kwargs["transcription_provider"] == "parakeet-onnx"
    assert service.kwargs["language"] == "en"
    transcription = service.kwargs["transcription_service_factory"]()
    assert transcription.local_stt_dispatcher is coordinator


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


def _direct_failed_attempt() -> dict[str, object]:
    return {
        "attempt_id": "attempt-1",
        "batch_id": None,
        "job_id": "ingest-job-1",
        "provider_id": "transcribe-cpp",
        "model_id": "local-gguf:whisper",
        "artifact_root": None,
        "artifact_dependencies": [],
        "precision": "native",
        "requested_device": "auto",
        "effective_device": None,
        "requested_language": "en",
        "effective_language": "en",
        "detected_language": None,
        "task": "transcribe",
        "error_code": "inference_failed",
    }


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
        # task-3301: overlap default is the generic schema default (100, the
        # value the UI shows), not the old hardcoded 50; ``max_size`` mirrors
        # ``size`` because ``improved_chunking_process`` reads that spelling;
        # no ``method`` is forced -- each consumer applies its own default.
        assert options["chunk_options"] == {
            "size": 1234,
            "max_size": 1234,
            "overlap": 100,
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
            "size": 2048,
            "max_size": 2048,
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
        assert options["transcription_batch_route_resolved"] is True
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
        assert options["transcription_batch_route_resolved"] is True

    def test_explicit_parakeet_f32_is_preserved_in_worker_options(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp4",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "parakeet-onnx",
                    "transcription_precision": "F32",
                    "language": "de",
                },
            },
        )

        options = app._ingest_job_options(job)

        assert options["transcription_model"] == "nemo-parakeet-tdt-0.6b-v3"
        assert options["transcription_precision"] == "f32"

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

    def test_document_group_options(self) -> None:
        """(task-3303 AC1) The document branch feeds ``process_document``."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/report.docx",
            ingest_options={
                "generic": {"chunk": True, "chunk_size": 800, "chunk_overlap": 80},
                "document": {
                    "processing_method": "docling",
                    "ocr": True,
                    "ocr_language": "de",
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["processing_method"] == "docling"
        assert options["enable_ocr"] is True
        assert options["ocr_language"] == "de"
        # The generic base group still applies to document files: analyze/
        # chunk/size travel exactly as they did when documents rode the
        # generic panel (task-3301's layering).
        assert options["chunk_options"] == {
            "size": 800,
            "max_size": 800,
            "overlap": 80,
        }

    def test_document_group_defaults_without_snapshot(self) -> None:
        app = _minimal_app()
        job = _make_job(source_path="/tmp/report.odt")

        options = app._ingest_job_options(job)

        assert options["processing_method"] == "auto"
        assert options["enable_ocr"] is False
        assert options["ocr_language"] == "en"

    def test_pdf_ocr_language_and_backend_travel(self) -> None:
        """(task-3303 AC2) OCR language/backend reach the pdf options."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.pdf",
            ingest_options={
                "pdf": {
                    "pdf_engine": "docext",
                    "ocr": True,
                    "ocr_language": "fr",
                    "ocr_backend": "tesseract",
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["pdf_engine"] == "docext"
        assert options["ocr"] is True
        assert options["ocr_language"] == "fr"
        assert options["ocr_backend"] == "tesseract"

    def test_pdf_ocr_detail_defaults(self) -> None:
        app = _minimal_app()
        job = _make_job(source_path="/tmp/test.pdf")

        options = app._ingest_job_options(job)

        assert options["ocr_language"] == "en"
        assert options["ocr_backend"] == "auto"

    def test_ebook_chapters_choice_maps_to_ebook_chapters_method(self) -> None:
        """(task-3303 AC3) The human "chapters" choice becomes the real method."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/book.epub",
            ingest_options={
                "generic": {"chunk": True, "chunk_size": 1000},
                "ebook": {"chunk_method": "chapters"},
            },
        )
        options = app._ingest_job_options(job)

        assert options["chunk_options"]["method"] == "ebook_chapters"

    def test_ebook_sentences_choice_travels_verbatim(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/book.epub",
            ingest_options={
                "generic": {"chunk": True},
                "ebook": {"chunk_method": "sentences"},
            },
        )
        options = app._ingest_job_options(job)

        assert options["chunk_options"]["method"] == "sentences"

    def test_ebook_legacy_snapshot_without_chunk_method_keeps_sentences(
        self,
    ) -> None:
        """(task-3303 xhigh review round 2, F11) A persisted job whose
        snapshot predates the ebook ``chunk_method`` field must keep the
        pre-branch chunking scheme on retry/requeue: the old builder forced
        ``method='sentences'`` for every group, so falling through to
        ``process_epub``'s chapters default silently switched a legacy
        job's scheme. Post-branch snapshots always carry the field (the
        submit-time snapshot seeds the schema default), so absence IS the
        legacy marker."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/book.epub",
            ingest_options={"generic": {"chunk": True}},
        )
        options = app._ingest_job_options(job)

        assert options["chunk_options"] is not None
        assert options["chunk_options"]["method"] == "sentences"

    def test_fresh_untouched_form_snapshot_maps_to_ebook_chapters(self) -> None:
        """(F11) A FRESH untouched submission still chunks e-books by
        chapter: the submit-time snapshot carries the schema default
        ("chapters"), which the builder maps to the chunker's real
        ``ebook_chapters`` -- distinguishing it from a legacy snapshot
        where the field is absent."""
        screen = object.__new__(LibraryScreen)
        screen._library_ingest_form = LibraryIngestFormState()
        snapshot = screen._build_ingest_options_snapshot()

        assert snapshot["ebook"]["chunk_method"] == "chapters"

        app = _minimal_app()
        job = _make_job(source_path="/tmp/book.epub", ingest_options=snapshot)
        options = app._ingest_job_options(job)

        assert options["chunk_options"]["method"] == "ebook_chapters"

    def test_ebook_method_ignored_when_chunking_off(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/book.epub",
            ingest_options={
                "generic": {"chunk": False},
                "ebook": {"chunk_method": "chapters"},
            },
        )
        options = app._ingest_job_options(job)

        assert options["chunk_options"] is None

    def test_translate_to_english_maps_to_target_language(self) -> None:
        """(task-3303 AC4) The translate toggle becomes target_language=en."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "faster-whisper",
                    "translate_to_english": True,
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["translation_target_language"] == "en"
        assert options["transcription_provider"] == "faster-whisper"

    def test_translate_under_default_provider_routes_to_faster_whisper(
        self,
    ) -> None:
        """Only faster-whisper translates; the semantic default must route
        there rather than to Parakeet when translation is requested."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {"translate_to_english": True},
            },
        )
        options = app._ingest_job_options(job)

        assert options["transcription_provider"] == "faster-whisper"
        assert options["translation_target_language"] == "en"

    def test_translate_off_sets_no_target_language(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {"translate_to_english": False},
            },
        )
        options = app._ingest_job_options(job)

        assert options["translation_target_language"] is None

    def test_explicit_target_language_wins_over_translate_checkbox(self) -> None:
        """An explicit target (retry overrides, older snapshots) stays
        authoritative; the checkbox only fills the gap."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "faster-whisper",
                    "translation_target_language": "en",
                    "translate_to_english": False,
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["translation_target_language"] == "en"

    def test_stale_translate_under_transcribe_cpp_is_ignored_not_fatal(
        self,
    ) -> None:
        """(task-3303 xhigh review round 2, F9) The translate checkbox is
        gated in the FORM to the default/faster-whisper providers
        (``enabled_when_values``), but a value checked under one provider
        and left stale after switching to transcribe-cpp used to be
        forwarded anyway -- ``resolve_batch_stt_route`` then raised
        ``BatchSTTRoutingError`` and every audio/video job in the batch
        FAILED at dispatch. The builder must consult the same schema gate
        the form does."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "transcribe-cpp",
                    "translate_to_english": True,
                },
            },
        )

        options = app._ingest_job_options(job)  # must not raise

        assert options["transcription_provider"] == "transcribe-cpp"
        assert options["translation_target_language"] is None

    def test_stale_translate_under_parakeet_is_ignored_not_fatal(self) -> None:
        """(F9) Same stale-checkbox hazard under parakeet-onnx, which
        rejects translation outright in ``_parakeet_route``."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "parakeet-onnx",
                    "language": "en",
                    "translate_to_english": True,
                },
            },
        )

        options = app._ingest_job_options(job)  # must not raise

        assert options["transcription_provider"] == "parakeet-onnx"
        assert options["translation_target_language"] is None

    def test_vad_filter_travels(self) -> None:
        """(task-3303 AC4) The VAD toggle reaches the transcription options."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={"audio_video": {"vad_filter": True}},
        )
        options = app._ingest_job_options(job)

        assert options["vad_filter"] is True

    def test_vad_filter_defaults_off(self) -> None:
        app = _minimal_app()
        job = _make_job(source_path="/tmp/test.mp3")

        options = app._ingest_job_options(job)

        assert options["vad_filter"] is False

    def test_trim_fields_travel_stripped(self) -> None:
        """(task-3306) Start/Stop trim bounds reach the worker options."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {"start_time": " 0:30 ", "end_time": "10:00"}
            },
        )
        options = app._ingest_job_options(job)

        assert options["start_time"] == "0:30"
        assert options["end_time"] == "10:00"

    def test_trim_fields_default_unbounded(self) -> None:
        """Untouched (or blank) trim inputs mean no trim at all."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={"audio_video": {"start_time": "", "end_time": "  "}},
        )
        options = app._ingest_job_options(job)

        assert options["start_time"] is None
        assert options["end_time"] is None

    def test_cookies_file_maps_to_use_cookies_and_cookies(self) -> None:
        """(task-3306) A cookies FILE PATH travels; its presence IS the
        use_cookies flag -- there is no separate toggle to go stale."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp4",
            ingest_options={
                "audio_video": {"cookies_file": " /home/u/cookies.txt "}
            },
        )
        options = app._ingest_job_options(job)

        assert options["use_cookies"] is True
        assert options["cookies"] == "/home/u/cookies.txt"

    def test_blank_cookies_file_means_no_cookies(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp4",
            ingest_options={"audio_video": {"cookies_file": "   "}},
        )
        options = app._ingest_job_options(job)

        assert options["use_cookies"] is False
        assert options["cookies"] is None

    def test_summarize_recursively_travels(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={"audio_video": {"summarize_recursively": True}},
        )
        options = app._ingest_job_options(job)

        assert options["summarize_recursively"] is True

    def test_summarize_recursively_defaults_off(self) -> None:
        app = _minimal_app()
        job = _make_job(source_path="/tmp/test.mp3")

        options = app._ingest_job_options(job)

        assert options["summarize_recursively"] is False

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

    def test_transcribe_cpp_reads_dedicated_path_into_private_worker_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        secret_path = "/private/models/speech.gguf"
        monkeypatch.setattr(
            app_module,
            "get_cli_setting",
            lambda key, *args: secret_path
            if key == "transcription.transcribe_cpp.model_path"
            else args[0]
            if args
            else None,
        )
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "transcribe-cpp",
                    "language": "en",
                    "timestamps": True,
                }
            },
        )

        options = app._ingest_job_options(job)

        assert options["transcription_provider"] == "transcribe-cpp"
        assert options["transcription_model"] is None
        assert options["transcription_precision"] == "native"
        assert options["language"] == "en"
        assert options["transcription_context"] == {
            "model_path": secret_path,
            "attempt_id": "ingest-job-test-attempt-1",
            "batch_id": None,
            "job_id": "ingest-job-test",
            "retry_of_attempt_id": None,
            "retry_of_job_id": None,
            "retry_source_failure_provenance": None,
        }
        assert "transcription_model_path" not in options
        assert secret_path not in str(job.ingest_options)

    def test_untouched_exact_faster_whisper_model_uses_visible_base_default(
        self,
    ) -> None:
        screen = object.__new__(LibraryScreen)
        screen._library_ingest_form = LibraryIngestFormState()
        screen._library_ingest_form.type_options["audio_video"] = {
            "transcription_provider": "faster-whisper",
        }
        snapshot = screen._build_ingest_options_snapshot()

        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options=snapshot,
        )

        options = app._ingest_job_options(job)

        assert options["transcription_provider"] == "faster-whisper"
        assert options["transcription_model"] == "base"

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

    def test_pdf_chunk_options_carry_explicit_words_method(self) -> None:
        """(task-3301/3303 xhigh review round 2, F12) The generic chunk-size
        hint promises WORDS ('words · 100–5000'), but ``process_pdf``
        setdefaults ``method='sentences'`` when none travels -- a ~10-30x
        unit lie (500 SENTENCES is roughly one chunk per document). The
        builder now makes the hint true by always sending the words
        method for the pdf group."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.pdf",
            ingest_options={"generic": {"chunk": True, "chunk_size": 500}},
        )
        options = app._ingest_job_options(job)

        assert options["chunk_options"]["method"] == "words"

    def test_audio_video_chunk_options_carry_explicit_words_method(self) -> None:
        """(F12) Same unit contract for the audio/video branch, whose
        processor defaults to sentences as well."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={"generic": {"chunk": True, "chunk_size": 500}},
        )
        options = app._ingest_job_options(job)

        assert options["chunk_options"]["method"] == "words"

    def test_pdf_chunk_size_governs_word_budget_through_processor_tail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(F12) GOVERNANCE, not kwargs-arrival: the builder's pdf
        chunk_options drive ``process_pdf``'s REAL chunking tail (its own
        sentences-setdefault seam plus the real chunking service), and a
        size of 120 must mean ~120 WORDS per chunk. Only the
        pymupdf-backed extraction/metadata seams are stubbed (pymupdf is
        absent in this venv); the chunker is never stubbed
        (kwargs-arrival-vs-governance lesson). With the method injection
        dropped, the processor's sentences default makes this word-soup
        content ONE chunk and the test goes RED."""
        from tldw_chatbook.Local_Ingestion import PDF_Processing_Lib

        content = " ".join(f"word{i}" for i in range(600))
        monkeypatch.setattr(
            PDF_Processing_Lib,
            "pymupdf4llm_parse_pdf",
            lambda *args, **kwargs: content,
        )

        class _FakePyMuPDF:
            class FileDataError(Exception):
                pass

            class EmptyFileError(Exception):
                pass

            @staticmethod
            def open(*args, **kwargs):
                raise RuntimeError("pymupdf absent in this test venv")

        monkeypatch.setattr(PDF_Processing_Lib, "pymupdf", _FakePyMuPDF)

        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/budget.pdf",
            ingest_options={
                "generic": {"chunk": True, "chunk_size": 120, "chunk_overlap": 0}
            },
        )
        options = app._ingest_job_options(job)

        result = PDF_Processing_Lib.process_pdf(
            file_input=b"%PDF-1.4 fake bytes",
            filename="budget.pdf",
            perform_chunking=True,
            chunk_options=options["chunk_options"],
        )

        chunks = result["chunks"]
        assert chunks and len(chunks) >= 4, (
            f"size 120 produced {len(chunks or [])} chunk(s) from 600 words "
            "-- the size unit is not words"
        )
        for chunk in chunks:
            assert len(chunk["text"].split()) <= 120

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


class TestIngestJobOptionsWiring:
    """task-3301: the dead controls resolve to real option values."""

    def test_untouched_overlap_default_is_schema_default(self) -> None:
        """Local fallback overlap == the generic schema default (100), the
        value the UI displays -- it used to be a hardcoded 50."""
        from tldw_chatbook.Library.ingest_capabilities import get_capabilities

        schema_overlap = next(
            f.default
            for f in get_capabilities("generic").fields
            if f.name == "chunk_overlap"
        )
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"chunk": True, "chunk_size": 1000}},
        )

        options = app._ingest_job_options(job)

        assert options["chunk_options"]["overlap"] == schema_overlap == 100

    def test_untouched_form_local_and_server_paths_agree_on_overlap(self) -> None:
        from tldw_chatbook.Library.server_ingest_request import (
            build_server_ingest_kwargs,
        )

        screen = object.__new__(LibraryScreen)
        screen._library_ingest_form = LibraryIngestFormState()
        snapshot = screen._build_ingest_options_snapshot()

        app = _minimal_app()
        job = _make_job(source_path="/tmp/test.txt", ingest_options=snapshot)
        local_options = app._ingest_job_options(job)
        server_kwargs = build_server_ingest_kwargs(
            "/tmp/test.txt", options=snapshot
        )

        assert local_options["chunk_options"] is not None
        assert (
            local_options["chunk_options"]["overlap"]
            == server_kwargs["chunk_overlap"]
        )
        assert (
            local_options["chunk_options"]["size"] == server_kwargs["chunk_size"]
        )

    def test_display_string_sizes_are_coerced_to_int(self) -> None:
        """The panel Inputs hand back display text; processors get ints."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={
                "generic": {
                    "chunk": True,
                    "chunk_size": "1000",
                    "chunk_overlap": "150",
                }
            },
        )

        options = app._ingest_job_options(job)

        assert options["chunk_options"]["size"] == 1000
        assert options["chunk_options"]["overlap"] == 150
        assert isinstance(options["chunk_options"]["size"], int)
        assert isinstance(options["chunk_options"]["overlap"], int)

    def test_chunk_options_carry_max_size_for_chunking_service(self) -> None:
        """``improved_chunking_process`` reads ``max_size``; the legacy
        audio/video option map reads ``size``. Both spellings must travel."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"chunk": True, "chunk_size": 777}},
        )

        options = app._ingest_job_options(job)

        assert options["chunk_options"]["size"] == 777
        assert options["chunk_options"]["max_size"] == 777

    def test_encoding_selection_reaches_options(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"encoding": "latin-1"}},
        )

        options = app._ingest_job_options(job)

        assert options["encoding"] == "latin-1"

    def test_analysis_provider_resolved_from_config(self) -> None:
        app = _minimal_app()
        app.app_config = {
            "analysis_defaults": {"provider": "OpenAI"},
            "api_settings": {"openai": {"api_key": "sk-test-configured"}},
        }
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert options["perform_analysis"] is True
        # (task-3301 xhigh review round) The NORMALIZED dispatch name
        # travels -- it is what `chat_api_call` (and the summarizer's
        # alias map) accept; the display spelling only ever fed logs.
        assert options["api_name"] == "openai"
        assert options["api_key"] == "sk-test-configured"
        assert "analysis_skipped_reason" not in options

    def test_analysis_provider_resolved_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-env")
        app = _minimal_app()
        app.app_config = {"analysis_defaults": {"provider": "OpenAI"}}
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert options["api_name"] == "openai"
        assert options["api_key"] == "sk-test-env"

    def test_unready_analysis_records_skip_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        app = _minimal_app()
        app.app_config = {"analysis_defaults": {"provider": "OpenAI"}}
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert options.get("api_key") is None
        assert options["analysis_skipped_reason"]
        assert "OpenAI" in options["analysis_skipped_reason"]

    def test_no_provider_configured_records_skip_reason(self) -> None:
        app = _minimal_app()
        app.app_config = {}
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert options["analysis_skipped_reason"]
        assert "provider" in options["analysis_skipped_reason"]

    def test_analyze_off_skips_provider_resolution_entirely(self) -> None:
        app = _minimal_app()
        app.app_config = {
            "analysis_defaults": {"provider": "OpenAI"},
            "api_settings": {"openai": {"api_key": "sk-test-configured"}},
        }
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": False}},
        )

        options = app._ingest_job_options(job)

        assert options["perform_analysis"] is False
        assert "api_name" not in options
        assert "api_key" not in options
        assert "analysis_skipped_reason" not in options

    def test_analysis_call_settings_travel(self) -> None:
        """(task-3301 xhigh review round, F10) The full [analysis_defaults]
        call shape travels to the worker, not just the provider name."""
        app = _minimal_app()
        app.app_config = {
            "analysis_defaults": {
                "provider": "OpenAI",
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "top_p": 0.9,
                "min_p": 0.01,
                "max_tokens": 512,
                "system_prompt": "Analyze thoroughly.",
            },
            "api_settings": {"openai": {"api_key": "sk-test-configured"}},
        }
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert options["analysis_call"] == {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "top_p": 0.9,
            "min_p": 0.01,
            "max_tokens": 512,
        }
        assert options["system_prompt"] == "Analyze thoroughly."

    def test_keyless_provider_sets_explicit_opt_in(self) -> None:
        """(task-3301 xhigh review round, F8) Keyless-ready providers get
        the explicit opt-in flag; keyed providers never do."""
        app = _minimal_app()
        app.app_config = {"analysis_defaults": {"provider": "Ollama"}}
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert options["api_name"] == "ollama"
        assert options.get("api_key") is None
        assert options["analysis_keyless_ok"] is True

    def test_keyed_provider_does_not_set_keyless_opt_in(self) -> None:
        app = _minimal_app()
        app.app_config = {
            "analysis_defaults": {"provider": "OpenAI"},
            "api_settings": {"openai": {"api_key": "sk-test-configured"}},
        }
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert "analysis_keyless_ok" not in options

    def test_undispatchable_provider_records_skip_reason(self) -> None:
        """(task-3301 xhigh review round, F5) A readiness-ready provider
        with no chat dispatch handler must skip with a reason, not error
        at analysis time."""
        app = _minimal_app()
        app.app_config = {"analysis_defaults": {"provider": "custom"}}
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert "api_name" not in options
        assert "not supported for ingest analysis" in (
            options["analysis_skipped_reason"]
        )


class TestIngestDoneProgress:
    """task-3301: the done row records analysis skipped-with-reason."""

    def test_plain_import_message(self) -> None:
        progress = app_module._library_ingest_done_progress(
            "/tmp/notes.txt", was_duplicate=False, payload={}
        )
        assert progress == {"message": "Imported notes.txt"}

    def test_analysis_skip_reason_appended(self) -> None:
        progress = app_module._library_ingest_done_progress(
            "/tmp/notes.txt",
            was_duplicate=False,
            payload={
                "analysis_skipped_reason": "OpenAI is not ready (Missing API key)"
            },
        )
        assert (
            progress["message"]
            == "Imported notes.txt — analysis skipped: OpenAI is not ready "
            "(Missing API key)"
        )
        assert (
            progress["analysis_skipped"]
            == "OpenAI is not ready (Missing API key)"
        )

    def test_analysis_failed_reason_appended(self) -> None:
        """(task-3301 xhigh review round, F4) A failed analysis annotates
        the done row the same way a skipped one does -- never silence."""
        progress = app_module._library_ingest_done_progress(
            "/tmp/notes.txt",
            was_duplicate=False,
            payload={"analysis_failed_reason": "Invalid API Name 'custom'"},
        )
        assert (
            progress["message"]
            == "Imported notes.txt — analysis failed: Invalid API Name 'custom'"
        )
        assert progress["analysis_failed"] == "Invalid API Name 'custom'"

    def test_duplicate_message_keeps_matched_prefix(self) -> None:
        from tldw_chatbook.Library.library_ingest_jobs import (
            INGEST_DUPLICATE_PROGRESS_PREFIX,
        )

        progress = app_module._library_ingest_done_progress(
            "/tmp/notes.txt",
            was_duplicate=True,
            payload={"analysis_skipped_reason": "whatever"},
        )
        assert progress["message"].startswith(INGEST_DUPLICATE_PROGRESS_PREFIX)


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

    def test_explicit_faster_whisper_retry_overrides_only_provider_and_links_job(
        self,
    ) -> None:
        app = _minimal_app(media_db="present")
        original = app.submit_library_ingest_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "generic": {"chunk": True},
                "audio_video": {
                    "transcription_provider": "transcribe-cpp",
                    "language": "en",
                    "timestamps": True,
                },
            },
        )
        failed = app.library_ingest_jobs.mark_failed(
            original.job_id,
            error="Speech-to-text inference failed.",
            stt_failure_provenance=_direct_failed_attempt(),
        )

        retry = app.retry_library_ingest_job_with_provider(
            failed.job_id,
            "faster-whisper",
        )

        assert retry.retry_of_job_id == failed.job_id
        assert retry.retry_source_failure_provenance == _direct_failed_attempt()
        assert retry.ingest_options == {
            "generic": {"chunk": True},
            "audio_video": {
                "transcription_provider": "faster-whisper",
                "language": "en",
                "timestamps": True,
            },
        }
        assert original.ingest_options["audio_video"]["transcription_provider"] == (
            "transcribe-cpp"
        )


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = object.__new__(TldwCli)
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    app._ingest_shutdown = False
    app._ingest_parse_worker_count = lambda: 1  # type: ignore[method-assign]
    app._ingest_heavy_lane_max_workers = lambda: 1  # type: ignore[method-assign]
    app._ingest_parse_pool_generation = 1
    app._ingest_parse_jobs_by_generation = {1: set()}
    app._ingest_local_stt_jobs = {}
    warning_messages: list[str] = []
    monkeypatch.setattr(
        "tldw_chatbook.app.logger.warning",
        lambda message: warning_messages.append(str(message)),
    )

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
    routing_warnings = [
        message
        for message in warning_messages
        if "batch STT routing failed" in message
    ]
    assert len(routing_warnings) == 1
    assert invalid.job_id in routing_warnings[0]
    assert "detected_type=audio" in routing_warnings[0]
    assert error_fragment in routing_warnings[0]


# --- task-3307: image group branch in _ingest_job_options --------------------


class TestImageIngestJobOptions:
    def test_image_group_options_travel(self) -> None:
        """The image branch feeds ``process_image``'s OCR knobs."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/scan.png",
            ingest_options={
                "generic": {"chunk": True, "chunk_size": 800, "chunk_overlap": 80},
                "image": {
                    "ocr": True,
                    "ocr_language": "de",
                    "ocr_backend": "tesseract",
                },
            },
        )

        options = app._ingest_job_options(job)

        assert options["ocr"] is True
        assert options["ocr_language"] == "de"
        assert options["ocr_backend"] == "tesseract"
        # The generic base still applies (task-3301 layering), and the
        # chunk method is the explicit words unit the size hint promises
        # (task-3303 F12) -- process_image chunks the OCR text itself via
        # improved_chunking_process.
        assert options["chunk_options"] == {
            "size": 800,
            "max_size": 800,
            "overlap": 80,
            "method": "words",
        }

    def test_image_group_defaults_without_snapshot(self) -> None:
        """An untouched form mirrors the schema/processor defaults: OCR on
        (the extracted text IS the imported content), auto backend, en."""
        app = _minimal_app()
        job = _make_job(source_path="/tmp/scan.jpg")

        options = app._ingest_job_options(job)

        assert options["ocr"] is True
        assert options["ocr_language"] == "en"
        assert options["ocr_backend"] == "auto"

    def test_image_ocr_off_travels(self) -> None:
        """OCR off must reach the processor as False -- the parse then
        extracts nothing and the persist seam fails the job honestly."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/scan.png",
            ingest_options={"image": {"ocr": False}},
        )

        options = app._ingest_job_options(job)

        assert options["ocr"] is False
