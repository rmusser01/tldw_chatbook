"""End-to-end wiring tests for the advertised Library ingest options (task-3301).

Three ingest-canvas options were silent no-ops on the local path:

* **Analyze after import** — ``_ingest_job_options`` never supplied
  ``api_name``/``api_key``, and plaintext/html/article hardcoded
  ``analysis: ""``.
* **Chunk content** — OFF was overridden by hardcoded
  ``perform_chunking=True`` for pdf/ebook/audio/video; ON never chunked
  text types because the DB layer ignores ``chunk_options``.
* **Encoding** — consumed nowhere; utf-8 hardcoded for plaintext/html.

These tests drive the REAL seams: ``parse_local_file_for_ingest`` with real
fixture files, ``persist_parsed_media`` against a real ``MediaDatabase`` on
disk (tmp_path), and processor stubs whose keyword arguments are validated
against the real processors' ``inspect.signature`` so a fake can never
drift from the seam it stands in for.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
    parse_local_file_for_ingest,
    persist_parsed_media,
)


# ---------------------------------------------------------------------------
# Signature guards: the stubs below must only ever be called with keyword
# arguments the REAL processor accepts. A stub that silently accepts a
# kwarg the real function would reject is a fake matching our own call.
# ---------------------------------------------------------------------------


def _assert_kwargs_accepted(real_func: Any, kwargs: Dict[str, Any]) -> None:
    """Fail when ``kwargs`` contains a name the real callable rejects."""
    sig = inspect.signature(real_func)
    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kw:
        return
    unknown = set(kwargs) - set(sig.parameters)
    assert not unknown, (
        f"call would crash against the real seam: unknown kwargs {sorted(unknown)}"
    )


def _real_process_pdf():
    from tldw_chatbook.Local_Ingestion.PDF_Processing_Lib import process_pdf

    return process_pdf


def _real_process_ebook():
    from tldw_chatbook.Local_Ingestion.Book_Ingestion_Lib import process_ebook

    return process_ebook


def _real_analyze():
    from tldw_chatbook.LLM_Calls.Summarization_General_Lib import analyze

    return analyze


@pytest.fixture()
def media_db(tmp_path: Path) -> MediaDatabase:
    db = MediaDatabase(tmp_path / "media.db", client_id="test-ingest-wiring")
    yield db
    db.close_connection()


def _chunk_rows(db: MediaDatabase, media_id: int) -> list[str]:
    cursor = db.execute_query(
        "SELECT chunk_text FROM UnvectorizedMediaChunks WHERE media_id = ? "
        "ORDER BY chunk_index",
        (media_id,),
    )
    return [row["chunk_text"] for row in cursor.fetchall()]


def _chunking_status(db: MediaDatabase, media_id: int) -> str:
    cursor = db.execute_query(
        "SELECT chunking_status FROM Media WHERE id = ?", (media_id,)
    )
    return cursor.fetchone()["chunking_status"]


_MANY_SENTENCES = " ".join(
    f"Sentence number {i} carries a bit of body text for chunking." for i in range(40)
)


# ---------------------------------------------------------------------------
# Encoding (AC #3)
# ---------------------------------------------------------------------------


class TestEncodingSelection:
    def test_plaintext_latin1_selection_decodes_correctly(self, tmp_path: Path):
        source = tmp_path / "latin1.txt"
        source.write_bytes("café résumé naïveté".encode("latin-1"))

        payload = parse_local_file_for_ingest(str(source), {"encoding": "latin-1"})

        assert payload["content"] == "café résumé naïveté"

    def test_plaintext_cp1252_selection_decodes_correctly(self, tmp_path: Path):
        source = tmp_path / "cp1252.txt"
        source.write_bytes("smart “quotes” and – dashes".encode("cp1252"))

        payload = parse_local_file_for_ingest(str(source), {"encoding": "cp1252"})

        assert payload["content"] == "smart “quotes” and – dashes"

    def test_plaintext_utf16_selection_decodes_correctly(self, tmp_path: Path):
        source = tmp_path / "utf16.txt"
        source.write_bytes("wide chars: äöü".encode("utf-16"))

        payload = parse_local_file_for_ingest(str(source), {"encoding": "utf-16"})

        assert payload["content"] == "wide chars: äöü"

    def test_plaintext_auto_keeps_clean_utf8(self, tmp_path: Path):
        source = tmp_path / "utf8.txt"
        source.write_text("plain utf-8 with é accents", encoding="utf-8")

        payload = parse_local_file_for_ingest(str(source), {"encoding": "auto"})

        assert payload["content"] == "plain utf-8 with é accents"

    def test_plaintext_absent_encoding_defaults_to_auto(self, tmp_path: Path):
        source = tmp_path / "plain.txt"
        source.write_text("no encoding option at all", encoding="utf-8")

        payload = parse_local_file_for_ingest(str(source), {})

        assert payload["content"] == "no encoding option at all"

    def test_html_latin1_selection_decodes_correctly(self, tmp_path: Path):
        source = tmp_path / "latin1.html"
        source.write_bytes(
            "<html><head><title>Résumé</title></head>"
            "<body><p>café société</p></body></html>".encode("latin-1")
        )

        payload = parse_local_file_for_ingest(str(source), {"encoding": "latin-1"})

        assert "café société" in payload["content"]
        # NOTE: the payload title is the file stem, not the <title> tag --
        # pre-existing behavior (the branch's title-tag extraction only
        # fires when no title was defaulted, and the stem default always
        # runs first). Pinned here as-is; out of task-3301's scope.
        assert payload["title"] == "latin1"

    def test_html_latin1_under_utf8_selection_does_not_crash(self, tmp_path: Path):
        """An explicit wrong choice degrades to replacement chars, not a crash.

        The pre-task code opened HTML with strict utf-8, so latin-1 bytes
        raised ``UnicodeDecodeError`` and failed the whole job.
        """
        source = tmp_path / "latin1.html"
        source.write_bytes("<html><body><p>café x</p></body></html>".encode("latin-1"))

        payload = parse_local_file_for_ingest(str(source), {"encoding": "utf-8"})

        assert "caf" in payload["content"]
        assert "café" not in payload["content"]


# ---------------------------------------------------------------------------
# Chunk toggle: OFF must reach the pdf/ebook/audio/video processors (AC #2)
# ---------------------------------------------------------------------------


class TestChunkToggleReachesProcessors:
    def test_pdf_chunk_off_passes_perform_chunking_false(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4 stub")
        real = _real_process_pdf()
        assert "perform_chunking" in inspect.signature(real).parameters
        calls: list[Dict[str, Any]] = []

        def fake_process_pdf(**kwargs):
            _assert_kwargs_accepted(real, kwargs)
            calls.append(kwargs)
            return {
                "content": "PDF text",
                "title": "t",
                "author": "a",
                "keywords": [],
                "chunks": [{"text": "PDF text", "metadata": {"chunk_num": 0}}],
                "analysis": "",
                "metadata": {},
                "error": None,
                "warnings": [],
            }

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_pdf",
            fake_process_pdf,
        )

        # Chunk OFF == no chunk_options in the job options.
        payload = parse_local_file_for_ingest(str(source), {})

        assert calls[0]["perform_chunking"] is False
        # The processor's chunking-disabled single-chunk fallback must not be
        # stored either: Chunk OFF means no chunk rows.
        assert payload["chunks"] is None

    def test_pdf_chunk_on_passes_perform_chunking_true(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4 stub")
        real = _real_process_pdf()
        calls: list[Dict[str, Any]] = []

        def fake_process_pdf(**kwargs):
            _assert_kwargs_accepted(real, kwargs)
            calls.append(kwargs)
            return {
                "content": "PDF text",
                "title": "t",
                "author": "a",
                "keywords": [],
                "chunks": [{"text": "PDF text", "metadata": {"chunk_num": 0}}],
                "analysis": "",
                "metadata": {},
                "error": None,
                "warnings": [],
            }

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_pdf",
            fake_process_pdf,
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {"chunk_options": {"method": "sentences", "max_size": 500, "overlap": 100}},
        )

        assert calls[0]["perform_chunking"] is True
        assert payload["chunks"] == [
            {"text": "PDF text", "metadata": {"chunk_num": 0}}
        ]

    def test_ebook_chunk_off_passes_perform_chunking_false(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "book.epub"
        source.write_bytes(b"PK\x03\x04" + b"\x00" * 32)
        real = _real_process_ebook()
        assert "perform_chunking" in inspect.signature(real).parameters
        calls: list[Dict[str, Any]] = []

        def fake_process_ebook(**kwargs):
            _assert_kwargs_accepted(real, kwargs)
            calls.append(kwargs)
            return {
                "content": "Ebook text",
                "title": "t",
                "author": "a",
                "keywords": [],
                "chunks": [],
                "analysis": "",
                "metadata": {},
                "error": None,
                "warnings": [],
            }

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_ebook",
            fake_process_ebook,
        )

        parse_local_file_for_ingest(str(source), {})

        assert calls[0]["perform_chunking"] is False

    def test_audio_chunk_off_passes_perform_chunking_false(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "sound.mp3"
        source.write_bytes(b"ID3\x00" + b"\x00" * 32)
        from tldw_chatbook.Local_Ingestion.audio_processing import (
            LocalAudioProcessor as RealAudioProcessor,
        )

        real_method = RealAudioProcessor.process_audio_files
        assert "perform_chunking" in inspect.signature(real_method).parameters
        calls: list[Dict[str, Any]] = []

        class _StubAudioProcessor:
            def __init__(self, media_db=None):
                self.media_db = media_db

            def process_audio_files(self, **kwargs):
                _assert_kwargs_accepted(real_method, kwargs)
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

        parse_local_file_for_ingest(str(source), {})

        assert calls[0]["perform_chunking"] is False

    def test_video_chunk_off_passes_perform_chunking_false(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "clip.mp4"
        source.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 32)
        # ``process_videos`` forwards **kwargs into the audio pipeline, whose
        # ``process_audio_files`` declares ``perform_chunking`` explicitly.
        from tldw_chatbook.Local_Ingestion.audio_processing import (
            LocalAudioProcessor as RealAudioProcessor,
        )

        real_sink = RealAudioProcessor.process_audio_files
        calls: list[Dict[str, Any]] = []

        class _StubVideoProcessor:
            def __init__(self, media_db=None):
                self.media_db = media_db

            def process_videos(self, **kwargs):
                kwargs.pop("inputs", None)
                kwargs.pop("download_video_flag", None)
                _assert_kwargs_accepted(real_sink, kwargs)
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

        parse_local_file_for_ingest(str(source), {})

        assert calls[0]["perform_chunking"] is False


# ---------------------------------------------------------------------------
# Chunk toggle: text types, end-to-end through the real DB (AC #2)
# ---------------------------------------------------------------------------


class TestTextTypeChunkingEndToEnd:
    def test_plaintext_chunk_on_stores_chunks(
        self, tmp_path: Path, media_db: MediaDatabase
    ):
        source = tmp_path / "many.txt"
        source.write_text(_MANY_SENTENCES, encoding="utf-8")

        # The exact shape ``_ingest_job_options`` emits for Chunk ON: no
        # method (each consumer applies its own default -- the text tail
        # uses the chunking service's word method), size in both
        # spellings, overlap.
        payload = parse_local_file_for_ingest(
            str(source),
            {"chunk_options": {"size": 40, "max_size": 40, "overlap": 10}},
        )
        assert payload["chunks"], "Chunk ON must produce chunks for plaintext"
        assert len(payload["chunks"]) > 1

        media_id, _uuid, _msg = persist_parsed_media(payload, media_db)
        rows = _chunk_rows(media_db, media_id)
        assert len(rows) == len(payload["chunks"])
        assert _chunking_status(media_db, media_id) == "completed"

    def test_plaintext_chunk_size_governs_chunk_count(self, tmp_path: Path):
        source = tmp_path / "many.txt"
        source.write_text(_MANY_SENTENCES, encoding="utf-8")

        small = parse_local_file_for_ingest(
            str(source),
            {"chunk_options": {"size": 40, "max_size": 40, "overlap": 10}},
        )
        large = parse_local_file_for_ingest(
            str(source),
            {"chunk_options": {"size": 4000, "max_size": 4000, "overlap": 10}},
        )

        assert len(small["chunks"]) > len(large["chunks"])

    def test_plaintext_chunk_overlap_governs_chunk_text(self, tmp_path: Path):
        source = tmp_path / "many.txt"
        source.write_text(_MANY_SENTENCES, encoding="utf-8")

        no_overlap = parse_local_file_for_ingest(
            str(source),
            {"chunk_options": {"size": 40, "max_size": 40, "overlap": 0}},
        )
        with_overlap = parse_local_file_for_ingest(
            str(source),
            {"chunk_options": {"size": 40, "max_size": 40, "overlap": 20}},
        )

        assert len(with_overlap["chunks"]) > len(no_overlap["chunks"])

    def test_plaintext_chunk_off_stores_no_chunks(
        self, tmp_path: Path, media_db: MediaDatabase
    ):
        source = tmp_path / "many.txt"
        source.write_text(_MANY_SENTENCES, encoding="utf-8")

        payload = parse_local_file_for_ingest(str(source), {})
        assert payload["chunks"] is None

        media_id, _uuid, _msg = persist_parsed_media(payload, media_db)
        assert _chunk_rows(media_db, media_id) == []
        assert _chunking_status(media_db, media_id) == "pending"

    def test_html_chunk_on_stores_chunks(
        self, tmp_path: Path, media_db: MediaDatabase
    ):
        source = tmp_path / "many.html"
        source.write_text(
            f"<html><body><p>{_MANY_SENTENCES}</p></body></html>", encoding="utf-8"
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {"chunk_options": {"size": 40, "max_size": 40, "overlap": 10}},
        )
        assert payload["chunks"]
        assert len(payload["chunks"]) > 1

        media_id, _uuid, _msg = persist_parsed_media(payload, media_db)
        assert len(_chunk_rows(media_db, media_id)) == len(payload["chunks"])
        assert _chunking_status(media_db, media_id) == "completed"

    def test_document_chunk_on_chunks_processor_content(
        self, tmp_path: Path, monkeypatch, media_db: MediaDatabase
    ):
        source = tmp_path / "report.docx"
        source.write_bytes(b"PK\x03\x04" + b"\x00" * 32)
        from tldw_chatbook.Local_Ingestion.Document_Processing_Lib import (
            process_document as real_process_document,
        )

        def fake_process_document(**kwargs):
            _assert_kwargs_accepted(real_process_document, kwargs)
            return {
                "content": _MANY_SENTENCES,
                "title": "Report",
                "author": "Author",
                "metadata": {},
                "extraction_successful": True,
            }

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_document",
            fake_process_document,
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {"chunk_options": {"size": 40, "max_size": 40, "overlap": 10}},
        )
        assert payload["chunks"]
        assert len(payload["chunks"]) > 1

        media_id, _uuid, _msg = persist_parsed_media(payload, media_db)
        assert len(_chunk_rows(media_db, media_id)) == len(payload["chunks"])


# ---------------------------------------------------------------------------
# Analyze after import: text types (AC #1)
# ---------------------------------------------------------------------------


class TestTextTypeAnalysis:
    def test_plaintext_analysis_runs_and_persists(
        self, tmp_path: Path, monkeypatch, media_db: MediaDatabase
    ):
        source = tmp_path / "notes.txt"
        source.write_text("Some meaningful notes to analyze.", encoding="utf-8")
        real_analyze = _real_analyze()
        calls: list[Dict[str, Any]] = []

        def fake_analyze(**kwargs):
            _assert_kwargs_accepted(real_analyze, kwargs)
            calls.append(kwargs)
            return "STUB ANALYSIS."

        monkeypatch.setattr(
            "tldw_chatbook.LLM_Calls.Summarization_General_Lib.analyze",
            fake_analyze,
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {
                "perform_analysis": True,
                "api_name": "openai",
                "api_key": "sk-test-not-real",
            },
        )

        assert payload["analysis_content"] == "STUB ANALYSIS."
        assert calls[0]["api_name"] == "openai"
        assert calls[0]["api_key"] == "sk-test-not-real"

        media_id, _uuid, _msg = persist_parsed_media(payload, media_db)
        cursor = media_db.execute_query(
            "SELECT analysis_content FROM DocumentVersions WHERE media_id = ?",
            (media_id,),
        )
        stored = [row["analysis_content"] for row in cursor.fetchall()]
        assert "STUB ANALYSIS." in stored

    def test_html_analysis_runs(self, tmp_path: Path, monkeypatch):
        source = tmp_path / "page.html"
        source.write_text(
            "<html><body><p>Body text worth analyzing.</p></body></html>",
            encoding="utf-8",
        )
        real_analyze = _real_analyze()

        def fake_analyze(**kwargs):
            _assert_kwargs_accepted(real_analyze, kwargs)
            return "HTML ANALYSIS."

        monkeypatch.setattr(
            "tldw_chatbook.LLM_Calls.Summarization_General_Lib.analyze",
            fake_analyze,
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {"perform_analysis": True, "api_name": "openai", "api_key": "sk-x"},
        )

        assert payload["analysis_content"] == "HTML ANALYSIS."

    def test_plaintext_analysis_skipped_without_provider(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "notes.txt"
        source.write_text("Content.", encoding="utf-8")
        called = []

        monkeypatch.setattr(
            "tldw_chatbook.LLM_Calls.Summarization_General_Lib.analyze",
            lambda **kwargs: called.append(kwargs) or "NEVER",
        )

        payload = parse_local_file_for_ingest(
            str(source), {"perform_analysis": True}
        )

        assert payload["analysis_content"] == ""
        assert called == []

    def test_plaintext_analysis_failure_is_warning_not_job_failure(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "notes.txt"
        source.write_text("Content that resists analysis.", encoding="utf-8")

        def exploding_analyze(**kwargs):
            raise RuntimeError("provider exploded")

        monkeypatch.setattr(
            "tldw_chatbook.LLM_Calls.Summarization_General_Lib.analyze",
            exploding_analyze,
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {"perform_analysis": True, "api_name": "openai", "api_key": "sk-x"},
        )

        assert payload["analysis_content"] == ""
        assert any("nalysis" in w for w in payload["warnings"])

    def test_analysis_skipped_reason_travels_to_payload(self, tmp_path: Path):
        source = tmp_path / "notes.txt"
        source.write_text("Content.", encoding="utf-8")

        payload = parse_local_file_for_ingest(
            str(source),
            {
                "perform_analysis": True,
                "analysis_skipped_reason": "OpenAI is not ready (Missing API key)",
            },
        )

        assert (
            payload["analysis_skipped_reason"]
            == "OpenAI is not ready (Missing API key)"
        )

    def test_no_skip_reason_key_when_not_requested(self, tmp_path: Path):
        source = tmp_path / "notes.txt"
        source.write_text("Content.", encoding="utf-8")

        payload = parse_local_file_for_ingest(str(source), {})

        assert not payload.get("analysis_skipped_reason")
