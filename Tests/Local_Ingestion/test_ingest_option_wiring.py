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
import shutil
import subprocess
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


def _real_chat_api_call():
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call

    return chat_api_call


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
# Local overwrite persistence
# ---------------------------------------------------------------------------


class TestLocalOverwritePersistence:
    def test_matching_content_skips_off_and_updates_metadata_on(
        self, media_db: MediaDatabase
    ) -> None:
        """The Local overwrite control must update the matched row in place.

        This is deliberately a real SQLite test: the duplicate decision and
        metadata update both belong to ``MediaDatabase``, not a mock.
        """
        initial_payload = {
            "file_type": "plaintext",
            "title": "Original title",
            "media_type": "document",
            "content": "The same local document content.",
            "keywords": ["original"],
            "url": "file:///fixtures/local-overwrite.txt",
            "analysis_content": "",
            "author": "Original author",
            "chunks": None,
            "chunk_options": None,
        }
        changed_metadata_payload = {
            **initial_payload,
            "title": "Updated title",
            "author": "Updated author",
            "keywords": ["updated"],
        }

        media_id, _media_uuid, _message = persist_parsed_media(
            initial_payload, media_db
        )
        assert media_id is not None

        skipped_id, _skipped_uuid, _skipped_message = persist_parsed_media(
            changed_metadata_payload, media_db
        )
        assert skipped_id is None
        skipped_row = media_db.execute_query(
            "SELECT title, author FROM Media WHERE id = ?", (media_id,)
        ).fetchone()
        assert dict(skipped_row) == {
            "title": "Original title",
            "author": "Original author",
        }

        updated_id, _updated_uuid, _updated_message = persist_parsed_media(
            changed_metadata_payload, media_db, overwrite_existing=True
        )
        assert updated_id == media_id
        updated_row = media_db.execute_query(
            "SELECT title, author FROM Media WHERE id = ?", (media_id,)
        ).fetchone()
        assert dict(updated_row) == {
            "title": "Updated title",
            "author": "Updated author",
        }


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

    def test_plaintext_unknown_encoding_degrades_with_warning(self, tmp_path: Path):
        """(task-3301 xhigh review round 2, F13) An explicit encoding name
        Python's codec registry doesn't know (``utf8-bom`` is not a codec;
        the real name is ``utf-8-sig``) must degrade to utf-8-with-replace
        plus a visible warning -- the documented degrade-not-fail contract.
        Before the fix, the explicit path let ``LookupError`` escape and the
        whole job failed while the auto path caught the same error class.
        """
        source = tmp_path / "bom.txt"
        source.write_text("perfectly fine utf-8 text", encoding="utf-8")

        payload = parse_local_file_for_ingest(str(source), {"encoding": "utf8-bom"})

        assert payload["content"] == "perfectly fine utf-8 text"
        encoding_warnings = [
            w for w in payload["warnings"] if "utf8-bom" in w
        ]
        assert encoding_warnings, (
            f"no warning names the unknown encoding: {payload['warnings']}"
        )

    def test_html_unknown_encoding_degrades_with_warning(self, tmp_path: Path):
        """(F13) Same contract on the HTML branch."""
        source = tmp_path / "bom.html"
        source.write_bytes(b"<html><body><p>body text here</p></body></html>")

        payload = parse_local_file_for_ingest(str(source), {"encoding": "utf8-bom"})

        assert "body text here" in payload["content"]
        assert any("utf8-bom" in w for w in payload["warnings"])


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
    """The text tail must land on a provably-dispatching call path.

    (task-3301 xhigh review round) The original tail called
    ``Summarization_General_Lib.analyze``, whose no-chunking direct dispatch
    sits in the dead ``else`` of ``if CHUNKER_AVAILABLE:`` -- with the chunk
    lib importable (every normal install) it returned
    ``'Error: Summarization failed unexpectedly.'`` WITHOUT any API call,
    and the tail's ``str-and-strip`` success check then persisted that
    in-band error string as the analysis. These tests therefore stub at the
    ``chat_api_call`` boundary -- the same unified dispatcher the Media
    viewer's analysis panel spends through -- and NEVER mock the tail's own
    helper: a stub that records a dispatch proves a dispatch.
    """

    def _install_chat_stub(self, monkeypatch, response: Any = "DISPATCHED ANALYSIS."):
        real = _real_chat_api_call()
        calls: list[Dict[str, Any]] = []

        def fake_chat_api_call(**kwargs):
            _assert_kwargs_accepted(real, kwargs)
            calls.append(kwargs)
            if isinstance(response, Exception):
                raise response
            return response

        monkeypatch.setattr(
            "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
            fake_chat_api_call,
        )
        return calls

    def _forbid_summarizer_path(self, monkeypatch):
        """The legacy analyze() path must not be what produces the result."""

        def exploding_analyze(*args, **kwargs):
            raise AssertionError(
                "text tail called Summarization analyze(); it must dispatch "
                "through chat_api_call"
            )

        monkeypatch.setattr(
            "tldw_chatbook.LLM_Calls.Summarization_General_Lib.analyze",
            exploding_analyze,
        )

    def test_plaintext_analysis_dispatches_and_persists(
        self, tmp_path: Path, monkeypatch, media_db: MediaDatabase
    ):
        source = tmp_path / "notes.txt"
        source.write_text("Some meaningful notes to analyze.", encoding="utf-8")
        calls = self._install_chat_stub(monkeypatch)
        self._forbid_summarizer_path(monkeypatch)

        payload = parse_local_file_for_ingest(
            str(source),
            {
                "perform_analysis": True,
                "api_name": "openai",
                "api_key": "sk-test-not-real",
            },
        )

        assert payload["analysis_content"] == "DISPATCHED ANALYSIS."
        assert calls, "no dispatch reached the chat_api_call boundary"
        assert calls[0]["api_endpoint"] == "openai"
        assert calls[0]["api_key"] == "sk-test-not-real"
        assert calls[0]["streaming"] is False
        # The document content must actually travel in the payload.
        assert "Some meaningful notes to analyze." in (
            calls[0]["messages_payload"][0]["content"]
        )

        media_id, _uuid, _msg = persist_parsed_media(payload, media_db)
        cursor = media_db.execute_query(
            "SELECT analysis_content FROM DocumentVersions WHERE media_id = ?",
            (media_id,),
        )
        stored = [row["analysis_content"] for row in cursor.fetchall()]
        assert "DISPATCHED ANALYSIS." in stored

    def test_analysis_call_settings_travel_to_the_dispatch(
        self, tmp_path: Path, monkeypatch
    ):
        """(F10) model/temperature/max_tokens/top_p/min_p + system prompt
        must reach the provider call, not be silently replaced."""
        source = tmp_path / "notes.txt"
        source.write_text("Configured analysis fidelity.", encoding="utf-8")
        calls = self._install_chat_stub(monkeypatch)

        payload = parse_local_file_for_ingest(
            str(source),
            {
                "perform_analysis": True,
                "api_name": "openai",
                "api_key": "sk-test-not-real",
                "system_prompt": "Analyze like the viewer would.",
                "analysis_call": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "min_p": 0.01,
                    "max_tokens": 512,
                },
            },
        )

        assert payload["analysis_content"] == "DISPATCHED ANALYSIS."
        call = calls[0]
        assert call["model"] == "gpt-4o-mini"
        assert call["temp"] == 0.2
        assert call["topp"] == 0.9
        assert call["minp"] == 0.01
        assert call["max_tokens"] == 512
        assert call["system_message"] == "Analyze like the viewer would."

    def test_html_analysis_dispatches(self, tmp_path: Path, monkeypatch):
        source = tmp_path / "page.html"
        source.write_text(
            "<html><body><p>Body text worth analyzing.</p></body></html>",
            encoding="utf-8",
        )
        self._install_chat_stub(monkeypatch, response="HTML ANALYSIS.")
        self._forbid_summarizer_path(monkeypatch)

        payload = parse_local_file_for_ingest(
            str(source),
            {"perform_analysis": True, "api_name": "openai", "api_key": "sk-x"},
        )

        assert payload["analysis_content"] == "HTML ANALYSIS."

    def test_openai_shaped_dict_response_is_extracted(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "notes.txt"
        source.write_text("Dict-shaped provider response.", encoding="utf-8")
        self._install_chat_stub(
            monkeypatch,
            response={"choices": [{"message": {"content": "DICT ANALYSIS."}}]},
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {"perform_analysis": True, "api_name": "openai", "api_key": "sk-x"},
        )

        assert payload["analysis_content"] == "DICT ANALYSIS."

    def test_plaintext_analysis_skipped_without_provider(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "notes.txt"
        source.write_text("Content.", encoding="utf-8")
        calls = self._install_chat_stub(monkeypatch)

        payload = parse_local_file_for_ingest(
            str(source), {"perform_analysis": True}
        )

        assert payload["analysis_content"] == ""
        assert calls == []

    def test_api_name_without_key_skips_instead_of_spending(
        self, tmp_path: Path, monkeypatch
    ):
        """(F8) Direct callers passing ``api_name`` with no credential get
        the historical silent skip -- never a call that would fall back to
        whatever key sits in config."""
        source = tmp_path / "notes.txt"
        source.write_text("Content.", encoding="utf-8")
        calls = self._install_chat_stub(monkeypatch)

        payload = parse_local_file_for_ingest(
            str(source), {"perform_analysis": True, "api_name": "openai"}
        )

        assert payload["analysis_content"] == ""
        assert calls == []

    def test_keyless_opt_in_allows_dispatch_without_key(
        self, tmp_path: Path, monkeypatch
    ):
        """(F8) The Library seam's explicit keyless opt-in -- set only after
        readiness said the provider is keyless-ready -- re-enables the call."""
        source = tmp_path / "notes.txt"
        source.write_text("Content.", encoding="utf-8")
        calls = self._install_chat_stub(monkeypatch, response="LOCAL ANALYSIS.")

        payload = parse_local_file_for_ingest(
            str(source),
            {
                "perform_analysis": True,
                "api_name": "ollama",
                "analysis_keyless_ok": True,
            },
        )

        assert payload["analysis_content"] == "LOCAL ANALYSIS."
        assert calls[0]["api_endpoint"] == "ollama"
        assert calls[0].get("api_key") is None

    def test_error_prefixed_response_is_failure_not_analysis(
        self, tmp_path: Path, monkeypatch
    ):
        """(F4) analyze()-style in-band error strings must never persist."""
        source = tmp_path / "notes.txt"
        source.write_text("Content.", encoding="utf-8")
        self._install_chat_stub(
            monkeypatch, response="Error: Invalid API Name 'openai'"
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {"perform_analysis": True, "api_name": "openai", "api_key": "sk-x"},
        )

        assert payload["analysis_content"] == ""
        assert any("Analysis failed" in w for w in payload["warnings"])
        assert "Invalid API Name" in payload["analysis_failed_reason"]

    def test_plaintext_analysis_failure_is_warning_not_job_failure(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "notes.txt"
        source.write_text("Content that resists analysis.", encoding="utf-8")
        self._install_chat_stub(
            monkeypatch, response=RuntimeError("provider exploded")
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {"perform_analysis": True, "api_name": "openai", "api_key": "sk-x"},
        )

        assert payload["analysis_content"] == ""
        assert any("nalysis" in w for w in payload["warnings"])
        assert "provider exploded" in payload["analysis_failed_reason"]

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


# ---------------------------------------------------------------------------
# (task-3301 xhigh review round, F4) Processor-returned analysis/summary
# values that are in-band error strings must never persist as analysis.
# ---------------------------------------------------------------------------


class TestProcessorAnalysisErrorStrings:
    def _pdf_stub_result(self, analysis: str) -> Dict[str, Any]:
        return {
            "content": "PDF text",
            "title": "t",
            "author": "a",
            "keywords": [],
            "chunks": [{"text": "PDF text", "metadata": {"chunk_num": 0}}],
            "analysis": analysis,
            "metadata": {},
            "error": None,
            "warnings": [],
        }

    def test_error_string_processor_analysis_is_not_persisted(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4 stub")
        real = _real_process_pdf()

        def fake_process_pdf(**kwargs):
            _assert_kwargs_accepted(real, kwargs)
            return self._pdf_stub_result(
                "Error: Summarization failed unexpectedly."
            )

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_pdf",
            fake_process_pdf,
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {"perform_analysis": True, "api_name": "openai", "api_key": "sk-x"},
        )

        assert payload["analysis_content"] == ""
        assert any("Analysis failed" in w for w in payload["warnings"])
        assert "Summarization failed" in payload["analysis_failed_reason"]

    def test_error_string_document_summary_is_not_persisted(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "report.docx"
        source.write_bytes(b"PK\x03\x04" + b"\x00" * 32)
        real = _real_process_document()

        def fake_process_document(**kwargs):
            _assert_kwargs_accepted(real, kwargs)
            result = _document_stub_result()
            result["summary"] = "Error: Invalid API Name 'koboldcpp'"
            return result

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_document",
            fake_process_document,
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {"perform_analysis": True, "api_name": "koboldcpp", "api_key": "k"},
        )

        assert payload["analysis_content"] == ""
        assert any("Analysis failed" in w for w in payload["warnings"])
        assert "Invalid API Name" in payload["analysis_failed_reason"]

    def test_real_document_summary_still_surfaces(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "report.docx"
        source.write_bytes(b"PK\x03\x04" + b"\x00" * 32)
        real = _real_process_document()

        def fake_process_document(**kwargs):
            _assert_kwargs_accepted(real, kwargs)
            result = _document_stub_result()
            result["summary"] = "A genuine document analysis."
            return result

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_document",
            fake_process_document,
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {"perform_analysis": True, "api_name": "openai", "api_key": "sk-x"},
        )

        assert payload["analysis_content"] == "A genuine document analysis."
        assert not payload.get("analysis_failed_reason")


# ---------------------------------------------------------------------------
# (task-3301 xhigh review round, F8) The pdf/ebook analysis gates require a
# credential OR the Library seam's explicit keyless opt-in. pymupdf/ebooklib
# are absent from this venv, so the gates themselves cannot be executed here;
# the shared predicate is unit-tested, its use is pinned in each gate's
# source, and the option's travel is checked at signature-checked stub
# boundaries.
# ---------------------------------------------------------------------------


class TestProcessorAnalysisCredentialGates:
    def test_credential_predicate_truth_table(self):
        from tldw_chatbook.Local_Ingestion.analysis_gate import (
            analysis_credentials_ok,
        )

        assert analysis_credentials_ok("sk-x") is True
        assert analysis_credentials_ok("sk-x", keyless_ok=True) is True
        assert analysis_credentials_ok(None, keyless_ok=True) is True
        assert analysis_credentials_ok(None) is False
        assert analysis_credentials_ok("") is False
        assert analysis_credentials_ok("", keyless_ok=False) is False

    @pytest.mark.parametrize(
        "module_name, func_name",
        [
            ("PDF_Processing_Lib", "process_pdf"),
            ("Book_Ingestion_Lib", "process_epub"),
            ("Book_Ingestion_Lib", "_process_markup_or_plain_text"),
            ("Book_Ingestion_Lib", "process_mobi"),
            ("Book_Ingestion_Lib", "process_fb2"),
        ],
    )
    def test_gates_accept_keyless_ok_and_use_the_predicate(
        self, module_name: str, func_name: str
    ):
        import importlib

        module = importlib.import_module(
            f"tldw_chatbook.Local_Ingestion.{module_name}"
        )
        func = getattr(module, func_name)
        assert "keyless_ok" in inspect.signature(func).parameters
        # The gate must consult the shared predicate -- a re-relaxed gate
        # (perform_analysis and api_name only) turns this RED.
        assert "analysis_credentials_ok(" in inspect.getsource(func)

    def test_keyless_ok_travels_to_process_pdf(self, tmp_path: Path, monkeypatch):
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
                "chunks": [],
                "analysis": "",
                "metadata": {},
                "error": None,
                "warnings": [],
            }

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_pdf",
            fake_process_pdf,
        )

        parse_local_file_for_ingest(
            str(source),
            {
                "perform_analysis": True,
                "api_name": "ollama",
                "analysis_keyless_ok": True,
            },
        )

        assert calls[0]["keyless_ok"] is True

    def test_keyless_ok_defaults_false_for_process_ebook(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "book.epub"
        source.write_bytes(b"PK\x03\x04" + b"\x00" * 32)
        real = _real_process_ebook()
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

        parse_local_file_for_ingest(
            str(source),
            {"perform_analysis": True, "api_name": "openai", "api_key": "sk-x"},
        )

        assert calls[0]["keyless_ok"] is False


# ---------------------------------------------------------------------------
# task-3303 AC1: document options reach process_document
# ---------------------------------------------------------------------------


def _real_process_document():
    from tldw_chatbook.Local_Ingestion.Document_Processing_Lib import (
        process_document,
    )

    return process_document


def _document_stub_result(content: str = "Document text") -> Dict[str, Any]:
    return {
        "content": content,
        "title": "Doc",
        "author": "Author",
        "metadata": {},
        "extraction_successful": True,
    }


class TestDocumentOptionWiring:
    def test_processing_options_reach_process_document(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "report.docx"
        source.write_bytes(b"PK\x03\x04" + b"\x00" * 32)
        real = _real_process_document()
        for name in ("processing_method", "enable_ocr", "ocr_language"):
            assert name in inspect.signature(real).parameters
        calls: list[Dict[str, Any]] = []

        def fake_process_document(**kwargs):
            _assert_kwargs_accepted(real, kwargs)
            calls.append(kwargs)
            return _document_stub_result()

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_document",
            fake_process_document,
        )

        parse_local_file_for_ingest(
            str(source),
            {
                "processing_method": "docling",
                "enable_ocr": True,
                "ocr_language": "de",
            },
        )

        assert calls[0]["processing_method"] == "docling"
        assert calls[0]["enable_ocr"] is True
        assert calls[0]["ocr_language"] == "de"

    def test_document_defaults_match_the_real_signature(
        self, tmp_path: Path, monkeypatch
    ):
        """Absent options must hand the processor its OWN declared defaults."""
        source = tmp_path / "report.rtf"
        source.write_bytes(b"{\\rtf1 stub}")
        real = _real_process_document()
        sig = inspect.signature(real)
        calls: list[Dict[str, Any]] = []

        def fake_process_document(**kwargs):
            _assert_kwargs_accepted(real, kwargs)
            calls.append(kwargs)
            return _document_stub_result()

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_document",
            fake_process_document,
        )

        parse_local_file_for_ingest(str(source), {})

        assert (
            calls[0]["processing_method"]
            == sig.parameters["processing_method"].default
        )
        assert calls[0]["enable_ocr"] == sig.parameters["enable_ocr"].default
        assert calls[0]["ocr_language"] == sig.parameters["ocr_language"].default

    def test_document_still_gets_generic_chunking(
        self, tmp_path: Path, monkeypatch, media_db: MediaDatabase
    ):
        """The document group layers ON TOP of generic: moving .docx out of
        the generic panel must not cost it task-3301's chunking tail."""
        source = tmp_path / "report.docx"
        source.write_bytes(b"PK\x03\x04" + b"\x00" * 32)
        real = _real_process_document()

        def fake_process_document(**kwargs):
            _assert_kwargs_accepted(real, kwargs)
            return _document_stub_result(content=_MANY_SENTENCES)

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_document",
            fake_process_document,
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {
                "processing_method": "native",
                "chunk_options": {"size": 40, "max_size": 40, "overlap": 10},
            },
        )
        assert payload["chunks"]
        assert len(payload["chunks"]) > 1

        media_id, _uuid, _msg = persist_parsed_media(payload, media_db)
        assert len(_chunk_rows(media_db, media_id)) == len(payload["chunks"])


# ---------------------------------------------------------------------------
# task-3303 AC2: PDF OCR detail reaches process_pdf
# ---------------------------------------------------------------------------


class TestPdfOcrDetailWiring:
    def test_ocr_language_and_backend_reach_process_pdf(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "scan.pdf"
        source.write_bytes(b"%PDF-1.4 stub")
        real = _real_process_pdf()
        for name in ("ocr_language", "ocr_backend"):
            assert name in inspect.signature(real).parameters
        calls: list[Dict[str, Any]] = []

        def fake_process_pdf(**kwargs):
            _assert_kwargs_accepted(real, kwargs)
            calls.append(kwargs)
            return {
                "content": "PDF text",
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
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_pdf",
            fake_process_pdf,
        )

        parse_local_file_for_ingest(
            str(source),
            {
                "pdf_engine": "docext",
                "ocr": True,
                "ocr_language": "fr",
                "ocr_backend": "tesseract",
            },
        )

        assert calls[0]["ocr_language"] == "fr"
        assert calls[0]["ocr_backend"] == "tesseract"
        assert calls[0]["ocr"] is True
        assert calls[0]["engine"] == "docext"

    def test_pdf_ocr_detail_defaults_match_the_real_signature(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4 stub")
        real = _real_process_pdf()
        sig = inspect.signature(real)
        calls: list[Dict[str, Any]] = []

        def fake_process_pdf(**kwargs):
            _assert_kwargs_accepted(real, kwargs)
            calls.append(kwargs)
            return {
                "content": "PDF text",
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
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_pdf",
            fake_process_pdf,
        )

        parse_local_file_for_ingest(str(source), {})

        assert (
            calls[0]["ocr_language"] == sig.parameters["ocr_language"].default
        )
        assert calls[0]["ocr_backend"] == sig.parameters["ocr_backend"].default


# ---------------------------------------------------------------------------
# task-3303 AC3: the ebook chunk-method choice reaches process_ebook
# ---------------------------------------------------------------------------


class TestEbookChunkMethodWiring:
    def test_chunk_method_reaches_process_ebook(self, tmp_path: Path, monkeypatch):
        source = tmp_path / "book.epub"
        source.write_bytes(b"PK\x03\x04" + b"\x00" * 32)
        real = _real_process_ebook()
        calls: list[Dict[str, Any]] = []

        def fake_process_ebook(**kwargs):
            _assert_kwargs_accepted(real, kwargs)
            calls.append(kwargs)
            return {
                "content": "Ebook text",
                "title": "t",
                "author": "a",
                "keywords": [],
                "chunks": [
                    {"text": "Chapter 1", "metadata": {"chunk_num": 0}}
                ],
                "analysis": "",
                "metadata": {},
                "error": None,
                "warnings": [],
            }

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_ebook",
            fake_process_ebook,
        )

        parse_local_file_for_ingest(
            str(source),
            {
                "chunk_options": {
                    "method": "ebook_chapters",
                    "size": 1000,
                    "max_size": 1000,
                    "overlap": 100,
                }
            },
        )

        assert calls[0]["perform_chunking"] is True
        assert calls[0]["chunk_options"]["method"] == "ebook_chapters"

    def test_chapter_method_is_one_the_chunker_implements(self):
        """The mapped method name must exist in the real chunking stack --
        asserted against Chunk_Lib's own dispatch, not a copied list."""
        import inspect as _inspect

        from tldw_chatbook.Chunking import Chunk_Lib

        dispatch_source = _inspect.getsource(Chunk_Lib.Chunker.chunk_text)
        assert '"ebook_chapters"' in dispatch_source


# ---------------------------------------------------------------------------
# task-3303 AC4: AV translation + VAD reach the transcription call
# ---------------------------------------------------------------------------


def _audio_stub_result() -> Dict[str, Any]:
    return {
        "results": [
            {
                "status": "Success",
                "content": "Transcript",
                "metadata": {"title": "Audio", "author": "Unknown"},
                "chunks": [],
                "analysis": "",
            }
        ]
    }


class TestAVTranslationAndVadWiring:
    def test_translation_target_reaches_audio_processor(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "talk.mp3"
        source.write_bytes(b"ID3\x00" + b"\x00" * 32)
        from tldw_chatbook.Local_Ingestion.audio_processing import (
            LocalAudioProcessor as RealAudioProcessor,
        )

        real_method = RealAudioProcessor.process_audio_files
        assert (
            "translation_target_language"
            in inspect.signature(real_method).parameters
        )
        calls: list[Dict[str, Any]] = []

        class _StubAudioProcessor:
            def __init__(self, media_db=None):
                self.media_db = media_db

            def process_audio_files(self, **kwargs):
                _assert_kwargs_accepted(real_method, kwargs)
                calls.append(kwargs)
                return _audio_stub_result()

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.LocalAudioProcessor",
            _StubAudioProcessor,
        )

        parse_local_file_for_ingest(
            str(source), {"translation_target_language": "en"}
        )

        assert calls[0]["translation_target_language"] == "en"

    def test_vad_filter_reaches_audio_processor_as_vad_use(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "talk.mp3"
        source.write_bytes(b"ID3\x00" + b"\x00" * 32)
        from tldw_chatbook.Local_Ingestion.audio_processing import (
            LocalAudioProcessor as RealAudioProcessor,
        )

        real_method = RealAudioProcessor.process_audio_files
        assert "vad_use" in inspect.signature(real_method).parameters
        calls: list[Dict[str, Any]] = []

        class _StubAudioProcessor:
            def __init__(self, media_db=None):
                self.media_db = media_db

            def process_audio_files(self, **kwargs):
                _assert_kwargs_accepted(real_method, kwargs)
                calls.append(kwargs)
                return _audio_stub_result()

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.LocalAudioProcessor",
            _StubAudioProcessor,
        )

        parse_local_file_for_ingest(str(source), {"vad_filter": True})

        assert calls[0]["vad_use"] is True

    def test_vad_filter_reaches_video_processor_as_vad_use(
        self, tmp_path: Path, monkeypatch
    ):
        source = tmp_path / "clip.mp4"
        source.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 32)
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

        parse_local_file_for_ingest(str(source), {"vad_filter": True})

        assert calls[0]["vad_use"] is True

    def test_vad_absent_defaults_off(self, tmp_path: Path, monkeypatch):
        source = tmp_path / "talk.mp3"
        source.write_bytes(b"ID3\x00" + b"\x00" * 32)
        from tldw_chatbook.Local_Ingestion.audio_processing import (
            LocalAudioProcessor as RealAudioProcessor,
        )

        real_method = RealAudioProcessor.process_audio_files
        calls: list[Dict[str, Any]] = []

        class _StubAudioProcessor:
            def __init__(self, media_db=None):
                self.media_db = media_db

            def process_audio_files(self, **kwargs):
                _assert_kwargs_accepted(real_method, kwargs)
                calls.append(kwargs)
                return _audio_stub_result()

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.LocalAudioProcessor",
            _StubAudioProcessor,
        )

        parse_local_file_for_ingest(str(source), {})

        assert calls[0]["vad_use"] is False


# ---------------------------------------------------------------------------
# (task-3301 xhigh review round 2, F7) Public wrapper chunk defaults.
#
# When task-3301 made ``chunk_options is None`` mean "do not chunk" at the
# parse seam (the Library queue's Chunk-content OFF), the public wrappers
# (``ingest_local_file``/``batch_ingest_files``/``quick_ingest``, exported
# via ``Local_Ingestion.__init__``) silently inherited that: their
# ``chunk_options=None`` DEFAULT now meant no chunking for every
# out-of-tree caller that previously got default chunking. The wrappers'
# omitted argument now means "chunk with defaults" ({}); an EXPLICIT
# ``None`` still means "do not chunk".
# ---------------------------------------------------------------------------

#: >500 words so the text tail's default word budget (500) must split it.
_MANY_WORDS = " ".join(
    f"word{i} filler body text keeps flowing steadily" for i in range(200)
)


class TestPublicWrapperChunkDefaults:
    def test_ingest_local_file_default_chunks_and_stores(
        self, tmp_path: Path, media_db: MediaDatabase
    ):
        from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
            ingest_local_file,
        )

        source = tmp_path / "wordy.txt"
        source.write_text(_MANY_WORDS, encoding="utf-8")

        result = ingest_local_file(source, media_db)

        assert result["chunks_created"] > 1, (
            "omitting chunk_options must mean 'chunk with defaults', not "
            "'never chunk'"
        )
        assert len(_chunk_rows(media_db, result["media_id"])) == (
            result["chunks_created"]
        )

    def test_ingest_local_file_explicit_none_stores_no_chunks(
        self, tmp_path: Path, media_db: MediaDatabase
    ):
        from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
            ingest_local_file,
        )

        source = tmp_path / "wordy_none.txt"
        source.write_text(_MANY_WORDS, encoding="utf-8")

        result = ingest_local_file(source, media_db, chunk_options=None)

        assert result["chunks_created"] == 0
        assert _chunk_rows(media_db, result["media_id"]) == []

    def test_batch_ingest_files_default_chunks(
        self, tmp_path: Path, media_db: MediaDatabase
    ):
        from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
            batch_ingest_files,
        )

        sources = []
        for i in range(2):
            source = tmp_path / f"batch{i}.txt"
            source.write_text(_MANY_WORDS, encoding="utf-8")
            sources.append(source)

        results = batch_ingest_files(sources, media_db)

        assert len(results) == 2
        for result in results:
            assert result["chunks_created"] > 1

    def test_quick_ingest_default_chunks(self, tmp_path: Path, monkeypatch):
        from tldw_chatbook.Local_Ingestion.local_file_ingestion import quick_ingest

        monkeypatch.setattr(
            "tldw_chatbook.config.get_media_db_path",
            lambda: tmp_path / "quick.db",
        )
        source = tmp_path / "quick.txt"
        source.write_text(_MANY_WORDS, encoding="utf-8")

        result = quick_ingest(source)

        assert result["chunks_created"] > 1


# ---------------------------------------------------------------------------
# task-3306: time-range trim, cookies file, recursive summary reach the
# processors; adaptive/multi-level chunking stays rejected while dead.
# ---------------------------------------------------------------------------


def _write_fake_mp3(tmp_path: Path, name: str = "talk.mp3") -> Path:
    source = tmp_path / name
    source.write_bytes(b"ID3\x00" + b"\x00" * 32)
    return source


def _write_fake_mp4(tmp_path: Path, name: str = "clip.mp4") -> Path:
    source = tmp_path / name
    source.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 32)
    return source


def _install_audio_stub(monkeypatch) -> list:
    """Stub LocalAudioProcessor, returning the captured call kwargs list."""
    from tldw_chatbook.Local_Ingestion.audio_processing import (
        LocalAudioProcessor as RealAudioProcessor,
    )

    real_method = RealAudioProcessor.process_audio_files
    calls: list = []

    class _StubAudioProcessor:
        def __init__(self, media_db=None):
            self.media_db = media_db

        def process_audio_files(self, **kwargs):
            _assert_kwargs_accepted(real_method, kwargs)
            calls.append(kwargs)
            return _audio_stub_result()

    monkeypatch.setattr(
        "tldw_chatbook.Local_Ingestion.local_file_ingestion.LocalAudioProcessor",
        _StubAudioProcessor,
    )
    return calls


def _install_video_stub(monkeypatch) -> list:
    """Stub LocalVideoProcessor, returning the captured call kwargs list.

    ``process_videos`` names inputs/download_video_flag/start_time/end_time
    itself and forwards the rest into the audio pipeline, so the signature
    guard runs against ``process_videos`` directly.
    """
    from tldw_chatbook.Local_Ingestion.video_processing import (
        LocalVideoProcessor as RealVideoProcessor,
    )

    real_method = RealVideoProcessor.process_videos
    calls: list = []

    class _StubVideoProcessor:
        def __init__(self, media_db=None):
            self.media_db = media_db

        def process_videos(self, **kwargs):
            _assert_kwargs_accepted(real_method, kwargs)
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
    return calls


class TestAVTrimWiring:
    def test_trim_reaches_audio_processor(self, tmp_path: Path, monkeypatch):
        source = _write_fake_mp3(tmp_path)
        from tldw_chatbook.Local_Ingestion.audio_processing import (
            LocalAudioProcessor as RealAudioProcessor,
        )

        params = inspect.signature(
            RealAudioProcessor.process_audio_files
        ).parameters
        assert "start_time" in params and "end_time" in params
        calls = _install_audio_stub(monkeypatch)

        parse_local_file_for_ingest(
            str(source), {"start_time": "0:15", "end_time": "1:30"}
        )

        assert calls[0]["start_time"] == "0:15"
        assert calls[0]["end_time"] == "1:30"

    def test_trim_reaches_video_processor(self, tmp_path: Path, monkeypatch):
        source = _write_fake_mp4(tmp_path)
        calls = _install_video_stub(monkeypatch)

        parse_local_file_for_ingest(
            str(source), {"start_time": "00:00:10", "end_time": "90"}
        )

        assert calls[0]["start_time"] == "00:00:10"
        assert calls[0]["end_time"] == "90"

    def test_trim_absent_defaults_unbounded(self, tmp_path: Path, monkeypatch):
        source = _write_fake_mp3(tmp_path)
        calls = _install_audio_stub(monkeypatch)

        parse_local_file_for_ingest(str(source), {})

        assert calls[0].get("start_time") is None
        assert calls[0].get("end_time") is None

    def test_video_extraction_trim_is_not_applied_twice(
        self, tmp_path: Path, monkeypatch
    ):
        """(task-3306) Governance: the ffmpeg trim must run ONCE.

        ``_process_single_video`` extracts audio with the requested
        start/end applied, then delegates to the audio stage with the same
        kwargs -- whose own trim path re-cuts any local non-YouTube input.
        Without dropping the bounds after extraction, a start of 60s is
        applied twice and the transcript window silently shifts to 120s.
        """
        from tldw_chatbook.Local_Ingestion.video_processing import (
            LocalVideoProcessor,
        )

        source = _write_fake_mp4(tmp_path)
        extracted = tmp_path / "extracted.mp3"
        extracted.write_bytes(b"ID3\x00" + b"\x00" * 32)

        processor = LocalVideoProcessor(None)
        extract_calls: list = []
        audio_calls: list = []

        def fake_extract(video_path, output_dir, start_time=None, end_time=None):
            extract_calls.append((start_time, end_time))
            return str(extracted)

        def fake_single_audio(input_item, processing_dir, **kwargs):
            audio_calls.append(kwargs)
            return {
                "status": "Success",
                "input_ref": input_item,
                "content": "Video transcript",
                "metadata": {"title": "Video", "author": "Unknown"},
                "segments": [],
                "chunks": [],
                "analysis": "",
                "warnings": [],
            }

        monkeypatch.setattr(
            processor, "_extract_audio_from_video", fake_extract
        )
        monkeypatch.setattr(
            processor.audio_processor, "_process_single_audio", fake_single_audio
        )

        result = processor.process_videos(
            inputs=[str(source)],
            download_video_flag=False,
            start_time="0:60",
            end_time="2:00",
        )

        assert result["results"][0]["status"] != "Error"
        assert extract_calls == [("0:60", "2:00")], (
            "the extraction stage must receive the requested bounds"
        )
        assert audio_calls[0].get("start_time") is None
        assert audio_calls[0].get("end_time") is None


# ---------------------------------------------------------------------------
# (task-3306 xhigh review round) "Stop at" must mean the SAME thing on both
# media paths. These are governance tests on the ffmpeg argv, not on ffmpeg
# output: neither ffmpeg nor a real media file is guaranteed present in this
# venv, so the assertion is on the WINDOW the constructed command line
# semantically requests. ``_interpret_ffmpeg_window`` below encodes ffmpeg's
# own rules independently of the builder, so a regression in the builder is
# not silently mirrored by the interpreter.
# ---------------------------------------------------------------------------


def _interpret_ffmpeg_window(argv: list[str]) -> tuple[float, float | None]:
    """Return the ABSOLUTE ``(start, end)`` source window ``argv`` requests.

    ffmpeg's rules, which the two ingest paths must not diverge on:

    * ``-ss`` BEFORE ``-i`` is input seeking; the output's timestamps are
      rebased to zero, so a subsequent output ``-to X`` stops at source
      ``start + X`` (a DURATION), while ``-t X`` is a duration too.
    * ``-ss`` AFTER ``-i`` is output seeking; timestamps are not rebased, so
      ``-to X`` stops at source ``X`` (ABSOLUTE) and ``-t X`` is a duration.
    """
    index = argv.index("-i")
    pre, post = argv[:index], argv[index + 2 :]

    def _flag(args: list[str], flag: str) -> float | None:
        if flag not in args:
            return None
        return _seconds(args[args.index(flag) + 1])

    def _seconds(text: str) -> float:
        total = 0.0
        for unit, part in enumerate(reversed(text.split(":"))):
            total += float(part) * (60**unit)
        return total

    input_ss = _flag(pre, "-ss")
    output_ss = _flag(post, "-ss")
    start = (input_ss or 0.0) + (output_ss or 0.0)

    duration = _flag(post, "-t")
    if duration is not None:
        return start, start + duration
    stop = _flag(post, "-to")
    if stop is not None:
        # Rebased timestamps (input seeking) make -to relative to the seek.
        return start, (input_ss + stop) if input_ss else stop
    return start, None


def _capture_video_extraction_argv(tmp_path: Path, monkeypatch, start, end):
    """Run ``_extract_audio_from_video`` with ffmpeg stubbed; return argv."""
    from tldw_chatbook.Local_Ingestion.video_processing import LocalVideoProcessor

    tmp_path.mkdir(parents=True, exist_ok=True)
    source = _write_fake_mp4(tmp_path)
    processor = LocalVideoProcessor(None)
    monkeypatch.setattr(processor, "_find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    captured: list[list[str]] = []

    def fake_run(command, **kwargs):
        captured.append(list(command))
        Path(command[-1]).write_bytes(b"ID3\x00")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    processor._extract_audio_from_video(
        str(source), str(tmp_path), start_time=start, end_time=end
    )
    return captured[0]


def _capture_audio_extraction_argv(tmp_path: Path, monkeypatch, start, end):
    """Run ``_extract_time_range`` with ffmpeg stubbed; return argv."""
    from tldw_chatbook.Local_Ingestion.audio_processing import LocalAudioProcessor

    tmp_path.mkdir(parents=True, exist_ok=True)
    source = _write_fake_mp3(tmp_path)
    processor = LocalAudioProcessor(None)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    captured: list[list[str]] = []

    def fake_run(command, **kwargs):
        captured.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    processor._extract_time_range(
        str(source), str(tmp_path), start_time=start, end_time=end
    )
    return captured[0]


class TestAVTrimArgvSemantics:
    def test_video_and_audio_mean_the_same_window(self, tmp_path: Path, monkeypatch):
        """Start 0:30 / Stop 1:00 must select 30s-60s on BOTH paths.

        The video path used input seeking (``-ss`` before ``-i``, which
        rebases output timestamps to zero) and then applied ``-to`` as an
        output option -- turning "Stop at 1:00" into "one minute AFTER the
        start", i.e. 0:30-1:30. The audio path, which puts ``-ss`` after
        ``-i``, produced the absolute 0:30-1:00 the label promises. Same
        two fields, same job, two different windows.
        """
        video_argv = _capture_video_extraction_argv(
            tmp_path / "v", monkeypatch, "0:30", "1:00"
        )
        audio_argv = _capture_audio_extraction_argv(
            tmp_path / "a", monkeypatch, "0:30", "1:00"
        )

        assert _interpret_ffmpeg_window(video_argv) == (30.0, 60.0), (
            f"video argv requests the wrong window: {video_argv}"
        )
        assert _interpret_ffmpeg_window(audio_argv) == (30.0, 60.0), (
            f"audio argv requests the wrong window: {audio_argv}"
        )
        assert _interpret_ffmpeg_window(video_argv) == _interpret_ffmpeg_window(
            audio_argv
        )

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            ("0:30", "1:00", (30.0, 60.0)),
            ("90", "150", (90.0, 150.0)),
            ("00:01:00", "00:01:30", (60.0, 90.0)),
            (None, "1:00", (0.0, 60.0)),
            ("0:30", None, (30.0, None)),
        ],
    )
    def test_both_paths_agree_across_formats(
        self, tmp_path: Path, monkeypatch, start, end, expected
    ):
        video_argv = _capture_video_extraction_argv(
            tmp_path / "v", monkeypatch, start, end
        )
        audio_argv = _capture_audio_extraction_argv(
            tmp_path / "a", monkeypatch, start, end
        )
        assert _interpret_ffmpeg_window(video_argv) == expected
        assert _interpret_ffmpeg_window(audio_argv) == expected

    def test_bounded_trim_keeps_fast_input_seeking(
        self, tmp_path: Path, monkeypatch
    ):
        """Correctness first, speed second -- but not speed sacrificed.

        Absolute-stop semantics could have been bought by moving ``-ss``
        after ``-i`` (output seeking decodes and throws away everything
        before the start). The shipped fix keeps input seeking and converts
        the absolute stop into the duration it implies, so a 2-hour file
        trimmed to its last minute still seeks instead of decoding.
        """
        argv = _capture_video_extraction_argv(
            tmp_path / "v", monkeypatch, "1:00:00", "1:01:00"
        )
        assert argv.index("-ss") < argv.index("-i"), (
            "bounded trims must keep pre-input (fast) seeking"
        )
        assert _interpret_ffmpeg_window(argv) == (3600.0, 3660.0)


class TestAVRecursiveSummaryWiring:
    def test_summarize_recursively_reaches_audio_processor(
        self, tmp_path: Path, monkeypatch
    ):
        source = _write_fake_mp3(tmp_path)
        from tldw_chatbook.Local_Ingestion.audio_processing import (
            LocalAudioProcessor as RealAudioProcessor,
        )

        params = inspect.signature(
            RealAudioProcessor.process_audio_files
        ).parameters
        assert "summarize_recursively" in params
        calls = _install_audio_stub(monkeypatch)

        parse_local_file_for_ingest(str(source), {"summarize_recursively": True})

        assert calls[0]["summarize_recursively"] is True

    def test_summarize_recursively_reaches_video_processor(
        self, tmp_path: Path, monkeypatch
    ):
        source = _write_fake_mp4(tmp_path)
        calls = _install_video_stub(monkeypatch)

        parse_local_file_for_ingest(str(source), {"summarize_recursively": True})

        assert calls[0]["summarize_recursively"] is True

    def test_legacy_chunk_options_spelling_still_works(
        self, tmp_path: Path, monkeypatch
    ):
        """Older callers tucked the flag into chunk_options; keep honoring it."""
        source = _write_fake_mp3(tmp_path)
        calls = _install_audio_stub(monkeypatch)

        parse_local_file_for_ingest(
            str(source), {"chunk_options": {"recursive_summary": True}}
        )

        assert calls[0]["summarize_recursively"] is True

    def test_absent_defaults_off(self, tmp_path: Path, monkeypatch):
        source = _write_fake_mp3(tmp_path)
        calls = _install_audio_stub(monkeypatch)

        parse_local_file_for_ingest(str(source), {})

        assert calls[0]["summarize_recursively"] is False

    def test_recursive_summary_changes_the_analysis_dispatch(self):
        """(task-3306) Governance at the consuming seam: with the flag ON
        and multiple chunks, ``_analyze_content`` runs the map-reduce path
        (one call per chunk + a combine call); OFF makes exactly one direct
        call. The exposed control provably changes the output shape.
        """
        from tldw_chatbook.Local_Ingestion.audio_processing import (
            LocalAudioProcessor,
        )

        processor = LocalAudioProcessor.__new__(LocalAudioProcessor)
        chunks = [{"text": "part one"}, {"text": "part two"}]
        for flag, expected_calls in ((True, 3), (False, 1)):
            calls: list = []

            def fake_chat_api_call(**kwargs):
                calls.append(kwargs)
                return f"summary {len(calls)}"

            import tldw_chatbook.Local_Ingestion.audio_processing as audio_mod

            original = audio_mod.chat_api_call
            audio_mod.chat_api_call = fake_chat_api_call
            try:
                processor._analyze_content(
                    content="part one part two",
                    chunks=chunks,
                    api_name="openai",
                    api_key="k",
                    custom_prompt=None,
                    system_prompt=None,
                    summarize_recursively=flag,
                )
            finally:
                audio_mod.chat_api_call = original
            assert len(calls) == expected_calls


class TestAVCookiesFileWiring:
    def test_cookies_file_reaches_video_processor(
        self, tmp_path: Path, monkeypatch
    ):
        source = _write_fake_mp4(tmp_path)
        calls = _install_video_stub(monkeypatch)

        parse_local_file_for_ingest(
            str(source),
            {"use_cookies": True, "cookies": "/home/user/cookies.txt"},
        )

        assert calls[0]["use_cookies"] is True
        assert calls[0]["cookies"] == "/home/user/cookies.txt"

    def test_cookies_absent_defaults_off_for_video(
        self, tmp_path: Path, monkeypatch
    ):
        source = _write_fake_mp4(tmp_path)
        calls = _install_video_stub(monkeypatch)

        parse_local_file_for_ingest(str(source), {})

        assert calls[0]["use_cookies"] is False
        assert calls[0].get("cookies") is None

    def test_cookies_never_forwarded_to_audio_processor(
        self, tmp_path: Path, monkeypatch
    ):
        """The audio downloader treats a cookies STRING as a JSON dict
        (``json.loads`` -> raw Cookie header); handing it the video path's
        cookiefile PATH would raise ``JSONDecodeError`` and fail the whole
        job. The audio branch therefore never forwards the option -- its
        yt-dlp (YouTube) path ignores cookies entirely anyway.
        """
        source = _write_fake_mp3(tmp_path)
        calls = _install_audio_stub(monkeypatch)

        parse_local_file_for_ingest(
            str(source),
            {"use_cookies": True, "cookies": "/home/user/cookies.txt"},
        )

        assert "use_cookies" not in calls[0]
        assert "cookies" not in calls[0]

    def test_cookies_problem_travels_to_the_payload(
        self, tmp_path: Path, monkeypatch
    ):
        """(xhigh review round) A cookies path the option boundary refused
        must be visible on the job, not swallowed. It rides the same
        options -> payload channel as the analysis skip reason."""
        source = _write_fake_mp4(tmp_path)
        _install_video_stub(monkeypatch)

        payload = parse_local_file_for_ingest(
            str(source),
            {
                "use_cookies": False,
                "cookies": None,
                "cookies_problem": "Cookies file not found: /tmp/gone.txt",
            },
        )

        assert payload["cookies_problem"] == (
            "Cookies file not found: /tmp/gone.txt"
        )
        assert "Cookies file not found: /tmp/gone.txt" in payload["warnings"]


class TestAdaptiveChunkingStaysRejected:
    def test_adaptive_and_multi_level_are_dead_at_the_real_chunker(self):
        """(task-3306) Rejection tripwire, not a wiring test.

        ``process_audio_files`` ACCEPTS use_adaptive_chunking /
        use_multi_level_chunking / chunk_language, but the only chunker on
        the audio/video path is ``ChunkingService.chunk_text``, which has
        no such parameters -- and ``_process_single_audio`` never reads the
        adaptive/multi-level kwargs at all (chunk_language lands only in
        per-chunk metadata). Exposing them would ship controls whose output
        cannot vary with the input. If this test ever fails, the chunker
        has grown the capability: re-open the exposure decision recorded in
        task-3306 instead of deleting the test.
        """
        from tldw_chatbook.RAG_Search.chunking_service import ChunkingService

        params = inspect.signature(ChunkingService.chunk_text).parameters
        assert "use_adaptive_chunking" not in params
        assert "adaptive" not in params
        assert "use_multi_level_chunking" not in params
        assert "multi_level" not in params
        assert "language" not in params

        from tldw_chatbook.Local_Ingestion.audio_processing import (
            LocalAudioProcessor,
        )

        single_audio_source = inspect.getsource(
            LocalAudioProcessor._process_single_audio
        )
        assert "use_adaptive_chunking" not in single_audio_source
        assert "use_multi_level_chunking" not in single_audio_source


# ---------------------------------------------------------------------------
# task-3307: image ingestion wiring (ship ruling recorded in task-3310)
# ---------------------------------------------------------------------------


def _real_process_image():
    from tldw_chatbook.Local_Ingestion.Image_Processing_Lib import process_image

    return process_image


def _real_extract_text_from_image():
    from tldw_chatbook.Local_Ingestion.Image_Processing_Lib import (
        extract_text_from_image,
    )

    return extract_text_from_image


def _write_tiny_png(path: Path) -> None:
    """A real 2x2 PNG -- Pillow is present in this venv (checked up front)."""
    from PIL import Image

    Image.new("RGB", (2, 2), (255, 255, 255)).save(path, "PNG")


def _install_ocr_stub(monkeypatch, text: str | None):
    """Stand in for ``extract_text_from_image`` at the OCR boundary.

    Signature-checked against the real function so the stub can never
    accept a call shape the real seam would reject. ``text=None`` models
    OCR failure / no backend installed (the real function returns None).
    """
    from types import SimpleNamespace

    real = _real_extract_text_from_image()
    calls: list[Dict[str, Any]] = []

    def fake_extract(image_path, **kwargs):
        _assert_kwargs_accepted(real, kwargs)
        calls.append({"image_path": image_path, **kwargs})
        if text is None:
            return None
        return SimpleNamespace(
            text=text,
            confidence=0.93,
            language=kwargs.get("language", "en"),
            backend="stub-backend",
            processing_time=0.01,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Local_Ingestion.Image_Processing_Lib.extract_text_from_image",
        fake_extract,
    )
    return calls


class TestImageWiring:
    def test_image_options_reach_process_image(self, tmp_path: Path, monkeypatch):
        """The parse branch forwards the panel's OCR knobs to the REAL
        ``process_image`` parameter names, keeps visual features off (their
        output is dropped by the persist path -- see the task notes), and
        keeps the processor's own analysis path off in favor of the arc's
        chat tail."""
        source = tmp_path / "scan.png"
        _write_tiny_png(source)

        real = _real_process_image()
        captured: Dict[str, Any] = {}

        def fake_process_image(file_path, **kwargs):
            _assert_kwargs_accepted(real, kwargs)
            captured.update({"file_path": file_path, **kwargs})
            return {
                "status": "Success",
                "content": "OCR TEXT",
                "title": "scan",
                "author": "Unknown",
                "keywords": [],
                "chunks": [{"text": "OCR TEXT", "metadata": {"chunk_num": 0}}],
                "analysis": "",
                "metadata": {},
                "error": None,
                "warnings": [],
            }

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_image",
            fake_process_image,
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {
                "ocr": True,
                "ocr_language": "de",
                "ocr_backend": "tesseract",
                "chunk_options": {"size": 500, "max_size": 500, "overlap": 50},
            },
        )

        assert captured["enable_ocr"] is True
        assert captured["ocr_language"] == "de"
        assert captured["ocr_backend"] == "tesseract"
        assert captured["extract_features"] is False
        assert captured["perform_analysis"] is False
        # (xhigh review round) chunk_options no longer travels into the
        # processor: the shared text-chunk tail owns image chunking now.
        # See ``test_image_chunking_has_one_authority``.
        assert captured["chunk_options"] is None
        assert payload["media_type"] == "image"
        assert payload["content"] == "OCR TEXT"

    def test_image_defaults_match_the_real_signature(
        self, tmp_path: Path, monkeypatch
    ):
        """The parse branch's fallbacks mirror ``process_image``'s own
        declared defaults, pinned against ``inspect.signature`` so a
        processor default change fails here instead of drifting."""
        source = tmp_path / "scan.png"
        _write_tiny_png(source)

        real = _real_process_image()
        sig = inspect.signature(real)
        captured: Dict[str, Any] = {}

        def fake_process_image(file_path, **kwargs):
            _assert_kwargs_accepted(real, kwargs)
            captured.update(kwargs)
            return {
                "status": "Success",
                "content": "x",
                "title": "scan",
                "author": "Unknown",
                "keywords": [],
                "chunks": [],
                "analysis": "",
                "metadata": {},
                "error": None,
                "warnings": [],
            }

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_image",
            fake_process_image,
        )

        parse_local_file_for_ingest(str(source), {})

        assert captured["enable_ocr"] == sig.parameters["enable_ocr"].default
        assert captured["ocr_language"] == sig.parameters["ocr_language"].default
        assert captured["ocr_backend"] == sig.parameters["ocr_backend"].default

    def test_image_end_to_end_real_png_persists(
        self, tmp_path: Path, monkeypatch, media_db: MediaDatabase
    ):
        """REAL ``process_image`` over a real 2x2 PNG (only the OCR boundary
        stubbed -- no backend is installed in this venv), persisted through
        the real ``persist_parsed_media`` into a real ``MediaDatabase``."""
        source = tmp_path / "receipt.png"
        _write_tiny_png(source)
        ocr_calls = _install_ocr_stub(monkeypatch, "TOTAL 12.50 EUR thank you")

        payload = parse_local_file_for_ingest(
            str(source),
            {
                "ocr": True,
                "ocr_language": "en",
                "ocr_backend": "auto",
                "chunk_options": {"size": 500, "max_size": 500, "overlap": 50},
            },
        )

        assert ocr_calls, "the OCR boundary was never reached"
        assert payload["media_type"] == "image"
        assert payload["content"] == "TOTAL 12.50 EUR thank you"
        # PIL metadata must survive: Pillow is installed here, and its
        # availability must not be hostage to the absent pillow_heif
        # (the coupled import guard this task decoupled).
        assert payload["metadata"]["width"] == 2
        assert payload["metadata"]["height"] == 2

        media_id, _uuid, _msg = persist_parsed_media(payload, media_db)
        assert media_id is not None

        cursor = media_db.execute_query(
            "SELECT type, content FROM Media WHERE id = ?", (media_id,)
        )
        row = cursor.fetchone()
        assert row["type"] == "image"
        assert row["content"] == "TOTAL 12.50 EUR thank you"
        assert _chunk_rows(media_db, media_id), "chunk ON stored no chunks"

    def test_image_chunk_off_stores_no_chunks(
        self, tmp_path: Path, monkeypatch, media_db: MediaDatabase
    ):
        source = tmp_path / "receipt.png"
        _write_tiny_png(source)
        _install_ocr_stub(monkeypatch, "some text")

        payload = parse_local_file_for_ingest(
            str(source), {"ocr": True, "chunk_options": None}
        )

        assert payload["chunks"] is None
        media_id, _uuid, _msg = persist_parsed_media(payload, media_db)
        assert _chunk_rows(media_db, media_id) == []

    def test_image_without_ocr_text_fails_honestly(
        self, tmp_path: Path, monkeypatch, media_db: MediaDatabase
    ):
        """No OCR text (no backend installed, or OCR off) must fail the job
        with a reason that names OCR -- never a 'done' row whose content is
        empty and silently unfindable in search/RAG."""
        source = tmp_path / "photo.png"
        _write_tiny_png(source)
        _install_ocr_stub(monkeypatch, None)  # OCR failed / no backend

        payload = parse_local_file_for_ingest(str(source), {"ocr": True})
        with pytest.raises(Exception, match="OCR"):
            persist_parsed_media(payload, media_db)

        cursor = media_db.execute_query("SELECT COUNT(*) AS n FROM Media", ())
        assert cursor.fetchone()["n"] == 0

    def test_image_ocr_off_also_fails_honestly(
        self, tmp_path: Path, monkeypatch, media_db: MediaDatabase
    ):
        source = tmp_path / "photo.png"
        _write_tiny_png(source)
        ocr_calls = _install_ocr_stub(monkeypatch, "never reached")

        payload = parse_local_file_for_ingest(str(source), {"ocr": False})

        assert not ocr_calls, "OCR ran despite the toggle being off"
        with pytest.raises(Exception, match="OCR"):
            persist_parsed_media(payload, media_db)

    def test_image_chunk_size_governs_chunk_count(self, tmp_path: Path, monkeypatch):
        """(task-3307 xhigh review round) OCR text must chunk per the form.

        The branch delegated to ``process_image``'s internal chunking,
        which only chunks for a TRUTHY ``chunk_options``. ``Chunk content``
        ON with untouched size/overlap arrives as ``{}`` (falsy), so the
        processor took its "no chunking options" fallback and returned ONE
        whole-text chunk; ``image`` was also absent from
        ``_TEXT_CHUNK_TYPES``, so the shared tail's repair never ran either.
        The image persisted as a single unchunked blob whatever the form
        said. Real chunker, no stub -- only the OCR boundary is stubbed.
        """
        source = tmp_path / "page.png"
        _write_tiny_png(source)
        _install_ocr_stub(monkeypatch, _MANY_SENTENCES)

        small = parse_local_file_for_ingest(
            str(source),
            {"ocr": True, "chunk_options": {"size": 40, "max_size": 40, "overlap": 10}},
        )
        large = parse_local_file_for_ingest(
            str(source),
            {
                "ocr": True,
                "chunk_options": {"size": 4000, "max_size": 4000, "overlap": 10},
            },
        )

        assert len(small["chunks"]) > 1, "OCR text was never chunked"
        assert len(small["chunks"]) > len(large["chunks"])

    def test_image_chunk_on_with_defaulted_empty_options_still_chunks(
        self, tmp_path: Path, monkeypatch
    ):
        """The exact shape the form produces for "chunk ON, nothing typed":
        an EMPTY options dict. The falsy-dict hole lived here -- the
        processor read it as "no chunking wanted" and returned one blob."""
        long_text = " ".join(
            f"Sentence number {i} carries a bit of body text for chunking."
            for i in range(200)
        )
        source = tmp_path / "page.png"
        _write_tiny_png(source)
        _install_ocr_stub(monkeypatch, long_text)

        payload = parse_local_file_for_ingest(
            str(source), {"ocr": True, "chunk_options": {}}
        )

        assert len(payload["chunks"]) > 1
        # The real chunker's metadata, not the processor's
        # ``{"chunk_num": 0}`` placeholder.
        assert "word_count" in payload["chunks"][0]

    def test_image_chunking_has_one_authority(self, tmp_path: Path, monkeypatch):
        """``process_image`` must never chunk on the ingest path.

        Two chunking layers is how the falsy-dict hole happened in the
        first place; the shared tail is the single authority, so the
        processor is called with ``chunk_options=None`` regardless of what
        the form asked for.
        """
        source = tmp_path / "page.png"
        _write_tiny_png(source)

        real = _real_process_image()
        captured: Dict[str, Any] = {}

        def fake_process_image(file_path, **kwargs):
            _assert_kwargs_accepted(real, kwargs)
            captured.update(kwargs)
            return {
                "status": "Success",
                "content": _MANY_SENTENCES,
                "title": "page",
                "author": "Unknown",
                "keywords": [],
                # The processor's convenience single whole-text "chunk",
                # returned even when it did no chunking at all.
                "chunks": [{"text": _MANY_SENTENCES, "metadata": {"chunk_num": 0}}],
                "analysis": "",
                "metadata": {},
                "error": None,
                "warnings": [],
            }

        monkeypatch.setattr(
            "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_image",
            fake_process_image,
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {"ocr": True, "chunk_options": {"size": 40, "max_size": 40, "overlap": 10}},
        )

        assert captured["chunk_options"] is None, (
            "the processor's own chunking must stay out of the ingest path"
        )
        assert len(payload["chunks"]) > 1, (
            "the processor's single fallback chunk was persisted as-is"
        )

    def test_image_analysis_dispatches_via_chat_tail(
        self, tmp_path: Path, monkeypatch
    ):
        """Analysis over the OCR text runs through the arc's chat_api_call
        tail (full [analysis_defaults] shape, keyless support) -- NOT
        ``process_image``'s own analyze() path, whose direct dispatch is
        the dead branch task-3301 documented."""
        source = tmp_path / "note.png"
        _write_tiny_png(source)
        _install_ocr_stub(monkeypatch, "OCR text worth analyzing.")

        real_chat = _real_chat_api_call()
        calls: list[Dict[str, Any]] = []

        def fake_chat_api_call(**kwargs):
            _assert_kwargs_accepted(real_chat, kwargs)
            calls.append(kwargs)
            return "IMAGE ANALYSIS."

        monkeypatch.setattr(
            "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
            fake_chat_api_call,
        )

        def exploding_analyze(*args, **kwargs):
            raise AssertionError(
                "image analysis went through Summarization analyze(); it "
                "must dispatch through chat_api_call"
            )

        monkeypatch.setattr(
            "tldw_chatbook.LLM_Calls.Summarization_General_Lib.analyze",
            exploding_analyze,
        )

        payload = parse_local_file_for_ingest(
            str(source),
            {
                "ocr": True,
                "perform_analysis": True,
                "api_name": "openai",
                "api_key": "sk-test-not-real",
            },
        )

        assert payload["analysis_content"] == "IMAGE ANALYSIS."
        assert calls, "no dispatch reached the chat_api_call boundary"
        assert "OCR text worth analyzing." in (
            calls[0]["messages_payload"][0]["content"]
        )
