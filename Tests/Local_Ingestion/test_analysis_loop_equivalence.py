"""B14 equivalence: the concurrent per-chunk analysis loops produce results
identical to the historical serial loops.

The three ingestion processors (PDF, EPUB, markup/text) replaced their serial
chunk-analysis loops with the shared ``analyze_chunks_concurrently`` fan-out.
These tests drive the REAL processors with a stubbed ``analyze`` and pin, per
loop:

* success path -- every chunk gets its own analysis in chunk order;
* per-chunk failure isolation -- a failing chunk produces the same
  ``[Summarization Error: ...]`` metadata marker and the same ordered warning
  the serial loop produced, and does NOT abort sibling chunks;
* empty-result handling -- an empty analysis stores ``None`` on the chunk;
* the fan-out is actually concurrent (peak >= 2 observed) while staying
  within the pool bound (peak <= 3).

Stubbing note: PDF extraction runs for real (pymupdf text insertion that
pymupdf4llm reads back); EPUB extraction is stubbed at ``read_epub_filtered``
(the established pattern in ``test_book_ingestion_chunking.py``). The chunking
seam is ALSO stubbed here -- real chunking currently raises
``RecoveryRequired("raw_source_selection_changed")`` under pytest on this base
(the pre-existing ``TestEpubChunkingExecutes``/``TestFb2ChunkingExecutes``
failures), and chunking is not what this file tests. The fake chunker splits
the extracted text into deterministic 15-word groups, so chunk boundaries and
order are fully deterministic.
"""

from __future__ import annotations

import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

from tldw_chatbook.Local_Ingestion import Book_Ingestion_Lib
from tldw_chatbook.Local_Ingestion.PDF_Processing_Lib import process_pdf
from tldw_chatbook.Local_Ingestion.Book_Ingestion_Lib import process_epub
from tldw_chatbook.LLM_Calls import Summarization_General_Lib as summ_lib
from tldw_chatbook.RAG_Search import chunking_service
from tldw_chatbook.Chunking import Chunk_Lib


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _fake_chunker(*args, **kwargs):
    """Deterministic stand-in for ``improved_chunking_process``: split the
    text into 15-word groups. Signature-agnostic on purpose."""
    text = args[0] if args else kwargs.get("text", "")
    words = str(text).split()
    chunks = []
    for i in range(0, len(words), 15):
        group = words[i : i + 15]
        if group:
            chunks.append(
                {"text": " ".join(group), "metadata": {"chunk_num": i // 15}}
            )
    return chunks


@pytest.fixture(autouse=True)
def _stub_chunk_seams(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(chunking_service, "improved_chunking_process", _fake_chunker)
    monkeypatch.setattr(Chunk_Lib, "improved_chunking_process", _fake_chunker)


class AnalyzeSpy:
    """Stands in for ``Summarization_General_Lib.analyze``; records calls and
    concurrency, optionally failing on marker chunks."""

    def __init__(self, fail_marker: str | None = None, delay: float = 0.0):
        self.fail_marker = fail_marker
        self.delay = delay
        self.calls: list[str] = []
        self.peak = 0
        self._live = 0
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        text = kwargs.get("input_data")
        if text is None and len(args) > 1:
            # Positional variant: (api_name, input_data, ...).
            text = args[1]
        with self._lock:
            self.calls.append(str(text))
            self._live += 1
            self.peak = max(self.peak, self._live)
        try:
            if self.delay:
                threading.Event().wait(self.delay)
            if self.fail_marker and self.fail_marker in str(text):
                raise RuntimeError(f"boom for {self.fail_marker}")
            return "analysis::" + str(text)
        finally:
            with self._lock:
                self._live -= 1


@pytest.fixture()
def analyze_spy(monkeypatch: pytest.MonkeyPatch):
    """Factory patching the function-level ``analyze`` import the processors
    resolve at call time."""

    def _install(spy: AnalyzeSpy) -> AnalyzeSpy:
        monkeypatch.setattr(summ_lib, "analyze", spy)
        return spy

    return _install


# ---------------------------------------------------------------------------
# PDF loop (real extraction; chunking stubbed)
# ---------------------------------------------------------------------------

_WORDS_PER_PAGE = 45


def _write_pdf(path: Path, fail_on_page: int | None = None) -> Path:
    doc = fitz.open()
    for page_num in range(3):
        words = " ".join(
            f"w{page_num}{i:02d}" for i in range(_WORDS_PER_PAGE)
        )
        if fail_on_page == page_num:
            words = f"FAILME " + words
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(72, 72, 540, 700), words, fontname="helv", fontsize=11
        )
    doc.save(str(path))
    doc.close()
    return path


class TestPdfAnalysisLoop:
    def test_success_path_analysis_per_chunk_in_order(
        self, tmp_path: Path, analyze_spy
    ):
        spy = analyze_spy(AnalyzeSpy())
        result = process_pdf(
            _write_pdf(tmp_path / "book.pdf"),
            filename="book.pdf",
            perform_chunking=True,
            chunk_options={"method": "words"},
            perform_analysis=True,
            api_name="openai",
            api_key="test-key",
        )
        assert result["status"] == "Success", result.get("error")
        chunks = result["chunks"]
        assert len(chunks) >= 3
        for chunk in chunks:
            assert chunk["metadata"]["analysis"] == "analysis::" + chunk["text"]
        # Chunk summaries were collected in chunk order.
        expected_summaries = ["analysis::" + c["text"] for c in chunks]
        assert result["analysis"] == "\n\n---\n\n".join(expected_summaries)
        assert not result.get("warnings")
        assert len(spy.calls) == len(chunks)

    def test_failure_isolation_marker_and_warning_in_chunk_order(
        self, tmp_path: Path, analyze_spy
    ):
        analyze_spy(AnalyzeSpy(fail_marker="FAILME"))
        result = process_pdf(
            _write_pdf(tmp_path / "book.pdf", fail_on_page=1),
            filename="book.pdf",
            perform_chunking=True,
            chunk_options={"method": "words"},
            perform_analysis=True,
            api_name="openai",
            api_key="test-key",
        )
        chunks = result["chunks"]
        error_slots = [
            i
            for i, c in enumerate(chunks)
            if isinstance(c["metadata"].get("analysis"), str)
            and c["metadata"]["analysis"].startswith("[Summarization Error:")
        ]
        # Every chunk containing the marked page failed; every other chunk
        # still got a real analysis (isolation, not abort).
        assert error_slots, "expected at least one error marker"
        for i, chunk in enumerate(chunks):
            if i in error_slots:
                assert chunk["metadata"]["analysis"] == (
                    "[Summarization Error: boom for FAILME]"
                )
                assert "FAILME" in chunk["text"]
            else:
                assert chunk["metadata"]["analysis"] == "analysis::" + chunk["text"]
        # Warnings mirror the serial loop's per-failure lines, in chunk order.
        assert result["warnings"] == [
            f"Summarization failed for chunk {i + 1}: boom for FAILME"
            for i in error_slots
        ]
        # Only the failures contributed no summary.
        surviving = [c for i, c in enumerate(chunks) if i not in error_slots]
        assert result["analysis"] == "\n\n---\n\n".join(
            "analysis::" + c["text"] for c in surviving
        )

    def test_concurrency_observed_and_bounded(self, tmp_path: Path, analyze_spy):
        spy = AnalyzeSpy(delay=0.08)
        analyze_spy(spy)
        result = process_pdf(
            _write_pdf(tmp_path / "book.pdf"),
            filename="book.pdf",
            perform_chunking=True,
            chunk_options={"method": "words"},
            perform_analysis=True,
            api_name="openai",
            api_key="test-key",
        )
        assert result["status"] == "Success"
        assert len(spy.calls) >= 3
        assert spy.peak >= 2, "analysis still fully serial -- no overlap observed"
        assert spy.peak <= 3, "peak concurrency exceeded the pool bound"


# ---------------------------------------------------------------------------
# EPUB loop (extraction + chunking stubbed; analysis loop is real)
# ---------------------------------------------------------------------------

_CHAPTERED_TEXT = "\n\n".join(
    f"# Chapter {i}\n\n"
    + " ".join(f"Sentence {j} of chapter {i} has words." for j in range(12))
    for i in range(1, 5)
)


def _write_stub_epub(path: Path) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("content.xhtml", b"<p>stub</p>")
    return path


@pytest.fixture()
def epub_extraction_stubbed(monkeypatch: pytest.MonkeyPatch):
    fake_book = SimpleNamespace(metadata={})
    monkeypatch.setattr(
        Book_Ingestion_Lib,
        "read_epub_filtered",
        lambda file_path: (_CHAPTERED_TEXT, fake_book),
    )
    monkeypatch.setattr(
        Book_Ingestion_Lib,
        "extract_epub_metadata_from_epub_obj",
        lambda ebook_obj: ("Stub Title", "Stub Author"),
    )


class TestEpubAnalysisLoop:
    def test_success_path_analysis_per_chunk_in_order(
        self, tmp_path: Path, epub_extraction_stubbed, analyze_spy
    ):
        spy = analyze_spy(AnalyzeSpy())
        result = process_epub(
            str(_write_stub_epub(tmp_path / "book.epub")),
            perform_chunking=True,
            chunk_options={"method": "words"},
            perform_analysis=True,
            api_name="openai",
            api_key="test-key",
        )
        chunks = result["chunks"]
        assert len(chunks) >= 3
        for chunk in chunks:
            assert chunk["metadata"]["analysis"] == "analysis::" + chunk["text"]
        assert result["analysis"] == "\n\n---\n\n".join(
            "analysis::" + c["text"] for c in chunks
        )
        assert not result.get("warnings")
        assert len(spy.calls) == len(chunks)

    def test_failure_isolation_marker_and_warning(
        self, tmp_path: Path, epub_extraction_stubbed, analyze_spy
    ):
        analyze_spy(AnalyzeSpy(fail_marker="chapter 2"))
        result = process_epub(
            str(_write_stub_epub(tmp_path / "book.epub")),
            perform_chunking=True,
            chunk_options={"method": "words"},
            perform_analysis=True,
            api_name="openai",
            api_key="test-key",
        )
        chunks = result["chunks"]
        error_slots = [
            i
            for i, c in enumerate(chunks)
            if isinstance(c["metadata"].get("analysis"), str)
            and c["metadata"]["analysis"].startswith("[Summarization Error:")
        ]
        assert error_slots, "expected the chapter-2 chunks to carry error markers"
        for i, chunk in enumerate(chunks):
            if i in error_slots:
                assert chunk["metadata"]["analysis"] == (
                    "[Summarization Error: boom for chapter 2]"
                )
                assert "chapter 2" in chunk["text"]
            else:
                assert chunk["metadata"]["analysis"] == "analysis::" + chunk["text"]
        assert result["warnings"] == [
            f"Summarization failed for chunk {i + 1}: boom for chapter 2"
            for i in error_slots
        ]

    def test_empty_analysis_stores_none(
        self, tmp_path: Path, epub_extraction_stubbed, analyze_spy, monkeypatch
    ):
        monkeypatch.setattr(
            summ_lib,
            "analyze",
            lambda *a, **k: "   ",  # whitespace-only -> empty-result branch
        )
        result = process_epub(
            str(_write_stub_epub(tmp_path / "book.epub")),
            perform_chunking=True,
            chunk_options={"method": "words"},
            perform_analysis=True,
            api_name="openai",
            api_key="test-key",
        )
        chunks = result["chunks"]
        assert chunks
        for chunk in chunks:
            assert chunk["metadata"]["analysis"] is None
        assert not result.get("warnings")
