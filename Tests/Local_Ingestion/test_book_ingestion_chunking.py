"""(task-3301/3303 xhigh review round 2, F2) E-book chunking must EXECUTE.

The three e-book processors (``process_epub``/``process_mobi``/``process_fb2``)
called ``improved_chunking_process(text=..., chunk_options_dict=...)`` while
importing it from ``RAG_Search.chunking_service``, whose signature is
``(text, options)`` -- a ``TypeError`` on EVERY call, degraded by each site's
broad ``except`` into one full-text chunk, so the ebook panel's chunk-method
select was dead on arrival. The call (and its surrounding comments -- the
``llm_call_function_for_chunker`` remark, "handles Chunker instantiation and
metadata enrichment") was written for ``Chunking.Chunk_Lib``'s
``improved_chunking_process``, the variant that accepts ``chunk_options_dict``
AND dispatches ``ebook_chapters`` -- the method the panel's default "chapters"
choice maps to. The ``chunking_service`` wrapper rejects ``ebook_chapters``
outright (``InvalidChunkingMethodError``), so merely renaming the kwarg would
have left the panel's DEFAULT choice dead; the fix points the import at the
callee the call was written for, and replaces the mobi/fb2 ``"recursive"``
method default (dispatched by NO chunker in this repo) with the pre-branch
``"sentences"``.

These tests run the REAL chunker end to end -- no stubs on the chunking seam
(the kwargs-arrival-vs-governance lesson in
``backlog/docs/lessons-testing-evidence.md``). Only optional-dependency
EXTRACTION seams are stubbed: ebooklib is absent in this venv, so the epub
tests stub ``read_epub_filtered``; ``process_fb2`` needs no stub at all (FB2
is XML, parsed via the stdlib fallback).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Local_Ingestion import Book_Ingestion_Lib
from tldw_chatbook.Local_Ingestion.Book_Ingestion_Lib import (
    process_epub,
    process_fb2,
)


@pytest.fixture(autouse=True)
def _offline_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real ebook chunking on the optional tokenizer fallback."""
    from tldw_chatbook.Chunking import token_chunker

    monkeypatch.setattr(token_chunker, "get_safe_import", lambda _name: None)


# ---------------------------------------------------------------------------
# FB2: fully real -- extraction (stdlib XML) AND chunking.
# ---------------------------------------------------------------------------

_FB2_NS = "http://www.gribuser.ru/xml/fictionbook/2.0"


def _write_fb2(path: Path, paragraphs: int = 30) -> Path:
    body = "\n".join(
        f"        <p>Sentence number {i} carries some body text.</p>"
        for i in range(paragraphs)
    )
    path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="{_FB2_NS}">
  <description>
    <title-info>
      <book-title>Chunking Test Book</book-title>
    </title-info>
  </description>
  <body>
    <section>
      <title><p>Chapter One</p></title>
{body}
    </section>
  </body>
</FictionBook>
""",
        encoding="utf-8",
    )
    return path


def _assert_no_chunking_failure(result: dict) -> None:
    failures = [w for w in result.get("warnings") or [] if "Chunking failed" in w]
    assert not failures, f"chunking degraded to the fallback: {failures}"


class TestFb2ChunkingExecutes:
    def test_explicit_sentences_options_chunk_for_real(self, tmp_path: Path):
        source = _write_fb2(tmp_path / "book.fb2")

        result = process_fb2(
            str(source),
            perform_chunking=True,
            chunk_options={"method": "sentences", "max_size": 3, "overlap": 0},
        )

        assert result["status"] == "Success"
        _assert_no_chunking_failure(result)
        chunks = result["chunks"]
        assert chunks and len(chunks) > 1, (
            "explicit sentence chunking produced a single chunk -- the "
            "improved_chunking_process call did not execute"
        )
        for chunk in chunks:
            assert chunk["text"].strip()
            assert chunk["metadata"]["chunk_method"] == "sentences"

    def test_default_options_use_a_real_method_not_recursive(self, tmp_path: Path):
        """No chunk_options at all: the processor's own default must be a
        method some chunker actually dispatches ('recursive' is not one)."""
        source = _write_fb2(tmp_path / "default.fb2")

        result = process_fb2(str(source), perform_chunking=True, chunk_options=None)

        assert result["status"] == "Success"
        _assert_no_chunking_failure(result)
        chunks = result["chunks"]
        assert chunks, "default chunking produced no chunks"
        assert chunks[0]["metadata"]["chunk_method"] == "sentences"


# ---------------------------------------------------------------------------
# EPUB: extraction stubbed (ebooklib absent in this venv), chunking real.
# ---------------------------------------------------------------------------

_CHAPTERED_TEXT = "\n\n".join(
    f"# Chapter {i}\n\n"
    + " ".join(f"Sentence {j} of chapter {i} has words." for j in range(12))
    for i in range(1, 5)
)


@pytest.fixture()
def epub_extraction_stubbed(monkeypatch: pytest.MonkeyPatch):
    """Stub ONLY the ebooklib-backed extraction seams of ``process_epub``.

    The chunking seam stays fully real. ``read_epub_filtered`` is the
    default ``extraction_method='filtered'`` dispatch target.
    """
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


class TestEpubChunkingExecutes:
    def test_default_chapters_method_chunks_by_chapter(
        self, tmp_path: Path, epub_extraction_stubbed
    ):
        source = tmp_path / "book.epub"
        source.write_bytes(b"PK\x03\x04 fake epub bytes")

        result = process_epub(str(source), perform_chunking=True, chunk_options=None)

        _assert_no_chunking_failure(result)
        chunks = result["chunks"]
        assert chunks and len(chunks) > 1, (
            "the ebook panel's default chapters method produced a single "
            "chunk -- ebook_chapters never executed"
        )
        assert chunks[0]["metadata"]["chunk_method"] == "ebook_chapters"

    def test_words_method_tracks_word_budget(
        self, tmp_path: Path, epub_extraction_stubbed
    ):
        """Governance, not kwargs-arrival: the panel's 'words' choice must
        change the OUTPUT (many small chunks under the word budget)."""
        source = tmp_path / "book.epub"
        source.write_bytes(b"PK\x03\x04 fake epub bytes")

        result = process_epub(
            str(source),
            perform_chunking=True,
            chunk_options={"method": "words", "max_size": 25, "overlap": 0},
        )

        _assert_no_chunking_failure(result)
        chunks = result["chunks"]
        assert chunks and len(chunks) > 3
        for chunk in chunks:
            assert len(chunk["text"].split()) <= 25
            assert chunk["metadata"]["chunk_method"] == "words"
