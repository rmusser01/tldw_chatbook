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

import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Local_Ingestion import Book_Ingestion_Lib
from tldw_chatbook.Local_Ingestion.Book_Ingestion_Lib import (
    epub_to_markdown,
    process_epub,
    process_fb2,
    read_epub,
    read_epub_filtered,
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


def _write_stub_epub(path: Path, *members: tuple[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members or (("content.xhtml", b"<p>stub</p>"),):
            archive.writestr(name, content)
    return path


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
        source = _write_stub_epub(tmp_path / "book.epub")

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
        source = _write_stub_epub(tmp_path / "book.epub")

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


@pytest.mark.parametrize(
    ("limit_name", "limit", "members"),
    [
        ("_MAX_EPUB_ARCHIVE_MEMBERS", 1, (("a.xhtml", b"a"), ("b.xhtml", b"b"))),
        ("_MAX_EPUB_MEMBER_BYTES", 8, (("chapter.xhtml", b"123456789"),)),
        (
            "_MAX_EPUB_TOTAL_BYTES",
            8,
            (("a.xhtml", b"12345"), ("b.xhtml", b"67890")),
        ),
        (
            "_MAX_EPUB_MARKUP_BYTES",
            8,
            (("a.xhtml", b"12345"), ("b.xhtml", b"67890")),
        ),
        ("_MAX_EPUB_COMPRESSION_RATIO", 1, (("chapter.xhtml", b"x" * 1_000),)),
    ],
    ids=(
        "member-count",
        "member-size",
        "total-size",
        "markup-size",
        "compression-ratio",
    ),
)
def test_process_epub_rejects_archive_limits_before_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit: int,
    members: tuple[tuple[str, bytes], ...],
) -> None:
    source = _write_stub_epub(tmp_path / "oversized.epub", *members)
    extractor_calls: list[str] = []
    fake_book = SimpleNamespace(metadata={})

    def extraction_probe(file_path: str):
        extractor_calls.append(file_path)
        return _CHAPTERED_TEXT, fake_book

    monkeypatch.setattr(Book_Ingestion_Lib, limit_name, limit, raising=False)
    monkeypatch.setattr(Book_Ingestion_Lib, "read_epub_filtered", extraction_probe)
    monkeypatch.setattr(
        Book_Ingestion_Lib,
        "extract_epub_metadata_from_epub_obj",
        lambda ebook_obj: ("Stub Title", "Stub Author"),
    )

    result = process_epub(str(source), perform_chunking=False)

    assert result["status"] == "Error"
    assert result["error"] == "EPUB archive exceeds safety limits."
    assert extractor_calls == []


def test_process_epub_counts_manifest_declared_xhtml_with_custom_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_xml = b"""<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OPS/package.opf"/></rootfiles>
</container>"""
    package_opf = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf">
  <manifest><item id="chapter" href="chapter.bin" media-type="application/xhtml+xml"/></manifest>
</package>"""
    chapter = b"<p>" + (b"x" * 100) + b"</p>"
    source = _write_stub_epub(
        tmp_path / "custom-extension.epub",
        ("META-INF/container.xml", container_xml),
        ("OPS/package.opf", package_opf),
        ("OPS/chapter.bin", chapter),
    )
    extractor_calls: list[str] = []
    limit = len(container_xml) + len(package_opf) + 10

    monkeypatch.setattr(Book_Ingestion_Lib, "_MAX_EPUB_MARKUP_BYTES", limit)
    monkeypatch.setattr(
        Book_Ingestion_Lib,
        "read_epub_filtered",
        lambda file_path: extractor_calls.append(file_path),
    )

    result = process_epub(str(source), perform_chunking=False)

    assert result["status"] == "Error"
    assert result["error"] == "EPUB archive exceeds safety limits."
    assert extractor_calls == []


def test_process_epub_fails_closed_when_manifest_classification_is_unsafe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_xml = b"""<?xml version="1.0"?>
<!DOCTYPE container [<!ENTITY package "OPS/package.opf">]>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="&package;"/></rootfiles>
</container>"""
    source = _write_stub_epub(
        tmp_path / "unsafe-manifest.epub",
        ("META-INF/container.xml", container_xml),
        ("OPS/package.opf", b"<package/ >"),
        ("OPS/chapter.bin", b"<p>custom extension</p>"),
    )
    extractor_calls: list[str] = []
    monkeypatch.setattr(
        Book_Ingestion_Lib,
        "read_epub_filtered",
        lambda file_path: extractor_calls.append(file_path),
    )

    result = process_epub(str(source), perform_chunking=False)

    assert result["status"] == "Error"
    assert result["error"] == "EPUB archive exceeds safety limits."
    assert extractor_calls == []


def test_process_epub_fails_closed_for_unresolvable_package_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_xml = b"""<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="%2e%2e/OPS/package.opf"/></rootfiles>
</container>"""
    source = _write_stub_epub(
        tmp_path / "unsafe-package-path.epub",
        ("META-INF/container.xml", container_xml),
        ("OPS/package.opf", b"<package/ >"),
        ("OPS/chapter.bin", b"<p>custom extension</p>"),
    )
    extractor_calls: list[str] = []
    monkeypatch.setattr(
        Book_Ingestion_Lib,
        "read_epub_filtered",
        lambda file_path: extractor_calls.append(file_path),
    )

    result = process_epub(str(source), perform_chunking=False)

    assert result["status"] == "Error"
    assert result["error"] == "EPUB archive exceeds safety limits."
    assert extractor_calls == []


@pytest.mark.parametrize(
    "reader",
    (epub_to_markdown, read_epub_filtered, read_epub),
    ids=("markdown", "filtered", "basic"),
)
def test_direct_epub_readers_apply_archive_guard_before_ebooklib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader,
) -> None:
    if not Book_Ingestion_Lib.EBOOKLIB_AVAILABLE:
        pytest.skip("ebooklib is not installed")
    source = _write_stub_epub(
        tmp_path / "oversized-direct-reader.epub",
        ("chapter.xhtml", b"x" * 100),
    )
    ebooklib_calls: list[str] = []

    monkeypatch.setattr(Book_Ingestion_Lib, "_MAX_EPUB_MARKUP_BYTES", 8)
    monkeypatch.setattr(
        Book_Ingestion_Lib.epub,
        "read_epub",
        lambda file_path: ebooklib_calls.append(file_path),
    )

    with pytest.raises(ValueError, match="EPUB archive exceeds safety limits"):
        reader(str(source))
    assert ebooklib_calls == []
