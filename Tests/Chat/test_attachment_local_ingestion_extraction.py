"""Task-19576: PDF/Word/ebook Console attachments used to return a
placeholder pointing at a retired "Media Ingestion tab" instead of
extracting real content -- and the plaintext-over-100KB handler pointed at
the same retired destination (5 sites total in `file_handlers.py`, not the
3 an earlier review summary named). All five now either extract real text
via `parse_local_file_for_ingest` (the same extractor Library ▸ Import
uses -- PDF/document/ebook) or, where the design intent is a deliberate
size-based deferral rather than a missing-capability stub (plaintext over
100KB), point at "Library" -- a real, live route -- instead of the retired
name.

Born-red: at base (before the fix), the PDF test below fails because the
handler returns the literal placeholder string ("[PDF File: ...]\\nTo
process this PDF file, please use the Media Ingestion tab.") instead of
extracted text; it passes once the handler routes through the real
extractor.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from tldw_chatbook.Local_Ingestion import local_file_ingestion
from tldw_chatbook.Utils.file_handlers import (
    DocumentFileHandler,
    EbookFileHandler,
    PDFFileHandler,
    PlaintextDatabaseHandler,
)

# The exact retired-destination string the old placeholders used. Every
# assertion below checks actual returned (user-visible) copy, not source
# comments -- this module's own docstrings/comments legitimately mention
# the retired name while explaining the fix, so a blanket source-file grep
# would false-positive on those.
_RETIRED_DESTINATION = "Media Ingestion tab"


def _make_processor_result(content: str, **overrides: Any) -> Dict[str, Any]:
    """Mirror the result shape `process_pdf`/`process_document`/
    `process_ebook` return (see `Tests/Local_Ingestion/
    test_local_file_ingestion.py`'s `_make_pdf_result`/`_make_ebook_result`
    for the same shape)."""
    result = {
        "status": "Success",
        "content": content,
        "title": "Extracted title",
        "author": "Extracted author",
        "keywords": [],
        "chunks": [],
        "analysis": "",
        "metadata": {},
        "error": None,
        "warnings": [],
    }
    result.update(overrides)
    return result


@pytest.mark.asyncio
async def test_pdf_attachment_extracts_real_text_not_placeholder(tmp_path):
    """Born-red: builds a REAL, minimal PDF with pymupdf/fitz (already a
    project dependency via PDF_Processing_Lib) and drives it through the
    real, unmocked extraction chain -- proving the handler returns
    genuinely extracted text, not a stand-in.
    """
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "report.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Task-19576 real extracted PDF content.")
    document.save(str(pdf_path))
    document.close()

    processed = await PDFFileHandler().process(pdf_path)

    assert "Task-19576 real extracted PDF content." in processed.content
    assert _RETIRED_DESTINATION not in processed.content


@pytest.mark.asyncio
async def test_document_attachment_routes_through_local_ingestion_extractor(
    tmp_path, monkeypatch
):
    """Document extraction routes through `parse_local_file_for_ingest` ->
    `process_document`, monkeypatched here (the idiom
    `test_local_file_ingestion.py` uses) so this stays fast/deterministic
    rather than depending on python-docx internals."""
    docx_path = tmp_path / "report.docx"
    # detect_file_type is purely extension-based; the bytes never need to
    # be a real docx since process_document is mocked below.
    docx_path.write_bytes(b"PK\x03\x04" + b"\x00" * 32)

    monkeypatch.setattr(
        local_file_ingestion,
        "process_document",
        lambda **kwargs: _make_processor_result("Real document body text."),
    )

    processed = await DocumentFileHandler().process(docx_path)

    assert "Real document body text." in processed.content
    assert _RETIRED_DESTINATION not in processed.content


@pytest.mark.asyncio
async def test_ebook_attachment_routes_through_local_ingestion_extractor(
    tmp_path, monkeypatch
):
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"PK\x03\x04" + b"\x00" * 32)

    monkeypatch.setattr(
        local_file_ingestion,
        "process_ebook",
        lambda **kwargs: _make_processor_result("Real chapter one text."),
    )

    processed = await EbookFileHandler().process(epub_path)

    assert "Real chapter one text." in processed.content
    assert _RETIRED_DESTINATION not in processed.content


@pytest.mark.asyncio
async def test_pdf_extraction_failure_is_honest_not_a_retired_placeholder(
    tmp_path, monkeypatch
):
    """A genuine extraction failure must not fall back to the old
    placeholder or name any retired destination -- it should say what
    actually happened."""
    pdf_path = tmp_path / "corrupt.pdf"
    pdf_path.write_bytes(b"not a real pdf")

    def _boom(**kwargs):
        raise RuntimeError("simulated extraction failure")

    monkeypatch.setattr(local_file_ingestion, "process_pdf", _boom)

    processed = await PDFFileHandler().process(pdf_path)

    assert _RETIRED_DESTINATION not in processed.content
    assert "simulated extraction failure" in processed.content


@pytest.mark.asyncio
async def test_plaintext_over_100kb_no_longer_names_the_retired_tab(tmp_path):
    """`PlaintextDatabaseHandler` intercepts text files over 100KB and
    defers to Library's RAG-searchable ingestion rather than inlining the
    whole file -- a deliberate cost/context-window choice, not a
    missing-capability stub. Only the (retired) destination name changes.
    """
    big_text = tmp_path / "notes.txt"
    big_text.write_text("x" * (150 * 1024))

    handler = PlaintextDatabaseHandler()
    assert handler.can_handle(big_text) is True

    processed = await handler.process(big_text)

    assert _RETIRED_DESTINATION not in processed.content
    assert "Library" in processed.content


@pytest.mark.asyncio
async def test_plaintext_over_10mb_no_longer_names_the_retired_tab(tmp_path):
    """The hard 10MB cap branch of the same handler (no content at all,
    just the too-large notice)."""
    huge_text = tmp_path / "huge.txt"
    huge_text.write_bytes(b"x" * (11 * 1024 * 1024))

    processed = await PlaintextDatabaseHandler().process(huge_text)

    assert _RETIRED_DESTINATION not in processed.content
    assert "Library" in processed.content


@pytest.mark.asyncio
async def test_no_handler_output_names_the_retired_media_ingestion_tab(
    tmp_path, monkeypatch
):
    """Census (task-19576 AC): drive every one of the five sites the task
    identified and grep the actual returned (user-visible) copy -- not
    source comments -- for the retired destination name."""
    monkeypatch.setattr(
        local_file_ingestion,
        "process_document",
        lambda **kwargs: _make_processor_result("doc body"),
    )
    monkeypatch.setattr(
        local_file_ingestion,
        "process_ebook",
        lambda **kwargs: _make_processor_result("ebook body"),
    )

    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "x.pdf"
    document = fitz.open()
    document.new_page()  # deliberately blank: exercises the "no
    # extractable text" fallback copy too
    document.save(str(pdf_path))
    document.close()

    docx_path = tmp_path / "x.docx"
    docx_path.write_bytes(b"PK\x03\x04")
    epub_path = tmp_path / "x.epub"
    epub_path.write_bytes(b"PK\x03\x04")
    big_text = tmp_path / "big.txt"
    big_text.write_text("y" * (150 * 1024))
    huge_text = tmp_path / "huge.txt"
    huge_text.write_bytes(b"y" * (11 * 1024 * 1024))

    outputs = {
        "pdf": (await PDFFileHandler().process(pdf_path)).content,
        "document": (await DocumentFileHandler().process(docx_path)).content,
        "ebook": (await EbookFileHandler().process(epub_path)).content,
        "plaintext_over_100kb": (
            await PlaintextDatabaseHandler().process(big_text)
        ).content,
        "plaintext_over_10mb": (
            await PlaintextDatabaseHandler().process(huge_text)
        ).content,
    }

    for site, output in outputs.items():
        assert _RETIRED_DESTINATION not in output, (
            f"{site} handler output still names the retired destination: "
            f"{output!r}"
        )
