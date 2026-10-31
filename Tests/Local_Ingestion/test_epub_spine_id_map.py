"""B27 (Book_Ingestion_Lib item): EPUB spine id lookup uses one dict map.

``read_epub_filtered`` used to call ebooklib's ``book.get_item_with_id`` once
per spine entry; that helper is a LINEAR scan over ``book.get_items()``, so a
book with S spine entries and I items paid O(S x I) identity checks per
extraction. The fix builds an ``{item.id: item}`` map ONCE before the spine
loop (first item wins, matching ``get_item_with_id``'s first-match contract).

Equivalence is pinned here over a 2-book fixture: a mixed-content book
(front matter skipped, TOC kept, non-document items skipped) and a
duplicate-id book pinning first-match-wins. Counted evidence: the stub book's
``get_item_with_id`` is called len(spine) times by the old code and 0 times
by the fixed code, while the extracted text is byte-identical to the
reference walk.
"""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Local_Ingestion import Book_Ingestion_Lib


try:
    import ebooklib

    _ITEM_DOCUMENT = ebooklib.ITEM_DOCUMENT
    _ITEM_IMAGE = ebooklib.ITEM_IMAGE
except ImportError:  # pragma: no cover - ebooklib is an optional dep
    pytest.skip("ebooklib not available", allow_module_level=True)


class StubItem:
    def __init__(self, item_id, file_name, html, item_type=_ITEM_DOCUMENT):
        self.id = item_id
        self.file_name = file_name
        self._html = html
        self._type = item_type

    def get_type(self):
        return self._type

    def get_content(self):
        return self._html.encode("utf-8")


class StubBook:
    """Duck-typed ``EpubBook``: real ``get_item_with_id`` semantics (first
    match wins, None on miss) plus a call counter for the evidence assert."""

    def __init__(self, items, spine):
        self._items = items
        self.spine = spine
        self.get_item_with_id_calls = 0

    def get_items(self):
        return list(self._items)

    def get_item_with_id(self, uid):
        self.get_item_with_id_calls += 1
        for item in self._items:
            if item.id == uid:
                return item
        return None


def _reference_spine_text(book) -> str:
    """The OLD implementation's walk (verbatim semantics): one
    ``get_item_with_id`` call per spine entry."""
    skip_front_matter = {
        "cover",
        "titlepage",
        "copy",
        "copyright",
        "colophon",
        "upgrade",
        "notice",
        "legal",
        "license",
    }
    import re

    from bs4 import BeautifulSoup

    segments = []
    for itemref in book.spine:
        item = book.get_item_with_id(itemref[0])
        if item.get_type() != _ITEM_DOCUMENT:
            continue
        filename_lower = item.file_name.lower()
        if any(name in filename_lower for name in skip_front_matter):
            continue
        content = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(content, "html.parser")
        text_chunks = []
        for elem in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol"]):
            text = elem.get_text().strip()
            if not text:
                continue
            if elem.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                level = int(elem.name[1])
                text_chunks.append(("#" * level) + " " + text)
            elif elem.name == "p":
                text_chunks.append(text)
            elif elem.name in ["ul", "ol"]:
                bullet = "-" if elem.name == "ul" else "1."
                for li in elem.find_all("li"):
                    li_text = li.get_text().strip()
                    if li_text:
                        text_chunks.append(f"{bullet} {li_text}")
        item_text = "\n\n".join(text_chunks)
        if item_text.strip():
            segments.append(item_text)
    full_text = "\n\n".join(segments)
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n\s*\n+", "\n\n", full_text)
    return full_text


@pytest.fixture()
def stubbed_book_opener(monkeypatch):
    def _install(book: StubBook) -> StubBook:
        monkeypatch.setattr(
            Book_Ingestion_Lib, "_read_epub_checked", lambda path: book
        )
        return book

    return _install


def test_book_one_mixed_content_output_unchanged(tmp_path, stubbed_book_opener):
    book = StubBook(
        items=[
            StubItem("img1", "images/cover.jpg", b"", item_type=_ITEM_IMAGE),
            StubItem("cover", "cover.xhtml", "<html><p>COVER PAGE</p></html>"),
            StubItem("titlepage", "titlepage.xhtml", "<html><p>TITLE PAGE</p></html>"),
            StubItem("copyright", "copyright.xhtml", "<html><p>ISBN 1234</p></html>"),
            StubItem("toc", "toc.xhtml", "<html><p>Table of Contents</p></html>"),
            StubItem(
                "ch1",
                "chapter1.xhtml",
                "<html><h1>Chapter One</h1><p>First paragraph.</p>"
                "<ul><li>alpha</li><li>beta</li></ul></html>",
            ),
            StubItem(
                "ch2",
                "chapter2.xhtml",
                "<html><h2>Chapter Two</h2><p>Second paragraph.</p></html>",
            ),
        ],
        spine=[
            ("img1", {}),
            ("cover", {}),
            ("titlepage", {}),
            ("copyright", {}),
            ("toc", {}),
            ("ch1", {}),
            ("ch2", {}),
        ],
    )
    stubbed_book_opener(book)

    text, returned_book = Book_Ingestion_Lib.read_epub_filtered(
        str(tmp_path / "whatever.epub")
    )
    # Snapshot BEFORE the reference walk below (it uses the same counter).
    production_get_item_calls = book.get_item_with_id_calls

    assert text == _reference_spine_text(book)
    assert returned_book is book
    # Evidence: the old code called the linear-scan helper once per spine
    # entry (7 here); the dict map replaces every one of those scans.
    assert production_get_item_calls == 0, (
        "spine loop still uses per-entry linear scans"
    )
    assert "# Chapter One" in text
    assert "COVER PAGE" not in text  # front matter still skipped
    assert "Table of Contents" in text  # toc still kept
    assert "- alpha" in text and "- beta" in text


def test_book_two_duplicate_ids_first_match_wins(tmp_path, stubbed_book_opener):
    """``get_item_with_id`` returns the FIRST item with an id; the map must
    preserve exactly that (setdefault, not last-write-wins)."""
    book = StubBook(
        items=[
            StubItem("dup", "a.xhtml", "<html><p>FIRST WINS</p></html>"),
            StubItem("dup", "b.xhtml", "<html><p>SECOND LOSES</p></html>"),
            StubItem("ch", "chapter.xhtml", "<html><p>Chapter text.</p></html>"),
        ],
        spine=[("dup", {}), ("ch", {})],
    )
    stubbed_book_opener(book)

    text, _ = Book_Ingestion_Lib.read_epub_filtered(str(tmp_path / "dup.epub"))
    production_get_item_calls = book.get_item_with_id_calls

    assert text == _reference_spine_text(book)
    assert "FIRST WINS" in text
    assert "SECOND LOSES" not in text
    assert production_get_item_calls == 0
