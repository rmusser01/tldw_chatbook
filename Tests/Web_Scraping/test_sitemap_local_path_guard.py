"""The local-sitemap entry point validates its path before parsing it.

Qodo review on PR #2806: `scrape_from_filtered_sitemap` handed the
caller-supplied `sitemap_file` straight to `_safe_parse`, which opens it.
Hardening the PARSER says nothing about the PATH -- so the only public
function in this module that reads a local file reached the filesystem
with no traversal or NUL check at all, while every other file path in the
repo goes through `Utils/path_validation.py`.

Both tests below are observable, not cosmetic: before the fix the
traversal string really did resolve and parse (a successful scrape out of
the intended directory), and the NUL escaped as an uncaught `ValueError`
from `open()` -- neither `xET.ParseError` nor `DefusedXmlException`, so
the documented "empty list on input it cannot use" contract was broken.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as m

SITEMAP = """<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.invalid/a</loc></url>
</urlset>
"""


@pytest.fixture
def sitemap(tmp_path, monkeypatch):
    """A readable benign sitemap, with scraping stubbed to a sync echo."""
    path = tmp_path / "sitemap.xml"
    path.write_text(SITEMAP, encoding="utf-8")
    monkeypatch.setattr(m, "scrape_article", lambda url, *a, **k: {"url": url})
    return path


def test_an_ordinary_path_still_scrapes(sitemap):
    """Anti-vacuity: the guard fails closed, so prove it passes the real shape."""
    assert m.scrape_from_filtered_sitemap(str(sitemap), lambda url: True) == [
        {"url": "https://example.invalid/a"}
    ]


def test_a_traversal_path_never_reaches_the_parser(sitemap):
    """`<tmp>/a/b/../../sitemap.xml` really does open the file -- and is refused.

    `a/b` are created so the traversal RESOLVES: POSIX `open()` walks every
    segment, so without them this would fail on a missing directory rather
    than on the guard, and prove nothing about traversal.
    """
    (sitemap.parent / "a" / "b").mkdir(parents=True)
    traversal = str(sitemap.parent / "a" / "b" / ".." / ".." / "sitemap.xml")
    assert Path(traversal).read_text(encoding="utf-8") == SITEMAP, "premise"

    assert m.scrape_from_filtered_sitemap(traversal, lambda url: True) == []


def test_an_embedded_nul_is_refused_instead_of_crashing(sitemap):
    assert m.scrape_from_filtered_sitemap(f"{sitemap}\x00.txt", lambda url: True) == []


def test_a_missing_file_returns_empty_rather_than_raising(tmp_path):
    assert m.scrape_from_filtered_sitemap(
        str(tmp_path / "absent.xml"), lambda url: True
    ) == []
