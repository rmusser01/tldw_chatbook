"""TASK-32894: the four remaining tier-2 XML parsers refuse entity expansion.

`Tests/Subscriptions/test_watchlist_opml_entity_expansion.py` keeps the
repo-wide census and the register of deliberate exceptions; it is AST-based,
so it proves a module *consults* defusedxml, not that the parse call
production takes actually goes through it. These four are the ones this task
hardened, and this is the behavioural half: the same bounded billion-laughs
payload that census file uses, handed to the exact symbol each module parses
with.

Bounded on purpose (6 levels, ~10^6 expansion) and asserted on the REFUSAL,
never on a timing -- a test that measures "this took too long" is a flake
generator and proves nothing about the fix.

`Local_Ingestion/XML_Ingestion.py` was hardened here originally, but PR #2803
then DELETED the module as dead code -- it was never wired to anything (see
`Chunking/auto_selection.py`, which calls it "retired", and
`Tests/Library/test_ingest_capabilities.py`, which calls it "never-wired").
Its three behavioural tests went with it: they raised
`ModuleNotFoundError` once that deletion reached this branch, and a test that
can only fail is worse than no test.
"""

from __future__ import annotations

import pytest

from Tests.Subscriptions.test_watchlist_opml_entity_expansion import (
    BENIGN_OPML,
    BILLION_LAUGHS_OPML,
)


@pytest.fixture
def no_config_reads(monkeypatch):
    """Neutralize the config reads these modules do at import/egress time.

    ADR-126's recovery gate fires on `load_cli_config_and_ensure_existence`
    in a clean worktree; none of it is the XML behaviour under test.
    """
    from tldw_chatbook import config as config_module
    from tldw_chatbook.Utils import egress

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )
    monkeypatch.setattr(
        egress,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )


def _entities_forbidden():
    from defusedxml.common import EntitiesForbidden

    return EntitiesForbidden


def test_eval_runner_rejects_entity_expansion_in_model_output():
    """The sharpest of the four: the payload is prompt-injection reachable."""
    from tldw_chatbook.Evals.eval_runner import BaseEvalRunner

    assert BaseEvalRunner._is_valid_xml(None, BENIGN_OPML) is True
    assert BaseEvalRunner._is_valid_xml(None, BILLION_LAUGHS_OPML) is False


def test_article_extractor_sitemap_parsers_refuse_the_payload(tmp_path):
    from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as AEL

    path = tmp_path / "sitemap.xml"
    path.write_text(BILLION_LAUGHS_OPML, encoding="utf-8")
    with pytest.raises(_entities_forbidden()):
        AEL._safe_parse(str(path))
    with pytest.raises(_entities_forbidden()):
        AEL._safe_fromstring(BILLION_LAUGHS_OPML)
    # Document BUILDING deliberately stays on stdlib ElementTree.
    assert AEL.xET.Element("urlset") is not None


def test_fetched_sitemap_refusal_returns_empty_rather_than_raising(
    tmp_path, monkeypatch, no_config_reads
):
    """The refusal is a ValueError, NOT an `ET.ParseError`: handler widened.

    `scrape_from_sitemap` caught only egress and `requests` errors, so an
    `EntitiesForbidden` would have escaped as an unhandled crash instead of
    the documented empty result.
    """
    from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as AEL

    class _Response:
        content = BILLION_LAUGHS_OPML.encode()

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        AEL, "guarded_fetch_requests", lambda url, **kwargs: _Response()
    )
    assert AEL.scrape_from_sitemap("https://example.invalid/sitemap.xml") == []


def test_file_sitemap_refusal_returns_empty_rather_than_raising(tmp_path):
    """Same widening on the file-based sibling."""
    from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as AEL

    path = tmp_path / "sitemap.xml"
    path.write_text(BILLION_LAUGHS_OPML, encoding="utf-8")
    assert AEL.scrape_from_filtered_sitemap(str(path), lambda url: True) == []


def test_crawler_sitemap_parser_refuses_the_payload():
    from tldw_chatbook.Web_Scraping.Article_Scraper import crawler

    with pytest.raises(_entities_forbidden()):
        crawler.ET.fromstring(BILLION_LAUGHS_OPML)


def test_stdlib_would_have_expanded_the_same_payload():
    """The defect, demonstrated on the primitive all four used to call."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(BILLION_LAUGHS_OPML)
    outline = root.find("./body/outline")
    assert outline is not None
    assert len(outline.get("text") or "") >= 3 * 10**5


# --------------------------------------------------------------------------
# Qodo review of #2800: the widened `ValueError` handler must guard the PARSE
# and nothing else. It originally enclosed the article loop and the
# caller-supplied `filter_function` too.
# --------------------------------------------------------------------------

_VALID_SITEMAP = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    "<url><loc>https://example.invalid/a</loc></url>"
    "</urlset>"
)


def test_a_filter_that_raises_is_not_reported_as_a_broken_sitemap(tmp_path):
    """A `ValueError` from caller-supplied filtering code used to be logged
    as "Error parsing sitemap" and swallowed into an empty list -- a broken
    filter was indistinguishable from an empty sitemap, and the caller was
    told nothing at all."""
    from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as AEL

    path = tmp_path / "sitemap.xml"
    path.write_text(_VALID_SITEMAP, encoding="utf-8")

    def _broken_filter(_url):
        raise ValueError("the filter itself is broken")

    with pytest.raises(ValueError, match="the filter itself is broken"):
        AEL.scrape_from_filtered_sitemap(str(path), _broken_filter)


def test_a_scrape_failure_is_not_reported_as_a_broken_sitemap(
    tmp_path, monkeypatch, no_config_reads
):
    """Same defect on the fetched sibling: its `try` enclosed the whole
    `scrape_article` comprehension."""
    from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as AEL

    class _Response:
        content = _VALID_SITEMAP.encode()

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        AEL, "guarded_fetch_requests", lambda url, **kwargs: _Response()
    )

    def _broken_scrape(_url):
        raise ValueError("scraping blew up")

    monkeypatch.setattr(AEL, "scrape_article", _broken_scrape)

    with pytest.raises(ValueError, match="scraping blew up"):
        AEL.scrape_from_sitemap("https://example.invalid/sitemap.xml")


@pytest.mark.parametrize(
    "source, expected",
    [
        ("https://user:secret@host.invalid/maps/sitemap.xml?token=abc",
         "https://host.invalid/maps/sitemap.xml"),
        ("/Users/someone/Documents/private/sitemap.xml", "sitemap.xml"),
        ("sitemap.xml", "sitemap.xml"),
    ],
)
def test_the_logged_sitemap_name_carries_no_secret_and_no_home_path(
    source, expected
):
    """The failure must be attributable without putting credentials, query
    tokens, or a profile-owned path into a persistent sink."""
    from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as AEL

    assert AEL._sitemap_log_name(source) == expected


# --------------------------------------------------------------------------
# Qodo review of #2800: the LOCAL sitemap path reached `_safe_parse` -- and so
# the filesystem -- without the central `Utils.path_validation` module, so a
# traversal spelling or a bogus absolute path was opened as-is.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    ["traversal", "home_expansion", "missing"],
)
def test_a_refused_local_sitemap_path_never_reaches_the_file_read(
    tmp_path, monkeypatch, spelling
):
    """Refusal happens before the open, not inside the XML parser.

    The traversal case deliberately resolves BACK to the real file, so the
    filesystem itself would have served it: only the validator stops it.
    """
    from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as AEL

    (tmp_path / "sitemap.xml").write_text(_VALID_SITEMAP, encoding="utf-8")

    parsed = []
    real_parse = AEL._safe_parse
    monkeypatch.setattr(
        AEL, "_safe_parse", lambda path: parsed.append(path) or real_parse(path)
    )

    source = {
        "traversal": str(
            tmp_path / "x" / ".." / ".." / tmp_path.name / "sitemap.xml"
        ),
        "home_expansion": "~/sitemap.xml",
        "missing": str(tmp_path / "nope" / "sitemap.xml"),
    }[spelling]

    assert AEL.scrape_from_filtered_sitemap(source, lambda url: False) == []
    assert parsed == [], f"{spelling} path reached the file read: {parsed}"


def test_a_plain_local_sitemap_path_still_parses(tmp_path):
    """Negative control for the guard above: ordinary paths are unaffected."""
    from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as AEL

    path = tmp_path / "sitemap.xml"
    path.write_text(_VALID_SITEMAP, encoding="utf-8")

    seen = []
    assert AEL.scrape_from_filtered_sitemap(str(path), seen.append) == []
    assert seen == ["https://example.invalid/a"]
