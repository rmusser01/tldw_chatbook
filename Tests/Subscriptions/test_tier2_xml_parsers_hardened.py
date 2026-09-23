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

`Local_Ingestion/XML_Ingestion.py` is hardened in the same commit, and its
behavioural tests are at the bottom of this file. They arrived late, and the
reason is worth recording: the module **cannot be imported** --
`from tldw_chatbook.DB.Client_Media_DB_v2 import add_media_to_database`
raises `ImportError` because that name does not exist (that module defines
`add_media_with_keywords`, a method, and no such module-level function).
That is a pre-existing defect this task did not cause and has not fixed;
nothing under `tldw_chatbook/` imports `XML_Ingestion` either, so the module
is currently orphaned. But it is a broken IMPORT, not an untestable parser:
stubbing that one name lets the real module load and its real `ET.parse` be
exercised, which is exactly what an AST census cannot do.
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
# `Local_Ingestion/XML_Ingestion.py` -- the behavioural half the module
# docstring said could not exist (Qodo review of #2800).
#
# It could not be imported because `from ...Client_Media_DB_v2 import
# add_media_to_database` names a symbol that module does not define. That is
# still true, and still a pre-existing defect this task did not cause -- but
# it is a broken IMPORT, not an untestable parser. Stubbing that import lets
# the real module load and the real `ET.parse` be exercised. Without this,
# swapping the alias back to the stdlib parser passes every check in the
# repo.
# --------------------------------------------------------------------------


def _load_xml_ingestion(monkeypatch):
    """Import XML_Ingestion with its broken and heavyweight deps stubbed."""
    import importlib
    import sys
    import types

    stubs = {
        # The name that does not exist -- the whole reason this module is
        # unimportable.
        "tldw_chatbook.DB.Client_Media_DB_v2": {
            "add_media_to_database": lambda *a, **k: None
        },
        "tldw_chatbook.LLM_Calls.Summarization_General_Lib": {
            "analyze": lambda *a, **k: ""
        },
        "tldw_chatbook.Chunking.Chunk_Lib": {"chunk_xml": lambda *a, **k: []},
        "tldw_chatbook.Metrics.metrics_logger": {
            "log_counter": lambda *a, **k: None,
            "log_histogram": lambda *a, **k: None,
        },
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(
        sys.modules, "tldw_chatbook.Local_Ingestion.XML_Ingestion", raising=False
    )
    return importlib.import_module("tldw_chatbook.Local_Ingestion.XML_Ingestion")


def test_xml_ingestion_parses_through_defusedxml(monkeypatch):
    """The alias must be defusedxml, asserted on the module object rather
    than on the source text an AST census already reads."""
    module = _load_xml_ingestion(monkeypatch)

    assert module.ET.__name__ == "defusedxml.ElementTree"


def test_xml_ingestion_refuses_entity_expansion(tmp_path, monkeypatch):
    """The behaviour, through `ET.parse` -- the exact symbol `xml_to_text`
    and `import_xml_file` both parse with."""
    module = _load_xml_ingestion(monkeypatch)

    path = tmp_path / "payload.xml"
    path.write_text(BILLION_LAUGHS_OPML, encoding="utf-8")

    with pytest.raises(_entities_forbidden()):
        module.ET.parse(str(path))


def test_xml_ingestion_still_reads_a_benign_document(tmp_path, monkeypatch):
    """Negative control: the hardening refuses the payload, not XML."""
    module = _load_xml_ingestion(monkeypatch)

    path = tmp_path / "benign.xml"
    path.write_text(BENIGN_OPML, encoding="utf-8")

    assert module.ET.parse(str(path)).getroot() is not None
