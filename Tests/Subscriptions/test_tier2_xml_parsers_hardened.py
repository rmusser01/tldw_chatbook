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

`Local_Ingestion/XML_Ingestion.py` is hardened in the same commit but has no
behavioural test here, and the reason is worth recording: at
`origin/dev d0face3ebe` the module **cannot be imported at all** --
`from tldw_chatbook.DB.Client_Media_DB_v2 import add_media_to_database`
raises `ImportError` because that name no longer exists. A pre-existing,
unrelated defect, not caused by this task; it leaves the AST census as the
only check that reaches that file.
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
