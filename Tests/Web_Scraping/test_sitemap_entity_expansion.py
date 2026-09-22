"""The two sitemap parsers in `Web_Scraping/` refuse entity expansion.

Tier-2 review S14, P2 [D1]: `Web_Scraping/Article_Extractor_Lib.py` and
`Web_Scraping/Article_Scraper/crawler.py` were the last two entries on the
repo's open-XXE register (`Tests/Subscriptions/
test_watchlist_opml_entity_expansion.py::_KNOWN_UNHARDENED`) that parse
FETCHED sitemaps with stdlib ElementTree.

A response size cap is no defence here -- amplification is the whole point
of a billion-laughs payload, so a few hundred bytes on the wire become
gigabytes inside the parser, well under `MAX_FETCH_BYTES_SITEMAP` (50 MB).

`crawler.py:403` is the one that goes live the moment the empty
`Article_Scraper/__init__.py` (S14's P1) is fixed: `Subscriptions`'
default generic scraper reaches it.

These assert the REFUSAL, not a timeout -- a "this took too long" test is
a flake generator and proves nothing (see the OPML test's own reasoning).
"""

from __future__ import annotations

import pytest
from defusedxml.common import EntitiesForbidden

#: Six nesting levels of a 10x entity: ~10^6 "lol" copies if expanded.
#: Bounded on purpose -- unmistakable if expanded, harmless if not.
BILLION_LAUGHS_SITEMAP = """<?xml version="1.0"?>
<!DOCTYPE urlset [
  <!ENTITY lol "lol">
  <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
  <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">
  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
  <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
  <!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;">
]>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.invalid/&lol5;</loc></url>
</urlset>
"""

BENIGN_SITEMAP = """<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.invalid/a</loc></url>
</urlset>
"""

_NS = ".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"


def test_the_parser_the_crawler_actually_uses_refuses_entity_expansion():
    """Asserted on `crawler.ET`, the binding `_parse_sitemap` calls."""
    from tldw_chatbook.Web_Scraping.Article_Scraper import crawler

    with pytest.raises(EntitiesForbidden):
        crawler.ET.fromstring(BILLION_LAUGHS_SITEMAP)


def test_the_crawler_still_parses_an_ordinary_sitemap():
    from tldw_chatbook.Web_Scraping.Article_Scraper import crawler

    root = crawler.ET.fromstring(BENIGN_SITEMAP)
    assert [e.text for e in root.findall(_NS)] == ["https://example.invalid/a"]


def test_the_article_extractors_sitemap_parsers_refuse_entity_expansion():
    """Both entry points: `fromstring` (fetched) and `parse` (file)."""
    from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as m

    with pytest.raises(EntitiesForbidden):
        m._safe_fromstring(BILLION_LAUGHS_SITEMAP)


def test_the_article_extractors_file_parser_refuses_entity_expansion(tmp_path):
    from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as m

    sitemap = tmp_path / "sitemap.xml"
    sitemap.write_text(BILLION_LAUGHS_SITEMAP, encoding="utf-8")

    with pytest.raises(EntitiesForbidden):
        m._safe_parse(str(sitemap))


def test_the_article_extractor_still_parses_an_ordinary_sitemap(tmp_path):
    from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as m

    sitemap = tmp_path / "sitemap.xml"
    sitemap.write_text(BENIGN_SITEMAP, encoding="utf-8")

    assert [e.text for e in m._safe_parse(str(sitemap)).getroot().findall(_NS)] == [
        "https://example.invalid/a"
    ]
    assert [e.text for e in m._safe_fromstring(BENIGN_SITEMAP).findall(_NS)] == [
        "https://example.invalid/a"
    ]


def test_document_building_still_uses_stdlib_elementtree():
    """`Element`/`SubElement` have no defusedxml counterpart, and a tree we
    build ourselves has no attacker-controlled input -- so the stdlib alias
    stays, exactly as `watchlist_opml_service.export()` keeps it."""
    from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as m

    assert m.xET.__name__ == "xml.etree.ElementTree"
