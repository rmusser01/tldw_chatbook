import pytest
from tldw_chatbook.Subscriptions.watchlist_opml_service import WatchlistOpmlService


def test_parse_opml():
    xml = '''<?xml version="1.0"?><opml version="2.0"><body><outline text="Tech" title="Tech"><outline text="AI" title="AI" type="rss" xmlUrl="http://example.com/ai"/></outline></body></opml>'''
    svc = WatchlistOpmlService()
    items = svc.parse(xml)
    assert len(items) == 1
    assert items[0]["url"] == "http://example.com/ai"
    assert items[0]["source_type"] == "rss"


def test_export_opml():
    svc = WatchlistOpmlService()
    xml = svc.export([
        {"name": "AI", "url": "http://example.com/ai", "source_type": "rss"}
    ])
    assert "http://example.com/ai" in xml


def test_a_c1_control_byte_in_an_imported_name_survives_parse():
    """Batch-4 review, I1 (the delivery vector). A raw 7-bit ESC embedded in
    an OPML `text=` attribute is rejected by `xml.etree.ElementTree` --
    `ParseError: not well-formed` -- so the classic ESC-injection form is
    blocked by XML's own grammar. A C1 control byte (0x80-0x9F, e.g. 0x9B,
    the single-byte CSI introducer `html_text.strip_control_characters`'s
    own docstring calls "just as capable" as `ESC [`) is valid in XML 1.0's
    character range and parses cleanly. This is the precondition for the
    next test: `parse()` itself does no sanitization at all, by design (it
    is a structural parser, not a content sanitizer) -- the display
    boundary is where this has to be stopped.
    """
    xml = (
        '<?xml version="1.0"?><opml version="2.0"><body>'
        '<outline text="Evil\x9b31mFeed" type="rss" '
        'xmlUrl="http://example.com/feed"/>'
        "</body></opml>"
    )
    svc = WatchlistOpmlService()
    items = svc.parse(xml)
    assert len(items) == 1
    assert items[0]["name"] == "Evil\x9b31mFeed", (
        "parse() must not silently sanitize -- that is not its job; if this "
        "assertion ever goes red because \\x9b vanished here, the render-"
        "boundary test below has stopped testing what it claims to"
    )


def test_a_c1_control_byte_from_opml_import_is_stripped_at_the_sources_table_cell():
    """Batch-4 review, I1 (the fix). The C1 byte confirmed to survive
    `parse()` above must not reach the Sources table's rendered cell --
    `SourcesPane._source_row_cells` is the render boundary
    `strip_control_characters` was extended to cover, the same discipline
    `content_pane.py` already applies to the reader.
    """
    from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane

    xml = (
        '<?xml version="1.0"?><opml version="2.0"><body>'
        '<outline text="Evil\x9b31mFeed" type="rss" '
        'xmlUrl="http://example.com/feed"/>'
        "</body></opml>"
    )
    payload = WatchlistOpmlService().parse(xml)[0]
    # The shape `_source_row_cells` actually reads: a normalized source
    # dict carrying the imported name.
    source = {"name": payload["name"], "source_type": payload["source_type"]}

    cells = SourcesPane._source_row_cells(source, False)
    name_cell = cells[0].plain

    assert "\x9b" not in name_cell, (
        f"a C1 control byte from an imported OPML name reached the Sources "
        f"table cell verbatim: {name_cell!r}"
    )
    assert "Evil" in name_cell and "31mFeed" in name_cell, (
        "the surrounding text must still render -- only the control byte is "
        "removed, not the characters around it"
    )
