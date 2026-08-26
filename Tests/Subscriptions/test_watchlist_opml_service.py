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
    xml = svc.export(
        [],
        [{"name": "AI", "url": "http://example.com/ai", "source_type": "rss"}],
    )
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


# --- TASK-3604 plan task 2: parse preserves folder structure (ADR-043) ---------


def test_parse_groups_feeds_under_their_folder():
    xml = (
        '<?xml version="1.0"?><opml version="2.0"><body>'
        '<outline text="Tech">'
        '<outline text="AI" type="rss" xmlUrl="http://example.com/ai"/>'
        '<outline text="ML" type="rss" xmlUrl="http://example.com/ml"/>'
        '</outline>'
        '<outline text="Loose" type="rss" xmlUrl="http://example.com/loose"/>'
        "</body></opml>"
    )
    items = WatchlistOpmlService().parse(xml)
    by_url = {item["url"]: item for item in items}
    assert by_url["http://example.com/ai"]["folder"] == "Tech"
    assert by_url["http://example.com/ml"]["folder"] == "Tech"
    assert by_url["http://example.com/loose"]["folder"] is None
    assert len(items) == 3, "the folder outline itself is not a feed"


def test_parse_flattens_nested_folders_to_the_innermost():
    """ADR-043 rule 2: the folder directly containing the feed wins."""
    xml = (
        '<?xml version="1.0"?><opml version="2.0"><body>'
        '<outline text="Tech"><outline text="AI">'
        '<outline text="Paperfeed" type="rss" xmlUrl="http://example.com/p"/>'
        "</outline></outline>"
        "</body></opml>"
    )
    items = WatchlistOpmlService().parse(xml)
    assert items[0]["folder"] == "AI"


def test_parse_a_feed_with_children_is_a_feed_and_children_inherit_its_context():
    """ADR-043 rule 3: an outline with a feed URL AND children is a feed;
    its children are evaluated under its folder context, never under a
    folder of the feed's own making."""
    xml = (
        '<?xml version="1.0"?><opml version="2.0"><body>'
        '<outline text="Tech">'
        '<outline text="ParentFeed" type="rss" xmlUrl="http://example.com/parent">'
        '<outline text="ChildFeed" type="rss" xmlUrl="http://example.com/child"/>'
        '</outline>'
        '</outline>'
        "</body></opml>"
    )
    items = WatchlistOpmlService().parse(xml)
    by_url = {item["url"]: item for item in items}
    assert by_url["http://example.com/parent"]["folder"] == "Tech"
    assert by_url["http://example.com/child"]["folder"] == "Tech", (
        "the child inherits the folder context -- ParentFeed is a feed, "
        "not a folder"
    )


def test_parse_surfaces_folder_names_faithfully():
    """Case variants surface RAW (the case-insensitive reuse is the
    assignment layer's job, task 3), and a hostile folder name is a literal
    string here -- parse is a structural parser, not a sanitizer (the C1
    test above established that contract for feed names)."""
    xml = (
        '<?xml version="1.0"?><opml version="2.0"><body>'
        '<outline text="AI"><outline text="a" type="rss" xmlUrl="http://example.com/a"/></outline>'
        '<outline text="ai"><outline text="b" type="rss" xmlUrl="http://example.com/b"/></outline>'
        '<outline text="&lt;script&gt;x">'
        '<outline text="c" type="rss" xmlUrl="http://example.com/c"/></outline>'
        "</body></opml>"
    )
    items = WatchlistOpmlService().parse(xml)
    by_url = {item["url"]: item for item in items}
    assert by_url["http://example.com/a"]["folder"] == "AI"
    assert by_url["http://example.com/b"]["folder"] == "ai"
    assert by_url["http://example.com/c"]["folder"] == "<script>x"


def test_parse_malformed_xml_still_raises():
    """The error path is unchanged: bad XML raises into the dialog's
    existing handler, never returns a partial import."""
    import xml.etree.ElementTree as ET

    with pytest.raises(ET.ParseError):
        WatchlistOpmlService().parse("<opml><body><outline")


# --- TASK-3604 plan task 3: import against a real DB ----------------------------


@pytest.mark.asyncio
async def test_import_assigns_membership_and_reimport_is_a_structural_noop(tmp_path):
    """ADR-043 end to end at the service layer: folders become watchlists,
    members join them, top-level feeds stay Unassigned -- and importing the
    same document twice changes nothing."""
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.Subscriptions.local_watchlists_service import (
        LocalWatchlistsService,
    )
    from tldw_chatbook.Subscriptions.watchlist_bundle_service import (
        WatchlistBundleService,
    )
    from tldw_chatbook.Subscriptions.watchlist_scope_service import (
        WatchlistBackend,
        WatchlistScopeService,
    )

    xml = (
        '<?xml version="1.0"?><opml version="2.0"><body>'
        '<outline text="Tech">'
        '<outline text="AI" type="rss" xmlUrl="http://example.com/ai"/>'
        '<outline text="ML" type="rss" xmlUrl="http://example.com/ml"/>'
        '</outline>'
        '<outline text="Loose" type="rss" xmlUrl="http://example.com/loose"/>'
        "</body></opml>"
    )

    db = SubscriptionsDB(str(tmp_path / "subs.db"), client_id="test")
    local = LocalWatchlistsService(db_factory=lambda: db)
    scope = WatchlistScopeService(local_service=local, server_service=None)
    bundle = WatchlistBundleService(db)

    result = await scope.import_opml(
        runtime_backend=WatchlistBackend.LOCAL, xml_text=xml
    )

    watchlists = bundle.list_watchlists()
    assert [w["name"] for w in watchlists] == ["Tech"]
    members = bundle.list_source_rows(watchlists[0]["id"])
    assert {row["name"] for row in members} == {"AI", "ML"}
    total_sources = db.conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
    assert total_sources == 3, "the loose feed exists but belongs nowhere"
    assert result["created"] == 3
    assert result["assignments"] == 2

    again = await scope.import_opml(
        runtime_backend=WatchlistBackend.LOCAL, xml_text=xml
    )
    assert again["created"] == 0 and again["existing"] == 3
    assert len(bundle.list_watchlists()) == 1, "no duplicate watchlist"
    assert len(bundle.list_source_rows(watchlists[0]["id"])) == 2
    total_sources = db.conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
    assert total_sources == 3, "no duplicate sources"


# --- TASK-3604 plan task 4: export nests watchlists as folders (ADR-043 rule 5) -


def test_export_nests_watchlists_as_folders_with_unassigned_top_level():
    """One folder per watchlist (name-ordered), members nested (name-
    ordered), a shared source under EACH of its watchlists, unassigned
    feeds flat at the top level."""
    import xml.etree.ElementTree as ET

    svc = WatchlistOpmlService()
    xml = svc.export(
        [
            {
                "name": "Tech",
                "sources": [
                    {"name": "Shared", "url": "http://x.com/shared", "source_type": "rss"},
                    {"name": "AI", "url": "http://x.com/ai", "source_type": "rss"},
                ],
            },
            {
                "name": "News",
                "sources": [
                    {"name": "Shared", "url": "http://x.com/shared", "source_type": "rss"},
                ],
            },
        ],
        [{"name": "Loose", "url": "http://x.com/loose", "source_type": "rss"}],
    )

    body = ET.fromstring(xml).find("body")
    top = list(body)
    folders = [el for el in top if el.get("xmlUrl") is None]
    loose = [el for el in top if el.get("xmlUrl") is not None]
    assert [f.get("text") for f in folders] == ["News", "Tech"], (
        "folders are name-ordered (case-insensitive)"
    )
    assert [el.get("text") for el in loose] == ["Loose"]
    tech = next(f for f in folders if f.get("text") == "Tech")
    assert [c.get("text") for c in tech] == ["AI", "Shared"], "members name-ordered"
    news = next(f for f in folders if f.get("text") == "News")
    assert [c.get("text") for c in news] == ["Shared"], (
        "membership is many-to-many: the shared source appears under both"
    )
    assert folders[0].get("xmlUrl") is None, "a folder outline carries no feed URL"


def test_export_escapes_hostile_watchlist_names():
    """A markup/quote-bearing watchlist name serializes escaped and
    re-parses to the literal string (the parse contract, both ends)."""
    svc = WatchlistOpmlService()
    hostile = '<script>alert("x")</script>'
    xml = svc.export(
        [{"name": hostile, "sources": [
            {"name": "F", "url": "http://x.com/f", "source_type": "rss"},
        ]}],
        [],
    )
    assert "<script>" not in xml, "the serializer must not emit raw markup"
    items = svc.parse(xml)
    assert items[0]["folder"] == hostile, (
        "the name round-trips as the literal string"
    )


def test_export_with_no_watchlists_is_flat():
    """Backward compatibility: a profile with no watchlists exports the
    same flat document the pre-mapping exporter produced."""
    import xml.etree.ElementTree as ET

    svc = WatchlistOpmlService()
    xml = svc.export(
        [],
        [
            {"name": "B", "url": "http://x.com/b", "source_type": "rss"},
            {"name": "A", "url": "http://x.com/a", "source_type": "rss"},
        ],
    )
    top = list(ET.fromstring(xml).find("body"))
    assert all(el.get("xmlUrl") is not None for el in top), "no folders at all"
    assert [el.get("text") for el in top] == ["A", "B"], "flat AND name-ordered"


# --- TASK-3604 plan task 5: the round-trip pin ---------------------------------


@pytest.mark.asyncio
async def test_import_counts_a_repeated_url_once_and_reports_unassigned_sources(
    tmp_path,
):
    """A shared exported feed is one source, not an "already present" one.

    Removing the in-import URL memo makes ``existing`` become 1 on a fresh
    database; deriving Unassigned from membership-edge count makes it become
    0 even though the loose feed is still top-level.
    """
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.Subscriptions.local_watchlists_service import (
        LocalWatchlistsService,
    )
    from tldw_chatbook.Subscriptions.watchlist_scope_service import (
        WatchlistBackend,
        WatchlistScopeService,
    )

    document = WatchlistOpmlService().export(
        [
            {
                "name": "News",
                "sources": [
                    {
                        "name": "Shared",
                        "url": "http://x.com/shared",
                        "source_type": "rss",
                    }
                ],
            },
            {
                "name": "Tech",
                "sources": [
                    {
                        "name": "Shared",
                        "url": "http://x.com/shared",
                        "source_type": "rss",
                    }
                ],
            },
        ],
        [{"name": "Loose", "url": "http://x.com/loose", "source_type": "rss"}],
    )
    db = SubscriptionsDB(str(tmp_path / "subs.db"), client_id="test")
    scope = WatchlistScopeService(
        local_service=LocalWatchlistsService(db_factory=lambda: db),
        server_service=None,
    )

    result = await scope.import_opml(
        runtime_backend=WatchlistBackend.LOCAL, xml_text=document
    )

    assert result["created"] == 2
    assert result["existing"] == 0
    assert result["assignments"] == 2
    assert result["unassigned"] == 1
    assert db.conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 2


@pytest.mark.asyncio
async def test_export_then_import_round_trips_the_structure(tmp_path):
    """The phase's done-when, machine-checked (ADR-043): exporting a
    structured profile and importing the document into a FRESH database
    reproduces the exact watchlists, memberships, and unassigned set."""
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.Subscriptions.local_watchlists_service import (
        LocalWatchlistsService,
    )
    from tldw_chatbook.Subscriptions.watchlist_bundle_service import (
        WatchlistBundleService,
    )
    from tldw_chatbook.Subscriptions.watchlist_scope_service import (
        WatchlistBackend,
        WatchlistScopeService,
    )

    db_a = SubscriptionsDB(str(tmp_path / "a.db"), client_id="test")
    bundle_a = WatchlistBundleService(db_a)
    tech = bundle_a.create("Tech")
    news = bundle_a.create("News")
    ai = db_a.add_subscription(name="AI", type="rss", source="http://x.com/ai")
    shared = db_a.add_subscription(name="Shared", type="rss", source="http://x.com/shared")
    db_a.add_subscription(name="Loose", type="rss", source="http://x.com/loose")
    bundle_a.add_source(tech["id"], ai)
    bundle_a.add_source(tech["id"], shared)
    bundle_a.add_source(news["id"], shared)
    scope_a = WatchlistScopeService(
        local_service=LocalWatchlistsService(db_factory=lambda: db_a),
        server_service=None,
    )

    document = await scope_a.export_opml(runtime_backend=WatchlistBackend.LOCAL)

    db_b = SubscriptionsDB(str(tmp_path / "b.db"), client_id="test")
    scope_b = WatchlistScopeService(
        local_service=LocalWatchlistsService(db_factory=lambda: db_b),
        server_service=None,
    )
    await scope_b.import_opml(runtime_backend=WatchlistBackend.LOCAL, xml_text=document)

    def structure(bundle) -> dict[str, list[str]]:
        return {
            w["name"]: sorted(r["name"] for r in bundle.list_source_rows(w["id"]))
            for w in bundle.list_watchlists()
        }

    bundle_b = WatchlistBundleService(db_b)
    assert structure(bundle_b) == structure(bundle_a) == {
        "News": ["Shared"],
        "Tech": ["AI", "Shared"],
    }
    assert {r["name"] for r in bundle_b.list_unassigned_source_rows()} == {"Loose"}
