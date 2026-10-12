"""Production catalog Dreams paging, selected bodies and collection notices."""

import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.Library.library_artifacts_catalog import LibraryArtifactsCatalog
from tldw_chatbook.Library.library_artifacts_state import ArtifactScope


@pytest.fixture
def dreams_catalog(tmp_path):
    db = DreamsDB(tmp_path / "dreams.sqlite", "library-dreams")
    collection = db.create_collection("2026-10-12", "manual", "digest")
    for index in range(25):
        db.insert_story(
            collection,
            title=f"Story {index:02}",
            url=f"https://example.com/{index}",
            snippet="snippet",
            body=f"Private body {index}",
            status="complete",
            source="web",
            kind="content",
            event_date=None,
            location=None,
            matched_topics=["science"],
            query="science",
        )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE dreams_collections SET status='partial', degradation_notes=?",
            ("[link=https://evil.invalid]Search unavailable[/link]",),
        )
    empty = db.create_collection("2026-10-11", "manual", "digest")
    with db.transaction() as conn:
        conn.execute(
            "UPDATE dreams_collections SET status='failed' WHERE id=?", (empty,)
        )
    try:
        yield (
            db,
            LibraryArtifactsCatalog(
                subscriptions_db=None, chachanotes_db=None, dreams_db_getter=lambda: db
            ),
        )
    finally:
        db.close_all()


def test_dreams_catalog_bounded_paging_locates_selected_body_and_notice(dreams_catalog):
    db, catalog = dreams_catalog
    scope = ArtifactScope(view="dreams", sort="title")
    statements = []
    with db.connection() as conn:
        conn.set_trace_callback(statements.append)
        page = catalog.read_page(scope)
        conn.set_trace_callback(None)
    assert page.total == 26 and len(page.items) == 20
    assert all("body" not in statement.lower() for statement in statements)
    next_page = catalog.read_page(scope, boundary=page.items[-1].order_key)
    assert next_page.start == 20 and len(next_page.items) == 6
    target = next_page.items[-1]
    located = catalog.locate(scope, target.key)
    assert located.start == 20 and target in located.items
    detail = catalog.read_detail(target.key)
    assert detail.body.startswith("Private body")
    assert dict(detail.details)["Collection status"] == "partial"
    assert (
        dict(detail.details)["Degradation"]
        == "[link=https://evil.invalid]Search unavailable[/link]"
    )
    empty = next(row for row in page.items if row.key.source == "dream_collection")
    assert catalog.read_detail(empty.key).body == ""
    assert empty.status == "failed"
    assert catalog.read_page(ArtifactScope(view="reports")).total == 0
    assert catalog.read_page(ArtifactScope(view="all")).total == 26


def test_normal_dream_collection_has_no_degradation_notice(dreams_catalog):
    db, catalog = dreams_catalog
    with db.transaction() as conn:
        conn.execute("UPDATE dreams_collections SET degradation_notes=NULL")
    row = next(
        row
        for row in catalog.read_page(ArtifactScope(view="dreams")).items
        if row.key.source == "dream_story"
    )
    assert "Degradation" not in dict(catalog.read_detail(row.key).details)


def test_tracked_status_keeps_origin_actions_and_missing_origin_is_readable(
    dreams_catalog,
):
    db, catalog = dreams_catalog
    first = db.list_stories(1)[0]["id"]
    tracked = db.create_tracked_item(
        origin_story_id=first,
        mechanism="question",
        intent="topic",
        query_template="science",
        cadence_seconds=3600,
    )
    missing = db.create_tracked_item(
        origin_story_id=9999,
        mechanism="question",
        intent="topic",
        query_template="[broken]Missing[/other]",
        cadence_seconds=3600,
    )
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO dream_track_runs(tracked_item_id,status,verdict_note,created_at) VALUES (?, 'changed', ?, '2026-10-12T00:00:00+00:00')",
            (tracked, "[link=https://evil.invalid]New result[/link]"),
        )
    page = catalog.read_page(ArtifactScope(view="dreams", query="science"))
    row = next(row for row in page.items if row.key.source == "dream_track")
    detail = catalog.read_detail(row.key)
    assert detail.dream_story["id"] == first
    assert dict(detail.details)["Last check"] == "changed"
    assert (
        dict(detail.details)["Verdict"]
        == "[link=https://evil.invalid]New result[/link]"
    )
    page = catalog.read_page(ArtifactScope(view="dreams", query="Missing"))
    row = next(row for row in page.items if row.key.native_id == missing)
    detail = catalog.read_detail(row.key)
    assert detail.dream_story is None
    assert dict(detail.details)["Tracking"] == "active"
    assert dict(detail.details)["Mechanism"] == "question"
    assert dict(detail.details)["Last check"] == "none yet"
    assert dict(detail.details)["Query"] == "[broken]Missing[/other]"
