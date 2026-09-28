"""dreams_view row shaping: full story rows, labels, failed-cycle synthetics.

Real ``DreamsDB`` on ``tmp_path`` (no fakes): pins the R2 contract -- every
``dream_stories`` column survives into the row for Task 7's modal, plus the
``label``/``collection_date`` shaping fields, kept-first ordering within a
collection, and the synthetic row that keeps a failed cycle visible even
when it produced no stories.
"""

import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.Dreams.dreams_view import format_dream_row, list_recent_dreams


@pytest.fixture()
def db(tmp_path):
    database = DreamsDB(tmp_path / "dreams.sqlite", "test-client")
    yield database
    database.close()


def _insert_story(db, collection_id, *, title, url, kept=False):
    story_id = db.insert_story(
        collection_id,
        title=title,
        url=url,
        snippet="Fares from $89",
        body="A story about fares.",
        status="complete",
        source="web",
        kind="deal",
        event_date=None,
        location="Japan",
        matched_topics=["visit japan"],
        query="cheap flights japan",
    )
    if kept:
        db.set_story_kept(story_id, True)
    return story_id


def test_brief_seed_orders_kept_first_and_ends_with_failed_cycle(db):
    complete = db.create_collection("2026-09-22", "scheduled", "digest-1")
    failed = db.create_collection("2026-09-21", "scheduled", "digest-2")
    db.set_collection_status(failed, "failed")
    kept_id = _insert_story(db, complete, title="Kept story", url="https://x/kept")
    plain_id = _insert_story(db, complete, title="Plain story", url="https://x/plain")

    rows = list_recent_dreams(db)

    assert [row["label"] for row in rows] == [
        "Kept story",
        "Plain story",
        "Cycle 2026-09-21: failed",
    ]
    assert rows[0]["id"] == kept_id
    assert rows[1]["id"] == plain_id
    assert rows[2] == {
        "label": "Cycle 2026-09-21: failed",
        "status": "failed",
        "kind": "unknown",
        "collection_date": "2026-09-21",
        "synthetic": True,
    }


def test_failed_cycle_newer_than_stories_leads_the_list(db):
    complete = db.create_collection("2026-09-21", "scheduled", "digest-1")
    failed = db.create_collection("2026-09-22", "scheduled", "digest-2")
    db.set_collection_status(failed, "failed")
    _insert_story(db, complete, title="Older story", url="https://x/older")

    rows = list_recent_dreams(db)

    assert rows[0]["synthetic"] is True
    assert rows[0]["collection_date"] == "2026-09-22"
    assert rows[1]["label"] == "Older story"
    assert rows[1]["collection_date"] == "2026-09-21"


def test_story_rows_carry_every_column_plus_shaping_fields(db):
    collection = db.create_collection("2026-09-22", "scheduled", "digest")
    story_id = _insert_story(
        db, collection, title="Cheap flights to Japan", url="https://x/1", kept=True
    )

    (row,) = list_recent_dreams(db)

    for column in (
        "id", "collection_id", "title", "url", "snippet", "body", "status",
        "source", "kind", "event_date", "location", "matched_topics", "query",
        "kept", "kept_at", "error", "created_at",
    ):
        assert column in row, f"Task 7's modal needs the {column!r} column"
    assert row["id"] == story_id
    assert row["collection_id"] == collection
    assert row["title"] == "Cheap flights to Japan"
    assert row["body"] == "A story about fares."
    assert row["matched_topics"] == ["visit japan"]
    assert row["query"] == "cheap flights japan"
    assert row["kept"] == 1
    assert row["label"] == "Cheap flights to Japan"
    assert row["collection_date"] == "2026-09-22"


def test_label_truncates_at_sixty_chars_but_title_survives(db):
    collection = db.create_collection("2026-09-22", "scheduled", "digest")
    long_title = "T" * 80
    _insert_story(db, collection, title=long_title, url="https://x/long")

    (row,) = list_recent_dreams(db)

    assert row["label"] == "T" * 60
    assert row["title"] == long_title


def test_format_dream_row_matches_report_row_idiom():
    assert (
        format_dream_row({"label": "Kept story", "kept": 1})
        == "> Dream: Kept story · kept"
    )
    assert (
        format_dream_row({"label": "Plain story", "kept": 0})
        == "> Dream: Plain story"
    )
    # The synthetic failed-cycle row renders with no kept badge.
    assert (
        format_dream_row(
            {"label": "Cycle 2026-09-21: failed", "synthetic": True}
        )
        == "> Dream: Cycle 2026-09-21: failed"
    )
