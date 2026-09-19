"""A card whose due time has passed is due today, not tomorrow.

TASK-32803.2. `next_review` was written with `.isoformat()`
(`2026-09-19T02:13:11.35+00:00`) and compared lexically against SQLite's
`CURRENT_TIMESTAMP` (`2026-09-19 02:13:11`). The `T` (0x54) sorts after a
space (0x20), so a card due at 02:13 today read as not-due until the UTC
day rolled over -- it came due "the day after". The write is canonical now
(space-separated UTC, matching CURRENT_TIMESTAMP), and both due queries
compare through `datetime()` so rows already written in the old shape read
correctly too.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db():
    # Qodo #1: yield + close so the in-memory connection is torn down
    # deterministically instead of leaking to GC.
    database = CharactersRAGDB(":memory:", "flashcard-due-test")
    yield database
    database.close_connection()


def _card(db) -> str:
    deck_id = db.create_deck("Biology", "Cell review")
    return db.create_flashcard(
        {"deck_id": deck_id, "front": "Q", "back": "A", "type": "basic"}
    )


def _set_next_review(db, card_id: str, value: str) -> None:
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE flashcards SET next_review = ? WHERE id = ?", (value, card_id)
        )


def test_a_card_due_earlier_today_is_returned_as_due(db):
    card_id = _card(db)
    # Due two hours ago, in the CANONICAL shape the fix writes.
    due = datetime.now(timezone.utc) - timedelta(hours=2)
    _set_next_review(db, card_id, due.strftime("%Y-%m-%d %H:%M:%S"))

    due_ids = [row["id"] for row in db.get_due_flashcards()]
    assert card_id in due_ids
    assert db.count_due_flashcards() >= 1


def test_a_card_due_later_today_is_not_yet_due(db):
    card_id = _card(db)
    later = datetime.now(timezone.utc) + timedelta(hours=2)
    _set_next_review(db, card_id, later.strftime("%Y-%m-%d %H:%M:%S"))

    assert card_id not in [row["id"] for row in db.get_due_flashcards()]


def test_a_row_in_the_old_isoformat_shape_is_read_correctly(db):
    """AC#3: a card written before the fix (T-separated, offset, micros)
    and already past due must still be returned as due."""
    card_id = _card(db)
    due = datetime.now(timezone.utc) - timedelta(hours=2)
    _set_next_review(db, card_id, due.isoformat())  # e.g. ...T..+00:00

    assert card_id in [row["id"] for row in db.get_due_flashcards()], (
        "an old-shape row that is genuinely past due was not read as due"
    )


def test_an_old_shape_row_due_later_is_still_not_due(db):
    card_id = _card(db)
    later = datetime.now(timezone.utc) + timedelta(hours=2)
    _set_next_review(db, card_id, later.isoformat())

    assert card_id not in [row["id"] for row in db.get_due_flashcards()]


def test_the_real_review_write_stores_the_canonical_shape(db):
    """AC#4: drive the actual review path, not a hand-set value.

    A rating that pushes the card out to a future interval must store
    `next_review` in SQLite's own timestamp shape (space-separated, no `T`,
    no offset), so the lexical due comparison stays a real time comparison.
    """
    card_id = _card(db)
    db.update_flashcard_review(card_id, rating=5)

    with db.transaction() as cursor:
        stored = cursor.execute(
            "SELECT CAST(next_review AS TEXT) FROM flashcards WHERE id = ?",
            (card_id,),
        ).fetchone()[0]

    assert stored is not None
    assert "T" not in stored, f"review wrote a T-separated timestamp: {stored!r}"
    assert "+" not in stored, f"review wrote a tz-offset timestamp: {stored!r}"
    assert "." not in stored, f"review wrote sub-second precision: {stored!r}"
    # And it is a value SQLite compares correctly against CURRENT_TIMESTAMP.
    with db.transaction() as cursor:
        parsed = cursor.execute("SELECT datetime(?)", (stored,)).fetchone()[0]
    assert parsed == stored


def test_mixed_format_same_date_cards_are_served_chronologically(db):
    """Qodo #5: a legacy T-separated row and a new space-separated row on the
    SAME date must be ordered by their normalized time, not raw text. With
    limit=1, the chronologically-earliest due card must win -- raw ordering
    (' ' 0x20 < 'T' 0x54) would place the space-separated (later) card first."""
    deck_id = db.create_deck("Chrono", "order check")
    earlier = db.create_flashcard(
        {"deck_id": deck_id, "front": "Q1", "back": "A1", "type": "basic"}
    )
    later = db.create_flashcard(
        {"deck_id": deck_id, "front": "Q2", "back": "A2", "type": "basic"}
    )
    _set_next_review(db, earlier, "2020-01-01T01:00:00")   # legacy T-format, 01:00
    _set_next_review(db, later, "2020-01-01 05:00:00")     # new space format, 05:00

    due = db.get_due_flashcards(limit=1)

    assert len(due) == 1
    assert due[0]["id"] == earlier, "due cards were not ordered chronologically"
