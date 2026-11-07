"""A card whose due time has passed is due today, not tomorrow.

TASK-32803.2 history: `next_review` was written with `.isoformat()`
(`2026-09-19T02:13:11.35+00:00`) and compared lexically against SQLite's
`CURRENT_TIMESTAMP` (`2026-09-19 02:13:11`). The `T` (0x54) sorts after a
space (0x20), so a card due at 02:13 today read as not-due until the UTC
day rolled over -- it came due "the day after". 32803.2 fixed the read by
comparing through ``datetime()`` and matched the write to the
space-separated shape.

ADR-224 flip: the v78→v79 migration normalized every stored
``next_review`` to the ADR-173 canonical shape
``YYYY-MM-DDTHH:MM:SS.mmmZ`` (fixed width, so raw TEXT order == time
order), and ``update_flashcard_review`` now writes that shape. The due
queries compare RAW ``next_review`` against a canonical now bound in
Python -- sargable via ``idx_flashcards_next_review``. Legacy-shape rows
are covered by the migration suite in ``test_sargable_timestamps.py``
(normalization + golden ordering), not by hand-seeding shapes no writer
emits anymore.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Utils.timestamps import parse_utc

#: ADR-173 canonical stored shape.
CANONICAL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


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


def _canonical(moment: datetime) -> str:
    """Render ``moment`` in the canonical stored shape (the only shape the
    post-v79 writers emit)."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def test_a_card_due_earlier_today_is_returned_as_due(db):
    card_id = _card(db)
    # Due two hours ago, in the canonical shape the fix writes.
    due = datetime.now(timezone.utc) - timedelta(hours=2)
    _set_next_review(db, card_id, _canonical(due))

    due_ids = [row["id"] for row in db.get_due_flashcards()]
    assert card_id in due_ids
    assert db.count_due_flashcards() >= 1


def test_a_card_due_later_today_is_not_yet_due(db):
    card_id = _card(db)
    later = datetime.now(timezone.utc) + timedelta(hours=2)
    _set_next_review(db, card_id, _canonical(later))

    assert card_id not in [row["id"] for row in db.get_due_flashcards()]


def test_a_card_due_exactly_now_is_due(db):
    """The raw comparison stays inclusive at the boundary, matching the old
    ``datetime(next_review) <= datetime('now')`` semantics."""
    card_id = _card(db)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    _set_next_review(db, card_id, _canonical(now))

    assert card_id in [row["id"] for row in db.get_due_flashcards()]


def test_a_never_reviewed_card_is_due(db):
    """NULL next_review (what ``create_flashcard`` leaves) means due now."""
    card_id = _card(db)

    assert card_id in [row["id"] for row in db.get_due_flashcards()]
    assert db.count_due_flashcards() >= 1


def test_the_real_review_write_stores_the_canonical_shape(db):
    """AC#4 of TASK-32803.2, carried into the ADR-224 contract: drive the
    actual review path, not a hand-set value.

    A passing rating that pushes the card out to a future interval must
    store ``next_review`` in the canonical shape (``T`` separator,
    millisecond precision, ``Z`` suffix, no offset), so the raw due
    comparison stays a real time comparison.
    """
    card_id = _card(db)
    before = datetime.now(timezone.utc)
    db.update_flashcard_review(card_id, rating=5)

    with db.transaction() as cursor:
        stored = cursor.execute(
            "SELECT CAST(next_review AS TEXT) FROM flashcards WHERE id = ?",
            (card_id,),
        ).fetchone()[0]

    assert stored is not None
    assert CANONICAL_RE.match(stored), (
        f"review wrote a non-canonical timestamp: {stored!r}"
    )
    # The next review is genuinely in the future and round-trips through
    # the tolerant parser.
    parsed = parse_utc(stored)
    assert parsed > before


def test_same_date_cards_are_served_chronologically_with_limit_one(db):
    """Qodo #5 / PR #2735 carried into the canonical world: two due cards
    on the same date must be ordered by their time, not anything else, so
    ``get_next_review_candidate`` (limit=1) serves the
    chronologically-earliest card."""
    deck_id = db.create_deck("Chrono", "order check")
    earlier = db.create_flashcard(
        {"deck_id": deck_id, "front": "Q1", "back": "A1", "type": "basic"}
    )
    later = db.create_flashcard(
        {"deck_id": deck_id, "front": "Q2", "back": "A2", "type": "basic"}
    )
    # Two days ago: safely due, fixed, away from the midnight edge.
    base = (datetime.now(timezone.utc) - timedelta(days=2)).replace(
        hour=8, minute=0, second=0, microsecond=0
    )
    _set_next_review(db, earlier, _canonical(base))
    _set_next_review(db, later, _canonical(base.replace(hour=20)))

    first = db.get_due_flashcards(limit=1)
    assert [row["id"] for row in first] == [earlier]
    assert [row["id"] for row in db.get_due_flashcards()] == [earlier, later]
