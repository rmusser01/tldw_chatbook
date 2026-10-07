"""Sargable SQL trio with timestamp normalization (TASK-34427, ADR-224).

Three non-console query families wrap their ordering/filtering columns in
normalizing SQL functions (``julianday``/``datetime``) because the stored
values mix legacy shapes with the ADR-173 canonical shape:

* ``get_conversations_for_character`` -- keyset + ORDER BY ``julianday(last_modified)``.
* ``get_due_flashcards`` / ``count_due_flashcards`` -- WHERE/ORDER BY
  ``datetime(next_review)``.
* Character browse list -- ``json_extract`` visibility predicate plus
  ``ORDER BY name COLLATE NOCASE`` with no serving index.

ADR-224 migrates the two timestamp columns to the canonical shape
(``YYYY-MM-DDTHH:MM:SS.mmmZ``, fixed width, so raw TEXT ordering equals time
ordering) and adds the indexes the raw comparisons need. This module holds:

* 15a -- the format audit: a genuinely-v78 fixture with BOTH legacy shapes
  seeded, documenting that raw text order diverges from normalized order
  (the defect the migration exists to remove).
* 15b/15c -- migration correctness on the mixed fixture: all values canonical,
  idempotent re-run, ordering golden preserved, persistence across reopen,
  no spurious sync events, unparseable garbage untouched.
* 15d -- the character-card visibility/NOCASE index.
* 15e -- EXPLAIN QUERY PLAN assertions: every family is index-driven, with no
  full table scan and no post-sort where an index can serve the order.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version

#: ADR-173 canonical stored shape: YYYY-MM-DDTHH:MM:SS.mmmZ (fixed 24 chars).
CANONICAL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

#: SQLite ``CURRENT_TIMESTAMP`` shape (space separator, second precision) --
#: what legacy rows and the ``DEFAULT`` carry.
SPACE_TS = "2026-10-02 10:00:00"

#: The pre-TASK-32803.2 ``.isoformat()`` shape (T separator, +00:00 offset).
ISOFFSET_TS = "2026-10-01T09:30:00.350+00:00"

#: The ADR-173 canonical shape (same instant as ``ISOFFSET_TS``).
CANONICAL_TS = "2026-10-01T09:30:00.350Z"

V78 = 78  # schema version this branch shipped before ADR-224's migration


@contextmanager
def v78_db(tmp_path: Path, name: str) -> Iterator[CharactersRAGDB]:
    """Open a genuinely-v78 DB (real chain replay) at ``tmp_path/name``."""
    with chachanotes_db_at_version(tmp_path / name, V78, client_id="audit") as db:
        yield db


def seed_mixed_conversations(db: CharactersRAGDB) -> list[str]:
    """Seed conversations for character 1 in every stored format.

    The tie pair (``z-tie-space``/``a-tie-canonical``) shares one instant in
    two shapes, and the ``10-03`` pair puts a LATER space-separated value
    against an EARLIER canonical value: both expose raw-text ordering as
    chronologically wrong, which is the defect the audit documents.

    Returns the ids in seeding order.
    """
    rows = [
        # (id, last_modified) -- chronological DESC target:
        ("c-new", "2026-10-05T09:30:15.250Z"),  # newest
        ("z-tie-space", SPACE_TS),  # same instant as a-tie-canonical
        ("a-tie-canonical", "2026-10-02T10:00:00.000Z"),
        ("c-space-later", "2026-10-03 23:00:00"),  # later than c-canon-earlier
        ("a-canon-earlier", "2026-10-03T01:00:00.000Z"),
        ("c-oldest", "2026-09-30 12:00:00"),  # oldest
    ]
    with db.transaction() as cursor:
        for cid, last_modified in rows:
            cursor.execute(
                "INSERT INTO conversations "
                "(id, root_id, character_id, title, created_at, last_modified, client_id) "
                "VALUES (?, ?, 1, ?, '2026-09-01T00:00:00.000Z', ?, 'audit-seed')",
                (cid, cid, cid, last_modified),
            )
    return [cid for cid, _ in rows]


def seed_mixed_flashcards(db: CharactersRAGDB) -> dict[str, str | None]:
    """Seed flashcards for one deck in every stored ``next_review`` shape.

    Returns ``{card_id: next_review}`` in seeding order. The due pair
    (``due-space-later`` vs ``due-iso-earlier``) is ordered one way by
    ``datetime()`` and the other by raw text on a v78 DB.
    """
    deck_id = db.create_deck("Audit", "mixed-format audit deck")
    cards: dict[str, str | None] = {
        "due-null": None,  # never reviewed: due now
        "due-space-later": "2026-10-01 10:00:00",  # due; later instant
        "due-iso-earlier": "2026-10-01T09:30:00.350+00:00",  # due; earlier
        "future-space": "2027-01-01 00:00:00",  # not due
        "future-canon": "2027-01-02T00:00:00.000Z",  # not due
    }
    with db.transaction() as cursor:
        for cid, next_review in cards.items():
            cursor.execute(
                "INSERT INTO flashcards "
                "(id, deck_id, front, back, next_review, created_by, last_modified_by) "
                "VALUES (?, ?, ?, ?, ?, 'audit-seed', 'audit-seed')",
                (cid, deck_id, f"Q {cid}", f"A {cid}", next_review),
            )
    return cards


# ---------------------------------------------------------------------------
# 15a -- format audit (documents the split; no production change required)
# ---------------------------------------------------------------------------


class TestFormatAuditV78:
    """The stored-format split, documented on a genuinely-v78 database."""

    def test_conversations_last_modified_holds_both_legacy_shapes(
        self, tmp_path: Path
    ) -> None:
        with v78_db(tmp_path, "audit-conv.db") as db:
            seed_mixed_conversations(db)
            with db.transaction() as cursor:
                shapes = [
                    row[0]
                    for row in cursor.execute(
                        # CAST: the connection registers DATETIME converters,
                        # which would hand back datetime objects, not the
                        # stored text under audit.
                        "SELECT CAST(last_modified AS TEXT) FROM conversations "
                        "WHERE character_id = 1 ORDER BY id"
                    )
                ]
            canonical = [v for v in shapes if CANONICAL_RE.match(v)]
            space = [v for v in shapes if "T" not in v]
            assert canonical, "fixture must contain canonical-shape rows"
            assert space, (
                "fixture must contain space-separated legacy rows "
                "(SQLite CURRENT_TIMESTAMP shape)"
            )

    def test_raw_text_ordering_diverges_from_julianday_ordering(
        self, tmp_path: Path
    ) -> None:
        """The defect: on mixed data, ``ORDER BY last_modified`` is not
        chronological, so the production query must normalize -- which is
        exactly what defeats the index."""
        with v78_db(tmp_path, "audit-order.db") as db:
            seed_mixed_conversations(db)
            with db.transaction() as cursor:
                raw = [
                    row[0]
                    for row in cursor.execute(
                        "SELECT id FROM conversations WHERE character_id = 1 "
                        "AND deleted = 0 AND scope_type = 'global' AND archived = 0 "
                        "ORDER BY last_modified DESC, id DESC"
                    )
                ]
                normalized = [
                    row[0]
                    for row in cursor.execute(
                        "SELECT id FROM conversations WHERE character_id = 1 "
                        "AND deleted = 0 AND scope_type = 'global' AND archived = 0 "
                        "ORDER BY julianday(last_modified) DESC, id DESC"
                    )
                ]
            assert raw != normalized, (
                "fixture failed to expose the split: raw and julianday order "
                "agreed; make the tie/same-date pairs diverge"
            )
            # The chronological truth the migration must preserve:
            # 10-05, then the 10-03 pair (23:00 before 01:00 -- raw text
            # flips these two), then the equal-instant 10-02 tie broken by
            # id DESC, then the oldest.
            assert normalized == [
                "c-new",
                "c-space-later",
                "a-canon-earlier",
                "z-tie-space",
                "a-tie-canonical",
                "c-oldest",
            ]

    def test_flashcards_next_review_holds_every_legacy_shape(
        self, tmp_path: Path
    ) -> None:
        with v78_db(tmp_path, "audit-cards.db") as db:
            cards = seed_mixed_flashcards(db)
            with db.transaction() as cursor:
                stored = {
                    row[0]: row[1]
                    for row in cursor.execute(
                        "SELECT id, CAST(next_review AS TEXT) FROM flashcards"
                    )
                }
            assert stored == cards
            assert stored["due-iso-earlier"] == ISOFFSET_TS
            assert "T" not in stored["due-space-later"]
            assert stored["due-null"] is None

    def test_flashcards_raw_ordering_diverges_from_datetime_ordering(
        self, tmp_path: Path
    ) -> None:
        with v78_db(tmp_path, "audit-cards-order.db") as db:
            seed_mixed_flashcards(db)
            with db.transaction() as cursor:
                raw = [
                    row[0]
                    for row in cursor.execute(
                        "SELECT id FROM flashcards "
                        "WHERE next_review IS NOT NULL ORDER BY next_review ASC"
                    )
                ]
                normalized = [
                    row[0]
                    for row in cursor.execute(
                        "SELECT id FROM flashcards "
                        "WHERE next_review IS NOT NULL "
                        "ORDER BY datetime(next_review) ASC"
                    )
                ]
            assert raw != normalized
            assert normalized.index("due-iso-earlier") < normalized.index(
                "due-space-later"
            )
            assert raw.index("due-space-later") < raw.index("due-iso-earlier")

    def test_conversation_app_writer_already_emits_canonical(
        self, tmp_path: Path
    ) -> None:
        """Writer audit: ``add_conversation`` stamps the canonical shape, so
        new rows cannot regress the mix from the write side."""
        with v78_db(tmp_path, "audit-writer.db") as db:
            conv_id = db.add_conversation(
                {"character_id": 1, "title": "Audit"}
            )
            with db.transaction() as cursor:
                stored = cursor.execute(
                    "SELECT CAST(last_modified AS TEXT) FROM conversations WHERE id = ?",
                    (conv_id,),
                ).fetchone()[0]
            assert CANONICAL_RE.match(stored), (
                f"add_conversation wrote a non-canonical timestamp: {stored!r}"
            )
