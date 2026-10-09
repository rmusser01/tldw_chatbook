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
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB import ChaChaNotes_DB as _chachanotes_module
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    _split_sql_statements,
)

#: The v78→v79 migration file under test (ADR-224).
MIGRATION_V79_PATH = (
    Path(_chachanotes_module.__file__).parent
    / "migrations"
    / "chachanotes_v78_to_v79_sargable_timestamp_normalization.sql"
)

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
            conv_id = db.add_conversation({"character_id": 1, "title": "Audit"})
            with db.transaction() as cursor:
                stored = cursor.execute(
                    "SELECT CAST(last_modified AS TEXT) FROM conversations WHERE id = ?",
                    (conv_id,),
                ).fetchone()[0]
            assert CANONICAL_RE.match(stored), (
                f"add_conversation wrote a non-canonical timestamp: {stored!r}"
            )


def seed_mixed_v78_db(tmp_path: Path, name: str = "mixed-v78.db") -> dict[str, Any]:
    """Build a genuinely-v78 DB holding every legacy shape, then close it.

    Reopening ``path`` with an unpatched ``CharactersRAGDB`` replays the
    v78→v79 migration (ADR-224). The returned "golden" orderings are captured
    through the pre-migration normalizing expressions -- the chronology the
    migration must preserve.
    """
    path = tmp_path / name
    with chachanotes_db_at_version(path, V78, client_id="audit") as db:
        seed_mixed_conversations(db)
        seed_mixed_flashcards(db)
        with db.transaction() as cursor:
            conv_golden = [
                row[0]
                for row in cursor.execute(
                    "SELECT id FROM conversations WHERE character_id = 1 "
                    "AND deleted = 0 AND scope_type = 'global' AND archived = 0 "
                    "ORDER BY julianday(last_modified) DESC, id DESC"
                )
            ]
            card_golden = [
                row[0]
                for row in cursor.execute(
                    "SELECT id FROM flashcards WHERE next_review IS NOT NULL "
                    "ORDER BY datetime(next_review) ASC"
                )
            ]
            sync_events = cursor.execute(
                "SELECT COUNT(*) FROM sync_log WHERE entity = 'conversations'"
            ).fetchone()[0]
    return {
        "path": path,
        "conv_golden": conv_golden,
        "card_golden": card_golden,
        "sync_events": sync_events,
    }


def reopen_migrated(facts: dict[str, Any]) -> CharactersRAGDB:
    """Open the seeded v78 fixture with current code (runs the migration)."""
    return CharactersRAGDB(str(facts["path"]), client_id="audit")


# ---------------------------------------------------------------------------
# 15b/15c -- the v78 -> v79 normalization migration on mixed-format data
# ---------------------------------------------------------------------------


class TestV78ToV79Migration:
    def test_conversations_last_modified_all_canonical_after_migration(
        self, tmp_path: Path
    ) -> None:
        facts = seed_mixed_v78_db(tmp_path)
        db = reopen_migrated(facts)
        try:
            with db.transaction() as cursor:
                values = [
                    row[0]
                    for row in cursor.execute(
                        "SELECT CAST(last_modified AS TEXT) FROM conversations "
                        "WHERE character_id = 1"
                    )
                ]
        finally:
            db.close_connection()
        assert values and all(CANONICAL_RE.match(v) for v in values), values
        # Every legacy instant survived: the same-instant tie pair now holds
        # byte-identical canonical values.
        assert values.count("2026-10-02T10:00:00.000Z") == 2

    def test_flashcards_next_review_all_canonical_after_migration(
        self, tmp_path: Path
    ) -> None:
        facts = seed_mixed_v78_db(tmp_path)
        db = reopen_migrated(facts)
        try:
            with db.transaction() as cursor:
                stored = {
                    row[0]: row[1]
                    for row in cursor.execute(
                        "SELECT id, CAST(next_review AS TEXT) FROM flashcards"
                    )
                }
        finally:
            db.close_connection()
        assert stored["due-null"] is None, "NULL (= due now) must stay NULL"
        assert stored["due-space-later"] == "2026-10-01T10:00:00.000Z"
        # The +00:00-offset legacy shape keeps its millisecond instant.
        assert stored["due-iso-earlier"] == "2026-10-01T09:30:00.350Z"
        assert stored["future-space"] == "2027-01-01T00:00:00.000Z"
        assert all(v is None or CANONICAL_RE.match(v) for v in stored.values()), stored

    def test_schema_version_is_79_after_migration(self, tmp_path: Path) -> None:
        facts = seed_mixed_v78_db(tmp_path)
        db = reopen_migrated(facts)
        try:
            assert (
                db._get_db_version(db.get_connection())
                == CharactersRAGDB._CURRENT_SCHEMA_VERSION
                == 79
            )
        finally:
            db.close_connection()

    def test_fresh_install_reaches_79_with_the_new_indexes(
        self, tmp_path: Path
    ) -> None:
        db = CharactersRAGDB(":memory:", "fresh-79")
        try:
            conn = db.get_connection()
            assert db._get_db_version(conn) == 79
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM pragma_index_list('conversations')"
                )
            } | {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM pragma_index_list('character_cards')"
                )
            }
            assert "idx_conv_char_lm" in names
            assert "idx_character_cards_visible_name" in names
        finally:
            db.close_connection()

    def test_conversation_ordering_golden_preserved(self, tmp_path: Path) -> None:
        facts = seed_mixed_v78_db(tmp_path)
        db = reopen_migrated(facts)
        try:
            rows = db.get_conversations_for_character(1, limit=10)
        finally:
            db.close_connection()
        assert [row["id"] for row in rows] == facts["conv_golden"]

    def test_flashcard_due_set_and_ordering_golden_preserved(
        self, tmp_path: Path
    ) -> None:
        facts = seed_mixed_v78_db(tmp_path)
        db = reopen_migrated(facts)
        try:
            due = db.get_due_flashcards(limit=10)
            count = db.count_due_flashcards()
        finally:
            db.close_connection()
        # NULL first (never-reviewed = due now), then chronological.
        assert [row["id"] for row in due] == [
            "due-null",
            "due-iso-earlier",
            "due-space-later",
        ]
        assert count == 3

    def test_keyset_pagination_agrees_with_offset_pagination_on_migrated_data(
        self, tmp_path: Path
    ) -> None:
        facts = seed_mixed_v78_db(tmp_path)
        db = reopen_migrated(facts)
        try:
            offset_pages: list[list[dict[str, Any]]] = [
                db.get_conversations_for_character(1, limit=2, offset=o)
                for o in (0, 2, 4)
            ]
            cursor_page = db.get_conversations_for_character(1, limit=2)
            second = db.get_conversations_for_character(
                1,
                limit=2,
                before_last_modified=cursor_page[-1]["last_modified"],
                before_id=cursor_page[-1]["id"],
            )
            # A cursor handed back as the SQLite-returned DATETIME object
            # (the personas controller pattern) must page identically.
            third_via_datetime = db.get_conversations_for_character(
                1,
                limit=2,
                before_last_modified=second[-1]["last_modified"],
                before_id=second[-1]["id"],
            )
        finally:
            db.close_connection()
        assert [r["id"] for r in second] == [r["id"] for r in offset_pages[1]]
        assert [r["id"] for r in third_via_datetime] == [
            r["id"] for r in offset_pages[2]
        ]

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        facts = seed_mixed_v78_db(tmp_path)
        db = reopen_migrated(facts)
        try:
            conn = db.get_connection()
            # Sorted-line equivalence, not literal dump equality: DROP +
            # CREATE of the sync trigger legitimately REORDERS sqlite_master
            # rows (creation order), which iterdump preserves; the DDL texts
            # and all data must be identical.
            before = sorted("\n".join(conn.iterdump()).splitlines())
            # Replay the file's statements directly on the migrated DB: the
            # guarded UPDATEs touch zero rows and the IF NOT EXISTS creates
            # are no-ops, so nothing may change.
            for statement in _split_sql_statements(
                MIGRATION_V79_PATH.read_text(encoding="utf-8")
            ):
                conn.execute(statement)
            conn.commit()
            after = sorted("\n".join(conn.iterdump()).splitlines())
        finally:
            db.close_connection()
        assert before == after

    def test_migration_persists_across_reopen(self, tmp_path: Path) -> None:
        facts = seed_mixed_v78_db(tmp_path)
        db = reopen_migrated(facts)
        db.close_connection()
        db = CharactersRAGDB(str(facts["path"]), client_id="audit")
        try:
            assert db._get_db_version(db.get_connection()) == 79
            rows = db.get_conversations_for_character(1, limit=10)
            assert [row["id"] for row in rows] == facts["conv_golden"]
        finally:
            db.close_connection()

    def test_migration_emits_no_spurious_sync_events(self, tmp_path: Path) -> None:
        """A storage-format change is not a content change: the
        normalization UPDATE must not enqueue per-conversation sync events
        (the v70→v71 drop/recreate trigger precedent)."""
        facts = seed_mixed_v78_db(tmp_path)
        db = reopen_migrated(facts)
        try:
            with db.transaction() as cursor:
                after = cursor.execute(
                    "SELECT COUNT(*) FROM sync_log WHERE entity = 'conversations'"
                ).fetchone()[0]
        finally:
            db.close_connection()
        assert after == facts["sync_events"]

    def test_migration_does_not_dirty_the_character_search_projection(
        self, tmp_path: Path
    ) -> None:
        """Same rule for the search projection: no revision bump, no dirty
        rows, no CURRENT_TIMESTAMP stamps -- a format-only change must be
        deterministic and must not invalidate searchable content."""
        facts = seed_mixed_v78_db(tmp_path)
        revision_before = dirty_before = None
        with sqlite3.connect(str(facts["path"])) as raw:
            revision_before = raw.execute(
                "SELECT data_revision, updated_at FROM character_conversation_search_revision"
            ).fetchall()
            dirty_before = raw.execute(
                "SELECT COUNT(*) FROM character_conversation_search_dirty"
            ).fetchone()[0]
        db = reopen_migrated(facts)
        try:
            with db.transaction() as cursor:
                revision_after = [
                    tuple(row)
                    for row in cursor.execute(
                        "SELECT data_revision, updated_at FROM character_conversation_search_revision"
                    )
                ]
                dirty_after = cursor.execute(
                    "SELECT COUNT(*) FROM character_conversation_search_dirty"
                ).fetchone()[0]
        finally:
            db.close_connection()
        assert revision_after == revision_before
        assert dirty_after == dirty_before == 0

    def test_unparseable_values_are_left_untouched(self, tmp_path: Path) -> None:
        path = tmp_path / "garbage.db"
        with chachanotes_db_at_version(path, V78, client_id="audit") as db:
            seed_mixed_conversations(db)
            with db.transaction() as cursor:
                cursor.execute(
                    "UPDATE conversations SET last_modified = 'not-a-timestamp' "
                    "WHERE id = 'c-oldest'"
                )
        db = CharactersRAGDB(str(path), client_id="audit")
        try:
            with db.transaction() as cursor:
                stored = cursor.execute(
                    "SELECT CAST(last_modified AS TEXT) FROM conversations "
                    "WHERE id = 'c-oldest'"
                ).fetchone()[0]
        finally:
            db.close_connection()
        # Documented residual risk (ADR-224): garbage is never silently
        # NULLed; it is reported, not rewritten.
        assert stored == "not-a-timestamp"

    def test_seek_cursor_as_datetime_object_pages_correctly(
        self, tmp_path: Path
    ) -> None:
        """The docstring contract: naive/aware datetime cursors are adapted
        as UTC to the canonical shape before the raw comparison."""
        from datetime import datetime

        facts = seed_mixed_v78_db(tmp_path)
        db = reopen_migrated(facts)
        try:
            aware = datetime(2026, 10, 3, 23, 0, 0, tzinfo=UTC)
            rows = db.get_conversations_for_character(
                1, limit=10, before_last_modified=aware, before_id="c-space-later"
            )
        finally:
            db.close_connection()
        assert [row["id"] for row in rows] == [
            "a-canon-earlier",
            "z-tie-space",
            "a-tie-canonical",
            "c-oldest",
        ]

    def test_unparseable_seek_cursor_text_is_rejected(self) -> None:
        db = CharactersRAGDB(":memory:", "cursor-validation")
        try:
            with pytest.raises(Exception, match="parseable timestamp"):
                db.get_conversations_for_character(
                    1, before_last_modified="not-a-timestamp", before_id="x"
                )
        finally:
            db.close_connection()


# ---------------------------------------------------------------------------
# 15d -- character cards: visibility + NOCASE index (result equivalence)
# ---------------------------------------------------------------------------


class TestCharacterCardsBrowseIndex:
    def _seeded(self) -> CharactersRAGDB:
        db = CharactersRAGDB(":memory:", "cards-browse")
        db.add_character_card({"name": "zeta", "description": "lower"})
        db.add_character_card({"name": "Apple", "description": "upper"})
        db.add_character_card({"name": "banana", "description": "lower"})
        db.add_character_card({"name": "Cherry", "description": "upper"})
        with db.transaction() as cursor:
            # An app-owned Actor Pack Persona portrait card: invisible to
            # user browse lists (_USER_VISIBLE_CHARACTER), plus a deleted
            # card. Both are planted via SQL because the public writer
            # rejects the reserved marker and the deleter bumps versions.
            cursor.execute(
                "UPDATE character_cards SET extensions = ? WHERE name = 'Cherry'",
                ('{"actor_pack_persona_portrait_owner": "pack-1"}',),
            )
            cursor.execute(
                "UPDATE character_cards SET deleted = 1 WHERE name = 'banana'"
            )
        return db

    def test_browse_page_orders_nocase_and_excludes_invisible_and_deleted(
        self,
    ) -> None:
        db = self._seeded()
        try:
            page = db.list_character_cards_page(limit=10, offset=0)
            # "Default Assistant" is the seeded user-visible card every
            # fresh database carries; NOCASE places it between Apple/zeta.
            assert [card["name"] for card in page] == [
                "Apple",
                "Default Assistant",
                "zeta",
            ]
            assert db.count_character_cards() == 3
        finally:
            db.close_connection()

    def test_visibility_expression_index_shape_is_registered(self) -> None:
        """Feature-detection (ADR-224): the shipped SQLite accepts the
        deterministic json_extract in a partial-index WHERE; the index
        exists with exactly that predicate."""
        db = CharactersRAGDB(":memory:", "cards-index-shape")
        try:
            row = (
                db.get_connection()
                .execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'index' "
                    "AND name = 'idx_character_cards_visible_name'"
                )
                .fetchone()
            )
        finally:
            db.close_connection()
        assert row is not None, (
            "partial expression index was rejected by this SQLite build; "
            "the ADR-224 stored-column fallback would be required"
        )
        sql = row[0]
        assert "COLLATE NOCASE" in sql
        assert "json_extract" in sql
        assert "actor_pack_persona_portrait_owner" in sql


# ---------------------------------------------------------------------------
# 15e -- EXPLAIN QUERY PLAN: index-driven, no full scans, no avoidable sort
# ---------------------------------------------------------------------------


def _plan(db: CharactersRAGDB, sql: str, params: tuple = ()) -> str:
    """EXPLAIN QUERY PLAN detail lines joined for substring assertions."""
    rows = db.get_connection().execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
    return "\n".join(str(row[3]) for row in rows)


def _assert_no_table_scan(plan: str, table: str) -> None:
    """Every plan line touching ``table`` must go through an index.

    ``SCAN <table> USING INDEX ...`` (an ordered index walk) passes; a bare
    ``SCAN <table>`` (full table scan) fails.
    """
    lines = [
        line
        for line in plan.splitlines()
        if re.search(rf"\b{re.escape(table)}\b", line)
    ]
    assert lines, f"no plan line mentions table {table!r}: {plan!r}"
    for line in lines:
        assert "USING" in line, f"full table scan on {table}: {line!r}"


class TestExplainQueryPlans:
    """Each family's production statement is index-driven.

    The assertions deliberately target the statements the methods execute
    (class-level SQL constants / named builder methods), per the
    ``_BACKLINK_SOURCES_SQL`` precedent.
    """

    @pytest.fixture()
    def db(self) -> Iterator[CharactersRAGDB]:
        database = CharactersRAGDB(":memory:", "explain-budget")
        seed_mixed_conversations(database)
        seed_mixed_flashcards(database)
        database.add_character_card({"name": "explain", "description": "d"})
        assert (
            database.get_connection()
            .execute("SELECT 1 FROM sqlite_schema WHERE name = 'sqlite_stat1'")
            .fetchone()
            is None
        )
        yield database
        database.close_connection()

    def test_conversation_first_page_is_index_driven(self, db) -> None:
        sql = CharactersRAGDB._CONVERSATIONS_FOR_CHARACTER_PAGE_SQL.format(
            archive="archived = 0"
        )
        plan = _plan(db, sql, (1, 50, 0))
        assert "SEARCH conversations USING INDEX" in plan
        _assert_no_table_scan(plan, "conversations")
        # Whichever composite wins -- idx_conv_char_lm (character-first,
        # ADR-224) or the pre-existing idx_conversations_archive (its
        # archived/deleted equality prefix leaves last_modified DESC,
        # id DESC serving the order) -- the (last_modified, id) order is
        # delivered by the index, never a sorter.
        assert "idx_conv_char_lm" in plan or "idx_conversations_archive" in plan, plan
        assert "USE TEMP B-TREE" not in plan

    def test_conversation_first_page_all_scope_uses_char_composite(self, db) -> None:
        """The archive index cannot serve order without its archived
        equality; the character-first composite carries the 'all' scope."""
        sql = CharactersRAGDB._CONVERSATIONS_FOR_CHARACTER_PAGE_SQL.format(
            archive="1 = 1"
        )
        plan = _plan(db, sql, (1, 50, 0))
        assert "idx_conv_char_lm" in plan
        _assert_no_table_scan(plan, "conversations")
        assert "USE TEMP B-TREE" not in plan

    def test_conversation_keyset_page_is_index_driven(self, db) -> None:
        sql = CharactersRAGDB._CONVERSATIONS_FOR_CHARACTER_KEYSET_SQL.format(
            archive="archived = 0"
        )
        plan = _plan(
            db,
            sql,
            (1, "2026-10-05T09:30:15.250Z", "2026-10-05T09:30:15.250Z", "c-new", 50),
        )
        assert "conversations USING INDEX" in plan
        _assert_no_table_scan(plan, "conversations")
        assert "USE TEMP B-TREE" not in plan

    def test_due_flashcards_use_the_next_review_index(self, db) -> None:
        sql = CharactersRAGDB._DUE_FLASHCARDS_SQL + " LIMIT ?"
        plan = _plan(db, sql, ("2027-06-01T00:00:00.000Z", 20))
        # Either an ordered index walk with early stop (SCAN ... USING
        # INDEX, LIMIT-aware) or index probes; the pre-ADR-224 plan was a
        # full table scan plus a sorter.
        assert "USING INDEX idx_flashcards_next_review" in plan
        _assert_no_table_scan(plan, "f")
        assert "USE TEMP B-TREE" not in plan

    def test_count_due_flashcards_use_the_next_review_index(self, db) -> None:
        plan = _plan(
            db,
            CharactersRAGDB._COUNT_DUE_FLASHCARDS_SQL,
            ("2027-06-01T00:00:00.000Z",),
        )
        assert "SEARCH f USING INDEX idx_flashcards_next_review" in plan
        _assert_no_table_scan(plan, "f")

    def test_character_browse_page_uses_visibility_name_index(self, db) -> None:
        sql = db._character_cards_browse_page_sql(
            order_by="name_asc", include_image=False
        )
        plan = _plan(db, sql, (10, 0))
        assert "USING INDEX idx_character_cards_visible_name" in plan
        _assert_no_table_scan(plan, "character_cards")
        # NOCASE order is served by the index itself.
        assert "USE TEMP B-TREE" not in plan

    def test_character_browse_count_uses_visibility_index(self, db) -> None:
        plan = _plan(db, db._character_cards_browse_count_sql())
        assert "USING INDEX idx_character_cards_visible_name" in plan
        _assert_no_table_scan(plan, "character_cards")
