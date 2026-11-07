"""Learning persistence (flashcards, decks, quizzes, attempts, study stats).

TASK-36949.2: moved out of ``CharactersRAGDB`` as a repository that runs on the
database's own connection and joins its managed ``transaction()``; it opens no
connection and no transaction of its own. Reach it through ``db.learning``.
Method bodies are unchanged apart from ``self.<db member>`` -> ``self.db.<member>``.
"""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
from datetime import timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_chatbook.Utils.fts5_match_forms import build_and_match_query
from tldw_chatbook.Utils.timestamps import to_utc_iso


class LearningRepository:
    """Flashcard, quiz and study-stat persistence over one ``CharactersRAGDB``."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db

    # --- Study Methods (Learning Paths, Flashcards, Mindmaps) ---

    def _recount_deck_card_count(self, cursor, deck_id: str) -> None:
        """Refresh the denormalized card_count for a deck from active child cards."""
        row = cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM flashcards
            WHERE deck_id = ? AND is_deleted = 0
            """,
            (deck_id,),
        ).fetchone()
        total = int(row["total"]) if row else 0
        cursor.execute(
            "UPDATE decks SET card_count = ? WHERE id = ?",
            (total, deck_id),
        )

    def _ensure_flashcard_fts_triggers(self, cursor) -> None:
        """Repair legacy FTS5 triggers before local flashcard mutations."""
        cursor.execute("DROP TRIGGER IF EXISTS flashcards_ad")
        cursor.execute("DROP TRIGGER IF EXISTS flashcards_au")
        cursor.execute(
            """
            CREATE TRIGGER flashcards_ad AFTER DELETE ON flashcards BEGIN
                INSERT INTO flashcards_fts(flashcards_fts, rowid, front, back, tags)
                VALUES('delete', old.rowid, old.front, old.back, old.tags);
            END
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER flashcards_au AFTER UPDATE ON flashcards BEGIN
                INSERT INTO flashcards_fts(flashcards_fts, rowid, front, back, tags)
                VALUES('delete', old.rowid, old.front, old.back, old.tags);
                INSERT INTO flashcards_fts(rowid, front, back, tags)
                VALUES (new.rowid, new.front, new.back, new.tags);
            END
            """
        )

    def create_flashcard(self, card_data: Dict[str, Any]) -> str:
        """
        Create a new flashcard.

        Args:
            card_data: Dictionary containing flashcard data with keys:
                - deck_id: ID of the deck this card belongs to
                - front: Front text of the card (question)
                - back: Back text of the card (answer)
                - tags: Optional space-separated tags
                - type: Card type (default: 'basic')
                - metadata: Optional JSON metadata

        Returns:
            The ID of the created flashcard.

        Raises:
            InputError: If required fields are missing or invalid.
            CharactersRAGDBError: If database operation fails.
        """
        deck_id = str(card_data.get("deck_id") or "").strip()
        if not deck_id:
            raise InputError("Flashcard must belong to a deck")
        if not card_data.get("front") or not card_data.get("back"):
            raise InputError("Flashcard must have both front and back text")

        card_id = self.db._generate_uuid()

        with self.db.transaction() as cursor:
            deck_row = cursor.execute(
                "SELECT id FROM decks WHERE id = ? AND is_deleted = 0",
                (deck_id,),
            ).fetchone()
            if not deck_row:
                raise InputError(f"Deck {deck_id} not found")

            cursor.execute(
                """
                INSERT INTO flashcards (
                    id, deck_id, front, back, tags, type,
                    created_by, last_modified_by, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    card_id,
                    deck_id,
                    card_data["front"],
                    card_data["back"],
                    card_data.get("tags", ""),
                    card_data.get("type", "basic"),
                    card_data.get("created_by", self.db.client_id),
                    card_data.get("last_modified_by", self.db.client_id),
                    json.dumps(card_data.get("metadata", {}))
                    if card_data.get("metadata")
                    else None,
                ),
            )
            self._recount_deck_card_count(cursor, deck_id)
        return card_id

    def get_flashcard(self, card_id: str) -> Optional[Dict[str, Any]]:
        """Get a flashcard by ID."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM flashcards 
            WHERE id = ? AND is_deleted = 0
        """,
            (card_id,),
        )

        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    @staticmethod
    def _normalize_flashcard_tags(tags: Any) -> str:
        """Normalize list/string tag payloads into the local space-separated format."""
        if isinstance(tags, str):
            raw_tags = tags.split()
        elif isinstance(tags, (list, tuple, set)):
            raw_tags = [str(item).strip() for item in tags]
        else:
            raw_tags = [str(tags).strip()]

        seen: set[str] = set()
        normalized: list[str] = []
        for tag in raw_tags:
            cleaned = str(tag or "").strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                normalized.append(cleaned)
        return " ".join(normalized)

    def update_flashcard(
        self,
        card_id: str,
        *,
        deck_id: Optional[str] = None,
        front: Optional[str] = None,
        back: Optional[str] = None,
        tags: Any = None,
        card_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        extra: Optional[str] = None,
        expected_version: Optional[int] = None,
    ) -> bool:
        """Update local flashcard content/metadata with optional optimistic locking."""
        try:
            with self.db.transaction() as cursor:
                row = cursor.execute(
                    """
                    SELECT deck_id, version, is_deleted, metadata
                    FROM flashcards
                    WHERE id = ?
                    """,
                    (card_id,),
                ).fetchone()
                if not row or int(row["is_deleted"]):
                    return False

                current_version = int(row["version"])
                if expected_version is not None and current_version != expected_version:
                    raise ConflictError(
                        "Version mismatch updating flashcard",
                        entity="flashcards",
                        entity_id=card_id,
                    )

                source_deck_id = row["deck_id"]
                updates: list[str] = []
                params: list[Any] = []

                if deck_id is not None:
                    target_deck_id = str(deck_id).strip()
                    if not target_deck_id:
                        raise InputError("Flashcard deck cannot be blank")
                    target_row = cursor.execute(
                        "SELECT id FROM decks WHERE id = ? AND is_deleted = 0",
                        (target_deck_id,),
                    ).fetchone()
                    if not target_row:
                        raise InputError(f"Target deck {target_deck_id} not found")
                    updates.append("deck_id = ?")
                    params.append(target_deck_id)

                if front is not None:
                    normalized_front = str(front).strip()
                    if not normalized_front:
                        raise InputError("Flashcard front cannot be blank")
                    updates.append("front = ?")
                    params.append(normalized_front)

                if back is not None:
                    normalized_back = str(back).strip()
                    if not normalized_back:
                        raise InputError("Flashcard back cannot be blank")
                    updates.append("back = ?")
                    params.append(normalized_back)

                if tags is not None:
                    updates.append("tags = ?")
                    params.append(self._normalize_flashcard_tags(tags))

                if card_type is not None:
                    normalized_type = str(card_type).strip() or "basic"
                    updates.append("type = ?")
                    params.append(normalized_type)

                metadata_update_requested = (
                    metadata is not None or notes is not None or extra is not None
                )
                if metadata_update_requested:
                    existing_metadata: dict[str, Any] = {}
                    if row["metadata"]:
                        with contextlib.suppress(json.JSONDecodeError, TypeError):
                            parsed = json.loads(row["metadata"])
                            if isinstance(parsed, dict):
                                existing_metadata = parsed
                    if metadata:
                        existing_metadata.update(metadata)
                    if notes is not None:
                        existing_metadata["notes"] = notes
                    if extra is not None:
                        existing_metadata["extra"] = extra
                    updates.append("metadata = ?")
                    params.append(
                        json.dumps(existing_metadata) if existing_metadata else None
                    )

                if not updates:
                    return True

                updates.extend(
                    [
                        "updated_at = CURRENT_TIMESTAMP",
                        "version = version + 1",
                        "last_modified_by = ?",
                    ]
                )
                params.append(self.db.client_id)
                params.append(card_id)
                cursor.execute(
                    f"UPDATE flashcards SET {', '.join(updates)} WHERE id = ? AND is_deleted = 0",
                    tuple(params),
                )

                if deck_id is not None and source_deck_id != str(deck_id).strip():
                    if source_deck_id:
                        self._recount_deck_card_count(cursor, source_deck_id)
                    self._recount_deck_card_count(cursor, str(deck_id).strip())
                return bool(cursor.rowcount)
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to update flashcard: {e}") from e

    def update_flashcard_review(self, card_id: str, rating: int) -> None:
        """
        Update flashcard after review using SM-2 algorithm.

        Args:
            card_id: ID of the flashcard
            rating: Review rating (0-5)
                0 - Complete blackout
                1 - Incorrect response; correct one remembered
                2 - Incorrect response; correct one seemed easy to recall
                3 - Correct response; with difficulty
                4 - Correct response; after hesitation
                5 - Perfect response
        """
        if rating < 0 or rating > 5:
            raise InputError("Rating must be between 0 and 5")

        with self.db.transaction() as cursor:
            # Get current card data
            cursor.execute(
                """
                SELECT interval, repetitions, ease_factor
                FROM flashcards
                WHERE id = ? AND is_deleted = 0
            """,
                (card_id,),
            )

            row = cursor.fetchone()
            if not row:
                raise InputError(f"Flashcard {card_id} not found")

            interval = row["interval"]
            repetitions = row["repetitions"]
            ease_factor = row["ease_factor"]

            # SM-2 algorithm
            if rating < 3:
                repetitions = 0
                interval = 1
            else:
                if repetitions == 0:
                    interval = 1
                elif repetitions == 1:
                    interval = 6
                else:
                    interval = int(interval * ease_factor)

                repetitions += 1

            # Update ease factor
            ease_factor = (
                ease_factor + 0.1 - (5 - rating) * (0.08 + (5 - rating) * 0.02)
            )
            if ease_factor < 1.3:
                ease_factor = 1.3

            # Calculate next review date
            from datetime import datetime, timedelta

            next_review = datetime.now(timezone.utc) + timedelta(days=interval)

            # Update flashcard
            cursor.execute(
                """
                UPDATE flashcards
                SET interval = ?, repetitions = ?, ease_factor = ?,
                    next_review = ?, last_review = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP, version = version + 1
                WHERE id = ?
            """,
                (
                    interval,
                    repetitions,
                    ease_factor,
                    # ADR-224 (superseding TASK-32803.2's write shape): store
                    # the package-canonical UTC shape
                    # (YYYY-MM-DDTHH:MM:SS.mmmZ). 32803.2 matched the legacy
                    # space-separated rows so the lexical `<=` in the due
                    # queries was a real time comparison; the v78→v79
                    # migration normalized those rows, so the canonical
                    # shape now provides that guarantee index-sargably.
                    # last_review/updated_at stay CURRENT_TIMESTAMP: they
                    # are payload-only (never filtered or ordered on).
                    to_utc_iso(next_review),
                    card_id,
                ),
            )

            # Add review history
            review_id = self.db._generate_uuid()
            cursor.execute(
                """
                INSERT INTO review_history (
                    id, flashcard_id, rating, interval_after, ease_after
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (review_id, card_id, rating, interval, ease_factor),
            )

    #: The due-card queries, ADR-224 style: RAW ``next_review`` comparison
    #: against a Python-produced canonical now (``_get_current_utc_timestamp_iso``)
    #: over the v78→v79-normalized column, so ``idx_flashcards_next_review``
    #: serves the range probes instead of a full-table ``datetime()`` scan.
    #: NULL means "never reviewed" and sorts due-first, exactly as the old
    #: ``datetime()`` expression ordered it. Named for the EXPLAIN QUERY PLAN
    #: assertions in ``Tests/ChaChaNotesDB/test_sargable_timestamps.py``.
    _DUE_FLASHCARDS_WHERE_SQL = (
        "SELECT f.* FROM flashcards f "
        "JOIN decks d ON d.id = f.deck_id "
        "WHERE f.is_deleted = 0 AND f.is_suspended = 0 "
        "AND d.is_deleted = 0 "
        "AND (f.next_review IS NULL OR f.next_review <= ?)"
    )
    _DUE_FLASHCARDS_SQL = _DUE_FLASHCARDS_WHERE_SQL + " ORDER BY f.next_review ASC"
    _COUNT_DUE_FLASHCARDS_SQL = (
        "SELECT COUNT(*) AS cnt FROM flashcards f "
        "JOIN decks d ON d.id = f.deck_id "
        "WHERE f.is_deleted = 0 AND f.is_suspended = 0 AND d.is_deleted = 0 "
        "AND (f.next_review IS NULL OR f.next_review <= ?)"
    )

    def get_due_flashcards(
        self, deck_id: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get flashcards due for review."""
        query = self._DUE_FLASHCARDS_WHERE_SQL
        params: List[Any] = [self.db._get_current_utc_timestamp_iso()]

        if deck_id:
            query += " AND f.deck_id = ?"
            params.append(deck_id)

        # TASK-32803.2 / Qodo #5 kept chronological order by sorting the
        # NORMALIZED time; ADR-224 makes raw order chronological (the column
        # is uniformly canonical since the v78→v79 migration), preserving
        # that guarantee for get_next_review_candidate (limit=1).
        query += " ORDER BY f.next_review ASC LIMIT ?"
        params.append(limit)

        # Qodo #3: use the standard read path like count_due_flashcards.
        cursor = self.db.execute_query(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    def count_due_flashcards(self) -> int:
        """Count flashcards due for review.

        Returns:
            The number of non-deleted, non-suspended flashcards in
            non-deleted decks whose ``next_review`` is unset or has
            passed -- the exact total the Library rail badge displays.
            Uses the identical WHERE clause as ``get_due_flashcards``.
        """
        query = self._COUNT_DUE_FLASHCARDS_SQL
        params: Tuple[Any, ...] = (self.db._get_current_utc_timestamp_iso(),)
        cursor = self.db.execute_query(query, params)
        row = cursor.fetchone()
        return int(row["cnt"] if row else 0)

    def get_deck(self, deck_id: str) -> Optional[Dict[str, Any]]:
        """Get a flashcard deck by ID."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM decks
            WHERE id = ? AND is_deleted = 0
            """,
            (deck_id,),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def list_decks(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List non-deleted flashcard decks for explicit deck selection flows."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM decks
            WHERE is_deleted = 0
            ORDER BY name COLLATE NOCASE ASC, created_at ASC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        return [dict(row) for row in cursor.fetchall()]

    def count_decks(self) -> int:
        """Count all non-deleted flashcard decks.

        Returns:
            The number of decks with ``is_deleted = 0`` -- the exact
            total the Library rail badge displays. Soft-deleted decks
            are excluded, matching ``list_decks``' visibility.
        """
        query = "SELECT COUNT(*) AS cnt FROM decks WHERE is_deleted = 0"
        cursor = self.db.execute_query(query)
        row = cursor.fetchone()
        return int(row["cnt"] if row else 0)

    def create_deck(self, name: str, description: Optional[str] = None) -> str:
        """Create a new flashcard deck."""
        deck_id = self.db._generate_uuid()

        with self.db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO decks (
                    id, name, description, created_by, last_modified_by
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (deck_id, name, description, self.db.client_id, self.db.client_id),
            )

        return deck_id

    def update_deck(
        self,
        deck_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        expected_version: Optional[int] = None,
    ) -> bool:
        """Update flashcard deck metadata with optional optimistic locking."""
        try:
            with self.db.transaction() as cursor:
                row = cursor.execute(
                    "SELECT version, is_deleted FROM decks WHERE id = ?",
                    (deck_id,),
                ).fetchone()
                if not row or int(row["is_deleted"]):
                    return False

                current_version = int(row["version"])
                if expected_version is not None and current_version != expected_version:
                    raise ConflictError(
                        "Version mismatch updating deck",
                        entity="decks",
                        entity_id=deck_id,
                    )

                updates: list[str] = []
                params: list[Any] = []
                if name is not None:
                    normalized_name = str(name).strip()
                    if not normalized_name:
                        raise InputError("Deck name cannot be blank")
                    updates.append("name = ?")
                    params.append(normalized_name)
                if description is not None:
                    updates.append("description = ?")
                    params.append(description)

                if not updates:
                    return True

                updates.extend(
                    [
                        "updated_at = CURRENT_TIMESTAMP",
                        "version = version + 1",
                        "last_modified_by = ?",
                    ]
                )
                params.append(self.db.client_id)
                params.append(deck_id)
                cursor.execute(
                    f"UPDATE decks SET {', '.join(updates)} WHERE id = ? AND is_deleted = 0",
                    tuple(params),
                )
                return bool(cursor.rowcount)
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to update deck: {e}") from e

    def delete_flashcard(
        self,
        card_id: str,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
    ) -> bool:
        """Delete a flashcard and refresh the owning deck count."""
        try:
            with self.db.transaction() as cursor:
                row = cursor.execute(
                    "SELECT deck_id, version, is_deleted FROM flashcards WHERE id = ?",
                    (card_id,),
                ).fetchone()
                if not row:
                    return False

                deck_id = row["deck_id"]
                current_version = int(row["version"])
                deleted = int(row["is_deleted"])

                if hard_delete:
                    cursor.execute("DELETE FROM flashcards WHERE id = ?", (card_id,))
                    if deck_id:
                        self._recount_deck_card_count(cursor, deck_id)
                    return True

                if deleted:
                    return True

                if expected_version is not None and current_version != expected_version:
                    raise ConflictError(
                        "Version mismatch deleting flashcard",
                        entity="flashcards",
                        entity_id=card_id,
                    )

                cursor.execute(
                    """
                    UPDATE flashcards
                    SET is_deleted = 1,
                        updated_at = CURRENT_TIMESTAMP,
                        version = version + 1,
                        last_modified_by = ?
                    WHERE id = ? AND is_deleted = 0
                    """,
                    (self.db.client_id, card_id),
                )
                if deck_id:
                    self._recount_deck_card_count(cursor, deck_id)
                return True
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to delete flashcard: {e}") from e

    def move_flashcard(
        self, card_id: str, target_deck_id: str, expected_version: Optional[int] = None
    ) -> bool:
        """Move a flashcard between decks and refresh both deck counts."""
        try:
            with self.db.transaction() as cursor:
                target_row = cursor.execute(
                    "SELECT id FROM decks WHERE id = ? AND is_deleted = 0",
                    (target_deck_id,),
                ).fetchone()
                if not target_row:
                    raise InputError(f"Target deck {target_deck_id} not found")

                row = cursor.execute(
                    "SELECT deck_id, version, is_deleted FROM flashcards WHERE id = ?",
                    (card_id,),
                ).fetchone()
                if not row or int(row["is_deleted"]):
                    return False

                source_deck_id = row["deck_id"]
                current_version = int(row["version"])
                if expected_version is not None and current_version != expected_version:
                    raise ConflictError(
                        "Version mismatch moving flashcard",
                        entity="flashcards",
                        entity_id=card_id,
                    )

                if source_deck_id == target_deck_id:
                    return True

                cursor.execute(
                    """
                    UPDATE flashcards
                    SET deck_id = ?,
                        updated_at = CURRENT_TIMESTAMP,
                        version = version + 1,
                        last_modified_by = ?
                    WHERE id = ? AND is_deleted = 0
                    """,
                    (target_deck_id, self.db.client_id, card_id),
                )
                if source_deck_id:
                    self._recount_deck_card_count(cursor, source_deck_id)
                self._recount_deck_card_count(cursor, target_deck_id)
                return True
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to move flashcard: {e}") from e

    def delete_deck(
        self,
        deck_id: str,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
    ) -> bool:
        """Soft-delete or hard-delete a deck, tombstoning its name on soft delete."""
        tombstone_name = f"__deleted_deck__:{deck_id}"
        try:
            with self.db.transaction() as cursor:
                row = cursor.execute(
                    "SELECT version, is_deleted FROM decks WHERE id = ?",
                    (deck_id,),
                ).fetchone()
                if not row:
                    return False

                current_version = int(row["version"])
                deleted = int(row["is_deleted"])

                if hard_delete:
                    cursor.execute(
                        "DELETE FROM flashcards WHERE deck_id = ?", (deck_id,)
                    )
                    cursor.execute("DELETE FROM decks WHERE id = ?", (deck_id,))
                    return True

                if deleted:
                    return True

                if expected_version is not None and current_version != expected_version:
                    raise ConflictError(
                        "Version mismatch deleting deck",
                        entity="decks",
                        entity_id=deck_id,
                    )

                cursor.execute(
                    """
                    UPDATE flashcards
                    SET is_deleted = 1,
                        updated_at = CURRENT_TIMESTAMP,
                        version = version + 1,
                        last_modified_by = ?
                    WHERE deck_id = ? AND is_deleted = 0
                    """,
                    (self.db.client_id, deck_id),
                )
                cursor.execute(
                    """
                    UPDATE decks
                    SET is_deleted = 1,
                        name = ?,
                        card_count = 0,
                        updated_at = CURRENT_TIMESTAMP,
                        version = version + 1,
                        last_modified_by = ?
                    WHERE id = ? AND is_deleted = 0
                    """,
                    (tombstone_name, self.db.client_id, deck_id),
                )
                return True
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to delete deck: {e}") from e

    @staticmethod
    def _flashcard_template_from_row(row: sqlite3.Row) -> Dict[str, Any]:
        record = dict(row)
        raw_placeholders = record.get("placeholder_definitions")
        if isinstance(raw_placeholders, str) and raw_placeholders.strip():
            try:
                parsed = json.loads(raw_placeholders)
                record["placeholder_definitions"] = (
                    parsed if isinstance(parsed, list) else []
                )
            except json.JSONDecodeError:
                record["placeholder_definitions"] = []
        elif raw_placeholders is None:
            record["placeholder_definitions"] = []
        return record

    def create_flashcard_template(
        self,
        *,
        name: str,
        model_type: str = "basic",
        front_template: str,
        back_template: Optional[str] = None,
        notes_template: Optional[str] = None,
        extra_template: Optional[str] = None,
        placeholder_definitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create a durable local flashcard template."""
        normalized_name = str(name or "").strip()
        normalized_model_type = str(model_type or "basic").strip() or "basic"
        if not normalized_name:
            raise InputError("Flashcard template name is required")
        if not str(front_template or "").strip():
            raise InputError("Flashcard template front_template is required")

        template_id = self.db._generate_uuid()
        placeholders_json = json.dumps(list(placeholder_definitions or []))
        try:
            with self.db.transaction() as cursor:
                cursor.execute(
                    """
                    INSERT INTO flashcard_templates (
                        id, name, model_type, front_template, back_template,
                        notes_template, extra_template, placeholder_definitions, client_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        template_id,
                        normalized_name,
                        normalized_model_type,
                        front_template,
                        back_template,
                        notes_template,
                        extra_template,
                        placeholders_json,
                        self.db.client_id,
                    ),
                )
            return self.get_flashcard_template(template_id)
        except sqlite3.Error as e:
            raise CharactersRAGDBError(
                f"Failed to create flashcard template: {e}"
            ) from e

    def list_flashcard_templates(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        """List non-deleted local flashcard templates."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        total_row = cursor.execute(
            "SELECT COUNT(*) AS total FROM flashcard_templates WHERE deleted = 0"
        ).fetchone()
        cursor.execute(
            """
            SELECT *
            FROM flashcard_templates
            WHERE deleted = 0
            ORDER BY name COLLATE NOCASE ASC, created_at ASC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        items = [self._flashcard_template_from_row(row) for row in cursor.fetchall()]
        total = int(total_row["total"]) if total_row else len(items)
        return {"items": items, "count": len(items), "total": total}

    def get_flashcard_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a non-deleted local flashcard template by ID."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        row = cursor.execute(
            """
            SELECT *
            FROM flashcard_templates
            WHERE id = ? AND deleted = 0
            """,
            (str(template_id),),
        ).fetchone()
        return self._flashcard_template_from_row(row) if row else None

    def update_flashcard_template(
        self,
        template_id: str,
        *,
        name: Optional[str] = None,
        model_type: Optional[str] = None,
        front_template: Optional[str] = None,
        back_template: Optional[str] = None,
        notes_template: Optional[str] = None,
        extra_template: Optional[str] = None,
        placeholder_definitions: Optional[List[Dict[str, Any]]] = None,
        expected_version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update a local flashcard template using optimistic locking when supplied."""
        try:
            with self.db.transaction() as cursor:
                row = cursor.execute(
                    "SELECT * FROM flashcard_templates WHERE id = ? AND deleted = 0",
                    (str(template_id),),
                ).fetchone()
                if not row:
                    return None
                current_version = int(row["version"])
                if expected_version is not None and current_version != expected_version:
                    raise ConflictError(
                        "Version mismatch updating flashcard template",
                        entity="flashcard_templates",
                        entity_id=template_id,
                    )
                target_name = name if name is not None else row["name"]
                target_model_type = (
                    model_type if model_type is not None else row["model_type"]
                )
                target_front_template = (
                    front_template
                    if front_template is not None
                    else row["front_template"]
                )
                if not str(target_name or "").strip():
                    raise InputError("Flashcard template name is required")
                if not str(target_front_template or "").strip():
                    raise InputError("Flashcard template front_template is required")

                cursor.execute(
                    """
                    UPDATE flashcard_templates
                    SET name = ?,
                        model_type = ?,
                        front_template = ?,
                        back_template = ?,
                        notes_template = ?,
                        extra_template = ?,
                        placeholder_definitions = ?,
                        last_modified = CURRENT_TIMESTAMP,
                        version = version + 1,
                        client_id = ?
                    WHERE id = ? AND deleted = 0
                    """,
                    (
                        target_name,
                        target_model_type,
                        target_front_template,
                        back_template
                        if back_template is not None
                        else row["back_template"],
                        notes_template
                        if notes_template is not None
                        else row["notes_template"],
                        extra_template
                        if extra_template is not None
                        else row["extra_template"],
                        json.dumps(list(placeholder_definitions or []))
                        if placeholder_definitions is not None
                        else row["placeholder_definitions"],
                        self.db.client_id,
                        str(template_id),
                    ),
                )
            return self.get_flashcard_template(str(template_id))
        except sqlite3.Error as e:
            raise CharactersRAGDBError(
                f"Failed to update flashcard template: {e}"
            ) from e

    def delete_flashcard_template(
        self, template_id: str, *, expected_version: int
    ) -> bool:
        """Soft-delete a local flashcard template."""
        try:
            with self.db.transaction() as cursor:
                row = cursor.execute(
                    "SELECT version, deleted FROM flashcard_templates WHERE id = ?",
                    (str(template_id),),
                ).fetchone()
                if not row:
                    return False
                if int(row["deleted"]):
                    return True
                if int(row["version"]) != int(expected_version):
                    raise ConflictError(
                        "Version mismatch deleting flashcard template",
                        entity="flashcard_templates",
                        entity_id=template_id,
                    )
                cursor.execute(
                    """
                    UPDATE flashcard_templates
                    SET deleted = 1,
                        last_modified = CURRENT_TIMESTAMP,
                        version = version + 1,
                        client_id = ?
                    WHERE id = ? AND deleted = 0
                    """,
                    (self.db.client_id, str(template_id)),
                )
                return True
        except sqlite3.Error as e:
            raise CharactersRAGDBError(
                f"Failed to delete flashcard template: {e}"
            ) from e

    def create_flashcard_asset(
        self,
        *,
        original_filename: Optional[str],
        mime_type: Optional[str],
        content: bytes,
    ) -> Dict[str, Any]:
        """Persist local flashcard asset bytes and return server-shaped metadata."""
        data = bytes(content or b"")
        if not data:
            raise InputError("Flashcard asset content is required")
        asset_uuid = self.db._generate_uuid()
        filename = str(original_filename or "asset").strip() or "asset"
        normalized_mime = (
            str(mime_type or "application/octet-stream").strip()
            or "application/octet-stream"
        )
        try:
            with self.db.transaction() as cursor:
                cursor.execute(
                    """
                    INSERT INTO flashcard_assets (
                        asset_uuid, original_filename, mime_type, byte_size, content, client_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        asset_uuid,
                        filename,
                        normalized_mime,
                        len(data),
                        sqlite3.Binary(data),
                        self.db.client_id,
                    ),
                )
            return {
                "asset_uuid": asset_uuid,
                "reference": f"flashcard-asset://{asset_uuid}",
                "markdown_snippet": f"![{filename}](flashcard-asset://{asset_uuid})",
                "mime_type": normalized_mime,
                "byte_size": len(data),
                "width": None,
                "height": None,
                "original_filename": filename,
            }
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to create flashcard asset: {e}") from e

    def get_flashcard_asset_content(self, asset_uuid: str) -> Optional[bytes]:
        """Return local flashcard asset bytes by UUID."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        row = cursor.execute(
            """
            SELECT content
            FROM flashcard_assets
            WHERE asset_uuid = ? AND deleted = 0
            """,
            (str(asset_uuid),),
        ).fetchone()
        if not row:
            return None
        return bytes(row["content"])

    #: Default row cap shared by the two flashcard search seams.
    _DEFAULT_FLASHCARD_LIMIT = 100

    def list_flashcards(
        self,
        deck_id: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = _DEFAULT_FLASHCARD_LIMIT,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List flashcards with optional deck filtering and FTS-backed search.

        task-19558: ``q`` used to reach MATCH raw. This is the Study screen's
        flashcard search box, so a card front containing a quote (or any
        typed ``"``) surfaced as a bare ``sqlite3.OperationalError``
        propagating out of the handler -- nothing on that path catches it.
        Every token is now quoted individually and ANDed, which keeps the
        raw bind's multi-word recall (``dragon lore`` still finds a card
        fronted "lore of the dragon reversed") while making operators inert.
        An unsearchable ``q`` -- ``None``, punctuation-only, or containing a
        NUL, which SQLite truncates the bound parameter at -- returns no
        rows rather than raising.
        """
        normalized_q = str(q or "").strip() or None
        params: List[Any] = []

        if normalized_q:
            query = """
                SELECT f.* FROM flashcards f
                JOIN flashcards_fts fts ON f.rowid = fts.rowid
                JOIN decks d ON d.id = f.deck_id
                WHERE flashcards_fts MATCH ? AND f.is_deleted = 0 AND d.is_deleted = 0
            """
            match_expression = build_and_match_query(normalized_q)
            if not match_expression:
                return []
            params.append(match_expression)
            if deck_id:
                query += " AND f.deck_id = ?"
                params.append(deck_id)
        else:
            query = """
                SELECT f.* FROM flashcards f
                JOIN decks d ON d.id = f.deck_id
                WHERE f.is_deleted = 0 AND d.is_deleted = 0
            """
            if deck_id:
                query += " AND f.deck_id = ?"
                params.append(deck_id)

        query += " ORDER BY f.updated_at DESC, f.created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # --- Quiz Methods ---

    def _normalize_multi_select_indices(self, answer: Any) -> List[int]:
        """Normalize multi-select answers into sorted unique option indices."""
        if isinstance(answer, (list, tuple, set)):
            values = list(answer)
        elif isinstance(answer, str):
            raw = answer.strip()
            if not raw:
                return []
            parsed = None
            with contextlib.suppress(json.JSONDecodeError):
                parsed = json.loads(raw)
            if isinstance(parsed, list):
                values = list(parsed)
            else:
                values = [part.strip() for part in raw.split(",")]
        else:
            values = [answer]

        indices: List[int] = []
        for value in values:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if index < 0:
                continue
            indices.append(index)
        return sorted(set(indices))

    def _normalize_matching_map(self, answer: Any) -> Dict[str, str]:
        """Normalize matching answers into a clean left-to-right map."""
        pairs: List[Tuple[str, str]] = []

        if isinstance(answer, dict):
            pairs = [(str(left), str(right)) for left, right in answer.items()]
        elif isinstance(answer, (list, tuple, set)):
            for item in answer:
                if isinstance(item, dict):
                    left = item.get("left")
                    right = item.get("right")
                    if left is not None and right is not None:
                        pairs.append((str(left), str(right)))
                    continue
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    pairs.append((str(item[0]), str(item[1])))
                    continue
                if isinstance(item, str):
                    token = item.strip()
                    delimiter = (
                        "=>" if "=>" in token else ("::" if "::" in token else None)
                    )
                    if not delimiter:
                        continue
                    left, right = token.split(delimiter, 1)
                    pairs.append((left, right))
        elif isinstance(answer, str):
            raw = answer.strip()
            if not raw:
                return {}
            parsed = None
            with contextlib.suppress(json.JSONDecodeError):
                parsed = json.loads(raw)
            if parsed is not None and parsed is not answer:
                return self._normalize_matching_map(parsed)
            for token in re.split(r"(?:\r?\n)|(?:\|\|)", raw):
                trimmed = token.strip()
                if not trimmed:
                    continue
                delimiter = (
                    "=>" if "=>" in trimmed else ("::" if "::" in trimmed else None)
                )
                if not delimiter:
                    continue
                left, right = trimmed.split(delimiter, 1)
                pairs.append((left, right))

        normalized: Dict[str, str] = {}
        for left, right in pairs:
            left_text = str(left).strip()
            right_text = str(right).strip()
            if not left_text or not right_text:
                continue
            normalized[left_text] = right_text
        return normalized

    def _normalize_matching_compare_map(self, answer: Any) -> Dict[str, str]:
        """Normalize matching maps for case-insensitive comparison."""
        normalized = self._normalize_matching_map(answer)
        return {
            self._normalize_fill_blank_text(left): self._normalize_fill_blank_text(
                right
            )
            for left, right in normalized.items()
            if self._normalize_fill_blank_text(left)
            and self._normalize_fill_blank_text(right)
        }

    def _normalize_fill_blank_text(self, value: Any) -> str:
        """Normalize a fill-blank answer for stable comparison."""
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def _clamp_fill_blank_threshold(self, value: Any) -> float:
        """Clamp fuzzy threshold to a sensible range."""
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.88
        return min(1.0, max(0.5, parsed))

    def _parse_fill_blank_json_rules(
        self, raw: str
    ) -> Optional[List[Tuple[str, bool, float]]]:
        """Parse JSON encoded fill-blank rules."""
        if not (raw.startswith("{") or raw.startswith("[")):
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if isinstance(parsed, list):
            accepted = [
                str(entry).strip() for entry in parsed if str(entry or "").strip()
            ]
            if not accepted:
                return None
            return [(entry, False, 0.88) for entry in accepted]

        if not isinstance(parsed, dict):
            return None

        accepted_raw = parsed.get("accepted_answers")
        if not isinstance(accepted_raw, list):
            return None
        accepted = [
            str(entry).strip() for entry in accepted_raw if str(entry or "").strip()
        ]
        if not accepted:
            return None

        fuzzy = bool(parsed.get("fuzzy"))
        threshold = self._clamp_fill_blank_threshold(parsed.get("fuzzy_threshold"))
        return [(entry, fuzzy, threshold) for entry in accepted]

    def _parse_fill_blank_token_rule(
        self, token: str
    ) -> Optional[Tuple[str, bool, float]]:
        """Parse a single delimited fill-blank token."""
        trimmed = token.strip()
        if not trimmed:
            return None
        if not trimmed.startswith("~"):
            return (trimmed, False, 0.88)

        body = trimmed[1:].strip()
        if not body:
            return None
        threshold_match = re.match(r"^(\d(?:\.\d+)?)\s*:\s*(.+)$", body)
        if threshold_match:
            return (
                threshold_match.group(2).strip(),
                True,
                self._clamp_fill_blank_threshold(threshold_match.group(1)),
            )
        return (body, True, 0.88)

    def _parse_fill_blank_answer_rules(
        self, correct_answer: Any
    ) -> List[Tuple[str, bool, float]]:
        """Parse supported fill-blank answer syntaxes into match rules."""
        raw = str(correct_answer or "").strip()
        if not raw:
            return []

        json_rules = self._parse_fill_blank_json_rules(raw)
        if json_rules:
            return json_rules

        delimited_rules: List[Tuple[str, bool, float]] = []
        for part in raw.split("||"):
            parsed_rule = self._parse_fill_blank_token_rule(part)
            if parsed_rule:
                delimited_rules.append(parsed_rule)
        if delimited_rules:
            return delimited_rules

        return [(raw, False, 0.88)]

    def _levenshtein_distance(self, left: str, right: str) -> int:
        """Compute Levenshtein distance for fuzzy fill-blank matching."""
        if left == right:
            return 0
        if not left:
            return len(right)
        if not right:
            return len(left)

        previous = list(range(len(right) + 1))
        current = [0] * (len(right) + 1)
        for left_index, left_char in enumerate(left, start=1):
            current[0] = left_index
            for right_index, right_char in enumerate(right, start=1):
                cost = 0 if left_char == right_char else 1
                current[right_index] = min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + cost,
                )
            previous, current = current, previous
        return previous[-1]

    def _fill_blank_similarity_ratio(self, left: str, right: str) -> float:
        """Compute normalized similarity in [0, 1]."""
        max_len = max(len(left), len(right))
        if max_len == 0:
            return 1.0
        return 1.0 - (self._levenshtein_distance(left, right) / max_len)

    def _check_fill_blank_answer(self, user_answer: Any, correct_answer: Any) -> bool:
        """Grade fill-blank answers using exact, alternates, and fuzzy rules."""
        normalized_user = self._normalize_fill_blank_text(user_answer)
        if not normalized_user:
            return False

        rules = self._parse_fill_blank_answer_rules(correct_answer)
        for value, fuzzy, threshold in rules:
            normalized_value = self._normalize_fill_blank_text(value)
            if not normalized_value:
                continue
            if normalized_value == normalized_user:
                return True
            if (
                fuzzy
                and self._fill_blank_similarity_ratio(normalized_user, normalized_value)
                >= threshold
            ):
                return True
        return False

    def _normalize_quiz_correct_answer(
        self, question_type: str, correct_answer: Any
    ) -> str:
        """Normalize correct answers into a stable stored string representation."""
        normalized_type = str(question_type or "").strip().lower()
        if normalized_type == "multiple_choice":
            try:
                return str(int(correct_answer))
            except (TypeError, ValueError):
                return "0"
        if normalized_type == "multi_select":
            return json.dumps(
                self._normalize_multi_select_indices(correct_answer), ensure_ascii=True
            )
        if normalized_type == "matching":
            normalized = self._normalize_matching_map(correct_answer)
            return json.dumps(normalized, ensure_ascii=True, sort_keys=True)
        if normalized_type == "true_false":
            value = str(correct_answer).strip().lower()
            return "true" if value in {"true", "1", "yes", "y"} else "false"
        return str(correct_answer or "").strip()

    def _deserialize_quiz_row(self, row: sqlite3.Row) -> Optional[Dict[str, Any]]:
        """Deserialize a quiz row into a plain dictionary."""
        if not row:
            return None
        return dict(row)

    def _deserialize_quiz_question(self, row: sqlite3.Row) -> Optional[Dict[str, Any]]:
        """Deserialize a quiz question row and expand JSON-backed fields."""
        item = self.db._deserialize_row_fields(
            row, ["options", "tags_json", "source_citations_json"]
        )
        if not item:
            return None

        tags = item.pop("tags_json", None)
        item["tags"] = tags if isinstance(tags, list) else None

        source_citations = item.pop("source_citations_json", None)
        if isinstance(source_citations, list):
            item["source_citations"] = source_citations
        elif isinstance(source_citations, dict):
            item["source_citations"] = [source_citations]
        else:
            item["source_citations"] = None

        hint_value = item.get("hint")
        if isinstance(hint_value, str):
            item["hint"] = hint_value.strip() or None
        elif hint_value is None:
            item["hint"] = None
        else:
            item["hint"] = str(hint_value).strip() or None

        try:
            item["hint_penalty_points"] = max(
                0, int(item.get("hint_penalty_points") or 0)
            )
        except (TypeError, ValueError):
            item["hint_penalty_points"] = 0

        question_type = str(item.get("question_type") or "").strip().lower()
        if (
            question_type == "multiple_choice"
            and item.get("correct_answer") is not None
        ):
            with contextlib.suppress(TypeError, ValueError):
                item["correct_answer"] = int(item["correct_answer"])
        elif question_type == "multi_select" and item.get("correct_answer") is not None:
            item["correct_answer"] = self._normalize_multi_select_indices(
                item["correct_answer"]
            )
        elif question_type == "matching" and item.get("correct_answer") is not None:
            item["correct_answer"] = self._normalize_matching_map(
                item["correct_answer"]
            )

        return item

    def _public_quiz_question(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Return the question payload safe for attempt-taking UI."""
        payload = dict(question)
        payload.pop("correct_answer", None)
        payload.pop("explanation", None)
        return payload

    def _recount_quiz_questions(self, conn: Any, quiz_id: str) -> int:
        """Recompute and persist the quiz question count."""
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM quiz_questions WHERE quiz_id = ? AND deleted = 0",
            (quiz_id,),
        ).fetchone()
        total = int(row["count"]) if row else 0
        now = self.db._get_current_utc_timestamp_iso()
        conn.execute(
            "UPDATE quizzes SET total_questions = ?, last_modified = ?, version = version + 1, client_id = ? WHERE id = ?",
            (total, now, self.db.client_id, quiz_id),
        )
        return total

    def create_quiz(
        self,
        name: str,
        description: Optional[str] = None,
        workspace_id: Optional[str] = None,
        time_limit_seconds: Optional[int] = None,
        passing_score: Optional[int] = None,
        client_id: str = "unknown",
    ) -> str:
        """Create a new quiz and return its local ID."""
        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise InputError("Quiz name is required.")

        now = self.db._get_current_utc_timestamp_iso()
        quiz_id = self.db._generate_uuid()
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO quizzes (
                    id, name, description, workspace_id, total_questions,
                    time_limit_seconds, passing_score, deleted, client_id,
                    version, created_at, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    quiz_id,
                    normalized_name,
                    (str(description or "").strip() or None),
                    workspace_id,
                    0,
                    time_limit_seconds,
                    passing_score,
                    0,
                    client_id or self.db.client_id,
                    1,
                    now,
                    now,
                ),
            )
        return quiz_id

    def get_quiz(
        self, quiz_id: str, include_deleted: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get a quiz by ID."""
        query = """
            SELECT id, name, description, workspace_id, total_questions,
                   time_limit_seconds, passing_score, deleted, client_id,
                   version, created_at, last_modified
              FROM quizzes
             WHERE id = ?
        """
        params: List[Any] = [quiz_id]
        if not include_deleted:
            query += " AND deleted = 0"
        cursor = self.db.execute_query(query, tuple(params))
        row = cursor.fetchone()
        return self._deserialize_quiz_row(row) if row else None

    def list_quizzes(
        self,
        q: Optional[str] = None,
        workspace_id: Optional[str] = None,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List quizzes with optional search and workspace filtering."""
        normalized_q = str(q or "").strip().lower() or None
        where_clauses = ["1=1"]
        params: List[Any] = []

        if not include_deleted:
            where_clauses.append("deleted = 0")
        if workspace_id is not None:
            where_clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if normalized_q:
            q_like = f"%{normalized_q}%"
            where_clauses.append(
                "(LOWER(name) LIKE ? OR LOWER(COALESCE(description, '')) LIKE ?)"
            )
            params.extend([q_like, q_like])

        where_sql = " AND ".join(where_clauses)
        query = f"""
            SELECT id, name, description, workspace_id, total_questions,
                   time_limit_seconds, passing_score, deleted, client_id,
                   version, created_at, last_modified
              FROM quizzes
             WHERE {where_sql}
             ORDER BY last_modified DESC, created_at DESC
             LIMIT ? OFFSET ?
        """
        count_query = f"SELECT COUNT(*) AS count FROM quizzes WHERE {where_sql}"

        cursor = self.db.execute_query(query, tuple(params + [limit, offset]))
        items = [self._deserialize_quiz_row(row) for row in cursor.fetchall()]
        items = [item for item in items if item is not None]

        count_cursor = self.db.execute_query(count_query, tuple(params))
        count_row = count_cursor.fetchone()
        total = int(count_row["count"]) if count_row else 0
        return {"items": items, "count": total}

    def count_quizzes(self) -> int:
        """Count all non-deleted quizzes.

        Returns:
            The number of quizzes with ``deleted = 0`` -- the exact
            total the Library rail badge displays. Note the quizzes
            table uses the column name ``deleted``, not ``is_deleted``
            as ``notes``/``decks``/``flashcards`` do.
        """
        query = "SELECT COUNT(*) AS cnt FROM quizzes WHERE deleted = 0"
        cursor = self.db.execute_query(query)
        row = cursor.fetchone()
        return int(row["cnt"] if row else 0)

    def update_quiz(
        self, quiz_id: str, updates: Dict[str, Any], client_id: str = "unknown"
    ) -> bool:
        """Update quiz fields using optimistic version checks when supplied."""
        expected_version = updates.pop("expected_version", None)
        allowed = {
            "name",
            "description",
            "workspace_id",
            "time_limit_seconds",
            "passing_score",
        }
        set_parts: List[str] = []
        params: List[Any] = []

        for key, value in updates.items():
            if key not in allowed:
                continue
            if key == "name":
                value = str(value or "").strip()
                if not value:
                    raise InputError("Quiz name is required.")
            elif key == "description":
                value = str(value or "").strip() or None
            set_parts.append(f"{key} = ?")
            params.append(value)

        now = self.db._get_current_utc_timestamp_iso()
        try:
            with self.db.transaction() as conn:
                row = conn.execute(
                    "SELECT version FROM quizzes WHERE id = ? AND deleted = 0",
                    (quiz_id,),
                ).fetchone()
                if not row:
                    return False
                current_version = int(row["version"])
                if expected_version is not None and current_version != expected_version:
                    raise ConflictError(
                        "Version mismatch updating quiz",
                        entity="quizzes",
                        entity_id=quiz_id,
                    )
                if not set_parts:
                    return True
                set_parts.extend(
                    ["last_modified = ?", "version = version + 1", "client_id = ?"]
                )
                params.extend([now, client_id or self.db.client_id, quiz_id])
                conn.execute(
                    f"UPDATE quizzes SET {', '.join(set_parts)} WHERE id = ? AND deleted = 0",
                    tuple(params),
                )
                return True
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to update quiz: {e}") from e

    def delete_quiz(
        self,
        quiz_id: str,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
    ) -> bool:
        """Soft-delete or hard-delete a quiz."""
        now = self.db._get_current_utc_timestamp_iso()
        try:
            with self.db.transaction() as conn:
                row = conn.execute(
                    "SELECT version, deleted FROM quizzes WHERE id = ?",
                    (quiz_id,),
                ).fetchone()
                if not row:
                    return False
                current_version = int(row["version"])
                deleted = int(row["deleted"])
                if hard_delete:
                    conn.execute("DELETE FROM quizzes WHERE id = ?", (quiz_id,))
                    return True
                if deleted:
                    return True
                if expected_version is not None and current_version != expected_version:
                    raise ConflictError(
                        "Version mismatch deleting quiz",
                        entity="quizzes",
                        entity_id=quiz_id,
                    )
                conn.execute(
                    "UPDATE quizzes SET deleted = 1, last_modified = ?, version = version + 1, client_id = ? "
                    "WHERE id = ? AND deleted = 0",
                    (now, self.db.client_id, quiz_id),
                )
                conn.execute(
                    "UPDATE quiz_questions SET deleted = 1, last_modified = ?, version = version + 1, client_id = ? "
                    "WHERE quiz_id = ? AND deleted = 0",
                    (now, self.db.client_id, quiz_id),
                )
                return True
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to delete quiz: {e}") from e

    def create_question(
        self,
        quiz_id: str,
        question_type: str,
        question_text: str,
        correct_answer: Union[int, str, List[int], Dict[str, str]],
        options: Optional[List[str]] = None,
        explanation: Optional[str] = None,
        hint: Optional[str] = None,
        hint_penalty_points: int = 0,
        source_citations: Optional[List[Dict[str, Any]]] = None,
        points: int = 1,
        order_index: int = 0,
        tags: Optional[List[str]] = None,
        client_id: str = "unknown",
    ) -> str:
        """Create a question for a quiz and update the quiz question count."""
        normalized_type = str(question_type or "").strip().lower()
        if normalized_type not in {
            "multiple_choice",
            "multi_select",
            "matching",
            "true_false",
            "fill_blank",
        }:
            raise InputError("Unsupported quiz question type.")

        normalized_text = str(question_text or "").strip()
        if not normalized_text:
            raise InputError("Question text is required.")

        now = self.db._get_current_utc_timestamp_iso()
        question_id = self.db._generate_uuid()
        normalized_hint = str(hint or "").strip() or None
        normalized_penalty = max(0, int(hint_penalty_points or 0))

        try:
            with self.db.transaction() as conn:
                quiz_row = conn.execute(
                    "SELECT id FROM quizzes WHERE id = ? AND deleted = 0",
                    (quiz_id,),
                ).fetchone()
                if not quiz_row:
                    raise ConflictError(
                        "Quiz not found", entity="quizzes", entity_id=quiz_id
                    )

                conn.execute(
                    """
                    INSERT INTO quiz_questions (
                        id, quiz_id, question_type, question_text, options,
                        correct_answer, explanation, hint, hint_penalty_points,
                        source_citations_json, points, order_index, tags_json,
                        deleted, client_id, version, created_at, last_modified
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        question_id,
                        quiz_id,
                        normalized_type,
                        normalized_text,
                        self.db._ensure_json_string(options),
                        self._normalize_quiz_correct_answer(
                            normalized_type, correct_answer
                        ),
                        (str(explanation or "").strip() or None),
                        normalized_hint,
                        normalized_penalty,
                        self.db._ensure_json_string(source_citations),
                        max(0, int(points or 0)),
                        int(order_index or 0),
                        self.db._ensure_json_string(tags),
                        0,
                        client_id or self.db.client_id,
                        1,
                        now,
                        now,
                    ),
                )
                self._recount_quiz_questions(conn, quiz_id)
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to create question: {e}") from e

        return question_id

    def get_question(
        self, question_id: str, include_deleted: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get a quiz question by ID."""
        query = """
            SELECT id, quiz_id, question_type, question_text, options,
                   correct_answer, explanation, hint, hint_penalty_points,
                   source_citations_json, points, order_index, tags_json,
                   deleted, client_id, version, created_at, last_modified
              FROM quiz_questions
             WHERE id = ?
        """
        params: List[Any] = [question_id]
        if not include_deleted:
            query += " AND deleted = 0"
        cursor = self.db.execute_query(query, tuple(params))
        row = cursor.fetchone()
        return self._deserialize_quiz_question(row) if row else None

    def list_questions(
        self,
        quiz_id: str,
        q: Optional[str] = None,
        include_answers: bool = False,
        limit: Optional[int] = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List questions for a quiz with optional text search."""
        normalized_q = str(q or "").strip().lower() or None
        where_clauses = ["quiz_id = ?", "deleted = 0"]
        params: List[Any] = [quiz_id]

        if normalized_q:
            q_like = f"%{normalized_q}%"
            where_clauses.append(
                "(LOWER(question_text) LIKE ? OR LOWER(COALESCE(explanation, '')) LIKE ?)"
            )
            params.extend([q_like, q_like])

        where_sql = " AND ".join(where_clauses)
        query = f"""
            SELECT id, quiz_id, question_type, question_text, options,
                   correct_answer, explanation, hint, hint_penalty_points,
                   source_citations_json, points, order_index, tags_json,
                   deleted, client_id, version, created_at, last_modified
              FROM quiz_questions
             WHERE {where_sql}
             ORDER BY order_index ASC, created_at ASC, id ASC
        """
        query_params: List[Any] = list(params)
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            query_params.extend([int(limit), int(offset)])

        count_query = f"SELECT COUNT(*) AS count FROM quiz_questions WHERE {where_sql}"
        cursor = self.db.execute_query(query, tuple(query_params))

        items: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            question = self._deserialize_quiz_question(row)
            if not question:
                continue
            if not include_answers:
                question = self._public_quiz_question(question)
            items.append(question)

        count_cursor = self.db.execute_query(count_query, tuple(params))
        count_row = count_cursor.fetchone()
        total = int(count_row["count"]) if count_row else 0
        return {"items": items, "count": total}

    def update_question(
        self, question_id: str, updates: Dict[str, Any], client_id: str = "unknown"
    ) -> bool:
        """Update quiz question fields using optimistic locking when supplied."""
        expected_version = updates.pop("expected_version", None)
        allowed = {
            "question_type",
            "question_text",
            "options",
            "correct_answer",
            "explanation",
            "hint",
            "hint_penalty_points",
            "source_citations",
            "points",
            "order_index",
            "tags",
        }
        set_parts: List[str] = []
        params: List[Any] = []

        for key, value in updates.items():
            if key not in allowed:
                continue
            if key == "options":
                set_parts.append("options = ?")
                params.append(self.db._ensure_json_string(value))
            elif key == "tags":
                set_parts.append("tags_json = ?")
                params.append(self.db._ensure_json_string(value))
            elif key == "correct_answer":
                question_type = updates.get("question_type")
                if not question_type:
                    row = self.db.execute_query(
                        "SELECT question_type FROM quiz_questions WHERE id = ? AND deleted = 0",
                        (question_id,),
                    ).fetchone()
                    question_type = row["question_type"] if row else None
                set_parts.append("correct_answer = ?")
                params.append(
                    self._normalize_quiz_correct_answer(
                        str(question_type or "fill_blank"), value
                    )
                )
            elif key == "hint":
                set_parts.append("hint = ?")
                params.append(str(value or "").strip() or None)
            elif key == "hint_penalty_points":
                try:
                    normalized_penalty = max(0, int(value or 0))
                except (TypeError, ValueError):
                    normalized_penalty = 0
                set_parts.append("hint_penalty_points = ?")
                params.append(normalized_penalty)
            elif key == "source_citations":
                set_parts.append("source_citations_json = ?")
                params.append(self.db._ensure_json_string(value))
            elif key == "question_text":
                normalized_text = str(value or "").strip()
                if not normalized_text:
                    raise InputError("Question text is required.")
                set_parts.append("question_text = ?")
                params.append(normalized_text)
            elif key == "question_type":
                normalized_type = str(value or "").strip().lower()
                if normalized_type not in {
                    "multiple_choice",
                    "multi_select",
                    "matching",
                    "true_false",
                    "fill_blank",
                }:
                    raise InputError("Unsupported quiz question type.")
                set_parts.append("question_type = ?")
                params.append(normalized_type)
            else:
                set_parts.append(f"{key} = ?")
                params.append(value)

        now = self.db._get_current_utc_timestamp_iso()
        try:
            with self.db.transaction() as conn:
                row = conn.execute(
                    "SELECT version, quiz_id FROM quiz_questions WHERE id = ? AND deleted = 0",
                    (question_id,),
                ).fetchone()
                if not row:
                    return False
                current_version = int(row["version"])
                quiz_id = row["quiz_id"]
                if expected_version is not None and current_version != expected_version:
                    raise ConflictError(
                        "Version mismatch updating question",
                        entity="quiz_questions",
                        entity_id=question_id,
                    )
                if not set_parts:
                    return True
                set_parts.extend(
                    ["last_modified = ?", "version = version + 1", "client_id = ?"]
                )
                params.extend([now, client_id or self.db.client_id, question_id])
                conn.execute(
                    f"UPDATE quiz_questions SET {', '.join(set_parts)} WHERE id = ? AND deleted = 0",
                    tuple(params),
                )
                self._recount_quiz_questions(conn, quiz_id)
                return True
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to update question: {e}") from e

    def delete_question(
        self,
        question_id: str,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
    ) -> bool:
        """Delete a quiz question and refresh the owning quiz question count."""
        now = self.db._get_current_utc_timestamp_iso()
        try:
            with self.db.transaction() as conn:
                row = conn.execute(
                    "SELECT quiz_id, version, deleted FROM quiz_questions WHERE id = ?",
                    (question_id,),
                ).fetchone()
                if not row:
                    return False
                quiz_id = row["quiz_id"]
                current_version = int(row["version"])
                deleted = int(row["deleted"])
                if hard_delete:
                    conn.execute(
                        "DELETE FROM quiz_questions WHERE id = ?", (question_id,)
                    )
                    self._recount_quiz_questions(conn, quiz_id)
                    return True
                if deleted:
                    return True
                if expected_version is not None and current_version != expected_version:
                    raise ConflictError(
                        "Version mismatch deleting question",
                        entity="quiz_questions",
                        entity_id=question_id,
                    )
                conn.execute(
                    "UPDATE quiz_questions SET deleted = 1, last_modified = ?, version = version + 1, client_id = ? "
                    "WHERE id = ? AND deleted = 0",
                    (now, self.db.client_id, question_id),
                )
                self._recount_quiz_questions(conn, quiz_id)
                return True
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to delete question: {e}") from e

    def start_attempt(self, quiz_id: str, client_id: str = "unknown") -> Dict[str, Any]:
        """Start a new quiz attempt and snapshot the current question set."""
        now = self.db._get_current_utc_timestamp_iso()
        try:
            with self.db.transaction() as conn:
                quiz_row = conn.execute(
                    "SELECT id FROM quizzes WHERE id = ? AND deleted = 0",
                    (quiz_id,),
                ).fetchone()
                if not quiz_row:
                    raise ConflictError(
                        "Quiz not found", entity="quizzes", entity_id=quiz_id
                    )

                questions_payload = self.list_questions(
                    quiz_id, include_answers=True, limit=None, offset=0
                )
                questions = questions_payload.get("items", [])
                total_possible = sum(
                    int(question.get("points") or 0) for question in questions
                )
                attempt_id = self.db._generate_uuid()
                conn.execute(
                    """
                    INSERT INTO quiz_attempts (
                        id, quiz_id, started_at, completed_at, score, total_possible,
                        time_spent_seconds, questions_snapshot, answers, client_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt_id,
                        quiz_id,
                        now,
                        None,
                        None,
                        total_possible,
                        None,
                        json.dumps(questions, ensure_ascii=True, default=str),
                        json.dumps([], ensure_ascii=True),
                        client_id or self.db.client_id,
                    ),
                )
                return {
                    "id": attempt_id,
                    "quiz_id": quiz_id,
                    "started_at": now,
                    "completed_at": None,
                    "score": None,
                    "total_possible": total_possible,
                    "time_spent_seconds": None,
                    "answers": [],
                    "questions": [
                        self._public_quiz_question(question) for question in questions
                    ],
                }
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to start attempt: {e}") from e

    def submit_attempt(
        self, attempt_id: str, answers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Submit and grade a quiz attempt."""
        now = self.db._get_current_utc_timestamp_iso()
        try:
            with self.db.transaction() as conn:
                row = conn.execute(
                    "SELECT quiz_id, started_at, total_possible, questions_snapshot, answers FROM quiz_attempts WHERE id = ?",
                    (attempt_id,),
                ).fetchone()
                if not row:
                    raise ConflictError(
                        "Attempt not found",
                        entity="quiz_attempts",
                        entity_id=attempt_id,
                    )

                row_data = dict(row)
                questions_snapshot = row_data.get("questions_snapshot") or "[]"
                try:
                    questions = json.loads(questions_snapshot)
                except json.JSONDecodeError:
                    questions = []

                questions_by_id = {
                    str(question.get("id")): question
                    for question in questions
                    if question.get("id") is not None
                }

                graded_answers: List[Dict[str, Any]] = []
                total_possible = sum(
                    int(question.get("points") or 0) for question in questions
                )
                score = 0
                total_time_ms = 0

                for answer in answers:
                    question_id = answer.get("question_id")
                    question = (
                        questions_by_id.get(str(question_id))
                        if question_id is not None
                        else None
                    )
                    user_answer = answer.get("user_answer")
                    hint_used = bool(answer.get("hint_used"))
                    hint_penalty_points = 0
                    points_awarded = 0
                    is_correct = False
                    correct_answer = None
                    explanation = None
                    source_citations = None

                    if question:
                        is_correct = self._check_answer(question, user_answer)
                        points_value = int(question.get("points") or 0)
                        configured_penalty = max(
                            0, int(question.get("hint_penalty_points") or 0)
                        )
                        if is_correct:
                            hint_penalty_points = (
                                min(points_value, configured_penalty)
                                if hint_used
                                else 0
                            )
                            points_awarded = max(0, points_value - hint_penalty_points)
                        score += points_awarded
                        correct_answer = question.get("correct_answer")
                        explanation = question.get("explanation")
                        source_citations = question.get("source_citations")

                    time_spent_ms = answer.get("time_spent_ms")
                    if isinstance(time_spent_ms, (int, float)):
                        total_time_ms += int(time_spent_ms)

                    graded_answers.append(
                        {
                            "question_id": question_id,
                            "user_answer": user_answer,
                            "is_correct": is_correct,
                            "correct_answer": correct_answer,
                            "explanation": explanation,
                            "hint_used": hint_used,
                            "hint_penalty_points": hint_penalty_points,
                            "source_citations": source_citations,
                            "points_awarded": points_awarded,
                            "time_spent_ms": time_spent_ms,
                        }
                    )

                time_spent_seconds = (
                    int(total_time_ms / 1000) if total_time_ms else None
                )
                conn.execute(
                    "UPDATE quiz_attempts SET completed_at = ?, score = ?, total_possible = ?, time_spent_seconds = ?, answers = ? "
                    "WHERE id = ?",
                    (
                        now,
                        score,
                        total_possible,
                        time_spent_seconds,
                        json.dumps(graded_answers, ensure_ascii=True),
                        attempt_id,
                    ),
                )

                return {
                    "id": attempt_id,
                    "quiz_id": row_data.get("quiz_id"),
                    "started_at": row_data.get("started_at"),
                    "completed_at": now,
                    "score": score,
                    "total_possible": total_possible,
                    "time_spent_seconds": time_spent_seconds,
                    "answers": graded_answers,
                }
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to submit attempt: {e}") from e

    def get_attempt(
        self,
        attempt_id: str,
        include_questions: bool = False,
        include_answers: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Load a quiz attempt and optionally expand its question snapshot."""
        cursor = self.db.execute_query(
            """
            SELECT id, quiz_id, started_at, completed_at, score, total_possible,
                   time_spent_seconds, questions_snapshot, answers
              FROM quiz_attempts
             WHERE id = ?
            """,
            (attempt_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        item = self.db._deserialize_row_fields(row, ["questions_snapshot", "answers"])
        if not item:
            return None

        questions = item.pop("questions_snapshot", None)
        if include_questions and isinstance(questions, list):
            if include_answers:
                item["questions"] = questions
            else:
                item["questions"] = [
                    self._public_quiz_question(question) for question in questions
                ]

        return item

    def list_attempts(
        self,
        quiz_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List quiz attempts with optional quiz filtering."""
        where_clauses = ["1=1"]
        params: List[Any] = []
        if quiz_id is not None:
            where_clauses.append("quiz_id = ?")
            params.append(quiz_id)

        where_sql = " AND ".join(where_clauses)
        query = f"""
            SELECT id, quiz_id, started_at, completed_at, score, total_possible, time_spent_seconds
              FROM quiz_attempts
             WHERE {where_sql}
             ORDER BY started_at DESC
             LIMIT ? OFFSET ?
        """
        count_query = f"SELECT COUNT(*) AS count FROM quiz_attempts WHERE {where_sql}"

        cursor = self.db.execute_query(query, tuple(params + [limit, offset]))
        items = [dict(row) for row in cursor.fetchall()]
        count_cursor = self.db.execute_query(count_query, tuple(params))
        count_row = count_cursor.fetchone()
        total = int(count_row["count"]) if count_row else 0
        return {"items": items, "count": total}

    def _check_answer(self, question: Dict[str, Any], user_answer: Any) -> bool:
        """Grade a single answer against the stored quiz question payload."""
        question_type = str(question.get("question_type") or "").strip().lower()
        correct_answer = question.get("correct_answer")

        if question_type == "multiple_choice":
            try:
                return int(user_answer) == int(correct_answer)
            except (TypeError, ValueError):
                return False
        if question_type == "multi_select":
            if user_answer is None or correct_answer is None:
                return False
            return self._normalize_multi_select_indices(
                user_answer
            ) == self._normalize_multi_select_indices(correct_answer)
        if question_type == "matching":
            if user_answer is None or correct_answer is None:
                return False
            return self._normalize_matching_compare_map(
                user_answer
            ) == self._normalize_matching_compare_map(correct_answer)
        if question_type == "true_false":
            if user_answer is None or correct_answer is None:
                return False
            return (
                str(user_answer).strip().lower() == str(correct_answer).strip().lower()
            )
        if question_type == "fill_blank":
            return self._check_fill_blank_answer(user_answer, correct_answer)
        return False

    def create_learning_path(
        self, title: str, description: Optional[str] = None
    ) -> str:
        """Create a new learning path."""
        path_id = self.db._generate_uuid()

        with self.db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO learning_paths (
                    id, title, description, created_by, last_modified_by
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (path_id, title, description, self.db.client_id, self.db.client_id),
            )

        return path_id

    def create_topic(self, topic_data: Dict[str, Any]) -> str:
        """Create a new topic within a learning path."""
        topic_id = self.db._generate_uuid()

        with self.db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO topics (
                    id, path_id, parent_id, title, content,
                    topic_order, created_by, last_modified_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    topic_id,
                    topic_data.get("path_id"),
                    topic_data.get("parent_id"),
                    topic_data["title"],
                    topic_data.get("content", ""),
                    topic_data.get("topic_order", 0),
                    self.db.client_id,
                    self.db.client_id,
                ),
            )

        return topic_id

    def update_topic_progress(
        self, topic_id: str, progress: float, status: Optional[str] = None
    ) -> None:
        """Update topic progress and status."""
        if progress < 0 or progress > 1:
            raise InputError("Progress must be between 0 and 1")

        with self.db.transaction() as cursor:
            query = """
                UPDATE topics
                SET progress = ?, updated_at = CURRENT_TIMESTAMP, version = version + 1
            """
            params = [progress]

            if status:
                if status not in ["not_started", "in_progress", "completed"]:
                    raise InputError("Invalid topic status")
                query += ", status = ?"
                params.append(status)

            query += " WHERE id = ? AND is_deleted = 0"
            params.append(topic_id)

            cursor.execute(query, params)

    # NOTE (task-19042): the write-only mindmap accessors (`create_mindmap`,
    # `add_mindmap_node`) were retired with the orphaned mindmap subsystem —
    # they had zero read counterparts, so nothing written could ever be
    # displayed. The `mindmaps`/`mindmap_nodes` tables (and their FTS mirror,
    # triggers, and index) deliberately remain in the schema, dormant, so
    # this retirement needs no schema-version bump; dropping them is a future
    # migration's job.

    def search_flashcards(
        self,
        query: str,
        deck_id: Optional[str] = None,
        limit: int = _DEFAULT_FLASHCARD_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Search flashcards using FTS; every token of ``query`` must appear.

        See ``list_flashcards`` for the form and its rationale (task-19558).

        ``limit`` caps the returned rows, most-recently-updated first. A
        non-positive or non-integer ``limit`` falls back to the default instead
        of reaching SQLite, where ``LIMIT -1`` would mean "no limit".
        """
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            limit = self._DEFAULT_FLASHCARD_LIMIT
        base_query = """
            SELECT f.* FROM flashcards f
            JOIN flashcards_fts fts ON f.rowid = fts.rowid
            WHERE flashcards_fts MATCH ? AND f.is_deleted = 0
        """
        match_expression = build_and_match_query(query)
        if not match_expression:
            return []
        params = [match_expression]

        if deck_id:
            base_query += " AND f.deck_id = ?"
            params.append(deck_id)

        base_query += " ORDER BY f.updated_at DESC, f.created_at DESC LIMIT ?"
        params.append(limit)

        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(base_query, params)

        return [dict(row) for row in cursor.fetchall()]

    def get_study_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get study statistics for the last N days."""
        from datetime import datetime, timedelta

        start_date = datetime.now(timezone.utc) - timedelta(days=days)

        conn = self.db.get_connection()
        cursor = conn.cursor()

        # Cards reviewed
        cursor.execute(
            """
            SELECT COUNT(*) as total_reviews,
                   AVG(rating) as avg_rating
            FROM review_history
            WHERE reviewed_at >= ?
        """,
            (start_date.isoformat(),),
        )

        review_stats = dict(cursor.fetchone())

        # Topics completed
        cursor.execute(
            """
            SELECT COUNT(*) as topics_completed
            FROM topics
            WHERE status = 'completed' AND updated_at >= ?
        """,
            (start_date.isoformat(),),
        )

        topic_stats = dict(cursor.fetchone())

        # Study sessions
        cursor.execute(
            """
            SELECT COUNT(*) as total_sessions,
                   SUM(duration) as total_duration,
                   AVG(duration) as avg_duration
            FROM study_sessions
            WHERE started_at >= ?
        """,
            (start_date.isoformat(),),
        )

        session_stats = dict(cursor.fetchone())

        return {
            "reviews": review_stats,
            "topics": topic_stats,
            "sessions": session_stats,
        }
