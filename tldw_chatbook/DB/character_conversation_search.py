"""Local character-conversation projection and selected-branch search storage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationCursor,
    CharacterConversationGroup,
    CharacterConversationPage,
    CharacterConversationRow,
    CharacterKeywordIndexStatus,
    CharacterRepairCandidate,
    CharacterRepairRequest,
    CharacterRepairResult,
    EligibleConversationDocument,
    LocalCharacterConversationTarget,
    ResolvedLocalCharacterKey,
    UnavailableCharacterReason,
    UnresolvedConversationKey,
)

if TYPE_CHECKING:
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


CHARACTER_CONVERSATION_SEARCH_SCHEMA_SQL = """
CREATE TABLE character_conversation_search_generations(
  generation_id TEXT PRIMARY KEY NOT NULL,
  data_authority_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('building', 'ready', 'failed')),
  policy_version INTEGER NOT NULL CHECK(policy_version > 0),
  source_revision INTEGER NOT NULL CHECK(source_revision >= 0),
  processed_conversations INTEGER NOT NULL DEFAULT 0 CHECK(processed_conversations >= 0),
  error_code TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at TEXT,
  lease_expires_at TEXT,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX character_conversation_search_generations_authority_status
  ON character_conversation_search_generations(data_authority_id, status);
CREATE UNIQUE INDEX character_conversation_search_one_ready_generation
  ON character_conversation_search_generations(data_authority_id)
  WHERE status = 'ready';

CREATE TABLE character_conversation_search_revision(
  singleton_id INTEGER PRIMARY KEY CHECK(singleton_id = 1),
  data_revision INTEGER NOT NULL CHECK(data_revision >= 0),
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO character_conversation_search_revision(singleton_id, data_revision)
VALUES(1, 0);
CREATE TRIGGER character_conversation_search_revision_no_delete
BEFORE DELETE ON character_conversation_search_revision
BEGIN
  SELECT RAISE(ABORT, 'character conversation search revision is required');
END;

CREATE TABLE character_conversation_search_state(
  singleton_id INTEGER PRIMARY KEY CHECK(singleton_id = 1),
  data_authority_id TEXT,
  active_policy_version INTEGER,
  activated INTEGER NOT NULL DEFAULT 0 CHECK(activated IN (0, 1)),
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO character_conversation_search_state(singleton_id) VALUES(1);

CREATE TABLE character_conversation_search_dirty(
  conversation_id TEXT PRIMARY KEY NOT NULL,
  data_authority_id TEXT NOT NULL,
  source_revision INTEGER NOT NULL CHECK(source_revision >= 0),
  enqueued_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX character_conversation_search_dirty_authority_revision
  ON character_conversation_search_dirty(data_authority_id, source_revision);

CREATE TABLE character_conversation_search_documents(
  document_id INTEGER PRIMARY KEY AUTOINCREMENT,
  data_authority_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  character_id INTEGER NOT NULL REFERENCES character_cards(id) ON DELETE CASCADE,
  character_label TEXT NOT NULL,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  eligibility_digest TEXT NOT NULL,
  validated_eligibility_digest TEXT NOT NULL,
  source_revision INTEGER NOT NULL CHECK(source_revision >= 0),
  generation_id TEXT NOT NULL
    REFERENCES character_conversation_search_generations(generation_id) ON DELETE CASCADE,
  UNIQUE(data_authority_id, generation_id, conversation_id)
);
CREATE INDEX character_conversation_search_documents_character
  ON character_conversation_search_documents(
    data_authority_id, character_id, generation_id, conversation_id
  );
CREATE INDEX character_conversation_search_documents_revision
  ON character_conversation_search_documents(data_authority_id, source_revision);

CREATE VIRTUAL TABLE character_conversation_fts USING fts5(
  character_label,
  title,
  body,
  content='character_conversation_search_documents',
  content_rowid='document_id'
);
CREATE TRIGGER character_conversation_search_documents_ai
AFTER INSERT ON character_conversation_search_documents BEGIN
  INSERT INTO character_conversation_fts(rowid, character_label, title, body)
  VALUES(new.document_id, new.character_label, new.title, new.body);
END;
CREATE TRIGGER character_conversation_search_documents_au
AFTER UPDATE ON character_conversation_search_documents BEGIN
  INSERT INTO character_conversation_fts(
    character_conversation_fts, rowid, character_label, title, body
  ) VALUES('delete', old.document_id, old.character_label, old.title, old.body);
  INSERT INTO character_conversation_fts(rowid, character_label, title, body)
  VALUES(new.document_id, new.character_label, new.title, new.body);
END;
CREATE TRIGGER character_conversation_search_documents_ad
AFTER DELETE ON character_conversation_search_documents BEGIN
  INSERT INTO character_conversation_fts(
    character_conversation_fts, rowid, character_label, title, body
  ) VALUES('delete', old.document_id, old.character_label, old.title, old.body);
END;

CREATE TRIGGER character_conversation_search_messages_ai
AFTER INSERT ON messages BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT new.conversation_id, state.data_authority_id, revision.data_revision
    FROM character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
CREATE TRIGGER character_conversation_search_messages_au
AFTER UPDATE ON messages BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT new.conversation_id, state.data_authority_id, revision.data_revision
    FROM character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
CREATE TRIGGER character_conversation_search_messages_ad
AFTER DELETE ON messages BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT old.conversation_id, state.data_authority_id, revision.data_revision
    FROM character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
CREATE TRIGGER character_conversation_search_conversations_ai
AFTER INSERT ON conversations BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT new.id, state.data_authority_id, revision.data_revision
    FROM character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
CREATE TRIGGER character_conversation_search_conversations_au
AFTER UPDATE ON conversations BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT new.id, state.data_authority_id, revision.data_revision
    FROM character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
CREATE TRIGGER character_conversation_search_conversations_ad
AFTER DELETE ON conversations BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT old.id, state.data_authority_id, revision.data_revision
    FROM character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
CREATE TRIGGER character_conversation_search_characters_au
AFTER UPDATE ON character_cards BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT c.id, state.data_authority_id, revision.data_revision
    FROM conversations AS c,
         character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE c.character_id = new.id
     AND state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
CREATE TRIGGER character_conversation_search_characters_ad
AFTER DELETE ON character_cards BEGIN
  UPDATE character_conversation_search_revision
     SET data_revision = data_revision + 1, updated_at = CURRENT_TIMESTAMP
   WHERE singleton_id = 1;
  INSERT INTO character_conversation_search_dirty(
    conversation_id, data_authority_id, source_revision
  )
  SELECT c.id, state.data_authority_id, revision.data_revision
    FROM conversations AS c,
         character_conversation_search_state AS state,
         character_conversation_search_revision AS revision
   WHERE c.character_id = old.id
     AND state.singleton_id = 1 AND state.activated = 1
  ON CONFLICT(conversation_id) DO UPDATE SET
    data_authority_id = excluded.data_authority_id,
    source_revision = excluded.source_revision,
    enqueued_at = CURRENT_TIMESTAMP;
END;
"""


class SelectedBranchEligibilityProjector:
    """Build the one canonical eligible document from one SQLite snapshot."""

    def __init__(self, database: CharactersRAGDB) -> None:
        self._database = database
        self._authority = database.get_local_authority_id()

    def project(
        self,
        conversation_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> EligibleConversationDocument | None:
        """Return selected visible user/assistant text, failing closed on bad graphs."""

        if connection is not None:
            return self._project(connection, conversation_id)
        with self._database.transaction() as snapshot:
            return self._project(snapshot, conversation_id)

    def _project(
        self, connection: sqlite3.Connection, conversation_id: str
    ) -> EligibleConversationDocument | None:
        authority = self._authority
        conversation = connection.execute(
            """
            SELECT conversations.id, conversations.title,
                   conversations.character_id,
                   conversations.assistant_kind,
                   conversations.assistant_authority_id,
                   conversations.runtime_backend,
                   conversations.active_leaf_message_id,
                   conversations.version
              FROM conversations
              JOIN character_cards AS card
                ON card.id = conversations.character_id
               AND card.deleted = 0
             WHERE conversations.id = ? AND conversations.deleted = 0
            """,
            (conversation_id,),
        ).fetchone()
        if (
            conversation is None
            or conversation["runtime_backend"] != "local"
            or conversation["assistant_kind"] != "character"
            or conversation["assistant_authority_id"] != authority
            or not isinstance(conversation["character_id"], int)
            or conversation["character_id"] < 1
        ):
            return None

        rows = connection.execute(
            """
            SELECT id, parent_message_id, role, content, deleted,
                   variant_of, is_selected_variant, timestamp, rowid
              FROM messages
             WHERE conversation_id = ?
             ORDER BY timestamp, rowid
            """,
            (conversation_id,),
        ).fetchall()
        live = {str(row["id"]): row for row in rows if not row["deleted"]}
        if not live:
            return None

        active_leaf = conversation["active_leaf_message_id"]
        if active_leaf is None:
            parent_ids = {
                str(row["parent_message_id"])
                for row in live.values()
                if row["parent_message_id"] is not None
                and str(row["parent_message_id"]) in live
            }
            leaves = [message_id for message_id in live if message_id not in parent_ids]
            if len(leaves) != 1:
                return None
            active_leaf = leaves[0]
        if not isinstance(active_leaf, str) or active_leaf not in live:
            return None

        ordered: list[sqlite3.Row] = []
        seen: set[str] = set()
        current_id: str | None = active_leaf
        while current_id is not None:
            if current_id in seen:
                return None
            seen.add(current_id)
            row = live.get(current_id)
            if row is None:
                return None
            ordered.append(row)
            parent = row["parent_message_id"]
            current_id = None if parent is None else str(parent)
        ordered.reverse()

        variant_groups: dict[str, list[sqlite3.Row]] = {}
        for candidate in live.values():
            variant_root = str(candidate["variant_of"] or candidate["id"])
            variant_groups.setdefault(variant_root, []).append(candidate)
        for row in ordered:
            variant_of = row["variant_of"]
            root_id = str(variant_of) if variant_of is not None else str(row["id"])
            group = variant_groups[root_id]
            if len(group) == 1 and variant_of is None:
                continue
            selected = [
                candidate
                for candidate in group
                if bool(candidate["is_selected_variant"])
            ]
            if len(selected) != 1 or selected[0]["id"] != row["id"]:
                return None

        eligible = [
            (str(row["id"]), str(row["content"]))
            for row in ordered
            if row["role"] in ("user", "assistant")
        ]
        if not eligible:
            return None
        title = str(conversation["title"] or "Untitled conversation")
        digest_input = json.dumps(
            {"policy": 1, "title": title, "messages": eligible},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        revision = self._read_revision(connection)
        target = LocalCharacterConversationTarget(
            character=ResolvedLocalCharacterKey(
                authority, int(conversation["character_id"])
            ),
            conversation_id=str(conversation["id"]),
        )
        return EligibleConversationDocument(
            target=target,
            title=title,
            body="\n\n".join(content for _message_id, content in eligible),
            source_revision=revision,
            eligibility_digest=hashlib.sha256(digest_input).hexdigest(),
        )

    @staticmethod
    def _read_revision(connection: sqlite3.Connection) -> int:
        try:
            row = connection.execute(
                "SELECT data_revision FROM character_conversation_search_revision "
                "WHERE singleton_id = 1"
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
        return 0 if row is None else int(row[0])


class CharacterConversationSearchRepository:
    """SQLite-backed browse, Keyword, and repair operations for the façade."""

    _POLICY_VERSION = 1
    _BACKFILL_BATCH_SIZE = 128

    def __init__(
        self,
        database: CharactersRAGDB,
        *,
        current_character: ResolvedLocalCharacterKey | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> None:
        self._database = database
        self._authority = database.get_local_authority_id()
        if (
            current_character is not None
            and current_character.data_authority_id != self._authority
        ):
            current_character = None
        self._current_character = current_character
        self._progress_callback = progress_callback
        self._projector = SelectedBranchEligibilityProjector(database)

    def recent_groups(
        self, *, group_limit: int = 4, row_limit: int = 5
    ) -> tuple[CharacterConversationGroup, ...]:
        """Return section-first recent groups, force-including the current card."""

        group_limit = self._bounded_limit(group_limit, maximum=4)
        row_limit = self._bounded_limit(row_limit, maximum=5)
        with self._database.transaction() as connection:
            current_card = self._resolved_current_card(connection)
            unavailable_total = self._unavailable_total(connection)
            reserve_unavailable = unavailable_total > 0 and (
                current_card is None or group_limit > 1
            )
            resolved_slots = (
                group_limit - int(current_card is not None) - int(reserve_unavailable)
            )
            summaries = self._recent_resolved_summaries(
                connection,
                exclude_character_id=(
                    None
                    if current_card is None
                    else self._current_character.character_id
                ),
                limit=resolved_slots,
            )

            groups: list[CharacterConversationGroup] = []
            if current_card is not None:
                groups.append(
                    self._resolved_group(
                        connection,
                        character_id=self._current_character.character_id,
                        character_label=str(current_card["name"]),
                        row_limit=row_limit,
                        is_current=True,
                    )
                )
            groups.extend(
                self._resolved_group(
                    connection,
                    character_id=int(summary["character_id"]),
                    character_label=str(summary["card_name"]),
                    row_limit=row_limit,
                    is_current=False,
                    total=int(summary["total"]),
                )
                for summary in summaries
            )
            if reserve_unavailable:
                groups.append(
                    self._unavailable_group(
                        connection, total=unavailable_total, row_limit=row_limit
                    )
                )
        return tuple(groups)

    def unavailable_page(
        self,
        *,
        offset: int = 0,
        limit: int = 20,
        query: str = "",
    ) -> CharacterConversationPage:
        """Return one explicit page containing only unresolved local chats."""

        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        limit = self._bounded_limit(limit, maximum=50)
        normalized_query = str(query or "").strip()[:200]
        with self._database.transaction() as connection:
            revision = self._revision(connection)
            total = self._unavailable_total(connection, query=normalized_query)
            sources = self._unavailable_sources(
                connection,
                limit=limit,
                offset=offset,
                query=normalized_query,
            )
        return CharacterConversationPage(
            rows=tuple(self._presentation_row(source) for source in sources),
            total=total,
            next_cursor=None,
            data_revision=revision,
        )

    def page_for_character(
        self,
        key: ResolvedLocalCharacterKey,
        *,
        cursor: CharacterConversationCursor | None = None,
        limit: int = 20,
    ) -> CharacterConversationPage:
        """Return one stable descending keyset page for an exact local character."""

        limit = self._bounded_limit(limit, maximum=20)
        if key.data_authority_id != self._authority:
            return CharacterConversationPage((), 0, None, self._revision())
        cursor_sql = ""
        params: list[object] = [self._authority, key.character_id]
        if cursor is not None:
            cursor_sql = "AND (c.last_modified, c.created_at, c.id) < (?, ?, ?)"
            params.extend(
                [cursor.last_modified, cursor.created_at, cursor.conversation_id]
            )
        params.append(limit + 1)
        with self._database.transaction() as connection:
            revision = self._revision(connection)
            total = connection.execute(
                """
                SELECT COUNT(*)
                  FROM conversations AS c
                  JOIN character_cards AS card ON card.id = c.character_id
                 WHERE c.deleted = 0 AND card.deleted = 0
                   AND c.runtime_backend = 'local'
                   AND c.assistant_kind = 'character'
                   AND c.assistant_authority_id = ?
                   AND c.character_id = ?
                """,
                (self._authority, key.character_id),
            ).fetchone()[0]
            sources = connection.execute(
                f"""
                SELECT c.id, c.title,
                       CAST(c.last_modified AS TEXT) AS last_modified,
                       CAST(c.created_at AS TEXT) AS created_at,
                       c.character_id,
                       c.assistant_id, c.assistant_authority_id,
                       c.runtime_backend, c.assistant_kind,
                       card.name AS card_name, card.deleted AS card_deleted
                  FROM conversations AS c
                  JOIN character_cards AS card ON card.id = c.character_id
                 WHERE c.deleted = 0 AND card.deleted = 0
                   AND c.runtime_backend = 'local'
                   AND c.assistant_kind = 'character'
                   AND c.assistant_authority_id = ?
                   AND c.character_id = ?
                   {cursor_sql}
                 ORDER BY c.last_modified DESC, c.created_at DESC, c.id DESC
                 LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        has_more = len(sources) > limit
        visible = sources[:limit]
        result_rows = tuple(self._presentation_row(source) for source in visible)
        next_cursor = None
        if has_more and result_rows:
            last = result_rows[-1]
            assert last.target is not None
            next_cursor = CharacterConversationCursor(
                last.last_modified, last.created_at, last.target.conversation_id
            )
        return CharacterConversationPage(result_rows, int(total), next_cursor, revision)

    def keyword_search(
        self,
        query: str,
        *,
        character: ResolvedLocalCharacterKey | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> CharacterConversationPage:
        """Search one read snapshot and fence it against a final source revision."""

        limit = self._bounded_limit(limit, maximum=50)
        if character is not None and character.data_authority_id != self._authority:
            return CharacterConversationPage((), 0, None, self._revision())
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if not isinstance(query, str) or not query.strip():
            revision = self._revision()
            return CharacterConversationPage(
                (), 0, None, revision, self.keyword_index_status()
            )
        for _attempt in range(2):
            if character is None:
                page = self._keyword_search_snapshot(query, offset=offset, limit=limit)
            else:
                page = self._keyword_search_snapshot(
                    query, character=character, offset=offset, limit=limit
                )
            if self._revision() == page.data_revision:
                return page
        revision = self._revision()
        return CharacterConversationPage(
            (), 0, None, revision, CharacterKeywordIndexStatus.ABSENT
        )

    def _keyword_search_snapshot(
        self,
        query: str,
        *,
        character: ResolvedLocalCharacterKey | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> CharacterConversationPage:
        match_query = '"' + query.strip().replace('"', '""') + '"'
        with self._database.transaction() as connection:
            revision = self._revision(connection)
            generation = connection.execute(
                """
                SELECT generation_id
                  FROM character_conversation_search_generations
                 WHERE data_authority_id = ? AND status = 'ready'
                   AND policy_version = ? AND source_revision = ?
                 LIMIT 1
                """,
                (self._authority, self._POLICY_VERSION, revision),
            ).fetchone()
            if generation is None:
                status = self._keyword_status_for_revision(connection, revision)
                return CharacterConversationPage((), 0, None, revision, status)
            search_params = (
                match_query,
                self._authority,
                generation["generation_id"],
                revision,
                character.character_id if character is not None else None,
                character.character_id if character is not None else None,
            )
            total = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                      FROM character_conversation_fts
                      JOIN character_conversation_search_documents AS d
                        ON d.document_id = character_conversation_fts.rowid
                      JOIN conversations AS c ON c.id = d.conversation_id
                      JOIN character_cards AS card ON card.id = c.character_id
                     WHERE character_conversation_fts MATCH ?
                       AND d.data_authority_id = ? AND d.generation_id = ?
                       AND d.source_revision = ?
                       AND (? IS NULL OR d.character_id = ?)
                       AND d.eligibility_digest = d.validated_eligibility_digest
                       AND c.deleted = 0 AND card.deleted = 0
                       AND c.runtime_backend = 'local'
                       AND c.assistant_kind = 'character'
                       AND c.assistant_authority_id = d.data_authority_id
                       AND c.character_id = d.character_id
                    """,
                    search_params,
                ).fetchone()[0]
            )
            validated: list[CharacterConversationRow] = []
            rejected = 0
            scanned = 0
            candidate_offset = offset
            batch_limit = limit
            while scanned < 50 and candidate_offset < total:
                candidates = self._keyword_candidates(
                    connection,
                    search_params=search_params,
                    limit=min(batch_limit, 50 - scanned),
                    offset=candidate_offset,
                )
                if not candidates:
                    break
                candidate_offset += len(candidates)
                scanned += len(candidates)
                batch_rejected = 0
                for candidate in candidates:
                    current = self._projector.project(
                        str(candidate["conversation_id"]), connection=connection
                    )
                    if (
                        current is None
                        or current.source_revision != revision
                        or current.eligibility_digest != candidate["eligibility_digest"]
                    ):
                        batch_rejected += 1
                        continue
                    validated.append(
                        self._presentation_row(
                            candidate, selected_excerpt=str(candidate["body"])[:240]
                        )
                    )
                rejected += batch_rejected
                if rejected == 0 or candidate_offset >= total:
                    break
                batch_limit = max(1, min(limit, total - candidate_offset))
            if rejected and (offset != 0 or candidate_offset < total):
                return CharacterConversationPage(
                    (), 0, None, revision, CharacterKeywordIndexStatus.FAILED
                )
            return CharacterConversationPage(
                tuple(validated[:limit]),
                total - rejected,
                None,
                revision,
                CharacterKeywordIndexStatus.READY,
            )

    @staticmethod
    def _keyword_candidates(
        connection: sqlite3.Connection,
        *,
        search_params: tuple[str, str, str, int, int | None, int | None],
        limit: int,
        offset: int,
    ) -> list[sqlite3.Row]:
        return connection.execute(
            """
            SELECT d.conversation_id, d.title, d.body,
                   d.eligibility_digest, d.source_revision,
                   CAST(c.last_modified AS TEXT) AS last_modified,
                   CAST(c.created_at AS TEXT) AS created_at,
                   c.character_id, c.assistant_id,
                   c.assistant_authority_id, c.runtime_backend,
                   c.assistant_kind, card.name AS card_name,
                   card.deleted AS card_deleted,
                   bm25(character_conversation_fts) AS search_rank
              FROM character_conversation_fts
              JOIN character_conversation_search_documents AS d
                ON d.document_id = character_conversation_fts.rowid
              JOIN conversations AS c ON c.id = d.conversation_id
              JOIN character_cards AS card ON card.id = c.character_id
             WHERE character_conversation_fts MATCH ?
               AND d.data_authority_id = ? AND d.generation_id = ?
               AND d.source_revision = ?
               AND (? IS NULL OR d.character_id = ?)
               AND d.eligibility_digest = d.validated_eligibility_digest
               AND c.deleted = 0 AND card.deleted = 0
               AND c.runtime_backend = 'local'
               AND c.assistant_kind = 'character'
               AND c.assistant_authority_id = d.data_authority_id
               AND c.character_id = d.character_id
             ORDER BY search_rank, c.last_modified DESC, c.id DESC
             LIMIT ? OFFSET ?
            """,
            (*search_params, limit, offset),
        ).fetchall()

    def repair_candidates(
        self, key: UnresolvedConversationKey
    ) -> tuple[CharacterRepairCandidate, ...]:
        """Enumerate only live cards owned by the unresolved key's authority."""

        if key.data_authority_id != self._authority:
            return ()
        with self._database.transaction() as connection:
            conversation = connection.execute(
                "SELECT c.character_id, c.assistant_authority_id, "
                "card.id AS live_card_id FROM conversations AS c "
                "LEFT JOIN character_cards AS card "
                "ON card.id = c.character_id AND card.deleted = 0 "
                "WHERE c.id = ? AND c.deleted = 0 "
                "AND c.runtime_backend = 'local' AND c.assistant_kind = 'character'",
                (key.conversation_id,),
            ).fetchone()
            if conversation is None or self._is_resolved_conversation(conversation):
                return ()
            rows = connection.execute(
                "SELECT id, name, version FROM character_cards "
                "WHERE deleted = 0 ORDER BY name COLLATE NOCASE, id"
            ).fetchall()
        return tuple(
            CharacterRepairCandidate(
                ResolvedLocalCharacterKey(self._authority, int(row["id"])),
                str(row["name"]),
                int(row["version"]),
            )
            for row in rows
        )

    def refresh_unresolved_evidence(
        self, key: UnresolvedConversationKey
    ) -> tuple[int, str] | None:
        """Read the current unresolved identity evidence and CAS version atomically."""

        if key.data_authority_id != self._authority:
            return None
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT c.version, c.assistant_id, c.character_id, "
                "c.assistant_authority_id, card.id AS live_card_id "
                "FROM conversations AS c LEFT JOIN character_cards AS card "
                "ON card.id = c.character_id AND card.deleted = 0 "
                "WHERE c.id = ? AND c.deleted = 0 "
                "AND c.runtime_backend = 'local' AND c.assistant_kind = 'character'",
                (key.conversation_id,),
            ).fetchone()
            if row is None or self._is_resolved_conversation(row):
                return None
            historical = str(row["assistant_id"] or row["character_id"] or "Unknown")
            return int(row["version"]), historical

    def validated_preview_messages(
        self,
        target: LocalCharacterConversationTarget,
        *,
        data_revision: int,
        limit: int = 200,
    ) -> tuple[dict[str, Any], ...] | None:
        """Atomically fence a preview read by revision and exact authority."""

        limit = self._bounded_limit(limit, maximum=200)
        if target.character.data_authority_id != self._authority:
            return None
        with self._database.transaction() as connection:
            if self._revision(connection) != data_revision:
                return None
            owner = connection.execute(
                "SELECT c.deleted, c.runtime_backend, c.assistant_kind, "
                "c.assistant_authority_id, c.character_id, "
                "card.id AS live_card_id, card.deleted AS card_deleted "
                "FROM conversations AS c "
                "LEFT JOIN character_cards AS card ON card.id = c.character_id "
                "WHERE c.id = ?",
                (target.conversation_id,),
            ).fetchone()
            if (
                owner is None
                or bool(owner["deleted"])
                or owner["runtime_backend"] != "local"
                or owner["assistant_kind"] != "character"
                or owner["assistant_authority_id"] != self._authority
                or owner["character_id"] != target.character.character_id
                or owner["live_card_id"] is None
                or bool(owner["card_deleted"])
            ):
                return None
            rows = connection.execute(
                "SELECT m.* FROM messages AS m "
                "WHERE m.conversation_id = ? AND m.deleted = 0 "
                "ORDER BY m.timestamp ASC, m.id ASC LIMIT ?",
                (target.conversation_id, limit),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def repair(self, request: CharacterRepairRequest) -> CharacterRepairResult:
        """Compare-and-set one exact unresolved local conversation identity."""

        if (
            request.unresolved.data_authority_id != self._authority
            or request.replacement.data_authority_id != self._authority
        ):
            return CharacterRepairResult.INVALID_CANDIDATE
        with self._database.transaction(immediate=True) as connection:
            candidate = connection.execute(
                "SELECT version FROM character_cards WHERE id = ? AND deleted = 0",
                (request.replacement.character_id,),
            ).fetchone()
            if candidate is None:
                return CharacterRepairResult.INVALID_CANDIDATE
            conversation = connection.execute(
                "SELECT c.version, c.character_id, c.assistant_authority_id, "
                "card.id AS live_card_id FROM conversations AS c "
                "LEFT JOIN character_cards AS card "
                "ON card.id = c.character_id AND card.deleted = 0 "
                "WHERE c.id = ? AND c.deleted = 0 "
                "AND c.runtime_backend = 'local' AND c.assistant_kind = 'character'",
                (request.unresolved.conversation_id,),
            ).fetchone()
            if conversation is None:
                return CharacterRepairResult.NOT_FOUND
            if self._is_resolved_conversation(conversation):
                return CharacterRepairResult.INVALID_CANDIDATE
            if int(conversation["version"]) != request.expected_conversation_version:
                return CharacterRepairResult.STALE_VERSION
            updated = connection.execute(
                """
                UPDATE conversations
                   SET character_id = ?, assistant_id = ?,
                       assistant_authority_id = ?, version = version + 1,
                       last_modified = CURRENT_TIMESTAMP, client_id = ?
                 WHERE id = ? AND version = ? AND deleted = 0
                   AND runtime_backend = 'local'
                   AND assistant_kind = 'character'
                """,
                (
                    request.replacement.character_id,
                    str(request.replacement.character_id),
                    self._authority,
                    self._database.client_id,
                    request.unresolved.conversation_id,
                    request.expected_conversation_version,
                ),
            )
            if updated.rowcount != 1:
                return CharacterRepairResult.STALE_VERSION
            connection.execute(
                "DELETE FROM character_conversation_search_documents "
                "WHERE data_authority_id = ? AND conversation_id = ?",
                (self._authority, request.unresolved.conversation_id),
            )
        return CharacterRepairResult.APPLIED

    def ensure_keyword_index(self) -> CharacterKeywordIndexStatus:
        """Activate and build or reconcile the local Keyword generation."""

        with self._database.transaction(immediate=True) as connection:
            revision = self._revision(connection)
            connection.execute(
                "UPDATE character_conversation_search_state "
                "SET data_authority_id = ?, active_policy_version = ?, "
                "activated = 1, updated_at = CURRENT_TIMESTAMP "
                "WHERE singleton_id = 1",
                (self._authority, self._POLICY_VERSION),
            )
            current = connection.execute(
                """
                SELECT generation_id, status,
                       lease_expires_at > CURRENT_TIMESTAMP AS lease_valid
                  FROM character_conversation_search_generations
                 WHERE data_authority_id = ? AND policy_version = ?
                   AND source_revision = ?
                 ORDER BY rowid DESC LIMIT 1
                """,
                (self._authority, self._POLICY_VERSION, revision),
            ).fetchone()
            if current is not None and current["status"] == "ready":
                return CharacterKeywordIndexStatus.READY
            if (
                current is not None
                and current["status"] == "building"
                and bool(current["lease_valid"])
            ):
                return CharacterKeywordIndexStatus.BUILDING
            if current is not None:
                connection.execute(
                    "DELETE FROM character_conversation_search_generations "
                    "WHERE generation_id = ?",
                    (current["generation_id"],),
                )
            ready = connection.execute(
                "SELECT generation_id FROM character_conversation_search_generations "
                "WHERE data_authority_id = ? AND policy_version = ? "
                "AND status = 'ready' LIMIT 1",
                (self._authority, self._POLICY_VERSION),
            ).fetchone()
            if ready is not None:
                reconcile_generation = str(ready["generation_id"])
            else:
                reconcile_generation = None
        if reconcile_generation is not None:
            maintained = self._apply_dirty_generation(reconcile_generation)
            if maintained is not None:
                return maintained
            return self._reconcile_generation(reconcile_generation)

        with self._database.transaction(immediate=True) as connection:
            revision = self._revision(connection)
            generation_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO character_conversation_search_generations(
                    generation_id, data_authority_id, status,
                    policy_version, source_revision, lease_expires_at
                ) VALUES(?, ?, 'building', ?, ?, DATETIME('now', '+5 minutes'))
                """,
                (
                    generation_id,
                    self._authority,
                    self._POLICY_VERSION,
                    revision,
                ),
            )

        try:
            processed = 0
            batch_processed = 0
            documents: list[EligibleConversationDocument] = []
            reported_at = time.monotonic()
            last_id = ""
            while True:
                with self._database.transaction() as connection:
                    ids = [
                        str(row[0])
                        for row in connection.execute(
                            """
                            SELECT c.id
                              FROM conversations AS c
                              JOIN character_cards AS card
                                ON card.id = c.character_id
                             WHERE c.deleted = 0 AND card.deleted = 0
                               AND c.runtime_backend = 'local'
                               AND c.assistant_kind = 'character'
                               AND c.assistant_authority_id = ?
                               AND c.id > ?
                             ORDER BY c.id
                             LIMIT 128
                            """,
                            (self._authority, last_id),
                        ).fetchall()
                    ]
                if not ids:
                    break
                last_id = ids[-1]
                for conversation_id in ids:
                    document = self._projector.project(conversation_id)
                    if document is not None:
                        documents.append(document)
                    processed += 1
                    batch_processed += 1
                    now = time.monotonic()
                    report_progress = (
                        batch_processed >= self._BACKFILL_BATCH_SIZE
                        or now - reported_at >= 1.0
                    )
                    if report_progress:
                        self._replace_documents(
                            generation_id, tuple(documents), processed=processed
                        )
                        documents.clear()
                        batch_processed = 0
                        if self._progress_callback is not None:
                            self._progress_callback(processed)
                        reported_at = now
                if len(ids) < self._BACKFILL_BATCH_SIZE:
                    break
            if batch_processed:
                self._replace_documents(
                    generation_id, tuple(documents), processed=processed
                )
            with self._database.transaction(immediate=True) as connection:
                if self._revision(connection) != revision:
                    connection.execute(
                        "UPDATE character_conversation_search_generations "
                        "SET status = 'failed', error_code = 'source_changed', "
                        "updated_at = CURRENT_TIMESTAMP "
                        "WHERE generation_id = ?",
                        (generation_id,),
                    )
                    connection.execute(
                        "DELETE FROM character_conversation_search_documents "
                        "WHERE generation_id = ?",
                        (generation_id,),
                    )
                    return CharacterKeywordIndexStatus.FAILED
                connection.execute(
                    "DELETE FROM character_conversation_search_generations "
                    "WHERE data_authority_id = ? AND generation_id != ?",
                    (self._authority, generation_id),
                )
                connection.execute(
                    "UPDATE character_conversation_search_generations "
                    "SET status = 'ready', processed_conversations = ?, "
                    "completed_at = CURRENT_TIMESTAMP, lease_expires_at = NULL, "
                    "updated_at = CURRENT_TIMESTAMP WHERE generation_id = ?",
                    (processed, generation_id),
                )
                connection.execute(
                    "DELETE FROM character_conversation_search_dirty "
                    "WHERE data_authority_id = ?",
                    (self._authority,),
                )
            return CharacterKeywordIndexStatus.READY
        except Exception:  # noqa: BLE001 - persist a typed failed generation for callers
            with self._database.transaction(immediate=True) as connection:
                connection.execute(
                    "UPDATE character_conversation_search_generations "
                    "SET status = 'failed', error_code = 'build_failed', "
                    "lease_expires_at = NULL, updated_at = CURRENT_TIMESTAMP "
                    "WHERE generation_id = ?",
                    (generation_id,),
                )
                connection.execute(
                    "DELETE FROM character_conversation_search_documents "
                    "WHERE generation_id = ?",
                    (generation_id,),
                )
            return CharacterKeywordIndexStatus.FAILED

    def keyword_index_status(self) -> CharacterKeywordIndexStatus:
        """Return status for the current authority, policy, and source revision."""

        with self._database.transaction() as connection:
            revision = self._revision(connection)
            return self._keyword_status_for_revision(connection, revision)

    def _keyword_status_for_revision(
        self, connection: sqlite3.Connection, revision: int
    ) -> CharacterKeywordIndexStatus:
        row = connection.execute(
            "SELECT status, lease_expires_at > CURRENT_TIMESTAMP AS lease_valid "
            "FROM character_conversation_search_generations "
            "WHERE data_authority_id = ? AND policy_version = ? "
            "AND source_revision = ? ORDER BY rowid DESC LIMIT 1",
            (self._authority, self._POLICY_VERSION, revision),
        ).fetchone()
        if row is not None:
            if row["status"] == "building" and not bool(row["lease_valid"]):
                return CharacterKeywordIndexStatus.FAILED
            return CharacterKeywordIndexStatus(str(row["status"]))
        failed = connection.execute(
            "SELECT 1 FROM character_conversation_search_generations "
            "WHERE data_authority_id = ? AND policy_version = ? "
            "AND status = 'failed' ORDER BY rowid DESC LIMIT 1",
            (self._authority, self._POLICY_VERSION),
        ).fetchone()
        if failed is not None:
            return CharacterKeywordIndexStatus.FAILED
        return CharacterKeywordIndexStatus.ABSENT

    def reconcile_keyword_index(self) -> CharacterKeywordIndexStatus:
        """Rebuild the activated ready generation from authoritative SQLite."""

        with self._database.transaction() as connection:
            state = connection.execute(
                "SELECT activated FROM character_conversation_search_state "
                "WHERE singleton_id = 1"
            ).fetchone()
            ready = connection.execute(
                "SELECT generation_id FROM character_conversation_search_generations "
                "WHERE data_authority_id = ? AND policy_version = ? "
                "AND status = 'ready' LIMIT 1",
                (self._authority, self._POLICY_VERSION),
            ).fetchone()
        if state is None or not state["activated"] or ready is None:
            return CharacterKeywordIndexStatus.ABSENT
        return self._reconcile_generation(str(ready["generation_id"]))

    def _apply_dirty_generation(
        self, generation_id: str
    ) -> CharacterKeywordIndexStatus | None:
        """Atomically replace/remove only ledger entries; return None if missed."""

        try:
            with self._database.transaction(immediate=True) as connection:
                revision = self._revision(connection)
                dirty_exists = connection.execute(
                    "SELECT 1 FROM character_conversation_search_dirty "
                    "WHERE data_authority_id = ? LIMIT 1",
                    (self._authority,),
                ).fetchone()
                if dirty_exists is None:
                    return None
                processed = 0
                last_id = ""
                while True:
                    ids = [
                        str(row[0])
                        for row in connection.execute(
                            "SELECT conversation_id "
                            "FROM character_conversation_search_dirty "
                            "WHERE data_authority_id = ? AND conversation_id > ? "
                            "ORDER BY conversation_id LIMIT 128",
                            (self._authority, last_id),
                        ).fetchall()
                    ]
                    if not ids:
                        break
                    last_id = ids[-1]
                    for conversation_id in ids:
                        document = self._projector.project(
                            conversation_id, connection=connection
                        )
                        if document is None:
                            connection.execute(
                                "DELETE FROM character_conversation_search_documents "
                                "WHERE data_authority_id = ? AND generation_id = ? "
                                "AND conversation_id = ?",
                                (self._authority, generation_id, conversation_id),
                            )
                        else:
                            self._upsert_document(connection, generation_id, document)
                        processed += 1
                    if len(ids) < self._BACKFILL_BATCH_SIZE:
                        break
                connection.execute(
                    "UPDATE character_conversation_search_documents "
                    "SET source_revision = ? WHERE data_authority_id = ? "
                    "AND generation_id = ?",
                    (revision, self._authority, generation_id),
                )
                connection.execute(
                    "UPDATE character_conversation_search_generations "
                    "SET source_revision = ?, processed_conversations = "
                    "processed_conversations + ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE generation_id = ? AND status = 'ready'",
                    (revision, processed, generation_id),
                )
                connection.execute(
                    "DELETE FROM character_conversation_search_dirty "
                    "WHERE data_authority_id = ?",
                    (self._authority,),
                )
                connection.execute(
                    "DELETE FROM character_conversation_search_generations "
                    "WHERE data_authority_id = ? AND generation_id != ?",
                    (self._authority, generation_id),
                )
            return CharacterKeywordIndexStatus.READY
        except Exception:  # noqa: BLE001 - maintenance is a typed boundary
            return CharacterKeywordIndexStatus.FAILED

    def _reconcile_generation(self, generation_id: str) -> CharacterKeywordIndexStatus:
        try:
            with self._database.transaction(immediate=True) as connection:
                revision = self._revision(connection)
                processed = 0
                last_id = ""
                while True:
                    ids = [
                        str(row[0])
                        for row in connection.execute(
                            """
                            SELECT conversation_id FROM (
                                SELECT id AS conversation_id FROM conversations
                                 WHERE runtime_backend = 'local'
                                   AND assistant_kind = 'character'
                                UNION
                                SELECT conversation_id
                                  FROM character_conversation_search_documents
                                 WHERE data_authority_id = ? AND generation_id = ?
                            )
                             WHERE conversation_id > ?
                             ORDER BY conversation_id
                             LIMIT 128
                            """,
                            (self._authority, generation_id, last_id),
                        ).fetchall()
                    ]
                    if not ids:
                        break
                    last_id = ids[-1]
                    for conversation_id in ids:
                        document = self._projector.project(
                            conversation_id, connection=connection
                        )
                        if document is None:
                            connection.execute(
                                "DELETE FROM character_conversation_search_documents "
                                "WHERE data_authority_id = ? AND generation_id = ? "
                                "AND conversation_id = ?",
                                (self._authority, generation_id, conversation_id),
                            )
                        else:
                            self._upsert_document(connection, generation_id, document)
                        processed += 1
                    if len(ids) < self._BACKFILL_BATCH_SIZE:
                        break
                connection.execute(
                    "UPDATE character_conversation_search_documents "
                    "SET source_revision = ? WHERE data_authority_id = ? "
                    "AND generation_id = ?",
                    (revision, self._authority, generation_id),
                )
                connection.execute(
                    "UPDATE character_conversation_search_generations "
                    "SET source_revision = ?, processed_conversations = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE generation_id = ?",
                    (revision, processed, generation_id),
                )
                connection.execute(
                    "DELETE FROM character_conversation_search_dirty "
                    "WHERE data_authority_id = ?",
                    (self._authority,),
                )
                connection.execute(
                    "DELETE FROM character_conversation_search_generations "
                    "WHERE data_authority_id = ? AND generation_id != ?",
                    (self._authority, generation_id),
                )
            return CharacterKeywordIndexStatus.READY
        except Exception:  # noqa: BLE001 - reconciliation is a typed boundary
            return CharacterKeywordIndexStatus.FAILED

    def _replace_documents(
        self,
        generation_id: str,
        documents: tuple[EligibleConversationDocument, ...],
        *,
        processed: int,
    ) -> None:
        with self._database.transaction(immediate=True) as connection:
            for document in documents:
                self._upsert_document(connection, generation_id, document)
            connection.execute(
                "UPDATE character_conversation_search_generations "
                "SET processed_conversations = ?, "
                "lease_expires_at = DATETIME('now', '+5 minutes'), "
                "updated_at = CURRENT_TIMESTAMP WHERE generation_id = ? "
                "AND status = 'building'",
                (processed, generation_id),
            )

    @staticmethod
    def _upsert_document(
        connection: sqlite3.Connection,
        generation_id: str,
        document: EligibleConversationDocument,
    ) -> None:
        character = connection.execute(
            "SELECT name FROM character_cards WHERE id = ? AND deleted = 0",
            (document.target.character.character_id,),
        ).fetchone()
        if character is None:
            return
        connection.execute(
            """
            INSERT INTO character_conversation_search_documents(
                data_authority_id, conversation_id, character_id,
                character_label, title, body, eligibility_digest,
                validated_eligibility_digest, source_revision, generation_id
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(data_authority_id, generation_id, conversation_id)
            DO UPDATE SET character_id = excluded.character_id,
                          character_label = excluded.character_label,
                          title = excluded.title,
                          body = excluded.body,
                          eligibility_digest = excluded.eligibility_digest,
                          validated_eligibility_digest =
                              excluded.validated_eligibility_digest,
                          source_revision = excluded.source_revision
            """,
            (
                document.target.character.data_authority_id,
                document.target.conversation_id,
                document.target.character.character_id,
                str(character["name"]),
                document.title,
                document.body,
                document.eligibility_digest,
                document.eligibility_digest,
                document.source_revision,
                generation_id,
            ),
        )

    def _resolved_current_card(
        self, connection: sqlite3.Connection
    ) -> sqlite3.Row | None:
        if self._current_character is None:
            return None
        return connection.execute(
            "SELECT id, name FROM character_cards WHERE id = ? AND deleted = 0",
            (self._current_character.character_id,),
        ).fetchone()

    def _unavailable_total(
        self,
        connection: sqlite3.Connection,
        *,
        query: str = "",
    ) -> int:
        escaped_query = self._escape_like_query(query)
        query_pattern = f"%{escaped_query.casefold()}%"
        return int(
            connection.execute(
                """
                SELECT COUNT(*)
                  FROM conversations AS c
                  LEFT JOIN character_cards AS card ON card.id = c.character_id
                 WHERE c.deleted = 0 AND c.runtime_backend = 'local'
                   AND c.assistant_kind = 'character'
                   AND (c.assistant_authority_id IS NOT ?
                        OR typeof(c.character_id) != 'integer'
                        OR c.character_id < 1
                        OR card.id IS NULL OR card.deleted != 0)
                   AND (? = ''
                        OR LOWER(COALESCE(c.title, '')) LIKE ? ESCAPE '\\'
                        OR LOWER(COALESCE(c.id, '')) LIKE ? ESCAPE '\\'
                        OR LOWER(COALESCE(c.assistant_id, '')) LIKE ? ESCAPE '\\'
                        OR LOWER(COALESCE(card.name, '')) LIKE ? ESCAPE '\\')
                """,
                (
                    self._authority,
                    escaped_query,
                    query_pattern,
                    query_pattern,
                    query_pattern,
                    query_pattern,
                ),
            ).fetchone()[0]
        )

    @staticmethod
    def _escape_like_query(query: str) -> str:
        """Escape a literal user query for SQLite LIKE with ``\\`` escape."""

        return query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def _recent_resolved_summaries(
        self,
        connection: sqlite3.Connection,
        *,
        exclude_character_id: int | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        if limit <= 0:
            return []
        return connection.execute(
            """
            SELECT c.character_id, card.name AS card_name,
                   COUNT(*) AS total,
                   MAX(CAST(c.last_modified AS TEXT)) AS latest
              FROM conversations AS c
              JOIN character_cards AS card
                ON card.id = c.character_id AND card.deleted = 0
             WHERE c.deleted = 0 AND c.runtime_backend = 'local'
               AND c.assistant_kind = 'character'
               AND c.assistant_authority_id = ?
               AND (? IS NULL OR c.character_id != ?)
             GROUP BY c.character_id, card.name
             ORDER BY latest DESC, c.character_id DESC
             LIMIT ?
            """,
            (
                self._authority,
                exclude_character_id,
                exclude_character_id,
                limit,
            ),
        ).fetchall()

    def _resolved_group(
        self,
        connection: sqlite3.Connection,
        *,
        character_id: int,
        character_label: str,
        row_limit: int,
        is_current: bool,
        total: int | None = None,
    ) -> CharacterConversationGroup:
        if total is None:
            total = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM conversations AS c
                     WHERE c.deleted = 0 AND c.runtime_backend = 'local'
                       AND c.assistant_kind = 'character'
                       AND c.assistant_authority_id = ? AND c.character_id = ?
                    """,
                    (self._authority, character_id),
                ).fetchone()[0]
            )
        sources = connection.execute(
            """
            SELECT c.id, c.title,
                   CAST(c.last_modified AS TEXT) AS last_modified,
                   CAST(c.created_at AS TEXT) AS created_at,
                   c.character_id, c.assistant_id, c.assistant_authority_id,
                   c.runtime_backend, c.assistant_kind,
                   card.name AS card_name, card.deleted AS card_deleted
              FROM conversations AS c
              JOIN character_cards AS card
                ON card.id = c.character_id AND card.deleted = 0
             WHERE c.deleted = 0 AND c.runtime_backend = 'local'
               AND c.assistant_kind = 'character'
               AND c.assistant_authority_id = ? AND c.character_id = ?
             ORDER BY c.last_modified DESC, c.created_at DESC, c.id DESC
             LIMIT ?
            """,
            (self._authority, character_id, row_limit),
        ).fetchall()
        key = ResolvedLocalCharacterKey(self._authority, character_id)
        return CharacterConversationGroup(
            key=key,
            character_label=character_label,
            rows=tuple(self._presentation_row(source) for source in sources),
            total=total,
            is_current=is_current,
        )

    def _unavailable_group(
        self,
        connection: sqlite3.Connection,
        *,
        total: int,
        row_limit: int,
    ) -> CharacterConversationGroup:
        sources = self._unavailable_sources(
            connection,
            limit=row_limit,
            offset=0,
        )
        return CharacterConversationGroup(
            key=UnresolvedConversationKey(
                self._authority, "unavailable-character-conversations"
            ),
            character_label="Chats with unavailable characters",
            rows=tuple(self._presentation_row(source) for source in sources),
            total=total,
            is_current=False,
        )

    def _unavailable_sources(
        self,
        connection: sqlite3.Connection,
        *,
        limit: int,
        offset: int,
        query: str = "",
    ) -> list[sqlite3.Row]:
        escaped_query = self._escape_like_query(query)
        query_pattern = f"%{escaped_query.casefold()}%"
        return connection.execute(
            """
            SELECT c.id, c.title,
                   CAST(c.last_modified AS TEXT) AS last_modified,
                   CAST(c.created_at AS TEXT) AS created_at,
                   c.character_id,
                   c.assistant_id, c.assistant_authority_id,
                   c.runtime_backend, c.assistant_kind,
                   card.name AS card_name, card.deleted AS card_deleted
              FROM conversations AS c
              LEFT JOIN character_cards AS card ON card.id = c.character_id
             WHERE c.deleted = 0 AND c.runtime_backend = 'local'
               AND c.assistant_kind = 'character'
               AND (c.assistant_authority_id IS NOT ?
                    OR typeof(c.character_id) != 'integer'
                    OR c.character_id < 1
                    OR card.id IS NULL OR card.deleted != 0)
               AND (? = ''
                    OR LOWER(COALESCE(c.title, '')) LIKE ? ESCAPE '\\'
                    OR LOWER(COALESCE(c.id, '')) LIKE ? ESCAPE '\\'
                    OR LOWER(COALESCE(c.assistant_id, '')) LIKE ? ESCAPE '\\'
                    OR LOWER(COALESCE(card.name, '')) LIKE ? ESCAPE '\\')
             ORDER BY c.last_modified DESC, c.created_at DESC, c.id DESC
             LIMIT ? OFFSET ?
            """,
            (
                self._authority,
                escaped_query,
                query_pattern,
                query_pattern,
                query_pattern,
                query_pattern,
                limit,
                offset,
            ),
        ).fetchall()

    def _presentation_row(
        self, source: Any, *, selected_excerpt: str = ""
    ) -> CharacterConversationRow:
        source_keys = source.keys()
        conversation_id = str(
            source["conversation_id"]
            if "conversation_id" in source_keys
            else source["id"]
        )
        card_name = source["card_name"]
        card_deleted = source["card_deleted"]
        character_id = source["character_id"]
        authority = source["assistant_authority_id"]
        label = str(card_name or source["assistant_id"] or "Unavailable character")
        title = str(source["title"] or "Untitled conversation")
        modified = str(source["last_modified"])
        resolved = (
            authority == self._authority
            and isinstance(character_id, int)
            and character_id > 0
            and card_name is not None
            and not card_deleted
        )
        if resolved:
            target = LocalCharacterConversationTarget(
                ResolvedLocalCharacterKey(self._authority, int(character_id)),
                conversation_id,
            )
            return CharacterConversationRow.resolved(
                target,
                character_label=label,
                title=title,
                last_modified=modified,
                created_at=str(source["created_at"]),
                is_current=target.character == self._current_character,
                selected_excerpt=selected_excerpt,
            )
        if card_name is None:
            reason = UnavailableCharacterReason.MISSING_CARD
        elif card_deleted:
            reason = UnavailableCharacterReason.DELETED_CARD
        elif authority is None:
            reason = UnavailableCharacterReason.AMBIGUOUS_LEGACY_LINK
        else:
            reason = UnavailableCharacterReason.MISSING_CHARACTER_AUTHORITY_LINK
        return CharacterConversationRow.unavailable(
            UnresolvedConversationKey(self._authority, conversation_id),
            reason=reason,
            character_label=label,
            title=title,
            last_modified=modified,
            created_at=str(source["created_at"]),
            selected_excerpt=selected_excerpt,
        )

    def _revision(self, connection: sqlite3.Connection | None = None) -> int:
        if connection is not None:
            row = connection.execute(
                "SELECT data_revision FROM character_conversation_search_revision "
                "WHERE singleton_id = 1"
            ).fetchone()
            return int(row[0])
        return self._database.get_character_conversation_search_revision()

    def _is_resolved_conversation(self, source: Any) -> bool:
        return (
            source["assistant_authority_id"] == self._authority
            and isinstance(source["character_id"], int)
            and source["character_id"] > 0
            and source["live_card_id"] is not None
        )

    @staticmethod
    def _bounded_limit(value: int, *, maximum: int) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
        ):
            raise ValueError(f"limit must be between 1 and {maximum}")
        return value
