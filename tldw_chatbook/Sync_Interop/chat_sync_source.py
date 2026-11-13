"""CharactersRAGDB's Chat Sync v2 surface: the ChatSyncIntentSource adapter.

The committed-intent readers, the restore enumerator and the incoming-apply
local-store methods moved here from ``CharactersRAGDB`` (TASK-36949.4) so the
DB layer no longer imports Sync_Interop. Every statement runs on the wrapped
database's own connection and ``transaction()``; this adapter never opens a
connection or transaction of its own.

The tombstone writers still stamp each deleted message's record hash inside
their own write transaction; they reach :func:`chat_record_payload_hash` through
the DB-layer port in ``DB/transaction_observer.py``, which importing this module
installs (composition roots also install it explicitly).
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping

from tldw_chatbook.Chat.assistant_generation_state import (
    normalize_assistant_generation_state,
)
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
    validated_provider_continuation,
    validated_thinking_blocks_json,
)
from tldw_chatbook.DB.transaction_observer import install_chat_record_hasher
from tldw_chatbook.Provider_Contract.provider_continuation import (
    ContinuationValidationError,
    dump_provider_continuation_json,
    parse_provider_continuation_json,
)
from tldw_chatbook.Sync_Interop.chat_outbox_producer import (
    ChatSyncDeleteIntentRecord,
    ChatSyncIntentRecord,
)
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash


_CHAT_SYNC_INTENT_PAYLOAD_KEYS = frozenset(
    {
        "id",
        "conversation_id",
        "parent_message_id",
        "sender",
        "content",
        "image_mime_type",
        "provider_continuation_json",
        "thinking_blocks_json",
        "assistant_generation_state",
        "timestamp",
        "ranking",
        "last_modified",
        "deleted",
        "client_id",
        "version",
    }
)


def _normalize_legacy_chat_sync_intent_payload(
    payload: object,
) -> dict[str, Any] | None:
    """Add nullable post-v44 keys, then enforce the exact Sync-v1 shape."""
    if type(payload) is not dict:
        return None
    normalized = dict(payload)
    normalized.setdefault("assistant_generation_state", None)
    normalized.setdefault("thinking_blocks_json", None)
    if set(normalized) != _CHAT_SYNC_INTENT_PAYLOAD_KEYS:
        return None
    return normalized


def _normalize_legacy_chat_delete_intent_payload(
    payload: object,
) -> dict[str, Any] | None:
    """Add nullable legacy keys, then enforce the delete intent shape."""
    if type(payload) is not dict:
        return None
    normalized = dict(payload)
    normalized.setdefault("assistant_generation_state", None)
    normalized.setdefault("base_payload_hash", None)
    normalized.setdefault("legacy_pre_v50_base_reconstruction", False)
    if set(normalized) != {
        "id",
        "deleted",
        "last_modified",
        "assistant_generation_state",
        "base_payload_hash",
        "legacy_pre_v50_base_reconstruction",
        "version",
        "client_id",
    }:
        return None
    base_payload_hash = normalized["base_payload_hash"]
    legacy_marker = normalized["legacy_pre_v50_base_reconstruction"]
    if type(legacy_marker) is not bool or (
        legacy_marker and base_payload_hash is not None
    ):
        return None
    if base_payload_hash is not None and (
        type(base_payload_hash) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", base_payload_hash) is None
    ):
        return None
    return normalized


def chat_record_payload_hash(row: Mapping[str, Any]) -> str:
    """Hash one valid live Chat record without retaining its private fields.

    This is the DB-layer chat-record hashing port's implementation: tombstone
    writers stamp its result on each delete intent inside their transaction.

    Args:
        row: A live message row with ``role``, ``content``,
            ``provider_continuation_json``, ``thinking_blocks_json`` and
            ``assistant_generation_state``.

    Returns:
        The canonical ``sha256:<hex>`` Sync v2 payload hash of the record.

    Raises:
        InputError: The row is not a valid Chat record; a tombstone writer
            propagates it and its transaction rolls back.
    """
    role = row["role"]
    content = row["content"]
    if type(role) is not str or type(content) is not str:
        raise InputError("Invalid chat sync record.")
    private_json = row["provider_continuation_json"]
    active_continuation = False
    if private_json is not None:
        if role != "assistant":
            raise InputError("Invalid chat sync record.")
        checkpoint, canonical_private = validated_provider_continuation(private_json)
        if canonical_private != private_json:
            raise InputError("Invalid chat sync record.")
        active_continuation = checkpoint.state == "active"
    try:
        state = normalize_assistant_generation_state(
            role=role,
            raw_state=row["assistant_generation_state"],
            has_valid_active_continuation=active_continuation,
        )
    except ValueError:
        raise InputError("Invalid chat sync record.") from None
    payload = {
        "assistant_generation_state": state.value if state is not None else None,
        "content": content,
        "role": role,
    }
    if private_json is not None:
        payload["provider_continuation_json"] = private_json
    thinking_json = row["thinking_blocks_json"]
    if thinking_json is not None:
        if role != "assistant":
            raise InputError("Invalid chat sync record.")
        canonical_thinking = validated_thinking_blocks_json(thinking_json)
        if canonical_thinking != thinking_json:
            raise InputError("Invalid chat sync record.")
        payload["thinking_blocks_json"] = canonical_thinking
    return canonical_payload_hash(payload)


@dataclass(frozen=True, slots=True, eq=False)
class CharactersRAGDBChatSyncSource:
    """``ChatSyncIntentSource`` and Chat Sync local store over one database.

    Adapters compare by identity; :func:`is_chat_sync_source_over` asks
    whether one wraps exactly a given database object.
    """

    db: CharactersRAGDB

    def read_committed_chat_sync_intent(
        self,
        *,
        message_id: str,
        message_version: int,
        payload_hash: str,
    ) -> ChatSyncIntentRecord | None:
        """Return one exact committed message intent without exposing private data.

        Args:
            message_id: Exact message owner ID.
            message_version: Exact committed message version.
            payload_hash: Expected canonical whole-message payload hash.

        Returns:
            The validated committed source record, or ``None`` when the
            source proof is absent, ambiguous, invalid, or uncommitted.
        """
        if (
            type(message_id) is not str
            or not message_id
            or type(message_version) is not int
            or message_version < 1
            or type(payload_hash) is not str
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", payload_hash)
        ):
            return None
        conn = self.db.get_connection()
        if conn.in_transaction:
            return None
        try:
            query = """
                SELECT m.id, m.conversation_id, m.parent_message_id, m.sender,
                       m.role, m.content, m.image_mime_type,
                       m.provider_continuation_json,
                       m.thinking_blocks_json,
                       m.assistant_generation_state, m.timestamp, m.ranking,
                       m.last_modified, m.deleted, m.client_id, m.version,
                       intent.operation, intent.payload
                  FROM messages AS m
                  JOIN sync_log AS intent
                    ON intent.entity = 'messages'
                   AND intent.entity_id = m.id
                   AND intent.version = m.version
                 WHERE m.id = ? AND m.version = ?
                 ORDER BY intent.change_id
                """
            with self.db.transaction() as conn:
                rows = conn.execute(
                    query,
                    (message_id, message_version),
                ).fetchall()
                if len(rows) != 1:
                    return None
                row = rows[0]
                base_payload_hash = self._previous_committed_chat_payload_hash(
                    conn,
                    message_id=message_id,
                    conversation_id=row["conversation_id"],
                    role=row["role"],
                    provider_continuation_json=row["provider_continuation_json"],
                    message_version=message_version,
                )
                if message_version > 1 and base_payload_hash is None:
                    return None
            if row["deleted"] or row["operation"] not in {"create", "update"}:
                return None
            intent_payload = _normalize_legacy_chat_sync_intent_payload(
                json.loads(row["payload"])
            )
            if intent_payload is None:
                return None

            def intent_value(value: Any) -> Any:
                if isinstance(value, datetime):
                    return (
                        value.astimezone(timezone.utc)
                        .isoformat(timespec="milliseconds")
                        .replace("+00:00", "Z")
                    )
                return value

            role = row["role"]
            content = row["content"]
            if type(role) is not str or type(content) is not str:
                return None
            thinking_json = None
            if row["thinking_blocks_json"] is not None:
                if role != "assistant":
                    return None
                thinking_json = validated_thinking_blocks_json(
                    row["thinking_blocks_json"]
                )
                if thinking_json != row["thinking_blocks_json"]:
                    return None
            if row["assistant_generation_state"] is not None and role != "assistant":
                return None
            private_json = row["provider_continuation_json"]
            has_active_continuation = False
            if private_json is not None:
                if role != "assistant" or type(private_json) is not str:
                    return None
                checkpoint = parse_provider_continuation_json(private_json)
                private_json = dump_provider_continuation_json(checkpoint)
                if private_json != row["provider_continuation_json"]:
                    return None
                has_active_continuation = checkpoint.state == "active"
            row_state = normalize_assistant_generation_state(
                role=role,
                raw_state=row["assistant_generation_state"],
                has_valid_active_continuation=has_active_continuation,
            )
            intent_state = normalize_assistant_generation_state(
                role=role,
                raw_state=intent_payload["assistant_generation_state"],
                has_valid_active_continuation=has_active_continuation,
            )
            if (
                intent_payload["assistant_generation_state"] is not None
                and role != "assistant"
            ):
                return None
            if (
                row_state is not None
                and row_state.value == "continuation_active"
                and not has_active_continuation
            ) or (
                intent_state is not None
                and intent_state.value == "continuation_active"
                and not has_active_continuation
            ):
                return None
            expected_intent = {
                "id": row["id"],
                "conversation_id": row["conversation_id"],
                "parent_message_id": row["parent_message_id"],
                "sender": row["sender"],
                "content": row["content"],
                "image_mime_type": row["image_mime_type"],
                "provider_continuation_json": row["provider_continuation_json"],
                "thinking_blocks_json": thinking_json,
                "assistant_generation_state": row_state.value
                if row_state is not None
                else None,
                "timestamp": intent_value(row["timestamp"]),
                "ranking": row["ranking"],
                "last_modified": intent_value(row["last_modified"]),
                "deleted": row["deleted"],
                "client_id": row["client_id"],
                "version": row["version"],
            }
            intent_payload["assistant_generation_state"] = (
                intent_state.value if intent_state is not None else None
            )
            if intent_payload != expected_intent:
                return None

            envelope_payload = {
                "assistant_generation_state": row_state.value
                if row_state is not None
                else None,
                "content": content,
                "role": role,
            }
            if private_json is not None:
                envelope_payload["provider_continuation_json"] = private_json
            if thinking_json is not None:
                envelope_payload["thinking_blocks_json"] = thinking_json
            if canonical_payload_hash(envelope_payload) != payload_hash:
                return None
            return ChatSyncIntentRecord(
                conversation_id=row["conversation_id"],
                message_id=row["id"],
                role=role,
                content=content,
                parent_message_id=row["parent_message_id"],
                provider_continuation_json=private_json,
                thinking_blocks_json=thinking_json,
                assistant_generation_state=row_state.value
                if row_state is not None
                else None,
                message_version=message_version,
                payload_hash=payload_hash,
                base_payload_hash=base_payload_hash,
            )
        except (
            ContinuationValidationError,
            InputError,
            ValueError,
            json.JSONDecodeError,
            sqlite3.Error,
        ):
            return None

    @staticmethod
    def _previous_committed_chat_payload_hash(
        conn: sqlite3.Connection,
        *,
        message_id: str,
        conversation_id: str,
        role: str,
        provider_continuation_json: str | None,
        message_version: int,
    ) -> str | None:
        """Return the immediate prior committed whole-record hash, if provable."""
        if message_version <= 1:
            return None
        rows = conn.execute(
            """
            SELECT operation, payload
              FROM sync_log
             WHERE entity = 'messages'
               AND entity_id = ?
               AND version = ?
             ORDER BY change_id
            """,
            (message_id, message_version - 1),
        ).fetchall()
        if len(rows) != 1:
            return None
        operation = rows[0]["operation"]
        raw_payload = json.loads(rows[0]["payload"])
        if operation == "delete":
            delete_payload = _normalize_legacy_chat_delete_intent_payload(raw_payload)
            if (
                delete_payload is None
                or delete_payload.get("id") != message_id
                or delete_payload.get("version") != message_version - 1
                or delete_payload.get("deleted") != 1
            ):
                return None
            has_active_continuation = False
            if provider_continuation_json is not None:
                if role != "assistant" or type(provider_continuation_json) is not str:
                    return None
                checkpoint = parse_provider_continuation_json(
                    provider_continuation_json
                )
                if (
                    dump_provider_continuation_json(checkpoint)
                    != provider_continuation_json
                ):
                    return None
                has_active_continuation = checkpoint.state == "active"
            raw_state = delete_payload["assistant_generation_state"]
            if raw_state is not None and role != "assistant":
                return None
            delete_state = normalize_assistant_generation_state(
                role=role,
                raw_state=raw_state,
                has_valid_active_continuation=has_active_continuation,
            )
            if (
                delete_state is not None
                and delete_state.value == "continuation_active"
                and not has_active_continuation
            ):
                return None
            return canonical_payload_hash({"deleted": True})
        if operation not in {"create", "update"}:
            return None
        payload = _normalize_legacy_chat_sync_intent_payload(raw_payload)
        if payload is None:
            return None
        if (
            payload.get("id") != message_id
            or payload.get("conversation_id") != conversation_id
            or payload.get("version") != message_version - 1
            or payload.get("deleted") != 0
            or type(payload.get("content")) is not str
        ):
            return None
        private_json = payload["provider_continuation_json"]
        has_active_continuation = False
        if private_json is not None:
            if role != "assistant" or type(private_json) is not str:
                return None
            checkpoint = parse_provider_continuation_json(private_json)
            private_json = dump_provider_continuation_json(checkpoint)
            if private_json != payload.get("provider_continuation_json"):
                return None
            has_active_continuation = checkpoint.state == "active"
        if payload["assistant_generation_state"] is not None and role != "assistant":
            return None
        state = normalize_assistant_generation_state(
            role=role,
            raw_state=payload["assistant_generation_state"],
            has_valid_active_continuation=has_active_continuation,
        )
        if (
            state is not None
            and state.value == "continuation_active"
            and not has_active_continuation
        ):
            return None
        base_payload = {
            "assistant_generation_state": state.value if state is not None else None,
            "content": payload["content"],
            "role": role,
        }
        if private_json is not None:
            base_payload["provider_continuation_json"] = private_json
        thinking_json = payload["thinking_blocks_json"]
        if thinking_json is not None:
            if role != "assistant":
                return None
            thinking_json = validated_thinking_blocks_json(thinking_json)
            if thinking_json != payload["thinking_blocks_json"]:
                return None
            base_payload["thinking_blocks_json"] = thinking_json
        return canonical_payload_hash(base_payload)

    def read_committed_chat_delete_intent(
        self,
        *,
        message_id: str,
        message_version: int,
        payload_hash: str,
    ) -> ChatSyncDeleteIntentRecord | None:
        """Return one exact committed message tombstone intent.

        Args:
            message_id: Exact tombstoned message ID.
            message_version: Exact committed tombstone version.
            payload_hash: Expected canonical delete payload hash.

        Returns:
            The validated committed delete source, or ``None`` when its
            proof is absent, ambiguous, invalid, or uncommitted.
        """
        if (
            type(message_id) is not str
            or not message_id
            or type(message_version) is not int
            or message_version < 1
            or type(payload_hash) is not str
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", payload_hash)
        ):
            return None
        conn = self.db.get_connection()
        if conn.in_transaction:
            return None
        try:
            query = """
                SELECT m.id, m.conversation_id, m.deleted, m.version,
                       m.last_modified, m.client_id, m.role, m.content,
                       m.provider_continuation_json,
                       m.assistant_generation_state,
                       intent.operation, intent.payload
                  FROM messages AS m
                  JOIN sync_log AS intent
                    ON intent.entity = 'messages'
                   AND intent.entity_id = m.id
                   AND intent.version = m.version
                 WHERE m.id = ? AND m.version = ?
                 ORDER BY intent.change_id
                """
            with self.db.transaction() as conn:
                rows = conn.execute(
                    query,
                    (message_id, message_version),
                ).fetchall()
                if len(rows) != 1:
                    return None
                row = rows[0]
            if not row["deleted"] or row["operation"] != "delete":
                return None
            intent_payload = _normalize_legacy_chat_delete_intent_payload(
                json.loads(row["payload"])
            )
            if (
                intent_payload is None
                or canonical_payload_hash({"deleted": True}) != payload_hash
            ):
                return None
            base_payload_hash = intent_payload.pop("base_payload_hash")
            legacy_marker = intent_payload.pop("legacy_pre_v50_base_reconstruction")
            if base_payload_hash is None and not legacy_marker:
                return None
            role = row["role"]
            content = row["content"]
            if type(role) is not str or type(content) is not str:
                return None
            private_json = row["provider_continuation_json"]
            has_active_continuation = False
            if private_json is not None:
                if role != "assistant" or type(private_json) is not str:
                    return None
                checkpoint = parse_provider_continuation_json(private_json)
                private_json = dump_provider_continuation_json(checkpoint)
                if private_json != row["provider_continuation_json"]:
                    return None
                has_active_continuation = checkpoint.state == "active"
            if row["assistant_generation_state"] is not None and role != "assistant":
                return None
            state = normalize_assistant_generation_state(
                role=role,
                raw_state=row["assistant_generation_state"],
                has_valid_active_continuation=has_active_continuation,
            )
            intent_state = normalize_assistant_generation_state(
                role=role,
                raw_state=intent_payload["assistant_generation_state"],
                has_valid_active_continuation=has_active_continuation,
            )
            if (
                intent_payload["assistant_generation_state"] is not None
                and role != "assistant"
            ):
                return None
            # A current-schema tombstone clears its private continuation and
            # thinking owners before commit. Its trigger-authored intent can
            # therefore retain the last generation-state label without the
            # sidecar that made that state live. The captured base hash is the
            # authoritative pre-delete proof; only legacy tombstones need the
            # surviving row fields to reconstruct it.
            if base_payload_hash is None and (
                (
                    state is not None
                    and state.value == "continuation_active"
                    and not has_active_continuation
                )
                or (
                    intent_state is not None
                    and intent_state.value == "continuation_active"
                    and not has_active_continuation
                )
            ):
                return None
            expected_intent = {
                "id": row["id"],
                "deleted": 1,
                "last_modified": row["last_modified"],
                "assistant_generation_state": state.value
                if state is not None
                else None,
                "version": row["version"],
                "client_id": row["client_id"],
            }
            if isinstance(expected_intent["last_modified"], datetime):
                expected_intent["last_modified"] = (
                    expected_intent["last_modified"]
                    .astimezone(timezone.utc)
                    .isoformat(timespec="milliseconds")
                    .replace("+00:00", "Z")
                )
            intent_payload["assistant_generation_state"] = (
                intent_state.value if intent_state is not None else None
            )
            if intent_payload != expected_intent:
                return None
            base_payload = {
                "assistant_generation_state": state.value
                if state is not None
                else None,
                "content": content,
                "role": role,
            }
            if private_json is not None:
                base_payload["provider_continuation_json"] = private_json
            return ChatSyncDeleteIntentRecord(
                conversation_id=row["conversation_id"],
                message_id=row["id"],
                message_version=message_version,
                payload_hash=payload_hash,
                base_payload_hash=(
                    base_payload_hash
                    if base_payload_hash is not None
                    else canonical_payload_hash(base_payload)
                ),
            )
        except (
            ContinuationValidationError,
            InputError,
            ValueError,
            json.JSONDecodeError,
            sqlite3.Error,
        ):
            return None

    @staticmethod
    def _split_chat_sync_stable_key(stable_key: str) -> tuple[str, str] | None:
        """Split the canonical ``conversation_id:message_id`` Sync key."""
        if type(stable_key) is not str:
            return None
        conversation_id, separator, message_id = stable_key.partition(":")
        if not separator or not conversation_id or not message_id:
            return None
        return conversation_id, message_id

    def get_chat_message_hash(self, stable_key: str) -> str | None:
        """Return the canonical whole-record Sync hash for one local message."""
        owner = self._split_chat_sync_stable_key(stable_key)
        if owner is None:
            return None
        conversation_id, message_id = owner
        row = (
            self.db.get_connection()
            .execute(
                "SELECT conversation_id, role, content, deleted, "
                "provider_continuation_json, thinking_blocks_json, "
                "assistant_generation_state FROM messages WHERE id = ?",
                (message_id,),
            )
            .fetchone()
        )
        if row is None or row["conversation_id"] != conversation_id:
            return None
        if row["deleted"]:
            return canonical_payload_hash({"deleted": True})
        try:
            role = row["role"]
            content = row["content"]
            if type(role) is not str or type(content) is not str:
                raise ValueError
            private_json = row["provider_continuation_json"]
            active_continuation = False
            if private_json is not None:
                if role != "assistant":
                    raise ValueError
                checkpoint, private_json = validated_provider_continuation(private_json)
                active_continuation = checkpoint.state == "active"
            state = normalize_assistant_generation_state(
                role=role,
                raw_state=row["assistant_generation_state"],
                has_valid_active_continuation=active_continuation,
            )
            payload = {
                "assistant_generation_state": state.value
                if state is not None
                else None,
                "content": content,
                "role": role,
            }
            if private_json is not None:
                payload["provider_continuation_json"] = private_json
            if row["thinking_blocks_json"] is not None:
                if role != "assistant":
                    raise ValueError
                payload["thinking_blocks_json"] = validated_thinking_blocks_json(
                    row["thinking_blocks_json"]
                )
            return canonical_payload_hash(payload)
        except (InputError, ValueError):
            # A present but unreadable owner must look divergent, not absent: this
            # keeps incoming Sync from replacing opaque local state by accident.
            return "invalid-local-chat-message"

    def append_chat_message(
        self, stable_key: str, payload: dict[str, Any], payload_hash: str
    ) -> None:
        """Apply one already-validated whole Chat Sync record atomically."""
        owner = self._split_chat_sync_stable_key(stable_key)
        if owner is None or canonical_payload_hash(payload) != payload_hash:
            raise InputError("Invalid chat sync record.")
        conversation_id, message_id = owner
        role = payload.get("role")
        content = payload.get("content")
        if type(role) is not str or type(content) is not str:
            raise InputError("Invalid chat sync record.")
        private_json = payload.get("provider_continuation_json")
        checkpoint = None
        if private_json is not None:
            if role != "assistant":
                raise InputError("Invalid chat sync record.")
            checkpoint, private_json = validated_provider_continuation(private_json)
        thinking_json = payload.get("thinking_blocks_json")
        if thinking_json is not None:
            if role != "assistant":
                raise InputError("Invalid chat sync record.")
            thinking_json = validated_thinking_blocks_json(thinking_json)
        try:
            state = normalize_assistant_generation_state(
                role=role,
                raw_state=payload.get("assistant_generation_state"),
                has_valid_active_continuation=(
                    checkpoint is not None and checkpoint.state == "active"
                ),
            )
        except ValueError:
            raise InputError("Invalid chat sync record.") from None
        existing = (
            self.db.get_connection()
            .execute(
                "SELECT conversation_id, role, version, deleted FROM messages WHERE id = ?",
                (message_id,),
            )
            .fetchone()
        )
        if existing is None:
            self.db.add_message(
                {
                    "id": message_id,
                    "conversation_id": conversation_id,
                    "sender": role,
                    "role": role,
                    "content": content,
                    "provider_continuation_json": private_json,
                    "thinking_blocks_json": thinking_json,
                    "assistant_generation_state": state.value
                    if state is not None
                    else None,
                }
            )
            return
        if (
            existing["conversation_id"] != conversation_id
            or existing["role"] != role
            or existing["deleted"]
        ):
            raise ConflictError(
                "Chat sync owner is incompatible.",
                entity="messages",
                entity_id=message_id,
            )
        now = self.db._get_current_utc_timestamp_iso()
        with self.db.transaction(immediate=True) as conn:

            def apply_sync_message(mutation_cursor: sqlite3.Cursor) -> None:
                updated = mutation_cursor.execute(
                    """
                    UPDATE messages
                       SET content = ?, provider_continuation_json = ?,
                           thinking_blocks_json = ?, assistant_generation_state = ?,
                           last_modified = ?, version = version + 1, client_id = ?
                     WHERE id = ? AND version = ? AND deleted = 0
                    """,
                    (
                        content,
                        private_json,
                        thinking_json,
                        state.value if state is not None else None,
                        now,
                        self.db.client_id,
                        message_id,
                        existing["version"],
                    ),
                )
                if updated.rowcount != 1:
                    raise ConflictError(
                        "Chat sync owner changed concurrently.",
                        entity="messages",
                        entity_id=message_id,
                    )

            self.db.coordinate_semantic_mutation(
                conn,
                message_id=message_id,
                creation_reason="sync_update",
                mutate=apply_sync_message,
            )

    def delete_chat_message(self, stable_key: str, payload_hash: str) -> None:
        """Apply one Chat Sync tombstone while retaining its semantic envelope."""
        owner = self._split_chat_sync_stable_key(stable_key)
        if owner is None or payload_hash != canonical_payload_hash({"deleted": True}):
            raise InputError("Invalid chat sync tombstone.")
        conversation_id, message_id = owner
        row = (
            self.db.get_connection()
            .execute(
                "SELECT conversation_id, version, deleted FROM messages WHERE id = ?",
                (message_id,),
            )
            .fetchone()
        )
        if row is None or row["conversation_id"] != conversation_id:
            return
        if not row["deleted"]:
            self.db.soft_delete_message(message_id, expected_version=row["version"])

    def list_current_committed_chat_sync_intents(
        self, conversation_id: str
    ) -> List[Dict[str, Any]]:
        """List exact current Chat intents for one restored conversation.

        Args:
            conversation_id: Restored conversation whose current message
                intents should be reconciled.

        Returns:
            Validated current intent descriptors safe to pass to the
            idempotent Chat Sync-v2 reconcilers.
        """
        if type(conversation_id) is not str or not conversation_id:
            return []
        conn = self.db.get_connection()
        if conn.in_transaction:
            return []
        try:
            query = """
                SELECT m.id, m.conversation_id, m.role, m.content,
                       m.provider_continuation_json,
                       m.thinking_blocks_json,
                       m.assistant_generation_state, m.deleted, m.version,
                       intent.operation
                  FROM messages AS m
                  JOIN conversations AS c
                    ON c.id = m.conversation_id AND c.deleted = 0
                  JOIN sync_log AS intent
                    ON intent.entity = 'messages'
                   AND intent.entity_id = m.id
                   AND intent.version = m.version
                 WHERE m.conversation_id = ?
                   AND (
                       (m.deleted = 1 AND intent.operation = 'delete') OR
                       (m.deleted = 0 AND intent.operation IN ('create', 'update'))
                   )
                   AND 1 = (
                       SELECT COUNT(*)
                         FROM sync_log AS duplicate
                        WHERE duplicate.entity = 'messages'
                          AND duplicate.entity_id = m.id
                          AND duplicate.version = m.version
                   )
                 ORDER BY m.timestamp, m.id
                """
            with self.db.transaction() as conn:
                rows = conn.execute(
                    query,
                    (conversation_id,),
                ).fetchall()
            intents: List[Dict[str, Any]] = []
            for row in rows:
                message_id = row["id"]
                message_version = row["version"]
                if row["deleted"]:
                    payload_hash = canonical_payload_hash({"deleted": True})
                    source = self.read_committed_chat_delete_intent(
                        message_id=message_id,
                        message_version=message_version,
                        payload_hash=payload_hash,
                    )
                    operation = "delete"
                else:
                    role = row["role"]
                    content = row["content"]
                    if type(role) is not str or type(content) is not str:
                        continue
                    private_json = row["provider_continuation_json"]
                    has_active_continuation = False
                    if private_json is not None:
                        if role != "assistant" or type(private_json) is not str:
                            continue
                        checkpoint = parse_provider_continuation_json(private_json)
                        private_json = dump_provider_continuation_json(checkpoint)
                        if private_json != row["provider_continuation_json"]:
                            continue
                        has_active_continuation = checkpoint.state == "active"
                    if (
                        row["assistant_generation_state"] is not None
                        and role != "assistant"
                    ):
                        continue
                    state = normalize_assistant_generation_state(
                        role=role,
                        raw_state=row["assistant_generation_state"],
                        has_valid_active_continuation=has_active_continuation,
                    )
                    if (
                        state is not None
                        and state.value == "continuation_active"
                        and not has_active_continuation
                    ):
                        continue
                    payload = {
                        "assistant_generation_state": state.value
                        if state is not None
                        else None,
                        "content": content,
                        "role": role,
                    }
                    if private_json is not None:
                        payload["provider_continuation_json"] = private_json
                    thinking_json = row["thinking_blocks_json"]
                    if thinking_json is not None:
                        if role != "assistant":
                            continue
                        try:
                            canonical_thinking = validated_thinking_blocks_json(
                                thinking_json
                            )
                        except InputError:
                            continue
                        if canonical_thinking != thinking_json:
                            continue
                        payload["thinking_blocks_json"] = canonical_thinking
                    payload_hash = canonical_payload_hash(payload)
                    source = self.read_committed_chat_sync_intent(
                        message_id=message_id,
                        message_version=message_version,
                        payload_hash=payload_hash,
                    )
                    operation = "upsert"
                if source is None or source.conversation_id != conversation_id:
                    continue
                intents.append(
                    {
                        "message_id": message_id,
                        "message_version": message_version,
                        "operation": operation,
                        "payload_hash": payload_hash,
                    }
                )
            return intents
        except (
            ContinuationValidationError,
            ValueError,
            json.JSONDecodeError,
            sqlite3.Error,
        ):
            return []


def chat_sync_source_for(database: object) -> CharactersRAGDBChatSyncSource | None:
    """Wrap a database in the chat-sync adapter.

    Args:
        database: Any object, typically a persistence service's ``db``.

    Returns:
        A new adapter over ``database`` if it is a ``CharactersRAGDB``, else
        ``None``.
    """
    if isinstance(database, CharactersRAGDB):
        return CharactersRAGDBChatSyncSource(database)
    return None


def is_chat_sync_source_over(source: object, database: object) -> bool:
    """Whether ``source`` is the chat-sync adapter over exactly ``database``.

    Args:
        source: A producer's ``source``, of any type.
        database: The database the caller persists to.

    Returns:
        ``True`` only if ``source`` is a :class:`CharactersRAGDBChatSyncSource`
        and its ``db`` is the same ``CharactersRAGDB`` object as ``database``.
    """
    return (
        isinstance(source, CharactersRAGDBChatSyncSource)
        and isinstance(database, CharactersRAGDB)
        and source.db is database
    )


def install_chat_sync_hashing() -> None:
    """Install :func:`chat_record_payload_hash` into the DB layer's port.

    Composition roots call this at startup; importing this module also calls
    it. Calling it again is harmless.
    """
    install_chat_record_hasher(chat_record_payload_hash)


install_chat_sync_hashing()
