"""Transactional persistence for Console dispatch recovery ownership."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping

from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleAssistantSettlement,
    ConsoleContinuationHandoff,
    ConsoleDispatchCheckpoint,
    ConsoleDispatchCheckpointState,
    ConsoleDispatchCheckpointValidationError,
    ConsoleDispatchReadResult,
    ConsoleDispatchResultStatus,
    ConsoleDispatchTransition,
    ConsoleDispatchWriteResult,
    ConsoleDurableTurnAcceptance,
    dump_console_dispatch_reconstructability_json,
    dump_console_resolved_destination_json,
    dump_console_turn_library_authority_json,
    parse_console_dispatch_reconstructability_json,
    parse_console_resolved_destination_json,
    parse_console_turn_library_authority_json,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleDispatchRecoveryKind,
    ConsoleDispatchRecoveryState,
    console_dispatch_recovery_from_checkpoint,
)
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationValidationError,
    dump_provider_continuation_json,
    parse_provider_continuation_json,
    read_provider_continuation_json,
)
from tldw_chatbook.Chat.thinking_blocks import (
    ThinkingEnvelopeValidationError,
    dump_thinking_blocks_json,
    parse_thinking_blocks_json,
    read_thinking_blocks_json,
)
from tldw_chatbook.Chat.console_transaction_contribution import (
    _scoped_console_transaction_writer,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash


_OWNER_SELECT = """
    SELECT checkpoint.assistant_message_id, checkpoint.user_message_id,
           checkpoint.conversation_id, checkpoint.schema_version,
           checkpoint.preparation_id, checkpoint.attempt_id, checkpoint.state,
           checkpoint.checkpoint_revision, checkpoint.user_message_version,
           checkpoint.assistant_message_version, checkpoint.origin,
           checkpoint.queue_entry_id, checkpoint.frozen_authority_json,
           checkpoint.resolved_destination_json,
           checkpoint.reconstructability_json,
           conversation.deleted AS conversation_deleted,
           user_message.conversation_id AS user_conversation_id,
           user_message.role AS user_role,
           user_message.version AS current_user_version,
           user_message.deleted AS user_deleted,
           assistant_message.conversation_id AS assistant_conversation_id,
           assistant_message.role AS assistant_role,
           assistant_message.content AS assistant_content,
           assistant_message.version AS current_assistant_version,
           assistant_message.deleted AS assistant_deleted,
           assistant_message.assistant_generation_state AS assistant_state,
           assistant_message.provider_continuation_json AS provider_continuation_json,
           assistant_message.thinking_blocks_json AS thinking_blocks_json
      FROM console_dispatch_checkpoints AS checkpoint
      JOIN conversations AS conversation
        ON conversation.id = checkpoint.conversation_id
      LEFT JOIN messages AS user_message
        ON user_message.id = checkpoint.user_message_id
      LEFT JOIN messages AS assistant_message
        ON assistant_message.id = checkpoint.assistant_message_id
"""

_ACTIVE_OWNER_SELECT = """
    WITH RECURSIVE active_path(message_id, parent_message_id) AS (
        SELECT active_message.id, active_message.parent_message_id
          FROM conversations AS active_conversation
          JOIN messages AS active_message
            ON active_message.id = active_conversation.active_leaf_message_id
           AND active_message.conversation_id = active_conversation.id
         WHERE active_conversation.id = ?
           AND active_conversation.deleted = 0
        UNION
        SELECT parent.id, parent.parent_message_id
          FROM messages AS parent
          JOIN active_path AS child
            ON child.parent_message_id = parent.id
         WHERE parent.conversation_id = ?
    )
"""

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")


class _ReconcileWriteNeeded:
    """Sentinel: the read-only reconcile pass found write-requiring work."""


_RECONCILE_WRITE_NEEDED = _ReconcileWriteNeeded()


class ConsoleDispatchRepository:
    """Own accepted insert, recovery validation, CAS, settlement, and handoff."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db

    def insert_with_messages(
        self,
        cursor: sqlite3.Cursor,
        acceptance: ConsoleDurableTurnAcceptance,
    ) -> ConsoleDispatchCheckpoint:
        """Insert one USER, assistant owner, and accepted checkpoint."""
        self._validate_acceptance(acceptance)
        conversation = cursor.execute(
            "SELECT deleted FROM conversations WHERE id = ?",
            (acceptance.conversation_id,),
        ).fetchone()
        if conversation is None or conversation["deleted"]:
            raise ConsoleDispatchCheckpointValidationError(
                "Conversation is missing or deleted."
            )
        existing_rows = cursor.execute(
            _OWNER_SELECT + " WHERE checkpoint.conversation_id = ?",
            (acceptance.conversation_id,),
        ).fetchall()
        if existing_rows:
            raise RuntimeError(
                "Conversation already has an active dispatch checkpoint."
            )
        if acceptance.parent_message_id is not None:
            parent = cursor.execute(
                "SELECT conversation_id, deleted FROM messages WHERE id = ?",
                (acceptance.parent_message_id,),
            ).fetchone()
            if (
                parent is None
                or parent["deleted"]
                or parent["conversation_id"] != acceptance.conversation_id
            ):
                raise ConsoleDispatchCheckpointValidationError(
                    "Parent message is unavailable."
                )
        attachments = self._validated_attachments(acceptance)
        first_attachment = next(
            (row for row in attachments if row[0] == 0),
            None,
        )
        now = self.db._get_current_utc_timestamp_iso()

        cursor.execute(
            """
            INSERT INTO messages (
                id, conversation_id, parent_message_id, sender, content,
                image_data, image_mime_type, timestamp, ranking,
                last_modified, client_id, version, deleted, role,
                usage_json, metadata_json, provider_continuation_json,
                assistant_generation_state
            ) VALUES (?, ?, ?, 'user', ?, ?, ?, ?, NULL,
                      ?, ?, 1, 0, 'user', NULL, NULL, NULL, NULL)
            """,
            (
                acceptance.user_message_id,
                acceptance.conversation_id,
                acceptance.parent_message_id,
                acceptance.user_content,
                first_attachment[1] if first_attachment is not None else None,
                first_attachment[2] if first_attachment is not None else None,
                now,
                now,
                self.db.client_id,
            ),
        )
        cursor.executemany(
            """
            INSERT INTO message_attachments (
                message_id, position, data, mime_type, display_name
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [(acceptance.user_message_id, *row) for row in attachments if row[0] >= 1],
        )
        cursor.execute(
            """
            INSERT INTO messages (
                id, conversation_id, parent_message_id, sender, content,
                image_data, image_mime_type, timestamp, ranking,
                last_modified, client_id, version, deleted, role,
                usage_json, metadata_json, provider_continuation_json,
                assistant_generation_state
            ) VALUES (?, ?, ?, 'assistant', '', NULL, NULL, ?, NULL,
                      ?, ?, 1, 0, 'assistant', NULL, NULL, NULL,
                      'accepted')
            """,
            (
                acceptance.assistant_message_id,
                acceptance.conversation_id,
                acceptance.user_message_id,
                now,
                now,
                self.db.client_id,
            ),
        )
        updated_leaf = cursor.execute(
            """
            UPDATE conversations
               SET active_leaf_message_id = ?
             WHERE id = ? AND deleted = 0
            """,
            (acceptance.assistant_message_id, acceptance.conversation_id),
        )
        if updated_leaf.rowcount != 1:
            raise sqlite3.IntegrityError(
                "Conversation became unavailable during acceptance."
            )
        authority_json = dump_console_turn_library_authority_json(
            acceptance.frozen_authority
        )
        destination_json = dump_console_resolved_destination_json(
            acceptance.resolved_destination
        )
        reconstructability_json = dump_console_dispatch_reconstructability_json(
            acceptance.reconstructability
        )
        cursor.execute(
            """
            INSERT INTO console_dispatch_checkpoints (
                assistant_message_id, user_message_id, conversation_id,
                schema_version, preparation_id, attempt_id, state,
                checkpoint_revision, user_message_version,
                assistant_message_version, origin, queue_entry_id,
                frozen_authority_json, resolved_destination_json,
                reconstructability_json, created_at, updated_at
            ) VALUES (?, ?, ?, 1, ?, ?, 'accepted', 1, 1, 1, ?, ?, ?, ?, ?,
                      CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                acceptance.assistant_message_id,
                acceptance.user_message_id,
                acceptance.conversation_id,
                acceptance.preparation_id,
                acceptance.attempt_id,
                acceptance.origin,
                acceptance.queue_entry_id,
                authority_json,
                destination_json,
                reconstructability_json,
            ),
        )
        message_ids: Mapping[str, str] = {
            "user": acceptance.user_message_id,
            "assistant": acceptance.assistant_message_id,
        }
        with _scoped_console_transaction_writer(
            cursor,
            acceptance.conversation_id,
        ) as writer:
            for contribution in acceptance.contributions:
                contribution.write(
                    writer=writer,
                    conversation_id=acceptance.conversation_id,
                    message_ids=message_ids,
                )
        row = cursor.execute(
            _OWNER_SELECT + " WHERE checkpoint.assistant_message_id = ?",
            (acceptance.assistant_message_id,),
        ).fetchone()
        if row is None:
            raise sqlite3.DatabaseError("Accepted checkpoint was not readable.")
        checkpoint, error_code = self._checkpoint_from_row(row)
        if checkpoint is None:
            raise ConsoleDispatchCheckpointValidationError(
                error_code or "Accepted checkpoint is invalid."
            )
        return checkpoint

    def read_for_session(self, conversation_id: str) -> ConsoleDispatchReadResult:
        """Read and validate at most one active-path recovery owner."""
        if type(conversation_id) is not str or not conversation_id.strip():
            return ConsoleDispatchReadResult(
                ConsoleDispatchResultStatus.NOT_FOUND,
                None,
            )
        try:
            rows = (
                self.db.get_connection()
                .execute(
                    _ACTIVE_OWNER_SELECT
                    + _OWNER_SELECT
                    + " JOIN active_path ON active_path.message_id = "
                    "checkpoint.assistant_message_id"
                    " WHERE checkpoint.conversation_id = ?",
                    (conversation_id, conversation_id, conversation_id),
                )
                .fetchall()
            )
        except sqlite3.Error:
            return ConsoleDispatchReadResult(
                ConsoleDispatchResultStatus.QUARANTINED,
                None,
                "checkpoint_read_error",
            )
        if not rows:
            return ConsoleDispatchReadResult(
                ConsoleDispatchResultStatus.NOT_FOUND,
                None,
            )
        if len(rows) != 1:
            return ConsoleDispatchReadResult(
                ConsoleDispatchResultStatus.QUARANTINED,
                None,
                "duplicate_active_path_owner",
            )
        checkpoint, error_code = self._checkpoint_from_row(rows[0])
        if checkpoint is None:
            return ConsoleDispatchReadResult(
                ConsoleDispatchResultStatus.QUARANTINED,
                None,
                error_code or "invalid_checkpoint",
            )
        return ConsoleDispatchReadResult(
            ConsoleDispatchResultStatus.COMMITTED,
            checkpoint,
        )

    def reconcile_for_session(
        self, conversation_id: str
    ) -> ConsoleDispatchRecoveryState | None:
        """Reconcile one active dispatch/continuation owner before queue wake.

        The checkpoint table remains device-local.  With no local checkpoint,
        synchronized ``accepted``/``dispatch_started`` values are therefore
        inert source-device facts, never replay authority.
        """

        if type(conversation_id) is not str or not conversation_id.strip():
            return None
        try:
            # TASK-22205: most reconciles have nothing to write (no
            # checkpoint at all, or a still-valid owner), so the first pass
            # is a plain read transaction that never issues a write
            # statement — it must not queue on the 15 s write-lock busy
            # timeout just to read recovery state. Only when the read pass
            # finds write-requiring work does a SECOND, fresh
            # ``BEGIN IMMEDIATE`` transaction re-run the full logic from
            # scratch (re-reading inside the write lock). The read pass
            # never upgrades in place: a DEFERRED read-then-write hits
            # SQLite's non-retryable snapshot-upgrade deadlock under
            # concurrent writers (the recorded task-21100 wave-1 lesson).
            outcome = self._reconcile_pass(conversation_id, allow_writes=False)
            if not isinstance(outcome, _ReconcileWriteNeeded):
                return outcome
            outcome = self._reconcile_pass(conversation_id, allow_writes=True)
            if isinstance(outcome, _ReconcileWriteNeeded):  # pragma: no cover
                raise sqlite3.IntegrityError("Reconcile write pass refused to write.")
            return outcome
        except sqlite3.Error:
            return self._quarantined(
                conversation_id,
                "",
                "checkpoint_reconcile_error",
            )

    def _reconcile_pass(
        self,
        conversation_id: str,
        *,
        allow_writes: bool,
    ) -> "ConsoleDispatchRecoveryState | None | _ReconcileWriteNeeded":
        """Run one reconcile pass; read passes may report write-needed."""

        with self.db.transaction(immediate=allow_writes) as cursor:
            rows = cursor.execute(
                _OWNER_SELECT + " WHERE checkpoint.conversation_id = ?",
                (conversation_id,),
            ).fetchall()
            if len(rows) > 1:
                return self._quarantined(
                    conversation_id,
                    "",
                    "duplicate_active_path_owner",
                )
            if rows:
                return self._reconcile_checkpoint_row(
                    cursor,
                    conversation_id,
                    rows[0],
                    allow_writes=allow_writes,
                )
            return self._reconcile_checkpoint_free_owner(
                cursor,
                conversation_id,
            )

    def _reconcile_checkpoint_row(
        self,
        cursor: sqlite3.Cursor,
        conversation_id: str,
        row: sqlite3.Row,
        *,
        allow_writes: bool = True,
    ) -> "ConsoleDispatchRecoveryState | None | _ReconcileWriteNeeded":
        assistant_id = str(row["assistant_message_id"] or "")
        if not self._valid_reconcile_pair(row):
            return self._quarantined(
                conversation_id,
                assistant_id,
                "invalid_checkpoint_owner",
            )
        active_ids = self._active_path_ids(cursor, conversation_id)
        if assistant_id not in active_ids:
            return self._quarantined(
                conversation_id,
                assistant_id,
                "checkpoint_not_active_path",
            )

        continuation = read_provider_continuation_json(
            row["provider_continuation_json"]
        )
        if continuation.checkpoint is not None:
            if continuation.checkpoint.state != "active":
                return self._quarantined(
                    conversation_id,
                    assistant_id,
                    "invalid_continuation",
                )
            if not allow_writes:
                return _RECONCILE_WRITE_NEEDED
            next_version = int(row["current_assistant_version"]) + 1
            now = self.db._get_current_utc_timestamp_iso()
            updated = cursor.execute(
                """
                UPDATE messages
                   SET assistant_generation_state = 'continuation_active',
                       version = ?, last_modified = ?, client_id = ?
                 WHERE id = ? AND conversation_id = ? AND role = 'assistant'
                   AND assistant_generation_state IS ? AND version = ?
                   AND deleted = 0 AND provider_continuation_json = ?
                """,
                (
                    next_version,
                    now,
                    self.db.client_id,
                    assistant_id,
                    conversation_id,
                    row["assistant_state"],
                    row["current_assistant_version"],
                    row["provider_continuation_json"],
                ),
            )
            if updated.rowcount != 1:
                raise sqlite3.IntegrityError(
                    "Continuation precedence message CAS failed."
                )
            self._delete_exact_checkpoint(cursor, row)
            return ConsoleDispatchRecoveryState(
                kind=ConsoleDispatchRecoveryKind.CONTINUATION,
                assistant_message_id=assistant_id,
                conversation_id=conversation_id,
                visible_copy="Response continuation is pending.",
                actions=(),
            )

        assistant_state = row["assistant_state"]
        if assistant_state in {"complete", "stopped", "failed", "discarded"}:
            if not allow_writes:
                return _RECONCILE_WRITE_NEEDED
            self._delete_exact_checkpoint(cursor, row)
            return None
        if (
            row["current_assistant_version"] != row["assistant_message_version"]
            or assistant_state != row["state"]
        ):
            return self._quarantined(
                conversation_id,
                assistant_id,
                "invalid_checkpoint_owner",
            )
        checkpoint, error_code = self._checkpoint_from_row(row)
        if checkpoint is None:
            return self._quarantined(
                conversation_id,
                assistant_id,
                error_code or "invalid_checkpoint",
            )
        return console_dispatch_recovery_from_checkpoint(checkpoint)

    def _reconcile_checkpoint_free_owner(
        self,
        cursor: sqlite3.Cursor,
        conversation_id: str,
    ) -> ConsoleDispatchRecoveryState | None:
        rows = cursor.execute(
            _ACTIVE_OWNER_SELECT
            + """
            SELECT message.id, message.conversation_id, message.role,
                   message.deleted, message.assistant_generation_state,
                   message.provider_continuation_json,
                   conversation.active_leaf_message_id
              FROM active_path
              JOIN messages AS message ON message.id = active_path.message_id
              JOIN conversations AS conversation
                ON conversation.id = message.conversation_id
             WHERE conversation.id = ? AND conversation.deleted = 0
            """,
            (conversation_id, conversation_id, conversation_id),
        ).fetchall()
        if not rows:
            return None
        valid_owners: list[sqlite3.Row] = []
        invalid_owner: tuple[sqlite3.Row, str] | None = None
        for candidate in rows:
            if (
                candidate["conversation_id"] != conversation_id
                or candidate["role"] != "assistant"
                or candidate["deleted"] != 0
            ):
                continue
            private_json = candidate["provider_continuation_json"]
            continuation = read_provider_continuation_json(private_json)
            if continuation.checkpoint is not None:
                if continuation.checkpoint.state == "active":
                    valid_owners.append(candidate)
                elif (
                    continuation.checkpoint.state == "complete"
                    and candidate["assistant_generation_state"] == "complete"
                ):
                    continue
                elif invalid_owner is None:
                    invalid_owner = (candidate, "invalid_continuation")
                continue
            if candidate["assistant_generation_state"] == "continuation_active":
                if invalid_owner is None:
                    invalid_owner = (candidate, "orphan_continuation")
            elif private_json is not None and invalid_owner is None:
                invalid_owner = (candidate, "invalid_continuation")

        if len(valid_owners) > 1:
            return self._quarantined(
                conversation_id,
                "",
                "duplicate_active_path_owner",
            )
        if invalid_owner is not None:
            candidate, error_code = invalid_owner
            return self._quarantined(
                conversation_id,
                str(candidate["id"]),
                error_code,
            )
        if valid_owners:
            owner = valid_owners[0]
            return ConsoleDispatchRecoveryState(
                kind=ConsoleDispatchRecoveryKind.CONTINUATION,
                assistant_message_id=str(owner["id"]),
                conversation_id=conversation_id,
                visible_copy="Response continuation is pending.",
                actions=(),
            )

        row = next(
            (
                candidate
                for candidate in rows
                if candidate["id"] == candidate["active_leaf_message_id"]
            ),
            None,
        )
        if row is None or row["role"] != "assistant" or row["deleted"] != 0:
            return None
        assistant_id = str(row["id"])
        state = row["assistant_generation_state"]
        if state == "accepted":
            return ConsoleDispatchRecoveryState(
                kind=ConsoleDispatchRecoveryKind.REMOTE_ACCEPTED,
                assistant_message_id=assistant_id,
                conversation_id=conversation_id,
                visible_copy=(
                    "Response accepted on another device; waiting for dispatch."
                ),
                actions=(),
            )
        if state == "dispatch_started":
            return ConsoleDispatchRecoveryState(
                kind=ConsoleDispatchRecoveryKind.REMOTE_DISPATCH_STARTED,
                assistant_message_id=assistant_id,
                conversation_id=conversation_id,
                visible_copy=(
                    "Response delivery status is unknown on the source device."
                ),
                actions=(),
            )
        return None

    @staticmethod
    def _valid_reconcile_pair(row: sqlite3.Row) -> bool:
        return bool(
            row["conversation_deleted"] == 0
            and row["user_role"] == "user"
            and row["assistant_role"] == "assistant"
            and row["user_conversation_id"] == row["conversation_id"]
            and row["assistant_conversation_id"] == row["conversation_id"]
            and row["user_deleted"] == 0
            and row["assistant_deleted"] == 0
            and row["current_user_version"] == row["user_message_version"]
            and ConsoleDispatchRepository._positive_versions(
                row["checkpoint_revision"],
                row["user_message_version"],
                row["assistant_message_version"],
                row["current_assistant_version"],
            )
        )

    @staticmethod
    def _active_path_ids(
        cursor: sqlite3.Cursor, conversation_id: str
    ) -> frozenset[str]:
        rows = cursor.execute(
            _ACTIVE_OWNER_SELECT + " SELECT message_id FROM active_path",
            (conversation_id, conversation_id),
        ).fetchall()
        return frozenset(str(row[0]) for row in rows)

    @staticmethod
    def _delete_exact_checkpoint(cursor: sqlite3.Cursor, row: sqlite3.Row) -> None:
        deleted = cursor.execute(
            """
            DELETE FROM console_dispatch_checkpoints
             WHERE assistant_message_id = ? AND state = ?
               AND checkpoint_revision = ? AND user_message_version = ?
               AND assistant_message_version = ?
            """,
            (
                row["assistant_message_id"],
                row["state"],
                row["checkpoint_revision"],
                row["user_message_version"],
                row["assistant_message_version"],
            ),
        )
        if deleted.rowcount != 1:
            raise sqlite3.IntegrityError("Checkpoint reconciliation delete failed.")

    @staticmethod
    def _quarantined(
        conversation_id: str,
        assistant_message_id: str,
        error_code: str,
    ) -> ConsoleDispatchRecoveryState:
        bounded = (
            error_code
            if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", error_code or "")
            else "invalid_checkpoint"
        )
        return ConsoleDispatchRecoveryState(
            kind=ConsoleDispatchRecoveryKind.QUARANTINED,
            assistant_message_id=assistant_message_id,
            conversation_id=conversation_id,
            visible_copy=(
                "Dispatch recovery is unavailable because persisted ownership "
                "is invalid."
            ),
            actions=(),
            error_code=bounded,
        )

    def cas_state(
        self, transition: ConsoleDispatchTransition
    ) -> ConsoleDispatchWriteResult:
        """Apply an expected-revision accepted/dispatch-started transition."""
        if (
            not isinstance(transition.expected_state, ConsoleDispatchCheckpointState)
            or not isinstance(transition.new_state, ConsoleDispatchCheckpointState)
            or transition.new_state
            is not ConsoleDispatchCheckpointState.DISPATCH_STARTED
            or transition.expected_state
            not in {
                ConsoleDispatchCheckpointState.ACCEPTED,
                ConsoleDispatchCheckpointState.DISPATCH_STARTED,
            }
            or not self._positive_versions(
                transition.expected_checkpoint_revision,
                transition.expected_user_message_version,
                transition.expected_assistant_message_version,
            )
            or type(transition.new_attempt_id) is not str
            or not self._valid_identifier(transition.new_attempt_id)
        ):
            raise ConsoleDispatchCheckpointValidationError(
                "Invalid dispatch transition."
            )
        with self.db.transaction(immediate=True) as cursor:
            row = self._owner_by_assistant(cursor, transition.assistant_message_id)
            if row is None:
                return self._write_status(ConsoleDispatchResultStatus.NOT_FOUND)
            if not self._matches_transition(row, transition):
                return self._write_status(ConsoleDispatchResultStatus.CONFLICT)
            next_message_version = transition.expected_assistant_message_version + 1
            now = self.db._get_current_utc_timestamp_iso()
            updated_message = cursor.execute(
                """
                UPDATE messages
                   SET assistant_generation_state = ?, version = ?,
                       last_modified = ?, client_id = ?
                 WHERE id = ? AND conversation_id = ? AND role = 'assistant'
                   AND assistant_generation_state = ? AND version = ? AND deleted = 0
                """,
                (
                    transition.new_state.value,
                    next_message_version,
                    now,
                    self.db.client_id,
                    transition.assistant_message_id,
                    row["conversation_id"],
                    transition.expected_state.value,
                    transition.expected_assistant_message_version,
                ),
            )
            if updated_message.rowcount != 1:
                return self._write_status(ConsoleDispatchResultStatus.CONFLICT)
            updated_checkpoint = cursor.execute(
                """
                UPDATE console_dispatch_checkpoints
                   SET state = ?, attempt_id = ?,
                       checkpoint_revision = checkpoint_revision + 1,
                       assistant_message_version = ?,
                       updated_at = CURRENT_TIMESTAMP
                 WHERE assistant_message_id = ? AND state = ?
                   AND checkpoint_revision = ? AND user_message_version = ?
                   AND assistant_message_version = ?
                """,
                (
                    transition.new_state.value,
                    transition.new_attempt_id,
                    next_message_version,
                    transition.assistant_message_id,
                    transition.expected_state.value,
                    transition.expected_checkpoint_revision,
                    transition.expected_user_message_version,
                    transition.expected_assistant_message_version,
                ),
            )
            if updated_checkpoint.rowcount != 1:
                raise sqlite3.IntegrityError(
                    "Checkpoint state CAS lost after owner CAS."
                )
            updated_row = self._owner_by_assistant(
                cursor, transition.assistant_message_id
            )
            if updated_row is None:
                raise sqlite3.DatabaseError("Committed checkpoint is unavailable.")
            checkpoint, error_code = self._checkpoint_from_row(updated_row)
            if checkpoint is None:
                raise ConsoleDispatchCheckpointValidationError(
                    error_code or "Committed checkpoint is invalid."
                )
            payload_hash = self._message_payload_hash(
                content=updated_row["assistant_content"],
                state=transition.new_state.value,
                thinking_blocks_json=updated_row["thinking_blocks_json"],
            )
            return ConsoleDispatchWriteResult(
                ConsoleDispatchResultStatus.COMMITTED,
                checkpoint,
                next_message_version,
                payload_hash,
            )

    def settle_with_assistant(
        self, settlement: ConsoleAssistantSettlement
    ) -> ConsoleDispatchWriteResult:
        """Commit terminal assistant state and delete its checkpoint atomically."""
        if (
            settlement.terminal_state
            not in {"complete", "stopped", "failed", "discarded"}
            or type(settlement.content) is not str
            or not isinstance(
                settlement.expected_checkpoint_state,
                ConsoleDispatchCheckpointState,
            )
            or not self._positive_versions(
                settlement.expected_checkpoint_revision,
                settlement.expected_user_message_version,
                settlement.expected_assistant_message_version,
            )
        ):
            raise ConsoleDispatchCheckpointValidationError(
                "Invalid assistant settlement."
            )
        if settlement.metadata_json is not None:
            try:
                metadata = json.loads(settlement.metadata_json)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ConsoleDispatchCheckpointValidationError(
                    "Invalid assistant metadata."
                ) from exc
            if type(metadata) is not dict:
                raise ConsoleDispatchCheckpointValidationError(
                    "Invalid assistant metadata."
                )
        if settlement.usage_json is not None:
            try:
                usage = json.loads(settlement.usage_json)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ConsoleDispatchCheckpointValidationError(
                    "Invalid assistant usage."
                ) from exc
            if type(usage) is not dict:
                raise ConsoleDispatchCheckpointValidationError(
                    "Invalid assistant usage."
                )
        terminal_continuation = None
        canonical_continuation = None
        canonical_thinking = None
        if settlement.thinking_blocks_json is not None:
            try:
                canonical_thinking = dump_thinking_blocks_json(
                    parse_thinking_blocks_json(settlement.thinking_blocks_json)
                )
            except ThinkingEnvelopeValidationError as exc:
                raise ConsoleDispatchCheckpointValidationError(
                    "Invalid terminal thinking."
                ) from exc
        if settlement.provider_continuation_json is not None:
            try:
                terminal_continuation = parse_provider_continuation_json(
                    settlement.provider_continuation_json
                )
                canonical_continuation = dump_provider_continuation_json(
                    terminal_continuation
                )
            except ContinuationValidationError as exc:
                raise ConsoleDispatchCheckpointValidationError(
                    "Invalid terminal continuation."
                ) from exc
            if (
                settlement.terminal_state != "complete"
                or terminal_continuation.state != "complete"
                or terminal_continuation.rounds[-1].assistant_content
                != settlement.content
            ):
                raise ConsoleDispatchCheckpointValidationError(
                    "Invalid terminal continuation."
                )
        with self.db.transaction(immediate=True) as cursor:
            row = self._owner_by_assistant(cursor, settlement.assistant_message_id)
            if row is None:
                return self._write_status(ConsoleDispatchResultStatus.NOT_FOUND)
            if not self._matches_settlement(row, settlement):
                return self._write_status(ConsoleDispatchResultStatus.CONFLICT)
            if (
                terminal_continuation is not None
                and not self._continuation_matches_destination(
                    row, terminal_continuation
                )
            ):
                return self._write_status(ConsoleDispatchResultStatus.CONFLICT)
            next_message_version = settlement.expected_assistant_message_version + 1
            now = self.db._get_current_utc_timestamp_iso()
            updated = cursor.execute(
                """
                UPDATE messages
                   SET content = ?, metadata_json = ?, usage_json = ?,
                       provider_continuation_json = ?, thinking_blocks_json = ?,
                       assistant_generation_state = ?, version = ?,
                       last_modified = ?, client_id = ?
                 WHERE id = ? AND conversation_id = ? AND role = 'assistant'
                   AND assistant_generation_state = ? AND version = ? AND deleted = 0
                """,
                (
                    settlement.content,
                    settlement.metadata_json,
                    settlement.usage_json,
                    canonical_continuation,
                    canonical_thinking,
                    settlement.terminal_state,
                    next_message_version,
                    now,
                    self.db.client_id,
                    settlement.assistant_message_id,
                    row["conversation_id"],
                    settlement.expected_checkpoint_state.value,
                    settlement.expected_assistant_message_version,
                ),
            )
            if updated.rowcount != 1:
                return self._write_status(ConsoleDispatchResultStatus.CONFLICT)
            deleted = cursor.execute(
                """
                DELETE FROM console_dispatch_checkpoints
                 WHERE assistant_message_id = ? AND state = ?
                   AND checkpoint_revision = ? AND user_message_version = ?
                   AND assistant_message_version = ?
                """,
                (
                    settlement.assistant_message_id,
                    settlement.expected_checkpoint_state.value,
                    settlement.expected_checkpoint_revision,
                    settlement.expected_user_message_version,
                    settlement.expected_assistant_message_version,
                ),
            )
            if deleted.rowcount != 1:
                raise sqlite3.IntegrityError(
                    "Checkpoint settlement delete lost after owner CAS."
                )
            return ConsoleDispatchWriteResult(
                ConsoleDispatchResultStatus.COMMITTED,
                None,
                next_message_version,
                self._message_payload_hash(
                    content=settlement.content,
                    state=settlement.terminal_state,
                    provider_continuation_json=canonical_continuation,
                    thinking_blocks_json=canonical_thinking,
                ),
            )

    def handoff_to_provider_continuation(
        self, handoff: ConsoleContinuationHandoff
    ) -> ConsoleDispatchWriteResult:
        """Commit ADR-063 ownership and remove dispatch ownership atomically."""
        if not self._positive_versions(
            handoff.expected_checkpoint_revision,
            handoff.expected_user_message_version,
            handoff.expected_assistant_message_version,
        ):
            raise ConsoleDispatchCheckpointValidationError(
                "Invalid continuation handoff."
            )
        try:
            continuation = parse_provider_continuation_json(
                handoff.provider_continuation_json
            )
            canonical = dump_provider_continuation_json(continuation)
        except ContinuationValidationError as exc:
            raise ConsoleDispatchCheckpointValidationError(
                "Invalid continuation handoff."
            ) from exc
        if continuation.state != "active" or canonical is None:
            raise ConsoleDispatchCheckpointValidationError(
                "Continuation handoff must be active."
            )
        with self.db.transaction(immediate=True) as cursor:
            row = self._owner_by_assistant(cursor, handoff.assistant_message_id)
            if row is None:
                return self._write_status(ConsoleDispatchResultStatus.NOT_FOUND)
            if not self._matches_handoff(row, handoff):
                return self._write_status(ConsoleDispatchResultStatus.CONFLICT)
            if not self._continuation_matches_destination(row, continuation):
                return self._write_status(ConsoleDispatchResultStatus.CONFLICT)
            expected_state = row["state"]
            next_message_version = handoff.expected_assistant_message_version + 1
            now = self.db._get_current_utc_timestamp_iso()
            updated = cursor.execute(
                """
                UPDATE messages
                   SET provider_continuation_json = ?,
                       assistant_generation_state = 'continuation_active',
                       version = ?, last_modified = ?, client_id = ?
                 WHERE id = ? AND conversation_id = ? AND role = 'assistant'
                   AND assistant_generation_state = ? AND version = ? AND deleted = 0
                """,
                (
                    canonical,
                    next_message_version,
                    now,
                    self.db.client_id,
                    handoff.assistant_message_id,
                    row["conversation_id"],
                    expected_state,
                    handoff.expected_assistant_message_version,
                ),
            )
            if updated.rowcount != 1:
                return self._write_status(ConsoleDispatchResultStatus.CONFLICT)
            deleted = cursor.execute(
                """
                DELETE FROM console_dispatch_checkpoints
                 WHERE assistant_message_id = ? AND state = ?
                   AND checkpoint_revision = ? AND user_message_version = ?
                   AND assistant_message_version = ?
                """,
                (
                    handoff.assistant_message_id,
                    expected_state,
                    handoff.expected_checkpoint_revision,
                    handoff.expected_user_message_version,
                    handoff.expected_assistant_message_version,
                ),
            )
            if deleted.rowcount != 1:
                raise sqlite3.IntegrityError(
                    "Checkpoint handoff delete lost after owner CAS."
                )
            content = row["assistant_content"]
            return ConsoleDispatchWriteResult(
                ConsoleDispatchResultStatus.COMMITTED,
                None,
                next_message_version,
                self._message_payload_hash(
                    content=content,
                    state="continuation_active",
                    provider_continuation_json=canonical,
                    thinking_blocks_json=row["thinking_blocks_json"],
                ),
            )

    def normalize_provider_continuation_owner(
        self,
        *,
        conversation_id: str,
        assistant_message_id: str,
        expected_message_version: int,
        expected_state: str | None,
        provider_continuation_json: str,
    ) -> ConsoleDispatchWriteResult:
        """CAS one checkpoint-free legacy owner to continuation_active."""
        if (
            not self._valid_identifier(conversation_id)
            or not self._valid_identifier(assistant_message_id)
            or not self._positive_versions(expected_message_version)
        ):
            raise ConsoleDispatchCheckpointValidationError(
                "Invalid continuation normalization."
            )
        try:
            continuation = parse_provider_continuation_json(provider_continuation_json)
            canonical = dump_provider_continuation_json(continuation)
        except ContinuationValidationError as exc:
            raise ConsoleDispatchCheckpointValidationError(
                "Invalid continuation normalization."
            ) from exc
        if continuation.state != "active" or canonical is None:
            raise ConsoleDispatchCheckpointValidationError(
                "Invalid continuation normalization."
            )
        with self.db.transaction(immediate=True) as cursor:
            active_ids = self._active_path_ids(cursor, conversation_id)
            if assistant_message_id not in active_ids:
                return self._write_status(ConsoleDispatchResultStatus.CONFLICT)
            row = cursor.execute(
                """
                SELECT id, conversation_id, role, deleted, version,
                       assistant_generation_state, provider_continuation_json,
                       content
                  FROM messages
                 WHERE id = ?
                """,
                (assistant_message_id,),
            ).fetchone()
            if (
                row is None
                or row["conversation_id"] != conversation_id
                or row["role"] != "assistant"
                or row["deleted"] != 0
                or row["version"] != expected_message_version
                or row["assistant_generation_state"] != expected_state
                or row["provider_continuation_json"] != canonical
            ):
                return self._write_status(ConsoleDispatchResultStatus.CONFLICT)
            next_version = expected_message_version + 1
            now = self.db._get_current_utc_timestamp_iso()
            updated = cursor.execute(
                """
                UPDATE messages
                   SET assistant_generation_state = 'continuation_active',
                       version = ?, last_modified = ?, client_id = ?
                 WHERE id = ? AND conversation_id = ? AND role = 'assistant'
                   AND deleted = 0 AND version = ?
                   AND assistant_generation_state IS ?
                   AND provider_continuation_json = ?
                """,
                (
                    next_version,
                    now,
                    self.db.client_id,
                    assistant_message_id,
                    conversation_id,
                    expected_message_version,
                    expected_state,
                    canonical,
                ),
            )
            if updated.rowcount != 1:
                return self._write_status(ConsoleDispatchResultStatus.CONFLICT)
            return ConsoleDispatchWriteResult(
                ConsoleDispatchResultStatus.COMMITTED,
                None,
                next_version,
                self._message_payload_hash(
                    content=str(row["content"] or ""),
                    state="continuation_active",
                    provider_continuation_json=canonical,
                ),
            )

    def provider_continuation_owner_snapshot(
        self, *, conversation_id: str, assistant_message_id: str
    ) -> Mapping[str, object] | None:
        """Freshly read one valid active-path ADR-063 owner for CAS recovery."""
        try:
            with self.db.transaction() as cursor:
                if assistant_message_id not in self._active_path_ids(
                    cursor, conversation_id
                ):
                    return None
                row = cursor.execute(
                    """
                    SELECT id, conversation_id, role, deleted, version,
                           assistant_generation_state, provider_continuation_json,
                           content
                      FROM messages
                     WHERE id = ?
                    """,
                    (assistant_message_id,),
                ).fetchone()
                if (
                    row is None
                    or row["conversation_id"] != conversation_id
                    or row["role"] != "assistant"
                    or row["deleted"] != 0
                    or type(row["version"]) is not int
                ):
                    return None
                safe = read_provider_continuation_json(
                    row["provider_continuation_json"]
                )
                if safe.checkpoint is None or safe.checkpoint.state != "active":
                    return None
                canonical = dump_provider_continuation_json(safe.checkpoint)
                if canonical is None:
                    return None
                return {
                    "checkpoint": safe.checkpoint,
                    "canonical": canonical,
                    "state": row["assistant_generation_state"],
                    "version": int(row["version"]),
                    "content": str(row["content"] or ""),
                }
        except sqlite3.Error:
            return None

    @staticmethod
    def _continuation_matches_destination(
        row: sqlite3.Row, continuation: object
    ) -> bool:
        try:
            destination = parse_console_resolved_destination_json(
                row["resolved_destination_json"]
            )
            return bool(
                continuation.provider == destination.provider
                and continuation.model == destination.model
                and continuation.api_base_url == destination.endpoint_identity
            )
        except (AttributeError, ConsoleDispatchCheckpointValidationError):
            return False

    @staticmethod
    def _validate_acceptance(acceptance: ConsoleDurableTurnAcceptance) -> None:
        if (
            not isinstance(acceptance, ConsoleDurableTurnAcceptance)
            or acceptance.origin not in {"manual", "queued"}
            or (acceptance.origin == "manual" and acceptance.queue_entry_id is not None)
            or (acceptance.origin == "queued" and not acceptance.queue_entry_id)
            or any(
                not ConsoleDispatchRepository._valid_identifier(value)
                for value in (
                    acceptance.conversation_id,
                    acceptance.user_message_id,
                    acceptance.assistant_message_id,
                    acceptance.preparation_id,
                    acceptance.attempt_id,
                )
            )
            or (
                acceptance.parent_message_id is not None
                and not ConsoleDispatchRepository._valid_identifier(
                    acceptance.parent_message_id
                )
            )
            or (
                acceptance.queue_entry_id is not None
                and not ConsoleDispatchRepository._valid_identifier(
                    acceptance.queue_entry_id
                )
            )
            or acceptance.user_message_id == acceptance.assistant_message_id
            or acceptance.attempt_id != acceptance.frozen_authority.attempt_id
            or type(acceptance.user_content) is not str
            or type(acceptance.attachments) is not tuple
            or type(acceptance.contributions) is not tuple
        ):
            raise ConsoleDispatchCheckpointValidationError(
                "Invalid durable turn acceptance."
            )

    @staticmethod
    def _validated_attachments(
        acceptance: ConsoleDurableTurnAcceptance,
    ) -> tuple[tuple[int, bytes, str, str], ...]:
        if (
            acceptance.attachments
            and not acceptance.reconstructability.attachments_reconstructable
        ):
            raise ConsoleDispatchCheckpointValidationError(
                "Accepted attachments must be reconstructable."
            )
        normalized: list[tuple[int, bytes, str, str]] = []
        positions: set[int] = set()
        for attachment in acceptance.attachments:
            if not isinstance(attachment, Mapping):
                raise ConsoleDispatchCheckpointValidationError(
                    "Invalid accepted attachment."
                )
            position = attachment.get("position")
            data = attachment.get("data")
            mime_type = attachment.get("mime_type")
            display_name = attachment.get("display_name", "")
            if (
                type(position) is not int
                or position < 0
                or position in positions
                or type(data) is not bytes
                or type(mime_type) is not str
                or not mime_type.strip()
                or type(display_name) is not str
            ):
                raise ConsoleDispatchCheckpointValidationError(
                    "Invalid accepted attachment."
                )
            positions.add(position)
            normalized.append((position, data, mime_type, display_name))
        return tuple(sorted(normalized))

    @staticmethod
    def _valid_identifier(value: object) -> bool:
        return type(value) is str and _IDENTIFIER_RE.fullmatch(value) is not None

    @staticmethod
    def _positive_versions(*values: int) -> bool:
        return all(type(value) is int and value > 0 for value in values)

    @staticmethod
    def _owner_by_assistant(
        cursor: sqlite3.Cursor, assistant_message_id: str
    ) -> sqlite3.Row | None:
        return cursor.execute(
            _OWNER_SELECT + " WHERE checkpoint.assistant_message_id = ?",
            (assistant_message_id,),
        ).fetchone()

    @staticmethod
    def _checkpoint_from_row(
        row: sqlite3.Row,
    ) -> tuple[ConsoleDispatchCheckpoint | None, str | None]:
        try:
            thinking = read_thinking_blocks_json(row["thinking_blocks_json"])
            if (
                row["schema_version"] != 1
                or row["conversation_deleted"] != 0
                or row["user_role"] != "user"
                or row["assistant_role"] != "assistant"
                or row["user_conversation_id"] != row["conversation_id"]
                or row["assistant_conversation_id"] != row["conversation_id"]
                or row["user_deleted"] != 0
                or row["assistant_deleted"] != 0
                or row["current_user_version"] != row["user_message_version"]
                or row["current_assistant_version"] != row["assistant_message_version"]
                or row["assistant_state"] != row["state"]
                or row["provider_continuation_json"] is not None
                or thinking.warning is not None
                or not thinking.generation_actions_enabled
                or not ConsoleDispatchRepository._positive_versions(
                    row["checkpoint_revision"],
                    row["user_message_version"],
                    row["assistant_message_version"],
                )
                or row["origin"] not in {"manual", "queued"}
                or (row["origin"] == "manual" and row["queue_entry_id"] is not None)
                or (row["origin"] == "queued" and not row["queue_entry_id"])
            ):
                return None, "invalid_checkpoint_owner"
            identity_values = (
                row["assistant_message_id"],
                row["user_message_id"],
                row["conversation_id"],
                row["preparation_id"],
                row["attempt_id"],
            )
            if any(
                not ConsoleDispatchRepository._valid_identifier(value)
                for value in identity_values
            ) or (
                row["queue_entry_id"] is not None
                and not ConsoleDispatchRepository._valid_identifier(
                    row["queue_entry_id"]
                )
            ):
                return None, "invalid_checkpoint_identity"
            frozen_authority = parse_console_turn_library_authority_json(
                row["frozen_authority_json"]
            )
            if (
                row["state"] == ConsoleDispatchCheckpointState.ACCEPTED.value
                and row["attempt_id"] != frozen_authority.attempt_id
            ):
                return None, "invalid_checkpoint_identity"
            checkpoint = ConsoleDispatchCheckpoint(
                assistant_message_id=row["assistant_message_id"],
                user_message_id=row["user_message_id"],
                conversation_id=row["conversation_id"],
                preparation_id=row["preparation_id"],
                attempt_id=row["attempt_id"],
                state=ConsoleDispatchCheckpointState(row["state"]),
                checkpoint_revision=row["checkpoint_revision"],
                user_message_version=row["user_message_version"],
                assistant_message_version=row["assistant_message_version"],
                origin=row["origin"],
                queue_entry_id=row["queue_entry_id"],
                frozen_authority=frozen_authority,
                resolved_destination=parse_console_resolved_destination_json(
                    row["resolved_destination_json"]
                ),
                reconstructability=parse_console_dispatch_reconstructability_json(
                    row["reconstructability_json"]
                ),
            )
            return checkpoint, None
        except (ValueError, TypeError, ConsoleDispatchCheckpointValidationError):
            return None, "invalid_checkpoint_payload"

    @staticmethod
    def _matches_common(
        row: sqlite3.Row,
        *,
        checkpoint_revision: int,
        user_version: int,
        assistant_version: int,
    ) -> bool:
        return (
            row["conversation_deleted"] == 0
            and row["checkpoint_revision"] == checkpoint_revision
            and row["user_message_version"] == user_version
            and row["assistant_message_version"] == assistant_version
            and row["current_user_version"] == user_version
            and row["current_assistant_version"] == assistant_version
            and row["user_deleted"] == 0
            and row["assistant_deleted"] == 0
            and row["user_role"] == "user"
            and row["assistant_role"] == "assistant"
            and row["user_conversation_id"] == row["conversation_id"]
            and row["assistant_conversation_id"] == row["conversation_id"]
            and row["provider_continuation_json"] is None
        )

    @classmethod
    def _matches_transition(
        cls, row: sqlite3.Row, transition: ConsoleDispatchTransition
    ) -> bool:
        return (
            cls._matches_common(
                row,
                checkpoint_revision=transition.expected_checkpoint_revision,
                user_version=transition.expected_user_message_version,
                assistant_version=transition.expected_assistant_message_version,
            )
            and row["state"] == transition.expected_state.value
            and row["assistant_state"] == transition.expected_state.value
        )

    @classmethod
    def _matches_settlement(
        cls, row: sqlite3.Row, settlement: ConsoleAssistantSettlement
    ) -> bool:
        return (
            cls._matches_common(
                row,
                checkpoint_revision=settlement.expected_checkpoint_revision,
                user_version=settlement.expected_user_message_version,
                assistant_version=settlement.expected_assistant_message_version,
            )
            and row["state"] == settlement.expected_checkpoint_state.value
            and row["assistant_state"] == settlement.expected_checkpoint_state.value
        )

    @classmethod
    def _matches_handoff(
        cls, row: sqlite3.Row, handoff: ConsoleContinuationHandoff
    ) -> bool:
        return (
            cls._matches_common(
                row,
                checkpoint_revision=handoff.expected_checkpoint_revision,
                user_version=handoff.expected_user_message_version,
                assistant_version=handoff.expected_assistant_message_version,
            )
            and row["state"] == ConsoleDispatchCheckpointState.DISPATCH_STARTED.value
            and row["assistant_state"] == row["state"]
        )

    @staticmethod
    def _write_status(
        status: ConsoleDispatchResultStatus,
    ) -> ConsoleDispatchWriteResult:
        return ConsoleDispatchWriteResult(status, None, None, None)

    @staticmethod
    def _message_payload_hash(
        *,
        content: str,
        state: str,
        provider_continuation_json: str | None = None,
        thinking_blocks_json: str | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "assistant_generation_state": state,
            "content": content,
            "role": "assistant",
        }
        if provider_continuation_json is not None:
            payload["provider_continuation_json"] = provider_continuation_json
        if thinking_blocks_json is not None:
            payload["thinking_blocks_json"] = thinking_blocks_json
        return canonical_payload_hash(payload)
