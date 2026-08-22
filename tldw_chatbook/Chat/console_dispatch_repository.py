"""Transactional persistence for Console dispatch recovery ownership."""

from __future__ import annotations

import json
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
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationValidationError,
    dump_provider_continuation_json,
    parse_provider_continuation_json,
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
           assistant_message.provider_continuation_json AS provider_continuation_json
      FROM console_dispatch_checkpoints AS checkpoint
      LEFT JOIN messages AS user_message
        ON user_message.id = checkpoint.user_message_id
      LEFT JOIN messages AS assistant_message
        ON assistant_message.id = checkpoint.assistant_message_id
"""


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
        now = self.db._get_current_utc_timestamp_iso()

        cursor.execute(
            """
            INSERT INTO messages (
                id, conversation_id, parent_message_id, sender, content,
                image_data, image_mime_type, timestamp, ranking,
                last_modified, client_id, version, deleted, role,
                usage_json, metadata_json, provider_continuation_json,
                assistant_generation_state
            ) VALUES (?, ?, ?, 'user', ?, NULL, NULL, ?, NULL,
                      ?, ?, 1, 0, 'user', NULL, NULL, NULL, NULL)
            """,
            (
                acceptance.user_message_id,
                acceptance.conversation_id,
                acceptance.parent_message_id,
                acceptance.user_content,
                now,
                now,
                self.db.client_id,
            ),
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
        for contribution in acceptance.contributions:
            contribution.write(
                cursor=cursor,
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
            rows = self.db.get_connection().execute(
                _OWNER_SELECT + " WHERE checkpoint.conversation_id = ?",
                (conversation_id,),
            ).fetchall()
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

    def cas_state(
        self, transition: ConsoleDispatchTransition
    ) -> ConsoleDispatchWriteResult:
        """Apply an expected-revision accepted/dispatch-started transition."""
        if (
            not isinstance(transition.expected_state, ConsoleDispatchCheckpointState)
            or not isinstance(transition.new_state, ConsoleDispatchCheckpointState)
            or transition.expected_state is not ConsoleDispatchCheckpointState.ACCEPTED
            or transition.new_state
            is not ConsoleDispatchCheckpointState.DISPATCH_STARTED
            or not self._positive_versions(
                transition.expected_checkpoint_revision,
                transition.expected_user_message_version,
                transition.expected_assistant_message_version,
            )
            or type(transition.new_attempt_id) is not str
            or not transition.new_attempt_id.strip()
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
                raise sqlite3.IntegrityError("Checkpoint state CAS lost after owner CAS.")
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
        with self.db.transaction(immediate=True) as cursor:
            row = self._owner_by_assistant(cursor, settlement.assistant_message_id)
            if row is None:
                return self._write_status(ConsoleDispatchResultStatus.NOT_FOUND)
            if not self._matches_settlement(row, settlement):
                return self._write_status(ConsoleDispatchResultStatus.CONFLICT)
            next_message_version = settlement.expected_assistant_message_version + 1
            now = self.db._get_current_utc_timestamp_iso()
            updated = cursor.execute(
                """
                UPDATE messages
                   SET content = ?, metadata_json = ?,
                       provider_continuation_json = NULL,
                       assistant_generation_state = ?, version = ?,
                       last_modified = ?, client_id = ?
                 WHERE id = ? AND conversation_id = ? AND role = 'assistant'
                   AND assistant_generation_state = ? AND version = ? AND deleted = 0
                """,
                (
                    settlement.content,
                    settlement.metadata_json,
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
                ),
            )

    @staticmethod
    def _validate_acceptance(acceptance: ConsoleDurableTurnAcceptance) -> None:
        if (
            not isinstance(acceptance, ConsoleDurableTurnAcceptance)
            or acceptance.origin not in {"manual", "queued"}
            or (acceptance.origin == "manual" and acceptance.queue_entry_id is not None)
            or (acceptance.origin == "queued" and not acceptance.queue_entry_id)
            or any(
                type(value) is not str or not value.strip()
                for value in (
                    acceptance.conversation_id,
                    acceptance.user_message_id,
                    acceptance.assistant_message_id,
                    acceptance.preparation_id,
                    acceptance.attempt_id,
                )
            )
            or acceptance.user_message_id == acceptance.assistant_message_id
            or type(acceptance.user_content) is not str
            or type(acceptance.attachments) is not tuple
            or type(acceptance.contributions) is not tuple
        ):
            raise ConsoleDispatchCheckpointValidationError(
                "Invalid durable turn acceptance."
            )

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
            if (
                row["schema_version"] != 1
                or row["user_role"] != "user"
                or row["assistant_role"] != "assistant"
                or row["user_conversation_id"] != row["conversation_id"]
                or row["assistant_conversation_id"] != row["conversation_id"]
                or row["user_deleted"] != 0
                or row["assistant_deleted"] != 0
                or row["current_user_version"] != row["user_message_version"]
                or row["current_assistant_version"]
                != row["assistant_message_version"]
                or row["assistant_state"] != row["state"]
                or row["provider_continuation_json"] is not None
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
                frozen_authority=parse_console_turn_library_authority_json(
                    row["frozen_authority_json"]
                ),
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
            row["checkpoint_revision"] == checkpoint_revision
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
            and row["assistant_state"]
            == settlement.expected_checkpoint_state.value
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
            and row["state"]
            == ConsoleDispatchCheckpointState.DISPATCH_STARTED.value
            and row["assistant_state"] == row["state"]
        )

    @staticmethod
    def _write_status(status: ConsoleDispatchResultStatus) -> ConsoleDispatchWriteResult:
        return ConsoleDispatchWriteResult(status, None, None, None)

    @staticmethod
    def _message_payload_hash(
        *,
        content: str,
        state: str,
        provider_continuation_json: str | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "assistant_generation_state": state,
            "content": content,
            "role": "assistant",
        }
        if provider_continuation_json is not None:
            payload["provider_continuation_json"] = provider_continuation_json
        return canonical_payload_hash(payload)
