"""Device-local persistence for per-conversation Console Library policy."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleConversationLibraryPolicy,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicyReadResult,
    ConsoleLibraryPolicyWriteResult,
    ConsoleLibraryPolicyWriteStatus,
    normalize_policy_read,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class ConsoleLibraryPolicyRepository:
    """Read and conditionally mutate device-local Library authority."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db

    def read(self, conversation_id: str) -> ConsoleLibraryPolicyReadResult:
        """Read one policy or an explicit fail-closed outcome."""
        if type(conversation_id) is not str or not conversation_id.strip():
            return normalize_policy_read(None)
        try:
            row = self.db.get_connection().execute(
                """
                SELECT conversation_id, schema_version, auto_retrieve_on_send,
                       assistant_library_access, policy_revision, updated_at
                  FROM console_conversation_library_policy
                 WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if row is None:
                return normalize_policy_read(None)
            if (
                type(row["schema_version"]) is not int
                or row["schema_version"] != 1
                or type(row["auto_retrieve_on_send"]) is not int
                or row["auto_retrieve_on_send"] not in (0, 1)
                or type(row["assistant_library_access"]) is not int
                or row["assistant_library_access"] not in (0, 1)
                or type(row["policy_revision"]) is not int
                or row["policy_revision"] < 1
                or not isinstance(row["updated_at"], (str, datetime))
                or not str(row["updated_at"]).strip()
            ):
                return normalize_policy_read(row)
            policy = ConsoleConversationLibraryPolicy(
                conversation_id=row["conversation_id"],
                auto_retrieve=(
                    ConsoleAutoRetrieve.AUTOMATIC
                    if row["auto_retrieve_on_send"] == 1
                    else ConsoleAutoRetrieve.NEVER
                ),
                assistant_access=(
                    ConsoleAssistantLibraryAccess.ALLOWED
                    if row["assistant_library_access"] == 1
                    else ConsoleAssistantLibraryAccess.BLOCKED
                ),
                policy_revision=row["policy_revision"],
                updated_at=str(row["updated_at"]),
            )
            return normalize_policy_read(policy)
        except Exception as exc:
            return normalize_policy_read(exc)

    def insert(
        self,
        conversation_id: str,
        candidate: ConsoleLibraryPolicyCandidate,
    ) -> ConsoleLibraryPolicyWriteResult:
        """Conditionally insert revision one without overwriting a race winner."""
        if not self._valid_candidate(candidate):
            return self._unavailable_write()
        try:
            with self.db.transaction(immediate=True) as cursor:
                conversation = cursor.execute(
                    "SELECT deleted FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if conversation is None or conversation["deleted"]:
                    return self._missing_conversation_write()
                try:
                    cursor.execute(
                        """
                        INSERT INTO console_conversation_library_policy (
                            conversation_id, schema_version,
                            auto_retrieve_on_send, assistant_library_access,
                            policy_revision, updated_at
                        ) VALUES (?, 1, ?, ?, 1, CURRENT_TIMESTAMP)
                        """,
                        (
                            conversation_id,
                            int(candidate.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC),
                            int(
                                candidate.assistant_access
                                is ConsoleAssistantLibraryAccess.ALLOWED
                            ),
                        ),
                    )
                except sqlite3.IntegrityError:
                    pass
                else:
                    row = self._read_row(cursor, conversation_id)
                    if row is None:
                        raise sqlite3.DatabaseError("Committed policy row unavailable")
                    snapshot = self._result_from_row(row).snapshot
                    return ConsoleLibraryPolicyWriteResult(
                        ConsoleLibraryPolicyWriteStatus.COMMITTED,
                        snapshot,
                    )
            winner = self.read(conversation_id)
            if winner.durable_policy is None:
                return self._unavailable_write()
            return ConsoleLibraryPolicyWriteResult(
                ConsoleLibraryPolicyWriteStatus.CONFLICT,
                winner.snapshot,
            )
        except Exception:
            return self._unavailable_write()

    def compare_and_swap(
        self,
        conversation_id: str,
        expected_revision: int,
        candidate: ConsoleLibraryPolicyCandidate,
    ) -> ConsoleLibraryPolicyWriteResult:
        """Commit exactly one expected revision or report conflict."""
        if (
            type(expected_revision) is not int
            or expected_revision < 1
            or not self._valid_candidate(candidate)
        ):
            return self._unavailable_write()
        try:
            with self.db.transaction(immediate=True) as cursor:
                conversation = cursor.execute(
                    "SELECT deleted FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if conversation is None or conversation["deleted"]:
                    return self._missing_conversation_write()
                updated = cursor.execute(
                    """
                    UPDATE console_conversation_library_policy
                       SET auto_retrieve_on_send = ?,
                           assistant_library_access = ?,
                           policy_revision = policy_revision + 1,
                           updated_at = CURRENT_TIMESTAMP
                     WHERE conversation_id = ? AND policy_revision = ?
                    """,
                    (
                        int(candidate.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC),
                        int(
                            candidate.assistant_access
                            is ConsoleAssistantLibraryAccess.ALLOWED
                        ),
                        conversation_id,
                        expected_revision,
                    ),
                )
                if updated.rowcount == 1:
                    row = self._read_row(cursor, conversation_id)
                    if row is None:
                        raise sqlite3.DatabaseError("Committed policy row unavailable")
                    return ConsoleLibraryPolicyWriteResult(
                        ConsoleLibraryPolicyWriteStatus.COMMITTED,
                        self._result_from_row(row).snapshot,
                    )
            current = self.read(conversation_id)
            return ConsoleLibraryPolicyWriteResult(
                ConsoleLibraryPolicyWriteStatus.CONFLICT,
                current.snapshot,
            )
        except Exception:
            return self._unavailable_write()

    @staticmethod
    def _read_row(cursor: sqlite3.Cursor, conversation_id: str) -> sqlite3.Row | None:
        return cursor.execute(
            """
            SELECT conversation_id, schema_version, auto_retrieve_on_send,
                   assistant_library_access, policy_revision, updated_at
              FROM console_conversation_library_policy
             WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()

    @staticmethod
    def _result_from_row(row: sqlite3.Row) -> ConsoleLibraryPolicyReadResult:
        return normalize_policy_read(
            ConsoleConversationLibraryPolicy(
                conversation_id=row["conversation_id"],
                auto_retrieve=(
                    ConsoleAutoRetrieve.AUTOMATIC
                    if row["auto_retrieve_on_send"] == 1
                    else ConsoleAutoRetrieve.NEVER
                ),
                assistant_access=(
                    ConsoleAssistantLibraryAccess.ALLOWED
                    if row["assistant_library_access"] == 1
                    else ConsoleAssistantLibraryAccess.BLOCKED
                ),
                policy_revision=row["policy_revision"],
                updated_at=str(row["updated_at"]),
            )
        )

    @staticmethod
    def _valid_candidate(candidate: object) -> bool:
        return isinstance(candidate, ConsoleLibraryPolicyCandidate) and isinstance(
            candidate.auto_retrieve, ConsoleAutoRetrieve
        ) and isinstance(candidate.assistant_access, ConsoleAssistantLibraryAccess)

    @staticmethod
    def _missing_conversation_write() -> ConsoleLibraryPolicyWriteResult:
        return ConsoleLibraryPolicyWriteResult(
            ConsoleLibraryPolicyWriteStatus.MISSING_CONVERSATION,
            normalize_policy_read(None).snapshot,
        )

    @staticmethod
    def _unavailable_write() -> ConsoleLibraryPolicyWriteResult:
        return ConsoleLibraryPolicyWriteResult(
            ConsoleLibraryPolicyWriteStatus.UNAVAILABLE,
            normalize_policy_read(RuntimeError()).snapshot,
        )
