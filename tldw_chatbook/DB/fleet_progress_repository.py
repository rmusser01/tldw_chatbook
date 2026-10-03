"""Chat-owned SQLite leaves for bounded pending reports (ADR-199).

These synchronous leaves use the existing owned transaction lifetime. They may
run under the progress queue lock and never call observers or live owners.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tldw_chatbook.Agents.fleet_messages import (
    MAX_CHILD_PENDING,
    MAX_CHILD_PENDING_CHARS,
    MAX_INBOX_PENDING,
    MAX_INBOX_PENDING_CHARS,
    MessageError,
    MessageIdentity,
    ProgressMessage,
)
from tldw_chatbook.Backup_Recovery.participants import _core_operation

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_transaction_contribution import (
        ConsoleTransactionWriter,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

_INSERT = """INSERT INTO fleet_progress_messages
(message_id, conversation_id, handle_id, run_id, parent_run_id, chain_id, agent, body, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""


def _parameters(conversation_id: str, message: ProgressMessage) -> tuple:
    identity = message.identity
    return (
        message.message_id,
        conversation_id,
        identity.handle_id,
        identity.run_id,
        identity.parent_run_id,
        identity.chain_id,
        identity.agent,
        message.body,
        message.created_at,
    )


class FleetProgressRepository:
    """Persist reports without allocating or restoring any execution capability."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db

    def load(self, conversation_id: str) -> tuple[ProgressMessage, ...]:
        """Read one bounded FIFO queue before acquiring native identity locks."""
        try:
            with self.db.transaction() as cursor:
                conversation = cursor.execute(
                    "SELECT deleted FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
                if conversation is None or conversation["deleted"]:
                    raise MessageError("unavailable")
                rows = cursor.execute(
                    "SELECT message_id, handle_id, run_id, parent_run_id, chain_id, agent, body, created_at "
                    "FROM fleet_progress_messages WHERE conversation_id = ? "
                    "ORDER BY sequence LIMIT ?",
                    (conversation_id, MAX_INBOX_PENDING + 1),
                ).fetchall()
        except MessageError:
            raise
        except Exception:  # noqa: BLE001 - loading cannot break unrelated owners
            raise MessageError("durable_unavailable") from None
        if (
            len(rows) > MAX_INBOX_PENDING
            or sum(len(row["body"]) for row in rows) > MAX_INBOX_PENDING_CHARS
        ):
            raise MessageError("queue_full")
        return tuple(
            ProgressMessage(
                row["message_id"],
                MessageIdentity(
                    row["handle_id"],
                    row["run_id"],
                    row["parent_run_id"],
                    row["chain_id"],
                    row["agent"],
                ),
                row["body"],
                row["created_at"],
            )
            for row in rows
        )

    def append(self, conversation_id: str, message: ProgressMessage) -> None:
        """Commit one admitted stable ID before its successful send receipt."""
        with _core_operation(self.db):
            if self.db.get_connection().in_transaction:
                raise MessageError("durable_unavailable")
            with self.db.transaction(immediate=True) as cursor:
                rows = cursor.execute(
                    "SELECT run_id, length(body) AS chars FROM fleet_progress_messages WHERE conversation_id = ? LIMIT ?",
                    (conversation_id, MAX_INBOX_PENDING + 1),
                ).fetchall()
                child = [
                    row for row in rows if row["run_id"] == message.identity.run_id
                ]
                if (
                    len(rows) >= MAX_INBOX_PENDING
                    or sum(row["chars"] for row in rows) + len(message.body)
                    > MAX_INBOX_PENDING_CHARS
                    or len(child) >= MAX_CHILD_PENDING
                    or sum(row["chars"] for row in child) + len(message.body)
                    > MAX_CHILD_PENDING_CHARS
                ):
                    raise MessageError("queue_full")
                cursor.execute(_INSERT, _parameters(conversation_id, message))

    def remove(self, conversation_id: str, message_ids: Sequence[str]) -> None:
        """Commit whole selected reports before returning collection/discard."""
        with _core_operation(self.db):
            if self.db.get_connection().in_transaction:
                raise MessageError("durable_unavailable")
            with self.db.transaction(immediate=True) as cursor:
                cursor.executemany(
                    "DELETE FROM fleet_progress_messages WHERE conversation_id = ? AND message_id = ?",
                    ((conversation_id, message_id) for message_id in message_ids),
                )
                if cursor.rowcount != len(message_ids):
                    raise MessageError("durable_unavailable")


@dataclass(frozen=True)
class FleetProgressPromotionContribution:
    """Frozen temporary reports inserted by the existing atomic Save writer."""

    messages: tuple[ProgressMessage, ...]

    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        if self.messages:
            writer.executemany(
                _INSERT,
                tuple(
                    _parameters(conversation_id, message) for message in self.messages
                ),
            )
