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


def _live_rows(cursor, conversation_id, budget):
    """Validate all committed history, then admit live metadata before bodies."""
    from tldw_chatbook.DB.mobile_consolidation_receipts import _imported_chat_records

    imported = _imported_chat_records(cursor.connection, budget)["fleet_message"]
    count, id_bytes = cursor.execute(
        "SELECT count(*),coalesce(sum(length(CAST(message_id AS BLOB))+length(CAST(run_id AS BLOB))),0) FROM fleet_progress_messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    budget.container(count * 8)
    budget.reserve(id_bytes * 4)
    scan = cursor.connection.execute(
        "SELECT message_id,run_id,length(body) AS chars,length(CAST(body AS BLOB)) AS body_bytes,instr(body,char(0)) AS nul FROM fleet_progress_messages WHERE conversation_id = ? ORDER BY sequence",
        (conversation_id,),
    )
    rows = []
    try:
        while True:
            page = scan.fetchmany(100)
            if not page:
                break
            for row in page:
                if row["message_id"] in imported:
                    continue
                if row["nul"] or row["chars"] is None or row["body_bytes"] is None:
                    raise MessageError("durable_unavailable")
                rows.append(row)
    finally:
        scan.close()
    return rows


def _body_rows(cursor, rows, budget):
    result = []
    for row in rows:
        size = cursor.execute(
            "SELECT length(CAST(message_id AS BLOB))+length(CAST(handle_id AS BLOB))+length(CAST(run_id AS BLOB))+length(CAST(parent_run_id AS BLOB))+coalesce(length(CAST(chain_id AS BLOB)),0)+length(CAST(agent AS BLOB))+length(CAST(body AS BLOB))+length(CAST(created_at AS BLOB)) FROM fleet_progress_messages WHERE message_id=?",
            (row["message_id"],),
        ).fetchone()
        if size is None or size[0] is None:
            raise MessageError("durable_unavailable")
        budget.reserve(size[0] * 4)
        budget.container(24)
        result.append(
            cursor.execute(
                "SELECT message_id,handle_id,run_id,parent_run_id,chain_id,agent,body,created_at FROM fleet_progress_messages WHERE message_id=?",
                (row["message_id"],),
            ).fetchone()
        )
    return result


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
                from tldw_chatbook.Chat.console_trace_read_budget import (
                    _TraceReadBudget,
                )

                budget = _TraceReadBudget()
                metadata = _live_rows(cursor, conversation_id, budget)
                if (
                    len(metadata) > MAX_INBOX_PENDING
                    or sum(row["chars"] for row in metadata) > MAX_INBOX_PENDING_CHARS
                ):
                    raise MessageError("queue_full")
                rows = _body_rows(cursor, metadata, budget)
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

    def filter_restored(self, conversation_id, messages):
        """Exclude retained imported IDs before in-memory restoration limits."""
        from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
        from tldw_chatbook.DB.mobile_consolidation_receipts import (
            _imported_chat_records,
        )

        budget = _TraceReadBudget()
        with self.db.transaction() as cursor:
            imported = _imported_chat_records(cursor.connection, budget)[
                "fleet_message"
            ]
            budget.container(len(messages) * 2)
            return tuple(
                message for message in messages if message.message_id not in imported
            )

    def append(self, conversation_id: str, message: ProgressMessage) -> None:
        """Commit one admitted stable ID before its successful send receipt."""
        with _core_operation(self.db):
            if self.db.get_connection().in_transaction:
                raise MessageError("durable_unavailable")
            with self.db.transaction(immediate=True) as cursor:
                from tldw_chatbook.Chat.console_trace_read_budget import (
                    _TraceReadBudget,
                )

                rows = _live_rows(cursor, conversation_id, _TraceReadBudget())
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
