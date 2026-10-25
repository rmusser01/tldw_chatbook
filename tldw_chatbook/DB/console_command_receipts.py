"""Bounded, body-free command evidence in the existing AgentRuns owner."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from tldw_chatbook.Canvas.limits import validate_opaque_identifier
from tldw_chatbook.Chat.console_remote_models import ConsoleHostIdentity, ConsoleTarget

if TYPE_CHECKING:
    from .AgentRuns_DB import AgentRunsDB

MAX_COMMAND_RECEIPTS = 4096
ACTIONS = frozenset(
    {
        "send",
        "steer",
        "redirect",
        "stop",
        "subagent_guidance",
        "subagent_stop",
        "decision",
    }
)
STATES = frozenset(
    {"pending", "accepted", "queued", "applied", "refused", "outcome_unknown"}
)
# This installed SQL is used by both the native owner and exact recovery registry.
COMMAND_RECEIPT_MIGRATION = tuple(
    statement.strip()
    for statement in (
        Path(__file__).with_name("migrations")
        / "agent_runs_v24_to_v25_console_command_receipts.sql"
    )
    .read_text(encoding="utf-8")
    .split(";")
    if statement.strip()
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def canonical(value: object) -> str:
    """Encode identities and payloads without ambiguous NaN or key ordering."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ConsoleCommandReceipt:
    command_id: str
    server_epoch: str
    console_id: str
    action: str
    target: ConsoleHostIdentity | ConsoleTarget
    payload_fingerprint: str
    state: str
    issued_ns: int
    expires_ns: int
    native_id: str | None = None
    reason: str | None = None
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        validate_opaque_identifier(self.command_id)
        if self.native_id is not None:
            validate_opaque_identifier(self.native_id)
        if (
            not isinstance(self.target, ConsoleHostIdentity)
            or self.target.server_epoch != self.server_epoch
            or self.target.console_id != self.console_id
            or self.action not in ACTIONS
            or self.state not in STATES
            or not isinstance(self.payload_fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.payload_fingerprint) is None
            or type(self.issued_ns) is not int
            or type(self.expires_ns) is not int
            or not 0 <= self.issued_ns < self.expires_ns < 2**63
            or not isinstance(self.updated_at, str)
            or len(self.updated_at) > 128
            or (
                self.reason is not None
                and (not isinstance(self.reason, str) or len(self.reason) > 256)
            )
        ):
            raise ValueError("invalid_command_receipt")

    @property
    def owner_epoch(self) -> str:
        return canonical([self.server_epoch, self.console_id])

    def public(self) -> dict:
        """Return inert receipt facts without payloads or authentication material."""
        return {
            "command_id": self.command_id,
            "server_epoch": self.server_epoch,
            "console_id": self.console_id,
            "action": self.action,
            "state": self.state,
            "native_id": self.native_id,
            "reason": self.reason,
            "issued_ns": self.issued_ns,
            "expires_ns": self.expires_ns,
            "updated_at": self.updated_at,
            "target": self.target.model_dump(),
        }


@dataclass(frozen=True)
class ConsoleCommandClaim:
    receipt: ConsoleCommandReceipt
    inserted: bool


@dataclass(frozen=True)
class ConsoleReceiptPage:
    items: tuple[ConsoleCommandReceipt, ...]
    next_cursor: str | None


class ConsoleReceiptCapacity(ValueError):
    """Retention cannot erase evidence to admit another command."""


class ConsoleCommandReceiptRepository:
    def __init__(self, runs_db: AgentRunsDB) -> None:
        self.runs_db = runs_db

    @staticmethod
    def _decode(row) -> ConsoleCommandReceipt | None:
        if row is None:
            return None
        values = dict(row)
        server_epoch, console_id = json.loads(values.pop("owner_epoch"))
        if console_id != values["console_id"]:
            raise ValueError("invalid_receipt_owner")
        target = json.loads(values.pop("target_json"))
        target_type = ConsoleTarget if "session_id" in target else ConsoleHostIdentity
        return ConsoleCommandReceipt(
            server_epoch=server_epoch,
            target=target_type.model_validate(target),
            **values,
        )

    def get(self, command_id: str) -> ConsoleCommandReceipt | None:
        with self.runs_db.connection() as connection:
            return self._decode(
                connection.execute(
                    "SELECT * FROM console_command_receipts WHERE command_id=?",
                    (command_id,),
                ).fetchone()
            )

    def claim(self, receipt: ConsoleCommandReceipt) -> ConsoleCommandClaim:
        """Only the atomic inserter may dispatch; pending is never ownership."""
        with self.runs_db.transaction(immediate=True) as connection:
            prior = self._decode(
                connection.execute(
                    "SELECT * FROM console_command_receipts WHERE command_id=?",
                    (receipt.command_id,),
                ).fetchone()
            )
            if prior is not None:
                if any(
                    getattr(prior, key) != getattr(receipt, key)
                    for key in (
                        "owner_epoch",
                        "action",
                        "target",
                        "payload_fingerprint",
                        "issued_ns",
                        "expires_ns",
                    )
                ):
                    raise ValueError("command_identity_mismatch")
                return ConsoleCommandClaim(prior, False)
            if (
                connection.execute(
                    "SELECT COUNT(*) FROM console_command_receipts"
                ).fetchone()[0]
                >= MAX_COMMAND_RECEIPTS
            ):
                raise ConsoleReceiptCapacity("command_receipt_capacity")
            connection.execute(
                "INSERT INTO console_command_receipts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    receipt.command_id,
                    receipt.owner_epoch,
                    receipt.console_id,
                    receipt.action,
                    canonical(receipt.target.model_dump()),
                    receipt.payload_fingerprint,
                    receipt.state,
                    receipt.issued_ns,
                    receipt.expires_ns,
                    receipt.native_id,
                    receipt.reason,
                    receipt.updated_at,
                ),
            )
            return ConsoleCommandClaim(receipt, True)

    def transition(
        self,
        command_id: str,
        *,
        expected_state: str,
        state: str,
        native_id: str | None = None,
        reason: str | None = None,
    ) -> ConsoleCommandReceipt:
        if state not in STATES or expected_state not in STATES:
            raise ValueError("invalid_receipt_state")
        if native_id is not None:
            validate_opaque_identifier(native_id)
        if reason is not None and (not isinstance(reason, str) or len(reason) > 256):
            raise ValueError("invalid_receipt_reason")
        with self.runs_db.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE console_command_receipts SET state=?, native_id=COALESCE(?,native_id), "
                "reason=?, updated_at=? WHERE command_id=? AND state=?",
                (state, native_id, reason, _now(), command_id, expected_state),
            )
            result = self._decode(
                connection.execute(
                    "SELECT * FROM console_command_receipts WHERE command_id=?",
                    (command_id,),
                ).fetchone()
            )
            if result is None:
                raise ValueError("unknown_command")
            return result

    def invalidate_prior_owner(self, owner_epoch: str) -> None:
        """A lost Preparing handoff has no transaction proving acceptance."""
        with self.runs_db.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE console_command_receipts SET state='outcome_unknown', "
                "reason='owner_lost',updated_at=? WHERE owner_epoch<>? AND state='pending'",
                (_now(), owner_epoch),
            )

    def prune_invalid(self, owner_epoch: str, now_ns: int) -> int:
        """Only expired completed evidence in this monotonic epoch is disposable."""
        with self.runs_db.transaction(immediate=True) as connection:
            return connection.execute(
                "DELETE FROM console_command_receipts WHERE owner_epoch=? AND expires_ns<=? "
                "AND state NOT IN ('pending','outcome_unknown')",
                (owner_epoch, now_ns),
            ).rowcount

    def list_unresolved(
        self, *, cursor: str | None = None, limit: int = 50
    ) -> ConsoleReceiptPage:
        if (
            type(limit) is not int
            or not 1 <= limit <= 100
            or (
                cursor is not None
                and (not isinstance(cursor, str) or len(cursor) > 256)
            )
        ):
            raise ValueError("invalid_receipt_page")
        with self.runs_db.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM console_command_receipts WHERE state IN ('pending','outcome_unknown') "
                "AND command_id>? ORDER BY command_id LIMIT ?",
                (cursor or "", limit + 1),
            ).fetchall()
        items = tuple(self._decode(row) for row in rows[:limit])
        return ConsoleReceiptPage(
            items, items[-1].command_id if len(rows) > limit else None
        )
