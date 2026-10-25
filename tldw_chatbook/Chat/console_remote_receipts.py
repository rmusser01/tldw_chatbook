"""Authenticated immutable submission tickets; native custody owns outcomes."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from tldw_chatbook.DB.console_command_receipts import (
    ACTIONS,
    ConsoleCommandReceipt,
    ConsoleCommandReceiptRepository,
    ConsoleReceiptCapacity,
    canonical,
)

from .console_remote_models import ConsoleHostIdentity, ConsoleTarget

if TYPE_CHECKING:
    from .console_remote_commands import ConsoleRemoteCommands

TICKET_LIFETIME_NS = 300 * 1_000_000_000


class ExpiredConsoleCommand(ValueError):
    """Submission expired; querying its durable outcome remains available."""


@dataclass(frozen=True)
class ConsoleCommandTicket:
    token: str
    command_id: str


class ConsoleRemoteReceipts:
    def __init__(
        self,
        commands: ConsoleRemoteCommands,
        repository: ConsoleCommandReceiptRepository,
        host_identity: ConsoleHostIdentity,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.commands = commands
        self.repository = repository
        self.host_identity = host_identity
        self.monotonic_ns = monotonic_ns
        self._key = secrets.token_bytes(32)
        repository.invalidate_prior_owner(
            canonical([host_identity.server_epoch, host_identity.console_id])
        )

    def _mac(self, domain: bytes, value: bytes) -> str:
        return hmac.new(self._key, domain + value, hashlib.sha256).hexdigest()

    def _fingerprint(self, payload: dict) -> str:
        if type(payload) is not dict:
            raise ValueError("invalid_command_payload")
        return self._mac(b"payload:", canonical(payload).encode())

    def issue(
        self, action: str, target: ConsoleHostIdentity | ConsoleTarget, payload: dict
    ) -> ConsoleCommandTicket:
        if action not in ACTIONS:
            raise ValueError("unsupported_command")
        if not isinstance(target, ConsoleHostIdentity) or any(
            getattr(target, key) != getattr(self.host_identity, key)
            for key in ("server_epoch", "console_id")
        ):
            raise ValueError("command_owner_changed")
        now = self.monotonic_ns()
        command_id = secrets.token_hex(16)
        value = dict(
            command_id=command_id,
            **self.host_identity.model_dump(),
            action=action,
            target=target.model_dump(),
            payload_fingerprint=self._fingerprint(payload),
            issued_ns=now,
            expires_ns=now + TICKET_LIFETIME_NS,
        )
        encoded = base64.urlsafe_b64encode(canonical(value).encode()).decode()
        return ConsoleCommandTicket(
            encoded + "." + self._mac(b"ticket:", encoded.encode()), command_id
        )

    def _verify(
        self, token: str, payload: dict, *, now_ns: int
    ) -> ConsoleCommandReceipt:
        try:
            if not isinstance(token, str) or len(token) > 16384:
                raise ValueError
            encoded, signature = token.split(".")
            if not hmac.compare_digest(
                self._mac(b"ticket:", encoded.encode()), signature
            ):
                raise ValueError
            data = json.loads(base64.urlsafe_b64decode(encoded))
            if any(
                data[key] != getattr(self.host_identity, key)
                for key in ("server_epoch", "console_id")
            ):
                raise ExpiredConsoleCommand("command_expired")
            if now_ns >= data["expires_ns"]:
                raise ExpiredConsoleCommand("command_expired")
            if not hmac.compare_digest(
                data["payload_fingerprint"], self._fingerprint(payload)
            ):
                raise ValueError
            target_type = (
                ConsoleTarget if "session_id" in data["target"] else ConsoleHostIdentity
            )
            data["target"] = target_type.model_validate(data["target"])
            return ConsoleCommandReceipt(**data, state="pending")
        except ExpiredConsoleCommand:
            raise
        except (ValueError, TypeError, KeyError, UnicodeError) as error:
            raise ValueError("invalid_command_ticket") from error

    async def submit(self, ticket: str, payload: dict) -> ConsoleCommandReceipt:
        # Use the same admission instant for verification and retention: a duplicate
        # must not prune its own receipt if the clock crosses expiry between them.
        now_ns = self.monotonic_ns()
        receipt = self._verify(ticket, payload, now_ns=now_ns)
        self.repository.prune_invalid(receipt.owner_epoch, now_ns)
        try:
            claim = self.repository.claim(receipt)
        except ConsoleReceiptCapacity:
            if self.monotonic_ns() >= receipt.expires_ns:
                raise ExpiredConsoleCommand("command_expired") from None
            if (
                receipt.action not in {"stop", "subagent_stop"}
                or not isinstance(receipt.target, ConsoleTarget)
                or not (receipt.target.run_id or receipt.target.turn_id)
                or payload
            ):
                raise
            outcome = await self.commands.execute(
                receipt.action, receipt.target, payload, receipt.command_id
            )
            return replace(
                receipt,
                state="accepted" if outcome["state"] == "accepted" else "refused",
                native_id=outcome.get("native_id"),
                reason=(
                    "stop_requested_exit_unconfirmed_receipt_unavailable_capacity"
                    if outcome["state"] == "accepted"
                    else "stop_refused_receipt_unavailable_capacity"
                ),
            )
        if not claim.inserted:
            return claim.receipt
        # Sole dispatch ownership is established, but repository work may have
        # consumed the remaining lifetime. No new native handoff may start then.
        if self.monotonic_ns() >= receipt.expires_ns:
            self.repository.transition(
                receipt.command_id,
                expected_state="pending",
                state="refused",
                reason="command_expired",
            )
            raise ExpiredConsoleCommand("command_expired")
        try:
            outcome = await self.commands.execute(
                receipt.action, receipt.target, payload, receipt.command_id
            )
            return self.repository.transition(
                receipt.command_id,
                expected_state="pending",
                state=outcome["state"],
                native_id=outcome.get("native_id"),
                reason={
                    "pending": "Preparing",
                    "accepted": (
                        "accepted_exit_unconfirmed"
                        if receipt.action in {"stop", "subagent_stop"}
                        else "answer_recorded"
                        if receipt.action == "decision"
                        else None
                    ),
                    "queued": "queued_consumption_unconfirmed",
                    "applied": None,
                    "refused": "command_refused",
                    "outcome_unknown": "handoff_interrupted",
                }[outcome["state"]],
            )
        except BaseException:
            self.repository.transition(
                receipt.command_id,
                expected_state="pending",
                state="outcome_unknown",
                reason="handoff_interrupted",
            )
            raise

    def query(self, command_id: str) -> ConsoleCommandReceipt | None:
        return self.repository.get(command_id)

    def observe(
        self,
        command_id: str,
        *,
        state: str,
        native_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Called only by native acceptance/queue/terminal custody observers."""
        current = self.repository.get(command_id)
        if current is not None and (
            current.state in {"pending", "queued", "outcome_unknown"}
            or current.action == "decision"
            and current.state == "accepted"
        ):
            self.repository.transition(
                command_id,
                expected_state=current.state,
                state=state,
                native_id=native_id,
                reason=reason,
            )
