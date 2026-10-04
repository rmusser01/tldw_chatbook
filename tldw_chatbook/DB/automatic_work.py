"""Atomic automatic-work ledger on AgentRunsDB's thread-local SQLite owner.

All execution-authorizing mutations commit under FULL synchronization. The
ordinary per-step run ledger keeps its existing connection policy. UI state,
provider billing, and message/tool bodies are not stored here.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from types import MappingProxyType
from typing import TYPE_CHECKING
from uuid import uuid4

from tldw_chatbook.Agents.agent_models import TERMINAL_RUN_STATUSES
from tldw_chatbook.Agents.automatic_work_budget import (
    AutomaticChatStartAttempt,
    AutomaticWakeAttempt,
    AutomaticWorkLimits,
    AutomaticWorkRefused,
    AutomaticWorkReservation,
    AutomaticWorkSnapshot,
)

if TYPE_CHECKING:
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


# A persisted monotonic reading is comparable only within this process. Store
# its owner so separate DB handles share the original anchor without a cache.
_CLOCK_OWNER_ID = uuid4().hex

# Failed durable settlement must deny the same runtime's sibling work, including
# through another handle. Only identities live here; recovery remains durable.
_UNCONFIRMED_STARTS: dict[tuple[str, str, str], tuple[str | None, object]] = {}
_UNCONFIRMED_STARTS_LOCK = threading.RLock()


def _identity(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError("identity must be a nonempty bounded string")
    return value


class AutomaticWorkLedger:
    """Chain allowance and admission state owned by one run database."""

    def __init__(
        self,
        db: AgentRunsDB,
        *,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._db = db
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock

    def _restriction_database(self) -> str:
        if not self._db.is_memory_db:
            return str(self._db.db_path.resolve())
        # Repeated :memory: names and eventual Python object-ID reuse must not
        # couple independent stores. A ledger shares its DB's private identity.
        with _UNCONFIRMED_STARTS_LOCK:
            identity = getattr(self._db, "_automatic_work_restriction_id", None)
            if identity is None:
                identity = "memory:" + uuid4().hex
                self._db._automatic_work_restriction_id = identity
            return identity

    def _restrict_chat_start(
        self, attempt_id: str, *, owner_id: str, chain_id: str | None
    ) -> None:
        """Retain uncertain authority before settlement I/O, without reading DB."""
        with _UNCONFIRMED_STARTS_LOCK:
            _UNCONFIRMED_STARTS[
                (self._restriction_database(), owner_id, attempt_id)
            ] = (chain_id, object())

    def _clear_chat_start_restriction(self, attempt_id: str, *, owner_id: str) -> None:
        with _UNCONFIRMED_STARTS_LOCK:
            _UNCONFIRMED_STARTS.pop(
                (self._restriction_database(), owner_id, attempt_id), None
            )

    def _check_unconfirmed_starts(
        self, conn: sqlite3.Connection, chain_id: str
    ) -> None:
        current = conn.execute(
            "SELECT owner_id FROM automatic_work_runtime_owner WHERE singleton=1"
        ).fetchone()
        with _UNCONFIRMED_STARTS_LOCK:
            members = tuple(
                entry[0]
                for (database, owner, _), entry in _UNCONFIRMED_STARTS.items()
                if database == self._restriction_database()
                and (current is None or owner == current["owner_id"])
            )
        if not members:
            return
        root = self._allowance_chain(conn, chain_id)["id"]
        if any(
            member is None or self._allowance_chain(conn, member)["id"] == root
            for member in members
        ):
            raise AutomaticWorkRefused("settlement_unconfirmed")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit an execution fence durably; restore policy on every exit.

        Nested transactions are rejected before changing the connection. Catch
        BaseException too: cancellation must never leave a fence half-open.
        """
        with self._db.connection() as conn:
            if conn.in_transaction:
                raise RuntimeError("automatic work transaction cannot be nested")
            previous = int(conn.execute("PRAGMA synchronous").fetchone()[0])
            conn.execute("PRAGMA synchronous=FULL")
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    yield conn
                    conn.commit()
                except BaseException:
                    conn.rollback()
                    raise
            finally:
                # The value comes only from SQLite's bounded pragma enum.
                conn.execute(f"PRAGMA synchronous={previous}")

    @contextmanager
    def _admission_transaction(self, chain_id: str) -> Iterator[sqlite3.Connection]:
        """Roll back refused work, retaining clock observations and its pause."""
        refusal = None
        with self.transaction() as conn:
            conn.execute("SAVEPOINT automatic_admission")
            try:
                yield conn
            except AutomaticWorkRefused as exc:
                observed = self._allowance_chain(conn, chain_id)
                root_id = observed["id"]
                conn.execute("ROLLBACK TO automatic_admission")
                # Elapsed-time observations are not an admission. In particular,
                # a refused first call in a replacement process must keep its
                # anchor so a retry cannot gain unattended time.
                conn.execute(
                    "UPDATE automatic_work_chains SET last_observed_at=?, clock_owner_id=?, started_monotonic=? WHERE id=?",
                    (
                        observed["last_observed_at"],
                        observed["clock_owner_id"],
                        observed["started_monotonic"],
                        root_id,
                    ),
                )
                refusal = exc
                reason = str(exc)
                status = (
                    "review_required"
                    if reason
                    in {
                        "clock_reversed",
                        "clock_unknown",
                        "usage_unknown",
                        "interrupted_work",
                        "review_required",
                    }
                    else "paused"
                )
                # A stale callback has no authority to pause its replacement's
                # otherwise healthy chain. Its admission alone is refused.
                if reason not in {"runtime_owner_replaced", "settlement_unconfirmed"}:
                    conn.execute(
                        "UPDATE automatic_work_chains SET status=?, pause_reason=? "
                        "WHERE id=? AND status!='review_required'",
                        (status, reason, root_id),
                    )
            finally:
                conn.execute("RELEASE automatic_admission")
        if refusal is not None:
            raise refusal

    def _check_admission(
        self,
        conn: sqlite3.Connection,
        chain_id: str,
        *,
        kind: str,
        amount: int = 0,
        limits: AutomaticWorkLimits | None = None,
    ) -> None:
        self._check_unconfirmed_starts(conn, chain_id)
        snapshot = self._snapshot(conn, chain_id)
        if snapshot.status == "review_required":
            raise AutomaticWorkRefused(snapshot.pause_reason or "review_required")
        # Monotonic exhaustion may precede the wall-clock deadline. It is final
        # for this chain, including on another handle with a fresh local clock.
        if snapshot.pause_reason == "wall_budget":
            raise AutomaticWorkRefused("wall_budget")
        effective = (
            snapshot.limits
            if limits is None
            else AutomaticWorkLimits(
                **{
                    key: min(value, getattr(limits, key))
                    for key, value in asdict(snapshot.limits).items()
                }
            )
        )
        chain = self._allowance_chain(conn, chain_id)
        chain_id = chain["id"]
        now, monotonic = self._wall_clock(), self._monotonic_clock()
        if not math.isfinite(now) or not math.isfinite(monotonic):
            raise AutomaticWorkRefused("clock_unknown")
        if now < chain["last_observed_at"]:
            raise AutomaticWorkRefused("clock_reversed")
        if effective.wall_seconds == 0:
            raise AutomaticWorkRefused("wall_budget")
        if snapshot.started_at is not None:
            started_monotonic = chain["started_monotonic"]
            if chain["clock_owner_id"] != _CLOCK_OWNER_ID:
                # Re-anchor an unambiguous recovered chain to this process,
                # carrying forward elapsed wall time and the original deadline.
                started_monotonic = monotonic - (now - snapshot.started_at)
                conn.execute(
                    "UPDATE automatic_work_chains SET clock_owner_id=?, started_monotonic=? WHERE id=?",
                    (_CLOCK_OWNER_ID, started_monotonic, chain_id),
                )
            if started_monotonic is None or not math.isfinite(started_monotonic):
                raise AutomaticWorkRefused("clock_unknown")
            if monotonic < started_monotonic:
                raise AutomaticWorkRefused("clock_reversed")
            elapsed_now = max(now, snapshot.started_at + monotonic - started_monotonic)
            deadline = min(
                snapshot.deadline_at, snapshot.started_at + effective.wall_seconds
            )
            if elapsed_now >= deadline:
                raise AutomaticWorkRefused("wall_budget")
        conn.execute(
            "UPDATE automatic_work_chains SET last_observed_at=? WHERE id=?",
            (now, chain_id),
        )
        for resource, ceiling in effective.resources().items():
            requested = amount if resource == kind else 0
            if (
                snapshot.used[resource] + snapshot.reserved[resource] + requested
                > ceiling
            ):
                raise AutomaticWorkRefused(f"{resource}_budget")
        conn.execute(
            "UPDATE automatic_work_chains SET status='active', pause_reason=NULL WHERE id=?",
            (chain_id,),
        )

    def _start_automatic(self, conn: sqlite3.Connection, chain_id: str) -> None:
        chain = self._allowance_chain(conn, chain_id)
        chain_id = chain["id"]
        if chain["started_at"] is None:
            limits = AutomaticWorkLimits(**json.loads(chain["limits_json"]))
            now = self._wall_clock()
            conn.execute(
                "UPDATE automatic_work_chains SET started_at=?, deadline_at=?, last_observed_at=?, "
                "clock_owner_id=?, started_monotonic=? WHERE id=?",
                (
                    now,
                    now + limits.wall_seconds,
                    now,
                    _CLOCK_OWNER_ID,
                    self._monotonic_clock(),
                    chain_id,
                ),
            )

    def create_chain(
        self,
        conversation_id: str,
        *,
        root_submission_id: str,
        limits: AutomaticWorkLimits | None = None,
    ) -> str:
        """Idempotently establish an immutable allowance for accepted user work."""
        _identity(conversation_id)
        _identity(root_submission_id)
        limits = limits if limits is not None else AutomaticWorkLimits.from_settings()
        if not isinstance(limits, AutomaticWorkLimits):
            raise TypeError("limits must be AutomaticWorkLimits")
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT id, conversation_id FROM automatic_work_chains WHERE root_submission_id=?",
                (root_submission_id,),
            ).fetchone()
            if existing:
                if existing["conversation_id"] != conversation_id:
                    raise ValueError("submission scope conflict")
                return str(existing["id"])
            chain_id = uuid4().hex
            now = self._wall_clock()
            conn.execute(
                "INSERT INTO automatic_work_chains "
                "(id, conversation_id, root_submission_id, limits_json, created_at, last_observed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    chain_id,
                    conversation_id,
                    root_submission_id,
                    json.dumps(asdict(limits)),
                    now,
                    now,
                ),
            )
        return chain_id

    @staticmethod
    def _check_runtime_owner(conn: sqlite3.Connection, owner_id: str) -> None:
        """Refuse revoked execution authority within the admission transaction."""
        _identity(owner_id)
        current = conn.execute(
            "SELECT owner_id FROM automatic_work_runtime_owner WHERE singleton=1"
        ).fetchone()
        if current is not None and current["owner_id"] != owner_id:
            raise AutomaticWorkRefused("runtime_owner_replaced")

    @staticmethod
    def _chain(conn: sqlite3.Connection, chain_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM automatic_work_chains WHERE id=?", (chain_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown automatic work chain")
        return row

    def _allowance_chain(self, conn: sqlite3.Connection, chain_id: str) -> sqlite3.Row:
        """Resolve one direct canonical root without changing local scope."""
        member = self._chain(conn, chain_id)
        root_id = member["allowance_root_chain_id"]
        if root_id is None:
            return member
        root = self._chain(conn, root_id)
        if root["allowance_root_chain_id"] is not None:
            raise ValueError("automatic allowance must name a direct root")
        return root

    def allowance_root(self, chain_id: str) -> str:
        """Return the canonical allowance identity without renewing limits."""
        with self._db.connection() as conn:
            return str(self._allowance_chain(conn, chain_id)["id"])

    @staticmethod
    def _reservation(
        conn: sqlite3.Connection, reservation_id: str, owner_id: str
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM automatic_work_reservations WHERE id=?", (reservation_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown automatic work reservation")
        if row["owner_id"] != owner_id:
            raise ValueError("reservation owner mismatch")
        return row

    @staticmethod
    def _reservation_view(row: sqlite3.Row) -> AutomaticWorkReservation:
        return AutomaticWorkReservation(
            **{key: row[key] for key in AutomaticWorkReservation.__dataclass_fields__}
        )

    def _snapshot(
        self, conn: sqlite3.Connection, chain_id: str
    ) -> AutomaticWorkSnapshot:
        member = self._chain(conn, chain_id)
        chain = self._allowance_chain(conn, chain_id)
        limits = AutomaticWorkLimits(**json.loads(chain["limits_json"]))
        used = dict.fromkeys(limits.resources(), 0)
        reserved = dict(used)
        uncertain = chain["status"] == "review_required"
        for row in conn.execute(
            "SELECT reservation.* FROM automatic_work_reservations AS reservation "
            "JOIN automatic_work_chains AS member ON member.id = reservation.chain_id "
            "WHERE member.id = ? OR member.allowance_root_chain_id = ?",
            (chain["id"], chain["id"]),
        ):
            if row["state"] == "released":
                continue
            if row["state"] in {"reserved", "uncertain"}:
                reserved[row["kind"]] += row["amount"]
            else:
                used[row["kind"]] += (
                    row["actual_amount"]
                    if row["actual_amount"] is not None
                    else row["amount"]
                )
            uncertain |= row["state"] == "uncertain"
        available = {
            kind: max(0, limit - used[kind] - reserved[kind])
            for kind, limit in limits.resources().items()
        }
        return AutomaticWorkSnapshot(
            chain_id,
            member["conversation_id"],
            limits,
            chain["status"],
            chain["pause_reason"],
            chain["started_at"],
            chain["deadline_at"],
            MappingProxyType(used),
            MappingProxyType(reserved),
            MappingProxyType(available),
            uncertain,
        )

    def snapshot(self, chain_id: str) -> AutomaticWorkSnapshot:
        """Read a consistent immutable projection without renewing allowance."""
        with self._db.connection() as conn:
            # A read transaction keeps chain state and reservations coherent
            # when another worker commits between the two SELECTs.
            if conn.in_transaction:
                return self._snapshot(conn, chain_id)
            conn.execute("BEGIN")
            try:
                return self._snapshot(conn, chain_id)
            finally:
                conn.rollback()

    def reserve(
        self,
        chain_id: str,
        *,
        reservation_id: str,
        owner_id: str,
        kind: str,
        amount: int,
        limits: AutomaticWorkLimits | None = None,
    ) -> AutomaticWorkReservation:
        """Reserve finite allowance atomically; duplicate IDs never recharge it."""
        _identity(reservation_id)
        _identity(owner_id)
        if kind not in AutomaticWorkLimits().resources():
            raise ValueError("unknown automatic resource kind")
        if type(amount) is not int or not 0 < amount < 2**63:
            raise ValueError("amount must be a positive SQLite integer")
        with self._admission_transaction(chain_id) as conn:
            self._check_runtime_owner(conn, owner_id)
            existing = conn.execute(
                "SELECT * FROM automatic_work_reservations WHERE id=?",
                (reservation_id,),
            ).fetchone()
            if existing:
                if (
                    existing["chain_id"],
                    existing["owner_id"],
                    existing["kind"],
                    existing["amount"],
                ) != (chain_id, owner_id, kind, amount):
                    raise ValueError("reservation identity conflict")
                return self._reservation_view(existing)
            self._check_admission(
                conn, chain_id, kind=kind, amount=amount, limits=limits
            )
            now = self._wall_clock()
            conn.execute(
                "INSERT INTO automatic_work_reservations "
                "(id, chain_id, owner_id, kind, amount, state, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'reserved', ?, ?)",
                (reservation_id, chain_id, owner_id, kind, amount, now, now),
            )
            return self._reservation_view(
                self._reservation(conn, reservation_id, owner_id)
            )

    def check_active(
        self,
        chain_id: str,
        *,
        owner_id: str | None = None,
        limits: AutomaticWorkLimits | None = None,
    ) -> AutomaticWorkSnapshot:
        """Observe elapsed/live bounds durably without consuming a resource."""
        with self._admission_transaction(chain_id) as conn:
            if owner_id is not None:
                self._check_runtime_owner(conn, owner_id)
            snapshot = self._snapshot(conn, chain_id)
            if (
                snapshot.status == "paused"
                and snapshot.pause_reason != "autowake_disabled"
            ):
                raise AutomaticWorkRefused(snapshot.pause_reason or "automatic_paused")
            self._check_admission(conn, chain_id, kind="model_call", limits=limits)
            return self._snapshot(conn, chain_id)

    def pause(
        self, chain_id: str, reason: str, *, review_required: bool = False
    ) -> None:
        """Persist a bounded metadata-only pause, never downgrading review."""
        if (
            not isinstance(reason, str)
            or not 1 <= len(reason) <= 64
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz_0123456789"
                for character in reason
            )
        ):
            raise ValueError("pause reason must be a bounded reason code")
        with self.transaction() as conn:
            root_id = self._allowance_chain(conn, chain_id)["id"]
            conn.execute(
                "UPDATE automatic_work_chains SET status=?, pause_reason=? "
                "WHERE id=? AND status!='review_required'",
                ("review_required" if review_required else "paused", reason, root_id),
            )

    def admit_call(
        self,
        chain_id: str,
        *,
        call_id: str,
        owner_id: str,
        input_tokens: int,
        output_tokens: int,
        limits: AutomaticWorkLimits | None = None,
    ) -> str | None:
        """Atomically charge one generation and its estimate before dispatch.

        Returns the token reservation ID only for a new dispatch; an identical
        retry returns None. Both resource records commit under FULL together.
        """
        _identity(call_id)
        _identity(owner_id)
        if (
            type(input_tokens) is not int
            or input_tokens < 0
            or type(output_tokens) is not int
            or output_tokens <= 0
            or input_tokens + output_tokens >= 2**63
        ):
            raise ValueError("call token counts must be bounded nonnegative integers")
        token_id = _identity(call_id + ":tokens")
        model_id = _identity(call_id + ":model")
        amount = input_tokens + output_tokens
        with self._admission_transaction(chain_id) as conn:
            self._check_runtime_owner(conn, owner_id)
            existing = conn.execute(
                "SELECT * FROM automatic_work_reservations WHERE id IN (?, ?)",
                (token_id, model_id),
            ).fetchall()
            if existing:
                if len(existing) != 2 or any(
                    (row["chain_id"], row["owner_id"], row["kind"], row["amount"])
                    != (
                        chain_id,
                        owner_id,
                        "tokens" if row["id"] == token_id else "model_call",
                        amount if row["id"] == token_id else 1,
                    )
                    for row in existing
                ):
                    raise ValueError("call reservation identity conflict")
                return None
            snapshot = self._snapshot(conn, chain_id)
            output_limit = snapshot.limits.output_tokens
            if limits is not None:
                output_limit = min(output_limit, limits.output_tokens)
            if output_tokens > output_limit:
                raise AutomaticWorkRefused("output_tokens_budget")
            self._check_admission(
                conn, chain_id, kind="model_call", amount=1, limits=limits
            )
            self._check_admission(
                conn, chain_id, kind="tokens", amount=amount, limits=limits
            )
            now = self._wall_clock()
            conn.executemany(
                "INSERT INTO automatic_work_reservations "
                "(id, chain_id, owner_id, kind, amount, state, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'committed', ?, ?)",
                [
                    (model_id, chain_id, owner_id, "model_call", 1, now, now),
                    (token_id, chain_id, owner_id, "tokens", amount, now, now),
                ],
            )
        return token_id

    def commit(
        self,
        reservation_id: str,
        *,
        owner_id: str,
        limits: AutomaticWorkLimits | None = None,
    ) -> bool:
        """Consume a reservation once; True alone authorizes its first dispatch."""
        with self._db.connection() as conn:
            chain_id = self._reservation(conn, reservation_id, owner_id)["chain_id"]
        with self._admission_transaction(chain_id) as conn:
            self._check_runtime_owner(conn, owner_id)
            row = self._reservation(conn, reservation_id, owner_id)
            if row["state"] != "reserved":
                return False
            self._check_admission(conn, chain_id, kind=row["kind"], limits=limits)
            now = self._wall_clock()
            conn.execute(
                "UPDATE automatic_work_reservations SET state='committed', updated_at=? WHERE id=?",
                (now, reservation_id),
            )
            if row["kind"] == "generation":
                self._start_automatic(conn, chain_id)
        return True

    def release(self, reservation_id: str, *, owner_id: str) -> bool:
        """Refund only the owner's proven uncommitted operation."""
        with self.transaction() as conn:
            row = self._reservation(conn, reservation_id, owner_id)
            if row["state"] != "reserved":
                return False
            conn.execute(
                "UPDATE automatic_work_reservations SET state='released', updated_at=? WHERE id=?",
                (self._wall_clock(), reservation_id),
            )
        return True

    def settle(
        self, reservation_id: str, *, owner_id: str, actual_amount: int | None
    ) -> bool:
        """Replace a known token estimate, retaining uncertainty on missing usage."""
        if actual_amount is not None and (
            type(actual_amount) is not int or not 0 <= actual_amount < 2**63
        ):
            raise ValueError("actual amount must be a nonnegative SQLite integer")
        with self.transaction() as conn:
            row = self._reservation(conn, reservation_id, owner_id)
            if row["kind"] != "tokens" or row["state"] in {"reserved", "released"}:
                raise ValueError("only dispatched token reservations can settle")
            if row["state"] == "settled":
                if row["actual_amount"] != actual_amount:
                    raise ValueError("settlement conflict")
                return False
            if row["state"] == "uncertain" and actual_amount is None:
                return False
            conn.execute(
                "UPDATE automatic_work_reservations SET state=?, actual_amount=?, updated_at=? WHERE id=?",
                (
                    "uncertain" if actual_amount is None else "settled",
                    actual_amount,
                    self._wall_clock(),
                    reservation_id,
                ),
            )
            root_id = self._allowance_chain(conn, row["chain_id"])["id"]
            if actual_amount is None:
                conn.execute(
                    "UPDATE automatic_work_chains SET status='review_required', pause_reason='usage_unknown' WHERE id=?",
                    (root_id,),
                )
            else:
                snapshot = self._snapshot(conn, row["chain_id"])
                if (
                    snapshot.used["tokens"] + snapshot.reserved["tokens"]
                    > snapshot.limits.budget_tokens
                ):
                    conn.execute(
                        "UPDATE automatic_work_chains SET status='paused', pause_reason='tokens_budget' "
                        "WHERE id=? AND status!='review_required' "
                        "AND (pause_reason IS NULL OR pause_reason!='wall_budget')",
                        (root_id,),
                    )
        return True

    def attach_run(self, run_id: str, chain_id: str) -> None:
        """Attach a same-conversation run once; never reparent old survivors."""
        with self.transaction() as conn:
            run = conn.execute(
                "SELECT child.conversation_id, child.work_chain_id, parent.conversation_id AS parent_conversation, "
                "parent.work_chain_id AS parent_chain FROM agent_runs child "
                "LEFT JOIN agent_runs parent ON parent.id=child.parent_run_id WHERE child.id=?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise ValueError("unknown run")
            chain = self._chain(conn, chain_id)
            if run["conversation_id"] != chain["conversation_id"]:
                raise ValueError("run chain scope mismatch")
            if (
                run["parent_conversation"] is not None
                and run["parent_conversation"] != run["conversation_id"]
            ) or (run["parent_chain"] is not None and run["parent_chain"] != chain_id):
                raise ValueError("run chain conflicts with parent")
            if run["work_chain_id"] is not None and run["work_chain_id"] != chain_id:
                raise ValueError("run chain is immutable")
            conn.execute(
                "UPDATE agent_runs SET work_chain_id=? WHERE id=?", (chain_id, run_id)
            )

    @staticmethod
    def _target_active(conn: sqlite3.Connection, conversation_id: str) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM automatic_wake_attempts WHERE conversation_id=? "
                "AND state IN ('prepared', 'accepted') UNION ALL "
                "SELECT 1 FROM automatic_chat_start_attempts WHERE conversation_id=? "
                "AND state IN ('prepared', 'accepted') LIMIT 1",
                (conversation_id, conversation_id),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _chat_start_attempt(
        conn: sqlite3.Connection, attempt_id: str, owner_id: str
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM automatic_chat_start_attempts WHERE id=?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown chat start attempt")
        if row["owner_id"] != owner_id:
            raise ValueError("attempt owner mismatch")
        return row

    @staticmethod
    def _chat_start_view(row: sqlite3.Row) -> AutomaticChatStartAttempt:
        return AutomaticChatStartAttempt(
            **{key: row[key] for key in AutomaticChatStartAttempt.__dataclass_fields__}
        )

    def read_chat_start_attempt(
        self, attempt_id: str, *, owner_id: str
    ) -> AutomaticChatStartAttempt:
        """Return exact persisted native authority for its immutable owner."""
        with self._db.connection() as conn:
            return self._chat_start_view(
                self._chat_start_attempt(conn, attempt_id, owner_id)
            )

    def prepare_chat_start(
        self,
        *,
        attempt_id: str,
        source_run_id: str,
        target_conversation_id: str,
        target_session_id: str,
        target_session_incarnation: str,
        owner_id: str,
        draft_revision: int,
        context_epoch: int,
        request_fingerprint: str,
        limits: AutomaticWorkLimits | None = None,
    ) -> AutomaticChatStartAttempt:
        """Atomically reserve one shared generation for an exact local target.

        The controller proves live source-session ownership before calling this
        trusted interface. Persisted run lineage and runtime ownership are checked
        here; target runs continue to have only same-conversation parents.
        """
        for value in (
            attempt_id,
            source_run_id,
            target_conversation_id,
            target_session_id,
            target_session_incarnation,
            owner_id,
        ):
            _identity(value)
        submission_id = _identity(f"chat-start:{attempt_id}")
        for value in (draft_revision, context_epoch):
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError(
                    "revision and epoch must be nonnegative SQLite integers"
                )
        if (
            not isinstance(request_fingerprint, str)
            or len(request_fingerprint) != 64
            or any(
                character not in "0123456789abcdef" for character in request_fingerprint
            )
        ):
            raise ValueError("request fingerprint must be a SHA-256 digest")
        with self._db.connection() as conn:
            source = conn.execute(
                "SELECT work_chain_id FROM agent_runs WHERE id=?", (source_run_id,)
            ).fetchone()
            if source is None or source["work_chain_id"] is None:
                raise AutomaticWorkRefused("source_chain_missing")
            source_chain_id = source["work_chain_id"]
        with self._admission_transaction(source_chain_id) as conn:
            self._check_runtime_owner(conn, owner_id)
            existing = conn.execute(
                "SELECT * FROM automatic_chat_start_attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            identity = (
                source_run_id,
                source_chain_id,
                target_conversation_id,
                target_session_id,
                target_session_incarnation,
                owner_id,
                draft_revision,
                context_epoch,
                request_fingerprint,
            )
            if existing is not None:
                if (
                    tuple(
                        existing[key]
                        for key in (
                            "source_run_id",
                            "source_chain_id",
                            "conversation_id",
                            "session_id",
                            "session_incarnation",
                            "owner_id",
                            "draft_revision",
                            "context_epoch",
                            "request_fingerprint",
                        )
                    )
                    != identity
                ):
                    raise ValueError("attempt identity conflict")
                return self._chat_start_view(existing)
            source = conn.execute(
                "SELECT status FROM agent_runs WHERE id=?", (source_run_id,)
            ).fetchone()
            if source["status"] in TERMINAL_RUN_STATUSES:
                raise AutomaticWorkRefused("source_run_terminal")
            self._check_admission(
                conn, source_chain_id, kind="generation", amount=1, limits=limits
            )
            if self._target_active(conn, target_conversation_id):
                raise AutomaticWorkRefused("conversation_wake_active")
            root = self._allowance_chain(conn, source_chain_id)
            chain_id, reservation_id = uuid4().hex, uuid4().hex
            now = self._wall_clock()
            conn.execute(
                "INSERT INTO automatic_work_chains (id, conversation_id, root_submission_id, limits_json, "
                "created_at, last_observed_at, allowance_root_chain_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    chain_id,
                    target_conversation_id,
                    submission_id,
                    root["limits_json"],
                    now,
                    now,
                    root["id"],
                ),
            )
            conn.execute(
                "INSERT INTO automatic_work_reservations (id, chain_id, owner_id, kind, amount, state, created_at, updated_at) "
                "VALUES (?, ?, ?, 'generation', 1, 'reserved', ?, ?)",
                (reservation_id, chain_id, owner_id, now, now),
            )
            conn.execute(
                "INSERT INTO automatic_chat_start_attempts (id, source_run_id, source_chain_id, chain_id, conversation_id, "
                "session_id, session_incarnation, owner_id, draft_revision, context_epoch, request_fingerprint, "
                "generation_reservation_id, state, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?)",
                (
                    attempt_id,
                    source_run_id,
                    source_chain_id,
                    chain_id,
                    target_conversation_id,
                    target_session_id,
                    target_session_incarnation,
                    owner_id,
                    draft_revision,
                    context_epoch,
                    request_fingerprint,
                    reservation_id,
                    now,
                ),
            )
            return self._chat_start_view(
                self._chat_start_attempt(conn, attempt_id, owner_id)
            )

    def accept_chat_start(
        self,
        attempt_id: str,
        *,
        owner_id: str,
        limits: AutomaticWorkLimits | None = None,
    ) -> bool:
        """Commit the ownership cutoff once; conversation acceptance follows."""
        with self._db.connection() as conn:
            chain_id = self._chat_start_attempt(conn, attempt_id, owner_id)["chain_id"]
        with self._admission_transaction(chain_id) as conn:
            self._check_runtime_owner(conn, owner_id)
            attempt = self._chat_start_attempt(conn, attempt_id, owner_id)
            if attempt["state"] != "prepared":
                return False
            self._check_admission(conn, chain_id, kind="generation", limits=limits)
            reservation = self._reservation(
                conn, attempt["generation_reservation_id"], owner_id
            )
            if reservation["state"] != "reserved":
                raise ValueError("attempt generation is not reserved")
            now = self._wall_clock()
            conn.execute(
                "UPDATE automatic_work_reservations SET state='committed', updated_at=? WHERE id=?",
                (now, reservation["id"]),
            )
            conn.execute(
                "UPDATE automatic_chat_start_attempts SET state='accepted', accepted_at=? WHERE id=?",
                (now, attempt_id),
            )
            self._start_automatic(conn, chain_id)
        return True

    def abort_chat_start(self, attempt_id: str, *, owner_id: str) -> bool:
        """Refund only this owner's still-prepared, uncommitted attempt."""
        with self.transaction() as conn:
            attempt = self._chat_start_attempt(conn, attempt_id, owner_id)
            if attempt["state"] != "prepared":
                return False
            reservation = self._reservation(
                conn, attempt["generation_reservation_id"], owner_id
            )
            if reservation["state"] != "reserved":
                return False
            self._check_runtime_owner(conn, owner_id)
            now = self._wall_clock()
            conn.execute(
                "UPDATE automatic_work_reservations SET state='released', updated_at=? WHERE id=?",
                (now, reservation["id"]),
            )
            conn.execute(
                "UPDATE automatic_chat_start_attempts SET state='aborted', completed_at=? WHERE id=?",
                (now, attempt_id),
            )
        return True

    def mark_chat_start_review_required(
        self, attempt_id: str, *, owner_id: str
    ) -> bool:
        """Atomically settle same-owner accepted uncertainty and pause its root.

        Preserve every charge and clock anchor. A stale owner cannot change its
        replacement, and nonaccepted terminal rows cannot be rewritten.

        Args:
            attempt_id: Exact durable native-start attempt identity.
            owner_id: Runtime owner that prepared and accepted the attempt.

        Returns:
            True after accepted or already-reviewed work is durably paused;
            False for prepared, aborted, or completed work.

        Raises:
            AutomaticWorkRefused: The durable runtime owner was replaced.
            ValueError: The attempt is unknown or belongs to another owner.
            sqlite3.Error: Durable settlement could not be confirmed.
        """
        with self.transaction() as conn:
            self._check_runtime_owner(conn, owner_id)
            attempt = self._chat_start_attempt(conn, attempt_id, owner_id)
            if attempt["state"] not in {"accepted", "review_required"}:
                return False
            root = self._allowance_chain(conn, attempt["chain_id"])["id"]
            conn.execute(
                "UPDATE automatic_chat_start_attempts SET state='review_required', completed_at=COALESCE(completed_at, ?) WHERE id=?",
                (self._wall_clock(), attempt_id),
            )
            conn.execute(
                "UPDATE automatic_work_chains SET status='review_required', pause_reason='interrupted_work' WHERE id=?",
                (root,),
            )
        self._clear_chat_start_restriction(attempt_id, owner_id=owner_id)
        return True

    def complete_chat_start(self, attempt_id: str, *, owner_id: str) -> bool:
        """Settle accepted native work once, without survivor delivery marks."""
        with self.transaction() as conn:
            attempt = self._chat_start_attempt(conn, attempt_id, owner_id)
            if attempt["state"] != "accepted":
                return False
            self._check_runtime_owner(conn, owner_id)
            conn.execute(
                "UPDATE automatic_chat_start_attempts SET state='completed', completed_at=? WHERE id=?",
                (self._wall_clock(), attempt_id),
            )
        return True

    @staticmethod
    def _attempt(
        conn: sqlite3.Connection, attempt_id: str, owner_id: str
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM automatic_wake_attempts WHERE id=?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown wake attempt")
        if row["owner_id"] != owner_id:
            raise ValueError("attempt owner mismatch")
        return row

    @staticmethod
    def _attempt_view(row: sqlite3.Row) -> AutomaticWakeAttempt:
        return AutomaticWakeAttempt(
            row["id"],
            row["chain_id"],
            row["conversation_id"],
            row["session_id"],
            row["owner_id"],
            row["state"],
            tuple(json.loads(row["run_ids_json"])),
        )

    def read_attempt(self, attempt_id: str, *, owner_id: str) -> AutomaticWakeAttempt:
        """Read a typed attempt only for its immutable runtime owner."""
        with self._db.connection() as conn:
            return self._attempt_view(self._attempt(conn, attempt_id, owner_id))

    def claim_wake(
        self,
        chain_id: str,
        *,
        attempt_id: str,
        owner_id: str,
        session_id: str,
        run_ids: Sequence[str],
        limits: AutomaticWorkLimits | None = None,
    ) -> AutomaticWakeAttempt:
        """Claim one immutable survivor batch and its generation in one commit."""
        for value in (attempt_id, owner_id, session_id):
            _identity(value)
        if isinstance(run_ids, str) or not 1 <= len(run_ids) <= 256:
            raise ValueError("wake batch must contain 1 to 256 run IDs")
        selected = tuple(sorted(_identity(run_id) for run_id in run_ids))
        if len(set(selected)) != len(selected):
            raise ValueError("wake batch contains duplicate run IDs")
        with self._admission_transaction(chain_id) as conn:
            self._check_runtime_owner(conn, owner_id)
            existing = conn.execute(
                "SELECT * FROM automatic_wake_attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if existing:
                if (
                    existing["chain_id"],
                    existing["owner_id"],
                    existing["session_id"],
                    tuple(json.loads(existing["run_ids_json"])),
                ) != (chain_id, owner_id, session_id, selected):
                    raise ValueError("attempt identity conflict")
                return self._attempt_view(existing)
            snapshot = self._snapshot(conn, chain_id)
            self._check_admission(
                conn, chain_id, kind="generation", amount=1, limits=limits
            )
            if self._target_active(conn, snapshot.conversation_id):
                raise AutomaticWorkRefused("conversation_wake_active")
            for run_id in selected:
                row = conn.execute(
                    "SELECT child.*, parent.status AS parent_status, parent.updated_at AS parent_updated_at "
                    "FROM agent_runs child LEFT JOIN agent_runs parent ON parent.id=child.parent_run_id WHERE child.id=?",
                    (run_id,),
                ).fetchone()
                if (
                    row is None
                    or row["work_chain_id"] != chain_id
                    or row["conversation_id"] != snapshot.conversation_id
                ):
                    raise ValueError("wake result scope mismatch")
                if (
                    row["agent_kind"] == "primary"
                    or row["status"] not in {"done", "error", "cancelled"}
                    or row["parent_status"] not in TERMINAL_RUN_STATUSES
                    or row["updated_at"] < row["parent_updated_at"]
                ):
                    raise ValueError("wake result must be a terminal survivor")
                if (
                    row["wake_delivered_at"] is not None
                    or conn.execute(
                        "SELECT 1 FROM automatic_wake_claims WHERE run_id=?", (run_id,)
                    ).fetchone()
                ):
                    raise AutomaticWorkRefused("result_already_claimed")
            now = self._wall_clock()
            reservation_id = uuid4().hex
            conn.execute(
                "INSERT INTO automatic_work_reservations (id, chain_id, owner_id, kind, amount, state, created_at, updated_at) VALUES (?, ?, ?, 'generation', 1, 'reserved', ?, ?)",
                (reservation_id, chain_id, owner_id, now, now),
            )
            conn.execute(
                "INSERT INTO automatic_wake_attempts (id, chain_id, conversation_id, session_id, owner_id, generation_reservation_id, run_ids_json, state, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'prepared', ?)",
                (
                    attempt_id,
                    chain_id,
                    snapshot.conversation_id,
                    session_id,
                    owner_id,
                    reservation_id,
                    json.dumps(selected),
                    now,
                ),
            )
            conn.executemany(
                "INSERT INTO automatic_wake_claims (run_id, attempt_id) VALUES (?, ?)",
                [(run_id, attempt_id) for run_id in selected],
            )
            return self._attempt_view(self._attempt(conn, attempt_id, owner_id))

    def accept_wake(
        self,
        attempt_id: str,
        *,
        owner_id: str,
        limits: AutomaticWorkLimits | None = None,
    ) -> bool:
        """Durably consume a prepared attempt; only the first True may dispatch."""
        with self._db.connection() as conn:
            chain_id = self._attempt(conn, attempt_id, owner_id)["chain_id"]
        with self._admission_transaction(chain_id) as conn:
            self._check_runtime_owner(conn, owner_id)
            attempt = self._attempt(conn, attempt_id, owner_id)
            if attempt["state"] != "prepared":
                return False
            self._check_admission(conn, chain_id, kind="generation", limits=limits)
            reservation = self._reservation(
                conn, attempt["generation_reservation_id"], owner_id
            )
            if reservation["state"] != "reserved":
                raise ValueError("attempt generation is not reserved")
            now = self._wall_clock()
            conn.execute(
                "UPDATE automatic_work_reservations SET state='committed', updated_at=? WHERE id=?",
                (now, reservation["id"]),
            )
            conn.execute(
                "UPDATE automatic_wake_attempts SET state='accepted', accepted_at=? WHERE id=?",
                (now, attempt_id),
            )
            self._start_automatic(conn, chain_id)
        return True

    def abort_wake(self, attempt_id: str, *, owner_id: str) -> bool:
        """Release a proven pre-acceptance refusal; accepted work never refunds."""
        with self.transaction() as conn:
            attempt = self._attempt(conn, attempt_id, owner_id)
            if attempt["state"] != "prepared":
                return False
            conn.execute(
                "UPDATE automatic_work_reservations SET state='released', updated_at=? WHERE id=? AND state='reserved'",
                (self._wall_clock(), attempt["generation_reservation_id"]),
            )
            conn.execute(
                "DELETE FROM automatic_wake_claims WHERE attempt_id=?", (attempt_id,)
            )
            conn.execute(
                "UPDATE automatic_wake_attempts SET state='aborted', completed_at=? WHERE id=?",
                (self._wall_clock(), attempt_id),
            )
        return True

    def complete_wake(self, attempt_id: str, *, owner_id: str) -> bool:
        """Atomically stamp the exact accepted batch without re-executing it."""
        from .AgentRuns_DB import _now_iso

        with self.transaction() as conn:
            attempt = self._attempt(conn, attempt_id, owner_id)
            if attempt["state"] != "accepted":
                return False
            conn.execute(
                "UPDATE agent_runs SET wake_delivered_at=? WHERE id IN (SELECT run_id FROM automatic_wake_claims WHERE attempt_id=?) AND wake_delivered_at IS NULL",
                (_now_iso(), attempt_id),
            )
            conn.execute(
                "UPDATE automatic_wake_attempts SET state='completed', completed_at=? WHERE id=?",
                (self._wall_clock(), attempt_id),
            )
        return True

    def recover(self, *, current_owner_id: str) -> int:
        """Explicit startup recovery; handle/view reopen must never call this.

        Foreign unfinished attempts and unresolved reservations retain charges.
        Known consumed counts remain known. Late usage may improve accounting,
        but cannot grant the old owner new execution authority.
        """
        _identity(current_owner_id)
        database = self._restriction_database()
        with self.transaction() as conn:
            # Revoke even completed owners: their surviving child callbacks
            # may otherwise look fully accounted and escape the unfinished scan.
            conn.execute(
                "INSERT INTO automatic_work_runtime_owner (singleton, owner_id) VALUES (1, ?) "
                "ON CONFLICT(singleton) DO UPDATE SET owner_id=excluded.owner_id",
                (current_owner_id,),
            )
            chains = {
                row[0]
                for row in conn.execute(
                    "SELECT chain_id FROM automatic_wake_attempts WHERE owner_id!=? AND state IN ('prepared', 'accepted') "
                    "UNION SELECT chain_id FROM automatic_chat_start_attempts WHERE owner_id!=? AND state IN ('prepared', 'accepted') "
                    "UNION SELECT chain_id FROM automatic_work_reservations WHERE owner_id!=? AND (state='reserved' OR (kind='tokens' AND state='committed'))",
                    (current_owner_id, current_owner_id, current_owner_id),
                )
            }
            chains = {
                self._allowance_chain(conn, chain_id)["id"] for chain_id in chains
            }
            conn.execute(
                "UPDATE automatic_chat_start_attempts SET state='review_required' WHERE owner_id!=? AND state IN ('prepared', 'accepted')",
                (current_owner_id,),
            )
            conn.execute(
                "UPDATE automatic_wake_attempts SET state='review_required' WHERE owner_id!=? AND state IN ('prepared', 'accepted')",
                (current_owner_id,),
            )
            conn.execute(
                "UPDATE automatic_work_reservations SET state='uncertain', updated_at=? WHERE owner_id!=? AND (state='reserved' OR (kind='tokens' AND state='committed'))",
                (self._wall_clock(), current_owner_id),
            )
            conn.executemany(
                "UPDATE automatic_work_chains SET status='review_required', pause_reason='interrupted_work' WHERE id=?",
                [(chain_id,) for chain_id in chains],
            )
            # Snapshot only the entries covered while this transaction owns
            # recovery. Later recoveries/setters may run before our return.
            with _UNCONFIRMED_STARTS_LOCK:
                recovered_restrictions = tuple(
                    (key, entry)
                    for key, entry in _UNCONFIRMED_STARTS.items()
                    if key[0] == database and key[1] != current_owner_id
                )
        with _UNCONFIRMED_STARTS_LOCK:
            for key, entry in recovered_restrictions:
                # A fresh write to even the same key is new uncertainty.
                if _UNCONFIRMED_STARTS.get(key) is entry:
                    _UNCONFIRMED_STARTS.pop(key)
        return len(chains)
