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
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from tldw_chatbook.Agents.agent_models import TERMINAL_RUN_STATUSES
from tldw_chatbook.Agents.automatic_work_budget import (
    AutomaticChatStartAttempt,
    AutomaticWakeAttempt,
    AutomaticWorkLimits,
    AutomaticWorkRefused,
    AutomaticWorkReservation,
    AutomaticWorkSnapshot,
    GoalAttempt,
    RuntimeRecoveryResult,
)
from tldw_chatbook.Agents.code_review_models import ReviewBoundaryReceipt

if TYPE_CHECKING:
    from tldw_chatbook.Agents.workflow_models import (
        WorkflowPhaseAttempt,
        WorkflowPlan,
        WorkflowTaskAttempt,
    )
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
        self._progress_claim_lock = threading.RLock()
        self._temporary_progress_attempts: dict[str, tuple[tuple[str, str], ...]] = {}
        self._temporary_progress_claims: dict[str, str] = {}
        self._review_authorities = {}
        self._review_attempts = {}

        # Establish the existing canonical key before this ledger can authorize
        # work. Cleanup must never need a fresh filesystem observation.
        if not db.is_memory_db:
            self._restriction_database_id = str(db.db_path.resolve())
        else:
            # Independent memory stores retain private UUIDs; ledgers for the
            # same store share the identity even if the property is rebuilt.
            with _UNCONFIRMED_STARTS_LOCK:
                identity = getattr(db, "_automatic_work_restriction_id", None)
                if identity is None:
                    identity = "memory:" + uuid4().hex
                    db._automatic_work_restriction_id = identity
                self._restriction_database_id = identity

    def register_review_boundary_authority(
        self, owner_id, *, validate, transition, acceptance_scope
    ):
        """Install the live native issuer, never a serialized authority object."""
        _identity(owner_id)
        with self._progress_claim_lock:
            previous = self._review_authorities.get(owner_id)
            if previous is not None and previous != (
                validate,
                transition,
                acceptance_scope,
            ):
                raise ValueError("review boundary issuer conflict")
            self._review_authorities[owner_id] = (
                validate,
                transition,
                acceptance_scope,
            )

    def unregister_review_boundary_authority(self, owner_id):
        """Release the in-memory owner only; durable claims and charges remain."""
        with self._progress_claim_lock:
            self._review_authorities.pop(owner_id, None)
            self._review_attempts = {
                key: value
                for key, value in self._review_attempts.items()
                if value.owner_id != owner_id
            }

    def _validate_review_boundary(
        self, receipt, *, chain_id, owner_id, session_id, conversation_id
    ):
        authority = self._review_authorities.get(owner_id)
        if (
            type(receipt) is not ReviewBoundaryReceipt
            or (
                receipt.chain_id,
                receipt.owner_id,
                receipt.session_id,
                receipt.conversation_id,
            )
            != (chain_id, owner_id, session_id, conversation_id)
            or authority is None
            or not authority[0](receipt)
        ):
            raise AutomaticWorkRefused("review_boundary_unavailable")

    def _transition_review_boundary(self, attempt_id, owner_id, state):
        receipt = self._review_attempts.get(attempt_id)
        authority = self._review_authorities.get(owner_id)
        if receipt is not None and authority is not None:
            authority[1](receipt, state)
        if state in {"released", "completed"}:
            with self._progress_claim_lock:
                self._review_attempts.pop(attempt_id, None)

    def _restriction_database(self) -> str:
        return self._restriction_database_id

    def _restrict_chat_start(
        self, attempt_id: str, *, owner_id: str, chain_id: str | None
    ) -> None:
        """Retain uncertain authority before settlement I/O, without reading DB."""
        self._retain_chat_start_restriction(
            attempt_id, owner_id=owner_id, chain_id=chain_id
        )

    def _retain_chat_start_restriction(
        self, attempt_id: str, *, owner_id: str, chain_id: str | None
    ) -> None:
        """Insert denial using only captured identity and the existing registry."""
        with _UNCONFIRMED_STARTS_LOCK:
            _UNCONFIRMED_STARTS[
                (self._restriction_database_id, owner_id, attempt_id)
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
    def transaction(self, *, _native=None) -> Iterator[sqlite3.Connection]:
        """Commit an execution fence durably; restore policy on every exit.

        Nested transactions are rejected before changing the connection. Catch
        BaseException too: cancellation must never leave a fence half-open.
        """
        if _native is not None:
            from tldw_chatbook.Chat.console_workflow_runs import _fence_connection

        with (
            self._db.connection()
            if _native is None
            else nullcontext(_fence_connection(_native, self))
        ) as conn:
            if conn.in_transaction:
                raise RuntimeError("automatic work transaction cannot be nested")
            previous = int(conn.execute("PRAGMA synchronous").fetchone()[0])
            conn.execute("PRAGMA synchronous=FULL")
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    yield conn
                    if _native is not None:
                        _fence_connection(_native, self)
                    conn.commit()
                except BaseException:
                    conn.rollback()
                    raise
            finally:
                # The value comes only from SQLite's bounded pragma enum.
                conn.execute(f"PRAGMA synchronous={previous}")

    @contextmanager
    def _admission_transaction(
        self, chain_id: str, *, _native=None
    ) -> Iterator[sqlite3.Connection]:
        """Roll back refused work, retaining clock observations and its pause."""
        refusal = None
        with self.transaction(_native=_native) as conn:
            conn.execute("SAVEPOINT automatic_admission")
            try:
                yield conn
            except AutomaticWorkRefused as exc:
                observed = self._allowance_chain(conn, chain_id)
                root_id = observed["id"]
                intervals = tuple(
                    dict(row)
                    for row in conn.execute(
                        "SELECT i.* FROM workflow_active_intervals i JOIN workflow_runs w ON w.id=i.workflow_id WHERE w.chain_id=?",
                        (root_id,),
                    )
                )
                conn.execute("ROLLBACK TO automatic_admission")
                for interval in intervals:
                    conn.execute(
                        "UPDATE workflow_active_intervals SET charged_seconds=?,observed_monotonic=? WHERE id=?",
                        (
                            interval["charged_seconds"],
                            interval["observed_monotonic"],
                            interval["id"],
                        ),
                    )
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
                if reason not in {
                    "runtime_owner_replaced",
                    "settlement_unconfirmed",
                    "review_boundary_already_claimed",
                    "review_boundary_unavailable",
                    "workflow_authority_changed",
                }:
                    conn.execute(
                        "UPDATE automatic_work_chains SET status=?, pause_reason=? "
                        "WHERE id=? AND status!='review_required'",
                        (status, reason, root_id),
                    )
            finally:
                conn.execute("RELEASE automatic_admission")
        if refusal is not None:
            raise refusal

    def policy_kind(self, chain_id: str) -> str:
        """Read immutable goal origin from the canonical allowance root."""
        with self._db.connection() as conn:
            root = self._allowance_chain(conn, chain_id)["id"]
            if conn.execute(
                "SELECT 1 FROM workflow_runs WHERE chain_id=?", (root,)
            ).fetchone():
                return "workflow_phase"
            return (
                "goal_iteration"
                if conn.execute(
                    "SELECT 1 FROM goal_runs WHERE chain_id=?", (root,)
                ).fetchone()
                else "fleet_wake"
            )

    def _check_admission(
        self,
        conn: sqlite3.Connection,
        chain_id: str,
        *,
        kind: str,
        amount: int = 0,
        limits: AutomaticWorkLimits | None = None,
        _dispatch=None,
        _native=None,
    ) -> None:
        if _native is not None:
            from tldw_chatbook.Chat.console_workflow_runs import _fence_connection

            _fence_connection(_native, self)
        self._check_unconfirmed_starts(conn, chain_id)
        snapshot = self._snapshot(conn, chain_id, _dispatch=_dispatch, _native=_native)
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
        workflow = conn.execute(
            "SELECT id FROM workflow_runs WHERE chain_id=?", (chain_id,)
        ).fetchone()
        if workflow is not None:
            self._workflow_enabled(_native=_native)
            for resource, ceiling in effective.resources().items():
                if ceiling == 0:
                    raise AutomaticWorkRefused(f"{resource}_budget")
            if effective.output_tokens == 0:
                raise AutomaticWorkRefused("output_tokens_budget")
            if conn.execute(
                "SELECT 1 FROM workflow_phase_attempts p JOIN workflow_runs w ON w.id=p.workflow_id JOIN automatic_wake_attempts a ON a.id=p.native_attempt_id WHERE w.id=? AND a.state='accepted' AND (p.control_revision!=w.control_revision OR w.status IN ('paused','stopped','closed','removed','completed','recovery_required'))",
                (workflow["id"],),
            ).fetchone():
                if _dispatch is None:
                    raise AutomaticWorkRefused("workflow_authority_changed")
                self._workflow_current(
                    conn,
                    _dispatch.phase.attempt.id,
                    _dispatch.phase.attempt.owner_id,
                    accepted=True,
                    _dispatch=_dispatch,
                )
            if (
                self._workflow_active_seconds(
                    conn,
                    workflow["id"],
                    observe=True,
                    _dispatch=_dispatch,
                    _native=_native,
                )
                >= effective.wall_seconds
            ):
                raise AutomaticWorkRefused("wall_budget")
        elif snapshot.started_at is not None:
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
        workflow = conn.execute(
            "SELECT id FROM workflow_runs WHERE chain_id=?", (chain_id,)
        ).fetchone()
        if workflow is not None:
            if not conn.execute(
                "SELECT 1 FROM workflow_active_intervals WHERE workflow_id=? AND ended_at IS NULL",
                (workflow["id"],),
            ).fetchone():
                owner = conn.execute(
                    "SELECT owner_id FROM automatic_work_runtime_owner WHERE singleton=1"
                ).fetchone()
                if owner is None:
                    raise AutomaticWorkRefused("runtime_owner_missing")
                now, mono = self._wall_clock(), self._monotonic_clock()
                conn.execute(
                    "INSERT INTO workflow_active_intervals (id,workflow_id,owner_id,started_at,clock_owner_id,started_monotonic,observed_monotonic) VALUES (?,?,?,?,?,?,?)",
                    (
                        uuid4().hex,
                        workflow["id"],
                        owner["owner_id"],
                        now,
                        _CLOCK_OWNER_ID,
                        mono,
                        mono,
                    ),
                )
            return
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
            return self._create_chain(
                conn,
                conversation_id,
                root_submission_id=root_submission_id,
                limits=limits,
            )

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
        self,
        conn: sqlite3.Connection,
        chain_id: str,
        *,
        _accounting=None,
        _dispatch=None,
        _native=None,
    ) -> AutomaticWorkSnapshot:
        member = self._chain(conn, chain_id)
        chain = self._allowance_chain(conn, chain_id)
        limits = AutomaticWorkLimits(**json.loads(chain["limits_json"]))
        used = dict.fromkeys(limits.resources(), 0)
        reserved = dict(used)
        uncertain = chain["status"] == "review_required"
        workflow = conn.execute(
            "SELECT id FROM workflow_runs WHERE chain_id=?", (chain["id"],)
        ).fetchone()
        if workflow is not None:
            uncertain |= self._workflow_clock_unknown(
                conn,
                workflow["id"],
                _accounting=_accounting,
                _dispatch=_dispatch,
                _native=_native,
            )
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
        _dispatch=None,
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
            self._check_admission(
                conn, chain_id, kind="model_call", limits=limits, _dispatch=_dispatch
            )
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
        _dispatch=None,
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
                conn,
                chain_id,
                kind="model_call",
                amount=1,
                limits=limits,
                _dispatch=_dispatch,
            )
            self._check_admission(
                conn,
                chain_id,
                kind="tokens",
                amount=amount,
                limits=limits,
                _dispatch=_dispatch,
            )
            self._resume_workflow_dispatch(
                conn, chain_id, owner_id, _dispatch=_dispatch
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
        _native=None,
    ) -> bool:
        """Consume a reservation once; True alone authorizes its first dispatch."""
        if _native is not None:
            from tldw_chatbook.Chat.console_workflow_runs import _fence_connection

        with (
            self._db.connection()
            if _native is None
            else nullcontext(_fence_connection(_native, self))
        ) as conn:
            chain_id = self._reservation(conn, reservation_id, owner_id)["chain_id"]
        with self._admission_transaction(chain_id, _native=_native) as conn:
            self._check_runtime_owner(conn, owner_id)
            row = self._reservation(conn, reservation_id, owner_id)
            if row["state"] != "reserved":
                return False
            self._check_admission(
                conn, chain_id, kind=row["kind"], limits=limits, _native=_native
            )
            now = self._wall_clock()
            conn.execute(
                "UPDATE automatic_work_reservations SET state='committed', updated_at=? WHERE id=?",
                (now, reservation_id),
            )
            if row["kind"] == "generation":
                self._start_automatic(conn, chain_id)
            else:
                self._resume_workflow_dispatch(
                    conn, chain_id, owner_id, _native=_native
                )
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

    def _confirm_chat_start_absent(self, attempt_id: str, *, owner_id: str) -> bool:
        """Confirm exact absence only after a successful current-owner transaction."""
        absent = False
        with self.transaction() as conn:
            owner = conn.execute(
                "SELECT owner_id FROM automatic_work_runtime_owner WHERE singleton=1"
            ).fetchone()
            if owner is not None and owner["owner_id"] == owner_id:
                absent = (
                    conn.execute(
                        "SELECT 1 FROM automatic_chat_start_attempts WHERE id=?",
                        (attempt_id,),
                    ).fetchone()
                    is None
                )
        # Commit and connection-policy restoration must both have succeeded.
        return absent

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
        conn: sqlite3.Connection,
        attempt_id: str,
        owner_id: str,
        kind: str = "fleet_wake",
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM automatic_wake_attempts WHERE id=?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown wake attempt")
        if row["owner_id"] != owner_id:
            raise ValueError("attempt owner mismatch")
        if row["attempt_kind"] != kind:
            raise ValueError("attempt kind mismatch")
        if kind == "goal_iteration":
            link = conn.execute(
                "SELECT g.chain_id, g.conversation_id FROM goal_iterations i JOIN goal_runs g ON g.id=i.goal_id WHERE i.attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if (
                link is None
                or link["chain_id"] != row["chain_id"]
                or link["conversation_id"] != row["conversation_id"]
                or row["session_id"] != row["conversation_id"]
                or json.loads(row["run_ids_json"]) != []
            ):
                raise ValueError("goal attempt link mismatch")

        if kind == "workflow_phase":
            link = conn.execute(
                "SELECT w.chain_id,w.conversation_id FROM workflow_phase_attempts p JOIN workflow_runs w ON w.id=p.workflow_id WHERE p.native_attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if (
                link is None
                or link["chain_id"] != row["chain_id"]
                or link["conversation_id"] != row["conversation_id"]
                or json.loads(row["run_ids_json"]) != []
                or json.loads(row["message_ids_json"]) != []
            ):
                raise ValueError("workflow attempt link mismatch")
        return row

    def _attempt_view(self, row: sqlite3.Row) -> AutomaticWakeAttempt:
        return AutomaticWakeAttempt(
            row["id"],
            row["chain_id"],
            row["conversation_id"],
            row["session_id"],
            row["owner_id"],
            row["state"],
            tuple(json.loads(row["run_ids_json"])),
            row["cause"],
            tuple(
                message_id
                for message_id, _ in self._temporary_progress_attempts[row["id"]]
            )
            if row["id"] in self._temporary_progress_attempts
            else tuple(json.loads(row["message_ids_json"])),
            row["review_boundary_id"],
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
        progress_messages: Sequence[tuple[str, str]] = (),
        persistent_progress: bool = True,
        review_boundary: ReviewBoundaryReceipt | None = None,
        limits: AutomaticWorkLimits | None = None,
    ) -> AutomaticWakeAttempt:
        """Claim one immutable survivor batch and its generation in one commit."""
        for value in (attempt_id, owner_id, session_id):
            _identity(value)
        if isinstance(run_ids, str) or not 0 <= len(run_ids) <= 256:
            raise ValueError("wake batch must contain at most 256 run IDs")
        if isinstance(progress_messages, str) or len(progress_messages) > 256:
            raise ValueError("progress batch must contain at most 256 messages")
        progress = tuple(
            sorted(
                (_identity(message_id), _identity(source_id))
                for message_id, source_id in progress_messages
            )
        )
        if len({message_id for message_id, _ in progress}) != len(progress):
            raise ValueError("wake batch contains duplicate message IDs")
        if not run_ids and not progress and review_boundary is None:
            raise ValueError("wake batch must not be empty")
        cause = (
            "mixed"
            if run_ids and progress
            else "progress"
            if progress
            else "completion"
        )
        selected = tuple(sorted(_identity(run_id) for run_id in run_ids))
        if len(set(selected)) != len(selected):
            raise ValueError("wake batch contains duplicate run IDs")
        with self._progress_claim_lock:
            with self._admission_transaction(chain_id) as conn:
                self._check_runtime_owner(conn, owner_id)
                existing = conn.execute(
                    "SELECT * FROM automatic_wake_attempts WHERE id=?", (attempt_id,)
                ).fetchone()
                if existing:
                    if existing["attempt_kind"] != "fleet_wake":
                        raise ValueError("attempt kind conflict")
                    if (
                        existing["chain_id"],
                        existing["owner_id"],
                        existing["session_id"],
                        tuple(json.loads(existing["run_ids_json"])),
                        existing["cause"],
                        existing["review_boundary_id"],
                        self._temporary_progress_attempts.get(attempt_id)
                        if attempt_id in self._temporary_progress_attempts
                        else tuple(
                            (row["message_id"], row["source_run_id"])
                            for row in conn.execute(
                                "SELECT message_id, source_run_id FROM automatic_progress_wake_claims WHERE attempt_id=? ORDER BY message_id",
                                (attempt_id,),
                            )
                        ),
                    ) != (
                        chain_id,
                        owner_id,
                        session_id,
                        selected,
                        cause,
                        getattr(review_boundary, "boundary_id", None),
                        progress,
                    ):
                        raise ValueError("attempt identity conflict")
                    if existing["state"] == "prepared" and review_boundary is not None:
                        self._validate_review_boundary(
                            review_boundary,
                            chain_id=chain_id,
                            owner_id=owner_id,
                            session_id=session_id,
                            conversation_id=existing["conversation_id"],
                        )
                    return self._attempt_view(existing)
                snapshot = self._snapshot(conn, chain_id)
                if review_boundary is not None:
                    self._validate_review_boundary(
                        review_boundary,
                        chain_id=chain_id,
                        owner_id=owner_id,
                        session_id=session_id,
                        conversation_id=snapshot.conversation_id,
                    )
                    if conn.execute(
                        "SELECT 1 FROM automatic_review_boundary_claims WHERE boundary_id=? OR (owner_id=? AND chain_id=? AND review_id=? AND slot_id=? AND slot_attempt_id=?)",
                        (
                            review_boundary.boundary_id,
                            owner_id,
                            chain_id,
                            review_boundary.review_id,
                            review_boundary.slot_id,
                            review_boundary.slot_attempt_id,
                        ),
                    ).fetchone():
                        raise AutomaticWorkRefused("review_boundary_already_claimed")
                    if not review_boundary.predecessor_run_ids or len(
                        review_boundary.predecessor_run_ids
                    ) != len(review_boundary.predecessor_settlement_ids):
                        raise ValueError("review_boundary predecessors missing")
                    for run_id, settlement in zip(
                        review_boundary.predecessor_run_ids,
                        review_boundary.predecessor_settlement_ids,
                    ):
                        row = conn.execute(
                            "SELECT status, conversation_id, work_chain_id, agent_kind FROM agent_runs WHERE id=?",
                            (run_id,),
                        ).fetchone()
                        if (
                            row is None
                            or row["status"] not in TERMINAL_RUN_STATUSES
                            or row["agent_kind"] != "subagent"
                            or row["conversation_id"] != snapshot.conversation_id
                            or row["work_chain_id"] != chain_id
                        ):
                            raise ValueError(
                                "review_boundary predecessor scope mismatch"
                            )
                        # The id names its step index (= seq): decode that one
                        # row, not the whole run, inside the write transaction.
                        # CAST keeps a malformed suffix a miss, never an error.
                        item = conn.execute(
                            "SELECT payload FROM agent_run_steps WHERE run_id=? AND seq=CAST(? AS INTEGER)",
                            (run_id, settlement.rpartition(":")[2]),
                        ).fetchone()
                        step = json.loads(item[0]) if item is not None else {}
                        if (
                            settlement != f"agent-step:{run_id}:{step.get('index')}"
                            or step.get("status") != row["status"]
                        ):
                            raise ValueError(
                                "review_boundary predecessor settlement missing"
                            )
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
                            "SELECT 1 FROM automatic_wake_claims WHERE run_id=?",
                            (run_id,),
                        ).fetchone()
                    ):
                        raise AutomaticWorkRefused("result_already_claimed")
                for message_id, source_id in progress:
                    source = conn.execute(
                        "SELECT agent_kind, conversation_id, work_chain_id, parent_run_id FROM agent_runs WHERE id=?",
                        (source_id,),
                    ).fetchone()
                    if (
                        source is None
                        or source["agent_kind"] == "primary"
                        or source["parent_run_id"] is None
                        or source["work_chain_id"] != chain_id
                        or source["conversation_id"] != snapshot.conversation_id
                    ):
                        raise ValueError("wake progress scope mismatch")
                    if (
                        message_id in self._temporary_progress_claims
                        or conn.execute(
                            "SELECT 1 FROM automatic_progress_wake_claims WHERE message_id=?",
                            (message_id,),
                        ).fetchone()
                    ):
                        raise AutomaticWorkRefused("progress_already_claimed")
                now = self._wall_clock()
                reservation_id = uuid4().hex
                conn.execute(
                    "INSERT INTO automatic_work_reservations (id, chain_id, owner_id, kind, amount, state, created_at, updated_at) VALUES (?, ?, ?, 'generation', 1, 'reserved', ?, ?)",
                    (reservation_id, chain_id, owner_id, now, now),
                )
                conn.execute(
                    "INSERT INTO automatic_wake_attempts (id, chain_id, conversation_id, session_id, owner_id, generation_reservation_id, run_ids_json, cause, message_ids_json, state, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?)",
                    (
                        attempt_id,
                        chain_id,
                        snapshot.conversation_id,
                        session_id,
                        owner_id,
                        reservation_id,
                        json.dumps(selected),
                        cause,
                        json.dumps(tuple(message_id for message_id, _ in progress))
                        if persistent_progress
                        else "[]",
                        now,
                    ),
                )
                conn.executemany(
                    "INSERT INTO automatic_wake_claims (run_id, attempt_id) VALUES (?, ?)",
                    [(run_id, attempt_id) for run_id in selected],
                )
                if review_boundary is not None:
                    receipt = review_boundary
                    conn.execute(
                        "UPDATE automatic_wake_attempts SET review_boundary_id=? WHERE id=?",
                        (receipt.boundary_id, attempt_id),
                    )
                    conn.execute(
                        "INSERT INTO automatic_review_boundary_claims (boundary_id, attempt_id, owner_id, chain_id, review_id, slot_id, slot_attempt_id, source_epoch, context_revision, grant_id, target_fingerprint, team_fingerprint, override_fingerprint, predecessor_run_ids_json, predecessor_settlement_ids_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            receipt.boundary_id,
                            attempt_id,
                            owner_id,
                            chain_id,
                            receipt.review_id,
                            receipt.slot_id,
                            receipt.slot_attempt_id,
                            receipt.source_epoch,
                            receipt.revision,
                            receipt.grant_id,
                            receipt.target_fingerprint,
                            receipt.team_fingerprint,
                            receipt.override_fingerprint,
                            json.dumps(receipt.predecessor_run_ids),
                            json.dumps(receipt.predecessor_settlement_ids),
                        ),
                    )
                if persistent_progress:
                    conn.executemany(
                        "INSERT INTO automatic_progress_wake_claims (message_id, source_run_id, attempt_id) VALUES (?, ?, ?)",
                        [
                            (message_id, source_id, attempt_id)
                            for message_id, source_id in progress
                        ],
                    )
            if review_boundary is not None:
                self._review_attempts[attempt_id] = review_boundary
            # Populate process-local authority only after the transaction commits.
            if progress and not persistent_progress:
                self._temporary_progress_attempts[attempt_id] = progress
                self._temporary_progress_claims.update(
                    (message_id, attempt_id) for message_id, _ in progress
                )
            return self.read_attempt(attempt_id, owner_id=owner_id)

    def progress_source_conversation(
        self, source_run_id: str, *, parent_run_id: str, chain_id: str
    ) -> str | None:
        """Read the immutable causal conversation for an exact child source.

        Args:
            source_run_id: Child named by the committed report metadata.
            parent_run_id: Immutable parent from that same report identity.
            chain_id: Immutable chain from that same report identity.

        Returns:
            Source conversation, or None if its exact lineage does not match.
        """
        for value in (source_run_id, parent_run_id, chain_id):
            _identity(value)
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT chain.conversation_id FROM agent_runs source "
                "JOIN automatic_work_chains chain ON chain.id=source.work_chain_id "
                "AND chain.conversation_id=source.conversation_id "
                "WHERE source.id=? AND source.parent_run_id=? "
                "AND source.work_chain_id=? AND source.agent_kind!='primary'",
                (source_run_id, parent_run_id, chain_id),
            ).fetchone()
            return row["conversation_id"] if row is not None else None

    def pending_progress_sources(
        self, conversation_id: str, progress_messages: Sequence[tuple[str, str]]
    ) -> tuple[tuple[str, str, str], ...]:
        """Filter source/claim metadata without reading or collecting any body."""
        values = []
        with self._progress_claim_lock, self._db.connection() as conn:
            for message_id, source_id in progress_messages:
                _identity(message_id)
                _identity(source_id)
                if message_id in self._temporary_progress_claims:
                    continue
                row = conn.execute(
                    "SELECT work_chain_id FROM agent_runs WHERE id=? AND conversation_id=? AND agent_kind!='primary' AND parent_run_id IS NOT NULL AND work_chain_id IS NOT NULL",
                    (source_id, conversation_id),
                ).fetchone()
                if (
                    row is not None
                    and not conn.execute(
                        "SELECT 1 FROM automatic_progress_wake_claims WHERE message_id=?",
                        (message_id,),
                    ).fetchone()
                ):
                    values.append((message_id, source_id, row["work_chain_id"]))
        return tuple(values)

    def accept_wake(
        self,
        attempt_id: str,
        *,
        owner_id: str,
        limits: AutomaticWorkLimits | None = None,
    ) -> bool:
        """Durably consume a prepared fleet attempt."""
        return self._accept_attempt(
            attempt_id, owner_id=owner_id, limits=limits, kind="fleet_wake"
        )

    def abort_wake(self, attempt_id: str, *, owner_id: str) -> bool:
        """Release a proven pre-acceptance refusal; accepted work never refunds."""
        return self._abort_attempt(attempt_id, owner_id=owner_id, kind="fleet_wake")

    def _abort_attempt(self, attempt_id: str, *, owner_id: str, kind: str) -> bool:
        """Release a proven pre-acceptance refusal; accepted work never refunds."""
        with self.transaction() as conn:
            attempt = self._attempt(conn, attempt_id, owner_id, kind)
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
                "DELETE FROM automatic_progress_wake_claims WHERE attempt_id=?",
                (attempt_id,),
            )
            conn.execute(
                "DELETE FROM automatic_review_boundary_claims WHERE attempt_id=?",
                (attempt_id,),
            )
            conn.execute(
                "UPDATE automatic_wake_attempts SET state='aborted', completed_at=? WHERE id=?",
                (self._wall_clock(), attempt_id),
            )
            if kind == "goal_iteration":
                conn.execute(
                    "DELETE FROM goal_payload_reservations WHERE attempt_id=?",
                    (attempt_id,),
                )
                conn.execute(
                    "UPDATE goal_iterations SET status='aborted', revision=revision+1 WHERE attempt_id=?",
                    (attempt_id,),
                )
        with self._progress_claim_lock:
            for message_id, _ in self._temporary_progress_attempts.get(attempt_id, ()):
                self._temporary_progress_claims.pop(message_id, None)
        self._transition_review_boundary(attempt_id, owner_id, "released")
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
        self._transition_review_boundary(attempt_id, owner_id, "completed")
        return True

    def _recover_clean_workflow_boundaries(self, conn, current_owner_id):
        """Close proven no-effect/quiescent old custody, retaining all consumed work."""
        candidates = conn.execute(
            "SELECT a.*,p.id AS phase_row,p.workflow_id,p.retirement FROM automatic_wake_attempts a JOIN workflow_phase_attempts p ON p.native_attempt_id=a.id WHERE a.owner_id!=? AND a.attempt_kind='workflow_phase' AND a.state IN ('prepared','accepted')",
            (current_owner_id,),
        ).fetchall()
        for phase in candidates:
            if conn.execute(
                "SELECT 1 FROM workflow_change_windows WHERE phase_attempt_id=? AND state IN ('active','unresolved') LIMIT 1",
                (phase["phase_row"],),
            ).fetchone():
                continue
            if phase["state"] == "accepted" and phase["retirement"] != "drained":
                continue
            if conn.execute(
                "SELECT 1 FROM workflow_active_intervals WHERE workflow_id=? AND ended_at IS NULL",
                (phase["workflow_id"],),
            ).fetchone():
                continue
            tasks = conn.execute(
                "SELECT t.*,r.attempt_id AS result_id,c.attempt_id AS checkpoint_id FROM workflow_task_attempts t JOIN workflow_task_claims claim ON claim.task_attempt_id=t.id LEFT JOIN workflow_results r ON r.attempt_id=t.id LEFT JOIN workflow_checkpoints c ON c.attempt_id=t.id WHERE claim.phase_attempt_id=?",
                (phase["phase_row"],),
            ).fetchall()
            if any(
                task["run_id"] is None
                and not self._db.workflow_runs._never_run_task(
                    conn, phase["workflow_id"], task["id"]
                )
                for task in tasks
            ):
                continue
            if any(
                task["run_id"] is not None
                and (
                    phase["state"] == "prepared"
                    or task["retirement"] != "drained"
                    or task["status"] not in {"completed", "failed"}
                    or task["result_id"] is None
                    or task["checkpoint_id"] is None
                )
                for task in tasks
            ):
                continue
            if conn.execute(
                "SELECT 1 FROM automatic_work_reservations r JOIN automatic_work_chains c ON c.id=r.chain_id WHERE (c.id=? OR c.allowance_root_chain_id=?) AND (r.state='uncertain' OR (r.kind='tokens' AND r.state='committed') OR (r.state='reserved' AND r.kind NOT IN ('generation','child_launch'))) LIMIT 1",
                (phase["chain_id"], phase["chain_id"]),
            ).fetchone():
                continue
            if (
                phase["state"] == "prepared"
                and conn.execute(
                    "SELECT 1 FROM automatic_work_reservations WHERE chain_id=? AND created_at>=? AND kind IN ('model_call','tokens') AND state NOT IN ('released','reserved') LIMIT 1",
                    (phase["chain_id"], phase["created_at"]),
                ).fetchone()
            ):
                continue
            # Validate this phase's exact records, not a second global history sweep.
            from tldw_chatbook.Agents.goal_iteration import digest
            from tldw_chatbook.Agents.workflow_models import WorkflowCheckpoint

            from .workflow_runs import WorkflowRunsStore

            try:
                for task in tasks:
                    if task["run_id"] is None:
                        continue
                    result = WorkflowRunsStore(self._db)._read_result(
                        conn, phase["workflow_id"], task["id"]
                    )
                    saved = conn.execute(
                        "SELECT payload_hash,payload_json FROM workflow_checkpoints WHERE attempt_id=? AND workflow_id=?",
                        (task["id"], phase["workflow_id"]),
                    ).fetchone()
                    checkpoint = WorkflowCheckpoint.model_validate_json(saved[1])
                    if (
                        digest(saved[1]) != saved[0]
                        or checkpoint.run_id != result.run_id
                        or checkpoint.result_digest != digest(result.canonical_json())
                    ):
                        raise ValueError("checkpoint_mismatch")
            except (ValueError, TypeError):
                continue
            conn.execute(
                "UPDATE automatic_wake_attempts SET state=?,completed_at=? WHERE id=?",
                (
                    "aborted" if phase["state"] == "prepared" else "completed",
                    self._wall_clock(),
                    phase["id"],
                ),
            )
            conn.execute(
                "UPDATE workflow_phase_attempts SET status='closed',retirement='drained',result_bytes=0 WHERE id=?",
                (phase["phase_row"],),
            )
            conn.execute(
                "UPDATE automatic_work_reservations SET state='released',updated_at=? WHERE state='reserved' AND (id=? OR id IN (SELECT c.child_reservation_id FROM workflow_task_claims c JOIN workflow_task_attempts t ON t.id=c.task_attempt_id WHERE c.phase_attempt_id=? AND t.run_id IS NULL))",
                (
                    self._wall_clock(),
                    phase["generation_reservation_id"],
                    phase["phase_row"],
                ),
            )
            conn.execute(
                "UPDATE workflow_task_attempts SET status='closed',retirement='drained' WHERE run_id IS NULL AND id IN (SELECT task_attempt_id FROM workflow_task_claims WHERE phase_attempt_id=?)",
                (phase["phase_row"],),
            )
            for task in tasks:
                WorkflowRunsStore._release_unused_result(
                    conn, phase["workflow_id"], task["id"]
                )
            conn.execute(
                "UPDATE workflow_runs SET status='awaiting_resume',control_revision=control_revision+1 WHERE id=? AND status NOT IN ('removed','closed','completed','stopped')",
                (phase["workflow_id"],),
            )

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
            self._recover_clean_workflow_boundaries(conn, current_owner_id)
            chains = {
                row[0]
                for row in conn.execute(
                    "SELECT chain_id FROM automatic_wake_attempts WHERE owner_id!=? AND state IN ('prepared', 'accepted') "
                    "UNION SELECT chain_id FROM automatic_chat_start_attempts WHERE owner_id!=? AND state IN ('prepared', 'accepted') "
                    "UNION SELECT chain_id FROM automatic_work_reservations WHERE owner_id!=? AND (state='reserved' OR (kind='tokens' AND state='committed'))",
                    (current_owner_id, current_owner_id, current_owner_id),
                )
            }
            # An open workflow interval is unsettled even if its logical run rows ended.
            intervals = conn.execute(
                "SELECT i.*,w.chain_id,c.limits_json FROM workflow_active_intervals i JOIN workflow_runs w ON w.id=i.workflow_id JOIN automatic_work_chains c ON c.id=w.chain_id WHERE i.ended_at IS NULL AND (i.owner_id!=? OR i.clock_owner_id IS NOT ?)",
                (current_owner_id, _CLOCK_OWNER_ID),
            ).fetchall()
            for interval in intervals:
                chains.add(interval["chain_id"])
                # Recovery cannot invent a historical drain timestamp.
                ceiling = AutomaticWorkLimits(
                    **json.loads(interval["limits_json"])
                ).wall_seconds
                conn.execute(
                    "UPDATE workflow_active_intervals SET charged_seconds=MAX(charged_seconds,?) WHERE id=?",
                    (ceiling, interval["id"]),
                )
            # Temporary progress IDs are deliberately process-local. After
            # restart a completed attempt's empty ID list cannot prove which
            # saved reports it already woke; retain a review fence on its chain.
            chains.update(
                row["chain_id"]
                for row in conn.execute(
                    "SELECT id, chain_id FROM automatic_wake_attempts "
                    "WHERE owner_id!=? AND state='completed' "
                    "AND cause IN ('progress', 'mixed') AND message_ids_json='[]'",
                    (current_owner_id,),
                )
                if row["id"] not in self._temporary_progress_attempts
            )
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
            conn.execute(
                "UPDATE workflow_phase_attempts SET status='uncertain',retirement='unknown' WHERE native_attempt_id IN (SELECT id FROM automatic_wake_attempts WHERE attempt_kind='workflow_phase' AND state='review_required')"
            )
            conn.executemany(
                "UPDATE workflow_runs SET status='recovery_required',control_revision=control_revision+1 WHERE chain_id=? AND status NOT IN ('removed','closed','completed')",
                [(chain_id,) for chain_id in chains],
            )
            goals = tuple(
                (
                    row["id"],
                    row["revision"],
                    "recovery_required"
                    if row["chain_id"] in chains or row["status"] == "recovery_required"
                    else "paused",
                )
                for row in conn.execute(
                    "SELECT g.* FROM goal_runs g WHERE g.status NOT IN ('completed','removed','closed','stopped','awaiting_result_review') "
                    "AND EXISTS (SELECT 1 FROM goal_iterations i JOIN automatic_wake_attempts a ON a.id=i.attempt_id WHERE i.goal_id=g.id AND a.state!='aborted')"
                )
            )
            workflows = tuple(
                (
                    row["id"],
                    row["control_revision"],
                    "recovery_required"
                    if row["chain_id"] in chains or row["status"] == "recovery_required"
                    else "awaiting_resume",
                )
                for row in conn.execute(
                    "SELECT id,control_revision,chain_id,status FROM workflow_runs WHERE status NOT IN ('removed','closed','completed','stopped')"
                )
            )
            audit = RuntimeRecoveryResult(
                current_owner_id, tuple(sorted(chains)), goals, workflows
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
        self.recovery_result = audit
        return len(chains)

    def _create_chain(
        self,
        conn: sqlite3.Connection,
        conversation_id: str,
        *,
        root_submission_id: str,
        limits: AutomaticWorkLimits,
    ) -> str:
        """Allocate inside the caller's transaction; never commit independently."""
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

    def read_goal_attempt(self, attempt_id: str, *, owner_id: str) -> GoalAttempt:
        """Read goal authority only when its immutable iteration link agrees."""
        with self._db.connection() as conn:
            row = self._attempt(conn, attempt_id, owner_id, "goal_iteration")
            link = conn.execute(
                "SELECT goal_id, ordinal FROM goal_iterations WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            return GoalAttempt(
                row["id"],
                link["goal_id"],
                link["ordinal"],
                row["chain_id"],
                row["conversation_id"],
                row["session_id"],
                row["owner_id"],
                row["state"],
            )

    def prepare_goal_iteration(self, goal_id: str, *, owner_id: str) -> GoalAttempt:
        """Reserve one generation and link it to real goal intent in FULL commit."""
        _identity(goal_id)
        goal = self._db.goal_runs.get(goal_id)
        with self._admission_transaction(goal.chain_id) as conn:
            self._check_runtime_owner(conn, owner_id)
            row = conn.execute(
                "SELECT * FROM goal_runs WHERE id=?", (goal_id,)
            ).fetchone()
            if row["status"] != "ready":
                raise AutomaticWorkRefused("goal_not_ready")
            try:
                goal.request.validate_verifier_invocations()
            except ValueError:
                raise AutomaticWorkRefused("ambiguous_verifier_invocation") from None
            existing = conn.execute(
                "SELECT a.id, a.owner_id FROM automatic_wake_attempts a JOIN goal_iterations i ON i.attempt_id=a.id WHERE i.goal_id=? AND a.state IN ('prepared','accepted')",
                (goal_id,),
            ).fetchone()
            if existing:
                self._attempt(conn, existing["id"], owner_id, "goal_iteration")
                attempt_id = existing["id"]
                if not conn.execute(
                    "SELECT 1 FROM goal_payload_reservations WHERE attempt_id=?",
                    (attempt_id,),
                ).fetchone():
                    self._db.goal_runs.reserve_result(conn, goal, attempt_id)
            else:
                self._check_admission(
                    conn,
                    goal.chain_id,
                    kind="generation",
                    amount=1,
                    limits=AutomaticWorkLimits.from_settings("goal_iteration"),
                )
                if self._target_active(conn, goal.conversation_id):
                    raise AutomaticWorkRefused("conversation_wake_active")
                ordinal = conn.execute(
                    "SELECT COALESCE(MAX(ordinal),0)+1 FROM goal_iterations WHERE goal_id=?",
                    (goal_id,),
                ).fetchone()[0]
                attempt_id, reservation_id = uuid4().hex, uuid4().hex
                now = self._wall_clock()
                conn.execute(
                    "INSERT INTO automatic_work_reservations (id,chain_id,owner_id,kind,amount,state,created_at,updated_at) VALUES (?,?,?,'generation',1,'reserved',?,?)",
                    (reservation_id, goal.chain_id, owner_id, now, now),
                )
                conn.execute(
                    "INSERT INTO automatic_wake_attempts (id,chain_id,conversation_id,session_id,owner_id,generation_reservation_id,run_ids_json,state,created_at,attempt_kind) VALUES (?,?,?,?,?,?,'[]','prepared',?,'goal_iteration')",
                    (
                        attempt_id,
                        goal.chain_id,
                        goal.conversation_id,
                        goal.conversation_id,
                        owner_id,
                        reservation_id,
                        now,
                    ),
                )
                conn.execute(
                    "INSERT INTO goal_iterations (id,goal_id,ordinal,launch_id,attempt_id,status) VALUES (?,?,?,?,?,'prepared')",
                    (uuid4().hex, goal_id, ordinal, uuid4().hex, attempt_id),
                )
                self._db.goal_runs.reserve_result(conn, goal, attempt_id)
        return self.read_goal_attempt(attempt_id, owner_id=owner_id)

    def accept_goal_iteration(
        self,
        attempt_id: str,
        *,
        owner_id: str,
        admission_guard: Callable[[], None] | None = None,
    ) -> bool:
        """Accept once; an optional runtime guard may only refuse admission.

        The guard runs synchronously inside the existing acceptance transaction;
        it must not access the database or perform asynchronous work.
        """
        return self._accept_attempt(
            attempt_id,
            owner_id=owner_id,
            limits=AutomaticWorkLimits.from_settings("goal_iteration"),
            kind="goal_iteration",
            admission_guard=admission_guard,
        )

    def abort_goal_iteration(self, attempt_id: str, *, owner_id: str) -> bool:
        """Refund only an unaccepted goal generation reservation."""
        return self._abort_attempt(attempt_id, owner_id=owner_id, kind="goal_iteration")

    def _accept_attempt(
        self,
        attempt_id: str,
        *,
        owner_id: str,
        limits: AutomaticWorkLimits | None,
        kind: str,
        admission_guard: Callable[[], None] | None = None,
    ) -> bool:
        with self._db.connection() as conn:
            attempt = self._attempt(conn, attempt_id, owner_id, kind)
            chain_id = attempt["chain_id"]
            review_boundary_id = attempt["review_boundary_id"]
            needs_review_scope = (
                review_boundary_id is not None and attempt["state"] == "prepared"
            )
        scope = nullcontext()
        has_review_scope = False
        if needs_review_scope:
            with self._progress_claim_lock:
                authority = self._review_authorities.get(owner_id)
                receipt = self._review_attempts.get(attempt_id)
            if authority is not None:
                scope = authority[2](receipt)
                has_review_scope = True
            # A concurrent abort may already have retired the receipt/issuer.
            # The original transaction checks owner and terminal state first;
            # a still-prepared attempt with missing authority fails validation.
        # Acquire native admission before SQLite, and keep it through the
        # post-commit native transition. Claim-time validation never takes this
        # scope, avoiding DB -> admission / admission -> DB lock inversion.
        with scope:
            return self._accept_attempt_transaction(
                chain_id,
                attempt_id,
                owner_id=owner_id,
                limits=limits,
                kind=kind,
                admission_guard=admission_guard,
                has_review_scope=has_review_scope,
            )

    def _accept_attempt_transaction(
        self,
        chain_id,
        attempt_id,
        *,
        owner_id,
        limits,
        kind,
        admission_guard,
        has_review_scope,
    ):
        with self._admission_transaction(chain_id) as conn:
            self._check_runtime_owner(conn, owner_id)
            attempt = self._attempt(conn, attempt_id, owner_id, kind)
            if attempt["state"] != "prepared":
                return False
            if attempt["review_boundary_id"] is not None:
                if not has_review_scope:
                    raise AutomaticWorkRefused("review_boundary_unavailable")
                receipt = self._review_attempts.get(attempt_id)
                self._validate_review_boundary(
                    receipt,
                    chain_id=chain_id,
                    owner_id=owner_id,
                    session_id=attempt["session_id"],
                    conversation_id=attempt["conversation_id"],
                )
                link = conn.execute(
                    "SELECT boundary_id FROM automatic_review_boundary_claims WHERE attempt_id=?",
                    (attempt_id,),
                ).fetchone()
                if link is None or link[0] != receipt.boundary_id:
                    raise AutomaticWorkRefused("review_boundary_unavailable")
            if kind == "workflow_phase":
                self._workflow_current(conn, attempt_id, owner_id)
                if limits.output_tokens == 0:
                    raise AutomaticWorkRefused("output_tokens_budget")
            if kind == "goal_iteration":
                goal = conn.execute(
                    "SELECT g.status, g.request_json FROM goal_runs g JOIN goal_iterations i ON i.goal_id=g.id WHERE i.attempt_id=?",
                    (attempt_id,),
                ).fetchone()
                if goal is None or goal["status"] != "ready":
                    raise AutomaticWorkRefused("goal_not_ready")
                from tldw_chatbook.Agents.goal_models import GoalRequest

                request = GoalRequest.model_validate_json(goal["request_json"])
                try:
                    request.validate_verifier_invocations()
                except ValueError:
                    raise AutomaticWorkRefused(
                        "ambiguous_verifier_invocation"
                    ) from None
                from tldw_chatbook.Agents.run_log import _setting
                from tldw_chatbook.config import coerce_bool_setting

                # Resolution/recovery may have awaited since the early gate.
                # Refuse inside the durable fence before consuming a generation.
                if not coerce_bool_setting(_setting("goal_runs_enabled", False), False):
                    raise AutomaticWorkRefused("goal_runs_disabled")
            self._check_admission(conn, chain_id, kind="generation", limits=limits)
            reservation = self._reservation(
                conn, attempt["generation_reservation_id"], owner_id
            )
            if reservation["state"] != "reserved":
                raise ValueError("attempt generation is not reserved")
            if admission_guard is not None:
                admission_guard()
            now = self._wall_clock()
            conn.execute(
                "UPDATE automatic_work_reservations SET state='committed', updated_at=? WHERE id=?",
                (now, reservation["id"]),
            )
            conn.execute(
                "UPDATE automatic_wake_attempts SET state='accepted', accepted_at=? WHERE id=?",
                (now, attempt_id),
            )
            if kind == "goal_iteration":
                conn.execute(
                    "UPDATE goal_iterations SET status='accepted', revision=revision+1 WHERE attempt_id=?",
                    (attempt_id,),
                )
            if kind == "workflow_phase":
                conn.execute(
                    "UPDATE workflow_phase_attempts SET status='accepted' WHERE native_attempt_id=?",
                    (attempt_id,),
                )
            self._start_automatic(conn, chain_id)
        self._transition_review_boundary(attempt_id, owner_id, "accepted")
        return True

    def read_workflow_attempt(
        self, attempt_id: str, *, owner_id: str
    ) -> WorkflowPhaseAttempt:
        """Read a durable reference, never reconstruct live dispatch custody."""
        from tldw_chatbook.Agents.workflow_models import WorkflowPhaseAttempt

        with self._db.connection() as conn:
            native = self._attempt(conn, attempt_id, owner_id, "workflow_phase")
            phase = conn.execute(
                "SELECT * FROM workflow_phase_attempts WHERE native_attempt_id=?",
                (attempt_id,),
            ).fetchone()
            return WorkflowPhaseAttempt(
                id=attempt_id,
                workflow_id=phase["workflow_id"],
                revision_id=phase["plan_revision"],
                phase_id=phase["phase_id"],
                purpose=phase["purpose"],
                control_revision=phase["control_revision"],
                command_id=phase["command_id"],
                repair_instruction_sha256=phase["repair_instruction_sha256"],
                chain_id=native["chain_id"],
                conversation_id=native["conversation_id"],
                session_id=native["session_id"],
                owner_id=native["owner_id"],
                state=native["state"],
                outcome={
                    "completed": "completed",
                    "failed": "failed",
                    "closed": "cancelled",
                }.get(phase["status"]),
            )

    def _workflow_enabled(self, *, _native=None) -> None:
        from tldw_chatbook.Agents.run_log import _setting
        from tldw_chatbook.config import coerce_bool_setting

        if _native is not None:
            from tldw_chatbook.Chat.console_workflow_runs import _fence_connection

            _fence_connection(_native, self)
            return
        if not coerce_bool_setting(_setting("workflow_runs_enabled", False), False):
            raise AutomaticWorkRefused("workflow_runs_disabled")

    def _workflow_current(
        self,
        conn,
        attempt_id,
        owner_id,
        *,
        accepted=False,
        _dispatch=None,
        _native=None,
    ):
        if _native is not None:
            from tldw_chatbook.Chat.console_workflow_runs import _fence_connection

            _fence_connection(_native, self)
        self._check_runtime_owner(conn, owner_id)
        if (
            conn.execute(
                "SELECT 1 FROM automatic_work_runtime_owner WHERE singleton=1 AND owner_id=?",
                (owner_id,),
            ).fetchone()
            is None
        ):
            raise AutomaticWorkRefused("runtime_owner_missing")
        native = self._attempt(conn, attempt_id, owner_id, "workflow_phase")
        phase = conn.execute(
            "SELECT p.*,w.control_revision AS current_revision,w.status AS workflow_status "
            "FROM workflow_phase_attempts p JOIN workflow_runs w ON w.id=p.workflow_id "
            "WHERE p.native_attempt_id=?",
            (attempt_id,),
        ).fetchone()
        if phase["control_revision"] != phase["current_revision"] or phase[
            "workflow_status"
        ] in {
            "stopped",
            "closed",
            "removed",
            "completed",
            "recovery_required",
            "paused",
        }:
            self._validate_workflow_dispatch(conn, _dispatch, native, phase)

        if accepted and native["state"] != "accepted":
            raise AutomaticWorkRefused("acceptance_required")
        self._workflow_enabled(_native=_native)
        return native, phase

    def _validate_workflow_dispatch(self, conn, proof, native, phase):
        from tldw_chatbook.Chat.console_workflow_runs import _dispatch_identity

        proof = _dispatch_identity(proof, self)
        task = proof.task.claim
        row = conn.execute(
            "SELECT t.status,t.run_id,c.launch_run_id,a.state,a.chain_id,a.owner_id "
            "FROM workflow_task_attempts t JOIN workflow_task_claims c ON c.task_attempt_id=t.id "
            "JOIN automatic_work_reservations a ON a.id=c.child_reservation_id "
            "WHERE t.id=? AND c.phase_attempt_id=? AND c.child_reservation_id=?",
            (task.id, phase["id"], task.reservation_id),
        ).fetchone()
        decision = conn.execute(
            "SELECT payload_json FROM workflow_decisions WHERE id=? AND workflow_id=? AND control_revision=?",
            (proof.decision_id, task.workflow_id, proof.revision - 1),
        ).fetchone()
        if (
            phase["workflow_status"] != "paused"
            or phase["current_revision"] != proof.revision
            or phase["native_attempt_id"] != proof.phase.attempt.id
            or phase["control_revision"] != proof.phase.attempt.control_revision
            or native["state"] != "accepted"
            or native["owner_id"] != task.owner_id
            or native["chain_id"] != task.chain_id
            or decision is None
            or decision[0] != proof.payload
            or row is None
            or tuple(row)
            not in (
                (
                    "accepted",
                    task.launch_run_id,
                    task.launch_run_id,
                    "committed",
                    task.chain_id,
                    task.owner_id,
                ),
                (
                    "accepted",
                    None,
                    task.launch_run_id,
                    "committed",
                    task.chain_id,
                    task.owner_id,
                ),
            )
        ):
            raise AutomaticWorkRefused("workflow_authority_changed")

    def prepare_workflow_phase(
        self,
        workflow_id: str,
        *,
        revision_id: int,
        phase_id: str,
        owner_id: str,
        session_id: str,
        command_id: str,
    ) -> WorkflowPhaseAttempt:
        """Reserve one execution phase and all mandatory task result capacity."""
        if type(revision_id) is not int or not 0 < revision_id < 2**63:
            raise ValueError("invalid workflow revision")
        _identity(phase_id)
        return self._prepare_workflow(
            workflow_id,
            revision_id=revision_id,
            phase_id=phase_id,
            purpose="execution",
            owner_id=owner_id,
            session_id=session_id,
            command_id=command_id,
        )

    def prepare_workflow_planning(
        self,
        workflow_id: str,
        *,
        owner_id: str,
        session_id: str,
        command_id: str,
        revision_id: int | None = None,
        repair_instruction: str | None = None,
    ) -> WorkflowPhaseAttempt:
        """Reserve read-only planning against the current immutable request/base plan."""
        if revision_id is not None and (
            type(revision_id) is not int or not 0 < revision_id < 2**63
        ):
            raise ValueError("invalid workflow revision")
        from hashlib import sha256

        from tldw_chatbook.Agents.workflow_models import RECORD_BYTES

        digest = None
        if repair_instruction is not None:
            if (
                type(repair_instruction) is not str
                or not repair_instruction
                or len(repair_instruction.encode("utf-8")) > RECORD_BYTES
                or revision_id is None
            ):
                raise ValueError("invalid workflow repair instruction")
            digest = sha256(repair_instruction.encode("utf-8")).hexdigest()
        return self._prepare_workflow(
            workflow_id,
            revision_id=revision_id,
            phase_id=None,
            purpose="planning",
            owner_id=owner_id,
            session_id=session_id,
            command_id=command_id,
            _repair_digest=digest,
        )

    def _prepare_workflow(
        self,
        workflow_id,
        *,
        revision_id,
        phase_id,
        purpose,
        owner_id,
        session_id,
        command_id,
        _remaining=None,
        _initial=None,
        _repair_digest=None,
    ):
        from tldw_chatbook.Agents.workflow_models import RECORD_BYTES

        for value in (workflow_id, owner_id, session_id, command_id):
            _identity(value)
        store = self._db.workflow_runs
        if _remaining is not None and _initial is not None:
            raise ValueError("workflow_decision_conflict")
        decision = _remaining if _remaining is not None else _initial
        with self._db.connection() as conn:
            chain_id = store._row(conn, workflow_id)["chain_id"]
        with self._admission_transaction(chain_id) as conn:
            self._check_runtime_owner(conn, owner_id)
            row = store._row(conn, workflow_id)
            prior = conn.execute(
                "SELECT p.*,a.owner_id,a.session_id FROM workflow_phase_attempts p "
                "JOIN automatic_wake_attempts a ON a.id=p.native_attempt_id WHERE p.command_id=?",
                (command_id,),
            ).fetchone()
            if prior:
                if tuple(
                    prior[key]
                    for key in (
                        "workflow_id",
                        "plan_revision",
                        "phase_id",
                        "purpose",
                        "owner_id",
                        "session_id",
                    )
                ) != (
                    workflow_id,
                    revision_id,
                    phase_id,
                    purpose,
                    owner_id,
                    session_id,
                ):
                    raise ValueError("workflow_command_conflict")
                if prior["repair_instruction_sha256"] != _repair_digest:
                    raise ValueError("workflow_command_conflict")
                if (
                    decision is not None
                    and store._attempt_decision(conn, prior) != decision
                ):
                    raise ValueError("workflow_command_conflict")
                attempt_id = prior["native_attempt_id"]
            else:
                self._workflow_enabled()
                if row["status"] not in {
                    "awaiting_plan",
                    "awaiting_approval",
                    "awaiting_resume",
                }:
                    raise AutomaticWorkRefused("workflow_not_ready")
                latest = conn.execute(
                    "SELECT MAX(revision) FROM workflow_plan_revisions WHERE workflow_id=?",
                    (workflow_id,),
                ).fetchone()[0]
                if revision_id != latest:
                    raise AutomaticWorkRefused("workflow_revision_changed")
                tasks = []
                if purpose == "execution":
                    tasks = conn.execute(
                        "SELECT task_id FROM workflow_tasks WHERE workflow_id=? AND plan_revision=? AND phase_id=? ORDER BY ordinal",
                        (workflow_id, revision_id, phase_id),
                    ).fetchall()
                    if not tasks:
                        raise ValueError("unknown_workflow_phase")
                    previous = conn.execute(
                        "SELECT 1 FROM workflow_phase_attempts WHERE workflow_id=? AND plan_revision=? AND phase_id=? AND native_attempt_id IS NOT NULL",
                        (workflow_id, revision_id, phase_id),
                    ).fetchone()
                    if _remaining is None and previous:
                        raise AutomaticWorkRefused("workflow_phase_already_attempted")
                    if _initial is not None and (
                        previous is not None
                        or _initial.workflow_id != workflow_id
                        or _initial.plan_revision != revision_id
                        or _initial.phase_id != phase_id
                        or _initial.action != "approve"
                        or _initial.reason != "ready"
                        or _initial.control_revision != row["control_revision"]
                    ):
                        raise ValueError("workflow_initial_decision_conflict")
                    if _remaining is not None:
                        if (
                            previous is None
                            or _remaining.workflow_id != workflow_id
                            or _remaining.plan_revision != revision_id
                            or _remaining.phase_id != phase_id
                            or _remaining.action != "approve"
                            or _remaining.reason != "remaining_ready"
                            or _remaining.control_revision != row["control_revision"]
                            or conn.execute(
                                "SELECT 1 FROM workflow_decisions WHERE id=?",
                                (_remaining.id,),
                            ).fetchone()
                        ):
                            raise ValueError("workflow_remaining_decision_conflict")
                        retained, remaining = store._remaining_partition(
                            conn, workflow_id, revision_id, phase_id
                        )
                        actual_same_phase = tuple(
                            ref
                            for ref in _remaining.result_refs
                            if conn.execute(
                                "SELECT 1 FROM workflow_task_attempts WHERE id=? AND workflow_id=? AND plan_revision=? AND phase_id=?",
                                (ref, workflow_id, revision_id, phase_id),
                            ).fetchone()
                        )
                        if set(actual_same_phase) != set(retained.values()) or len(
                            actual_same_phase
                        ) != len(retained):
                            raise ValueError("workflow_remaining_decision_conflict")
                        tasks = [task for task in tasks if task["task_id"] in remaining]
                limits = AutomaticWorkLimits.from_settings("workflow_phase")
                if limits.output_tokens == 0:
                    raise AutomaticWorkRefused("output_tokens_budget")
                self._check_admission(
                    conn, chain_id, kind="generation", amount=1, limits=limits
                )
                if self._target_active(conn, row["conversation_id"]):
                    raise AutomaticWorkRefused("conversation_wake_active")
                store._capacity(
                    conn,
                    workflow_id,
                    RECORD_BYTES * (len(tasks) or 1)
                    + (64 if _repair_digest is not None else 0),
                )
                if decision is not None:
                    store._save_decision(conn, decision, status="awaiting_approval")
                    row = store._row(conn, workflow_id)
                attempt_id, reservation_id, phase_key = (
                    uuid4().hex,
                    uuid4().hex,
                    uuid4().hex,
                )
                now = self._wall_clock()
                conn.execute(
                    "INSERT INTO automatic_work_reservations (id,chain_id,owner_id,kind,amount,state,created_at,updated_at) VALUES (?,?,?,'generation',1,'reserved',?,?)",
                    (reservation_id, chain_id, owner_id, now, now),
                )
                conn.execute(
                    "INSERT INTO automatic_wake_attempts (id,chain_id,conversation_id,session_id,owner_id,generation_reservation_id,run_ids_json,state,created_at,attempt_kind) VALUES (?,?,?,?,?,?,'[]','prepared',?,'workflow_phase')",
                    (
                        attempt_id,
                        chain_id,
                        row["conversation_id"],
                        session_id,
                        owner_id,
                        reservation_id,
                        now,
                    ),
                )
                ordinal = conn.execute(
                    "SELECT COALESCE(MAX(ordinal),0)+1 FROM workflow_phase_attempts WHERE workflow_id=?",
                    (workflow_id,),
                ).fetchone()[0]
                conn.execute(
                    "INSERT INTO workflow_phase_attempts (id,workflow_id,plan_revision,phase_id,ordinal,native_attempt_id,status,purpose,command_id,control_revision,result_bytes,repair_instruction_sha256) VALUES (?,?,?,?,?,?,'prepared',?,?,?,?,?)",
                    (
                        phase_key,
                        workflow_id,
                        revision_id,
                        phase_id,
                        ordinal,
                        attempt_id,
                        purpose,
                        command_id,
                        row["control_revision"],
                        RECORD_BYTES if purpose == "planning" else 0,
                        _repair_digest,
                    ),
                )
                for task in tasks:
                    task_key = uuid4().hex
                    task_ordinal = conn.execute(
                        "SELECT COALESCE(MAX(ordinal),0)+1 FROM workflow_task_attempts WHERE workflow_id=? AND plan_revision=? AND task_id=?",
                        (workflow_id, revision_id, task["task_id"]),
                    ).fetchone()[0]
                    conn.execute(
                        "INSERT INTO workflow_task_attempts (id,workflow_id,plan_revision,phase_id,task_id,ordinal,status) VALUES (?,?,?,?,?,?,'prepared')",
                        (
                            task_key,
                            workflow_id,
                            revision_id,
                            phase_id,
                            task["task_id"],
                            task_ordinal,
                        ),
                    )
                    conn.execute(
                        "INSERT INTO workflow_task_claims (task_attempt_id,phase_attempt_id,launch_run_id) VALUES (?,?,?)",
                        (task_key, phase_key, uuid4().hex),
                    )
                    conn.execute(
                        "INSERT INTO workflow_payload_reservations VALUES (?,?,?,'reserved')",
                        (task_key, workflow_id, RECORD_BYTES),
                    )
        return self.read_workflow_attempt(attempt_id, owner_id=owner_id)

    def accept_workflow_phase(self, attempt_id: str, *, owner_id: str) -> bool:
        """Commit acceptance once under FULL durability, before any billable work."""
        return self._accept_attempt(
            attempt_id,
            owner_id=owner_id,
            limits=AutomaticWorkLimits.from_settings("workflow_phase"),
            kind="workflow_phase",
        )

    def abort_workflow_planning(self, original, *, _prepared=None) -> bool:
        """Abort only the original returned, unaccepted planning invocation."""
        import inspect

        from tldw_chatbook.Agents.workflow_models import WorkflowPhaseAttempt
        from tldw_chatbook.Chat.console_workflow_preparation import (
            _PLANNING_ABORT_CALLER,
            _PreparedWorkflowReply,
        )

        caller = inspect.currentframe().f_back
        if (
            type(original) is not WorkflowPhaseAttempt
            or type(_prepared) is not _PreparedWorkflowReply
            or caller.f_code is not _PLANNING_ABORT_CALLER
            or caller.f_locals.get("self") is not _prepared
        ):
            raise PermissionError("original prepared planning abort required")
        _prepared._planning_abort_source(self, original)
        with self.transaction() as conn:
            _prepared._planning_abort_source(self, original)
            self._check_runtime_owner(conn, original.owner_id)
            native = self._attempt(
                conn, original.id, original.owner_id, "workflow_phase"
            )
            phase = conn.execute(
                "SELECT * FROM workflow_phase_attempts WHERE native_attempt_id=?",
                (original.id,),
            ).fetchone()
            workflow = self._db.workflow_runs._row(conn, original.workflow_id)
            current = self.read_workflow_attempt(
                original.id, owner_id=original.owner_id
            )
            if (
                current != original
                or original.state != "prepared"
                or original.purpose != "planning"
                or original.phase_id is not None
                or phase["status"] != "prepared"
                or phase["retirement"] != "pending"
                or workflow["control_revision"] != original.control_revision
                or conn.execute(
                    "SELECT MAX(revision) FROM workflow_plan_revisions WHERE workflow_id=?",
                    (original.workflow_id,),
                ).fetchone()[0]
                != original.revision_id
                or conn.execute(
                    "SELECT 1 FROM workflow_task_claims WHERE phase_attempt_id=? "
                    "UNION ALL SELECT 1 FROM workflow_change_windows WHERE phase_attempt_id=? "
                    "UNION ALL SELECT 1 FROM workflow_active_intervals WHERE workflow_id=? AND ended_at IS NULL "
                    "UNION ALL SELECT 1 FROM workflow_task_attempts WHERE native_attempt_id=? LIMIT 1",
                    (phase["id"], phase["id"], original.workflow_id, original.id),
                ).fetchone()
                or conn.execute(
                    "SELECT 1 FROM automatic_work_reservations WHERE chain_id=? AND created_at>=? "
                    "AND kind IN ('model_call','tokens') AND state NOT IN ('released','reserved') LIMIT 1",
                    (original.chain_id, native["created_at"]),
                ).fetchone()
            ):
                raise AutomaticWorkRefused("workflow_planning_abort_unproven")
            now = self._wall_clock()
            if (
                conn.execute(
                    "UPDATE automatic_work_reservations SET state='released',updated_at=? "
                    "WHERE id=? AND chain_id=? AND owner_id=? AND kind='generation' AND amount=1 AND state='reserved'",
                    (
                        now,
                        native["generation_reservation_id"],
                        original.chain_id,
                        original.owner_id,
                    ),
                ).rowcount
                != 1
            ):
                raise AutomaticWorkRefused("workflow_planning_abort_unproven")
            if (
                conn.execute(
                    "UPDATE automatic_wake_attempts SET state='aborted',completed_at=? "
                    "WHERE id=? AND owner_id=? AND attempt_kind='workflow_phase' AND state='prepared'",
                    (now, original.id, original.owner_id),
                ).rowcount
                != 1
            ):
                raise AutomaticWorkRefused("workflow_planning_abort_unproven")
            if (
                conn.execute(
                    "UPDATE workflow_phase_attempts SET status='closed',retirement='drained',result_bytes=0 "
                    "WHERE id=? AND native_attempt_id=? AND purpose='planning' AND phase_id IS NULL "
                    "AND status='prepared' AND retirement='pending'",
                    (phase["id"], original.id),
                ).rowcount
                != 1
            ):
                raise AutomaticWorkRefused("workflow_planning_abort_unproven")
        return True

    def claim_workflow_task(
        self, attempt_id: str, *, task_id: str, owner_id: str
    ) -> WorkflowTaskAttempt:
        """Consume one launch once; an existing claim is lookup-only on redelivery."""
        from tldw_chatbook.Agents.workflow_models import WorkflowTaskClaim

        _identity(task_id)
        attempt = self.read_workflow_attempt(attempt_id, owner_id=owner_id)
        with self._admission_transaction(attempt.chain_id) as conn:
            _, phase = self._workflow_current(conn, attempt_id, owner_id, accepted=True)
            if phase["purpose"] != "execution":
                raise AutomaticWorkRefused("planning_has_no_tasks")
            row = conn.execute(
                "SELECT t.*,c.child_reservation_id,c.launch_run_id FROM workflow_task_attempts t JOIN workflow_task_claims c ON c.task_attempt_id=t.id WHERE c.phase_attempt_id=? AND t.task_id=?",
                (phase["id"], task_id),
            ).fetchone()
            if not row:
                raise ValueError("unknown_workflow_task")
            if row["child_reservation_id"] is None:
                self._check_admission(
                    conn,
                    attempt.chain_id,
                    kind="child_launch",
                    amount=1,
                    limits=AutomaticWorkLimits.from_settings("workflow_phase"),
                )
                reservation_id, now = uuid4().hex, self._wall_clock()
                conn.execute(
                    "INSERT INTO automatic_work_reservations (id,chain_id,owner_id,kind,amount,state,created_at,updated_at) VALUES (?,?,?,'child_launch',1,'reserved',?,?)",
                    (reservation_id, attempt.chain_id, owner_id, now, now),
                )
                conn.execute(
                    "UPDATE workflow_task_claims SET child_reservation_id=? WHERE task_attempt_id=?",
                    (reservation_id, row["id"]),
                )
                conn.execute(
                    "UPDATE workflow_task_attempts SET status='accepted' WHERE id=?",
                    (row["id"],),
                )
            values = dict(
                conn.execute(
                    "SELECT * FROM workflow_task_attempts WHERE id=?", (row["id"],)
                ).fetchone()
            )
            return WorkflowTaskClaim(
                **values,
                reservation_id=row["child_reservation_id"] or reservation_id,
                launch_run_id=row["launch_run_id"],
                phase_attempt_id=attempt_id,
                chain_id=attempt.chain_id,
                conversation_id=attempt.conversation_id,
                session_id=attempt.session_id,
                owner_id=owner_id,
            )

    def _workflow_clock_unknown(
        self, conn, workflow_id, *, _accounting=None, _dispatch=None, _native=None
    ):
        rows = tuple(
            conn.execute(
                "SELECT p.*,n.state AS native_state,n.owner_id AS native_owner_id "
                "FROM workflow_phase_attempts p JOIN automatic_wake_attempts n "
                "ON n.id=p.native_attempt_id WHERE p.workflow_id=? AND p.retirement='unknown' "
                "AND n.state IN ('accepted','review_required')",
                (workflow_id,),
            )
        )
        if not rows:
            return False
        from tldw_chatbook.Chat.console_workflow_runs import _quiescence_evidence

        return any(
            row["native_state"] == "review_required"
            or not _quiescence_evidence(
                self, conn, row, context=_accounting, dispatch=_dispatch, native=_native
            )
            for row in rows
        )

    def _workflow_active_seconds(
        self,
        conn,
        workflow_id,
        *,
        observe=False,
        _accounting=None,
        _dispatch=None,
        _native=None,
    ):
        """Union intervals are serialized per workflow, never per task."""
        if self._workflow_clock_unknown(
            conn,
            workflow_id,
            _accounting=_accounting,
            _dispatch=_dispatch,
            _native=_native,
        ):
            raise AutomaticWorkRefused("clock_unknown")
        total = 0.0
        now, mono = self._wall_clock(), self._monotonic_clock()
        for row in conn.execute(
            "SELECT * FROM workflow_active_intervals WHERE workflow_id=?",
            (workflow_id,),
        ):
            charged = row["charged_seconds"]
            if row["ended_at"] is None:
                if (
                    row["clock_owner_id"] != _CLOCK_OWNER_ID
                    or row["started_monotonic"] is None
                    or row["observed_monotonic"] is None
                    or not all(
                        math.isfinite(v)
                        for v in (
                            now,
                            mono,
                            row["started_monotonic"],
                            row["observed_monotonic"],
                        )
                    )
                ):
                    raise AutomaticWorkRefused("clock_unknown")
                if now < row["started_at"] or mono < row["observed_monotonic"]:
                    raise AutomaticWorkRefused("clock_reversed")
                charged = max(
                    charged, now - row["started_at"], mono - row["started_monotonic"]
                )
                if observe:
                    conn.execute(
                        "UPDATE workflow_active_intervals SET charged_seconds=?,observed_monotonic=? WHERE id=?",
                        (charged, mono, row["id"]),
                    )
            total += charged
        return total

    def workflow_active_seconds(self, workflow_id: str) -> float:
        """Read cumulative charge; foreign open intervals refuse rather than reset."""
        with self._db.connection() as conn:
            return self._workflow_active_seconds(conn, workflow_id)

    def _resume_workflow_dispatch(
        self, conn, chain_id, owner_id, *, _dispatch=None, _native=None
    ) -> bool:
        """Reopen workflow time inside the same transaction as native dispatch."""
        if _native is not None:
            from tldw_chatbook.Chat.console_workflow_runs import _fence_connection

            _fence_connection(_native, self)
        root = self._allowance_chain(conn, chain_id)["id"]
        workflow = conn.execute(
            "SELECT id FROM workflow_runs WHERE chain_id=?", (root,)
        ).fetchone()
        if workflow is None:
            return False
        native = conn.execute(
            "SELECT id FROM automatic_wake_attempts WHERE chain_id=? "
            "AND attempt_kind='workflow_phase' AND state='accepted'",
            (root,),
        ).fetchone()
        if native is None:
            raise AutomaticWorkRefused("acceptance_required")
        self._workflow_current(
            conn,
            native["id"],
            owner_id,
            accepted=True,
            _dispatch=_dispatch,
            _native=_native,
        )
        conn.execute(
            "UPDATE workflow_phase_attempts SET retirement='pending' WHERE native_attempt_id=?",
            (native["id"],),
        )
        already_open = conn.execute(
            "SELECT 1 FROM workflow_active_intervals WHERE workflow_id=? AND ended_at IS NULL",
            (workflow["id"],),
        ).fetchone()
        self._start_automatic(conn, root)
        return already_open is None

    def resume_workflow_phase(
        self,
        attempt_id: str,
        *,
        owner_id: str,
        _dispatch=None,
        _quiescence=None,
        _cleanup=False,
    ) -> bool:
        """Resume attributable tool work after a positively drained capacity wait.

        Native provider admission and child commit also reopen transactionally;
        polling and reserving an unlaunched child do not resume active time.
        """
        attempt = self.read_workflow_attempt(attempt_id, owner_id=owner_id)
        if _quiescence is not None or _cleanup:
            from tldw_chatbook.Chat.console_workflow_runs import _quiescence_receiving

            mode = "cleanup" if _cleanup else "resume"
            _quiescence_receiving(_quiescence, self, mode)
            if _quiescence.phase.attempt != attempt:
                raise PermissionError("original workflow accounting attempt changed")
        if _cleanup:
            with self.transaction() as conn:
                _quiescence_receiving(_quiescence, self, "cleanup")
                self._check_runtime_owner(conn, owner_id)
                native = self._attempt(conn, attempt_id, owner_id, "workflow_phase")
                phase = conn.execute(
                    "SELECT * FROM workflow_phase_attempts WHERE native_attempt_id=?",
                    (attempt_id,),
                ).fetchone()
                if (
                    native["state"] != "accepted"
                    or phase["retirement"] != "unknown"
                    or phase["workflow_id"] != attempt.workflow_id
                    or self._workflow_clock_unknown(conn, attempt.workflow_id)
                ):
                    raise AutomaticWorkRefused("workflow_cleanup_accounting_unproven")
                conn.execute(
                    "UPDATE workflow_phase_attempts SET retirement='pending' WHERE id=?",
                    (phase["id"],),
                )
                self._start_automatic(conn, attempt.chain_id)
                return True
        with self._admission_transaction(attempt.chain_id) as conn:
            self._workflow_current(
                conn, attempt_id, owner_id, accepted=True, _dispatch=_dispatch
            )
            self._check_admission(
                conn,
                attempt.chain_id,
                kind="generation",
                limits=AutomaticWorkLimits.from_settings("workflow_phase"),
                _dispatch=_dispatch,
            )
            return self._resume_workflow_dispatch(
                conn, attempt.chain_id, owner_id, _dispatch=_dispatch
            )

    def _workflow_unsettled(self, conn, phase, chain_id, *, quiescing=False):
        # Unclaimed/reserved children have not dispatched. Keep their receipts
        # through a capacity wait, but require every task for phase completion.
        task_filter = (
            "(t.run_id IS NOT NULL OR a.state NOT IN ('reserved','released')) AND "
            if quiescing
            else ""
        )
        unsettled = conn.execute(
            "SELECT 1 FROM workflow_task_claims c "
            "JOIN workflow_task_attempts t ON t.id=c.task_attempt_id "
            "LEFT JOIN automatic_work_reservations a ON a.id=c.child_reservation_id "
            "LEFT JOIN workflow_results r ON r.attempt_id=t.id "
            "WHERE c.phase_attempt_id=? AND "
            + task_filter
            + "(c.child_reservation_id IS NULL OR t.retirement!='drained' "
            "OR t.status NOT IN ('completed','failed') OR r.attempt_id IS NULL) LIMIT 1",
            (phase["id"],),
        ).fetchone()
        reservation_filter = (
            "AND NOT (r.state='reserved' AND r.kind='child_launch')"
            if quiescing
            else ""
        )
        return (
            unsettled
            or conn.execute(
                "SELECT 1 FROM workflow_change_windows WHERE phase_attempt_id=? AND state IN ('active','unresolved','disclosed_failure') LIMIT 1",
                (phase["id"],),
            ).fetchone()
            or conn.execute(
                "SELECT 1 FROM automatic_work_reservations r "
                "JOIN automatic_work_chains c ON c.id=r.chain_id "
                "WHERE (c.id=? OR c.allowance_root_chain_id=?) "
                "AND (r.state IN ('reserved','uncertain') OR (r.kind='tokens' AND r.state='committed')) "
                + reservation_filter
                + " LIMIT 1",
                (chain_id, chain_id),
            ).fetchone()
        )

    def _close_workflow_interval(self, conn, workflow_id) -> bool:
        self._workflow_active_seconds(conn, workflow_id, observe=True)
        return (
            conn.execute(
                "UPDATE workflow_active_intervals SET ended_at=? WHERE workflow_id=? AND ended_at IS NULL",
                (self._wall_clock(), workflow_id),
            ).rowcount
            > 0
        )

    def quiesce_workflow_phase(
        self, attempt_id: str, *, owner_id: str, drained: bool, _quiescence=None
    ) -> bool:
        """Close only a positively drained wait, retaining acceptance and receipts.

        The native owner must observe physical drain, including tools and owned
        effects. Terminal task rows alone are not that observation.
        """
        if type(drained) is not bool:
            raise ValueError("invalid drain observation")
        attempt = self.read_workflow_attempt(attempt_id, owner_id=owner_id)
        if _quiescence is not None:
            from tldw_chatbook.Chat.console_workflow_runs import _quiescence_receiving

            _quiescence_receiving(_quiescence, self, "quiesce")
            if _quiescence.phase.attempt != attempt:
                raise PermissionError("original workflow accounting attempt changed")
        with self.transaction() as conn:
            self._check_runtime_owner(conn, owner_id)
            native = self._attempt(conn, attempt_id, owner_id, "workflow_phase")
            if native["state"] != "accepted" or not drained:
                return False
            phase = conn.execute(
                "SELECT * FROM workflow_phase_attempts WHERE native_attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if self._workflow_unsettled(conn, phase, attempt.chain_id, quiescing=True):
                raise AutomaticWorkRefused("workflow_not_settled")
            if _quiescence is not None:
                _quiescence_receiving(_quiescence, self, "quiesce")
                self._workflow_current(conn, attempt_id, owner_id, accepted=True)
                interval = conn.execute(
                    "SELECT id FROM workflow_active_intervals WHERE workflow_id=? AND ended_at IS NULL",
                    (attempt.workflow_id,),
                ).fetchone()
                if interval is None or not self._close_workflow_interval(
                    conn, attempt.workflow_id
                ):
                    raise AutomaticWorkRefused("workflow_original_interval_missing")
                conn.execute(
                    "UPDATE workflow_phase_attempts SET retirement='unknown' WHERE id=?",
                    (phase["id"],),
                )
                return tuple(
                    conn.execute(
                        "SELECT * FROM workflow_active_intervals WHERE id=?",
                        (interval[0],),
                    ).fetchone()
                )
            conn.execute(
                "UPDATE workflow_phase_attempts SET retirement='drained' WHERE id=?",
                (phase["id"],),
            )
            return self._close_workflow_interval(conn, attempt.workflow_id)

    def settle_workflow_phase(
        self,
        attempt_id: str,
        *,
        owner_id: str,
        drained: bool,
        plan: WorkflowPlan | None = None,
        outcome: Literal["completed", "failed", "cancelled"] = "completed",
    ) -> bool:
        """Close only after the owner proves physical drain and required durable results.

        The app-owned coordinator supplies its physical-drain observation. Logical
        completion, permission waits and user acknowledgments are not observations.
        Planning failure/cancellation publishes no plan and does not change control
        authority. Existing phase statuses failed/closed retain those outcomes.
        """
        if type(drained) is not bool:
            raise ValueError("invalid drain observation")
        if outcome not in {"completed", "failed", "cancelled"}:
            raise ValueError("invalid workflow outcome")
        attempt = self.read_workflow_attempt(attempt_id, owner_id=owner_id)
        if outcome != "completed" and (
            attempt.purpose != "planning" or plan is not None
        ):
            raise ValueError(
                "only unsuccessful planning may settle without publication"
            )
        with self.transaction() as conn:
            self._check_runtime_owner(conn, owner_id)
            native = self._attempt(conn, attempt_id, owner_id, "workflow_phase")
            if native["state"] != "accepted":
                return False
            if not drained:
                return False
            phase = conn.execute(
                "SELECT * FROM workflow_phase_attempts WHERE native_attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if self._workflow_unsettled(conn, phase, attempt.chain_id):
                raise AutomaticWorkRefused("workflow_not_settled")
            if phase["purpose"] == "planning":
                if outcome == "completed" and plan is None:
                    raise AutomaticWorkRefused("planning_result_missing")
            elif plan is not None:
                raise ValueError("execution phase cannot publish a plan")
            self._close_workflow_interval(conn, attempt.workflow_id)
            conn.execute(
                "UPDATE automatic_wake_attempts SET state='completed',completed_at=? WHERE id=?",
                (self._wall_clock(), attempt_id),
            )
            conn.execute(
                "UPDATE workflow_phase_attempts SET status=?,retirement='drained',result_bytes=0 WHERE id=?",
                ("closed" if outcome == "cancelled" else outcome, phase["id"]),
            )
            if phase["purpose"] == "planning" and outcome == "completed":
                self._db.workflow_runs._save_plan(
                    conn,
                    attempt.workflow_id,
                    plan,
                    expected_control_revision=phase["control_revision"],
                )
        return True

    def bind_workflow_task_run(
        self,
        attempt_id: str,
        *,
        task_id: str,
        run_id: str,
        owner_id: str,
        _dispatch=None,
    ) -> None:
        """Attach the native run to an already claimed launch, exactly once."""
        from tldw_chatbook.Agents.workflow_models import WorkflowTaskAttempt

        _identity(task_id)
        _identity(run_id)
        with self.transaction() as conn:
            _, phase = self._workflow_current(
                conn, attempt_id, owner_id, accepted=True, _dispatch=_dispatch
            )
            row = conn.execute(
                "SELECT t.*,c.launch_run_id FROM workflow_task_attempts t JOIN workflow_task_claims c ON c.task_attempt_id=t.id WHERE c.phase_attempt_id=? AND t.task_id=? AND c.child_reservation_id IS NOT NULL",
                (phase["id"], task_id),
            ).fetchone()
            if row is None:
                raise AutomaticWorkRefused("task_claim_required")
            if row["run_id"] not in (None, run_id) or row["launch_run_id"] != run_id:
                raise ValueError("workflow_run_conflict")
            values = dict(row)
            values.pop("launch_run_id")
            values["run_id"] = run_id
            self._db.workflow_runs._ownership(conn, WorkflowTaskAttempt(**values))
            conn.execute(
                "UPDATE workflow_task_attempts SET run_id=? WHERE id=?",
                (run_id, row["id"]),
            )


# Finite original workflow fence receivers; captured when these definitions load.
_WORKFLOW_NATIVE_RECORDS = tuple(
    (
        name,
        method,
        tuple(
            (
                function,
                function.__code__,
                function.__defaults__,
                function.__kwdefaults__,
                tuple((function.__kwdefaults__ or {}).items()),
                function.__closure__,
                tuple(
                    (cell, cell.cell_contents) for cell in function.__closure__ or ()
                ),
            )
            for function in (
                method,
                *([method.__wrapped__] if hasattr(method, "__wrapped__") else []),
            )
        ),
    )
    for name in (
        "transaction",
        "_admission_transaction",
        "commit",
        "_check_admission",
        "_workflow_enabled",
        "_workflow_current",
        "_resume_workflow_dispatch",
        "quiesce_workflow_phase",
        "resume_workflow_phase",
        "_workflow_clock_unknown",
        "_workflow_active_seconds",
        "_snapshot",
    )
    for method in (vars(AutomaticWorkLedger)[name],)
)
