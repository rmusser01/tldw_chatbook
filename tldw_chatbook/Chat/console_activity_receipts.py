"""App-lifetime coordination for local Console activity receipts."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from loguru import logger

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@dataclass(frozen=True)
class ConsoleActivityReceipt:
    """Immutable switcher-facing projection of one durable unseen receipt."""

    activity_id: str
    origin: str
    logical_outcome_id: str
    transition_revision: int
    session_id: str | None
    conversation_id: str | None
    run_id: str | None
    assistant_message_id: str | None
    status: str
    created_at: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ConsoleActivityReceipt":
        return cls(
            activity_id=str(row["activity_id"]),
            origin=str(row["origin"]),
            logical_outcome_id=str(row["logical_outcome_id"]),
            transition_revision=int(row["transition_revision"]),
            session_id=(str(row["session_id"]) if row.get("session_id") else None),
            conversation_id=(
                str(row["conversation_id"]) if row.get("conversation_id") else None
            ),
            run_id=str(row["run_id"]) if row.get("run_id") else None,
            assistant_message_id=(
                str(row["assistant_message_id"])
                if row.get("assistant_message_id")
                else None
            ),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
        )


@dataclass(frozen=True)
class FleetReceiptPublication:
    """Result of publishing every eligible child from one FLEET drain."""

    activity_ids: tuple[str, ...]
    complete: bool


class ConsoleActivityReceiptService:
    """Serialize receipt persistence, cache publication, and FLEET marks."""

    _FLEET_STATUS = {
        "done": "done",
        "error": "failed",
        "stuck": "stuck",
        "cancelled": "cancelled",
    }

    def __init__(self, runs_db: AgentRunsDB, marks: Any | None) -> None:
        self._db = runs_db
        self._marks = marks
        self._lock = threading.RLock()
        self._snapshot: tuple[ConsoleActivityReceipt, ...] = ()
        # A receipt write can fail after a FLEET survivor has settled.  Keep
        # that conversation-level evidence independently from hydration so a
        # later successful empty reload cannot erase the only remaining cue.
        self._fleet_fallback_debt: set[str] = set()
        self._fleet_marks_seeded = False
        self._projection_generation = 0
        self._state: Literal["cold", "loading", "ready", "degraded"] = (
            "cold" if runs_db.receipt_capability_available else "degraded"
        )

    def hydration_state(self) -> Literal["cold", "loading", "ready", "degraded"]:
        """Return the content-free storage readiness state."""
        with self._lock:
            return self._state

    @property
    def projection_generation(self) -> int:
        """Return the monotonic generation of the immutable unseen snapshot."""
        with self._lock:
            return self._projection_generation

    @property
    def degraded(self) -> bool:
        """Whether receipt storage or coarse-mark reconciliation is degraded."""
        return self.hydration_state() == "degraded"

    def unseen_snapshot(self) -> tuple[ConsoleActivityReceipt, ...]:
        """Return the immutable in-memory unseen snapshot without SQLite I/O."""
        with self._lock:
            return self._snapshot

    def _set_degraded(self, operation: str, exc: Exception | None = None) -> None:
        self._state = "degraded"
        logger.warning(
            "Console activity receipts degraded operation={} exception_type={}",
            operation,
            type(exc).__name__ if exc is not None else "invalid_state",
        )

    def _replace_snapshot(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._snapshot = tuple(ConsoleActivityReceipt.from_row(row) for row in rows)
        self._projection_generation += 1

    def _reload_locked(self) -> bool:
        try:
            rows: list[Mapping[str, Any]] = []
            cursor: tuple[str, str] | None = None
            while True:
                page = (
                    self._db.list_unseen_console_activity(cursor=cursor)
                    if cursor is not None
                    else self._db.list_unseen_console_activity()
                )
                rows.extend(page)
                next_cursor = getattr(page, "next_cursor", None)
                if next_cursor is None:
                    break
                if next_cursor == cursor:
                    raise RuntimeError("Console activity cursor did not advance.")
                cursor = next_cursor
        except Exception as exc:  # noqa: BLE001 - live receipt paths degrade, never settle
            self._set_degraded("read", exc)
            return False
        self._replace_snapshot(rows)
        return True

    def _reconcile_fleet_marks_locked(self) -> bool:
        if self._state == "degraded" or self._marks is None:
            return self._marks is None
        mark = getattr(self._marks, "FLEET_UNSEEN", None)
        if mark is None:
            return True
        fallback_mark = getattr(self._marks, "FLEET_RECEIPT_FALLBACK", None)
        try:
            if fallback_mark is not None:
                durable_fallback = set(
                    self._marks.list_marked_conversation_ids(fallback_mark)
                )
                if not self._fleet_marks_seeded:
                    self._fleet_fallback_debt.update(durable_fallback)
                for conversation_id in sorted(
                    self._fleet_fallback_debt - durable_fallback
                ):
                    self._marks.set_mark(conversation_id, fallback_mark)
            self._fleet_marks_seeded = True
        except Exception as exc:  # noqa: BLE001 - preserve coarse marks
            self._set_degraded("fleet_fallback_reconcile", exc)
            return False
        receipt_required = {
            receipt.conversation_id
            for receipt in self._snapshot
            if receipt.origin == "fleet_survivor" and receipt.conversation_id
        }
        try:
            marked = set(self._marks.list_marked_conversation_ids(mark))
            required = receipt_required | self._fleet_fallback_debt
            for conversation_id in sorted(required - marked):
                self._marks.set_mark(conversation_id, mark)
            for conversation_id in sorted(marked - required):
                self._marks.clear_mark(conversation_id, mark)
        except Exception as exc:  # noqa: BLE001 - a stale badge is safer than a crash
            self._set_degraded("fleet_mark", exc)
            return False
        return True

    def _set_fleet_fallback_locked(self, conversation_id: str) -> bool:
        self._fleet_fallback_debt.add(conversation_id)
        if self._marks is None:
            return True
        mark = getattr(self._marks, "FLEET_RECEIPT_FALLBACK", None)
        if mark is None:
            return True
        try:
            self._marks.set_mark(conversation_id, mark)
        except Exception as exc:  # noqa: BLE001 - coarse badge remains best effort
            self._set_degraded("fleet_fallback_set", exc)
            return False
        return True

    def _clear_fleet_fallback_locked(self, conversation_id: str) -> bool:
        if self._marks is not None:
            mark = getattr(self._marks, "FLEET_RECEIPT_FALLBACK", None)
            if mark is not None:
                try:
                    self._marks.clear_mark(conversation_id, mark)
                except Exception as exc:  # noqa: BLE001 - retain evidence
                    self._set_degraded("fleet_fallback_clear", exc)
                    return False
        self._fleet_fallback_debt.discard(conversation_id)
        return True

    def ensure_fleet_mark(self, conversation_id: str) -> bool:
        """Best-effort set the coarse badge under receipt serialization.

        This is the compatibility request used when a wake lands outside the
        visible conversation.  It deliberately preserves the incumbent badge
        even when no receipt exists (for example, an owed pre-v15 launch wake);
        the next successful hydration/reconciliation may remove that
        false-positive once durable receipt state is known.
        """
        conversation_id = str(conversation_id or "").strip()
        with self._lock:
            if not conversation_id or self._marks is None:
                return False
            mark = getattr(self._marks, "FLEET_UNSEEN", None)
            if mark is None:
                return False
            try:
                self._marks.set_mark(conversation_id, mark)
            except Exception as exc:  # noqa: BLE001 - presentation bookkeeping
                self._set_degraded("fleet_mark_set", exc)
                return False
            return True

    def clear_fleet_mark_if_seen(self, conversation_id: str) -> bool:
        """Clear one coarse badge only from a ready, receipt-free snapshot.

        A cold, loading, or degraded cache is uncertainty, so the safe result
        is to retain the badge.  With a ready cache, any unseen survivor
        receipt for the conversation also retains/reasserts it.  The decision
        and mutation share the same lock as publication and acknowledgement.
        """
        conversation_id = str(conversation_id or "").strip()
        with self._lock:
            if not conversation_id or self._marks is None:
                return False
            mark = getattr(self._marks, "FLEET_UNSEEN", None)
            if mark is None:
                return False
            if self._state != "ready":
                return False
            # Visiting the conversation is the explicit compatibility
            # acknowledgement for survivor evidence that could not be
            # represented by a durable receipt.
            if not self._clear_fleet_fallback_locked(conversation_id):
                return False
            if any(
                receipt.origin == "fleet_survivor"
                and receipt.conversation_id == conversation_id
                for receipt in self._snapshot
            ):
                try:
                    self._marks.set_mark(conversation_id, mark)
                except Exception as exc:  # noqa: BLE001 - keep stale badge on failure
                    self._set_degraded("fleet_mark_set", exc)
                return False
            try:
                marked = set(self._marks.list_marked_conversation_ids(mark))
                if conversation_id not in marked:
                    return False
                self._marks.clear_mark(conversation_id, mark)
            except Exception as exc:  # noqa: BLE001 - keep stale badge on failure
                self._set_degraded("fleet_mark_clear", exc)
                return False
            return True

    def hydrate_from_storage(self) -> int:
        """Hydrate once off-loop; degraded calls are explicit retries."""
        with self._lock:
            if self._state == "ready":
                return len(self._snapshot)
            self._state = "loading"
            if not self._reload_locked():
                return len(self._snapshot)
            self._state = "ready"
            if not self._reconcile_fleet_marks_locked():
                return len(self._snapshot)
            return len(self._snapshot)

    def publish_ordinary(
        self,
        *,
        logical_outcome_id: str,
        status: str,
        session_id: str,
        conversation_id: str | None,
        assistant_message_id: str | None = None,
    ) -> str | None:
        """Publish one ordinary inactive-session terminal outcome."""
        with self._lock:
            try:
                activity_id, _created = self._db.publish_console_activity(
                    origin="ordinary",
                    logical_outcome_id=logical_outcome_id,
                    status=status,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    assistant_message_id=assistant_message_id,
                )
            except Exception as exc:  # noqa: BLE001 - execution settlement must continue
                self._set_degraded("publish_ordinary", exc)
                return None
            if not self._reload_locked():
                return activity_id
            self._state = "ready"
            self._reconcile_fleet_marks_locked()
            return activity_id

    def publish_fleet_drain(self, event: Any) -> FleetReceiptPublication:
        """Publish post-turn survivors and report all-or-partial durability."""
        with self._lock:
            conversation_id = str(getattr(event, "conversation_id", "") or "")
            drain_id = str(getattr(event, "drain_id", "") or "")
            if not conversation_id or not drain_id:
                self._set_degraded("fleet_identity")
                return FleetReceiptPublication(activity_ids=(), complete=False)
            published: list[str] = []
            complete = True
            eligible_children = 0
            incomplete_operation = "fleet_status"
            for ordinal, child in enumerate(getattr(event, "children", ()) or ()):
                if getattr(child, "settled_after_turn", False) is not True:
                    continue
                eligible_children += 1
                raw_status = str(getattr(child, "status", "") or "")
                status = self._FLEET_STATUS.get(raw_status)
                if status is None:
                    complete = False
                    continue
                run_id = getattr(child, "run_id", None)
                logical_outcome_id = (
                    f"fleet-run:{run_id}"
                    if run_id
                    else f"fleet-drain:{drain_id}:{ordinal}"
                )
                try:
                    activity_id, _created = self._db.publish_console_activity(
                        origin="fleet_survivor",
                        logical_outcome_id=logical_outcome_id,
                        status=status,
                        session_id=str(getattr(child, "session_id", "") or "") or None,
                        conversation_id=conversation_id,
                        run_id=str(run_id) if run_id else None,
                        assistant_message_id=(
                            str(getattr(child, "assistant_message_id", "") or "")
                            or None
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 - child settlement is authoritative
                    self._set_degraded("publish_fleet", exc)
                    incomplete_operation = "publish_fleet"
                    complete = False
                    continue
                published.append(activity_id)
            reloaded = self._reload_locked()
            if complete and reloaded:
                self._state = "ready"
                if eligible_children:
                    # A complete replay has now represented every survivor as
                    # a durable receipt, which supersedes the coarse debt.
                    complete = self._clear_fleet_fallback_locked(conversation_id)
                if complete:
                    complete = self._reconcile_fleet_marks_locked()
            if not complete or not reloaded:
                # A partial receipt set must never reconcile away the one
                # compatibility signal that still tells the user news exists.
                self._set_fleet_fallback_locked(conversation_id)
                self.ensure_fleet_mark(conversation_id)
                self._set_degraded(
                    incomplete_operation if not complete else "fleet_reload"
                )
            return FleetReceiptPublication(
                activity_ids=tuple(published), complete=complete and reloaded
            )

    def acknowledge(self, activity_ids: Sequence[str]) -> int:
        """Acknowledge exact frozen IDs and return durably confirmed count."""
        with self._lock:
            requested = tuple(
                dict.fromkeys(
                    str(activity_id).strip()
                    for activity_id in activity_ids
                    if str(activity_id).strip()
                )
            )
            if not requested:
                return 0
            try:
                updated = self._db.acknowledge_console_activity(requested)
            except Exception as exc:  # noqa: BLE001 - keep unseen state on uncertainty
                self._set_degraded("acknowledge", exc)
                return 0
            if not self._reload_locked():
                return updated
            self._state = "ready"
            self._reconcile_fleet_marks_locked()
            remaining = {receipt.activity_id for receipt in self._snapshot}
            return sum(activity_id not in remaining for activity_id in requested)

    def reconcile_fleet_marks(self) -> None:
        """Reconcile coarse FLEET marks without acknowledging receipts."""
        with self._lock:
            self._reconcile_fleet_marks_locked()
