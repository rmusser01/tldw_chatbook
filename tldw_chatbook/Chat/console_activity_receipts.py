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
            rows = self._db.list_unseen_console_activity()
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
        required = {
            receipt.conversation_id
            for receipt in self._snapshot
            if receipt.origin == "fleet_survivor" and receipt.conversation_id
        }
        try:
            marked = set(self._marks.list_marked_conversation_ids(mark))
            for conversation_id in sorted(required - marked):
                self._marks.set_mark(conversation_id, mark)
            for conversation_id in sorted(marked - required):
                self._marks.clear_mark(conversation_id, mark)
        except Exception as exc:  # noqa: BLE001 - a stale badge is safer than a crash
            self._set_degraded("fleet_mark", exc)
            return False
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

    def publish_fleet_drain(self, event: Any) -> tuple[str, ...]:
        """Publish post-turn survivors from one stable FLEET drain event."""
        with self._lock:
            conversation_id = str(getattr(event, "conversation_id", "") or "")
            drain_id = str(getattr(event, "drain_id", "") or "")
            if not conversation_id or not drain_id:
                self._set_degraded("fleet_identity")
                return ()
            published: list[str] = []
            invalid_status = False
            for ordinal, child in enumerate(getattr(event, "children", ()) or ()):
                if getattr(child, "settled_after_turn", False) is not True:
                    continue
                raw_status = str(getattr(child, "status", "") or "")
                status = self._FLEET_STATUS.get(raw_status)
                if status is None:
                    invalid_status = True
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
                    continue
                published.append(activity_id)
            if self._reload_locked():
                self._state = "ready"
                self._reconcile_fleet_marks_locked()
            if invalid_status:
                self._set_degraded("fleet_status")
            return tuple(published)

    def acknowledge(self, activity_ids: Sequence[str]) -> int:
        """Acknowledge exact frozen receipt IDs and reconcile coarse marks."""
        with self._lock:
            try:
                updated = self._db.acknowledge_console_activity(activity_ids)
            except Exception as exc:  # noqa: BLE001 - keep unseen state on uncertainty
                self._set_degraded("acknowledge", exc)
                return 0
            if not self._reload_locked():
                return updated
            self._state = "ready"
            self._reconcile_fleet_marks_locked()
            return updated

    def reconcile_fleet_marks(self) -> None:
        """Reconcile coarse FLEET marks without acknowledging receipts."""
        with self._lock:
            self._reconcile_fleet_marks_locked()
