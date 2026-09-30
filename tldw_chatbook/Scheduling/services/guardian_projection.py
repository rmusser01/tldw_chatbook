"""Read-only projection of the [guardian] daily trend task as a queue task.

Mirrors ``dreams_projection.py``'s shape and its two hard rules:

* **One id-shape definition point.** ``GUARDIAN_TASK_PREFIX`` /
  :func:`parse_guardian_task_id` are the ONLY places the ``guardian:<job>``
  id shape is built and parsed; the handler imports from here rather than
  holding its own copy (the two-copies-drift lesson).
* **Watermark-owned ``next_run_at``.** The queue reloads every ~30 minutes
  and REPLACES projected tasks, so a due time derived from projection time
  would slide forward on every reload and never arrive (Qodo #6, PR
  #2890). The daily slot derives from the ``trends_last_run`` watermark
  the handler writes into ``guardian_meta`` after each run:
  ``next_run_at = watermark + 24h``, and a never-run (or unparseable)
  watermark is due NOW -- a freshly enabled Guardian fires on the next
  tick, not one cadence later.

Projection choice (brief: separate projection OR fold into an existing
daily projection, "the lighter against the queue's named-parameter rule"):
SEPARATE, because folding into ``dreams_projection`` would tie the
``[guardian] enabled`` gate and the Guardian store read to the Dreams
projection's early-return chain, while the queue seam already costs only
one more named parameter (``guardian_projection``) under
``PriorityQueue``'s established one-name-per-projection rule -- the same
increment Dreams itself made over briefing/watchlist.

Emits exactly ONE task: ``guardian:trends`` (spec §Trend analyzer, the
daily half of the trend cadence; local analysis, no LLM, no network,
budget-free). Emission is gated live on ``[guardian] enabled``; a
disabled Guardian projects nothing (ADR-204 contract 1).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from tldw_chatbook.Guardian.settings import guardian_setting
from tldw_chatbook.Scheduling.models import ScheduledTask, TaskStatus

#: The one place this project's guardian scheduled-task ids are built AND
#: parsed -- same single-definition-point discipline as
#: ``BRIEFING_TASK_PREFIX`` / ``DREAMS_TASK_PREFIX``.
GUARDIAN_TASK_PREFIX = "guardian"

#: The single projected job's task id (one trend run per day).
GUARDIAN_TRENDS_TASK_ID = f"{GUARDIAN_TASK_PREFIX}:trends"

#: The task type the scheduler routes to the guardian trend handler.
GUARDIAN_TRENDS_TASK_TYPE = "guardian_trends"

#: The ``guardian_meta`` key carrying the daily watermark (written by the
#: handler after each run; read here to derive the next slot).
TRENDS_LAST_RUN_META_KEY = "trends_last_run"

#: One trend run per day (spec §Trend analyzer, cadence (b)).
TRENDS_CADENCE = timedelta(hours=24)


def parse_guardian_task_id(task_id: Any) -> str | None:
    """Extract the job name from a ``guardian:<job>`` task id.

    The one parser for this id shape -- see ``GUARDIAN_TASK_PREFIX``'s
    docstring for why it lives here rather than in the handler.

    Args:
        task_id: The scheduled task's ``id`` field, expected to look like
            ``"guardian:trends"``.

    Returns:
        The parsed job name, or ``None`` if ``task_id`` is not a string,
        has no ``:``, has the wrong prefix, or has an empty job name.
    """
    if not isinstance(task_id, str) or ":" not in task_id:
        return None
    prefix, job = task_id.split(":", 1)
    if prefix != GUARDIAN_TASK_PREFIX or not job:
        return None
    return job


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    """Normalize a stored watermark to a timezone-aware datetime."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class GuardianProjection:
    """Project the Guardian daily trend run as one watermark-driven task."""

    def __init__(
        self,
        db_getter: Callable[[], Any],
        *,
        cadence_hours: int | None = None,
    ) -> None:
        """Initialize the projection.

        Args:
            db_getter: Zero-arg callable returning the
                :class:`~tldw_chatbook.DB.Guardian_DB.GuardianDB` to read
                the ``trends_last_run`` watermark from, or ``None`` when no
                store is available (the projection then emits nothing). A
                GETTER, not the instance -- resolved fresh on every read,
                never frozen at wiring time (same discipline as
                ``DreamsProjection``).
            cadence_hours: Fixed cadence override; ``None`` (the default)
                uses :data:`TRENDS_CADENCE` (24h).
        """
        self._db_getter = db_getter
        self._cadence = (
            timedelta(hours=int(cadence_hours))
            if cadence_hours is not None
            else TRENDS_CADENCE
        )

    def tasks(self, now: datetime) -> list[ScheduledTask]:
        """Project the single ``guardian:trends`` daily task.

        Args:
            now: The "due now" instant used when no watermark exists.

        Returns:
            ``[]`` when ``[guardian] enabled`` is false or no store is
            available; otherwise exactly one ``guardian:trends`` task whose
            ``next_run_at`` is the ``trends_last_run`` watermark plus the
            cadence (due NOW when never run).
        """
        if not guardian_setting("enabled"):
            return []
        db = self._db_getter()
        if db is None:
            return []
        current = now if now.tzinfo is not None else now.replace(
            tzinfo=timezone.utc
        )
        current = current.astimezone(timezone.utc)
        watermark = None
        try:
            watermark = _parse_iso_timestamp(
                db.get_meta(TRENDS_LAST_RUN_META_KEY)
            )
        except Exception:  # noqa: BLE001 - a broken read means due now
            watermark = None
        if watermark is None:
            next_run_at = current
        else:
            next_run_at = watermark.astimezone(timezone.utc) + self._cadence
        return [
            ScheduledTask(
                id=GUARDIAN_TRENDS_TASK_ID,
                title="Guardian daily trend analysis",
                type=GUARDIAN_TRENDS_TASK_TYPE,
                status=TaskStatus.WAITING,
                next_run_at=next_run_at,
                owner_id="local",
            )
        ]

    def list_jobs(
        self, owner_id: str = "local", *, now: datetime | None = None
    ) -> list[ScheduledTask]:
        """Queue-feed adapter: ``tasks()`` with the owner the queue stamps.

        ``PriorityQueue._append_projected`` consumes every projection
        through this signature, so the Guardian projection exposes it too.

        Args:
            owner_id: Owner to stamp on the emitted task.
            now: Injected clock; defaults to the current UTC time.

        Returns:
            ``tasks(now)`` with ``owner_id`` overridden.
        """
        current = now if now is not None else datetime.now(timezone.utc)
        return [
            task.model_copy(update={"owner_id": owner_id})
            for task in self.tasks(current)
        ]
