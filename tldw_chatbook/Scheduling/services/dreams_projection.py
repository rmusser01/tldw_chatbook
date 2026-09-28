"""Read-only projection of the [dreams] scheduled work as queue tasks.

Mirrors ``briefing_projection.py``'s shape and its two hard-won rules:

* **One id-shape definition point.** ``DREAMS_TASK_PREFIX`` /
  :func:`parse_dreams_task_id` and ``DREAMS_TRACK_PREFIX`` /
  :func:`parse_dream_track_task_id` are the ONLY places their id shapes
  are built and parsed; the handlers import from here rather than holding
  their own copies (the two-copies-drift lesson -- see
  ``BRIEFING_TASK_PREFIX``'s docstring for the incident).
* **Attempt-aware ``next_run_at``.** A completion watermark alone would pin
  the schedule to the last success forever after any later failure, so the
  queue reload would re-emit the job every ~30 minutes, uncapped. The
  Dreams form of the rule, per the phase-1 controller ruling:

  - completion watermark: ``MAX(completed_at)`` over
    ``status IN ('complete', 'partial')`` -- a failed or still-
    ``generating`` cycle never advances it;
  - attempt watermark: ``MAX(created_at)`` over ``status = 'failed'`` --
    a failed latest cycle retries one cadence after the ATTEMPT, not after
    the stale completion. Terminal successes are excluded because their
    ``created_at`` precedes their own ``completed_at`` (a cycle always
    finishes after it starts), so counting them could only push the next
    run later than the completion watermark already does;
  - rows still ``generating`` contribute to NEITHER watermark: the
    in-process cycle claim plus the stale-reclaim own a live cycle, and a
    ``generating`` row is either about to become terminal (advancing a
    watermark then) or a crashed run the next cycle's stale-reclaim
    converts to ``failed`` first;
  - ``next_run_at = max(completion, attempt) + cadence_hours``;
    never-attempted (both watermarks ``None``) is due NOW -- a freshly
    enabled Dreams fires on the next tick, not one cadence later.

Phase 2 (Track) adds one task per ACTIVE tracked item: ``dream_track:<id>``
whose ``next_run_at`` is ``last_checked + cadence_seconds`` with two clamps
-- a never-checked item is due immediately, and one overdue by more than
48 hours skips to ``now + cadence`` (a slept laptop must not fire a pileup
of stale checks on wake; spec's bounded catch-up). Paused and retired
items emit nothing.

Unlike ``BriefingProjection`` (one task per watchlist row), the cycle half
projects exactly ONE task: the single dated discovery cycle,
``dreams:cycle``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable

from tldw_chatbook.Dreams.settings import dreams_setting
from tldw_chatbook.Scheduling.models import ScheduledTask, TaskStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.DB.Dreams_DB import DreamsDB

#: The one place this project's dreams scheduled-task ids are built AND
#: parsed -- same single-definition-point discipline as
#: ``BRIEFING_TASK_PREFIX`` (see that constant's docstring for the drift
#: incident this avoids). ``dreams_handler.py`` imports this and
#: ``parse_dreams_task_id`` rather than holding its own copy.
DREAMS_TASK_PREFIX = "dreams"

#: The single projected job's task id (one cycle per cadence period).
DREAMS_CYCLE_TASK_ID = f"{DREAMS_TASK_PREFIX}:cycle"

#: The task type the scheduler routes to the dreams handler.
DREAMS_CYCLE_TASK_TYPE = "dreams_cycle"

#: Phase 2 (Track): the one place the ``dream_track:<item_id>`` id shape is
#: built AND parsed -- same single-definition-point discipline as
#: ``DREAMS_TASK_PREFIX``. ``dream_track_handler.py`` imports both from
#: here rather than holding its own copy. Deliberately NOT a
#: ``dreams:*`` id: ``parse_dreams_task_id`` must never accept one of
#: these, and vice versa.
DREAMS_TRACK_PREFIX = "dream_track"

#: The task type the scheduler routes to the track-check handler.
DREAMS_TRACK_TASK_TYPE = "dream_track_check"

#: Overdue clamp (spec's bounded catch-up): a check more than 48 hours past
#: its scheduled time skips the stale pileup and lands one cadence out.
_TRACK_OVERDUE_CLAMP = timedelta(hours=48)

#: Defensive floor for a tracked item's cadence. ``track_page`` /
#: ``track_question`` clamp to the configured minimum interval, but rows
#: can be inserted by other writers; a zero cadence would otherwise make
#: an overdue item due "now" forever, firing every tick.
_TRACK_CADENCE_FLOOR_SECONDS = 1


def parse_dreams_task_id(task_id: Any) -> str | None:
    """Extract the job name from a ``dreams:<job>`` task id.

    The one parser for this id shape -- see ``DREAMS_TASK_PREFIX``'s own
    docstring for why it lives here rather than being reimplemented in
    ``dreams_handler.py``.

    Args:
        task_id: The scheduled task's ``id`` field, expected to look like
            ``"dreams:cycle"``.

    Returns:
        The parsed job name, or ``None`` if ``task_id`` is not a string,
        has no ``:``, has the wrong prefix, or has an empty job name.
    """
    if not isinstance(task_id, str) or ":" not in task_id:
        return None
    prefix, job = task_id.split(":", 1)
    if prefix != DREAMS_TASK_PREFIX or not job:
        return None
    return job


def parse_dream_track_task_id(task_id: Any) -> int | None:
    """Extract the tracked-item id from a ``dream_track:<id>`` task id.

    The one parser for this id shape -- see ``DREAMS_TRACK_PREFIX``'s own
    docstring for the single-definition-point discipline. Deliberately
    rejects every ``dreams:*`` id (and vice versa) so a cycle task can
    never be routed to a track check.

    Args:
        task_id: The scheduled task's ``id`` field, expected to look like
            ``"dream_track:7"``.

    Returns:
        The tracked-item id (always ``> 0``), or ``None`` if ``task_id``
        is not a string, has no ``:``, has the wrong prefix, or does not
        carry a positive integer id.
    """
    if not isinstance(task_id, str) or ":" not in task_id:
        return None
    prefix, item = task_id.split(":", 1)
    if prefix != DREAMS_TRACK_PREFIX:
        return None
    try:
        tracked_item_id = int(item)
    except ValueError:
        return None
    return tracked_item_id if tracked_item_id > 0 else None


def _parse_iso_timestamp(value: str | datetime | None) -> datetime | None:
    """Normalize a stored timestamp to a timezone-aware datetime.

    ``DreamsDB`` writes timezone-aware ISO strings (``datetime.now(
    timezone.utc).isoformat()``); naive values are still tolerated the same
    way ``briefing_projection._parse_iso_timestamp`` tolerates them. Kept
    as its own small copy for the same reason that module keeps its own:
    the one-definition-point lesson is about the id prefix/parser contract,
    not a generic timestamp shim.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _read_watermarks(db: "DreamsDB") -> tuple[datetime | None, datetime | None]:
    """Read the completion and attempt watermarks (one read connection).

    Runs on whatever thread the caller is on; the queue's reload path
    already executes ``load()`` under ``asyncio.to_thread``.
    """
    with db.connection() as conn:
        completed = conn.execute(
            "SELECT MAX(completed_at) AS m FROM dreams_collections"
            " WHERE status IN ('complete', 'partial')"
        ).fetchone()["m"]
        attempted = conn.execute(
            "SELECT MAX(created_at) AS m FROM dreams_collections"
            " WHERE status = 'failed'"
        ).fetchone()["m"]
    return _parse_iso_timestamp(completed), _parse_iso_timestamp(attempted)


class DreamsProjection:
    """Project the Dreams discovery cycle as one cadence-driven task."""

    def __init__(
        self,
        db_getter: Callable[[], "DreamsDB | None"],
        *,
        cadence_hours: int | None = None,
    ) -> None:
        """Initialize the projection.

        Args:
            db_getter: Zero-arg callable returning the ``DreamsDB`` to read
                watermarks from, or ``None`` when no database is available
                (the projection then emits nothing). A GETTER, not the
                instance: ``app.py`` wires this during ``__init__`` and the
                same getter-lambda discipline as ``BriefingJobHandler``'s
                ``chachanotes_db_getter`` applies -- resolved fresh on
                every read, never frozen at wiring time.
            cadence_hours: Fixed cadence override; ``None`` (the default)
                reads ``[dreams] cadence_hours`` live on every ``tasks()``
                call so a settings change applies without a restart.
        """
        self._db_getter = db_getter
        self._cadence_hours = cadence_hours

    def tasks(self, now: datetime) -> list[ScheduledTask]:
        """Project the cycle task plus one task per active tracked item.

        Args:
            now: The "due now" instant used for a never-attempted cycle
                and a never-checked tracked item.

        Returns:
            ``[]`` when ``[dreams] enabled`` is false or no database is
            available; otherwise the ``dreams:cycle`` task (whose
            ``next_run_at`` follows the attempt-aware watermark rule in
            the module docstring) followed by one ``dream_track:<id>``
            task per ``status='active'`` tracked item under the clamps in
            the module docstring. Paused/retired items emit nothing.
        """
        if not dreams_setting("enabled"):
            return []
        db = self._db_getter()
        if db is None:
            return []
        current = now if now.tzinfo is not None else now.replace(
            tzinfo=timezone.utc
        )
        current = current.astimezone(timezone.utc)
        completed, attempted = _read_watermarks(db)
        last_activity = max(
            (
                value
                for value in (completed, attempted)
                if value is not None
            ),
            default=None,
        )
        if last_activity is None:
            next_run_at = current
        else:
            next_run_at = last_activity + timedelta(
                hours=int(
                    self._cadence_hours
                    if self._cadence_hours is not None
                    else dreams_setting("cadence_hours")
                )
            )
        projected = [
            ScheduledTask(
                id=DREAMS_CYCLE_TASK_ID,
                title="Dreams discovery cycle",
                type=DREAMS_CYCLE_TASK_TYPE,
                status=TaskStatus.WAITING,
                next_run_at=next_run_at,
                owner_id="local",
            )
        ]
        # Phase 2 (Track): one task per ACTIVE tracked item. The list read
        # runs on whatever thread the caller is on -- the queue's reload
        # path already executes ``load()`` under ``asyncio.to_thread``.
        for row in db.list_tracked_items("active"):
            projected.append(self._track_task(row, current))
        return projected

    @staticmethod
    def _track_task(row: dict, now: datetime) -> ScheduledTask:
        """One tracked item's projected task under the two clamps.

        ``next_run_at = last_checked + cadence`` with: a never-checked
        (or unparseable ``last_checked``) item due NOW, and an item
        overdue by more than 48 hours skipping to ``now + cadence`` (the
        bounded catch-up -- wake-from-sleep must not fire a stale pileup).
        """
        cadence = timedelta(seconds=max(
            _TRACK_CADENCE_FLOOR_SECONDS, int(row.get("cadence_seconds") or 0)))
        last_checked = _parse_iso_timestamp(row.get("last_checked"))
        if last_checked is None:
            next_run_at = now
        else:
            scheduled = last_checked + cadence
            if scheduled + _TRACK_OVERDUE_CLAMP < now:
                next_run_at = now + cadence
            else:
                next_run_at = scheduled
        title = "Dreams tracked check"
        template = str(row.get("query_template") or "").strip()
        if template:
            title = f"{title}: {template[:40]}"
        return ScheduledTask(
            id=f"{DREAMS_TRACK_PREFIX}:{int(row['id'])}",
            title=title,
            type=DREAMS_TRACK_TASK_TYPE,
            status=TaskStatus.WAITING,
            next_run_at=next_run_at,
            owner_id="local",
        )

    def list_jobs(
        self, owner_id: str = "local", *, now: datetime | None = None
    ) -> list[ScheduledTask]:
        """Queue-feed adapter: ``tasks()`` with the owner the queue stamps.

        ``PriorityQueue._append_projected`` consumes every projection
        through this signature (``watchlist_projection`` and
        ``briefing_projection`` both expose it), so the Dreams projection
        does too rather than the queue growing a second feed path.

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
