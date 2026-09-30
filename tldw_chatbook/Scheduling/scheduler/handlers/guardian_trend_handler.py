"""Handler for the scheduled [guardian] daily trend task.

The ``dream_track_handler`` spawn pattern (Locked Decision 3 discipline):
``handle`` does only synchronous, in-memory work (parse the id, resolve
deps) and spawns the run as an independent ``asyncio.Task``, so
``SchedulerLoop.tick`` never waits behind it -- even though the run is
local SQLite analysis with no LLM/network (budget-free, spec §Trend
analyzer), a store hiccup (WAL lock contention, a slow disk) must not
stall reminders, watchlist checks, briefings, and cycles.

What the run does, in order: ``analyze()`` (detection + capped trend-row
inserts), one ``NotificationDispatchService.dispatch(category="guardian",
...)`` per NEW trend notice, the retention sweep
(``prune_alerts(now - alert_retention_days)``), and finally the
``trends_last_run`` watermark write that pins tomorrow's projected slot.
The watermark is written LAST on purpose: a run that dies mid-way leaves
it untouched, so the next tick re-projects the task due and the cadence
self-heals.

Unlike ``dream_track_handler`` there is deliberately NO ``shutdown()``
seam: a spawned trend run touches only the local Guardian store, settles
in milliseconds, and app-exit teardown cancels the loop that owns it --
the dream seam exists for checks that can be waiting on an LLM or a
search engine mid-flight.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from loguru import logger

from tldw_chatbook.Scheduling.services.guardian_projection import (
    GUARDIAN_TRENDS_TASK_ID,
    parse_guardian_task_id,
)

#: Strong references to in-flight spawned trend runs. ``handle`` spawns
#: each run as a bare ``asyncio.Task`` (same reasoning as
#: ``dream_track_handler._SPAWNED_TRACK_CHECKS``: nothing else would hold
#: the task, and a bare ``create_task`` result is only weakly held by the
#: event loop and can be garbage-collected mid-flight). Entries are
#: discarded by a done callback.
_SPAWNED_TREND_RUNS: set[asyncio.Task[None]] = set()


@dataclass(frozen=True)
class GuardianTrendDeps:
    """The handles one scheduled trend run needs, resolved per dispatch.

    Attributes:
        guardian_db: The ``GuardianDB`` to analyze and prune. The deps
            getter returns ``None`` (no deps at all) rather than a bag
            with a broken store; ``None`` here still skips cleanly.
        dispatch_service: The ``NotificationDispatchService`` used for
            trend notices, or ``None`` to degrade to insert-without-
            delivery (the trend rows still land).
    """

    guardian_db: Any
    dispatch_service: Any | None = None


class GuardianTrendHandler:
    """Fire-and-forget scheduled Guardian trend analysis.

    ``deps_getter`` is a zero-arg GETTER resolved fresh on every dispatch
    -- never a bag captured at construction time -- because ``app.py``
    wires this handler before the handles it carries exist as attributes;
    a getter returning ``None`` (feature disabled, no Guardian store)
    simply skips that dispatch, and the queue's periodic reload re-emits
    the task for a later attempt.
    """

    def __init__(
        self, *, deps_getter: Callable[[], "GuardianTrendDeps | None"]
    ) -> None:
        """Initialize the handler.

        Args:
            deps_getter: Zero-arg callable returning a populated
                :class:`GuardianTrendDeps`, or ``None`` when the run must
                not happen right now.
        """
        self._deps_getter = deps_getter

    async def handle(self, task: dict[str, Any]) -> None:
        """Process one scheduled ``guardian:trends`` task.

        Never raises into the scheduler loop: an unparseable id or absent
        deps is logged and dropped, and the run itself lives inside a
        spawned task whose coroutine contains every failure.

        Args:
            task: Projected scheduled task dict from ``GuardianProjection``.
        """
        job = parse_guardian_task_id(task.get("id"))
        if job != "trends":
            logger.warning(
                f"Invalid guardian trend task id: {task.get('id')!r}"
                f" (expected {GUARDIAN_TRENDS_TASK_ID!r})"
            )
            return
        deps = self._deps_getter()
        if deps is None or deps.guardian_db is None:
            # Not an error: Guardian disabled (the projection would
            # normally not have emitted the task at all) or the store is
            # unavailable right now. The queue's reload re-emits the
            # task, so a later dispatch retries on its own.
            logger.info(
                "Skipping scheduled guardian trend run: no deps available."
            )
            return
        spawned = asyncio.create_task(
            self._run_daily_trends(deps), name="guardian_trends"
        )
        _SPAWNED_TREND_RUNS.add(spawned)
        spawned.add_done_callback(_SPAWNED_TREND_RUNS.discard)

    async def _run_daily_trends(self, deps: "GuardianTrendDeps") -> None:
        """Run one daily analysis to completion, containing every failure.

        This coroutine is the whole body of the spawned task, so nothing
        it does may raise: an exception escaping a task nobody awaits
        becomes asyncio's "Task exception was never retrieved". Logged
        with the exception's type name only, never a message that could
        embed alert content.

        ADR-097 (boot-census ratchet): the Guardian import chain stays off
        the boot path by importing ``analyze`` here, at first dispatch --
        this handler module is imported during ``app.py``'s
        post-``_ui_ready`` wiring, well after the module census.
        """
        try:
            from tldw_chatbook.Guardian.settings import guardian_setting
            from tldw_chatbook.Guardian.trend_analyzer import analyze
            from tldw_chatbook.Utils.timestamps import utc_now_iso

            db = deps.guardian_db
            now_iso = utc_now_iso()
            # Date-scoped visit id: the analyzer's per-visit frequency cap
            # then bounds trend notices to one per label per DAY, and a
            # new day is a fresh scope by construction.
            visit_id = f"trends:{now_iso[:10]}"
            notices = analyze(
                db, now=now_iso, visit_id=visit_id, session_id="daily"
            )
            dispatched = 0
            dispatch = getattr(deps.dispatch_service, "dispatch", None)
            for notice in notices:
                if not callable(dispatch):
                    break  # degrade: rows landed, delivery unavailable
                try:
                    dispatch(
                        category="guardian",
                        title=f"Guardian trend: {notice.label}",
                        message=notice.message,
                        severity="warning",
                    )
                    dispatched += 1
                except Exception as dispatch_exc:  # noqa: BLE001 - one denial stops nothing
                    logger.debug(
                        "Guardian trend notification delivery failed "
                        f"({type(dispatch_exc).__name__})"
                    )
            # Retention (contract 9): best-effort sweep at each daily run.
            retention_days = float(
                guardian_setting("alert_retention_days", 180) or 180
            )
            try:
                cutoff = (
                    datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
                    - timedelta(days=retention_days)
                ).isoformat()
                pruned = db.prune_alerts(cutoff)
                if pruned:
                    logger.info(
                        "Guardian retention pruned {} alert(s) at the "
                        "daily trend run",
                        pruned,
                    )
            except Exception:  # noqa: BLE001 - retention never blocks the run
                logger.opt(exception=True).debug(
                    "Guardian retention sweep failed"
                )
            # Watermark LAST: a run that dies above leaves it untouched
            # so the next tick re-projects the task and retries.
            db.set_meta("trends_last_run", now_iso)
            logger.info(
                "Scheduled guardian trend run finished: {} new notice(s), "
                "{} dispatched",
                len(notices),
                dispatched,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - must never escape uncaught
            logger.warning(
                "Scheduled guardian trend run failed: "
                f"{type(exc).__name__}"
            )

    async def __call__(self, task: dict[str, Any]) -> None:
        """Allow the handler to be invoked directly by the scheduler loop."""
        await self.handle(task)
