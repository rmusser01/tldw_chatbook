"""Handler for scheduled `automation_definition` rows (schedules-handoff §7.2).

Local-owner, `lifecycle=configured` automation definitions with a
`next_run_at` feed `PriorityQueue.load` as real rows, keyed by `"type":
"automation_definition"` -- one handler instance registers in
`SchedulerLoop.handlers` for all of them, dispatching to a family-keyed
executor registry (`agent_task` drops in a second registry entry later;
this module never grows a second handler class for it).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from tldw_chatbook.Scheduling.constants import HANDLER_TIMEOUT_SECONDS, coerce_positive_float
from tldw_chatbook.Scheduling.schedule_compute import compute_next_run_at, schedule_slot_for
from tldw_chatbook.Scheduling.slot_keys import canonical_hash

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.Notifications.notification_dispatch_service import (
        NotificationDispatchService,
    )
    from tldw_chatbook.Scheduling.automation_execution import ExecutionOutcome
    from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB

#: One definition row in, one `ExecutionOutcome` out (see
#: `automation_execution.ExecutionOutcome`, imported lazily below so this
#: module's own top level never carries the Library seams the real
#: implementation pulls in). `Any` here, not the real dataclass -- the
#: alias must be evaluable at import time even though the class it
#: describes is only ever imported inside the spawned coroutine.
Executor = Callable[[Any, dict], Awaitable[Any]]

#: `NotificationKind` values this handler can emit, per terminal run
#: status (server `NOTIFICATION_KIND_BY_STATUS` parity,
#: `tldw_api/notifications_reminders_schemas.py`).
_NOTIFICATION_KIND_BY_STATUS = {
    "completed": "automation_run_succeeded",
    "failed": "automation_run_failed",
    "timed_out": "automation_run_timed_out",
    "skipped": "automation_run_skipped",
}

#: (notification_policy key, default when the key is absent, severity) per
#: status. `on_failure` gates both `failed` and `timed_out` -- a timeout is
#: a failure from the notification policy's point of view. `on_skip`
#: defaults False (server parity: skipped runs are silent unless asked
#: for), the only status here whose default is not True.
_NOTIFICATION_GATE_BY_STATUS = {
    "completed": ("on_success", True, "information"),
    "failed": ("on_failure", True, "warning"),
    "timed_out": ("on_failure", True, "warning"),
    "skipped": ("on_skip", False, "information"),
}


def _parse_next_run_at(value: Any) -> datetime | None:
    """Parse a definition row's `next_run_at` into an aware UTC datetime.

    Same naive-means-UTC discipline as `schedule_compute._naive_as_utc`.
    Junk or missing input yields `None` rather than raising -- a malformed
    row must not kill the queue.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


async def _execute_recurring_question(app: Any, definition_row: dict) -> "ExecutionOutcome":
    """Default `recurring_question` executor: a lazy-import wrapper.

    `automation_execution` imports two Library RAG seams at its own top
    level -- heavier than the boot-census ratchet (ADR-097) allows this
    handler module to carry. The import happens here, inside the
    coroutine the scheduler loop never awaits directly (it is only ever
    reached from inside `AutomationDefinitionHandler._run`, itself only
    reached from a spawned `asyncio.Task`), so it costs nothing until a
    `recurring_question` definition actually fires.
    """
    from tldw_chatbook.Scheduling.automation_execution import execute_recurring_question

    return await execute_recurring_question(app, definition_row)


class AutomationDefinitionHandler:
    """Fire-and-forget scheduled automation-definition execution.

    Follows `BriefingJobHandler`'s spawn shape (spec §7.2, design-review
    correction: not optional). `SchedulerLoop.tick` awaits every handler
    serially, inline (briefings phase-4 locked decision), and a
    `recurring_question` run is RAG retrieval plus an LLM call -- awaiting
    it here would stall every other due task behind whichever provider is
    slowest. `handle` therefore does only synchronous, in-memory-cheap
    work (family lookup, the claim check, the run-row insert, and
    advancing the schedule) and spawns the actual execution as an
    independent `asyncio.Task`, returning before that task has had a
    chance to run at all.

    Claim guard (briefing precedent, spec §7.2 "Overlap guard"): `_claimed`
    holds the definition ids with a run currently in flight, from just
    before the `running` row is inserted (`handle`) until the spawned
    task's own `finally` releases it (`_run`) -- so an interval shorter
    than the run's own duration degrades to back-to-back runs, never
    concurrent ones. `_pending` is the strong-reference set that keeps a
    spawned task alive (a bare `asyncio.create_task` result with nothing
    else holding it is only weakly referenced by the event loop); the name
    is pinned by a later end-to-end test that awaits it directly to drain
    in-flight runs.
    """

    def __init__(
        self,
        db: "ScheduledTasksDB",
        app_getter: Callable[[], Any] | None = None,
        dispatch_service: "NotificationDispatchService | None" = None,
        handler_timeout_seconds: float | None = None,
        executors: dict[str, Executor] | None = None,
    ) -> None:
        """Initialize the handler.

        Args:
            db: The `ScheduledTasksDB` automation runs/results/definitions
                are read from and written to.
            app_getter: Zero-arg getter for the running app, resolved
                fresh per execution (`ReminderHandler`/`BriefingJobHandler`
                late-binding discipline) and passed both to the executor
                and to `dispatch_service.dispatch`.
            dispatch_service: Notification seam. `None` (the default)
                makes every notification a no-op -- headless/tests.
            handler_timeout_seconds: Per-handler execution timeout
                override; `None` (the default) falls back to the module
                default `HANDLER_TIMEOUT_SECONDS`, itself disabled by a
                zero/negative config value (`coerce_positive_float`
                discipline, `SchedulerLoop.__init__` precedent).
            executors: Family-keyed executor registry. `None` (the
                default) registers only `"recurring_question"`, wired to a
                lazy-import wrapper around Task 3's
                `execute_recurring_question` -- so `agent_task` can add a
                second entry later without touching this handler's shape.
                Tests inject fakes here directly.
        """
        self.db = db
        self.app_getter = app_getter
        self.dispatch_service = dispatch_service
        self.handler_timeout_seconds = coerce_positive_float(
            handler_timeout_seconds
            if handler_timeout_seconds is not None
            else HANDLER_TIMEOUT_SECONDS,
            HANDLER_TIMEOUT_SECONDS,
            allow_zero=True,
        )
        self.executors: dict[str, Executor] = (
            dict(executors)
            if executors is not None
            else {"recurring_question": _execute_recurring_question}
        )
        #: Definition ids with a run currently claimed -- see class
        #: docstring for the exact window this covers.
        self._claimed: set[str] = set()
        #: Strong references to spawned run tasks -- see class docstring.
        self._pending: set[asyncio.Task[Any]] = set()

    async def handle(self, task: dict[str, Any]) -> None:
        """Process one scheduled `automation_definition` row.

        `task` is the definition row dict (Task 5's queue projection adds
        `"type": "automation_definition"`, not consulted here -- the loop
        already routed on it). Every DB write below runs through
        `asyncio.to_thread` (`dispatch_reminder`'s own convention for this
        loop), so `handle` itself never blocks the event loop even though
        it is all synchronous, in-order work.

        Args:
            task: The definition row being dispatched.
        """
        definition_id = task.get("id")
        family = task.get("family")
        executor = self.executors.get(family)
        if executor is None:
            logger.warning(
                f"No executor registered for automation family {family!r} "
                f"(definition {definition_id!r}); skipping dispatch."
            )
            return

        next_run = _parse_next_run_at(task.get("next_run_at"))
        if next_run is None:
            logger.warning(
                f"Automation definition {definition_id!r} has no usable "
                f"next_run_at; skipping dispatch."
            )
            return
        slot = schedule_slot_for(next_run)

        await self._dispatch(
            task,
            executor=executor,
            trigger_reason="scheduled",
            schedule_slot=slot,
            advance_schedule=True,
        )

    async def run_now(self, definition_row: dict[str, Any]) -> str | None:
        """Dispatch one definition immediately (manual "Run now", Task 6).

        Reuses `handle`'s claim/spawn machinery via `_dispatch`, with
        `trigger_reason="manual"` and `schedule_slot=None` -- manual runs
        never slot-collide (NULL is distinct from every other value in the
        run's `(definition_id, definition_version, schedule_slot)`
        UNIQUE, so a manual run never dedupes against, or is deduped by, a
        scheduled one) -- and no schedule advance: a manual run does not
        consume or move the definition's next scheduled occurrence.

        The service seam (`SchedulingService.run_automation_now`) has
        already applied the owner/lifecycle/transfer/health refusals
        before calling this; the only refusal left here is the same
        overlap claim `handle` itself enforces.

        Args:
            definition_row: The definition row, shaped like `handle`'s
                `task` argument (a `"next_run_at"` is not required -- a
                manual run needs no slot).

        Returns:
            The new run's id, or `None` when a run was already claimed for
            this definition -- the skip is recorded exactly as `handle`'s
            own overlap guard records it (`status="skipped"`,
            `trigger_reason="manual"`).
        """
        definition_id = definition_row.get("id")
        family = definition_row.get("family")
        executor = self.executors.get(family)
        if executor is None:
            logger.warning(
                f"No executor registered for automation family {family!r} "
                f"(definition {definition_id!r}); skipping manual dispatch."
            )
            return None

        return await self._dispatch(
            definition_row,
            executor=executor,
            trigger_reason="manual",
            schedule_slot=None,
            advance_schedule=False,
        )

    async def _dispatch(
        self,
        task: dict[str, Any],
        *,
        executor: Executor,
        trigger_reason: str,
        schedule_slot: str | None,
        advance_schedule: bool,
    ) -> str | None:
        """Claim, insert the `running` row, spawn the run -- the shared post-claim body.

        Extracted from `handle` (Task 6): `handle` calls this with
        `trigger_reason="scheduled"`, a real `schedule_slot`, and
        `advance_schedule=True`; `run_now` calls it with
        `trigger_reason="manual"`, `schedule_slot=None`, and
        `advance_schedule=False`. See the class docstring's "Claim guard"
        for the exact window `_claimed` covers.

        Returns the new run's id, or `None` when the definition already
        has a run claimed (a `skipped` row is written first, exactly as
        the pre-extraction `handle` wrote it) or the slot's UNIQUE deduped
        the insert.
        """
        definition_id = task.get("id")
        owner_id = task.get("owner_id") or "local"
        version = task.get("version") or 1

        if definition_id in self._claimed:
            # Not an error: another dispatch (scheduled or manual) fired
            # before the previous run finished. The claim -- held by that
            # still-running execution -- is the refusal.
            logger.info(
                f"Skipping automation definition {definition_id!r}: a run "
                f"is already in flight for it."
            )
            skipped_run_id = await asyncio.to_thread(
                self.db.create_automation_run,
                owner_id,
                definition_id,
                version,
                trigger_reason,
                status="skipped",
                outcome="none",
                schedule_slot=None,
                run_summary={"skipped": "overlap", "claimed_slot": schedule_slot},
            )
            self._notify(
                task,
                run_id=skipped_run_id,
                status="skipped",
                outcome_value="none",
                message=(
                    f"{task.get('name') or 'Automation'} skipped: a run "
                    f"was already in progress."
                ),
            )
            return None

        self._claimed.add(definition_id)
        # From here to the successful spawn below, `_run`'s own `finally`
        # does not exist yet to release the claim -- and this span has two
        # real awaits (`create_automation_run`, `update_automation_
        # definition`), either of which can raise (DB busy/locked) or be
        # cancelled (the loop's per-handler `wait_for`). Without this
        # try/finally an exception here stranded the claim forever (the
        # definition became un-runnable until process restart) and, past
        # the insert, orphaned the `running` row. `claim_released_by_spawn`
        # is the only path that must NOT discard here: once `_run` is
        # actually spawned, IT owns the release (its own `finally`).
        claim_released_by_spawn = False
        try:
            now = datetime.now(timezone.utc)
            config = task.get("config")
            run_id = await asyncio.to_thread(
                self.db.create_automation_run,
                owner_id,
                definition_id,
                version,
                trigger_reason,
                status="running",
                schedule_slot=schedule_slot,
                started_at=now,
                scope_snapshot=config.get("scope") if isinstance(config, dict) else None,
                finding_policy_snapshot=task.get("finding_policy"),
            )
            if run_id is None:
                # The (definition, version, slot) UNIQUE fired: this slot
                # already ran. Dedupe is a result, not an error -- nothing
                # else to do; the finally below releases the claim.
                return None

            if advance_schedule:
                schedule = task.get("schedule")
                await asyncio.to_thread(
                    self.db.update_automation_definition,
                    definition_id,
                    next_run_at=compute_next_run_at(
                        schedule if isinstance(schedule, dict) else {}, now=now
                    ),
                )

            spawned = asyncio.create_task(
                self._run(executor, task, run_id=run_id, definition_id=definition_id, owner_id=owner_id),
                name=f"automation_run_{run_id}",
            )
            self._pending.add(spawned)
            spawned.add_done_callback(self._pending.discard)
            claim_released_by_spawn = True
            return run_id
        finally:
            if not claim_released_by_spawn:
                self._claimed.discard(definition_id)

    async def _run(
        self,
        executor: Executor,
        task: dict[str, Any],
        *,
        run_id: str,
        definition_id: str,
        owner_id: str,
    ) -> None:
        """Run one execution to completion, containing every failure.

        This coroutine is the whole body of the spawned task, so nothing
        it does may raise: an exception escaping a task nobody awaits
        becomes asyncio's "Task exception was never retrieved". The
        execution timeout (task-18939 semantics) wraps only the executor
        call, per `handler_timeout_seconds` (`<=0` disables the bound,
        `SchedulerLoop._effective_timeout_seconds` precedent).
        """
        try:
            app = self.app_getter() if self.app_getter is not None else None
            timeout = self.handler_timeout_seconds
            try:
                if timeout is not None and timeout > 0:
                    outcome: "ExecutionOutcome" = await asyncio.wait_for(
                        executor(app, task), timeout=timeout
                    )
                else:
                    outcome = await executor(app, task)
            except asyncio.TimeoutError:
                await asyncio.to_thread(
                    self.db.update_automation_run,
                    run_id,
                    status="timed_out",
                    outcome="degraded",
                    ended_at=datetime.now(timezone.utc),
                    failure_reason={"code": "execution_timeout"},
                )
                logger.warning(
                    f"Automation run {run_id} for definition "
                    f"{definition_id!r} timed out after {timeout}s"
                )
                self._notify(
                    task,
                    run_id=run_id,
                    status="timed_out",
                    outcome_value="degraded",
                    message=(
                        f"{task.get('name') or 'Automation'} timed out "
                        f"after {timeout}s."
                    ),
                )
                return
            except Exception as exc:  # noqa: BLE001 - must never escape uncaught
                logger.warning(
                    f"Automation run {run_id} for definition "
                    f"{definition_id!r} failed: {type(exc).__name__}"
                )
                await asyncio.to_thread(
                    self.db.update_automation_run,
                    run_id,
                    status="failed",
                    outcome="degraded",
                    ended_at=datetime.now(timezone.utc),
                    failure_reason={
                        "code": "execution_error",
                        "error_type": type(exc).__name__,
                    },
                )
                self._notify(
                    task,
                    run_id=run_id,
                    status="failed",
                    outcome_value="degraded",
                    message=(
                        f"{task.get('name') or 'Automation'} failed: "
                        f"{type(exc).__name__}."
                    ),
                )
                return

            await asyncio.to_thread(
                self.db.update_automation_run,
                run_id,
                status="completed",
                outcome=outcome.outcome,
                run_summary={"title": outcome.title, "summary": outcome.summary},
                evidence_summary=outcome.evidence_summary,
                failure_reason=outcome.failure_reason,
                ended_at=datetime.now(timezone.utc),
            )
            if outcome.outcome == "finding":
                dedupe_key = canonical_hash(
                    {
                        "definition_id": definition_id,
                        "run_id": run_id,
                        "kind": "finding",
                    }
                )
                await asyncio.to_thread(
                    self.db.create_automation_result,
                    owner_id,
                    definition_id,
                    run_id,
                    "finding",
                    outcome.title,
                    outcome.summary,
                    dedupe_key,
                    answer=outcome.answer,
                    answer_mode=outcome.answer_mode,
                    confidence=outcome.confidence,
                    source_refs=outcome.source_refs,
                )
            self._notify(
                task,
                run_id=run_id,
                status="completed",
                outcome_value=outcome.outcome,
                message=outcome.summary or outcome.title or "Completed.",
            )
        finally:
            self._claimed.discard(definition_id)

    def _notify(
        self,
        task: dict[str, Any],
        *,
        run_id: str | None,
        status: str,
        outcome_value: str,
        message: str,
    ) -> None:
        """Dispatch one notification for a terminal run status.

        No-op without a dispatch service; the `notification_policy` gate
        (`_NOTIFICATION_GATE_BY_STATUS`) ports the server's
        `_notification_enabled(definition, status)` semantics. Never
        raises -- a notification failure must never surface as a
        scheduling failure (same containment rule as
        `BriefingJobHandler._notify_result`).
        """
        if self.dispatch_service is None:
            return
        kind = _NOTIFICATION_KIND_BY_STATUS.get(status)
        gate = _NOTIFICATION_GATE_BY_STATUS.get(status)
        if kind is None or gate is None:
            return
        gate_key, gate_default, severity = gate
        policy = task.get("notification_policy")
        if not isinstance(policy, dict):
            policy = {}
        if not bool(policy.get(gate_key, gate_default)):
            return
        definition_id = task.get("id")
        try:
            app = self.app_getter() if self.app_getter is not None else None
            self.dispatch_service.dispatch(
                app=app,
                category="automation",
                title=str(task.get("name") or "Automation"),
                message=message,
                severity=severity,
                source_entity_kind="automation_definition",
                source_entity_id=definition_id,
                payload={"kind": kind, "run_id": run_id, "outcome": outcome_value},
            )
        except Exception as exc:  # noqa: BLE001 - must never escape uncaught
            logger.warning(
                f"Automation notification dispatch failed for definition "
                f"{definition_id!r}: {type(exc).__name__}"
            )

    async def __call__(self, task: dict[str, Any]) -> None:
        """Allow the handler to be invoked directly by the scheduler loop."""
        await self.handle(task)
