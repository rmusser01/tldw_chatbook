"""Local-first facade for scheduled task operations.

The ``SchedulingService`` is the single entry point used by the UI. It routes
reads and writes to the local ``ScheduledTasksDB`` cache, and prefers the server
API when a ``SchedulingServerClient`` is available and the current owner is a
server identity (``server:<user_id>``).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from croniter import croniter
from loguru import logger

from tldw_chatbook.Scheduling.automation_health import compute_local_health
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.models import (
    ReminderTask,
    ReviewState,
    ScheduleKind,
    ScheduledTask,
)
from tldw_chatbook.Scheduling.services.briefing_projection import BriefingProjection
from tldw_chatbook.Scheduling.services.server_client import (
    SchedulingServerClient,
    ServerUnavailableError,
)
from tldw_chatbook.Scheduling.services.sync_engine import SyncEngine
from tldw_chatbook.Scheduling.services.watchlist_projection import WatchlistProjection

_REMINDER_PRIMITIVE = "reminder_task"

#: Matches ScheduledTasksDB._RESULT_REVIEW_PRIMITIVE / SyncEngine's module
#: constant of the same value -- duplicated locally rather than imported,
#: mirroring how _REMINDER_PRIMITIVE is independently defined in each of
#: those modules too.
_RESULT_REVIEW_PRIMITIVE = "automation_result_review"

# Fields that are local-only and should not be sent to the server.
_LOCAL_ONLY_FIELDS = {
    "id",
    "server_id",
    "owner_id",
    "created_at",
    "updated_at",
    "sync_version",
}

# Fields accepted by ServerNotificationsService.create_reminder().
_REMINDER_SERVER_CREATE_FIELDS = {
    "title",
    "body",
    "schedule_kind",
    "run_at",
    "cron",
    "timezone",
    "enabled",
    "link_type",
    "link_id",
    "link_url",
}


class SchedulingService:
    """Facade for scheduling CRUD and sync operations.

    Args:
        db: The local scheduled-tasks database.
        server_client: Optional wrapper around the server reminder API.
        runtime_source: Initial owner identity; ``"local"`` or ``"server:<user_id>"``.
    """

    def __init__(
        self,
        db: ScheduledTasksDB,
        server_client: SchedulingServerClient | None = None,
        runtime_source: str = "local",
        watchlist_projection: WatchlistProjection | None = None,
        briefing_projection: BriefingProjection | None = None,
        on_queue_changed: Callable[[], None] | None = None,
        app_getter: Callable[[], Any] | None = None,
        automation_handler_getter: Callable[[], Any] | None = None,
    ) -> None:
        self.db = db
        self.server_client = server_client or SchedulingServerClient()
        self.runtime_source = runtime_source
        self.owner_id = runtime_source
        self.watchlist_projection = watchlist_projection
        self.briefing_projection = briefing_projection
        self.sync_engine = SyncEngine(db, self.server_client, self.owner_id)
        #: Zero-arg getter for the running app, used by
        #: ``run_automation_now`` to compute read-time health
        #: (``compute_local_health``). Late-binding like
        #: ``on_queue_changed`` below: the app wires this to ``lambda:
        #: self`` at construction time, before ``self`` itself is fully
        #: built out.
        self.app_getter = app_getter
        #: Zero-arg getter for the (lazily constructed, memoized)
        #: ``AutomationDefinitionHandler``. Injected rather than imported
        #: directly -- this module must not import the handler module
        #: (ADR-097 boot-census rule) -- and resolved fresh per call so a
        #: manual run reuses the SAME handler instance (and its
        #: ``_claimed``/``_pending`` overlap-guard state) as the scheduled
        #: dispatch path.
        self.automation_handler_getter = automation_handler_getter
        #: Called after any reminder mutation that can change what the
        #: scheduler should dispatch (create/update/delete, local or
        #: server-persisted). The app wires this to
        #: ``SchedulerLoop.request_reload`` so a reminder created mid-session
        #: reaches the live queue on the next tick instead of waiting for
        #: the periodic ~30-minute reload (task-18937). Kept optional and
        #: exception-guarded: a broken callback must never fail the mutation.
        self.on_queue_changed = on_queue_changed

    def _notify_queue_changed(self) -> None:
        """Invoke the queue-changed callback, tolerating a broken one.

        The exception log carries the owner and the callback's qualified
        name so a wiring failure can be correlated with the scheduler
        instance it affected (review finding: bare message, no context).
        """
        if self.on_queue_changed is None:
            return
        callback = self.on_queue_changed
        try:
            callback()
        except Exception:  # noqa: BLE001 - callback failure is not the caller's
            logger.exception(
                "Scheduling on_queue_changed callback failed for owner "
                "{owner} (callback {callback}); the mutation itself "
                "succeeded and the scheduler queue will reload on its "
                "periodic interval",
                owner=self.owner_id,
                callback=getattr(callback, "__qualname__", repr(callback)),
            )

    def set_owner(self, owner_id: str) -> None:
        """Switch the active owner and propagate it to the sync engine."""
        self.owner_id = owner_id
        self.sync_engine.owner_id = owner_id

    async def create_reminder(self, payload: dict[str, Any]) -> ReminderTask:
        """Create a reminder, preferring the server API when connected.

        If the server is unreachable or returns an error, the reminder is stored
        locally and a pending mutation is recorded so the sync engine can push it
        later.
        """
        task = ReminderTask(**payload)
        task.next_run_at = self._compute_next_run_at(task)
        server_payload = self._server_create_payload(task)
        db_fields = task.model_dump(
            exclude={
                "id",
                "server_id",
                "owner_id",
                "created_at",
                "updated_at",
                "sync_version",
            }
        )

        use_server = self._use_server()
        if use_server:
            assert self.server_client is not None
            try:
                response = await self.server_client.create_reminder(**server_payload)
                return await self._persist_server_reminder_response(response)
            except ServerUnavailableError:
                logger.warning(
                    f"Server unavailable while creating reminder for {self.owner_id}"
                )
            except Exception as exc:  # noqa: BLE001 - server errors should fall back
                logger.exception(
                    f"Server create_reminder failed for {self.owner_id}: {exc}"
                )

        task_id = self.db.create_reminder_task(owner_id=self.owner_id, **db_fields)
        if use_server:
            self.db.record_pending_mutation(
                task_id,
                _REMINDER_PRIMITIVE,
                self.owner_id,
                {"action": "create", "fields": server_payload},
            )
        self._notify_queue_changed()

        row = self.db.get_reminder_task(task_id)
        assert row is not None
        return self._row_to_reminder(row)

    async def list_reminders(self) -> list[ReminderTask]:
        """Return reminders for the current owner from the local cache."""
        rows = self.db.list_reminder_tasks(owner_id=self.owner_id)
        return [self._row_to_reminder(row) for row in rows]

    async def list_tasks(self) -> list[ReminderTask | ScheduledTask]:
        """Return reminders plus watchlist/briefing projections for the current owner."""
        tasks: list[ReminderTask | ScheduledTask] = list(await self.list_reminders())
        if self.watchlist_projection is not None:
            tasks.extend(self.watchlist_projection.list_jobs(owner_id=self.owner_id))
        if self.briefing_projection is not None:
            tasks.extend(self.briefing_projection.list_jobs(owner_id=self.owner_id))
        # Sort by next_run_at (None sorts last)
        tasks.sort(
            key=lambda t: t.next_run_at or datetime.max.replace(tzinfo=timezone.utc)
        )
        return tasks

    async def get_reminder(self, task_id: str) -> ReminderTask | None:
        """Fetch a single reminder by local id."""
        row = self.db.get_reminder_task(task_id)
        if row is None:
            return None
        return self._row_to_reminder(row)

    async def update_reminder(
        self, task_id: str, payload: dict[str, Any]
    ) -> ReminderTask | None:
        """Update a reminder, preferring the server API when connected.

        Falls back to a local update plus a pending mutation if the server is
        unavailable or returns an error.
        """
        row = self.db.get_reminder_task(task_id)
        if row is None:
            return None

        use_server = self._use_server()
        if use_server:
            assert self.server_client is not None
            server_id = row.get("server_id")
            try:
                if server_id:
                    response = await self.server_client.update_reminder(
                        server_id, **payload
                    )
                else:
                    merged_task = ReminderTask(**{**row, **payload})
                    merged_payload = self._server_create_payload(merged_task)
                    response = await self.server_client.create_reminder(
                        **merged_payload
                    )
                return await self._persist_server_reminder_response(
                    response, local_id=task_id
                )
            except ServerUnavailableError:
                logger.warning(
                    f"Server unavailable while updating reminder {task_id} for {self.owner_id}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    f"Server update_reminder failed for {task_id} ({self.owner_id}): {exc}"
                )

        # Local path: compute next_run_at and clear stale schedule fields
        # when the schedule is being changed.
        if any(key in payload for key in ("schedule_kind", "run_at", "cron", "timezone")):
            row_task = self._row_to_reminder(row)
            merged_data = row_task.model_dump()
            merged_data.update(payload)
            merged_task = ReminderTask(**merged_data)
            payload = dict(payload)
            if merged_task.schedule_kind == ScheduleKind.ONE_TIME:
                payload["cron"] = None
                payload["timezone"] = None
            elif merged_task.schedule_kind == ScheduleKind.RECURRING:
                payload["run_at"] = None
            payload["next_run_at"] = self._compute_next_run_at(merged_task)

        self.db.update_reminder_task(task_id, **payload)
        if use_server:
            self.db.record_pending_mutation(
                task_id,
                _REMINDER_PRIMITIVE,
                self.owner_id,
                {"action": "update", "fields": dict(payload)},
            )
        self._notify_queue_changed()

        row = self.db.get_reminder_task(task_id)
        assert row is not None
        return self._row_to_reminder(row)

    async def delete_reminder(self, task_id: str) -> bool:
        """Delete a reminder locally and on the server when connected.

        If the server is unavailable or returns an error, a tombstone is recorded
        so the delete can be pushed later.
        """
        row = self.db.get_reminder_task(task_id)
        if row is None:
            return False

        use_server = self._use_server()
        if use_server:
            assert self.server_client is not None
            server_id = row.get("server_id")
            try:
                if server_id:
                    await self.server_client.delete_reminder(server_id)
                self.db.delete_reminder_task(task_id)
                self.db.delete_sync_mapping(task_id, _REMINDER_PRIMITIVE, self.owner_id)
                self.db.delete_pending_mutation_for_record(
                    task_id, _REMINDER_PRIMITIVE, self.owner_id
                )
                self._notify_queue_changed()
                return True
            except ServerUnavailableError:
                logger.warning(
                    f"Server unavailable while deleting reminder {task_id} for {self.owner_id}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    f"Server delete_reminder failed for {task_id} ({self.owner_id}): {exc}"
                )

            if server_id is None:
                # No server copy exists; drop any stale pending mutation and
                # fall back to a local-only delete.
                self.db.delete_pending_mutation_for_record(
                    task_id, _REMINDER_PRIMITIVE, self.owner_id
                )

            self.db.record_tombstone(task_id, _REMINDER_PRIMITIVE, self.owner_id)
            self.db.delete_reminder_task(task_id)
            self.db.delete_pending_mutation_for_record(
                task_id, _REMINDER_PRIMITIVE, self.owner_id
            )
            self._notify_queue_changed()
            return True

        deleted = self.db.delete_reminder_task(task_id)
        if deleted:
            self._notify_queue_changed()
        return deleted

    async def sync_now(self, owner_id: str | None = None):
        """Trigger a full sync for the given owner (defaults to current owner).

        Returns the engine's ``SyncOutcome`` so callers can report what
        the sync actually did -- the engine swallows server errors into
        persisted sync-error state, so the return value is the only way
        to distinguish a failed sync from a no-op (task-23105 review F3).

        A successful sync can insert, update, and delete reminder rows the
        scheduler has already queued, so it fires ``on_queue_changed`` like
        every other mutation path (review finding: sync left the live queue
        stale until the ~30-minute periodic reload -- pulled reminders did
        not dispatch on time and remotely-deleted ones kept firing).
        """
        target_owner = owner_id if owner_id is not None else self.owner_id
        outcome = await self.sync_engine.sync_now(target_owner)
        self._notify_queue_changed()
        return outcome

    async def run_reminder_now(self, task_id: str, loop: Any = None) -> ReminderTask | None:
        """Dispatch a reminder immediately through the scheduler's own path.

        The service seam for the workbench's Run-now action (task-18938):
        it delegates to ``SchedulerLoop.run_reminder_now`` -- the SAME
        dispatch unit ``tick`` uses -- so a manual run is a real dispatch
        (recurring next occurrence persisted; one_time consumed), never a
        parallel code path. The task keeps its enabled/disabled state.

        Args:
            task_id: The reminder's local id.
            loop: The app's ``SchedulerLoop``. When omitted, manual dispatch
                is refused honestly (returned as ``None`` with a log line)
                rather than silently skipped -- without the loop there is no
                registered handler to run.

        Returns:
            The refreshed task after dispatch, or ``None`` when the task is
            missing, the loop/handler is unavailable, or the handler failed
            (the failure is already recorded on the task's ``last_status``
            by the dispatch seam).
        """
        if loop is None:
            logger.warning(
                "Manual reminder run refused for task {task_id}: no scheduler "
                "loop available",
                task_id=task_id,
            )
            return None
        row = self.db.get_reminder_task(task_id)
        if row is None:
            return None

        # ADR-077 decision 1: server-scoped rows are executed by the
        # server. The workbench surfaces a precise refusal message; the
        # loop carries the same guard for direct callers.
        from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

        if is_server_scoped_owner(row.get("owner_id")):
            logger.warning(
                "Manual reminder run refused for task {task_id}: "
                "server-scoped (executed by the server per ADR-077)",
                task_id=task_id,
            )
            return None

        succeeded = await loop.run_reminder_now(task_id)
        self._notify_queue_changed()

        row = self.db.get_reminder_task(task_id)
        if row is None or not succeeded:
            return None
        return self._row_to_reminder(row)

    async def run_automation_now(self, definition_id: str) -> dict[str, Any] | None:
        """Dispatch a local automation definition immediately (manual run).

        The service seam for the workbench's Run-now action on
        `automation_definition` rows (schedules-handoff PR-2 Task 6):
        delegates to ``AutomationDefinitionHandler.run_now`` -- the SAME
        claim/spawn machinery ``handle`` (the scheduled dispatch path)
        uses, with ``trigger_reason="manual"`` and no schedule slot -- so
        a manual run reuses the identical overlap guard and never
        double-executes against a concurrent scheduled run, and never
        advances the definition's next scheduled occurrence.

        Refuses (returns ``None``, reason logged) without dispatching for,
        in order: no automation handler wired (mirrors
        ``run_reminder_now``'s honest refusal when no loop is available),
        a server-scoped owner (ADR-077 decision 1: the server executes
        those), a ``lifecycle`` outside ``{configured, paused}``, a
        pending transfer (``transfer_state`` not ``NULL``), or a
        read-time health other than ``"ready"`` (``compute_local_health``
        -- never the possibly-stale ``health`` column).

        Args:
            definition_id: The definition's local id.

        Returns:
            ``None`` on refusal or when the definition does not exist.
            On success, ``{"run_id": ..., "deduped": bool}``: ``run_id``
            is ``None`` and ``deduped`` is ``True`` when the handler's own
            overlap claim declined the run (a run was already in flight
            for this definition) -- that is still a "success" from this
            method's own refusal checks above, just a no-op dispatch.
        """
        if self.automation_handler_getter is None:
            logger.warning(
                "Manual automation run refused for definition {definition_id}: "
                "no automation handler available",
                definition_id=definition_id,
            )
            return None

        row = await asyncio.to_thread(self.db.get_automation_definition, definition_id)
        if row is None:
            return None

        # ADR-077 decision 1: server-scoped rows are executed by the
        # server (same guard `run_reminder_now` applies to reminders).
        from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

        if is_server_scoped_owner(row.get("owner_id")):
            logger.warning(
                "Manual automation run refused for definition {definition_id}: "
                "server-scoped (executed by the server per ADR-077)",
                definition_id=definition_id,
            )
            return None

        if row.get("lifecycle") not in ("configured", "paused"):
            logger.warning(
                "Manual automation run refused for definition {definition_id}: "
                "lifecycle {lifecycle!r} is not configured/paused",
                definition_id=definition_id,
                lifecycle=row.get("lifecycle"),
            )
            return None

        if row.get("transfer_state") is not None:
            logger.warning(
                "Manual automation run refused for definition {definition_id}: "
                "a transfer is in progress",
                definition_id=definition_id,
            )
            return None

        app = self.app_getter() if self.app_getter is not None else None
        health, reason = compute_local_health(app, row)
        if health != "ready":
            logger.warning(
                "Manual automation run refused for definition {definition_id}: "
                "health is {health!r} ({reason})",
                definition_id=definition_id,
                health=health,
                reason=reason,
            )
            return None

        handler = self.automation_handler_getter()
        run_id = await handler.run_now(row)
        return {"run_id": run_id, "deduped": run_id is None}

    async def review_automation_result(
        self,
        result_id: str,
        review_state: str,
        review_note: str | None = None,
    ) -> bool:
        """Set a local automation result's review state (local entry point).

        Writes the local row via a single ``update_result_review`` call.
        When that row is a server mirror (``server_id`` set), the same
        call also records an ``automation_result_review`` pending
        mutation carrying the SERVER result id -- in the SAME DB
        transaction as the review UPDATE, so a crash between the two
        writes can never leave a local review that never pushes (or an
        outbox row for a review that was rolled back) -- so
        ``SyncEngine._replay_review_mutations`` can push it without a
        local join (spec §5.1's payload-not-reference rule). The
        mutation is recorded under the ROW's own ``owner_id`` (falling
        back to ``self.owner_id`` only if the row has none) since the
        workbench can toggle the service's active owner independently of
        which owner a given result row belongs to -- recording under
        ``self.owner_id`` would strand the mutation where
        ``get_pending_mutations`` for the row's real owner never sees it.
        Never notifies the queue: results don't arm anything the scheduler
        dispatches.

        Args:
            result_id: The result's local id.
            review_state: New review state; must be a valid
                ``ReviewState`` value or the call is refused.
            review_note: Optional free-text note attached to the review.

        Returns:
            ``True`` on a successful local write; ``False`` for an
            invalid ``review_state`` or an unknown ``result_id`` (no DB
            write in either case).
        """
        valid_states = {state.value for state in ReviewState}
        if review_state not in valid_states:
            logger.warning(
                "review_automation_result refused for {result_id}: invalid "
                "review_state {review_state!r} (must be one of {valid_states})",
                result_id=result_id,
                review_state=review_state,
                valid_states=sorted(valid_states),
            )
            return False

        row = await asyncio.to_thread(self.db.get_automation_result, result_id)
        if row is None:
            return False

        server_id = row.get("server_id")
        pending_mutation: dict[str, Any] | None = None
        if server_id:
            mutation_owner = row.get("owner_id") or self.owner_id
            pending_mutation = {
                "local_id": result_id,
                "primitive": _RESULT_REVIEW_PRIMITIVE,
                "owner_id": mutation_owner,
                "payload": {
                    "server_result_id": server_id,
                    "review_state": review_state,
                    "review_note": review_note,
                },
            }

        updated = await asyncio.to_thread(
            self.db.update_result_review,
            result_id,
            review_state,
            review_note,
            pending_mutation=pending_mutation,
        )
        return updated

    def _use_server(self) -> bool:
        """Return True when server operations should be attempted."""
        return self.server_client is not None and self.owner_id.startswith("server:")

    def _map_server_response_to_local(self, response: dict[str, Any]) -> dict[str, Any]:
        """Convert a server reminder response into local reminder-task fields."""
        local: dict[str, Any] = {}

        server_id = response.get("id")
        if server_id is not None:
            local["server_id"] = server_id

        for key in (
            "title",
            "body",
            "schedule_kind",
            "run_at",
            "cron",
            "timezone",
            "enabled",
            "last_status",
            "next_run_at",
            "last_run_at",
            "missed_at",
            "link_type",
            "link_id",
            "link_url",
            "created_at",
            "updated_at",
        ):
            if key in response:
                local[key] = response[key]

        return local

    def _server_create_payload(
        self,
        task: ReminderTask,
    ) -> dict[str, Any]:
        """Build the payload for ``ServerNotificationsService.create_reminder``.

        Only server-accepted fields are included, and datetimes are serialized to
        ISO-8601 strings so the server client receives the expected types.
        """
        payload: dict[str, Any] = {}
        for field in _REMINDER_SERVER_CREATE_FIELDS:
            value = getattr(task, field)
            if value is None:
                continue
            if isinstance(value, datetime):
                value = value.isoformat()
            payload[field] = value
        return payload

    async def _persist_server_reminder_response(
        self,
        response: dict[str, Any],
        local_id: str | None = None,
    ) -> ReminderTask:
        """Insert or update the local cache from a server reminder response."""
        local_fields = self._map_server_response_to_local(response)
        server_id = response.get("id")

        if local_id is not None:
            self.db.update_reminder_task(local_id, **local_fields)
            task_id = local_id
        else:
            existing = None
            if server_id:
                existing = self.db.get_reminder_task_by_server_id(
                    self.owner_id, server_id
                )
            if existing is not None:
                task_id = existing["id"]
                self.db.update_reminder_task(task_id, **local_fields)
            else:
                task_id = self.db.create_reminder_task(
                    owner_id=self.owner_id, **local_fields
                )

        if server_id:
            self.db.set_sync_mapping(
                task_id, server_id, _REMINDER_PRIMITIVE, self.owner_id
            )

        self.db.delete_pending_mutation_for_record(
            task_id, _REMINDER_PRIMITIVE, self.owner_id
        )
        self._notify_queue_changed()

        row = self.db.get_reminder_task(task_id)
        assert row is not None
        return self._row_to_reminder(row)

    def _compute_next_run_at(self, task: ReminderTask) -> datetime | None:
        """Compute the next scheduled run time for a reminder task.

        For one-time schedules the ``run_at`` value is returned directly. For
        recurring schedules the next cron occurrence after the current time in
        the task's timezone is computed and returned as UTC.
        """
        if task.schedule_kind == ScheduleKind.ONE_TIME:
            return task.run_at

        if task.schedule_kind == ScheduleKind.RECURRING:
            if not task.cron or not task.timezone:
                return None
            tz = ZoneInfo(task.timezone)
            now = datetime.now(tz)
            next_run = croniter(task.cron, now).get_next(datetime)
            return next_run.astimezone(timezone.utc)

        return None

    @staticmethod
    def _row_to_reminder(row: dict[str, Any]) -> ReminderTask:
        """Build a ``ReminderTask`` from a DB row.

        Removes ``None`` values for fields that have Pydantic defaults so the
        defaults are applied instead of failing validation.
        """
        data = dict(row)
        if data.get("last_status") is None:
            data.pop("last_status", None)
        if data.get("missed_count") is None:
            data.pop("missed_count", None)
        if data.get("timeout_seconds") is None:
            data.pop("timeout_seconds", None)
        return ReminderTask(**data)
