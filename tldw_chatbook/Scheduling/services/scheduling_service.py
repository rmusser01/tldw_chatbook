"""Local-first facade for scheduled task operations.

The ``SchedulingService`` is the single entry point used by the UI. It routes
reads and writes to the local ``ScheduledTasksDB`` cache, and prefers the server
API when a ``SchedulingServerClient`` is available and the current owner is a
server identity (``server:<user_id>``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field as _dataclass_field
from datetime import datetime, timedelta, timezone
from types import EllipsisType
from typing import Any, Callable
from zoneinfo import ZoneInfo

from croniter import croniter
from loguru import logger
from pydantic import ValidationError

# ADR-097 boot ratchet: automation_health loads on first use. A thin module-
# level proxy (not a plain deferred import) keeps `compute_local_health`
# patchable as an attribute of THIS module -- Tests/Scheduling/test_run_now.py
# stubs it here, and a function-local import would silently bypass the stub.
def compute_local_health(app, row):
    from tldw_chatbook.Scheduling.automation_health import (
        compute_local_health as _impl,
    )

    return _impl(app, row)

# ADR-097: automation_preview / automation_validation / schedule_compute are
# imported function-level in the authoring facade below -- this module is
# boot-resident and eager imports of the authoring stack breached the
# ui-ready census (975 > 972).
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import (
    DORMANT_TRANSFER_STATES,
    IN_FLIGHT_TRANSFER_STATES,
    ScheduledTasksDB,
)
from tldw_chatbook.Scheduling.models import (
    AutomationFamily,
    AutomationPreview,
    PreviewStatus,
    ReminderTask,
    ReviewState,
    ScheduleKind,
    ScheduledTask,
)
from tldw_chatbook.Scheduling.services.briefing_projection import BriefingProjection
from tldw_chatbook.Scheduling.services.server_client import (
    SchedulingServerClient,
    ServerClientError,
    ServerClientNotFoundError,
    ServerClientPolicyError,
    ServerClientValidationError,
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

#: Matches SyncEngine._DEFINITION_PRIMITIVE -- same duplication precedent
#: as _RESULT_REVIEW_PRIMITIVE above (this module is the producer of these
#: mutations, SyncEngine the consumer/replayer).
_DEFINITION_PRIMITIVE = "automation_definition"

#: v1 scope guard (schedules-handoff PR-4, task 4): only this family can be
#: authored through `preview_definition`/`save_definition`. `agent_task`
#: authoring rides a follow-up program -- see `_reject_unsupported_family`.
_SUPPORTED_AUTOMATION_FAMILY = AutomationFamily.RECURRING_QUESTION.value

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

#: spec §6.4: a one-time task whose `run_at` falls inside this window (or
#: has already passed) gets a `transfer_warnings` entry, not a refusal --
#: "server behavior on a past run_at is unverified" (flagged for live
#: verification, spec §10), so both directions of "close to firing" warn
#: the same way rather than trying to guess which side is safe.
_TRANSFER_IMMINENT_WINDOW = timedelta(minutes=5)

#: spec §6.3 cancel-table reason text -- module constants so
#: `cancel_refusal` (a side-effect-free preview) and `cancel_transfer`
#: (the actual mutation) share exactly one source of truth instead of
#: two copies of the same branching drifting apart (Task 7 fix round
#: finding 1: the UI used to re-derive this locally).
_CANCEL_TOO_LATE_REASON = "Too late to cancel -- start a reverse transfer instead."
_CANCEL_NOT_IN_PROGRESS_REASON = (
    "No transfer in progress on this row -- if it already moved, start a "
    "reverse transfer instead."
)

#: Spec §6.4's "a transfer is already in progress" refusal, shared by
#: `transfer_refusal`, `begin_transfer_to_server`'s CAS backstop and
#: `begin_transfer_to_local`'s queued-release guard (final review I5).
_TRANSFER_IN_PROGRESS_REASON = "A transfer is already in progress on this row."

#: Spec §6.3's read-only rule: while a transfer is in flight the row's
#: content is frozen (final review I7). `begin_transfer_to_server`
#: snapshots the create payload at begin time, so a later edit would ship
#: the PRE-edit content to the server and then be overwritten locally by
#: the first mirror pull -- silently discarding the user's edit.
_TRANSFER_READ_ONLY_REASON = (
    "This row is moving between this device and the server -- it is "
    "read-only until the move finishes. Cancel the transfer first."
)


@dataclass(slots=True)
class SaveDefinitionOutcome:
    """Result of `SchedulingService.save_definition` (Task 5's modal seam).

    Attributes:
        status: ``"saved"`` (committed locally, or to the server while
            online), ``"queued"`` (server owner, seam unreachable -- the
            local row was written and a mutation queued for the next
            sync), ``"invalid"`` (the preview rejected the payload;
            nothing was written), or ``"error"`` (an operational failure
            unrelated to payload validity, e.g. an unknown
            ``definition_id``).
        errors: Field-addressed validation errors (``{"field", "code",
            "message"}``), populated for ``"invalid"``/``"error"``.
        definition_id: The local row id, when one exists after the call.
            ``None`` only for ``"invalid"``/``"error"`` on a create (no
            row was ever written).
    """

    status: str
    errors: list[dict[str, Any]] = _dataclass_field(default_factory=list)
    definition_id: str | None = None


@dataclass(slots=True)
class TransferOutcome:
    """Result of a transfer-machine facade call (Task 6, spec §6.3/§6.4).

    Attributes:
        status: ``"pending"`` (armed the machine -- to_server_pending set,
            or a release's dormant local copy created), ``"cancelled"``
            (spec §6.3's cancel actually undid something), ``"refused"``
            (a `transfer_refusal` gate, or a losing CAS race, blocked the
            call -- see ``reason``), or ``"not_found"`` (``row_id`` does
            not exist).
        reason: Human-readable refusal text, populated for ``"refused"``.
        row_id: The dormant local copy's id, populated only by a
            successful ``begin_transfer_to_local`` -- Task 7's UI needs it
            to show/select the new row without a separate lookup.
    """

    status: str
    reason: str | None = None
    row_id: str | None = None


@dataclass(slots=True)
class ResolveOutcome:
    """Result of `SchedulingService.resolve_definition` (schedules-handoff
    PR-6 Task 2 -- definition-level mark-solved/reopen).

    Attributes:
        status: ``"saved"`` (written locally, or applied on the server and
            mirrored back) or ``"error"`` (nothing was written -- an
            unknown row, or -- plan ruling 2, a deliberate v1 gap -- a
            server-owned row whose server connection is unreachable or
            refused; there is no offline queue for this action yet).
        reason: Human-readable failure text, populated for ``"error"``.
    """

    status: str
    reason: str | None = None


@dataclass(slots=True)
class ReminderEditOutcome:
    """Result of `SchedulingService.edit_reminder_fields` (PR-3 task 3's
    reminder row editors).

    Gives the reminder side the same ``{field, code, message}`` validation
    surface `save_definition` already gives the definition side (survey
    §2's asymmetry: `update_reminder` alone raises a bare, uncaught
    `pydantic.ValidationError` on a bad schedule/timezone value and
    returns `None` -- with no per-field detail -- on a locked row).

    Attributes:
        status: ``"saved"`` (persisted, via `update_reminder`) or
            ``"error"`` (nothing written -- an unknown row, a locked
            row, or a validation failure).
        errors: Field-addressed validation errors (``{"field", "code",
            "message"}``), populated for ``"error"``. A locked or unknown
            row addresses the pseudo-field ``"_row"``.
        task: The updated `ReminderTask`, populated only for ``"saved"``.
    """

    status: str
    errors: list[dict[str, Any]] = _dataclass_field(default_factory=list)
    task: ReminderTask | None = None


def _seam_failure_warning(exc: Exception) -> dict[str, str]:
    """Build the ``_owner`` warning `preview_definition` appends on a seam failure.

    Not a `field_error` (those are validation errors, not warnings) --
    same ``{"field", "code", "message"}`` shape as `automation_preview.
    py`'s own warning entries, addressed to the pseudo-field ``"_owner"``
    since the failure is about the owner's server connection/permissions,
    not any one authoring field. `ServerClientPolicyError` (a deterministic,
    pre-network refusal) gets its own code/wording -- review round 1 finding
    3: telling a permanently-refused user "showing local validation only"
    with no other context reads as "try again once you're back online",
    which retrying will never fix.
    """
    if isinstance(exc, ServerClientPolicyError):
        return {
            "field": "_owner",
            "code": "policy_denied",
            "message": (
                f"The server refused this automation ({exc}); showing local "
                "validation only -- this will not resolve by retrying."
            ),
        }
    return {
        "field": "_owner",
        "code": "server_unreachable",
        "message": (
            f"Could not reach the server to preview this automation ({exc}); "
            "showing local validation only."
        ),
    }


def _server_refused_outcome(
    exc: Exception, definition_id: str | None
) -> SaveDefinitionOutcome:
    """Build the `SaveDefinitionOutcome` for a refusal that replaying cannot fix.

    Covers every `ServerClientValidationError` -- a local pre-network
    policy refusal (`ServerClientPolicyError`) AND any server-side 4xx
    (`server_client._call_with_retry` maps 400..499 except 404 to this
    class and never retries it): 409 `definition_version_conflict`, 409
    `definition_archived`, 422 `schedule_invalid`, ...

    None of these may fall back to the offline queue path (review round 1
    finding 1, final review C2): `SyncEngine._push_definition_create`/
    `_push_definition_update` would replay the identical request and hit
    the identical refusal forever, and for a policy refusal `_run_phase`
    even treats it as "not applicable" and swallows it silently -- no sync
    error recorded, mutation never cleared, never surfaced. A save this
    facade knows can never succeed must be reported as failed now, not
    queued as if it will eventually sync.

    Retryable failures (`ServerUnavailableError`/`ServerClientTimeoutError`
    /`ServerClientServerError`, and a 404 the sync engine converts to a
    create) still take the offline-queue path.
    """
    from tldw_chatbook.Scheduling.automation_validation import field_error

    code = (
        "policy_denied"
        if isinstance(exc, ServerClientPolicyError)
        else "server_rejected"
    )
    return SaveDefinitionOutcome(
        status="error",
        errors=[
            field_error(
                "_owner",
                code,
                f"The server refused this automation: {exc}",
            )
        ],
        definition_id=definition_id,
    )


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

    async def create_reminder(
        self, payload: dict[str, Any], *, owner_id: str | None = None
    ) -> ReminderTask:
        """Create a reminder, preferring the server API when connected.

        If the server is unreachable or returns an error, the reminder is stored
        locally and a pending mutation is recorded so the sync engine can push it
        later.

        Args:
            payload: `ReminderTask` fields to create.
            owner_id: Write this one reminder under a DIFFERENT owner than
                the service's active one (the form's "Runs on" selector);
                defaults to `self.owner_id`, so every existing caller is
                unchanged. Explicit rather than a `set_owner` flip around
                the awaited call: `owner_id` is shared mutable state that
                concurrent workers (sync, refresh, run-now) read, and a
                flip held across a network round-trip is visible to all of
                them. Mirrors `_owner_uses_server(owner_id)`, the same
                precedent `preview_definition`/`save_definition` use.
        """
        owner_id = owner_id or self.owner_id
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

        use_server = self._owner_uses_server(owner_id)
        if use_server:
            assert self.server_client is not None
            try:
                response = await self.server_client.create_reminder(**server_payload)
                return await self._persist_server_reminder_response(
                    response, owner_id=owner_id
                )
            except ServerUnavailableError:
                logger.warning(
                    f"Server unavailable while creating reminder for {owner_id}"
                )
            except Exception as exc:  # noqa: BLE001 - server errors should fall back
                logger.exception(
                    f"Server create_reminder failed for {owner_id}: {exc}"
                )

        task_id = self.db.create_reminder_task(owner_id=owner_id, **db_fields)
        if use_server:
            self.db.record_pending_mutation(
                task_id,
                _REMINDER_PRIMITIVE,
                owner_id,
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

    async def list_tasks(
        self,
        owner_id: str | None | EllipsisType = ...,
        *,
        include_projections: bool = True,
    ) -> list[ReminderTask | ScheduledTask]:
        """Return reminders plus (optionally) watchlist/briefing projections.

        Args:
            owner_id: Reminder owner scope. The default (``...``, i.e. the
                parameter is left unset) preserves every existing caller
                byte-for-byte: reminders scope to ``self.owner_id``, same
                as before this parameter existed. Pass ``None`` for a
                spans-owners listing -- every owner's reminders, via
                `ScheduledTasksDB.list_reminder_tasks`'s own
                ``owner_id=None`` "no WHERE clause" behavior (redesign
                PR-2's unified Queue list) -- or a specific owner id
                string to scope reminders to that one owner.
                Watchlist/briefing projections always stay scoped to
                ``self.owner_id`` regardless of this argument: their
                `list_jobs` only STAMPS the given owner id onto every row
                (their underlying read has no per-owner filter at all),
                so a ``None`` owner would fail their `ScheduledTask.
                owner_id: str` field instead of "spanning owners".
            include_projections: Default ``True`` preserves every existing
                caller byte-for-byte. ``False`` skips both `list_jobs`
                calls entirely -- redesign PR-2 Task 2's review, finding
                2: the unified Queue's `load_tasks` immediately filters
                every `ScheduledTask` row back out, so building AND
                sorting them was pure waste on that path (a full
                `Subscriptions_DB.get_all_subscriptions` scan plus a
                comparable briefing read, on every Queue refresh). The
                Queue passes ``include_projections=False``.

        Returns:
            Reminders (in the requested owner scope) plus, when
            ``include_projections`` is true, this device's watchlist/
            briefing projections -- sorted by ``next_run_at`` ascending
            (``None`` last).
        """
        reminder_owner_id = self.owner_id if owner_id is ... else owner_id
        rows = self.db.list_reminder_tasks(owner_id=reminder_owner_id)
        tasks: list[ReminderTask | ScheduledTask] = [
            self._row_to_reminder(row) for row in rows
        ]
        if include_projections:
            if self.watchlist_projection is not None:
                tasks.extend(
                    self.watchlist_projection.list_jobs(owner_id=self.owner_id)
                )
            if self.briefing_projection is not None:
                tasks.extend(
                    self.briefing_projection.list_jobs(owner_id=self.owner_id)
                )
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
        self, task_id: str, payload: dict[str, Any], *, owner_id: str | None = None
    ) -> ReminderTask | None:
        """Update a reminder, preferring the server API when connected.

        Falls back to a local update plus a pending mutation if the server is
        unavailable or returns an error.

        Args:
            task_id: The local reminder row to update.
            payload: The fields to change.
            owner_id: The owner this update belongs to; defaults to
                `self.owner_id`. Same rationale as `create_reminder`'s --
                threaded explicitly so a cross-owner save never has to flip
                the service's shared `owner_id` around an awaited call.
        """
        owner_id = owner_id or self.owner_id
        row = self.db.get_reminder_task(task_id)
        if row is None:
            return None

        # spec §6.3: in-flight rows are read-only except cancel. The UI
        # disables Edit/Enable/Disable with this same reason; this is the
        # backstop for every other caller (final review I7).
        locked = self.transfer_lock_reason(row)
        if locked is not None:
            logger.warning(
                "Reminder update refused for task {task_id}: {reason}",
                task_id=task_id,
                reason=locked,
            )
            return None

        use_server = self._owner_uses_server(owner_id)
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
                    response, local_id=task_id, owner_id=owner_id
                )
            except ServerUnavailableError:
                logger.warning(
                    f"Server unavailable while updating reminder {task_id} for {owner_id}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    f"Server update_reminder failed for {task_id} ({owner_id}): {exc}"
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
                owner_id,
                {"action": "update", "fields": dict(payload)},
            )
        self._notify_queue_changed()

        row = self.db.get_reminder_task(task_id)
        assert row is not None
        return self._row_to_reminder(row)

    async def edit_reminder_fields(
        self, task_id: str, payload: dict[str, Any]
    ) -> ReminderEditOutcome:
        """Validate and persist a single-row reminder edit (PR-3 task 3).

        Wraps `update_reminder` with the create form's own schedule
        validators (`schedule_input_parsing.parse_forgiving_datetime`/
        `is_valid_zone` -- the pure module `ReminderForm` itself also
        imports these from, task 3's folded-in refactor -- plus
        `croniter.is_valid` for cron) -- reused
        rather than re-derived, so a Repeat/At/Timezone row editor gets
        the same forgiving-local-datetime and known-zone behavior the
        create form gives, and the same errors when it doesn't parse.
        `update_reminder` alone does none of this: an invalid `cron`/
        `timezone` would silently persist (the `ReminderTask` model
        never validates their contents), and an invalid `run_at`/
        `schedule_kind` combination raises a bare `pydantic.
        ValidationError` with no field-addressed detail for a row editor
        to render.

        Args:
            task_id: The local reminder row to update.
            payload: The one or few fields the row editor is changing --
                same partial shape `update_reminder` already accepts.

        Returns:
            A `ReminderEditOutcome`. ``status="error"`` covers: an
            unknown ``task_id``, a locked (in-transfer) row (`errors`
            addresses the pseudo-field ``"_row"``, per
            `transfer_lock_reason`), or a bad schedule/timezone value
            (`errors` addresses the offending field). ``status="saved"``
            means `update_reminder` was called -- threading the ROW's
            OWN owner, not `self.owner_id` (PR-2's Queue lists reminders
            across owners, so the row under a cursor is not necessarily
            the service's active owner -- same rationale as
            `delete_reminder`'s) -- and returned the updated task.
        """
        from tldw_chatbook.Scheduling.automation_validation import field_error

        row = self.db.get_reminder_task(task_id)
        if row is None:
            return ReminderEditOutcome(
                status="error",
                errors=[field_error("_row", "not_found", f"Reminder {task_id} was not found.")],
            )

        locked = self.transfer_lock_reason(row)
        if locked is not None:
            return ReminderEditOutcome(
                status="error",
                errors=[field_error("_row", "transfer_in_progress", locked)],
            )

        # redesign PR-3, task 3's folded-in refactor (task-2-review.md
        # finding 1): these two used to live in `reminder_form.py`, a
        # UI-layer module this service-layer file had no business
        # importing from. Hoisted to a pure `Scheduling/`-side module --
        # still function-local (ADR-097 boot ratchet), though this one no
        # longer carries any Textual weight, so hoisting the IMPORT to
        # module level would also be safe; left function-local anyway for
        # a minimal diff against Task 2's shape.
        from tldw_chatbook.Scheduling.schedule_input_parsing import (
            is_valid_zone,
            parse_forgiving_datetime,
        )

        errors: list[dict[str, Any]] = []
        cleaned = dict(payload)
        if "run_at" in cleaned and isinstance(cleaned["run_at"], str):
            parsed, _assumed_local = parse_forgiving_datetime(cleaned["run_at"])
            if parsed is None:
                errors.append(
                    field_error(
                        "run_at",
                        "invalid_datetime",
                        "Run At must be a date and time like 2026-08-28 09:00.",
                    )
                )
            else:
                cleaned["run_at"] = parsed
        if "cron" in cleaned and cleaned["cron"] is not None:
            if not croniter.is_valid(cleaned["cron"]):
                errors.append(
                    field_error("cron", "invalid_cron", "Cron expression is invalid.")
                )
        if "timezone" in cleaned and cleaned["timezone"] is not None:
            new_zone = cleaned["timezone"]
            # Same stored-zone round-trip carve-out as the create form
            # (reminder_form.py's `_save`): a zone already on the row must
            # keep validating even if this machine's tzdata can't resolve
            # it -- only a genuinely NEW zone value is checked.
            if new_zone != row.get("timezone") and not is_valid_zone(new_zone):
                errors.append(
                    field_error(
                        "timezone", "invalid_timezone", f"Unknown timezone: {new_zone}"
                    )
                )

        if errors:
            return ReminderEditOutcome(status="error", errors=errors)

        try:
            task = await self.update_reminder(
                task_id, cleaned, owner_id=row.get("owner_id")
            )
        except ValidationError as exc:
            # Belt-and-suspenders: a schedule_kind/run_at/cron combination
            # the checks above don't cover (e.g. a one_time edit that
            # leaves run_at unset) still routes through `ReminderTask`'s
            # own model validation inside `update_reminder` -- catch that
            # surface here too so it never escapes as a raw exception.
            return ReminderEditOutcome(
                status="error",
                errors=[
                    field_error(
                        ".".join(str(part) for part in err["loc"]) or "_row",
                        str(err["type"]),
                        str(err["msg"]),
                    )
                    for err in exc.errors()
                ],
            )

        if task is None:
            # `update_reminder` refuses (returns None) if the row was
            # deleted or newly locked between the check above and this
            # call -- a narrow race, not a validation failure.
            return ReminderEditOutcome(
                status="error",
                errors=[
                    field_error(
                        "_row",
                        "update_refused",
                        "This reminder could not be updated -- it may have "
                        "just been deleted or locked by a transfer.",
                    )
                ],
            )

        return ReminderEditOutcome(status="saved", task=task)

    async def delete_reminder(
        self, task_id: str, *, owner_id: str | None = None
    ) -> bool:
        """Delete a reminder locally and on the server when connected.

        If the server is unavailable or returns an error, a tombstone is recorded
        so the delete can be pushed later.

        Args:
            task_id: The local reminder row to delete.
            owner_id: The owner this delete belongs to; defaults to
                `self.owner_id`. Same rationale (and same `_owner_uses_
                server` seam) as `create_reminder`/`update_reminder`'s --
                redesign PR-2's Queue lists reminders across owners, so
                "the row under the cursor" is no longer guaranteed to be
                the service's active owner. Deleting a `server:` row
                while the active owner was local previously took the
                local-only branch: no server call, no tombstone, and the
                row came back on the next pull (final review F4).
        """
        owner_id = owner_id or self.owner_id
        row = self.db.get_reminder_task(task_id)
        if row is None:
            return False

        # spec §6.3 (final review I7): deleting a row mid-transfer either
        # strands a live server task with no local trace, or -- for a
        # dormant release copy -- silently discards the only row the
        # release is about to arm. Cancel first.
        locked = self.transfer_lock_reason(row)
        if locked is not None:
            logger.warning(
                "Reminder delete refused for task {task_id}: {reason}",
                task_id=task_id,
                reason=locked,
            )
            return False

        use_server = self._owner_uses_server(owner_id)
        if use_server:
            assert self.server_client is not None
            server_id = row.get("server_id")
            try:
                if server_id:
                    await self.server_client.delete_reminder(server_id)
                self.db.delete_reminder_task(task_id)
                self.db.delete_sync_mapping(task_id, _REMINDER_PRIMITIVE, owner_id)
                self.db.delete_pending_mutation_for_record(
                    task_id, _REMINDER_PRIMITIVE, owner_id
                )
                self._notify_queue_changed()
                return True
            except ServerUnavailableError:
                logger.warning(
                    f"Server unavailable while deleting reminder {task_id} for {owner_id}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    f"Server delete_reminder failed for {task_id} ({owner_id}): {exc}"
                )

            if server_id is None:
                # No server copy exists; drop any stale pending mutation and
                # fall back to a local-only delete.
                self.db.delete_pending_mutation_for_record(
                    task_id, _REMINDER_PRIMITIVE, owner_id
                )

            self.db.record_tombstone(task_id, _REMINDER_PRIMITIVE, owner_id)
            self.db.delete_reminder_task(task_id)
            self.db.delete_pending_mutation_for_record(
                task_id, _REMINDER_PRIMITIVE, owner_id
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

        # spec §6.1 ruling 2: a row that has actually been sent to the
        # server (or is a dormant server-release copy) is not this side's
        # to run manually -- mirrors run_automation_now's same refusal.
        if row.get("transfer_state") in DORMANT_TRANSFER_STATES:
            logger.warning(
                "Manual reminder run refused for task {task_id}: "
                "a transfer is in progress",
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
        dormant transfer (``transfer_state`` in `DORMANT_TRANSFER_STATES`
        -- spec §6.1 ruling 2; a merely-queued or failed transfer keeps
        arming and does NOT refuse here), or a read-time health other than
        ``"ready"`` (``compute_local_health`` -- never the possibly-stale
        ``health`` column).

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

        if row.get("transfer_state") in DORMANT_TRANSFER_STATES:
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

    # ------------------------------------------------------------------
    # Transfer machine facade (schedules-handoff PR-5, Task 6, spec §6)
    # ------------------------------------------------------------------

    def _active_server_owner_id(self) -> str | None:
        """The single connected server's owner scope (``"server:<id>"``), or
        ``None`` when no server identity is currently resolved.

        Mirrors the workbench's own ``_active_server_id()``/`_runs_on_
        options` precedent (`schedules_workbench.py`): the app's
        `active_server_id` property, NOT ``self.owner_id`` -- `self.
        owner_id` is a UI-togglable VIEW (the user can flip to "This
        device" while a server stays connected), so it cannot stand in
        for "which server this session is connected to" (Task 4's own
        finding, carried forward here for the transfer machine's
        destination-owner rule).
        """
        app = self.app_getter() if self.app_getter is not None else None
        active_server_id = getattr(app, "active_server_id", None) if app is not None else None
        if not active_server_id:
            return None
        return f"server:{active_server_id}"

    def _get_transfer_row(self, table_kind: str, row_id: str) -> dict[str, Any] | None:
        if table_kind == _REMINDER_PRIMITIVE:
            return self.db.get_reminder_task(row_id)
        if table_kind == _DEFINITION_PRIMITIVE:
            return self.db.get_automation_definition(row_id)
        raise ValueError(f"Unknown table_kind for transfer: {table_kind!r}")

    def _delete_transfer_row(self, table_kind: str, row_id: str) -> bool:
        if table_kind == _REMINDER_PRIMITIVE:
            return self.db.delete_reminder_task(row_id)
        if table_kind == _DEFINITION_PRIMITIVE:
            return self.db.delete_automation_definition(row_id)
        raise ValueError(f"Unknown table_kind for transfer: {table_kind!r}")

    @staticmethod
    def _definition_transfer_payload(row: dict[str, Any]) -> dict[str, Any]:
        """Build a `transfer_to_server` mutation's `definition_payload`.

        Same CLIENT-vocabulary shape `_build_definition_request` produces
        for a create (minus `mode`/`definition_id`, which `SyncEngine`'s
        replay overrides itself) -- sourced straight from the row's own
        stored fields, the same way `_merge_definition_payload` treats an
        edit's base state. `SyncEngine._server_vocab_definition_payload`
        translates `schedule` to server vocabulary once, at push time.
        """
        return {
            "family": row.get("family"),
            "name": row.get("name"),
            "description": row.get("description"),
            "schedule": row.get("schedule") or {},
            "input": row.get("input") or {},
            "config": row.get("config") or {},
            "visibility_policy": row.get("visibility_policy") or {},
            "notification_policy": row.get("notification_policy") or {},
            "approval_policy": row.get("approval_policy") or {},
        }

    def transfer_refusal(self, row: dict[str, Any], direction: str) -> str | None:
        """Return why a transfer must be refused, or `None` when allowed.

        Spec §6.4's own priority order, so the ONE reason surfaced when
        several apply at once (e.g. an archived `agent_task` row) matches
        what the spec lists first: no server connection; ownership
        doesn't match `direction` (a `to_server` transfer needs a LOCAL
        row, a `to_local` release needs a server-owned mirror -- a
        structural prerequisite the spec's bullets assume, checked here
        alongside connection/identity since none of the bullets below are
        meaningful without it); no server identity resolved (`to_server`
        only -- a release already knows its destination from the mirror
        row's own `owner_id`); whether LOCAL can actually run the family
        (`to_local` only) -- `agent_task` always refuses in v1,
        `recurring_question` refuses when `compute_local_health` is not
        ``"ready"``, quoting its reason verbatim; a transfer already in
        progress (`row["transfer_state"]` in `{to_server_pending,
        to_server_sent, from_server_pending}` -- keyed off state, never
        mutation existence, same rule `cancel_transfer` follows;
        `to_server_failed` is deliberately EXCLUDED here --
        `begin_transfer_to_server` re-begins a failed transfer as a
        retry, spec obligation (f)); and lifecycle outside `{configured,
        paused}` (``archived``/``solved`` have nothing left to execute).

        Args:
            row: The reminder-task or automation-definition row to
                transfer, as returned by the DB layer.
            direction: ``"to_server"`` or ``"to_local"``. Any other value
                skips the direction-specific checks and is evaluated on
                the shared ones alone.

        Returns:
            The single highest-priority refusal reason as user-facing
            copy, or ``None`` when the transfer is allowed.
        """
        if (
            self.server_client is None
            or getattr(self.server_client, "notifications_service", None) is None
        ):
            return "No server connection is configured."

        owner_id = str(row.get("owner_id") or "")
        if direction == "to_server":
            if owner_id.startswith("server:"):
                return "This row already lives on the server."
            if self._active_server_owner_id() is None:
                return "No server identity is configured."
        elif direction == "to_local":
            if not owner_id.startswith("server:") or not row.get("server_id"):
                return "This row is not server-owned."

        if direction == "to_local":
            family = row.get("family")
            if family == "agent_task":
                return "Agent-task automations cannot run locally yet."
            if family == "recurring_question":
                app = self.app_getter() if self.app_getter is not None else None
                health, reason = compute_local_health(app, row)
                if health != "ready":
                    return reason

        if row.get("transfer_state") in IN_FLIGHT_TRANSFER_STATES:
            return _TRANSFER_IN_PROGRESS_REASON

        lifecycle = row.get("lifecycle")
        if lifecycle is not None and lifecycle not in ("configured", "paused"):
            return f"This automation is {lifecycle} and cannot transfer."

        return None

    @staticmethod
    def transfer_lock_reason(row: dict[str, Any]) -> str | None:
        """Why ``row`` is read-only right now, or `None` when it is editable.

        Spec §6.3: "dormant and in-flight rows are read-only except
        cancel". This is the ONE source of truth for that rule -- the
        facade's own edit/delete/enable-disable guards call it, and the UI
        calls it to disable those affordances with the same words
        (UX-073), rather than re-deriving the state set in two places
        (the drift `cancel_refusal` was introduced to stop).

        A `to_server_failed` row is NOT locked: it re-armed locally,
        nothing is queued against it, and editing before a retry is
        exactly what should be possible.

        Args:
            row: The reminder-task or automation-definition row to test.

        Returns:
            The read-only reason as user-facing copy when ``row``'s
            ``transfer_state`` is in `IN_FLIGHT_TRANSFER_STATES`, else
            ``None``.
        """
        if row.get("transfer_state") in IN_FLIGHT_TRANSFER_STATES:
            return _TRANSFER_READ_ONLY_REASON
        return None

    def transfer_warnings(self, row: dict[str, Any], direction: str) -> list[str]:
        """Non-blocking warnings for a transfer (spec §6.4).

        An imminent (or already past) one-time `run_at` warns rather than
        refuses -- "the transfer can outlive the moment, and server
        behavior on a past run_at is unverified". Reminders also warn
        about `timeout_seconds`, a local-only field that never transfers
        (definitions have no equivalent: their local-only `next_run_at`
        is expected to recompute, not silently dropped data).

        Args:
            row: The row being transferred. A ``family`` key identifies
                it as a definition; anything else is read as a reminder.
            direction: The transfer direction. Accepted for call-site
                symmetry with `transfer_refusal`; the warnings below
                depend on the row's own shape, not on which way it moves.

        Returns:
            Zero or more user-facing warning strings. Empty means nothing
            to confirm -- never a refusal, which is `transfer_refusal`'s
            job alone.
        """
        warnings: list[str] = []
        is_definition = "family" in row

        run_at_raw: Any = None
        if is_definition:
            schedule = row.get("schedule") or {}
            if isinstance(schedule, dict) and schedule.get("kind") == "one_time":
                run_at_raw = schedule.get("run_at")
        elif row.get("schedule_kind") == "one_time":
            run_at_raw = row.get("run_at")

        run_at = self._parse_transfer_run_at(run_at_raw)
        if run_at is not None:
            remaining = run_at - datetime.now(timezone.utc)
            if remaining <= _TRANSFER_IMMINENT_WINDOW:
                warnings.append(
                    "This one-time run fires within the next 5 minutes (or "
                    "has already passed); server behavior on a transfer "
                    "this close to run time is unverified."
                )

        if not is_definition and row.get("timeout_seconds") is not None:
            warnings.append(
                "The per-run timeout (timeout_seconds) is local-only and "
                "will not transfer."
            )

        return warnings

    @staticmethod
    def _parse_transfer_run_at(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    async def begin_transfer_to_server(self, table_kind: str, row_id: str) -> "TransferOutcome":
        """Start (or retry) a local -> server transfer (spec §6.1).

        Refuses via `transfer_refusal` first; then CASes `transfer_state`
        `None` OR `to_server_failed` -> `to_server_pending` (Task 1's
        compare-and-set, guarding against a concurrent second `begin` on
        the same row) and records a `transfer_to_server` mutation under
        the DESTINATION server scope (Task 4's finding: never the row's
        own -- still ``"local"`` -- `owner_id`). The row keeps executing
        locally while merely queued (spec §6.1.1) -- `SyncEngine`'s push
        disarms it later, only once an actual send attempt starts.

        The CAS and that mutation are ONE transaction (`set_transfer_
        state`'s `pending_mutation` kwarg, Qodo review fix wave 2). Two
        transactions left a crash window whose loser is silent and
        permanent: the row sits `to_server_pending` -- read-only per §6.3,
        excluded from the armable set -- with no outbox row for any
        replay to find, so it neither runs locally nor ever reaches the
        server. The mutation payload is therefore built BEFORE the CAS;
        nothing else about the ordering changed.

        `to_server_failed` -> `to_server_pending` is the RETRY leg
        (obligation (f)): `transfer_refusal` deliberately lets a
        definitively-failed row reach here, and `record_pending_mutation`
        is a plain `local_id`/`primitive`/`owner_id`-keyed upsert, so
        recording a FRESH payload (built the same way a first-time begin
        builds one, below) atomically replaces the retained mutation --
        stripping its `transfer_errors` for free rather than editing it
        in place. `_replay_definition_mutations`/`_network_phase`'s skip
        checks key off a TRUTHY `payload["transfer_errors"]` (not mere key
        presence), so this is what makes the row eligible for replay
        again.
        """
        row = self._get_transfer_row(table_kind, row_id)
        if row is None:
            return TransferOutcome(status="not_found", reason="No such row.")

        reason = self.transfer_refusal(row, "to_server")
        if reason is not None:
            return TransferOutcome(status="refused", reason=reason)

        destination_owner_id = self._active_server_owner_id()
        if destination_owner_id is None:
            return TransferOutcome(
                status="refused", reason="No server identity is configured."
            )

        if table_kind == _DEFINITION_PRIMITIVE:
            mutation_payload = {
                "action": "transfer_to_server",
                "definition_payload": self._definition_transfer_payload(row),
            }
        else:
            mutation_payload = {
                "action": "transfer_to_server",
                "task_payload": self._server_create_payload(self._row_to_reminder(row)),
            }

        armed = self.db.set_transfer_state(
            table_kind,
            row_id,
            "to_server_pending",
            expected=(None, "to_server_failed"),
            pending_mutation={
                "primitive": table_kind,
                "owner_id": destination_owner_id,
                "payload": mutation_payload,
            },
        )
        if not armed:
            return TransferOutcome(
                status="refused", reason=_TRANSFER_IN_PROGRESS_REASON
            )

        self._notify_queue_changed()
        return TransferOutcome(status="pending")

    async def begin_transfer_to_local(self, table_kind: str, row_id: str) -> "TransferOutcome":
        """Start a server -> local release (spec §6.2).

        Refuses via `transfer_refusal` first (which already confirms
        ``row`` is a server-owned mirror with a ``server_id``); then
        creates the dormant local copy (`create_local_copy_from_mirror`,
        Task 5) and records a `release_from_server` mutation keyed by the
        MIRROR's own ``local_id`` under the mirror's own ``owner_id`` --
        Task 5's documented convention, so its replay
        (`_push_definition_release`/`_push_reminder_release`) finds it.
        The mirror row itself is untouched and keeps executing
        server-side until the release actually acks.

        Refuses when a release is ALREADY queued for this mirror. Unlike
        every other in-progress check in this facade, that one cannot key
        off `transfer_state`: the release marks only the dormant COPY, and
        the mirror the user is pressing on carries no state at all. Two
        presses therefore built two copies while the second mutation
        REPLACED the first (same `(local_id, primitive, owner_id)` upsert
        key), stranding copy #1 dormant forever -- invisible to every
        armable query, named by no mutation (final review I5). Mutation
        existence is the only observable here, and it is safe as a
        REFUSAL: cancel stays state-keyed (Task 5 adjudication), so a
        definitively-failed release whose mutation is already cleared is
        still cancelable.

        That refusal is checked TWICE (Qodo review, fix wave 2). The
        check here is a cheap early exit; the authoritative one runs
        inside `create_local_copy_from_mirror`'s own transaction, which
        also writes the mutation. Splitting the copy and the mutation
        across two transactions reopened I5's exact strand by a different
        route -- a crash, or a second `begin` landing in the gap, left
        the copy with no mutation naming it -- and no amount of
        pre-checking outside the write closes that gap.
        """
        row = self._get_transfer_row(table_kind, row_id)
        if row is None:
            return TransferOutcome(status="not_found", reason="No such row.")

        reason = self.transfer_refusal(row, "to_local")
        if reason is not None:
            return TransferOutcome(status="refused", reason=reason)

        queued = self.db.get_pending_mutation_for_local_id(row_id, table_kind)
        if queued is not None and (queued.get("payload") or {}).get(
            "action"
        ) == "release_from_server":
            return TransferOutcome(
                status="refused", reason=_TRANSFER_IN_PROGRESS_REASON
            )

        owner_id = row["owner_id"]
        server_id = row["server_id"]
        server_field = (
            "server_definition_id"
            if table_kind == _DEFINITION_PRIMITIVE
            else "server_task_id"
        )
        copy_id = self.db.create_local_copy_from_mirror(
            table_kind,
            row_id,
            pending_mutation={
                "local_id": row_id,
                "primitive": table_kind,
                "owner_id": owner_id,
                # `local_copy_id` is filled in by the DB call itself --
                # the copy's id does not exist until its INSERT runs.
                "payload": {
                    "action": "release_from_server",
                    server_field: server_id,
                },
            },
        )
        if copy_id is None:
            # The in-transaction re-check found a release already queued
            # (a second `begin` that got past the pre-check above).
            return TransferOutcome(
                status="refused", reason=_TRANSFER_IN_PROGRESS_REASON
            )
        self._notify_queue_changed()
        return TransferOutcome(status="pending", row_id=copy_id)

    def _find_release_mutation(
        self, table_kind: str, local_copy_id: str
    ) -> dict[str, Any] | None:
        """Find a queued `release_from_server` mutation by its `local_copy_id`.

        The mutation's own `local_id` is the MIRROR's id (Task 5's
        keying convention), not the dormant copy's -- so a cancel called
        with the COPY's id (the only row that actually carries
        `from_server_pending`, per `cancel_transfer`'s docstring) has to
        search by the payload's nested field instead of a direct lookup.

        Scanned across ALL owners (final review C2). Scoping the scan to
        "today's active server" was the bug: with no server connected --
        the state a user is most likely to cancel in -- the lookup was
        skipped entirely, the copy was deleted, and the release mutation
        survived to delete the task server-side on the next reconnect.
        The mutation's own `owner_id` is the answer, not a guess, exactly
        as `get_pending_mutation_for_local_id` already established.
        """
        for mutation in self.db.get_pending_mutations(primitive=table_kind):
            payload = mutation.get("payload") or {}
            if (
                payload.get("action") == "release_from_server"
                and payload.get("local_copy_id") == local_copy_id
            ):
                return mutation
        return None

    def _delete_transfer_mutation(self, table_kind: str, row_id: str) -> None:
        """Drop the queued `transfer_to_server` mutation for ``row_id``.

        Read via `get_pending_mutation_for_local_id` -- the mutation's OWN
        `owner_id` column -- never via "today's active server". Guessing
        left the mutation behind on every offline or post-server-switch
        cancel: the state cleared, so the UI said cancelled, while the
        mutation sat in the queue forever, CAS-skipped each cycle and
        suppressing pull-apply for that row via `pending_local_ids`
        (final review C2/I3). Same lesson as the Task 7 retry-error
        lookup, applied to the write side.
        """
        mutation = self.db.get_pending_mutation_for_local_id(row_id, table_kind)
        if mutation is None:
            return
        if (mutation.get("payload") or {}).get("action") != "transfer_to_server":
            # Not this machine's mutation (a plain create/update edit on a
            # server-owned row) -- cancelling a transfer must not discard
            # the user's unrelated queued edit.
            return
        self.db.delete_pending_mutation(mutation["id"])

    def cancel_refusal(self, row: dict[str, Any]) -> str | None:
        """Preview whether `cancel_transfer` would refuse ``row``, without
        mutating anything (spec §6.3) -- the UI's Cancel-button disabled-
        reason source of truth (Task 7 fix round finding 1: the UI used
        to re-derive this same state branching locally with no shared
        source, risking silent drift if `cancel_transfer`'s branching
        ever changed).

        Mirrors `cancel_transfer`'s own branching exactly (same module
        constants back both), with one necessary gap: a losing
        compare-and-set race (a concurrent push disarming the row between
        this call and an actual `cancel_transfer`) is a live race, not
        something a row snapshot can predict -- same limitation
        `transfer_refusal` already has for its own CAS backstop.
        """
        state = row.get("transfer_state")
        if state in ("to_server_pending", "to_server_failed", "from_server_pending"):
            return None
        if state == "to_server_sent":
            return _CANCEL_TOO_LATE_REASON
        return _CANCEL_NOT_IN_PROGRESS_REASON

    async def cancel_transfer(self, table_kind: str, row_id: str) -> "TransferOutcome":
        """Cancel an in-progress transfer (spec §6.3 table, exactly).

        Keyed OFF ``row["transfer_state"]``, never off whether a pending
        mutation still exists: a release that definitively failed
        server-side settles by clearing its own mutation (same
        reject-and-clear every other definitive failure gets) but leaves
        the dormant copy's `from_server_pending` state untouched -- cancel
        must still recover that copy, so mutation absence cannot mean
        "nothing to cancel" here.

        - `to_server_pending` / `to_server_failed` (unattempted, or a
          settled definitive failure -- both re-armed locally, nothing
          sent): CAS to ``None``, drop the queued mutation, row stays
          local. A losing CAS (a concurrent push just disarmed it) is
          reported the same as `to_server_sent` below -- too late.
        - `to_server_sent`: too late -- refused, offering a reverse
          transfer once this one lands.
        - `from_server_pending` (the dormant COPY row -- unpushed release,
          or one that definitively failed): drop any live release
          mutation naming it, then delete the copy. Nothing further is
          sent. NOT the same as "no server-side effect": this state also
          covers a release whose delete already landed but whose ack was
          lost, and that delete cannot be undone from here (spec §6.3;
          the user-facing copy says "nothing further will be sent", not
          "nothing happened").
        - Anything else (``None`` -- never transferring, or a release
          that already acked and armed): too late -- refused, offering a
          reverse transfer.
        """
        row = self._get_transfer_row(table_kind, row_id)
        if row is None:
            return TransferOutcome(status="not_found", reason="No such row.")

        state = row.get("transfer_state")
        too_late = TransferOutcome(status="refused", reason=_CANCEL_TOO_LATE_REASON)

        if state in ("to_server_pending", "to_server_failed"):
            cleared = self.db.clear_transfer_state(
                table_kind, row_id, expected=("to_server_pending", "to_server_failed")
            )
            if not cleared:
                return too_late
            self._delete_transfer_mutation(table_kind, row_id)
            self._notify_queue_changed()
            return TransferOutcome(status="cancelled")

        if state == "from_server_pending":
            # The mutation goes FIRST: if the copy delete were to land
            # without it, the surviving release would still delete the
            # task server-side on the next sync, with no local copy left.
            mutation = self._find_release_mutation(table_kind, row_id)
            if mutation is not None:
                self.db.delete_pending_mutation(mutation["id"])
            self._delete_transfer_row(table_kind, row_id)
            self._notify_queue_changed()
            return TransferOutcome(status="cancelled")

        if state == "to_server_sent":
            return too_late

        return TransferOutcome(
            status="refused", reason=_CANCEL_NOT_IN_PROGRESS_REASON
        )

    #: Which `lifecycle` value each lifecycle action lands the row on.
    #: The action names are the server's own endpoint verbs and the
    #: `pending_mutations` payload actions `SyncEngine._push_definition_
    #: lifecycle` replays -- one table, so a rename cannot drift between
    #: the local write and the queued mutation.
    _LIFECYCLE_ACTIONS = {
        "pause": "paused",
        "resume": "configured",
        "archive": "archived",
    }

    async def set_definition_lifecycle(
        self, row_id: str, action: str
    ) -> SaveDefinitionOutcome:
        """Pause / resume / archive an automation definition.

        The missing PRODUCER for the `pause`/`resume`/`archive` pending
        mutations `SyncEngine._push_definition_lifecycle` (PR-4 Task 2)
        replays: until this existed, that replay leg -- and the four
        client methods under it -- had no caller at all outside the
        release leg's archive, so the whole seam was inert (final review
        M9).

        Local rows: a direct lifecycle write, no mutation (nothing to
        sync). Server-owned rows: the local row is updated optimistically
        and ONE lifecycle mutation is recorded in the SAME transaction
        (`update_automation_definition`'s `pending_mutation` kwarg), for
        the replay to push -- so a lifecycle change made offline survives
        and lands on the next sync, exactly like an offline edit.

        **No UI is wired to this yet, deliberately**: the Automations
        tab's pause/resume/archive affordances belong to the schedules
        redesign program, not to PR-5, whose scope is the transfer
        machine. This method exists so the replay leg below it is
        reachable and tested rather than dead code.

        Args:
            row_id: The definition's LOCAL row id.
            action: ``"pause"``, ``"resume"``, or ``"archive"``.

        Returns:
            `SaveDefinitionOutcome` -- ``"saved"`` for a local row or a
            server row written and queued, ``"error"`` for an unknown
            action, a missing row, or a row locked by an in-flight
            transfer.
        """
        from tldw_chatbook.Scheduling.automation_validation import field_error

        lifecycle = self._LIFECYCLE_ACTIONS.get(action)
        if lifecycle is None:
            return SaveDefinitionOutcome(
                status="error",
                errors=[
                    field_error(
                        "_lifecycle",
                        "unknown_action",
                        f"Unknown lifecycle action {action!r}.",
                    )
                ],
                definition_id=row_id,
            )

        row = await asyncio.to_thread(self.db.get_automation_definition, row_id)
        if row is None:
            return SaveDefinitionOutcome(
                status="error",
                errors=[
                    field_error(
                        "_definition",
                        "not_found",
                        f"Automation definition {row_id} was not found.",
                    )
                ],
                definition_id=row_id,
            )

        locked = self.transfer_lock_reason(row)
        if locked is not None:
            return SaveDefinitionOutcome(
                status="error",
                errors=[field_error("_transfer", "transfer_in_progress", locked)],
                definition_id=row_id,
            )

        owner_id = str(row.get("owner_id") or "local")
        pending_mutation = None
        if self._owner_uses_server(owner_id):
            pending_mutation = {
                "primitive": _DEFINITION_PRIMITIVE,
                "owner_id": owner_id,
                "payload": {
                    "action": action,
                    "server_definition_id": row.get("server_id"),
                },
            }

        await asyncio.to_thread(
            lambda: self.db.update_automation_definition(
                row_id,
                lifecycle=lifecycle,
                pending_mutation=pending_mutation,
            )
        )
        self._notify_queue_changed()
        return SaveDefinitionOutcome(status="saved", definition_id=row_id)

    async def resolve_definition(
        self, definition_id: str, solved: bool, result_id: str | None = None
    ) -> "ResolveOutcome":
        """Mark a definition solved, or reopen a solved one (schedules-handoff
        PR-6 Task 2, spec §4.3).

        Resolution is DEFINITION-level (plan ruling 2), not per-result:
        writes ``resolution_state``/``resolved_at``/``resolved_by``/
        ``resolved_result_id`` on the definition row, recording the
        triggering result id.

        Local rows write directly via `ScheduledTasksDB.set_definition_
        resolution` -- there is nothing to sync, same reasoning `set_
        definition_lifecycle` uses for local rows. Server-owned rows call
        the server's mark-solved/reopen endpoint IMMEDIATELY (unlike `set_
        definition_lifecycle`'s optimistic-write-plus-queue pattern) and
        mirror the echoed row back via `db.upsert_automation_definitions_
        from_server` on success -- ``result_id`` is translated from this
        row's LOCAL result id to the mirrored result's ``server_id``
        first, since the server has never heard of a local UUID. When the
        seam is unreachable or the server refuses, this returns an honest
        ``status="error"`` rather than queuing a mutation (plan ruling 2,
        a deliberate v1 gap): a solved/reopen mutation queued against a
        row whose true resolution state may already have changed
        server-side is a worse lie than an outright refusal, and there is
        no offline primitive for this action yet.

        Never notifies the queue: like `review_automation_result`,
        nothing in `list_armable_automation_definitions` reads the
        resolution columns, so a resolution change never changes what the
        scheduler should dispatch.

        Refuses (fix round 1) while `transfer_lock_reason` reports the row
        dormant/in-flight -- same "read-only except cancel" rule `save_
        definition`/`set_definition_lifecycle` already enforce (spec
        §6.3). Resolution fields are row-state exactly like `lifecycle`:
        a local write mid-transfer would be shipped by a create snapshot
        taken before this task existed, then silently clobbered back to
        `"open"` by the first mirror pull -- the identical corruption
        class I7 was written to close, just via a column its guard didn't
        cover yet.

        Args:
            definition_id: The definition's LOCAL row id.
            solved: ``True`` to mark solved, ``False`` to reopen.
            result_id: The LOCAL id of the result that triggered the
                resolution (mark-solved only; ignored for reopen).

        Returns:
            `ResolveOutcome` -- ``"saved"`` on a successful local write or
            server round trip, ``"error"`` (with ``reason``) for an
            unknown ``definition_id``, a row locked by an in-flight
            transfer, an unsynced result id, or an unreachable/refused
            server row.
        """
        row = await asyncio.to_thread(self.db.get_automation_definition, definition_id)
        if row is None:
            return ResolveOutcome(
                status="error",
                reason=f"Automation definition {definition_id} was not found.",
            )

        locked = self.transfer_lock_reason(row)
        if locked is not None:
            return ResolveOutcome(status="error", reason=locked)

        owner_id = str(row.get("owner_id") or "local")
        action_desc = "mark this definition solved" if solved else "reopen this definition"

        if not self._owner_uses_server(owner_id):
            updated = await asyncio.to_thread(
                self.db.set_definition_resolution,
                definition_id,
                state="solved" if solved else "open",
                result_id=result_id,
                resolved_by="local",
            )
            if not updated:
                return ResolveOutcome(
                    status="error",
                    reason=f"Automation definition {definition_id} was not found.",
                )
            return ResolveOutcome(status="saved")

        server_id = row.get("server_id")
        if not server_id:
            return ResolveOutcome(
                status="error",
                reason=f"Could not {action_desc}: this row has no server identity.",
            )

        assert self.server_client is not None
        server_result_id = None
        if solved and result_id:
            result_row = await asyncio.to_thread(
                self.db.get_automation_result, result_id
            )
            server_result_id = (result_row or {}).get("server_id")
            if not server_result_id:
                # Fail closed rather than forwarding the raw LOCAL uuid --
                # the server has never heard of it and would refuse with
                # its own opaque `result_not_found`, which the generic
                # except below would then misreport as a connectivity
                # problem for a user who is actually connected fine.
                return ResolveOutcome(
                    status="error",
                    reason=(
                        f"Could not {action_desc}: this result has not been "
                        "synced to the server yet."
                    ),
                )

        try:
            if solved:
                response = await self.server_client.mark_automation_definition_solved(
                    server_id, result_id=server_result_id
                )
            else:
                response = await self.server_client.reopen_automation_definition(
                    server_id
                )
        except ServerClientNotFoundError:
            # 404 sits in its own class (not a ValidationError subclass),
            # so the definitive-refusal catch below would miss it and the
            # connectivity branch would blame the network (final-review
            # finding 3). A vanished server row is just as definitive.
            return ResolveOutcome(
                status="error",
                reason=(
                    f"Could not {action_desc}: the server no longer has "
                    "this automation."
                ),
            )
        except ServerClientValidationError as exc:
            # Every DEFINITIVE refusal, not just the pre-network policy
            # one (`ServerClientPolicyError` is a subclass, so widening
            # the catch subsumes it -- the same policy-vs-connectivity
            # split `_server_refused_outcome` already applies to
            # save_definition, and the PR-4 wave before it).
            #
            # `server_client._call_with_retry` maps every 4xx except 404
            # to this class and never retries it. Live (task 6 round 2,
            # D9), releasing a definition to this device archives the
            # server's copy, and mark-solving it then returns a 409
            # `scheduled_task_definition_archived` with `retryable:
            # false` -- which the old narrow catch let fall through to
            # the connectivity branch below, telling a plainly-connected
            # user to check their network while the real reason sat in
            # the response body. That release-then-solve path is the flow
            # immediately preceding this action, not a contrived one.
            logger.warning(
                "resolve_definition refused by the server for {definition_id} "
                "(server row {server_id}): {exc}",
                definition_id=definition_id,
                server_id=server_id,
                exc=exc,
            )
            return ResolveOutcome(
                status="error",
                reason=(
                    f"The server refused to {action_desc} ({exc}) -- this "
                    "will not resolve by retrying."
                ),
            )
        except ServerClientError as exc:
            logger.warning(
                "resolve_definition could not {action} for {definition_id} "
                "(server row {server_id}): {exc}",
                action=action_desc,
                definition_id=definition_id,
                server_id=server_id,
                exc=exc,
            )
            return ResolveOutcome(
                status="error",
                reason=f"Could not {action_desc} -- this action requires a server connection.",
            )

        await asyncio.to_thread(
            self.db.upsert_automation_definitions_from_server, owner_id, [response]
        )
        return ResolveOutcome(status="saved")

    async def recover_inflight_transfers(self) -> None:
        """Startup recovery for rows stuck `to_server_sent` (spec §6.1.3).

        An ambiguous timeout between a transfer's send and its ack is the
        ONE scenario this replaces -- `SyncEngine`'s own push replay
        deliberately refuses to touch a `to_server_sent` row (Task 4/5),
        so this is the only path that un-sticks one. Mirrors the
        `reconcile_stale_automation_runs` on_mount precedent: each
        sub-step is independently exception-guarded, so a broken recovery
        pass can never block app startup.

        Definitions: CAS straight back to `to_server_pending` -- the
        server's create is hash-idempotent (ruling 4), so a blind retry
        is safe. Reminders: list-and-match on `link_id` first (their
        create is NOT idempotent) -- found means the transfer actually
        landed (convert to the mirror, clear the mutation); absent means
        CAS back to `to_server_pending` for a normal retry.

        Every failure is logged with the row id, the owner scope the
        recovery is acting under, and the primitive (Qodo review, fix
        wave 2) -- never any payload content, which on this path carries
        the user's own reminder/definition text. A pass-level log names
        the primitive alone: no single row is implicated when the
        listing itself is what failed. Guarding per row also stops one
        bad row from cancelling recovery for every row behind it.
        """
        try:
            self._recover_stuck_definitions()
        except Exception:
            logger.exception(
                "Inflight-transfer recovery pass failed for primitive "
                "{primitive}",
                primitive=_DEFINITION_PRIMITIVE,
            )
        try:
            await self._recover_stuck_reminders()
        except Exception:
            logger.exception(
                "Inflight-transfer recovery pass failed for primitive "
                "{primitive}",
                primitive=_REMINDER_PRIMITIVE,
            )

    def _recover_stuck_definitions(self) -> None:
        stuck = [
            row
            for row in self.db.list_automation_definitions()
            if row.get("transfer_state") == "to_server_sent"
        ]
        for row in stuck:
            try:
                self.db.set_transfer_state(
                    _DEFINITION_PRIMITIVE,
                    row["id"],
                    "to_server_pending",
                    expected=("to_server_sent",),
                )
            except Exception:
                logger.exception(
                    "Inflight-transfer recovery failed for row {row_id} "
                    "(owner {owner_id}, primitive {primitive})",
                    row_id=row["id"],
                    owner_id=row.get("owner_id"),
                    primitive=_DEFINITION_PRIMITIVE,
                )

    async def _recover_stuck_reminders(self) -> None:
        """List-and-match recovery, scoped to rows this server actually owns.

        The list-and-match is only meaningful against the server the
        transfer was SENT to, and each stuck row records that server
        itself -- its own mutation's `owner_id`, the same
        `get_pending_mutation_for_local_id` answer `_delete_transfer_
        mutation` and the Task 7 retry-error lookup already key off.
        Reconciling every stuck row against "today's active server" was
        wrong after a server switch (Qodo review, fix wave 2): an
        ambiguously-successful send to server A is absent from server B's
        listing, so it was CAS'd back to `to_server_pending` and replayed
        -- creating a SECOND task on A once A reconnects, the duplicate
        §6.1.3 exists to prevent.

        A row recorded under another owner is therefore SKIPPED, not
        guessed at: it stays `to_server_sent` with its mutation intact
        until that server is the connected one, and says so in the log.
        Deferring is honest; a wrong-server answer is not, and it is not
        recoverable afterwards.

        A stuck row with NO mutation has no recorded owner to defer to
        (its mutation was cleared without the row's state following) and
        is reconciled against the active server, exactly as before --
        skipping it would leave it stuck with nothing left that could
        ever un-stick it.
        """
        stuck = [
            row
            for row in self.db.list_reminder_tasks()
            if row.get("transfer_state") == "to_server_sent"
        ]
        if not stuck:
            return
        if (
            self.server_client is None
            or getattr(self.server_client, "notifications_service", None) is None
        ):
            logger.info(
                "Skipping reminder inflight-transfer recovery: no server connection"
            )
            return
        destination_owner_id = self._active_server_owner_id()
        if destination_owner_id is None:
            logger.info(
                "Skipping reminder inflight-transfer recovery: no active "
                "server identity"
            )
            return

        recoverable: list[dict[str, Any]] = []
        for row in stuck:
            mutation = self.db.get_pending_mutation_for_local_id(
                row["id"], _REMINDER_PRIMITIVE
            )
            recorded_owner_id = (mutation or {}).get("owner_id") or destination_owner_id
            if recorded_owner_id != destination_owner_id:
                logger.info(
                    "Deferring inflight-transfer recovery for row {row_id} "
                    "(primitive {primitive}): sent under {owner_id}, waiting "
                    "for that connection",
                    row_id=row["id"],
                    primitive=_REMINDER_PRIMITIVE,
                    owner_id=recorded_owner_id,
                )
                continue
            recoverable.append(row)
        if not recoverable:
            return

        try:
            response = await self.server_client.list_reminders()
        except Exception as exc:  # noqa: BLE001 - recovery must never crash startup
            logger.warning(
                f"Reminder inflight-transfer recovery could not reach the "
                f"server ({exc}); leaving {len(recoverable)} row(s) for the "
                "next startup"
            )
            return

        items = response.get("items", []) if isinstance(response, dict) else []
        by_link_id = {
            item.get("link_id"): item
            for item in items
            if item.get("link_type") == "chatbook_transfer" and item.get("link_id")
        }

        for row in recoverable:
            local_id = row["id"]
            try:
                matched = by_link_id.get(local_id)
                if matched is not None:
                    # Delete the mutation regardless of outcome (matches
                    # `SyncEngine._push_reminder_transfer`'s own precedent
                    # for this same DB call) -- a "vanished" row (deleted
                    # between the scan above and this convert) has nothing
                    # left to replay for, so leaving the mutation queued
                    # would just strand it forever.
                    self.db.convert_row_to_server_mirror(
                        _REMINDER_PRIMITIVE, local_id, matched, destination_owner_id
                    )
                    self.db.delete_pending_mutation_for_record(
                        local_id, _REMINDER_PRIMITIVE, destination_owner_id
                    )
                else:
                    self.db.set_transfer_state(
                        _REMINDER_PRIMITIVE,
                        local_id,
                        "to_server_pending",
                        expected=("to_server_sent",),
                    )
            except Exception:
                logger.exception(
                    "Inflight-transfer recovery failed for row {row_id} "
                    "(owner {owner_id}, primitive {primitive})",
                    row_id=local_id,
                    owner_id=destination_owner_id,
                    primitive=_REMINDER_PRIMITIVE,
                )
        self._notify_queue_changed()

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

    async def preview_definition(
        self, payload: dict[str, Any], owner_id: str
    ) -> AutomationPreview:
        """Preview an automation-definition authoring payload (Task 5's live-feedback seam).

        Routes by ``owner_id`` (not ``self.owner_id`` -- the modal can
        preview for a different owner than the service's current active
        one): a local owner runs Task 1's pure `preview_automation_
        definition` directly (no I/O, so no `asyncio.to_thread`). A
        server owner round-trips through `SchedulingServerClient.
        preview_automation_definition`; when that round trip fails for
        ANY reason (offline, timeout, 5xx, policy refusal), this falls
        back to the same local pure preview with an extra warning
        (``field="_owner"``) appended, so the modal still shows schedule
        feedback instead of a dead form -- the warning's ``code``/
        ``message`` distinguish a deterministic policy refusal
        (``"policy_denied"``) from an actual connectivity failure
        (``"server_unreachable"``), since only one of those is worth
        retrying (review round 1 finding 3).

        v1 scope guard: `family` other than `"recurring_question"` is
        rejected before any preview runs (`_reject_unsupported_family`) --
        Task 1's pure preview fabricates a `family: unsupported` error for
        `agent_task` that is a scope cut, not real server parity, and must
        never reach a caller through this facade.
        """
        from tldw_chatbook.Scheduling.automation_preview import preview_automation_definition

        guard = self._reject_unsupported_family(payload)
        if guard is not None:
            return guard

        if not self._owner_uses_server(owner_id):
            return preview_automation_definition(payload)

        assert self.server_client is not None
        try:
            response = await self.server_client.preview_automation_definition(
                dict(payload)
            )
        except ServerClientError as exc:
            logger.warning(
                "Server preview unreachable for owner {owner_id} ({exc}); "
                "falling back to local validation",
                owner_id=owner_id,
                exc=exc,
            )
            local_preview = preview_automation_definition(payload)
            warnings = [*(local_preview.warnings or []), _seam_failure_warning(exc)]
            return local_preview.model_copy(update={"warnings": warnings})
        return self._server_preview_to_model(response)

    async def save_definition(
        self,
        payload: dict[str, Any],
        owner_id: str,
        definition_id: str | None = None,
    ) -> SaveDefinitionOutcome:
        """Create or update a local `recurring_question` automation definition.

        Always previews first (create/save ruling 3): an invalid payload
        writes nothing and returns its validation errors. ``definition_id``
        (the LOCAL row id, ``None`` for a create) is this facade's own
        source of truth for create-vs-update -- not whatever `payload`
        happens to carry -- mirroring `SyncEngine`'s replay precedent
        (Task 3) of overriding `mode`/`definition_id`/`definition_version`
        on the outgoing request itself rather than trusting the payload.

        Local owner: computes `next_run_at` and writes straight to
        `ScheduledTasksDB` (`create_automation_definition`/
        `update_automation_definition`).

        Server owner, reachable: preview -> commit (`create_automation_
        definition`/`update_automation_definition` on the server client)
        -> mirror the echo locally. An existing local row is updated in
        place (`adopt_server_definition_identity`, so an edit never
        creates a second local row for the same definition); a brand-new
        definition has no local row to adopt onto, so it goes through the
        same server-mirror upsert the sync pull uses (`upsert_automation_
        definitions_from_server`), then the freshly inserted row is
        looked up by its new server id.

        Server owner, seam unreachable (offline create, or an edit of a
        row that was never synced): the LOCAL pure preview stands in for
        the server's verdict (still "invalid -> write nothing"), the
        local row is written (or updated), and one `automation_definition`
        pending mutation is recorded atomically with it (same transaction,
        `create_automation_definition`/`update_automation_definition`'s
        `pending_mutation` kwarg) for `SyncEngine` to replay later.

        Server owner, refused for good (`ServerClientValidationError` --
        a deterministic pre-network policy refusal, or any server-side
        4xx: 409 version conflict, 422 schedule invalid, ...): returns
        `status="error"` and writes nothing. NOT queued -- a replay would
        hit the identical refusal forever (review round 1 finding 1,
        final review C2). See `_server_refused_outcome`.

        Editing (`definition_id` given) MERGES the payload onto the stored
        row (`_merge_definition_payload`), so a caller that omits a field
        it does not author keeps that field's stored value instead of
        wiping it (final review I4).
        """
        from tldw_chatbook.Scheduling.automation_preview import preview_automation_definition
        from tldw_chatbook.Scheduling.automation_validation import field_error
        from tldw_chatbook.Scheduling.schedule_vocabulary import to_server_schedule

        guard = self._reject_unsupported_family(payload)
        if guard is not None:
            return SaveDefinitionOutcome(
                status="invalid", errors=guard.validation_errors or [], definition_id=definition_id
            )

        local_row: dict[str, Any] | None = None
        if definition_id is not None:
            local_row = await asyncio.to_thread(
                self.db.get_automation_definition, definition_id
            )
            if local_row is None:
                return SaveDefinitionOutcome(
                    status="error",
                    errors=[
                        field_error(
                            "_definition",
                            "not_found",
                            f"Automation definition {definition_id} was not found.",
                        )
                    ],
                    definition_id=definition_id,
                )
            # spec §6.3 (final review I7): the transfer snapshotted this
            # row's payload at begin time, so an edit now would ship the
            # PRE-edit content and then be overwritten by the mirror pull.
            locked = self.transfer_lock_reason(local_row)
            if locked is not None:
                return SaveDefinitionOutcome(
                    status="error",
                    errors=[field_error("_transfer", "transfer_in_progress", locked)],
                    definition_id=definition_id,
                )
            payload = self._merge_definition_payload(payload, local_row)

        if not self._owner_uses_server(owner_id):
            mode = "update" if local_row is not None else "create"
            request = self._build_definition_request(
                payload,
                mode=mode,
                definition_id=definition_id if mode == "update" else None,
                definition_version=local_row.get("version") if local_row else None,
            )
            preview = preview_automation_definition(request)
            if preview.status != PreviewStatus.VALID:
                return SaveDefinitionOutcome(
                    status="invalid",
                    errors=preview.validation_errors or [],
                    definition_id=definition_id,
                )
            saved_id = await self._write_local_definition(
                preview, owner_id, local_row, definition_id
            )
            self._notify_queue_changed()
            return SaveDefinitionOutcome(status="saved", definition_id=saved_id)

        # Server owner: offline-authored (never synced) rows push as a
        # server create even when a local row already exists (Task 3's
        # `_push_definition_mutation` precedent).
        server_mode = (
            "create" if local_row is None or not local_row.get("server_id") else "update"
        )
        request = self._build_definition_request(
            payload,
            mode=server_mode,
            definition_id=local_row.get("server_id") if local_row else None,
            definition_version=local_row.get("version") if local_row else None,
        )

        assert self.server_client is not None
        # `request` is in CLIENT schedule vocabulary (schedule_compute.py)
        # and must stay that way -- it is also handed to
        # `_save_definition_offline` below on a seam failure, which feeds
        # it to the LOCAL pure preview and queues it verbatim as the
        # pending mutation's `definition_payload` (SyncEngine's push
        # translates THAT at replay time). Only the network-bound copy
        # gets translated, so the server's preview doesn't pass an
        # untranslated schedule that later fails to arm (task 3 review,
        # finding 2).
        network_request = dict(request)
        if isinstance(network_request.get("schedule"), dict):
            network_request["schedule"] = to_server_schedule(network_request["schedule"])
        try:
            response = await self.server_client.preview_automation_definition(
                network_request
            )
        except ServerClientValidationError as exc:
            return _server_refused_outcome(exc, definition_id)
        except ServerClientError as exc:
            return await self._save_definition_offline(
                request, owner_id, local_row, definition_id, server_mode, exc
            )

        preview_dict = response if isinstance(response, dict) else {}
        if preview_dict.get("status") != "valid":
            return SaveDefinitionOutcome(
                status="invalid",
                errors=preview_dict.get("validation_errors") or [],
                definition_id=definition_id,
            )

        preview_id = preview_dict.get("id")
        try:
            if server_mode == "update":
                committed = await self.server_client.update_automation_definition(
                    local_row["server_id"], preview_id
                )
            else:
                committed = await self.server_client.create_automation_definition(
                    preview_id
                )
        except ServerClientValidationError as exc:
            return _server_refused_outcome(exc, definition_id)
        except ServerClientError as exc:
            return await self._save_definition_offline(
                request, owner_id, local_row, definition_id, server_mode, exc
            )

        committed = committed if isinstance(committed, dict) else {}
        saved_id = await self._mirror_server_definition(
            committed, owner_id, local_row, definition_id
        )
        self._notify_queue_changed()
        return SaveDefinitionOutcome(status="saved", definition_id=saved_id)

    async def _save_definition_offline(
        self,
        request: dict[str, Any],
        owner_id: str,
        local_row: dict[str, Any] | None,
        definition_id: str | None,
        server_mode: str,
        exc: Exception,
    ) -> SaveDefinitionOutcome:
        """Server seam unreachable during save: local-first fallback.

        Ruling 3 ("save always previews") still holds offline: the LOCAL
        pure preview stands in for the server's verdict, so an invalid
        payload still writes nothing. A valid one writes the local row
        and queues exactly one `automation_definition` mutation in the
        SAME transaction as that write.

        The update branch passes `bump_version=False` (final review C1):
        a server-owned row's `version` column MIRRORS the server's
        (`upsert_automation_definitions_from_server`/`adopt_server_
        definition_identity` copy it verbatim) and the server checks the
        queued `definition_version` for exact equality. Bumping it locally
        desynchronizes the mirror, and because `pending_mutations` is
        `UNIQUE(local_id, primitive, owner_id)` a second offline edit
        REPLACES the first with one carrying the drifted version -- which
        the server then rejects (409) forever.
        """
        from tldw_chatbook.Scheduling.automation_preview import preview_automation_definition

        logger.warning(
            "Server unreachable while saving automation definition for "
            "{owner_id} ({exc}); queuing for later sync",
            owner_id=owner_id,
            exc=exc,
        )
        local_preview = preview_automation_definition(request)
        if local_preview.status != PreviewStatus.VALID:
            return SaveDefinitionOutcome(
                status="invalid",
                errors=local_preview.validation_errors or [],
                definition_id=definition_id,
            )

        fields = self._definition_db_fields_from_preview(local_preview)
        mutation_payload = {
            "action": server_mode,
            "definition_payload": request,
            "server_definition_id": local_row.get("server_id") if local_row else None,
        }
        pending_mutation = {
            "primitive": _DEFINITION_PRIMITIVE,
            "owner_id": owner_id,
            "payload": mutation_payload,
        }

        if local_row is not None:
            await asyncio.to_thread(
                self.db.update_automation_definition,
                definition_id,
                bump_version=False,
                pending_mutation=pending_mutation,
                **fields,
            )
            saved_id = definition_id
        else:
            name = fields.pop("name")
            saved_id = await asyncio.to_thread(
                self.db.create_automation_definition,
                owner_id,
                _SUPPORTED_AUTOMATION_FAMILY,
                name,
                pending_mutation=pending_mutation,
                **fields,
            )

        self._notify_queue_changed()
        return SaveDefinitionOutcome(status="queued", definition_id=saved_id)

    async def _write_local_definition(
        self,
        preview: AutomationPreview,
        owner_id: str,
        local_row: dict[str, Any] | None,
        definition_id: str | None,
    ) -> str:
        """Write a valid local-owner preview's normalized fields to the DB."""
        fields = self._definition_db_fields_from_preview(preview)
        if local_row is not None:
            await asyncio.to_thread(
                self.db.update_automation_definition, definition_id, **fields
            )
            return definition_id

        name = fields.pop("name")
        return await asyncio.to_thread(
            self.db.create_automation_definition,
            owner_id,
            _SUPPORTED_AUTOMATION_FAMILY,
            name,
            **fields,
        )

    async def _mirror_server_definition(
        self,
        server_item: dict[str, Any],
        owner_id: str,
        local_row: dict[str, Any] | None,
        definition_id: str | None,
    ) -> str | None:
        """Mirror a create/update server echo onto the local cache; returns the local id.

        An existing local row (an edit, synced or not) adopts the echoed
        identity/fields in place (`adopt_server_definition_identity`) so
        this never creates a second row for the same definition. A
        brand-new definition has no local row to adopt onto, so it is
        mirrored via the same server-mirror upsert the sync pull uses
        (`upsert_automation_definitions_from_server`), then looked up by
        its new server id -- that upsert reports only insert/update
        counts, not the generated id.

        Either way, any pending `automation_definition` mutation left over
        from an earlier offline save on this row is cleared once this
        online save actually lands -- same precedent as
        `_persist_server_reminder_response`'s `delete_pending_mutation_
        for_record` call. Without this, a stale queued mutation survives a
        later successful online save and the next sync replays it: for a
        never-synced row that just got its first server identity via
        create, that means a SECOND server-side definition and an orphaned
        adopt; for an already-synced row, it means the OLDER queued
        payload silently overwriting this save's newer edit (review round
        1 finding 2).
        """
        if local_row is not None:
            await asyncio.to_thread(
                self.db.adopt_server_definition_identity, definition_id, server_item
            )
            saved_id: str | None = definition_id
        else:
            await asyncio.to_thread(
                self.db.upsert_automation_definitions_from_server, owner_id, [server_item]
            )
            server_id = server_item.get("id")
            if server_id is None:
                return None
            mirrored = await asyncio.to_thread(
                self.db.get_automation_definition_by_server_id, owner_id, server_id
            )
            saved_id = mirrored.get("id") if mirrored else None

        if saved_id is not None:
            await asyncio.to_thread(
                self.db.delete_pending_mutation_for_record,
                saved_id,
                _DEFINITION_PRIMITIVE,
                owner_id,
            )
        return saved_id

    @staticmethod
    def _merge_definition_payload(
        payload: dict[str, Any], local_row: dict[str, Any]
    ) -> dict[str, Any]:
        """Overlay an edit payload onto the stored row's fields (final review I4).

        An authoring payload only carries the fields its author exposes --
        the v1 modal has no input for `description`, `visibility_policy`,
        `approval_policy` or `config.retention_policy`, and none for
        `input.max_tokens` (which `automation_execution` reads). Without
        this merge those fields are absent from the preview request, so the
        normalizer defaults them, `_definition_db_fields_from_preview`
        writes the defaults over the row, AND the server-owned update
        PATCH sends the defaults too -- four fields silently destroyed by
        a rename.

        Rule: an OMITTED key keeps its stored value; a key the payload
        carries wins, including an explicit `None` (so a caller that does
        expose a field can still clear it). `config`/`input`/
        `notification_policy` merge one level deep for the same reason,
        which is why the form emits `provider`/`model` explicitly as
        `None` when blank rather than omitting them.

        `stored` may be SQL NULL (Python `None`) rather than `{}` -- a
        server-mirrored row whose server item omitted the key entirely
        (`upsert_automation_definitions_from_server`'s INSERT path,
        task-4 review Finding 1) leaves the column NULL. Treated as `{}`
        here so the merge still runs instead of silently skipping (which
        used to drop the whole group -- e.g. `input.question` -- on any
        edit that didn't itself touch that group).
        """
        merged: dict[str, Any] = {}
        for key in ("description", "visibility_policy", "approval_policy"):
            if key in local_row and local_row[key] is not None:
                merged[key] = local_row[key]
        merged.update(payload)
        for key in ("config", "input", "notification_policy"):
            stored = local_row.get(key)
            incoming = payload.get(key)
            if isinstance(incoming, dict):
                merged[key] = {
                    **(stored if isinstance(stored, dict) else {}),
                    **incoming,
                }
        return merged

    @staticmethod
    def _build_definition_request(
        payload: dict[str, Any],
        *,
        mode: str,
        definition_id: str | None,
        definition_version: int | None,
    ) -> dict[str, Any]:
        """Build a server-preview-shaped request from an authoring payload.

        Overrides `mode`/`definition_id`/`definition_version` from the
        caller's own resolved state rather than trusting whatever the raw
        payload carries -- this facade is the payload's producer, so it is
        the authority on which mode a save/preview actually is (mirrors
        `SyncEngine`'s replay precedent, Task 3).
        """
        request = dict(payload)
        request["family"] = _SUPPORTED_AUTOMATION_FAMILY
        request["mode"] = mode
        if mode == "update":
            request["definition_id"] = definition_id
            request["definition_version"] = definition_version
        else:
            request.pop("definition_id", None)
            request.pop("definition_version", None)
        return request

    @staticmethod
    def _definition_db_fields_from_preview(preview: AutomationPreview) -> dict[str, Any]:
        """Map a valid preview's normalized config onto automation-definition DB columns.

        `visibility_policy` comes from the preview's own top-level field
        (already wrapped `{"mode": str}`, matching the DB column's/server's
        `ScheduledTaskDefinitionResponse` shape) rather than `normalized_
        config["visibility_policy"]`, which Task 1's preview deliberately
        leaves as the flat mode string.

        `config.scope.resolved_sources` (task 6 E2E finding) is stripped
        before storage: `normalize_recurring_question_scope`'s
        `"all_searchable_library"` branch (`recurring_question_scope.py`)
        computes that key fresh on every call as an OUTPUT projection, not
        an accepted input field (`SUPPORTED_SCOPE_FIELDS` has no such
        entry). Persisting it verbatim made every later re-normalization of
        this stored scope -- `automation_execution.py`'s own dispatch,
        `automation_health.py`'s sources-readable check -- report a
        spurious "unsupported field" error, degrading every scheduled run
        of a default-scope (the common case) definition. It is always
        recomputed on read, so dropping it here loses nothing.
        """
        from tldw_chatbook.Scheduling.schedule_compute import compute_next_run_at

        normalized = preview.normalized_config or {}
        schedule = normalized.get("schedule") or {}
        config = dict(normalized.get("config") or {})
        scope = config.get("scope")
        if isinstance(scope, dict) and "resolved_sources" in scope:
            config["scope"] = {k: v for k, v in scope.items() if k != "resolved_sources"}
        return {
            "name": normalized.get("name"),
            "description": normalized.get("description"),
            "schedule": schedule,
            "input": normalized.get("input") or {},
            "config": config,
            "visibility_policy": preview.visibility_policy or {},
            "notification_policy": normalized.get("notification_policy") or {},
            "approval_policy": normalized.get("approval_policy") or {},
            # Dedicated DB columns, not merely `config` members: the executor
            # reads `row["finding_policy"]` (`automation_execution.py`'s
            # `_resolve_finding_policy`) and the run snapshot copies
            # `task["finding_policy"]` (`automation_handler.py`), so a policy
            # left only inside `config` reaches neither -- every locally
            # authored or offline-queued definition ran with the column
            # DEFAULT (`balanced_findings`) whatever the author picked.
            # `retention_policy` has the same column and the same exposure.
            "finding_policy": config.get("finding_policy") or {},
            "retention_policy": config.get("retention_policy") or {},
            "next_run_at": compute_next_run_at(schedule, now=datetime.now(timezone.utc)),
        }

    def _reject_unsupported_family(
        self, payload: dict[str, Any]
    ) -> AutomationPreview | None:
        """v1 scope guard: only `family="recurring_question"` may be authored here.

        Runs before any preview call so Task 1's local pure preview's
        fabricated `family: unsupported` error for `agent_task` (a scope
        cut it documents itself, not real server parity) never reaches a
        caller through this facade.

        Returns:
            An already-invalid `AutomationPreview` when `family` is
            anything other than `recurring_question`; `None` when the
            guard passes and the caller should proceed to a real preview.
        """
        from tldw_chatbook.Scheduling.automation_validation import field_error

        family_value = payload.get("family")
        if family_value == _SUPPORTED_AUTOMATION_FAMILY:
            return None
        try:
            family_enum = AutomationFamily(family_value)
        except ValueError:
            family_enum = AutomationFamily.RECURRING_QUESTION
        return AutomationPreview(
            mode=payload.get("mode") or "create",
            family=family_enum,
            definition_id=payload.get("definition_id"),
            definition_version=payload.get("definition_version"),
            status=PreviewStatus.INVALID,
            validation_errors=[
                field_error(
                    "family",
                    "unsupported",
                    "Only recurring_question automations can be authored "
                    "here (agent_task authoring is not yet available).",
                )
            ],
        )

    def _owner_uses_server(self, owner_id: str) -> bool:
        """Return True when server operations should be attempted for `owner_id`.

        Same rule as `_use_server`, parameterized: `preview_definition`/
        `save_definition` take an explicit owner (the modal can target a
        different owner than the service's current active one), so they
        cannot use `_use_server`'s `self.owner_id`-bound check.
        """
        return self.server_client is not None and owner_id.startswith("server:")

    def _server_preview_to_model(self, response: dict[str, Any]) -> AutomationPreview:
        """Convert a server `ScheduledTaskPreviewResponse` dict into the model.

        Drops null values before construction, mirroring `_row_to_
        reminder`'s established idiom: every nullable server field already
        has a model default (`None`, or `created_at`'s `default_factory`),
        so a missing/null value falls back cleanly instead of failing
        Pydantic validation (`created_at` is typed `datetime`, not
        `datetime | None`).
        """
        if not isinstance(response, dict):
            response = {}
        fields = {key: value for key, value in response.items() if value is not None}
        return AutomationPreview(**fields)

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
        *,
        owner_id: str | None = None,
    ) -> ReminderTask:
        """Insert or update the local cache from a server reminder response.

        `owner_id` defaults to the service's active owner; `create_reminder`/
        `update_reminder` pass their own so a cross-owner save lands under
        the owner it was authored for.
        """
        owner_id = owner_id or self.owner_id
        local_fields = self._map_server_response_to_local(response)
        server_id = response.get("id")

        if local_id is not None:
            self.db.update_reminder_task(local_id, **local_fields)
            task_id = local_id
        else:
            existing = None
            if server_id:
                existing = self.db.get_reminder_task_by_server_id(owner_id, server_id)
            if existing is not None:
                task_id = existing["id"]
                self.db.update_reminder_task(task_id, **local_fields)
            else:
                task_id = self.db.create_reminder_task(
                    owner_id=owner_id, **local_fields
                )

        if server_id:
            self.db.set_sync_mapping(task_id, server_id, _REMINDER_PRIMITIVE, owner_id)

        self.db.delete_pending_mutation_for_record(
            task_id, _REMINDER_PRIMITIVE, owner_id
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
