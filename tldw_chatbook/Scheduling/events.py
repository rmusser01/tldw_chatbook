"""Scheduling domain events."""

from __future__ import annotations

from typing import Any

from textual.message import Message

from ..Widgets.detail_value_row import DetailValueRow
from .models import ReminderTask


class ReminderFormSubmitted(Message):
    """Posted when the reminder create/edit form is saved."""

    def __init__(self, form_data: dict[str, Any], task_id: str | None = None) -> None:
        super().__init__()
        self.form_data = form_data
        self.task_id = task_id


class DeleteTaskRequested(Message):
    """Posted when the user confirms deletion of a scheduled task."""

    def __init__(self, task: ReminderTask) -> None:
        super().__init__()
        self.task = task


class EditTaskRequested(Message):
    """Posted when the user asks to edit a reminder."""

    def __init__(self, task: ReminderTask) -> None:
        super().__init__()
        self.task = task


class AcknowledgeIncidentRequested(Message):
    """TASK-26027: posted when the user acknowledges a failure incident."""

    def __init__(self, incident_id: int) -> None:
        super().__init__()
        self.incident_id = incident_id


class EnableTaskRequested(Message):
    """Posted when the user asks to enable a reminder."""

    def __init__(self, task: ReminderTask) -> None:
        super().__init__()
        self.task = task


class DisableTaskRequested(Message):
    """Posted when the user asks to disable a reminder."""

    def __init__(self, task: ReminderTask) -> None:
        super().__init__()
        self.task = task


class RunReminderNowRequested(Message):
    """Posted when the user asks to run a reminder immediately (task-18938)."""

    def __init__(self, task: ReminderTask) -> None:
        super().__init__()
        self.task = task


# redesign PR-4, task 4 (ruling 2): `TransferToServerRequested`/
# `TransferToLocalRequested`/`CancelTransferRequested`/
# `RetryTransferRequested` (schedules-handoff spec §6, PR-5 task 7) were
# posted only by `TaskDetail`'s now-retired legacy Move/Retry/Cancel
# buttons and consumed only by `SchedulesWorkbench`'s now-retired
# `_begin_transfer`/`_cancel_transfer` -- deleted with both ends (`git
# grep` verified zero remaining producers/consumers before removal). The
# Runs-on row's dropdown + mini-bar uses `ReminderOwnerActionRequested`
# below instead.


class ReminderFieldEditRequested(Message):
    """Posted when a reminder-pane Frequency row's inline editor commits
    an edit (redesign PR-3, task 3).

    ``row`` travels with the message so the handler (the workbench, which
    owns the `SchedulingService` `TaskDetail` does not reach into) can
    render a field-specific failure directly on the row that started the
    edit (`DetailValueRow.show_error`) without a second lookup -- the
    same "carry the widget the caller will need" idiom
    `DetailValueRow.Activated` already uses for its own ``row``.
    `TaskDetail` has already called `row.end_edit()` before posting this
    (closing the editor, restoring the OLD display) -- a failure needs no
    separate "restore" step beyond that.
    """

    def __init__(
        self, task: ReminderTask, payload: dict[str, Any], row: DetailValueRow
    ) -> None:
        super().__init__()
        self.task = task
        self.payload = payload
        self.row = row


class DefinitionFieldEditRequested(Message):
    """Posted when a definition-pane Details/Frequency row's inline editor
    commits an edit (redesign PR-3, task 4).

    ``definition`` is the raw definition dict `DefinitionDetail.set_
    definition` was last painted with (a local DB row or a raw server
    list-response dict -- either shape). Its own ``id`` may be a LOCAL row
    id or, for a row shown from a pure server fetch that has no local
    shadow yet, the SERVER's id -- the handler resolves the actual local
    id `SchedulingService.save_definition` needs the same way the
    existing full-modal Edit action already does
    (`SchedulesWorkbench._resolve_local_definition_id`), rather than
    assuming ``id`` is always local.

    ``payload`` is already shaped for `save_definition` (family/name/
    schedule resent verbatim per `definition_detail._definition_edit_
    payload`'s own docstring -- `_merge_definition_payload` does not
    default those from storage the way its docstring implies). ``row``
    travels with it (same "carry the widget the handler needs" idiom
    `DetailValueRow.Activated`/`ReminderFieldEditRequested` already use)
    so the workbench can call `row.show_error(...)` directly on failure.
    `DefinitionDetail` has already called `row.end_edit()` before posting
    this.
    """

    def __init__(
        self, definition: dict[str, Any], payload: dict[str, Any], row: DetailValueRow
    ) -> None:
        super().__init__()
        self.definition = definition
        self.payload = payload
        self.row = row


class DefinitionLifecycleToggleRequested(Message):
    """Posted when the definition pane's header Pause/Resume button is
    pressed (redesign PR-3, task 4 -- `SchedulingService.set_definition_
    lifecycle`'s first UI caller).

    ``action`` is ``"pause"`` or ``"resume"`` (`DefinitionDetail` decides
    which, from the definition's current ``lifecycle``) -- the same two
    action strings `SchedulingService._LIFECYCLE_ACTIONS` accepts.
    """

    def __init__(self, definition: dict[str, Any], action: str) -> None:
        super().__init__()
        self.definition = definition
        self.action = action


class ReminderOwnerActionRequested(Message):
    """Posted by the reminder pane's 'Runs on' row (redesign PR-3, task 5):
    a dropdown owner pick (``action`` is the transfer direction,
    ``"to_server"``/``"to_local"``), or the row's own proactively-shown
    Cancel/Retry affordance for an in-flight/failed transfer (``action``
    is ``"cancel"``/``"retry"``).

    ``row`` travels with it (same "carry the widget the handler needs"
    idiom `DetailValueRow.Activated`/`ReminderFieldEditRequested` already
    use) so the workbench can call `row.show_error(...)` directly on a
    refusal -- this row's transfer refusals render inline (health-quoting
    preserved) rather than as a toast. That inline rendering was
    originally the coexisting alternative to the legacy Move/Cancel/Retry
    buttons' `SchedulesWorkbench._begin_transfer` toast (PR-3 task 5
    brief); redesign PR-4 task 4 deleted those buttons and that method, so
    this row's dropdown is now the ONE transfer surface (ruling 2).
    `TaskDetail` has already called `row.end_edit()` (a dropdown commit)
    before posting this.
    """

    def __init__(self, task: ReminderTask, action: str, row: DetailValueRow) -> None:
        super().__init__()
        self.task = task
        self.action = action
        self.row = row


class DefinitionOwnerActionRequested(Message):
    """Definition-pane counterpart of `ReminderOwnerActionRequested`
    (redesign PR-3, task 5). ``definition`` is the raw dict
    `DefinitionDetail.set_definition` was last painted with, same
    resolve-to-a-local-id handling as `DefinitionFieldEditRequested`.
    """

    def __init__(
        self, definition: dict[str, Any], action: str, row: DetailValueRow
    ) -> None:
        super().__init__()
        self.definition = definition
        self.action = action
        self.row = row


class ViewDefinitionResultsRequested(Message):
    """Posted when a definition pane's 'Unread results' row is activated
    (redesign PR-4, task 2 -- the retired "See Results tab" pointer's
    live replacement). ``definition`` is the raw dict `DefinitionDetail.
    set_definition` was last painted with; the workbench resolves both
    its local/server id spaces (`index_definitions_by_id`'s own caveat)
    and pushes a `ResultsTab` scoped to this one definition.
    """

    def __init__(self, definition: dict[str, Any]) -> None:
        """
        Args:
            definition: The raw dict `DefinitionDetail.set_definition`
                was last painted with (local DB row or raw server
                list-response dict).
        """
        super().__init__()
        self.definition = definition


class DefinitionRunNowRequested(Message):
    """Posted when a definition pane's header 'Run now' button is pressed
    (redesign PR-4, task 3 -- the retired Automations-tab `r` key's live
    replacement, ruling 2). ``definition`` is the raw dict
    `DefinitionDetail.set_definition` was last painted with; the workbench
    routes it to the existing owner-routed dispatch
    (`SchedulesWorkbench._run_automation_now`) unchanged.
    """

    def __init__(self, definition: dict[str, Any]) -> None:
        """
        Args:
            definition: The raw dict `DefinitionDetail.set_definition`
                was last painted with (local DB row or raw server
                list-response dict).
        """
        super().__init__()
        self.definition = definition


class ViewDefinitionAuditRequested(Message):
    """Posted when a definition pane's 'Last run' row is activated
    (redesign PR-4, task 3 -- the retired Automations-tab's third
    (run-history/audit) pane's live replacement; the row's own "...see
    Run history" copy for a server-owned definition is the pointer this
    activation makes live). ``definition`` is the raw dict
    `DefinitionDetail.set_definition` was last painted with; the
    workbench pushes a `definition_audit_view.DefinitionAuditView` scoped
    to this one definition.
    """

    def __init__(self, definition: dict[str, Any]) -> None:
        """
        Args:
            definition: The raw dict `DefinitionDetail.set_definition`
                was last painted with (local DB row or raw server
                list-response dict).
        """
        super().__init__()
        self.definition = definition


class SyncCompleted(Message):
    """Posted when a non-failing sync attempt completes.

    ``outcome`` is the engine's ``SyncOutcome`` (task-23105 review F3):
    it says whether the attempt was applicable at all and how many items
    were pulled/pushed, so the UI can report honestly. ``None`` means
    the sender predates outcomes (treated as a plain completion). A
    failed attempt posts ``SyncFailed`` instead.

    Args:
        owner_id: The sync owner whose attempt completed.
        conflict_count: Number of unresolved reminder-task conflicts
            outstanding after the attempt; 0 means the queue is clean.
        outcome: The engine's ``SyncOutcome`` for the attempt, or None
            from senders that predate outcomes. Typed ``object`` to keep
            this event module free of a service-layer import.
    """

    def __init__(
        self,
        owner_id: str,
        conflict_count: int,
        outcome: object | None = None,
    ) -> None:
        super().__init__()
        self.owner_id = owner_id
        self.conflict_count = conflict_count
        self.outcome = outcome


class SyncFailed(Message):
    """Posted when a sync attempt fails.

    Args:
        owner_id: The sync owner whose attempt failed.
        error: Human-readable failure text for the UI. Sourced either
            from a raised exception or from the error the engine recorded
            on a ``SyncOutcome`` whose status is ``"error"``.
    """

    def __init__(self, owner_id: str, error: str) -> None:
        super().__init__()
        self.owner_id = owner_id
        self.error = error
