"""Scheduling domain events."""

from __future__ import annotations

from typing import Any

from textual.message import Message

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
