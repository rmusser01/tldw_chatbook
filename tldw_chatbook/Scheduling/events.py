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


class SyncCompleted(Message):
    """Posted when a sync attempt completes."""

    def __init__(self, owner_id: str, conflict_count: int) -> None:
        super().__init__()
        self.owner_id = owner_id
        self.conflict_count = conflict_count


class SyncFailed(Message):
    """Posted when a sync attempt fails."""

    def __init__(self, owner_id: str, error: str) -> None:
        super().__init__()
        self.owner_id = owner_id
        self.error = error
