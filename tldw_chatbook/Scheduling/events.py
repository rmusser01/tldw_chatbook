"""Scheduling domain events."""

from __future__ import annotations

from typing import Any

from textual.message import Message


class ReminderFormSubmitted(Message):
    """Posted when the reminder create/edit form is saved."""

    def __init__(self, form_data: dict[str, Any]) -> None:
        super().__init__()
        self.form_data = form_data
