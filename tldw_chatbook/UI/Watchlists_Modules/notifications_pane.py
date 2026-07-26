"""Client-owned local notifications inbox for the Watchlists destination."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Static

from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard


class NotificationSelected(Message):
    """Posted when the user selects a local notification."""

    def __init__(self, notification: dict[str, Any] | None) -> None:
        self.notification = notification
        super().__init__()


class RefreshNotificationsRequested(Message):
    """Posted when the user requests an inbox refresh."""


class MarkNotificationReadRequested(Message):
    """Posted when the user marks a notification as read."""

    def __init__(self, notification_id: int) -> None:
        self.notification_id = notification_id
        super().__init__()


class DismissNotificationRequested(Message):
    """Posted when the user dismisses a notification."""

    def __init__(self, notification_id: int) -> None:
        self.notification_id = notification_id
        super().__init__()


class NotificationsPane(RecomposeCaptureGuard, Vertical):
    """Review and update the local client-notification inbox."""

    notifications = reactive[list[dict[str, Any]]]([], recompose=True)
    selected_notification = reactive[dict[str, Any] | None](None, recompose=True)

    def compose(self):
        yield Static(
            "Local client inbox · stored on this device",
            id="notifications-local-ownership",
        )
        with Horizontal(id="notifications-toolbar", classes="destination-filter-strip"):
            yield Button(
                "Refresh",
                id="notifications-refresh-button",
                variant="primary",
            )
            yield Button(
                "Mark read",
                id="notifications-mark-read-button",
                disabled=self.selected_notification is None,
            )
            yield Button(
                "Dismiss",
                id="notifications-dismiss-button",
                disabled=self.selected_notification is None,
            )

        table = DataTable(id="notifications-table")
        table.add_columns("Status", "Title", "Category", "Severity", "Created")
        for notification in self.notifications:
            table.add_row(
                "Read" if notification.get("is_read") else "Unread",
                Text(str(notification.get("title") or "Notification")),
                Text(str(notification.get("category") or "-")),
                Text(str(notification.get("severity") or "-")),
                Text(str(notification.get("created_at") or "-")),
                key=str(notification.get("id")),
            )
        yield table

        selected = self.selected_notification
        yield Static("Notification detail", classes="pane-title")
        yield Static(
            Text(
                str(selected.get("message") or "No notification selected.")
                if selected
                else "No notification selected."
            ),
            id="notifications-detail-message",
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.select_notification_by_id(str(event.row_key.value))

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        event.stop()
        self.select_notification_by_id(str(event.cell_key.row_key.value))

    def select_notification_by_id(self, notification_id: str) -> None:
        """Select one visible notification by its stable local id."""
        self.selected_notification = next(
            (
                notification
                for notification in self.notifications
                if str(notification.get("id")) == notification_id
            ),
            None,
        )

    def watch_selected_notification(self, notification: dict[str, Any] | None) -> None:
        if self.is_mounted:
            self.post_message(NotificationSelected(notification))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        selected = self.selected_notification
        if button_id == "notifications-refresh-button":
            self.post_message(RefreshNotificationsRequested())
        elif button_id == "notifications-mark-read-button" and selected:
            notification_id = selected.get("id")
            if notification_id not in (None, ""):
                self.post_message(MarkNotificationReadRequested(int(notification_id)))
        elif button_id == "notifications-dismiss-button" and selected:
            notification_id = selected.get("id")
            if notification_id not in (None, ""):
                self.post_message(DismissNotificationRequested(int(notification_id)))
        event.stop()
