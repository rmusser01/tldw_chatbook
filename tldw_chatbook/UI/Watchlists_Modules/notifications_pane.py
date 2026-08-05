"""Client-owned local notifications inbox for the Watchlists destination."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.coordinate import Coordinate
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Static

from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard
from .table_selection import highlight_is_user_driven


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

    #: task-876: Rich's own terminal-agnostic "current item" idiom (see
    #: `snippet_editor.py`'s `_WHITESPACE_MARKER_STYLE` and
    #: `library_media_viewer.py`'s search-match highlighting), used here
    #: because a `DataTable` cell's `Text` cannot reference Textual CSS
    #: variables ($ds-focus-bg etc.) the way a widget's own styles can.
    #: `selected_notification` is `recompose=True`, so -- unlike
    #: Sources/RunsPane below -- this can be applied entirely in `compose()`.
    _SELECTED_ROW_STYLE = "reverse bold"

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

        selected_id = (
            str(self.selected_notification.get("id"))
            if self.selected_notification
            else None
        )
        table = DataTable(id="notifications-table")
        table.add_columns("Status", "Title", "Category", "Severity", "Created")
        selected_index: int | None = None
        for index, notification in enumerate(self.notifications):
            row_id = str(notification.get("id"))
            if row_id == selected_id:
                selected_index = index
            # Distinguishable from the DataTable's own focus cursor, which
            # is a keyboard-navigation affordance that always sits
            # somewhere -- including on a row this pane does not consider
            # selected (task-876).
            style = self._SELECTED_ROW_STYLE if row_id == selected_id else ""
            table.add_row(
                Text("Read" if notification.get("is_read") else "Unread", style=style),
                Text(str(notification.get("title") or "Notification"), style=style),
                Text(str(notification.get("category") or "-"), style=style),
                Text(str(notification.get("severity") or "-"), style=style),
                Text(str(notification.get("created_at") or "-"), style=style),
                key=row_id,
            )
        if selected_index is not None:
            # TASK-1105. `selected_notification` is `recompose=True`, so
            # selecting a row rebuilds this pane and constructs a BRAND NEW
            # `DataTable` whose cursor starts at row 0. That fresh cursor
            # posts `CellHighlighted(row 0)`, which the handlers below turn
            # straight back into "select the first notification" -- so before
            # this line, clicking row 2 selected row 2 and was then dragged
            # back to row 0 by the rebuild it had just caused. Seeding the new
            # table's cursor from the surviving selection makes the re-fired
            # highlight agree with it, which both stops the bounce and keeps
            # the keyboard cursor where the user left it.
            table.cursor_coordinate = Coordinate(selected_index, 0)
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

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Select on cursor movement, which is what a mouse click produces.

        TASK-1105, matching `SourcesPane`. `RowSelected`/`CellSelected` fire on
        *activation* -- Enter, or a second click on an already-current cell --
        so a single click on any row but the current one moved the cursor and
        selected nothing.
        """
        event.stop()
        if not highlight_is_user_driven(event):
            return
        if event.row_key is not None and event.row_key.value is not None:
            self.select_notification_by_id(str(event.row_key.value))

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        """Same, for a table whose cursor is cell-shaped rather than row-shaped."""
        event.stop()
        if not highlight_is_user_driven(event):
            return
        row_key = getattr(event.cell_key, "row_key", None)
        if row_key is not None and row_key.value is not None:
            self.select_notification_by_id(str(row_key.value))

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
