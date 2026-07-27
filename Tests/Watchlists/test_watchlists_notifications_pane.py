"""Tests for the Watchlists notifications pane.

task-876: `NotificationsPane` had no way to distinguish a selected row from
`DataTable`'s own focus cursor, which always sits somewhere -- including on
a row the pane does not consider selected. `selected_notification` is
already `recompose=True` here (unlike `SourcesPane`/`RunsPane`, where a
selection change must not rebuild the table), so the highlight is applied
entirely in `compose()`.
"""

import pytest
from rich.style import Style
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable

from tldw_chatbook.UI.Watchlists_Modules.notifications_pane import (
    DismissNotificationRequested,
    MarkNotificationReadRequested,
    NotificationSelected,
    NotificationsPane,
    RefreshNotificationsRequested,
)


class NotificationsPaneHarness(App):
    def __init__(self):
        super().__init__()
        self.captured_messages = []

    def compose(self) -> ComposeResult:
        yield NotificationsPane()

    def on_notification_selected(self, message: NotificationSelected) -> None:
        self.captured_messages.append(("notification_selected", message.notification))

    def on_refresh_notifications_requested(
        self, message: RefreshNotificationsRequested
    ) -> None:
        self.captured_messages.append(("refresh_requested", None))

    def on_mark_notification_read_requested(
        self, message: MarkNotificationReadRequested
    ) -> None:
        self.captured_messages.append(("mark_read_requested", message.notification_id))

    def on_dismiss_notification_requested(
        self, message: DismissNotificationRequested
    ) -> None:
        self.captured_messages.append(("dismiss_requested", message.notification_id))


@pytest.fixture
def sample_notifications():
    return [
        {
            "id": 1,
            "title": "Research complete",
            "category": "research",
            "severity": "info",
            "created_at": "2026-07-18 10:00",
            "is_read": False,
            "message": "The synthesis is ready.",
        },
        {
            "id": 2,
            "title": "Check failed",
            "category": "error",
            "severity": "error",
            "created_at": "2026-07-18 11:00",
            "is_read": True,
            "message": "Source unreachable.",
        },
    ]


def _cell_style(table: DataTable, row_key: str, column_index: int) -> Style:
    column_key = list(table.columns.keys())[column_index]
    raw_style = table.get_cell(row_key, column_key).style
    return Style.parse(raw_style) if isinstance(raw_style, str) else raw_style


@pytest.mark.asyncio
async def test_notifications_pane_renders_table_and_toolbar():
    app = NotificationsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(NotificationsPane)
        assert pane.query_one("#notifications-table", DataTable)
        assert pane.query_one("#notifications-refresh-button", Button)
        assert pane.query_one("#notifications-mark-read-button", Button)
        assert pane.query_one("#notifications-dismiss-button", Button)


@pytest.mark.asyncio
async def test_notifications_pane_populates_table(sample_notifications):
    app = NotificationsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(NotificationsPane)
        pane.notifications = sample_notifications
        await pilot.pause()

        table = pane.query_one("#notifications-table", DataTable)
        assert table.row_count == 2


@pytest.mark.asyncio
async def test_notifications_pane_selects_and_posts_message(sample_notifications):
    app = NotificationsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(NotificationsPane)
        pane.notifications = sample_notifications
        await pilot.pause()

        pane.select_notification_by_id("1")
        await pilot.pause()

        assert pane.selected_notification == sample_notifications[0]
        assert app.captured_messages == [
            ("notification_selected", sample_notifications[0])
        ]


# --- task-876: selected row is distinguishable from a merely-focused one ---


@pytest.mark.asyncio
async def test_selected_notification_row_is_styled_distinctly_from_others(
    sample_notifications,
):
    app = NotificationsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(NotificationsPane)
        pane.notifications = sample_notifications
        pane.select_notification_by_id("1")
        await pilot.pause()

        table = pane.query_one("#notifications-table", DataTable)
        assert _cell_style(table, "1", 0).reverse
        assert not _cell_style(table, "2", 0).reverse


@pytest.mark.asyncio
async def test_notification_selection_highlight_moves_on_reselection(
    sample_notifications,
):
    app = NotificationsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(NotificationsPane)
        pane.notifications = sample_notifications
        pane.select_notification_by_id("1")
        await pilot.pause()

        pane.select_notification_by_id("2")
        await pilot.pause()

        table = pane.query_one("#notifications-table", DataTable)
        assert not _cell_style(table, "1", 0).reverse
        assert _cell_style(table, "2", 0).reverse
