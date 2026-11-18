# Tests/Notifications/test_notification_toast_markup.py
"""Review of #2921, round 4, C1 (TASK-34803): a notification's title and
message are runtime text -- a watchlist rule's name, a reminder's title, a
research query -- and reach ``App.notify`` through one shared helper. Rich
markup was left on there, so an unmatched closing tag in that text exited
the app, a bracketed word was swallowed, and ``[@click=...]`` became a live
action link.

These run in a real Textual app WITH NOTIFICATIONS ENABLED: ``run_test``
disables them by default, which is why no UI test ever saw this.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static
from textual.widgets._toast import Toast

from tldw_chatbook.Notifications import (
    ClientNotificationsDB,
    NotificationDispatchService,
)
from tldw_chatbook.Notifications import notification_dispatch_service
from tldw_chatbook.Scheduling.scheduler.handlers.reminder_handler import (
    ReminderHandler,
)
from tldw_chatbook.Utils.NotificationHelper import show_notification

_RUNTIME_TEXT = [
    "a [/b] c",                    # unmatched closing tag: MarkupError
    "[PDF] x",                     # a bracketed word: swallowed
    "[@click=app.quit]x[/]",       # became an action link
    "TIL [/s] cheap flights",
    "see [/r/boardgames] meetup",
]


class _Probe(App):
    def compose(self) -> ComposeResult:
        yield Static("probe")


def _toast_text(app: App) -> str:
    (toast,) = app.screen.query(Toast)
    return str(toast.render())


@pytest.fixture()
def service(tmp_path):
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    try:
        yield NotificationDispatchService(store=store)
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", _RUNTIME_TEXT)
async def test_a_dispatched_notification_shows_its_text_literally(
        service, text):
    app = _Probe()
    async with app.run_test(size=(100, 30), notifications=True) as pilot:
        service.dispatch(
            app=app, category="watchlists", title=f"Alert: {text}",
            message=f"1 new item for {text}")
        await pilot.pause(0.3)

        shown = _toast_text(app)

    assert f"Alert: {text}: 1 new item for {text}" in shown


@pytest.mark.asyncio
@pytest.mark.parametrize("text", _RUNTIME_TEXT)
async def test_a_fired_reminder_shows_its_title_and_body_literally(
        service, text):
    """The scheduler's real reminder handler: Dreams titles its event
    reminder from the watched query."""
    app = _Probe()
    handler = ReminderHandler(dispatch_service=service, app_getter=lambda: app)
    async with app.run_test(size=(100, 30), notifications=True) as pilot:
        await handler.handle(
            {"id": "r-1", "title": f"Dreams: {text}", "body": f"about {text}"})
        await pilot.pause(0.3)

        shown = _toast_text(app)

    assert f"Dreams: {text}: about {text}" in shown


def test_the_helper_turns_markup_off_on_the_notify_fallback():
    app = SimpleNamespace(notify=Mock())

    show_notification(app, "a [/b] c", severity="warning", timeout=3.0)

    app.notify.assert_called_once_with(
        "a [/b] c", severity="warning", timeout=3.0, markup=False)


def test_the_dispatch_services_own_fallback_turns_markup_off_too(
        service, monkeypatch):
    """When the helper itself raises, the dispatch service calls
    ``app.notify`` directly -- the second sink."""

    def broken(*args, **kwargs):
        raise RuntimeError("helper unavailable")

    monkeypatch.setattr(
        notification_dispatch_service, "show_notification", broken)
    app = SimpleNamespace(notify=Mock())

    service.dispatch(
        app=app, category="watchlists", title="Alert: [PDF] x",
        message="1 new item")

    app.notify.assert_called_once_with(
        "Alert: [PDF] x: 1 new item", severity="information", timeout=None,
        markup=False)
