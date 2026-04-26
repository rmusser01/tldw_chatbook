from __future__ import annotations

import pytest

from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB
from tldw_chatbook.UI.Subscription_Modules.notifications_inbox_controller import (
    NotificationsInboxController,
)


class PolicyDecision:
    def __init__(self, allowed: bool) -> None:
        self.allowed = allowed


class FakeApp:
    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed = allowed
        self.action_ids: list[str] = []

    def require_ui_action_allowed(self, *, action_id: str):
        self.action_ids.append(action_id)
        return PolicyDecision(self.allowed)


@pytest.mark.asyncio
async def test_notifications_inbox_controller_filters_rows_through_store(tmp_path):
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    store.insert(
        category="watchlists",
        title="Watchlist alert",
        message="Remote source changed.",
        severity="warning",
        source_backend="server",
    )
    store.insert(
        category="research",
        title="Research complete",
        message="Bundle ready.",
        severity="info",
        source_backend="local",
    )
    app = FakeApp()
    controller = NotificationsInboxController(app_instance=app, store=store)

    rows = await controller.load_rows(category="research", source_backend="local")

    assert [row["title"] for row in rows] == ["Research complete"]
    assert app.action_ids == ["notifications.queue.list.local"]


@pytest.mark.asyncio
async def test_notifications_inbox_controller_gates_preferences(tmp_path):
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    app = FakeApp()
    controller = NotificationsInboxController(app_instance=app, store=store)

    preferences = await controller.update_preferences(
        delivery_enabled=False,
        muted_categories=["watchlists"],
    )
    loaded = await controller.load_preferences()

    assert preferences == loaded
    assert preferences["delivery_enabled"] is False
    assert preferences["muted_categories"] == ["watchlists"]
    assert app.action_ids == [
        "notifications.preferences.configure.local",
        "notifications.preferences.list.local",
    ]


@pytest.mark.asyncio
async def test_notifications_inbox_controller_denies_preference_updates(tmp_path):
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    app = FakeApp(allowed=False)
    controller = NotificationsInboxController(app_instance=app, store=store)

    preferences = await controller.update_preferences(delivery_enabled=False)

    assert preferences is None
    assert store.get_preferences()["delivery_enabled"] is True
    assert app.action_ids == ["notifications.preferences.configure.local"]
