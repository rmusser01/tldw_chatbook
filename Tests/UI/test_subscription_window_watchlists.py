from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB
from tldw_chatbook.Notifications.notification_dispatch_service import NotificationDispatchService
from tldw_chatbook.UI.SubscriptionWindow import SubscriptionWindow


class _PolicyRecorder:
    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls: list[dict[str, object]] = []

    def __call__(self, *, action_id: str, scope_type: str | None = None):
        call = {"action_id": action_id, "scope_type": scope_type}
        self.calls.append(call)
        return SimpleNamespace(allowed=self.allowed, action_id=action_id, scope_type=scope_type)


class _StubListView:
    def __init__(self) -> None:
        self.items: list[object] = []
        self.highlighted_child: object | None = None
        self.index: int | None = None

    async def clear(self) -> None:
        self.items.clear()
        self.highlighted_child = None
        self.index = None

    async def append(self, item: object) -> None:
        self.items.append(item)


class _StubWidget:
    def __init__(self, *, display: bool = True, value: object = False, text: str = "") -> None:
        self.display = display
        self.value = value
        self.text = text
        self.disabled = False
        self.updated: list[object] = []
        self.focused = False

    def update(self, value: object) -> None:
        self.updated.append(value)

    def load_text(self, value: str) -> None:
        self.text = value

    def focus(self) -> None:
        self.focused = True


def _normalized_watch_row(runtime_backend: str) -> dict[str, object]:
    entity_kind = "subscription" if runtime_backend == "local" else "watchlist_source"
    return {
        "id": f"{runtime_backend}:{entity_kind}:17",
        "source_id": 17,
        "title": "Example Source",
        "url": "https://example.com/feed.xml",
        "source_type": "rss",
        "backend": runtime_backend,
        "entity_kind": entity_kind,
        "active": True,
        "tags": ["tech", "news"],
        "settings": {"preserve": True} if runtime_backend == "server" else None,
    }


def _select_list_item(list_view: _StubListView, index: int = 0) -> object:
    item = list_view.items[index]
    list_view.highlighted_child = item
    list_view.index = index
    return item


def build_window(*, tmp_path: Path, runtime_backend: str, policy_allowed: bool = True) -> SubscriptionWindow:
    notifications_store = ClientNotificationsDB(
        db_path=tmp_path / f"notifications-{runtime_backend}.sqlite3",
        client_id=f"tests-{runtime_backend}",
    )
    require_ui_action_allowed = _PolicyRecorder(allowed=policy_allowed)
    app = SimpleNamespace(
        runtime_backend=runtime_backend,
        current_runtime_backend=runtime_backend,
        watchlist_scope_service=Mock(),
        client_notifications_db=notifications_store,
        notification_dispatch_service=NotificationDispatchService(store=notifications_store),
        require_ui_action_allowed=require_ui_action_allowed,
        notify=Mock(),
    )
    app.watchlist_scope_service.list_watch_items = AsyncMock(
        return_value=[_normalized_watch_row(runtime_backend)]
    )
    app.watchlist_scope_service.save_watch_item = AsyncMock(
        return_value=_normalized_watch_row(runtime_backend)
    )
    app.watchlist_scope_service.delete_watch_item = AsyncMock(
        return_value={
            "deleted": True,
            "source_id": 17,
            "restore_window_seconds": 10,
            "restore_expires_at": "2026-04-21T12:00:00Z",
        }
    )

    window = SubscriptionWindow(app)
    window.notify = Mock()
    window.run_worker = Mock()

    widgets = {
        "#subscription-list": _StubListView(),
        "#notifications-list": _StubListView(),
        "#review-main": _StubWidget(display=True),
        "#review-local-only-state": _StubWidget(display=False),
        "#enable-scheduler": _StubWidget(display=True, value=False),
        "#sub-name": _StubWidget(display=True, value=""),
        "#sub-type": _StubWidget(display=True, value="rss"),
        "#sub-url": _StubWidget(display=True, value=""),
        "#sub-description": _StubWidget(display=True, text=""),
        "#sub-tags": _StubWidget(display=True, value=""),
        "#sub-folder": _StubWidget(display=True, value=""),
        "#sub-priority": _StubWidget(display=True, value="3"),
        "#sub-frequency": _StubWidget(display=True, value="3600"),
        "#sub-auto-ingest": _StubWidget(display=True, value=False),
        "#sub-auth-type": _StubWidget(display=True, value="none"),
        "#sub-headers": _StubWidget(display=True, text=""),
    }

    def query_one(selector: str, _widget_type=None):
        if selector not in widgets:
            raise AssertionError(f"Unexpected selector requested in test: {selector}")
        return widgets[selector]

    window.query_one = Mock(side_effect=query_one)
    window.notifications_store = notifications_store
    window._test_widgets = widgets
    return window


@pytest.mark.asyncio
async def test_server_mode_refresh_skips_local_scheduler_and_loads_watchlist_list(tmp_path: Path):
    window = build_window(tmp_path=tmp_path, runtime_backend="server")

    await window.refresh_backend_view()

    assert window.scheduler_worker is None
    assert window.backend_controller.last_loaded_backend == "server"
    subscription_list = window._test_widgets["#subscription-list"]
    assert len(subscription_list.items) == 1
    assert subscription_list.items[0].data["id"] == "server:watchlist_source:17"
    assert subscription_list.items[0].data["title"] == "Example Source"


@pytest.mark.asyncio
async def test_normalized_watch_item_selection_populates_editor_fields(tmp_path: Path):
    window = build_window(tmp_path=tmp_path, runtime_backend="server")
    await window.refresh_backend_view()

    subscription_list = window._test_widgets["#subscription-list"]
    selected_item = _select_list_item(subscription_list)

    await window.handle_subscription_selected(SimpleNamespace(item=selected_item))

    assert window.selected_subscription == "server:watchlist_source:17"
    assert window.query_one("#sub-name").value == "Example Source"
    assert window.query_one("#sub-type").value == "rss"
    assert window.query_one("#sub-url").value == "https://example.com/feed.xml"
    assert window.query_one("#sub-tags").value == "tech, news"


@pytest.mark.asyncio
async def test_save_uses_scope_service_with_normalized_payload_and_refreshes(tmp_path: Path):
    window = build_window(tmp_path=tmp_path, runtime_backend="server")
    refresh_backend_view = AsyncMock()
    window.refresh_backend_view = refresh_backend_view
    selected_row = _normalized_watch_row("server")
    window.selected_subscription = selected_row["id"]
    window._selected_watch_item = selected_row

    window.query_one("#sub-name").value = "Updated Source"
    window.query_one("#sub-type").value = "site"
    window.query_one("#sub-url").value = "https://example.com"
    window.query_one("#sub-tags").value = "alpha, beta"

    await window.handle_save_subscription(None)

    window.app_instance.watchlist_scope_service.save_watch_item.assert_awaited_once_with(
        runtime_backend="server",
        payload={
            "id": "server:watchlist_source:17",
            "name": "Updated Source",
            "url": "https://example.com",
            "source_type": "site",
            "active": True,
            "tags": ["alpha", "beta"],
            "settings": {"preserve": True},
        },
    )
    refresh_backend_view.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_clears_stale_selected_row_before_create_save(tmp_path: Path):
    window = build_window(tmp_path=tmp_path, runtime_backend="server")
    await window.refresh_backend_view()
    subscription_list = window._test_widgets["#subscription-list"]
    _select_list_item(subscription_list)
    refresh_backend_view = AsyncMock()
    window.refresh_backend_view = refresh_backend_view

    await window.handle_add_subscription(None)
    window.query_one("#sub-name").value = "Brand New Source"
    window.query_one("#sub-type").value = "site"
    window.query_one("#sub-url").value = "https://example.net"
    window.query_one("#sub-tags").value = "fresh"

    await window.handle_save_subscription(None)

    window.app_instance.watchlist_scope_service.save_watch_item.assert_awaited_once_with(
        runtime_backend="server",
        payload={
            "name": "Brand New Source",
            "url": "https://example.net",
            "source_type": "site",
            "active": True,
            "tags": ["fresh"],
        },
    )
    refresh_backend_view.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_after_save_uses_active_edit_context_when_list_selection_is_cleared(tmp_path: Path):
    window = build_window(tmp_path=tmp_path, runtime_backend="server")
    await window.refresh_backend_view()
    subscription_list = window._test_widgets["#subscription-list"]
    selected_item = _select_list_item(subscription_list)

    await window.handle_subscription_selected(SimpleNamespace(item=selected_item))
    window.query_one("#sub-name").value = "Updated Source"
    await window.handle_save_subscription(None)

    assert subscription_list.highlighted_child is None
    assert subscription_list.index is None

    await window.handle_delete_subscription(None)

    window.app_instance.watchlist_scope_service.delete_watch_item.assert_awaited_once_with(
        runtime_backend="server",
        item_id="server:watchlist_source:17",
    )


@pytest.mark.asyncio
async def test_backend_switch_clears_stale_edit_context(tmp_path: Path):
    window = build_window(tmp_path=tmp_path, runtime_backend="server")
    await window.refresh_backend_view()
    subscription_list = window._test_widgets["#subscription-list"]
    selected_item = _select_list_item(subscription_list)

    await window.handle_subscription_selected(SimpleNamespace(item=selected_item))
    await window.handle_runtime_backend_changed("local")

    assert window.selected_subscription is None
    assert window._selected_watch_item is None
    assert subscription_list.highlighted_child is None
    assert subscription_list.index is None


@pytest.mark.asyncio
async def test_delete_button_routes_through_backend_controller_and_refreshes(tmp_path: Path):
    window = build_window(tmp_path=tmp_path, runtime_backend="server")
    await window.refresh_backend_view()
    subscription_list = window._test_widgets["#subscription-list"]
    _select_list_item(subscription_list)
    refresh_backend_view = AsyncMock()
    window.refresh_backend_view = refresh_backend_view

    await window.handle_delete_subscription(None)

    window.app_instance.watchlist_scope_service.delete_watch_item.assert_awaited_once_with(
        runtime_backend="server",
        item_id="server:watchlist_source:17",
    )
    refresh_backend_view.assert_awaited_once()


@pytest.mark.asyncio
async def test_notifications_tab_marks_read_through_policy_and_store(tmp_path: Path):
    window = build_window(tmp_path=tmp_path, runtime_backend="local")
    row = window.notifications_store.insert_notification(
        category="watchlists",
        title="Queued",
        message="Queued for review",
        source_backend="local",
        source_entity_id="local:subscription:7",
        source_entity_kind="subscription",
        payload={"state": "queued"},
    )

    await window.notifications_controller.mark_read(row["id"], is_read=True)

    assert window.app_instance.require_ui_action_allowed.calls[-1]["action_id"] == "notifications.queue.update.local"
    stored = window.notifications_store.list_notifications(limit=1)[0]
    assert stored["id"] == row["id"]
    assert stored["is_read"] is True


@pytest.mark.asyncio
async def test_notifications_policy_denial_skips_store_reads_and_mutations(tmp_path: Path):
    denied_window = build_window(tmp_path=tmp_path, runtime_backend="local", policy_allowed=False)
    row = denied_window.notifications_store.insert_notification(
        category="watchlists",
        title="Queued",
        message="Queued for review",
        source_backend="local",
        source_entity_id="local:subscription:7",
        source_entity_kind="subscription",
    )

    rows = await denied_window.notifications_controller.load_rows()
    updated = await denied_window.notifications_controller.mark_read(row["id"], is_read=True)

    assert rows == []
    assert updated is False
    stored = denied_window.notifications_store.list_notifications(limit=1)[0]
    assert stored["is_read"] is False


@pytest.mark.asyncio
async def test_remote_delete_notification_preserves_restore_window_metadata(tmp_path: Path):
    window = build_window(tmp_path=tmp_path, runtime_backend="server")

    await window.backend_controller.delete_watch_item("server:watchlist_source:17")

    notification = window.notifications_store.list_notifications(limit=1)[0]
    assert notification["payload"]["restore_window_seconds"] == 10
    assert notification["payload"]["restore_expires_at"] == "2026-04-21T12:00:00Z"
