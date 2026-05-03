from typing import get_type_hints

from tldw_chatbook.Home.active_work_adapter import (
    HomeConsoleLaunch,
    HomeControlAction,
    HomeControlResult,
    HomeControlResultStatus,
    LocalNotificationHomeActiveWorkAdapter,
    UnavailableHomeActiveWorkAdapter,
)


def test_unavailable_home_adapter_builds_dashboard_input_from_runtime_context():
    adapter = UnavailableHomeActiveWorkAdapter()

    dashboard_input = adapter.build_dashboard_input(
        providers_models={"OpenAI": ["gpt-4.1"]},
        has_recent_work=True,
    )

    assert dashboard_input.model_ready is True
    assert dashboard_input.has_recent_work is True
    assert dashboard_input.pending_approval_count == 0
    assert dashboard_input.active_run_count == 0
    assert dashboard_input.active_detail_route == "chat"


def test_local_notification_adapter_counts_unread_local_notifications():
    class FakeNotificationsService:
        def __init__(self):
            self.calls = []

        def list_queue(self, *, limit=100, include_dismissed=False, category=None):
            self.calls.append(
                {
                    "limit": limit,
                    "include_dismissed": include_dismissed,
                    "category": category,
                }
            )
            return [
                {"id": "read", "is_read": True},
                {"id": "unread-1", "is_read": False},
                {"id": "unread-2", "is_read": False},
            ]

    service = FakeNotificationsService()
    adapter = LocalNotificationHomeActiveWorkAdapter(notification_service=service)

    dashboard_input = adapter.build_dashboard_input(
        providers_models={"OpenAI": ["gpt-4.1"]},
        has_recent_work=True,
    )

    assert dashboard_input.model_ready is True
    assert dashboard_input.has_recent_work is True
    assert dashboard_input.notification_count == 2
    assert dashboard_input.pending_approval_count == 0
    assert dashboard_input.active_run_count == 0
    assert service.calls == [{"limit": 100, "include_dismissed": False, "category": None}]


def test_local_notification_adapter_maps_local_watchlist_runs_to_active_work():
    class FakeWatchlistsService:
        def __init__(self):
            self.calls = []

        def list_home_run_snapshot(self, *, limit=20):
            self.calls.append({"limit": limit})
            return [
                {
                    "id": "local:watchlist_run:5",
                    "run_id": 5,
                    "source_id": 7,
                    "status": "failed",
                    "source_title": "Daily security feed",
                    "backend": "local",
                },
                {
                    "id": "local:watchlist_run:4",
                    "run_id": 4,
                    "source_id": 7,
                    "status": "completed",
                    "source_title": "Completed feed",
                    "backend": "local",
                },
                {
                    "id": "local:watchlist_run:3",
                    "run_id": 3,
                    "source_id": 8,
                    "status": "queued",
                    "source_title": "Queued release feed",
                    "backend": "local",
                },
            ]

    service = FakeWatchlistsService()
    adapter = LocalNotificationHomeActiveWorkAdapter(watchlist_service=service)

    dashboard_input = adapter.build_dashboard_input(
        providers_models={"OpenAI": ["gpt-4.1"]},
        has_recent_work=False,
    )

    assert service.calls == [{"limit": 20}]
    assert [item.item_id for item in dashboard_input.active_work_items] == [
        "local:watchlist_run:5",
        "local:watchlist_run:3",
    ]
    assert dashboard_input.active_work_items[0].title == "Daily security feed"
    assert dashboard_input.active_work_items[0].source == "W+C"
    assert dashboard_input.active_work_items[0].status == "failed"
    assert dashboard_input.active_work_items[0].detail_route == "subscriptions"
    assert dashboard_input.active_work_items[0].console_available is False


def test_local_notification_adapter_fails_closed_when_snapshot_unavailable():
    class BrokenNotificationsService:
        def list_queue(self, *, limit=100, include_dismissed=False, category=None):
            raise RuntimeError("notification store unavailable")

    adapter = LocalNotificationHomeActiveWorkAdapter(
        notification_service=BrokenNotificationsService()
    )

    dashboard_input = adapter.build_dashboard_input(
        providers_models={},
        has_recent_work=False,
    )

    assert dashboard_input.model_ready is False
    assert dashboard_input.notification_count == 0
    assert dashboard_input.pending_approval_count == 0
    assert dashboard_input.active_run_count == 0


def test_unavailable_home_adapter_returns_honest_recovery_result():
    adapter = UnavailableHomeActiveWorkAdapter()

    result = adapter.handle_control(HomeControlAction.APPROVE, target_id="approval-1")

    assert result.action is HomeControlAction.APPROVE
    assert result.status is HomeControlResultStatus.UNAVAILABLE
    assert result.target_id == "approval-1"
    assert result.severity == "warning"
    assert "Approve is not connected" in result.message
    assert result.recovery_route == "chat"


def test_unavailable_home_adapter_keeps_detail_and_console_actions_recoverable():
    adapter = UnavailableHomeActiveWorkAdapter()

    detail_result = adapter.handle_control(
        HomeControlAction.OPEN_DETAILS,
        target_route="workflows",
    )
    console_result = adapter.handle_control(
        HomeControlAction.OPEN_IN_CONSOLE,
        target_route="chat",
    )

    assert detail_result.status is HomeControlResultStatus.UNAVAILABLE
    assert detail_result.target_route is None
    assert detail_result.recovery_route == "workflows"
    assert "Open details is not connected" in detail_result.message

    assert console_result.status is HomeControlResultStatus.UNAVAILABLE
    assert console_result.console_launch is None
    assert console_result.recovery_route == "chat"
    assert "Open in Console is not connected" in console_result.message


def test_home_control_result_can_carry_route_and_console_launch_payload():
    launch = HomeConsoleLaunch(
        source="workflows",
        title="Daily digest",
        payload={"run_id": "run-1"},
    )

    detail_result = HomeControlResult(
        action=HomeControlAction.OPEN_DETAILS,
        status=HomeControlResultStatus.HANDLED,
        message="Opening workflow details.",
        target_route="workflows",
    )
    console_result = HomeControlResult(
        action=HomeControlAction.OPEN_IN_CONSOLE,
        status=HomeControlResultStatus.HANDLED,
        message="Opening Console for Daily digest.",
        console_launch=launch,
    )

    assert detail_result.target_route == "workflows"
    assert console_result.console_launch is launch
    assert console_result.console_launch.payload == {"run_id": "run-1"}


def test_home_control_result_status_contract_uses_enum_only():
    assert get_type_hints(HomeControlResult)["status"] is HomeControlResultStatus
