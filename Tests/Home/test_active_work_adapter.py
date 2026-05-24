from typing import get_type_hints
from types import SimpleNamespace

from tldw_chatbook.Home.active_work_adapter import (
    HomeConsoleLaunch,
    HomeControlAction,
    HomeControlResult,
    HomeControlResultStatus,
    LocalNotificationHomeActiveWorkAdapter,
    UnavailableHomeActiveWorkAdapter,
)
from tldw_chatbook.Home.dashboard_state import HomeActiveWorkItem, summarize_home_dashboard
from tldw_chatbook.Notifications.notifications_scope_service import ServerEventScopeRequiredError
from tldw_chatbook.runtime_policy.types import RuntimeSourceState


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
    assert dashboard_input.runtime_source == "local"
    assert dashboard_input.server_configured is False


def test_home_adapter_surfaces_runtime_server_status_without_credentials():
    runtime_policy = SimpleNamespace(
        state=RuntimeSourceState(
            active_source="server",
            active_server_id="primary",
            server_configured=True,
            server_reachability="reachable",
            server_auth_state="auth_required",
            last_known_server_label="Primary Server",
        )
    )
    adapter = UnavailableHomeActiveWorkAdapter(runtime_policy=runtime_policy)

    dashboard_input = adapter.build_dashboard_input(
        providers_models={},
        has_recent_work=False,
    )
    status = summarize_home_dashboard(dashboard_input).sections[0].lines[0]

    assert dashboard_input.runtime_source == "server"
    assert dashboard_input.active_server_id == "primary"
    assert dashboard_input.server_label == "Primary Server"
    assert dashboard_input.server_configured is True
    assert dashboard_input.server_reachability == "reachable"
    assert dashboard_input.server_auth_state == "auth_required"
    assert "Mode: Server" in status
    assert "Server: Auth required" in status


def test_home_adapter_runtime_server_status_matrix_is_source_honest():
    cases = [
        (
            RuntimeSourceState(active_source="local"),
            "Mode: Local",
            "Server: Not configured (local mode)",
        ),
        (
            RuntimeSourceState(
                active_source="server",
                active_server_id=None,
                server_configured=False,
            ),
            "Mode: Server",
            "Server: Missing active server",
        ),
        (
            RuntimeSourceState(
                active_source="server",
                active_server_id=None,
                server_configured=True,
                server_reachability="reachable",
                server_auth_state="authenticated",
            ),
            "Mode: Server",
            "Server: Missing active server",
        ),
        (
            RuntimeSourceState(
                active_source="server",
                active_server_id="primary",
                server_configured=True,
                server_reachability="unreachable",
                server_auth_state="unknown",
            ),
            "Mode: Server",
            "Server: Unreachable",
        ),
        (
            RuntimeSourceState(
                active_source="server",
                active_server_id="primary",
                server_configured=True,
                server_reachability="reachable",
                server_auth_state="authenticated",
            ),
            "Mode: Server",
            "Server: Ready",
        ),
    ]

    for runtime_state, mode_text, server_text in cases:
        adapter = UnavailableHomeActiveWorkAdapter(
            runtime_policy=SimpleNamespace(state=runtime_state)
        )
        dashboard_input = adapter.build_dashboard_input(
            providers_models={"OpenAI": ["gpt-4.1"]},
            has_recent_work=False,
        )
        status = summarize_home_dashboard(dashboard_input).sections[0].lines[0]

        assert mode_text in status
        assert server_text in status


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
    assert dashboard_input.active_work_items[0].source == "Watchlists"
    assert dashboard_input.active_work_items[0].status == "failed"
    assert dashboard_input.active_work_items[0].detail_route == "subscriptions"
    assert dashboard_input.active_work_items[0].console_available is True


def test_local_notification_adapter_opens_local_watchlist_run_details():
    class FakeWatchlistsService:
        def list_home_run_snapshot(self, *, limit=20):
            return [
                {
                    "id": "local:watchlist_run:5",
                    "run_id": 5,
                    "source_id": 7,
                    "status": "failed",
                    "source_title": "Daily security feed",
                    "backend": "local",
                },
            ]

    adapter = LocalNotificationHomeActiveWorkAdapter(
        watchlist_service=FakeWatchlistsService()
    )

    result = adapter.handle_control(
        HomeControlAction.OPEN_DETAILS,
        target_id="local:watchlist_run:5",
        target_route="subscriptions",
    )
    missing_result = adapter.handle_control(
        HomeControlAction.OPEN_DETAILS,
        target_id="local:watchlist_run:404",
        target_route="subscriptions",
    )

    assert result.status is HomeControlResultStatus.HANDLED
    assert result.target_id == "local:watchlist_run:5"
    assert result.target_route == "subscriptions"
    assert result.message == "Opening Watchlists run details for Daily security feed."

    assert missing_result.status is HomeControlResultStatus.UNAVAILABLE
    assert missing_result.target_id == "local:watchlist_run:404"


def test_local_notification_adapter_opens_local_watchlist_run_details_with_synthesized_id():
    class FakeWatchlistsService:
        def list_home_run_snapshot(self, *, limit=20):
            return [
                {
                    "run_id": 6,
                    "source_id": 8,
                    "status": "running",
                    "source_title": "Running release feed",
                    "backend": "local",
                },
            ]

    adapter = LocalNotificationHomeActiveWorkAdapter(
        watchlist_service=FakeWatchlistsService()
    )

    result = adapter.handle_control(
        HomeControlAction.OPEN_DETAILS,
        target_id="local:watchlist_run:6",
        target_route="subscriptions",
    )

    assert result.status is HomeControlResultStatus.HANDLED
    assert result.target_id == "local:watchlist_run:6"
    assert result.target_route == "subscriptions"
    assert result.message == "Opening Watchlists run details for Running release feed."


def test_local_notification_adapter_opens_local_watchlist_run_in_console():
    class FakeWatchlistsService:
        def list_home_run_snapshot(self, *, limit=20):
            return [
                {
                    "id": "local:watchlist_run:5",
                    "run_id": 5,
                    "job_id": 31,
                    "source_id": 7,
                    "status": "failed",
                    "source_title": "Daily security feed",
                    "backend": "local",
                },
            ]

    adapter = LocalNotificationHomeActiveWorkAdapter(
        watchlist_service=FakeWatchlistsService()
    )

    dashboard_input = adapter.build_dashboard_input(
        providers_models={"OpenAI": ["gpt-4.1"]},
        has_recent_work=False,
    )
    result = adapter.handle_control(
        HomeControlAction.OPEN_IN_CONSOLE,
        target_id="local:watchlist_run:5",
        target_route="chat",
    )
    missing_result = adapter.handle_control(
        HomeControlAction.OPEN_IN_CONSOLE,
        target_id="local:watchlist_run:404",
        target_route="chat",
    )

    assert dashboard_input.active_work_items[0].console_available is True
    assert result.status is HomeControlResultStatus.HANDLED
    assert result.console_launch is not None
    assert result.console_launch.source == "Watchlists"
    assert result.console_launch.title == "Daily security feed"
    assert result.console_launch.status == "failed"
    assert result.console_launch.recovery == "Review the Watchlists run details or retry from Watchlists."
    assert result.console_launch.action_label == "Open Watchlists run"
    assert result.console_launch.payload == {
        "run_id": 5,
        "job_id": 31,
        "source_id": 7,
        "target_id": "local:watchlist_run:5",
    }
    assert missing_result.status is HomeControlResultStatus.UNAVAILABLE
    assert missing_result.console_launch is None


def test_local_notification_adapter_maps_console_saved_chatbook_to_active_work():
    class FakeChatbookService:
        def __init__(self):
            self.calls = []

        def list_home_artifact_snapshot(self, *, limit=20):
            self.calls.append({"limit": limit})
            return [
                {
                    "chatbook_id": 77,
                    "id": "77",
                    "name": "Grounded Answer",
                    "description": "Saved from Console assistant response.",
                    "metadata": {
                        "artifact_source": "console",
                        "artifact_kind": "assistant-response",
                        "message_id": "msg-456",
                        "provider": "OpenAI",
                        "model": "gpt-4.1",
                        "content": "Grounded answer body.",
                        "content_truncated": False,
                    },
                    "updated_at": "2026-05-05T20:00:00Z",
                },
            ]

    service = FakeChatbookService()
    adapter = LocalNotificationHomeActiveWorkAdapter(chatbook_service=service)
    adapter.refresh_chatbook_artifact_snapshot()

    dashboard_input = adapter.build_dashboard_input(
        providers_models={"OpenAI": ["gpt-4.1"]},
        has_recent_work=False,
    )

    assert service.calls == [{"limit": 20}]
    assert dashboard_input.active_work_items == (
        HomeActiveWorkItem(
            item_id="local:chatbook:77",
            title="Grounded Answer",
            source="Artifacts",
            status="ready",
            detail_route="artifacts",
            console_available=True,
        ),
    )


def test_local_notification_adapter_maps_only_latest_console_saved_chatbook_to_active_work():
    class FakeChatbookService:
        def list_home_artifact_snapshot(self, *, limit=20):
            return [
                {
                    "chatbook_id": 78,
                    "name": "Latest Grounded Answer",
                    "metadata": {
                        "artifact_source": "console",
                        "artifact_kind": "assistant-response",
                    },
                },
                {
                    "chatbook_id": 77,
                    "name": "Older Grounded Answer",
                    "metadata": {
                        "artifact_source": "console",
                        "artifact_kind": "assistant-response",
                    },
                },
            ]

    adapter = LocalNotificationHomeActiveWorkAdapter(
        chatbook_service=FakeChatbookService()
    )
    adapter.refresh_chatbook_artifact_snapshot()

    dashboard_input = adapter.build_dashboard_input(
        providers_models={"OpenAI": ["gpt-4.1"]},
        has_recent_work=False,
    )

    assert [item.item_id for item in dashboard_input.active_work_items] == [
        "local:chatbook:78"
    ]


def test_local_notification_adapter_opens_console_saved_chatbook_details_and_console():
    class FakeChatbookService:
        def list_home_artifact_snapshot(self, *, limit=20):
            return [
                {
                    "chatbook_id": 77,
                    "id": "77",
                    "name": "Grounded Answer",
                    "description": "Saved from Console assistant response.",
                    "file_path": "/tmp/grounded-answer.chatbook",
                    "tags": ["console", "artifact"],
                    "categories": ["Console", "Artifacts"],
                    "metadata": {
                        "artifact_source": "console",
                        "artifact_kind": "assistant-response",
                        "conversation_id": "conv-123",
                        "message_id": "msg-456",
                        "message_role": "Assistant",
                        "provider": "OpenAI",
                        "model": "gpt-4.1",
                        "content": "Grounded answer body.",
                        "content_truncated": False,
                    },
                    "updated_at": "2026-05-05T20:00:00Z",
                },
            ]

    adapter = LocalNotificationHomeActiveWorkAdapter(
        chatbook_service=FakeChatbookService()
    )
    adapter.refresh_chatbook_artifact_snapshot()

    detail_result = adapter.handle_control(
        HomeControlAction.OPEN_DETAILS,
        target_id="local:chatbook:77",
        target_route="artifacts",
    )
    console_result = adapter.handle_control(
        HomeControlAction.OPEN_IN_CONSOLE,
        target_id="local:chatbook:77",
        target_route="chat",
    )

    assert detail_result.status is HomeControlResultStatus.HANDLED
    assert detail_result.target_id == "local:chatbook:77"
    assert detail_result.target_route == "artifacts"
    assert detail_result.message == "Opening Artifacts for Grounded Answer."

    assert console_result.status is HomeControlResultStatus.HANDLED
    assert console_result.console_launch is not None
    assert console_result.console_launch.source == "artifacts"
    assert console_result.console_launch.title == "Grounded Answer"
    assert console_result.console_launch.status == "ready"
    assert console_result.console_launch.recovery == (
        "Review this Chatbook artifact in Console or return to Home."
    )
    assert console_result.console_launch.action_label == "Open Chatbook artifact"
    assert console_result.console_launch.payload is not None
    assert console_result.console_launch.payload["file_path"].endswith(
        "/grounded-answer.chatbook"
    )
    assert console_result.console_launch.payload == {
        "target_id": "local:chatbook:77",
        "chatbook_id": 77,
        "record_id": "77",
        "file_path": console_result.console_launch.payload["file_path"],
        "description": "Saved from Console assistant response.",
        "tags": "console, artifact",
        "categories": "Console, Artifacts",
        "updated_at": "2026-05-05T20:00:00Z",
        "artifact_source": "console",
        "artifact_kind": "assistant-response",
        "conversation_id": "conv-123",
        "message_id": "msg-456",
        "message_role": "Assistant",
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "content_preview": "Grounded answer body.",
        "content_truncated": False,
    }


def test_local_notification_adapter_preserves_console_saved_chatbook_citation_metadata():
    class FakeChatbookService:
        def list_home_artifact_snapshot(self, *, limit=20):
            return [
                {
                    "chatbook_id": 77,
                    "id": "77",
                    "name": "Grounded Answer",
                    "metadata": {
                        "artifact_source": "console",
                        "artifact_kind": "assistant-response",
                        "content": "The credential expired [S1].",
                        "citation_validation": {
                            "status": "validated",
                            "citations": [
                                {
                                    "evidence_id": "S1",
                                    "source_id": "note-1",
                                    "status": "validated",
                                    "quote": "The credential expired [S1].",
                                }
                            ],
                            "cited_evidence_ids": ["S1"],
                            "unknown_citation_ids": [],
                            "uncited_evidence_ids": [],
                            "recovery": "",
                        },
                        "evidence_bundle": {
                            "bundle_id": "library-rag:incident",
                            "query": "Why did the incident happen?",
                            "status": "available",
                            "references": [
                                {
                                    "evidence_id": "S1",
                                    "source_id": "note-1",
                                    "source_type": "note",
                                    "title": "Incident Review",
                                    "snippet": "Expired credential caused the incident.",
                                    "authority_label": "Source authority: local",
                                    "status": "available",
                                }
                            ],
                        },
                    },
                },
            ]

    adapter = LocalNotificationHomeActiveWorkAdapter(
        chatbook_service=FakeChatbookService()
    )
    adapter.refresh_chatbook_artifact_snapshot()

    console_result = adapter.handle_control(
        HomeControlAction.OPEN_IN_CONSOLE,
        target_id="local:chatbook:77",
        target_route="chat",
    )

    assert console_result.status is HomeControlResultStatus.HANDLED
    payload = console_result.console_launch.payload
    assert payload["citation_status"] == "validated"
    assert payload["citation_cited_evidence_ids"] == "S1"
    assert payload["citation_count"] == 1
    assert payload["evidence_bundle_id"] == "library-rag:incident"
    assert payload["evidence_source_count"] == 1
    assert payload["evidence_snippet_count"] == 1


def test_local_notification_adapter_bounds_console_saved_chatbook_preview():
    long_content = "x" * 1500

    class FakeChatbookService:
        def list_home_artifact_snapshot(self, *, limit=20):
            return [
                {
                    "chatbook_id": 77,
                    "id": "77",
                    "name": "Long Answer",
                    "metadata": {
                        "artifact_source": "console",
                        "artifact_kind": "assistant-response",
                        "content": long_content,
                        "content_truncated": False,
                    },
                },
            ]

    adapter = LocalNotificationHomeActiveWorkAdapter(
        chatbook_service=FakeChatbookService()
    )
    adapter.refresh_chatbook_artifact_snapshot()

    result = adapter.handle_control(
        HomeControlAction.OPEN_IN_CONSOLE,
        target_id="local:chatbook:77",
        target_route="chat",
    )

    assert result.console_launch is not None
    payload = result.console_launch.payload
    assert payload is not None
    assert payload["content_preview"] == "x" * 1000
    assert payload["content_truncated"] is True


def test_local_notification_adapter_sanitizes_console_saved_chatbook_payload_text():
    class FakeChatbookService:
        def list_home_artifact_snapshot(self, *, limit=20):
            return [
                {
                    "chatbook_id": 77,
                    "id": "77",
                    "name": "Unsafe Answer",
                    "description": "<script>alert('x')</script>\nDescription",
                    "file_path": "../../secret.chatbook",
                    "tags": ["safe", "<script>tag</script>"],
                    "categories": ["Console\nArtifacts"],
                    "metadata": {
                        "artifact_source": "console",
                        "artifact_kind": "assistant-response",
                        "provider": "OpenAI\n<script>",
                        "model": "gpt-4.1",
                        "content": "<script>alert('x')</script>\nGrounded answer body.",
                        "content_truncated": False,
                        "citation_validation": {
                            "status": "<script>validated</script>",
                            "cited_evidence_ids": ["S1", "javascript:bad"],
                            "citations": [{"evidence_id": "S1"}],
                        },
                        "evidence_bundle": {
                            "bundle_id": "bundle-onclick=bad",
                            "query": "<script>query</script>",
                            "references": [{"snippet": "safe"}],
                        },
                    },
                },
            ]

    adapter = LocalNotificationHomeActiveWorkAdapter(
        chatbook_service=FakeChatbookService()
    )
    adapter.refresh_chatbook_artifact_snapshot()

    result = adapter.handle_control(
        HomeControlAction.OPEN_IN_CONSOLE,
        target_id="local:chatbook:77",
        target_route="chat",
    )

    assert result.console_launch is not None
    payload = result.console_launch.payload
    assert payload is not None
    assert payload["file_path"] is None
    assert "<script" not in str(payload["description"]).lower()
    assert "<script" not in str(payload["tags"]).lower()
    assert "\n" not in str(payload["categories"])
    assert "<script" not in str(payload.get("provider", "")).lower()
    assert "<script" not in str(payload["content_preview"]).lower()
    assert "<script" not in str(payload.get("citation_status", "")).lower()
    assert "javascript:" not in str(payload.get("citation_cited_evidence_ids", "")).lower()
    assert "onclick=" not in str(payload.get("evidence_bundle_id", "")).lower()
    assert "<script" not in str(payload.get("evidence_query", "")).lower()


def test_local_notification_adapter_bounds_console_saved_chatbook_payload_fields():
    class FakeChatbookService:
        def list_home_artifact_snapshot(self, *, limit=20):
            return [
                {
                    "chatbook_id": 77,
                    "id": "77",
                    "name": "Large Answer",
                    "description": "d" * 1500,
                    "file_path": f"/tmp/{'f' * 2500}.chatbook",
                    "tags": ["t" * 1500],
                    "categories": ["c" * 1500],
                    "metadata": {
                        "artifact_source": "console",
                        "artifact_kind": "assistant-response",
                        "provider": "p" * 300,
                        "model": "m" * 300,
                        "content": "body",
                    },
                },
            ]

    adapter = LocalNotificationHomeActiveWorkAdapter(
        chatbook_service=FakeChatbookService()
    )
    adapter.refresh_chatbook_artifact_snapshot()

    result = adapter.handle_control(
        HomeControlAction.OPEN_IN_CONSOLE,
        target_id="local:chatbook:77",
        target_route="chat",
    )

    assert result.console_launch is not None
    payload = result.console_launch.payload
    assert payload is not None
    assert len(str(payload["description"])) <= 1000
    assert len(str(payload["file_path"])) <= 2000
    assert len(str(payload["tags"])) <= 1000
    assert len(str(payload["categories"])) <= 1000
    assert len(str(payload["provider"])) <= 256
    assert len(str(payload["model"])) <= 256


def test_local_notification_adapter_dashboard_build_uses_cached_chatbook_snapshot():
    class FakeChatbookService:
        def __init__(self):
            self.calls = 0

        def list_home_artifact_snapshot(self, *, limit=20):
            self.calls += 1
            return [
                {
                    "chatbook_id": 77,
                    "name": "Grounded Answer",
                    "metadata": {
                        "artifact_source": "console",
                        "artifact_kind": "assistant-response",
                    },
                },
            ]

    service = FakeChatbookService()
    adapter = LocalNotificationHomeActiveWorkAdapter(chatbook_service=service)
    adapter.refresh_chatbook_artifact_snapshot()
    assert service.calls == 1

    dashboard_input = adapter.build_dashboard_input(
        providers_models={"OpenAI": ["gpt-4.1"]},
        has_recent_work=False,
    )

    assert service.calls == 1
    assert [item.item_id for item in dashboard_input.active_work_items] == [
        "local:chatbook:77"
    ]


def test_local_notification_adapter_fails_closed_when_chatbook_snapshot_unavailable():
    class FakeWatchlistsService:
        def list_home_run_snapshot(self, *, limit=20):
            return [
                {
                    "id": "local:watchlist_run:5",
                    "title": "Daily Feed",
                    "status": "running",
                    "target_route": "watchlists",
                    "console_available": True,
                },
            ]

    class BrokenChatbookService:
        def list_home_artifact_snapshot(self, *, limit=20):
            raise RuntimeError("chatbook registry unavailable")

    adapter = LocalNotificationHomeActiveWorkAdapter(
        watchlist_service=FakeWatchlistsService(),
        chatbook_service=BrokenChatbookService(),
    )
    adapter.refresh_chatbook_artifact_snapshot()

    dashboard_input = adapter.build_dashboard_input(
        providers_models={"OpenAI": ["gpt-4.1"]},
        has_recent_work=True,
    )

    assert [item.item_id for item in dashboard_input.active_work_items] == [
        "local:watchlist_run:5"
    ]


def test_local_notification_adapter_fails_closed_when_watchlist_snapshot_unavailable():
    class BrokenWatchlistsService:
        def list_home_run_snapshot(self, *, limit=20):
            raise RuntimeError("watchlist store unavailable")

    adapter = LocalNotificationHomeActiveWorkAdapter(
        watchlist_service=BrokenWatchlistsService()
    )

    dashboard_input = adapter.build_dashboard_input(
        providers_models={"OpenAI": ["gpt-4.1"]},
        has_recent_work=True,
    )

    assert dashboard_input.model_ready is True
    assert dashboard_input.active_work_items == ()


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


def test_local_notification_adapter_distinguishes_server_events_from_local_notifications():
    class FakeNotificationsService:
        def list_queue(self, *, limit=100, include_dismissed=False, category=None):
            return [
                {"id": "local-read", "is_read": True},
                {"id": "local-unread", "is_read": False},
            ]

    class FakeServerEventService:
        def list_observed_server_feed(self, *, limit=20, mark_presented=False):
            return {
                "items": [{"record_id": "server:notification:8", "title": "Server event"}],
                "total": 1,
                "backend": "server",
                "source": "event_state_repository",
                "replay": {
                    "state": "available",
                    "server_refetch_required": False,
                },
            }

    adapter = LocalNotificationHomeActiveWorkAdapter(
        notification_service=FakeNotificationsService(),
        server_event_service=FakeServerEventService(),
    )

    dashboard_input = adapter.build_dashboard_input(
        providers_models={"OpenAI": ["gpt-4.1"]},
        has_recent_work=False,
    )
    system_status = summarize_home_dashboard(dashboard_input).sections[3].lines

    assert dashboard_input.notification_count == 1
    assert dashboard_input.server_event_count == 1
    assert dashboard_input.server_event_state == "available"
    assert "Local notifications: 1 unread" not in system_status
    assert "Server events: 1 observed via server event feed" in system_status


def test_local_notification_adapter_surfaces_server_event_replay_gap_requery_state():
    class FakeServerEventService:
        def list_observed_server_feed(self, *, limit=20, mark_presented=False):
            return {
                "items": [],
                "total": 0,
                "replay": {
                    "state": "retention_gap",
                    "server_refetch_required": True,
                    "last_pruned_cursor": "cursor-10",
                },
            }

    adapter = LocalNotificationHomeActiveWorkAdapter(
        server_event_service=FakeServerEventService()
    )

    dashboard_input = adapter.build_dashboard_input(
        providers_models={"OpenAI": ["gpt-4.1"]},
        has_recent_work=False,
    )
    system_status = summarize_home_dashboard(dashboard_input).sections[3].lines

    assert dashboard_input.server_event_state == "requery_required"
    assert dashboard_input.server_event_recovery == "Requery server events from the active server."
    assert "Server events: Replay gap - requery server events" in system_status


def test_local_notification_adapter_surfaces_server_event_reconnect_state():
    class FakeServerEventService:
        def list_observed_server_feed(self, *, limit=20, mark_presented=False):
            raise ServerEventScopeRequiredError("No selected event scope.")

    adapter = LocalNotificationHomeActiveWorkAdapter(
        server_event_service=FakeServerEventService()
    )

    dashboard_input = adapter.build_dashboard_input(
        providers_models={"OpenAI": ["gpt-4.1"]},
        has_recent_work=False,
    )
    system_status = summarize_home_dashboard(dashboard_input).sections[3].lines

    assert dashboard_input.server_event_state == "reconnect_required"
    assert dashboard_input.server_event_recovery == "Reconnect or select an active server."
    assert "Server events: Reconnect required" in system_status


def test_local_notification_adapter_surfaces_server_event_unavailable_state():
    class FakeServerEventService:
        def list_observed_server_feed(self, *, limit=20, mark_presented=False):
            raise RuntimeError("Server notifications backend is unavailable.")

    adapter = LocalNotificationHomeActiveWorkAdapter(
        server_event_service=FakeServerEventService()
    )

    dashboard_input = adapter.build_dashboard_input(
        providers_models={"OpenAI": ["gpt-4.1"]},
        has_recent_work=False,
    )
    system_status = summarize_home_dashboard(dashboard_input).sections[3].lines

    assert dashboard_input.server_event_state == "unavailable"
    assert dashboard_input.server_event_recovery == "Server event feed is unavailable."
    assert "Server events: Unavailable" in system_status


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
