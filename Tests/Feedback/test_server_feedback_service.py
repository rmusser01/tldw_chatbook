from unittest.mock import Mock

import pytest

from tldw_chatbook.Feedback_Interop import ServerFeedbackService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeFeedbackClient:
    def __init__(self):
        self.calls = []

    async def submit_explicit_feedback(self, request_data):
        self.calls.append(("submit_explicit_feedback", request_data.model_dump(exclude_none=True, mode="json")))
        return {"ok": True, "feedback_id": "fb-1"}

    async def list_feedback(self, conversation_id):
        self.calls.append(("list_feedback", conversation_id))
        return {
            "ok": True,
            "feedback": [
                {
                    "id": "fb-1",
                    "conversation_id": conversation_id,
                    "message_id": "msg-1",
                    "query": "Summarize",
                    "helpful": True,
                }
            ],
        }

    async def update_feedback(self, feedback_id, request_data):
        self.calls.append(("update_feedback", feedback_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"ok": True, "feedback_id": feedback_id}

    async def delete_feedback(self, feedback_id):
        self.calls.append(("delete_feedback", feedback_id))
        return {"ok": True, "deleted": True}


@pytest.mark.asyncio
async def test_server_feedback_service_routes_crud_with_policy_actions():
    client = FakeFeedbackClient()
    policy = Mock()
    service = ServerFeedbackService(client=client, policy_enforcer=policy)

    submitted = await service.submit_feedback(
        conversation_id="conv-1",
        message_id="msg-1",
        feedback_type="helpful",
        helpful=True,
        query="Summarize",
    )
    listed = await service.list_feedback("conv-1")
    updated = await service.update_feedback("fb-1", issues=["missing_details"], user_notes="Needs detail")
    deleted = await service.delete_feedback("fb-1")

    assert submitted["feedback_id"] == "fb-1"
    assert listed["feedback"][0]["id"] == "fb-1"
    assert updated["feedback_id"] == "fb-1"
    assert deleted["deleted"] is True
    assert client.calls == [
        (
            "submit_explicit_feedback",
            {
                "conversation_id": "conv-1",
                "message_id": "msg-1",
                "feedback_type": "helpful",
                "helpful": True,
                "query": "Summarize",
            },
        ),
        ("list_feedback", "conv-1"),
        ("update_feedback", "fb-1", {"issues": ["missing_details"], "user_notes": "Needs detail"}),
        ("delete_feedback", "fb-1"),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "feedback.create.server",
        "feedback.list.server",
        "feedback.update.server",
        "feedback.delete.server",
    ]


@pytest.mark.asyncio
async def test_server_feedback_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_auth_required",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeFeedbackClient()
    service = ServerFeedbackService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_feedback("conv-1")

    assert exc.value.reason_code == "server_auth_required"
    assert client.calls == []


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


class ExplodingClientProvider:
    def __init__(self):
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        raise AssertionError("provider should not be used when direct client exists")


class FreshClientProvider:
    def __init__(self):
        self.build_calls = 0
        self.clients = []

    def build_client(self):
        self.build_calls += 1
        client = object()
        self.clients.append(client)
        return client


def test_server_feedback_service_direct_client_takes_precedence_over_provider():
    client = object()
    provider = ExplodingClientProvider()
    service = ServerFeedbackService(client=client, client_provider=provider)

    assert service._require_client() is client
    assert provider.build_calls == 0


def test_server_feedback_service_from_server_context_provider_is_lazy():
    client = object()
    provider = FakeClientProvider(client)
    service = ServerFeedbackService.from_server_context_provider(provider)

    assert isinstance(service, ServerFeedbackService)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0
    assert service._require_client() is client
    assert service.client is None
    assert provider.build_calls == 1


def test_server_feedback_service_re_resolves_provider_without_service_local_client_cache():
    provider = FreshClientProvider()
    service = ServerFeedbackService.from_server_context_provider(provider)

    first_client = service._require_client()
    second_client = service._require_client()

    assert service.client is None
    assert provider.build_calls == 2
    assert first_client is not second_client
    assert provider.clients == [first_client, second_client]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


def test_server_feedback_service_from_config_returns_provider_backed_service():
    service = ServerFeedbackService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerFeedbackService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client
