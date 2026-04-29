import inspect
from unittest.mock import Mock

import pytest

import tldw_chatbook.Study_Interop.server_quiz_service as quiz_module
from tldw_chatbook.Study_Interop.server_quiz_service import ServerQuizService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeQuizClient:
    def __init__(self):
        self.calls = []

    async def list_quizzes(self, **kwargs):
        self.calls.append(("list_quizzes", kwargs))
        return {"items": [], "total": 0}

    async def get_quiz(self, quiz_id):
        self.calls.append(("get_quiz", quiz_id))
        return {"id": quiz_id, "name": "Quiz"}

    async def create_quiz(self, request_data):
        self.calls.append(("create_quiz", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 7, "name": request_data.name}

    async def delete_quiz(self, quiz_id, **kwargs):
        self.calls.append(("delete_quiz", quiz_id, kwargs))
        return {"status": "deleted"}

    async def list_quiz_questions(self, quiz_id, **kwargs):
        self.calls.append(("list_quiz_questions", quiz_id, kwargs))
        return {"items": [], "total": 0}

    async def create_quiz_question(self, quiz_id, request_data):
        self.calls.append(("create_quiz_question", quiz_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 11, "quiz_id": quiz_id}

    async def delete_quiz_question(self, quiz_id, question_id, **kwargs):
        self.calls.append(("delete_quiz_question", quiz_id, question_id, kwargs))
        return {"status": "deleted"}

    async def start_quiz_attempt(self, quiz_id):
        self.calls.append(("start_quiz_attempt", quiz_id))
        return {"id": 21, "quiz_id": quiz_id}

    async def submit_quiz_attempt(self, attempt_id, request_data):
        self.calls.append(("submit_quiz_attempt", attempt_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": attempt_id, "status": "submitted"}

    async def list_quiz_attempts(self, **kwargs):
        self.calls.append(("list_quiz_attempts", kwargs))
        return {"items": [], "total": 0}

    async def get_quiz_attempt(self, attempt_id, **kwargs):
        self.calls.append(("get_quiz_attempt", attempt_id, kwargs))
        return {"id": attempt_id, "status": "submitted"}


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


class FreshClientProvider:
    def __init__(self, factory):
        self.factory = factory
        self.build_calls = 0
        self.clients = []

    def build_client(self):
        self.build_calls += 1
        client = self.factory()
        self.clients.append(client)
        return client


class ExplodingClientProvider:
    def __init__(self):
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        raise AssertionError("provider should not be used when direct client exists")


def test_server_quiz_service_module_does_not_reference_legacy_config_client_builders():
    source = inspect.getsource(quiz_module)

    assert "build_runtime_api_client_from_config" not in source
    assert "build_runtime_api_client(app_config" not in source


@pytest.mark.asyncio
async def test_server_quiz_service_direct_client_takes_precedence_over_provider():
    client = FakeQuizClient()
    provider = ExplodingClientProvider()
    service = ServerQuizService(client=client, client_provider=provider)

    result = await service.list_quizzes(q="bio")

    assert result["total"] == 0
    assert provider.build_calls == 0
    assert client.calls == [("list_quizzes", {"q": "bio", "limit": 100, "offset": 0})]


@pytest.mark.asyncio
async def test_server_quiz_service_from_server_context_provider_is_lazy():
    client = FakeQuizClient()
    provider = FakeClientProvider(client)
    service = ServerQuizService.from_server_context_provider(provider)

    assert isinstance(service, ServerQuizService)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0

    result = await service.list_quizzes(q="bio")

    assert result["total"] == 0
    assert service.client is None
    assert provider.build_calls == 1
    assert client.calls == [("list_quizzes", {"q": "bio", "limit": 100, "offset": 0})]


@pytest.mark.asyncio
async def test_server_quiz_service_re_resolves_provider_without_service_local_client_cache():
    provider = FreshClientProvider(FakeQuizClient)
    service = ServerQuizService.from_server_context_provider(provider)

    await service.list_quizzes(q="bio")
    await service.list_quizzes(q="chem")

    assert service.client is None
    assert provider.build_calls == 2
    assert len(provider.clients) == 2
    assert provider.clients[0] is not provider.clients[1]
    assert provider.clients[0].calls == [("list_quizzes", {"q": "bio", "limit": 100, "offset": 0})]
    assert provider.clients[1].calls == [("list_quizzes", {"q": "chem", "limit": 100, "offset": 0})]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


def test_server_quiz_service_from_config_returns_provider_backed_service():
    service = ServerQuizService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerQuizService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client


@pytest.mark.asyncio
async def test_server_quiz_service_enforces_policy_actions():
    client = FakeQuizClient()
    policy = Mock()
    service = ServerQuizService(client=client, policy_enforcer=policy)

    await service.list_quizzes(q="bio")
    await service.get_quiz(7)
    await service.create_quiz(name="Biology")
    await service.delete_quiz(7, expected_version=2)
    await service.list_questions(7, include_answers=True)
    await service.create_question(
        7,
        question_type="true_false",
        question_text="Q?",
        correct_answer="true",
    )
    await service.delete_question(11, quiz_id=7, expected_version=2)
    await service.start_attempt(7)
    await service.submit_attempt(21, answers=[{"question_id": 11, "user_answer": "true"}])
    await service.list_attempts(quiz_id=7)
    await service.get_attempt(21)

    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "quiz.list.server",
        "quiz.detail.server",
        "quiz.create.server",
        "quiz.delete.server",
        "quiz.question.list.server",
        "quiz.question.detail.server",
        "quiz.question.detail.server",
        "quiz.attempt.create.server",
        "quiz.attempt.observe.server",
        "quiz.attempt.observe.server",
        "quiz.attempt.observe.server",
    ]


@pytest.mark.asyncio
async def test_server_quiz_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_unreachable",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeQuizClient()
    service = ServerQuizService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_quizzes(q="bio")

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
