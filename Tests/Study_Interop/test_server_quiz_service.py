from unittest.mock import Mock

import pytest

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
