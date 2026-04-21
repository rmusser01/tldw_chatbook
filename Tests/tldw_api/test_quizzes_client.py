"""Tests for quiz endpoint wiring on the shared TLDW API client."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    QuizAttemptListResponse,
    QuizAttemptResponse,
    QuizAttemptSubmitRequest,
    QuizCreateRequest,
    QuizQuestionCreateRequest,
    QuizQuestionListResponse,
    QuizQuestionResponse,
    QuizResponse,
    QuizUpdateRequest,
    TLDWAPIClient,
)


@pytest.mark.asyncio
async def test_quiz_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "id": 7,
                "name": "Renal Review",
                "description": "Kidney basics",
                "workspace_id": None,
                "total_questions": 0,
                "time_limit_seconds": 300,
                "passing_score": 70,
                "deleted": False,
                "client_id": "server-client",
                "version": 1,
                "created_at": "2026-04-20T00:00:00Z",
                "last_modified": "2026-04-20T00:01:00Z",
            },
            {
                "items": [
                    {
                        "id": 7,
                        "name": "Renal Review",
                        "description": "Kidney basics",
                        "workspace_id": None,
                        "total_questions": 0,
                        "time_limit_seconds": 300,
                        "passing_score": 70,
                        "deleted": False,
                        "client_id": "server-client",
                        "version": 1,
                        "created_at": "2026-04-20T00:00:00Z",
                        "last_modified": "2026-04-20T00:01:00Z",
                    }
                ],
                "count": 1,
            },
            {
                "id": 7,
                "name": "Renal Review",
                "description": "Kidney basics",
                "workspace_id": None,
                "total_questions": 0,
                "time_limit_seconds": 300,
                "passing_score": 70,
                "deleted": False,
                "client_id": "server-client",
                "version": 1,
                "created_at": "2026-04-20T00:00:00Z",
                "last_modified": "2026-04-20T00:01:00Z",
            },
            {
                "id": 7,
                "name": "Renal Review v2",
                "description": "Kidney basics",
                "workspace_id": None,
                "total_questions": 0,
                "time_limit_seconds": 300,
                "passing_score": 70,
                "deleted": False,
                "client_id": "server-client",
                "version": 2,
                "created_at": "2026-04-20T00:00:00Z",
                "last_modified": "2026-04-20T00:02:00Z",
            },
            {"status": "deleted"},
            {
                "id": 11,
                "quiz_id": 7,
                "question_type": "fill_blank",
                "question_text": "The capital of France is ____.",
                "options": None,
                "correct_answer": "Paris",
                "explanation": "Paris is the capital city.",
                "hint": None,
                "hint_penalty_points": 0,
                "source_citations": None,
                "points": 2,
                "order_index": 0,
                "tags": ["geography"],
                "deleted": False,
                "client_id": "server-client",
                "version": 1,
                "created_at": "2026-04-20T00:00:00Z",
                "last_modified": "2026-04-20T00:01:00Z",
            },
            {
                "items": [
                    {
                        "id": 11,
                        "quiz_id": 7,
                        "question_type": "fill_blank",
                        "question_text": "The capital of France is ____.",
                        "options": None,
                        "hint": None,
                        "hint_penalty_points": 0,
                        "source_citations": None,
                        "points": 2,
                        "order_index": 0,
                        "tags": ["geography"],
                        "deleted": False,
                        "client_id": "server-client",
                        "version": 1,
                        "created_at": "2026-04-20T00:00:00Z",
                        "last_modified": "2026-04-20T00:01:00Z",
                    }
                ],
                "count": 1,
            },
            {
                "id": 11,
                "quiz_id": 7,
                "question_type": "fill_blank",
                "question_text": "The capital of France is ____.",
                "options": None,
                "correct_answer": "Paris",
                "explanation": "Paris is the capital city.",
                "hint": None,
                "hint_penalty_points": 0,
                "source_citations": None,
                "points": 2,
                "order_index": 0,
                "tags": ["geography"],
                "deleted": False,
                "client_id": "server-client",
                "version": 2,
                "created_at": "2026-04-20T00:00:00Z",
                "last_modified": "2026-04-20T00:02:00Z",
            },
            {"status": "deleted"},
            {
                "id": 41,
                "quiz_id": 7,
                "started_at": "2026-04-20T00:00:00Z",
                "completed_at": None,
                "score": None,
                "total_possible": 2,
                "time_spent_seconds": None,
                "answers": [],
                "questions": [
                    {
                        "id": 11,
                        "quiz_id": 7,
                        "question_type": "fill_blank",
                        "question_text": "The capital of France is ____.",
                        "options": None,
                        "hint": None,
                        "hint_penalty_points": 0,
                        "source_citations": None,
                        "points": 2,
                        "order_index": 0,
                        "tags": ["geography"],
                        "deleted": False,
                        "client_id": "server-client",
                        "version": 1,
                        "created_at": "2026-04-20T00:00:00Z",
                        "last_modified": "2026-04-20T00:01:00Z",
                    }
                ],
            },
            {
                "id": 41,
                "quiz_id": 7,
                "started_at": "2026-04-20T00:00:00Z",
                "completed_at": "2026-04-20T00:02:00Z",
                "score": 2,
                "total_possible": 2,
                "time_spent_seconds": 2,
                "answers": [
                    {
                        "question_id": 11,
                        "user_answer": "Paris",
                        "is_correct": True,
                        "correct_answer": "Paris",
                        "explanation": "Paris is the capital city.",
                        "points_awarded": 2,
                        "time_spent_ms": 1200,
                    }
                ],
            },
            {
                "items": [
                    {
                        "id": 41,
                        "quiz_id": 7,
                        "started_at": "2026-04-20T00:00:00Z",
                        "completed_at": "2026-04-20T00:02:00Z",
                        "score": 2,
                        "total_possible": 2,
                        "time_spent_seconds": 2,
                        "answers": [],
                    }
                ],
                "count": 1,
            },
            {
                "id": 41,
                "quiz_id": 7,
                "started_at": "2026-04-20T00:00:00Z",
                "completed_at": "2026-04-20T00:02:00Z",
                "score": 2,
                "total_possible": 2,
                "time_spent_seconds": 2,
                "answers": [
                    {
                        "question_id": 11,
                        "user_answer": "Paris",
                        "is_correct": True,
                        "correct_answer": "Paris",
                        "explanation": "Paris is the capital city.",
                        "points_awarded": 2,
                        "time_spent_ms": 1200,
                    }
                ],
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created_quiz = await client.create_quiz(
        QuizCreateRequest(
            name="Renal Review",
            description="Kidney basics",
            time_limit_seconds=300,
            passing_score=70,
        )
    )
    listed_quizzes = await client.list_quizzes(limit=10, offset=2)
    loaded_quiz = await client.get_quiz(7)
    updated_quiz = await client.update_quiz(7, QuizUpdateRequest(name="Renal Review v2", expected_version=1))
    deleted_quiz = await client.delete_quiz(7, expected_version=2)
    created_question = await client.create_quiz_question(
        7,
        QuizQuestionCreateRequest(
            question_type="fill_blank",
            question_text="The capital of France is ____.",
            correct_answer="Paris",
            explanation="Paris is the capital city.",
            points=2,
        ),
    )
    listed_questions = await client.list_quiz_questions(7, include_answers=False, limit=10, offset=1)
    updated_question = await client.update_quiz_question(
        7,
        11,
        {"explanation": "Paris is the capital city.", "expected_version": 1},
    )
    deleted_question = await client.delete_quiz_question(7, 11, expected_version=2)
    started_attempt = await client.start_quiz_attempt(7)
    submitted_attempt = await client.submit_quiz_attempt(
        41,
        QuizAttemptSubmitRequest(
            answers=[
                {
                    "question_id": 11,
                    "user_answer": "Paris",
                    "hint_used": False,
                    "time_spent_ms": 1200,
                }
            ]
        ),
    )
    listed_attempts = await client.list_quiz_attempts(quiz_id=7, limit=10, offset=1)
    loaded_attempt = await client.get_quiz_attempt(41, include_questions=True, include_answers=True)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/quizzes")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/quizzes")
    assert mocked.await_args_list[1].kwargs["params"] == {"limit": 10, "offset": 2}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/quizzes/7")
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/quizzes/7")
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/quizzes/7")
    assert mocked.await_args_list[4].kwargs["params"] == {"expected_version": 2, "hard": False}
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/quizzes/7/questions")
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/quizzes/7/questions")
    assert mocked.await_args_list[6].kwargs["params"] == {
        "include_answers": False,
        "limit": 10,
        "offset": 1,
    }
    assert mocked.await_args_list[7].args[:2] == ("PATCH", "/api/v1/quizzes/7/questions/11")
    assert mocked.await_args_list[8].args[:2] == ("DELETE", "/api/v1/quizzes/7/questions/11")
    assert mocked.await_args_list[8].kwargs["params"] == {"expected_version": 2, "hard": False}
    assert mocked.await_args_list[9].args[:2] == ("POST", "/api/v1/quizzes/7/attempts")
    assert mocked.await_args_list[10].args[:2] == ("PUT", "/api/v1/quizzes/attempts/41")
    assert mocked.await_args_list[11].args[:2] == ("GET", "/api/v1/quizzes/attempts")
    assert mocked.await_args_list[11].kwargs["params"] == {"quiz_id": 7, "limit": 10, "offset": 1}
    assert mocked.await_args_list[12].args[:2] == ("GET", "/api/v1/quizzes/attempts/41")
    assert mocked.await_args_list[12].kwargs["params"] == {"include_questions": True, "include_answers": True}

    assert isinstance(created_quiz, QuizResponse)
    assert isinstance(loaded_quiz, QuizResponse)
    assert isinstance(created_question, QuizQuestionResponse)
    assert isinstance(listed_questions, QuizQuestionListResponse)
    assert isinstance(started_attempt, QuizAttemptResponse)
    assert isinstance(submitted_attempt, QuizAttemptResponse)
    assert isinstance(listed_attempts, QuizAttemptListResponse)
    assert isinstance(loaded_attempt, QuizAttemptResponse)
    assert updated_quiz.version == 2
    assert updated_question.version == 2
    assert deleted_quiz["status"] == "deleted"
    assert deleted_question["status"] == "deleted"
    assert listed_quizzes[0].name == "Renal Review"
