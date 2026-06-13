from tldw_chatbook.tldw_api import (
    QuizAttemptAnswerInput,
    QuizAttemptListResponse,
    QuizAttemptResponse,
    QuizAttemptSubmitRequest,
    QuizCreateRequest,
    QuizListResponse,
    QuizQuestionCreateRequest,
    QuizQuestionListResponse,
    QuizQuestionResponse,
    QuizResponse,
    QuizUpdateRequest,
)


def test_quiz_create_request_round_trips_core_fields():
    payload = QuizCreateRequest(
        name="Renal Review",
        description="Kidney basics",
        workspace_id="ws-1",
        time_limit_seconds=300,
        passing_score=70,
    )

    assert payload.model_dump(exclude_none=True) == {
        "name": "Renal Review",
        "description": "Kidney basics",
        "workspace_id": "ws-1",
        "time_limit_seconds": 300,
        "passing_score": 70,
    }


def test_quiz_response_preserves_metadata_and_counts():
    payload = QuizResponse.model_validate(
        {
            "id": 7,
            "name": "Renal Review",
            "description": "Kidney basics",
            "workspace_id": None,
            "total_questions": 3,
            "time_limit_seconds": 300,
            "passing_score": 70,
            "deleted": False,
            "client_id": "server-client",
            "version": 2,
            "created_at": "2026-04-20T00:00:00Z",
            "last_modified": "2026-04-20T00:02:00Z",
        }
    )

    assert payload.id == 7
    assert payload.total_questions == 3
    assert payload.time_limit_seconds == 300


def test_quiz_question_response_preserves_public_and_admin_fields():
    payload = QuizQuestionResponse.model_validate(
        {
            "id": 11,
            "quiz_id": 7,
            "question_type": "fill_blank",
            "question_text": "The capital of France is ____.",
            "options": None,
            "correct_answer": "Paris",
            "explanation": "Paris is the capital city.",
            "hint": "Think Eiffel Tower.",
            "hint_penalty_points": 1,
            "source_citations": [{"source_type": "note", "source_id": "note-1"}],
            "points": 2,
            "order_index": 0,
            "tags": ["geography"],
            "deleted": False,
            "client_id": "server-client",
            "version": 1,
            "created_at": "2026-04-20T00:00:00Z",
            "last_modified": "2026-04-20T00:01:00Z",
        }
    )

    assert payload.correct_answer == "Paris"
    assert payload.explanation == "Paris is the capital city."
    assert payload.tags == ["geography"]


def test_quiz_question_list_response_wraps_questions():
    payload = QuizQuestionListResponse.model_validate(
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
                    "points": 1,
                    "order_index": 0,
                    "tags": None,
                    "deleted": False,
                    "client_id": "server-client",
                    "version": 1,
                    "created_at": "2026-04-20T00:00:00Z",
                    "last_modified": "2026-04-20T00:01:00Z",
                }
            ],
            "count": 1,
        }
    )

    assert payload.count == 1
    assert payload.items[0].question_text == "The capital of France is ____."


def test_quiz_attempt_submit_request_preserves_answer_payloads():
    payload = QuizAttemptSubmitRequest(
        answers=[
            QuizAttemptAnswerInput(
                question_id=11,
                user_answer="Paris",
                hint_used=False,
                time_spent_ms=1200,
            )
        ]
    )

    assert payload.model_dump() == {
        "answers": [
            {
                "question_id": 11,
                "user_answer": "Paris",
                "hint_used": False,
                "time_spent_ms": 1200,
            }
        ]
    }


def test_quiz_attempt_response_can_include_questions_and_answers():
    payload = QuizAttemptResponse.model_validate(
        {
            "id": 41,
            "quiz_id": 7,
            "started_at": "2026-04-20T00:00:00Z",
            "completed_at": "2026-04-20T00:03:00Z",
            "score": 2,
            "total_possible": 2,
            "time_spent_seconds": 3,
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
        }
    )

    assert payload.score == 2
    assert payload.answers[0].is_correct is True
    assert payload.questions[0].question_type == "fill_blank"


def test_quiz_list_and_attempt_list_responses_wrap_items():
    quizzes = QuizListResponse.model_validate(
        {
            "items": [
                {
                    "id": 7,
                    "name": "Renal Review",
                    "description": "Kidney basics",
                    "workspace_id": None,
                    "total_questions": 3,
                    "time_limit_seconds": None,
                    "passing_score": None,
                    "deleted": False,
                    "client_id": "server-client",
                    "version": 1,
                    "created_at": "2026-04-20T00:00:00Z",
                    "last_modified": "2026-04-20T00:02:00Z",
                }
            ],
            "count": 1,
        }
    )
    attempts = QuizAttemptListResponse.model_validate(
        {
            "items": [
                {
                    "id": 41,
                    "quiz_id": 7,
                    "started_at": "2026-04-20T00:00:00Z",
                    "completed_at": None,
                    "score": None,
                    "total_possible": 2,
                    "time_spent_seconds": None,
                    "answers": [],
                }
            ],
            "count": 1,
        }
    )
    update = QuizUpdateRequest(name="Renal Review v2", expected_version=1)

    assert quizzes.items[0].name == "Renal Review"
    assert attempts.items[0].quiz_id == 7
    assert update.model_dump(exclude_none=True) == {"name": "Renal Review v2", "expected_version": 1}
