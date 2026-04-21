from tldw_chatbook.Study_Interop.quiz_normalizers import (
    normalize_quiz_attempt_record,
    normalize_quiz_question_record,
    normalize_quiz_record,
)


def test_normalize_local_quiz_record_sets_local_defaults():
    record = normalize_quiz_record(
        "local",
        {
            "id": "quiz-local-1",
            "name": "Renal Review",
            "description": "Kidney basics",
            "total_questions": 3,
        },
    )

    assert record["record_id"] == "local:quiz:quiz-local-1"
    assert record["record_type"] == "quiz"
    assert record["workspace_id"] is None
    assert record["total_questions"] == 3


def test_normalize_server_quiz_question_record_preserves_admin_fields():
    record = normalize_quiz_question_record(
        "server",
        {
            "id": 11,
            "quiz_id": 7,
            "question_type": "fill_blank",
            "question_text": "The capital of France is ____.",
            "correct_answer": "Paris",
            "explanation": "Paris is the capital city.",
            "hint": "Think Eiffel Tower.",
            "hint_penalty_points": 1,
            "source_citations": [{"source_type": "note", "source_id": "note-1"}],
            "points": 2,
            "order_index": 0,
            "tags": ["geography"],
        },
    )

    assert record["record_id"] == "server:quiz_question:11"
    assert record["quiz_record_id"] == "server:quiz:7"
    assert record["correct_answer"] == "Paris"
    assert record["answer_visible"] is True


def test_normalize_quiz_attempt_record_normalizes_nested_questions_and_answers():
    record = normalize_quiz_attempt_record(
        "server",
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
                    "points_awarded": 2,
                }
            ],
            "questions": [
                {
                    "id": 11,
                    "quiz_id": 7,
                    "question_type": "fill_blank",
                    "question_text": "The capital of France is ____.",
                    "points": 2,
                    "order_index": 0,
                }
            ],
        },
    )

    assert record["record_id"] == "server:quiz_attempt:41"
    assert record["quiz_record_id"] == "server:quiz:7"
    assert record["questions"][0]["record_id"] == "server:quiz_question:11"
    assert record["answers"][0]["question_record_id"] == "server:quiz_question:11"
    assert record["score"] == 2
