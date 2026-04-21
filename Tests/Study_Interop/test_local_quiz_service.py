from tldw_chatbook.Study_Interop.local_quiz_service import LocalQuizService


class FakeDB:
    def __init__(self):
        self.calls = []

    def list_quizzes(self, *, q=None, limit=100, offset=0):
        self.calls.append(("list_quizzes", q, limit, offset))
        return {"items": [{"id": "quiz-local-1", "name": "Renal Review", "total_questions": 1}], "count": 1}

    def create_quiz(self, **payload):
        self.calls.append(("create_quiz", payload))
        return "quiz-local-1"

    def get_quiz(self, quiz_id):
        self.calls.append(("get_quiz", quiz_id))
        return {"id": quiz_id, "name": "Renal Review", "description": "Kidney basics", "total_questions": 1}

    def list_questions(self, quiz_id, *, q=None, include_answers=False, limit=100, offset=0):
        self.calls.append(("list_questions", quiz_id, q, include_answers, limit, offset))
        return {
            "items": [
                {
                    "id": "question-local-1",
                    "quiz_id": quiz_id,
                    "question_type": "fill_blank",
                    "question_text": "The capital of France is ____.",
                    "correct_answer": "Paris",
                    "points": 2,
                    "order_index": 0,
                }
            ],
            "count": 1,
        }

    def create_question(self, quiz_id, **payload):
        self.calls.append(("create_question", quiz_id, payload))
        return "question-local-1"

    def delete_quiz(self, quiz_id, *, expected_version=None, hard_delete=False):
        self.calls.append(("delete_quiz", quiz_id, expected_version, hard_delete))
        return True

    def delete_question(self, question_id, *, expected_version=None, hard_delete=False):
        self.calls.append(("delete_question", question_id, expected_version, hard_delete))
        return True

    def get_question(self, question_id):
        self.calls.append(("get_question", question_id))
        return {
            "id": question_id,
            "quiz_id": "quiz-local-1",
            "question_type": "fill_blank",
            "question_text": "The capital of France is ____.",
            "correct_answer": "Paris",
            "points": 2,
            "order_index": 0,
        }

    def start_attempt(self, quiz_id):
        self.calls.append(("start_attempt", quiz_id))
        return {
            "id": "attempt-local-1",
            "quiz_id": quiz_id,
            "started_at": "2026-04-20T00:00:00Z",
            "completed_at": None,
            "score": None,
            "total_possible": 2,
            "time_spent_seconds": None,
            "answers": [],
            "questions": [
                {
                    "id": "question-local-1",
                    "quiz_id": quiz_id,
                    "question_type": "fill_blank",
                    "question_text": "The capital of France is ____.",
                    "points": 2,
                    "order_index": 0,
                }
            ],
        }

    def submit_attempt(self, attempt_id, answers):
        self.calls.append(("submit_attempt", attempt_id, answers))
        return {
            "id": attempt_id,
            "quiz_id": "quiz-local-1",
            "started_at": "2026-04-20T00:00:00Z",
            "completed_at": "2026-04-20T00:02:00Z",
            "score": 2,
            "total_possible": 2,
            "time_spent_seconds": 2,
            "answers": [{"question_id": "question-local-1", "user_answer": "Paris", "is_correct": True, "points_awarded": 2}],
        }

    def list_attempts(self, *, quiz_id=None, limit=100, offset=0):
        self.calls.append(("list_attempts", quiz_id, limit, offset))
        return {
            "items": [
                {
                    "id": "attempt-local-1",
                    "quiz_id": quiz_id,
                    "started_at": "2026-04-20T00:00:00Z",
                    "completed_at": "2026-04-20T00:02:00Z",
                    "score": 2,
                    "total_possible": 2,
                    "time_spent_seconds": 2,
                    "answers": [],
                }
            ],
            "count": 1,
        }

    def get_attempt(self, attempt_id, *, include_questions=False, include_answers=False):
        self.calls.append(("get_attempt", attempt_id, include_questions, include_answers))
        return {
            "id": attempt_id,
            "quiz_id": "quiz-local-1",
            "started_at": "2026-04-20T00:00:00Z",
            "completed_at": "2026-04-20T00:02:00Z",
            "score": 2,
            "total_possible": 2,
            "time_spent_seconds": 2,
            "answers": [],
            "questions": [],
        }


def test_local_quiz_service_lists_and_creates_quizzes():
    db = FakeDB()
    service = LocalQuizService(db=db)

    listed = service.list_quizzes(q="renal", limit=5, offset=1)
    created = service.create_quiz(name="Renal Review", description="Kidney basics", time_limit_seconds=300, passing_score=70)

    assert listed["items"][0]["name"] == "Renal Review"
    assert created["id"] == "quiz-local-1"
    assert db.calls == [
        ("list_quizzes", "renal", 5, 1),
        ("create_quiz", {"name": "Renal Review", "description": "Kidney basics", "workspace_id": None, "time_limit_seconds": 300, "passing_score": 70}),
        ("get_quiz", "quiz-local-1"),
    ]


def test_local_quiz_service_normalizes_blank_question_search_to_list_query():
    db = FakeDB()
    service = LocalQuizService(db=db)

    listed = service.list_questions("quiz-local-1", q="   ", include_answers=False, limit=7, offset=3)

    assert listed["items"][0]["quiz_id"] == "quiz-local-1"
    assert db.calls == [("list_questions", "quiz-local-1", None, False, 7, 3)]


def test_local_quiz_service_creates_questions_and_records_attempts():
    db = FakeDB()
    service = LocalQuizService(db=db)

    created_question = service.create_question(
        "quiz-local-1",
        question_type="fill_blank",
        question_text="The capital of France is ____.",
        correct_answer="Paris",
        explanation="Paris is the capital city.",
        points=2,
    )
    started = service.start_attempt("quiz-local-1")
    submitted = service.submit_attempt(
        "attempt-local-1",
        answers=[{"question_id": "question-local-1", "user_answer": "Paris", "time_spent_ms": 1200}],
    )
    attempts = service.list_attempts(quiz_id="quiz-local-1", limit=5, offset=1)
    loaded = service.get_attempt("attempt-local-1", include_questions=True, include_answers=True)

    assert created_question["id"] == "question-local-1"
    assert started["id"] == "attempt-local-1"
    assert submitted["score"] == 2
    assert attempts["count"] == 1
    assert loaded["id"] == "attempt-local-1"


def test_local_quiz_service_deletes_quiz_and_question():
    db = FakeDB()
    service = LocalQuizService(db=db)

    deleted_quiz = service.delete_quiz("quiz-local-1", expected_version=2)
    deleted_question = service.delete_question("question-local-1", expected_version=3)

    assert deleted_quiz is True
    assert deleted_question is True
    assert db.calls == [
        ("delete_quiz", "quiz-local-1", 2, False),
        ("delete_question", "question-local-1", 3, False),
    ]
