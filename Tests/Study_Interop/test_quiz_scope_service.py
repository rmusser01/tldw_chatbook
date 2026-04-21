import pytest

from tldw_chatbook.Study_Interop.quiz_scope_service import QuizScopeService


class FakeLocalQuizService:
    def __init__(self):
        self.calls = []

    def list_quizzes(self, *, q=None, limit=100, offset=0):
        self.calls.append(("list_quizzes", q, limit, offset))
        return {"items": [{"id": "quiz-local-1", "name": "Renal Review", "total_questions": 1}], "count": 1}

    def create_quiz(self, *, name, description=None, time_limit_seconds=None, passing_score=None, workspace_id=None):
        self.calls.append(("create_quiz", name, description, time_limit_seconds, passing_score, workspace_id))
        return {"id": "quiz-local-1", "name": name, "description": description, "total_questions": 0}

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
        return {
            "id": "question-local-1",
            "quiz_id": quiz_id,
            "question_type": payload["question_type"],
            "question_text": payload["question_text"],
            "correct_answer": payload["correct_answer"],
            "explanation": payload.get("explanation"),
            "points": payload.get("points", 1),
            "order_index": payload.get("order_index", 0),
        }

    def delete_quiz(self, quiz_id, *, expected_version=None, hard_delete=False):
        self.calls.append(("delete_quiz", quiz_id, expected_version, hard_delete))
        return True

    def delete_question(self, question_id, *, expected_version=None, hard_delete=False):
        self.calls.append(("delete_question", question_id, expected_version, hard_delete))
        return True

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

    def submit_attempt(self, attempt_id, *, answers):
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
            "answers": [
                {
                    "question_id": "question-local-1",
                    "user_answer": "Paris",
                    "is_correct": True,
                    "points_awarded": 2,
                }
            ],
            "questions": [
                {
                    "id": "question-local-1",
                    "quiz_id": "quiz-local-1",
                    "question_type": "fill_blank",
                    "question_text": "The capital of France is ____.",
                    "correct_answer": "Paris",
                    "points": 2,
                    "order_index": 0,
                }
            ],
        }


class FakeServerQuizService:
    def __init__(self):
        self.calls = []

    async def list_quizzes(self, *, q=None, limit=100, offset=0):
        self.calls.append(("list_quizzes", q, limit, offset))
        return {"items": [{"id": 7, "name": "Renal Review", "total_questions": 1, "time_limit_seconds": 300}], "count": 1}

    async def start_attempt(self, quiz_id):
        self.calls.append(("start_attempt", quiz_id))
        return {
            "id": 41,
            "quiz_id": quiz_id,
            "started_at": "2026-04-20T00:00:00Z",
            "completed_at": None,
            "score": None,
            "total_possible": 2,
            "time_spent_seconds": None,
            "answers": [],
            "questions": [
                {
                    "id": 11,
                    "quiz_id": quiz_id,
                    "question_type": "fill_blank",
                    "question_text": "The capital of France is ____.",
                    "points": 2,
                    "order_index": 0,
                }
            ],
        }

    async def delete_quiz(self, quiz_id, *, expected_version=None, hard_delete=False):
        self.calls.append(("delete_quiz", quiz_id, expected_version, hard_delete))
        return True

    async def delete_question(self, question_id, *, expected_version=None, hard_delete=False):
        self.calls.append(("delete_question", question_id, expected_version, hard_delete))
        return True

    async def submit_attempt(self, attempt_id, *, answers):
        self.calls.append(("submit_attempt", attempt_id, answers))
        return {
            "id": attempt_id,
            "quiz_id": 7,
            "started_at": "2026-04-20T00:00:00Z",
            "completed_at": "2026-04-20T00:02:00Z",
            "score": 2,
            "total_possible": 2,
            "time_spent_seconds": 2,
            "answers": [{"question_id": 11, "user_answer": "Paris", "is_correct": True, "points_awarded": 2}],
        }

    async def list_attempts(self, *, quiz_id=None, limit=100, offset=0):
        self.calls.append(("list_attempts", quiz_id, limit, offset))
        return {
            "items": [
                {
                    "id": 41,
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

    async def get_attempt(self, attempt_id, *, include_questions=False, include_answers=False):
        self.calls.append(("get_attempt", attempt_id, include_questions, include_answers))
        return {
            "id": attempt_id,
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
                    "points_awarded": 2,
                }
            ],
            "questions": [
                {
                    "id": 11,
                    "quiz_id": 7,
                    "question_type": "fill_blank",
                    "question_text": "The capital of France is ____.",
                    "correct_answer": "Paris",
                    "points": 2,
                    "order_index": 0,
                }
            ],
        }


class PagedFakeServerQuizService:
    def __init__(self, pages, *, fail_on_offset=None):
        self.calls = []
        self.pages = pages
        self.fail_on_offset = fail_on_offset

    async def list_quizzes(self, *, q=None, limit=100, offset=0):
        self.calls.append(("list_quizzes", q, limit, offset))
        if self.fail_on_offset is not None and offset == self.fail_on_offset:
            raise RuntimeError("quiz page load failed")
        items = list(self.pages.get(offset, []))
        return {"items": items, "count": len(items)}

    async def create_quiz(self, *, name, description=None, time_limit_seconds=None, passing_score=None, workspace_id=None):
        self.calls.append(("create_quiz", name, description, time_limit_seconds, passing_score, workspace_id))
        return {"id": "quiz-created", "name": name, "description": description, "workspace_id": workspace_id}


@pytest.mark.asyncio
async def test_quiz_scope_service_routes_quiz_list_by_backend():
    scope = QuizScopeService(
        local_service=FakeLocalQuizService(),
        server_service=FakeServerQuizService(),
    )

    local_quizzes = await scope.list_quizzes(mode="local")
    server_quizzes = await scope.list_quizzes(mode="server")

    assert local_quizzes[0]["record_id"] == "local:quiz:quiz-local-1"
    assert server_quizzes[0]["record_id"] == "server:quiz:7"
    assert server_quizzes[0]["time_limit_seconds"] == 300


@pytest.mark.asyncio
async def test_quiz_scope_service_filters_global_and_workspace_quizzes_without_mixing():
    server = PagedFakeServerQuizService(
        {
            0: [
                {"id": 1, "name": "Global One", "workspace_id": None},
                {"id": 2, "name": "Workspace One A", "workspace_id": "ws-1"},
            ],
            2: [
                {"id": 3, "name": "Workspace Two", "workspace_id": "ws-2"},
                {"id": 4, "name": "Global Two", "workspace_id": None},
            ],
            4: [
                {"id": 5, "name": "Workspace One B", "workspace_id": "ws-1"},
            ],
        }
    )
    scope = QuizScopeService(local_service=FakeLocalQuizService(), server_service=server)

    global_quizzes = await scope.list_quizzes(mode="server", scope_type="global", limit=2, offset=0)
    workspace_quizzes = await scope.list_quizzes(
        mode="server",
        scope_type="workspace",
        workspace_id="ws-1",
        limit=2,
        offset=0,
    )

    assert [quiz["backing_id"] for quiz in global_quizzes] == ["1", "4"]
    assert [quiz["backing_id"] for quiz in workspace_quizzes] == ["2", "5"]
    assert server.calls == [
        ("list_quizzes", None, 2, 0),
        ("list_quizzes", None, 2, 2),
        ("list_quizzes", None, 2, 4),
        ("list_quizzes", None, 2, 0),
        ("list_quizzes", None, 2, 2),
        ("list_quizzes", None, 2, 4),
    ]


@pytest.mark.asyncio
async def test_quiz_scope_service_applies_offset_after_client_side_quiz_scope_filtering():
    server = PagedFakeServerQuizService(
        {
            0: [
                {"id": 1, "name": "Global One", "workspace_id": None},
                {"id": 2, "name": "Workspace One A", "workspace_id": "ws-1"},
            ],
            2: [
                {"id": 3, "name": "Workspace Two", "workspace_id": "ws-2"},
                {"id": 4, "name": "Global Two", "workspace_id": None},
            ],
            4: [
                {"id": 5, "name": "Workspace One B", "workspace_id": "ws-1"},
            ],
        }
    )
    scope = QuizScopeService(local_service=FakeLocalQuizService(), server_service=server)

    global_quizzes = await scope.list_quizzes(mode="server", scope_type="global", limit=2, offset=1)
    workspace_quizzes = await scope.list_quizzes(
        mode="server",
        scope_type="workspace",
        workspace_id="ws-1",
        limit=2,
        offset=1,
    )

    assert [quiz["record_id"] for quiz in global_quizzes] == ["server:quiz:4"]
    assert [quiz["record_id"] for quiz in workspace_quizzes] == ["server:quiz:5"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("list_quizzes", {"mode": "server", "scope_type": "invalid", "limit": 1}),
        (
            "create_quiz",
            {"mode": "server", "scope_type": "invalid", "name": "Workspace quiz", "description": None},
        ),
    ],
)
async def test_quiz_scope_service_rejects_invalid_scope_type_for_scoped_methods(method_name, kwargs):
    scope = QuizScopeService(local_service=FakeLocalQuizService(), server_service=FakeServerQuizService())

    with pytest.raises(ValueError, match="Invalid quiz scope_type: invalid"):
        await getattr(scope, method_name)(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("list_quizzes", {"mode": "server", "scope_type": "workspace", "limit": 1}),
        (
            "create_quiz",
            {"mode": "server", "scope_type": "workspace", "name": "Workspace quiz", "description": None},
        ),
    ],
)
async def test_quiz_scope_service_rejects_missing_workspace_id_for_workspace_scope(method_name, kwargs):
    scope = QuizScopeService(local_service=FakeLocalQuizService(), server_service=FakeServerQuizService())

    with pytest.raises(ValueError, match="workspace_id is required when scope_type='workspace'"):
        await getattr(scope, method_name)(**kwargs)


@pytest.mark.asyncio
async def test_quiz_scope_service_rejects_workspace_scope_attempt_calls_in_local_mode():
    scope = QuizScopeService(local_service=FakeLocalQuizService(), server_service=FakeServerQuizService())

    with pytest.raises(ValueError, match="Workspace Study is unavailable in local mode"):
        await scope.start_attempt(
            mode="local",
            scope_type="workspace",
            workspace_id="ws-1",
            quiz_id="quiz-local-1",
        )

    with pytest.raises(ValueError, match="Workspace Study is unavailable in local mode"):
        await scope.submit_attempt(
            mode="local",
            scope_type="workspace",
            workspace_id="ws-1",
            attempt_id="attempt-local-1",
            answers=[],
        )


@pytest.mark.asyncio
async def test_quiz_scope_service_rejects_workspace_scope_for_attempt_history_loading_in_local_mode():
    scope = QuizScopeService(local_service=FakeLocalQuizService(), server_service=FakeServerQuizService())

    with pytest.raises(ValueError, match="Workspace Study is unavailable in local mode"):
        await scope.list_attempts(
            mode="local",
            scope_type="workspace",
            workspace_id="ws-1",
            quiz_id="quiz-local-1",
        )

    with pytest.raises(ValueError, match="Workspace Study is unavailable in local mode"):
        await scope.get_attempt(
            mode="local",
            scope_type="workspace",
            workspace_id="ws-1",
            attempt_id="attempt-local-1",
            include_questions=True,
            include_answers=True,
        )


@pytest.mark.asyncio
async def test_quiz_scope_service_forwards_workspace_id_on_server_create_and_nulls_global_create():
    server = PagedFakeServerQuizService({0: []})
    scope = QuizScopeService(local_service=FakeLocalQuizService(), server_service=server)

    await scope.create_quiz(
        mode="server",
        scope_type="workspace",
        workspace_id="ws-7",
        name="Workspace quiz",
        description="Scoped",
        time_limit_seconds=300,
        passing_score=80,
    )
    await scope.create_quiz(
        mode="server",
        scope_type="global",
        workspace_id="ws-7",
        name="Global quiz",
        description=None,
    )

    assert server.calls == [
        ("create_quiz", "Workspace quiz", "Scoped", 300, 80, "ws-7"),
        ("create_quiz", "Global quiz", None, None, None, None),
    ]


@pytest.mark.asyncio
async def test_quiz_scope_service_keeps_workspace_load_errors_scoped():
    server = PagedFakeServerQuizService(
        {
            0: [
                {"id": 1, "name": "Global One", "workspace_id": None},
                {"id": 2, "name": "Workspace One", "workspace_id": "ws-2"},
            ],
            2: [{"id": 3, "name": "Workspace Two", "workspace_id": "ws-1"}],
        },
        fail_on_offset=2,
    )
    scope = QuizScopeService(local_service=FakeLocalQuizService(), server_service=server)

    with pytest.raises(RuntimeError, match="quiz page load failed"):
        await scope.list_quizzes(mode="server", scope_type="workspace", workspace_id="ws-1", limit=2, offset=0)


@pytest.mark.asyncio
async def test_quiz_scope_service_routes_question_create_and_list():
    local = FakeLocalQuizService()
    scope = QuizScopeService(local_service=local, server_service=FakeServerQuizService())

    created = await scope.create_question(
        mode="local",
        quiz_id="quiz-local-1",
        question_type="fill_blank",
        question_text="The capital of France is ____.",
        correct_answer="Paris",
        explanation="Paris is the capital city.",
        points=2,
    )
    listed = await scope.list_questions(mode="local", quiz_id="quiz-local-1", q="  ", include_answers=True, limit=10, offset=1)

    assert created["record_id"] == "local:quiz_question:question-local-1"
    assert listed[0]["quiz_record_id"] == "local:quiz:quiz-local-1"
    assert local.calls[0][0] == "create_question"
    assert local.calls[1] == ("list_questions", "quiz-local-1", None, True, 10, 1)


@pytest.mark.asyncio
async def test_quiz_scope_service_normalizes_attempt_start_and_submit():
    server = FakeServerQuizService()
    scope = QuizScopeService(local_service=FakeLocalQuizService(), server_service=server)

    started = await scope.start_attempt(mode="server", quiz_id=7)
    submitted = await scope.submit_attempt(
        mode="server",
        attempt_id=41,
        answers=[{"question_id": 11, "user_answer": "Paris", "time_spent_ms": 1200}],
    )

    assert started["record_id"] == "server:quiz_attempt:41"
    assert started["questions"][0]["record_id"] == "server:quiz_question:11"
    assert submitted["answers"][0]["question_record_id"] == "server:quiz_question:11"
    assert submitted["score"] == 2
    assert server.calls == [
        ("start_attempt", 7),
        ("submit_attempt", 41, [{"question_id": 11, "user_answer": "Paris", "time_spent_ms": 1200}]),
    ]


@pytest.mark.asyncio
async def test_quiz_scope_service_routes_delete_and_attempt_history():
    local = FakeLocalQuizService()
    server = FakeServerQuizService()
    scope = QuizScopeService(local_service=local, server_service=server)

    deleted_quiz = await scope.delete_quiz(mode="server", quiz_id=7, expected_version=2)
    deleted_question = await scope.delete_question(mode="local", question_id="question-local-1", expected_version=3)
    attempts = await scope.list_attempts(mode="server", quiz_id=7, limit=5, offset=1)
    loaded_attempt = await scope.get_attempt(
        mode="local",
        attempt_id="attempt-local-1",
        include_questions=True,
        include_answers=True,
    )

    assert deleted_quiz is True
    assert deleted_question is True
    assert attempts[0]["record_id"] == "server:quiz_attempt:41"
    assert loaded_attempt["record_id"] == "local:quiz_attempt:attempt-local-1"
    assert loaded_attempt["questions"][0]["record_id"] == "local:quiz_question:question-local-1"
    assert server.calls[0] == ("delete_quiz", 7, 2, False)
    assert local.calls[0] == ("delete_question", "question-local-1", 3, False)
