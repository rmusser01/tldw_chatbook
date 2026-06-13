from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import Button, Input, ListView, Select, Static, TextArea

from tldw_chatbook.UI.Screens.study_scope_models import StudyScopeContext, StudyScopeState, StudyScopeType
from tldw_chatbook.UI.Screens.study_screen import StudyScreen
from tldw_chatbook.UI.Study_Window import StudyWindow


class FakeQuizScopeService:
    def __init__(self):
        self.calls = []
        self.quizzes = [{"record_id": "local:quiz:quiz-local-1", "backing_id": "quiz-local-1", "name": "Renal Review", "total_questions": 1}]
        self.questions = [
            {
                "record_id": "local:quiz_question:question-local-1",
                "backing_id": "question-local-1",
                "quiz_record_id": "local:quiz:quiz-local-1",
                "question_type": "fill_blank",
                "question_text": "The capital of France is ____.",
                "correct_answer": "Paris",
                "points": 2,
                "order_index": 0,
            }
        ]
        self.attempts = [
            {
                "record_id": "local:quiz_attempt:attempt-1",
                "backing_id": "attempt-1",
                "quiz_record_id": "local:quiz:quiz-local-1",
                "started_at": "2026-04-20T00:00:00Z",
                "completed_at": "2026-04-20T00:02:00Z",
                "score": 2,
                "total_possible": 2,
                "time_spent_seconds": 2,
                "answers": [],
            }
        ]

    async def list_quizzes(self, *, mode=None, scope_type=None, workspace_id=None, q=None, limit=100, offset=0):
        self.calls.append(("list_quizzes", mode, scope_type, workspace_id, q, limit, offset))
        return list(self.quizzes)

    async def create_quiz(
        self,
        *,
        mode=None,
        scope_type=None,
        workspace_id=None,
        name,
        description=None,
        time_limit_seconds=None,
        passing_score=None,
    ):
        self.calls.append(("create_quiz", mode, scope_type, workspace_id, name, description, time_limit_seconds, passing_score))
        created = {
            "record_id": f"{mode}:quiz:new-quiz",
            "backing_id": "new-quiz",
            "name": name,
            "description": description,
            "total_questions": 0,
        }
        self.quizzes.append(created)
        return created

    async def list_questions(self, *, mode=None, quiz_id=None, q=None, include_answers=False, limit=100, offset=0):
        self.calls.append(("list_questions", mode, quiz_id, q, include_answers, limit, offset))
        return [question for question in self.questions if quiz_id is None or question["quiz_record_id"].endswith(str(quiz_id))]

    async def create_question(
        self,
        *,
        mode=None,
        quiz_id=None,
        question_type,
        question_text,
        correct_answer,
        options=None,
        explanation=None,
        hint=None,
        hint_penalty_points=0,
        points=1,
        order_index=0,
        tags=None,
        source_citations=None,
    ):
        self.calls.append(("create_question", mode, quiz_id, question_type, question_text, correct_answer, points))
        created = {
            "record_id": f"{mode}:quiz_question:new-question",
            "backing_id": "new-question",
            "quiz_record_id": f"{mode}:quiz:{quiz_id}",
            "question_type": question_type,
            "question_text": question_text,
            "correct_answer": correct_answer,
            "explanation": explanation,
            "points": points,
            "order_index": order_index,
        }
        self.questions.append(created)
        return created

    async def delete_quiz(self, *, mode=None, quiz_id=None, expected_version=None, hard_delete=False):
        self.calls.append(("delete_quiz", mode, quiz_id, expected_version, hard_delete))
        self.quizzes = [quiz for quiz in self.quizzes if str(quiz.get("backing_id")) != str(quiz_id)]
        self.questions = [question for question in self.questions if not str(question.get("quiz_record_id", "")).endswith(str(quiz_id))]
        self.attempts = [attempt for attempt in self.attempts if not str(attempt.get("quiz_record_id", "")).endswith(str(quiz_id))]
        return True

    async def delete_question(self, *, mode=None, quiz_id=None, question_id=None, expected_version=None, hard_delete=False):
        self.calls.append(("delete_question", mode, question_id, expected_version, hard_delete))
        self.questions = [question for question in self.questions if str(question.get("backing_id")) != str(question_id)]
        return True

    async def start_attempt(self, *, mode=None, scope_type=None, workspace_id=None, quiz_id=None):
        self.calls.append(("start_attempt", mode, scope_type, workspace_id, quiz_id))
        return {
            "record_id": f"{mode}:quiz_attempt:attempt-1",
            "backing_id": "attempt-1",
            "quiz_record_id": f"{mode}:quiz:{quiz_id}",
            "started_at": "2026-04-20T00:00:00Z",
            "completed_at": None,
            "score": None,
            "total_possible": 2,
            "time_spent_seconds": None,
            "answers": [],
            "questions": [
                {
                    "record_id": f"{mode}:quiz_question:question-local-1",
                    "backing_id": "question-local-1",
                    "quiz_record_id": f"{mode}:quiz:{quiz_id}",
                    "question_type": "fill_blank",
                    "question_text": "The capital of France is ____.",
                    "points": 2,
                    "order_index": 0,
                }
            ],
        }

    async def submit_attempt(self, *, mode=None, scope_type=None, workspace_id=None, attempt_id=None, answers=None):
        self.calls.append(("submit_attempt", mode, scope_type, workspace_id, attempt_id, answers))
        return {
            "record_id": f"{mode}:quiz_attempt:{attempt_id}",
            "backing_id": attempt_id,
            "quiz_record_id": f"{mode}:quiz:quiz-local-1",
            "started_at": "2026-04-20T00:00:00Z",
            "completed_at": "2026-04-20T00:02:00Z",
            "score": 2,
            "total_possible": 2,
            "time_spent_seconds": 2,
            "answers": [
                {
                    "question_record_id": f"{mode}:quiz_question:question-local-1",
                    "question_id": "question-local-1",
                    "user_answer": "Paris",
                    "is_correct": True,
                    "points_awarded": 2,
                }
            ],
            "questions": [],
        }

    async def list_attempts(self, *, mode=None, scope_type=None, workspace_id=None, quiz_id=None, limit=100, offset=0):
        self.calls.append(("list_attempts", mode, scope_type, workspace_id, quiz_id, limit, offset))
        return [attempt for attempt in self.attempts if quiz_id is None or str(attempt.get("quiz_record_id", "")).endswith(str(quiz_id))]

    async def get_attempt(
        self,
        *,
        mode=None,
        scope_type=None,
        workspace_id=None,
        attempt_id=None,
        include_questions=False,
        include_answers=False,
    ):
        self.calls.append(("get_attempt", mode, scope_type, workspace_id, attempt_id, include_questions, include_answers))
        return {
            "record_id": f"{mode}:quiz_attempt:{attempt_id}",
            "backing_id": attempt_id,
            "quiz_record_id": f"{mode}:quiz:quiz-local-1",
            "started_at": "2026-04-20T00:00:00Z",
            "completed_at": "2026-04-20T00:02:00Z",
            "score": 2,
            "total_possible": 2,
            "time_spent_seconds": 2,
            "answers": [
                {
                    "question_record_id": f"{mode}:quiz_question:question-local-1",
                    "question_id": "question-local-1",
                    "user_answer": "Paris",
                    "is_correct": True,
                    "points_awarded": 2,
                }
            ],
            "questions": [
                {
                    "record_id": f"{mode}:quiz_question:question-local-1",
                    "backing_id": "question-local-1",
                    "quiz_record_id": f"{mode}:quiz:quiz-local-1",
                    "question_type": "fill_blank",
                    "question_text": "The capital of France is ____.",
                    "points": 2,
                    "order_index": 0,
                }
            ] if include_questions else [],
        }


class EmptyQuizScopeService(FakeQuizScopeService):
    def __init__(self):
        super().__init__()
        self.quizzes = []
        self.questions = []


class WorkspaceFilteredQuizScopeService(FakeQuizScopeService):
    def __init__(self):
        super().__init__()
        self.workspace_id = "ws-1"
        self.global_quizzes = [
            {
                "record_id": "server:quiz:quiz-global-1",
                "backing_id": "quiz-global-1",
                "name": "Global Review",
                "total_questions": 1,
            }
        ]
        self.workspace_quizzes = [
            {
                "record_id": "server:quiz:quiz-workspace-1",
                "backing_id": "quiz-workspace-1",
                "name": "Workspace Review",
                "workspace_id": self.workspace_id,
                "total_questions": 1,
            }
        ]
        self.global_attempts = [
            {
                "record_id": "server:quiz_attempt:attempt-global-1",
                "backing_id": "attempt-global-1",
                "quiz_record_id": "server:quiz:quiz-global-1",
                "started_at": "2026-04-20T00:00:00Z",
                "completed_at": "2026-04-20T00:02:00Z",
                "score": 1,
                "total_possible": 1,
                "time_spent_seconds": 2,
                "answers": [],
            }
        ]
        self.workspace_attempts = [
            {
                "record_id": "server:quiz_attempt:attempt-workspace-1",
                "backing_id": "attempt-workspace-1",
                "quiz_record_id": "server:quiz:quiz-workspace-1",
                "started_at": "2026-04-20T00:00:00Z",
                "completed_at": "2026-04-20T00:03:00Z",
                "score": 2,
                "total_possible": 2,
                "time_spent_seconds": 3,
                "answers": [],
            }
        ]

    async def list_quizzes(self, *, mode=None, scope_type=None, workspace_id=None, q=None, limit=100, offset=0):
        self.calls.append(("list_quizzes", mode, scope_type, workspace_id, q, limit, offset))
        if scope_type == "workspace":
            assert workspace_id == self.workspace_id
            return list(self.workspace_quizzes)
        return list(self.global_quizzes)

    async def create_quiz(
        self,
        *,
        mode=None,
        scope_type=None,
        workspace_id=None,
        name,
        description=None,
        time_limit_seconds=None,
        passing_score=None,
    ):
        self.calls.append(("create_quiz", mode, scope_type, workspace_id, name, description, time_limit_seconds, passing_score))
        created = {
            "record_id": f"{mode}:quiz:new-{scope_type or 'global'}-quiz",
            "backing_id": f"new-{scope_type or 'global'}-quiz",
            "name": name,
            "description": description,
            "workspace_id": workspace_id if scope_type == "workspace" else None,
            "total_questions": 0,
        }
        if scope_type == "workspace":
            self.workspace_quizzes.append(created)
        else:
            self.global_quizzes.append(created)
        return created

    async def list_attempts(self, *, mode=None, scope_type=None, workspace_id=None, quiz_id=None, limit=100, offset=0):
        self.calls.append(("list_attempts", mode, scope_type, workspace_id, quiz_id, limit, offset))
        attempts = self.workspace_attempts if scope_type == "workspace" else self.global_attempts
        return [attempt for attempt in attempts if quiz_id is None or str(attempt.get("quiz_record_id", "")).endswith(str(quiz_id))]

    async def start_attempt(self, *, mode=None, scope_type=None, workspace_id=None, quiz_id=None):
        self.calls.append(("start_attempt", mode, scope_type, workspace_id, quiz_id))
        return {
            "record_id": f"{mode}:quiz_attempt:attempt-workspace-start",
            "backing_id": "attempt-workspace-start",
            "quiz_record_id": f"{mode}:quiz:{quiz_id}",
            "started_at": "2026-04-20T00:00:00Z",
            "completed_at": None,
            "score": None,
            "total_possible": 2,
            "time_spent_seconds": None,
            "answers": [],
            "questions": [
                {
                    "record_id": f"{mode}:quiz_question:question-workspace-1",
                    "backing_id": "question-workspace-1",
                    "quiz_record_id": f"{mode}:quiz:{quiz_id}",
                    "question_type": "fill_blank",
                    "question_text": "Workspace question ____.",
                    "points": 2,
                    "order_index": 0,
                }
            ],
        }

    async def submit_attempt(self, *, mode=None, scope_type=None, workspace_id=None, attempt_id=None, answers=None):
        self.calls.append(("submit_attempt", mode, scope_type, workspace_id, attempt_id, answers))
        return {
            "record_id": f"{mode}:quiz_attempt:{attempt_id}",
            "backing_id": attempt_id,
            "quiz_record_id": f"{mode}:quiz:quiz-workspace-1",
            "started_at": "2026-04-20T00:00:00Z",
            "completed_at": "2026-04-20T00:02:00Z",
            "score": 2,
            "total_possible": 2,
            "time_spent_seconds": 2,
            "answers": [],
            "questions": [],
        }

    async def get_attempt(
        self,
        *,
        mode=None,
        scope_type=None,
        workspace_id=None,
        attempt_id=None,
        include_questions=False,
        include_answers=False,
    ):
        self.calls.append(("get_attempt", mode, scope_type, workspace_id, attempt_id, include_questions, include_answers))
        return {
            "record_id": f"{mode}:quiz_attempt:{attempt_id}",
            "backing_id": attempt_id,
            "quiz_record_id": f"{mode}:quiz:quiz-workspace-1",
            "started_at": "2026-04-20T00:00:00Z",
            "completed_at": "2026-04-20T00:02:00Z",
            "score": 2,
            "total_possible": 2,
            "time_spent_seconds": 2,
            "answers": [
                {
                    "question_record_id": f"{mode}:quiz_question:question-workspace-1",
                    "question_id": "question-workspace-1",
                    "user_answer": "Answer",
                    "is_correct": True,
                    "points_awarded": 2,
                }
            ],
            "questions": [
                {
                    "record_id": f"{mode}:quiz_question:question-workspace-1",
                    "backing_id": "question-workspace-1",
                    "quiz_record_id": f"{mode}:quiz:quiz-workspace-1",
                    "question_type": "fill_blank",
                    "question_text": "Workspace question ____.",
                    "points": 2,
                    "order_index": 0,
                }
            ] if include_questions else [],
        }


class StudyTestApp(App):
    def __init__(self, app_instance):
        super().__init__()
        self._screen = StudyScreen(app_instance=app_instance)

    async def on_mount(self) -> None:
        await self.push_screen(self._screen)


def _text(widget) -> str:
    return str(widget.render())


def _is_blank(value) -> bool:
    return value in {None, "", False, Select.BLANK} or str(value).startswith("Select.")


async def _wait_for_quiz_status(app: App, pilot, expected_substring: str, attempts: int = 20) -> None:
    """Wait for the quiz attempt status to contain the expected text."""
    needle = expected_substring.lower()
    for _ in range(attempts):
        status = app.screen.query_one("#quiz-attempt-status", Static)
        if needle in _text(status).lower():
            return
        await pilot.pause(0.1)


@pytest.mark.asyncio
async def test_quizzes_view_loads_scope_backed_quizzes():
    scope = FakeQuizScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=None,
        study_quiz_scope_service=scope,
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.3)

        quiz_select = app.screen.query_one("#quiz-select", Select)
        status = app.screen.query_one("#quiz-attempt-status", Static)

        assert ("list_quizzes", "local", "global", None, None, 100, 0) in scope.calls
        assert str(quiz_select.value) == "quiz-local-1"
        assert "Ready to manage selected quiz" in _text(status)


@pytest.mark.asyncio
async def test_quizzes_view_loads_workspace_scoped_quizzes_and_attempt_history():
    scope = WorkspaceFilteredQuizScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=None,
        study_quiz_scope_service=scope,
        pending_study_scope_context=StudyScopeContext(
            scope_type=StudyScopeType.WORKSPACE,
            workspace_id="ws-1",
            workspace_name="Research",
        ),
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.3)

        quiz_select = app.screen.query_one("#quiz-select", Select)
        history_select = app.screen.query_one("#quiz-attempt-history-select", Select)
        status = app.screen.query_one("#quiz-attempt-status", Static)
        controller = app.screen.query_one(StudyWindow).quizzes_controller

        assert ("list_quizzes", "server", "workspace", "ws-1", None, 100, 0) in scope.calls
        assert str(quiz_select.value) == "quiz-workspace-1"
        assert "Ready to manage selected quiz" in _text(status)

        history_select.value = "attempt-workspace-1"
        await controller.load_selected_attempt()
        await pilot.pause(0.1)

        assert ("list_attempts", "server", "workspace", "ws-1", "quiz-workspace-1", 100, 0) in scope.calls
        assert ("get_attempt", "server", "workspace", "ws-1", "attempt-workspace-1", True, True) in scope.calls
        assert "Score: 2 / 2" in _text(app.screen.query_one("#quiz-attempt-status", Static))


@pytest.mark.asyncio
async def test_quizzes_view_creates_workspace_quiz_with_active_scope():
    scope = WorkspaceFilteredQuizScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=None,
        study_quiz_scope_service=scope,
        pending_study_scope_context=StudyScopeContext(
            scope_type=StudyScopeType.WORKSPACE,
            workspace_id="ws-1",
            workspace_name="Research",
        ),
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.3)

        controller = app.screen.query_one(StudyWindow).quizzes_controller
        app.screen.query_one("#new-quiz-name-input", Input).value = "Workspace Quiz"
        app.screen.query_one("#new-quiz-description-input", Input).value = "Workspace scoped"

        await controller.create_quiz()
        await pilot.pause(0.1)

        assert ("create_quiz", "server", "workspace", "ws-1", "Workspace Quiz", "Workspace scoped", None, None) in scope.calls
        assert str(app.screen.query_one("#quiz-select", Select).value) == "new-workspace-quiz"


@pytest.mark.asyncio
async def test_workspace_scope_local_mode_disables_quiz_actions_and_shows_unavailable_message():
    scope = FakeQuizScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=None,
        study_quiz_scope_service=scope,
        pending_study_scope_context=StudyScopeContext(
            scope_type=StudyScopeType.WORKSPACE,
            workspace_id="ws-1",
            workspace_name="Research",
        ),
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-quizzes-btn")
        await _wait_for_quiz_status(app, pilot, "server mode")

        status = app.screen.query_one("#quiz-attempt-status", Static)
        question_list = app.screen.query_one("#quiz-question-list", ListView)
        create_quiz_button = app.screen.query_one("#create-quiz-button", Button)
        delete_quiz_button = app.screen.query_one("#delete-quiz-button", Button)
        create_question_button = app.screen.query_one("#create-quiz-question-button", Button)
        delete_question_button = app.screen.query_one("#delete-quiz-question-button", Button)
        start_attempt_button = app.screen.query_one("#start-quiz-attempt-button", Button)
        submit_answer_button = app.screen.query_one("#submit-quiz-answer-button", Button)
        load_attempt_button = app.screen.query_one("#load-quiz-attempt-history-button", Button)

        assert "server mode" in _text(status).lower()
        assert not any(call[0] == "list_quizzes" for call in scope.calls)
        assert question_list.disabled is True
        assert create_quiz_button.disabled is True
        assert delete_quiz_button.disabled is True
        assert create_question_button.disabled is True
        assert delete_question_button.disabled is True
        assert start_attempt_button.disabled is True
        assert submit_answer_button.disabled is True
        assert load_attempt_button.disabled is True
        for button in (
            create_quiz_button,
            delete_quiz_button,
            create_question_button,
            delete_question_button,
            start_attempt_button,
            submit_answer_button,
            load_attempt_button,
        ):
            assert "Workspace Study requires server mode" in str(button.tooltip)


@pytest.mark.asyncio
async def test_backend_flip_resets_quiz_question_attempt_and_answer_state_through_scope_application_path():
    scope = WorkspaceFilteredQuizScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=None,
        study_quiz_scope_service=scope,
        pending_study_scope_context=StudyScopeContext(
            scope_type=StudyScopeType.WORKSPACE,
            workspace_id="ws-1",
            workspace_name="Research",
        ),
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.3)

        window = app.screen.query_one(StudyWindow)
        controller = window.quizzes_controller
        quiz_select = app.screen.query_one("#quiz-select", Select)
        question_list = app.screen.query_one("#quiz-question-list", ListView)
        history_select = app.screen.query_one("#quiz-attempt-history-select", Select)
        answer_input = app.screen.query_one("#quiz-answer-input", Input)
        status = app.screen.query_one("#quiz-attempt-status", Static)

        quiz_select.value = "quiz-workspace-1"
        await controller.refresh_questions()
        await controller.refresh_attempt_history()
        question_list.index = 0
        history_select.value = "attempt-workspace-1"
        answer_input.value = "Paris"
        controller.current_attempt_id = "attempt-workspace-1"
        controller.current_attempt_questions = [
            {"backing_id": "question-workspace-1", "question_text": "Workspace question ____."}
        ]
        controller.current_attempt_answers = [{"question_id": "question-workspace-1", "user_answer": "Paris"}]
        controller.current_question_index = 0
        controller._set_attempt_status("Question 1 of 1.")
        controller._set_attempt_question("Workspace question ____.")

        await app.screen.handle_runtime_backend_changed("local")
        await pilot.pause(0.3)

        assert _is_blank(quiz_select.value)
        assert _is_blank(history_select.value)
        assert controller.current_attempt_id is None
        assert controller.current_attempt_questions == []
        assert controller.current_attempt_answers == []
        assert controller.current_question_index == 0
        assert answer_input.value == ""
        assert controller.current_quiz_questions == []
        assert not question_list.children
        assert question_list.disabled is True
        assert "server mode" in _text(status).lower()


@pytest.mark.asyncio
async def test_quizzes_view_creates_quiz_and_question_through_scope_service():
    scope = FakeQuizScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=None,
        study_quiz_scope_service=scope,
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.3)
        controller = app.screen.query_one(StudyWindow).quizzes_controller

        app.screen.query_one("#new-quiz-name-input", Input).value = "Geography Review"
        app.screen.query_one("#new-quiz-description-input", Input).value = "Capitals"
        await controller.create_quiz()

        quiz_select = app.screen.query_one("#quiz-select", Select)
        assert str(quiz_select.value) == "new-quiz"

        app.screen.query_one("#quiz-question-text", TextArea).text = "The capital of France is ____."
        app.screen.query_one("#quiz-correct-answer-input", Input).value = "Paris"
        await controller.create_question()

        question_list = app.screen.query_one("#quiz-question-list", ListView)

        assert ("create_quiz", "local", "global", None, "Geography Review", "Capitals", None, None) in scope.calls
        assert ("create_question", "local", "new-quiz", "fill_blank", "The capital of France is ____.", "Paris", 1) in scope.calls
        assert question_list.children


@pytest.mark.asyncio
async def test_quizzes_attempt_flow_submits_answer_and_shows_summary():
    scope = FakeQuizScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=None,
        study_quiz_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.3)
        controller = app.screen.query_one(StudyWindow).quizzes_controller

        quiz_select = app.screen.query_one("#quiz-select", Select)
        quiz_select.value = "quiz-local-1"

        await controller.start_attempt()
        await pilot.pause(0.1)

        question = app.screen.query_one("#quiz-attempt-question", Static)
        answer = app.screen.query_one("#quiz-answer-input", Input)

        assert "capital of France" in _text(question)
        answer.value = "Paris"

        await controller.submit_current_answer()
        await pilot.pause(0.1)

        status = app.screen.query_one("#quiz-attempt-status", Static)

        assert (
            "submit_attempt",
            "server",
            "global",
            None,
            "attempt-1",
            [{"question_id": "question-local-1", "user_answer": "Paris"}],
        ) in scope.calls
        assert "Score: 2 / 2" in _text(status)


@pytest.mark.asyncio
async def test_active_attempt_disables_quiz_mutations_and_blocks_second_start():
    scope = FakeQuizScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=None,
        study_quiz_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.3)
        controller = app.screen.query_one(StudyWindow).quizzes_controller

        quiz_select = app.screen.query_one("#quiz-select", Select)
        quiz_select.value = "quiz-local-1"
        await controller.start_attempt()
        await pilot.pause(0.1)

        create_quiz_button = app.screen.query_one("#create-quiz-button", Button)
        delete_quiz_button = app.screen.query_one("#delete-quiz-button", Button)
        create_question_button = app.screen.query_one("#create-quiz-question-button", Button)
        delete_question_button = app.screen.query_one("#delete-quiz-question-button", Button)
        start_attempt_button = app.screen.query_one("#start-quiz-attempt-button", Button)
        submit_answer_button = app.screen.query_one("#submit-quiz-answer-button", Button)
        load_attempt_button = app.screen.query_one("#load-quiz-attempt-history-button", Button)
        history_select = app.screen.query_one("#quiz-attempt-history-select", Select)
        answer_input = app.screen.query_one("#quiz-answer-input", Input)

        first_attempt_id = controller.current_attempt_id
        first_questions = list(controller.current_attempt_questions)

        assert quiz_select.disabled is True
        assert create_quiz_button.disabled is True
        assert delete_quiz_button.disabled is True
        assert create_question_button.disabled is True
        assert delete_question_button.disabled is True
        assert start_attempt_button.disabled is True
        assert load_attempt_button.disabled is True
        assert history_select.disabled is True
        assert answer_input.disabled is False
        assert submit_answer_button.disabled is False
        for button in (
            create_quiz_button,
            delete_quiz_button,
            create_question_button,
            delete_question_button,
            start_attempt_button,
            load_attempt_button,
        ):
            assert "Submit the active quiz attempt" in str(button.tooltip)
        assert submit_answer_button.tooltip is None

        await controller.start_attempt()
        await pilot.pause(0.1)

        start_attempt_calls = [call for call in scope.calls if call[0] == "start_attempt"]
        assert len(start_attempt_calls) == 1
        assert controller.current_attempt_id == first_attempt_id
        assert controller.current_attempt_questions == first_questions


@pytest.mark.asyncio
async def test_active_attempt_blocks_create_and_delete_quiz_without_refreshing_selection():
    scope = FakeQuizScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=None,
        study_quiz_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.3)
        controller = app.screen.query_one(StudyWindow).quizzes_controller

        quiz_select = app.screen.query_one("#quiz-select", Select)
        quiz_select.value = "quiz-local-1"
        await controller.start_attempt()
        await pilot.pause(0.1)

        app.screen.query_one("#new-quiz-name-input", Input).value = "Blocked Quiz"
        app.screen.query_one("#new-quiz-description-input", Input).value = "Should not create"

        await controller.create_quiz()
        await controller.delete_quiz()
        await pilot.pause(0.1)

        assert not any(call[0] == "create_quiz" for call in scope.calls)
        assert not any(call[0] == "delete_quiz" for call in scope.calls)
        assert str(quiz_select.value) == "quiz-local-1"
        assert controller.current_attempt_id == "attempt-1"
        assert controller.current_attempt_questions


@pytest.mark.asyncio
async def test_active_attempt_blocks_loading_attempt_history_directly():
    scope = FakeQuizScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=None,
        study_quiz_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.3)
        controller = app.screen.query_one(StudyWindow).quizzes_controller

        quiz_select = app.screen.query_one("#quiz-select", Select)
        history_select = app.screen.query_one("#quiz-attempt-history-select", Select)
        quiz_select.value = "quiz-local-1"
        history_select.value = "attempt-1"

        await controller.start_attempt()
        await pilot.pause(0.1)
        await controller.load_selected_attempt()
        await pilot.pause(0.1)

        assert not any(call[0] == "get_attempt" for call in scope.calls)
        assert controller.current_attempt_id == "attempt-1"
        assert controller.current_attempt_questions


@pytest.mark.asyncio
async def test_quizzes_view_shows_explicit_empty_state_when_no_quizzes_exist():
    app_instance = SimpleNamespace(
        study_scope_service=None,
        study_quiz_scope_service=EmptyQuizScopeService(),
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.3)

        status = app.screen.query_one("#quiz-attempt-status", Static)

        status_text = _text(status)
        assert "No quizzes yet." in status_text
        assert "Create a quiz" in status_text
        assert "add questions" in status_text
        assert "start an attempt" in status_text


@pytest.mark.asyncio
async def test_workspace_quizzes_empty_state_explains_workspace_recovery_path():
    scope = WorkspaceFilteredQuizScopeService()
    scope.workspace_quizzes = []
    scope.workspace_attempts = []
    app_instance = SimpleNamespace(
        study_scope_service=None,
        study_quiz_scope_service=scope,
        pending_study_scope_context=StudyScopeContext(
            scope_type=StudyScopeType.WORKSPACE,
            workspace_id="ws-1",
            workspace_name="Research",
        ),
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.3)

        status = app.screen.query_one("#quiz-attempt-status", Static)
        create_button = app.screen.query_one("#create-quiz-button", Button)

        status_text = _text(status)
        assert "No quizzes in this workspace yet." in status_text
        assert "Create a workspace quiz" in status_text
        assert "switch to Global Study" in status_text
        assert create_button.disabled is False


@pytest.mark.asyncio
async def test_quizzes_view_deletes_selected_quiz_and_resets_selection():
    scope = FakeQuizScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=None,
        study_quiz_scope_service=scope,
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.3)
        controller = app.screen.query_one(StudyWindow).quizzes_controller

        app.screen.query_one("#delete-quiz-button", Button)

        await controller.delete_quiz()
        await pilot.pause(0.1)

        quiz_select = app.screen.query_one("#quiz-select", Select)
        status = app.screen.query_one("#quiz-attempt-status", Static)

        assert ("delete_quiz", "local", "quiz-local-1", None, False) in scope.calls
        assert _is_blank(quiz_select.value)
        assert "No quizzes yet." in _text(status)


@pytest.mark.asyncio
async def test_quizzes_view_deletes_selected_question_and_refreshes_question_list():
    scope = FakeQuizScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=None,
        study_quiz_scope_service=scope,
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.3)
        controller = app.screen.query_one(StudyWindow).quizzes_controller
        app.screen.query_one("#delete-quiz-question-button", Button)

        question_list = app.screen.query_one("#quiz-question-list", ListView)
        question_list.index = 0
        await controller.delete_question()
        await pilot.pause(0.1)

        assert ("delete_question", "local", "question-local-1", None, False) in scope.calls
        assert "No questions in this quiz." in _text(question_list.children[0].query_one(Static))


@pytest.mark.asyncio
async def test_quizzes_view_loads_attempt_history_into_summary_panel():
    scope = FakeQuizScopeService()
    app_instance = SimpleNamespace(
        study_scope_service=None,
        study_quiz_scope_service=scope,
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = StudyTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.3)
        controller = app.screen.query_one(StudyWindow).quizzes_controller

        history_select = app.screen.query_one("#quiz-attempt-history-select", Select)
        history_select.value = "attempt-1"

        await controller.load_selected_attempt()
        await pilot.pause(0.1)

        history_summary = app.screen.query_one("#quiz-attempt-history-summary", Static)
        status = app.screen.query_one("#quiz-attempt-status", Static)

        assert ("get_attempt", "local", "global", None, "attempt-1", True, True) in scope.calls
        assert "Score: 2 / 2" in _text(status)
        assert "The capital of France is ____." in _text(history_summary)
