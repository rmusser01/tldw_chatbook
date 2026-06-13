from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.UI.Screens.study_screen import StudyScreen
from tldw_chatbook.UI.Screens.study_scope_models import StudyScopeContext, StudyScopeType
from tldw_chatbook.UI.Study_Window import StudyWindow


class DashboardStudyScopeService:
    def __init__(self):
        self.calls = []
        self.decks = [
            {"record_id": "local:study_deck:deck-1", "backing_id": "deck-1", "name": "Binary Trees"},
            {"record_id": "local:study_deck:deck-2", "backing_id": "deck-2", "name": "Graph Theory"},
        ]
        self.due_cards = [
            {"record_id": "local:study_flashcard:card-1", "backing_id": "card-1"},
            {"record_id": "local:study_flashcard:card-2", "backing_id": "card-2"},
        ]

    async def list_decks(self, *, mode=None, scope_type=None, workspace_id=None, limit=100, offset=0):
        self.calls.append(("list_decks", mode, scope_type, workspace_id, limit, offset))
        return list(self.decks)

    async def get_due_flashcards(self, *, mode=None, scope_type=None, workspace_id=None, limit=25):
        self.calls.append(("get_due_flashcards", mode, scope_type, workspace_id, limit))
        return list(self.due_cards[:limit])


class DashboardQuizScopeService:
    def __init__(self):
        self.calls = []
        self.quizzes = [
            {
                "record_id": "local:quiz:quiz-1",
                "backing_id": "quiz-1",
                "name": "Tree Drill",
                "total_questions": 1,
            },
            {
                "record_id": "local:quiz:quiz-2",
                "backing_id": "quiz-2",
                "name": "Graph Drill",
                "total_questions": 2,
            },
        ]
        self.questions = [
            {
                "record_id": "local:quiz_question:question-1",
                "backing_id": "question-1",
                "quiz_record_id": "local:quiz:quiz-1",
                "question_type": "fill_blank",
                "question_text": "A balanced binary tree has height ____.",
                "correct_answer": "log n",
                "points": 1,
                "order_index": 0,
            }
        ]

    async def list_quizzes(self, *, mode=None, scope_type=None, workspace_id=None, q=None, limit=100, offset=0):
        self.calls.append(("list_quizzes", mode, scope_type, workspace_id, q, limit, offset))
        return list(self.quizzes)

    async def list_questions(self, *, mode=None, quiz_id=None, q=None, include_answers=False, limit=100, offset=0):
        self.calls.append(("list_questions", mode, quiz_id, q, include_answers, limit, offset))
        return [question for question in self.questions if str(question.get("quiz_record_id", "")).endswith(str(quiz_id))]

    async def list_attempts(self, *, mode=None, scope_type=None, workspace_id=None, quiz_id=None, limit=100, offset=0):
        self.calls.append(("list_attempts", mode, scope_type, workspace_id, quiz_id, limit, offset))
        return []

    async def start_attempt(self, *, mode=None, scope_type=None, workspace_id=None, quiz_id=None):
        self.calls.append(("start_attempt", mode, scope_type, workspace_id, quiz_id))
        return {
            "record_id": "local:quiz_attempt:attempt-1",
            "backing_id": "attempt-1",
            "quiz_record_id": f"local:quiz:{quiz_id}",
            "questions": [
                {
                    "record_id": "local:quiz_question:question-1",
                    "backing_id": "question-1",
                    "quiz_record_id": f"local:quiz:{quiz_id}",
                    "question_type": "fill_blank",
                    "question_text": "A balanced binary tree has height ____.",
                    "points": 1,
                    "order_index": 0,
                }
            ],
        }


class StudyDashboardTestApp(App[None]):
    def __init__(self, app_instance):
        super().__init__()
        self._screen = StudyScreen(app_instance=app_instance)

    async def on_mount(self) -> None:
        await self.push_screen(self._screen)


def _build_app_instance() -> SimpleNamespace:
    return SimpleNamespace(
        study_scope_service=DashboardStudyScopeService(),
        study_quiz_scope_service=DashboardQuizScopeService(),
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )


def _text(widget: Static) -> str:
    return str(widget.render())


@pytest.mark.asyncio
async def test_study_section_bar_explains_compact_labels():
    app = StudyDashboardTestApp(_build_app_instance())

    async with app.run_test() as pilot:
        await pilot.pause(0.3)

        expected_tooltips = {
            "#view-dashboard-btn": "Review due cards, recent decks, quizzes, and resume study sessions.",
            "#view-structured-btn": "Build or follow structured learning paths.",
            "#view-flashcards-btn": "Review decks and spaced-repetition cards.",
            "#view-quizzes-btn": "Create, start, and review quizzes.",
            "#view-study-guide-btn": "Generate or open study guides from your material.",
            "#view-mindmaps-btn": "Explore topics as visual knowledge maps.",
            "#view-course-btn": "Create course outlines and study sequences.",
            "#view-learning-map-btn": "Open the learning map for relationships across study material.",
        }

        for selector, tooltip in expected_tooltips.items():
            button = app.screen.query_one(selector, Button)

            assert str(button.tooltip) == tooltip


@pytest.mark.asyncio
async def test_study_dashboard_surfaces_due_and_recent_items():
    app = StudyDashboardTestApp(_build_app_instance())

    async with app.run_test() as pilot:
        await pilot.pause(0.3)

        dashboard = app.screen.query_one("#study-dashboard")
        due = app.screen.query_one("#study-due-today", Static)
        recent_decks = app.screen.query_one("#study-recent-decks", Static)
        recent_quizzes = app.screen.query_one("#study-recent-quizzes", Static)
        resume_button = app.screen.query_one("#study-resume-last", Button)

        assert dashboard is not None
        assert "2 due today" in _text(due)
        assert "Binary Trees" in _text(recent_decks)
        assert "Tree Drill" in _text(recent_quizzes)
        assert resume_button is not None
        assert resume_button.disabled is True
        assert "No study session to resume" in str(resume_button.tooltip)
        assert "Open flashcards or quizzes" in str(resume_button.tooltip)


@pytest.mark.asyncio
async def test_study_dashboard_resume_action_returns_to_last_session():
    app = StudyDashboardTestApp(_build_app_instance())

    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        app.screen.current_study_session = {"section": "quizzes", "topic": "Binary Trees"}
        await pilot.pause(0.1)

        resume_button = app.screen.query_one("#study-resume-last", Button)
        assert resume_button.disabled is False
        assert "Resume the most recent study session" in str(resume_button.tooltip)

        await pilot.click("#study-resume-last")
        await pilot.pause(0.2)

        assert getattr(app.screen, "current_section", None) == "quizzes"


@pytest.mark.asyncio
async def test_study_quizzes_section_offers_shell_level_start_flow():
    app = StudyDashboardTestApp(_build_app_instance())

    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.4)

        quiz_session = app.screen.query_one("#quiz-session")
        quiz_start = app.screen.query_one("#quiz-start", Button)
        quiz_summary = app.screen.query_one("#quiz-session-summary", Static)

        assert quiz_session is not None
        assert quiz_start is not None
        assert "Tree Drill" in _text(quiz_summary)
        assert quiz_start.disabled is False
        assert "Start the selected quiz" in str(quiz_start.tooltip)

        await pilot.click("#quiz-start")
        await pilot.pause(0.3)

        study_window = app.screen.query_one(StudyWindow)
        quiz_status = app.screen.query_one("#quiz-session-status", Static)

        assert study_window.current_view == "quizzes"
        assert "Question 1 of 1." in _text(quiz_status)


@pytest.mark.asyncio
async def test_study_quizzes_start_action_explains_no_quiz_recovery():
    app_instance = _build_app_instance()
    app_instance.study_quiz_scope_service.quizzes = []
    app_instance.study_quiz_scope_service.questions = []
    app = StudyDashboardTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.4)

        quiz_start = app.screen.query_one("#quiz-start", Button)
        review_in_chat = app.screen.query_one("#quiz-open-in-chat", Button)
        quiz_summary = app.screen.query_one("#quiz-session-summary", Static)

        assert "No quizzes available yet." in _text(quiz_summary)
        assert quiz_start.disabled is True
        assert "Create or import a quiz" in str(quiz_start.tooltip)
        assert review_in_chat.disabled is True
        assert "Select a quiz before reviewing it in Chat" in str(review_in_chat.tooltip)


@pytest.mark.asyncio
async def test_study_quizzes_review_in_chat_stages_selected_quiz_context():
    app_instance = _build_app_instance()
    app_instance.open_chat_with_handoff = Mock()
    app = StudyDashboardTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.4)

        review_in_chat = app.screen.query_one("#quiz-open-in-chat", Button)

        assert review_in_chat.disabled is False
        assert "Review the selected quiz in Chat" in str(review_in_chat.tooltip)

        await pilot.click("#quiz-open-in-chat")
        await pilot.pause(0.2)

        app_instance.open_chat_with_handoff.assert_called_once()
        payload = app_instance.open_chat_with_handoff.call_args.args[0]

        assert isinstance(payload, ChatHandoffPayload)
        assert payload.source == "study"
        assert payload.item_type == "quiz"
        assert payload.title == "Tree Drill"
        assert "Tree Drill" in payload.body
        assert "Quiz status" in payload.body
        assert payload.suggested_prompt == "Help me review this quiz and identify what to study next."


@pytest.mark.asyncio
async def test_study_quizzes_review_in_chat_preserves_workspace_scope_metadata():
    app_instance = _build_app_instance()
    app_instance.current_runtime_backend = "server"
    app_instance.pending_study_scope_context = StudyScopeContext(
        scope_type=StudyScopeType.WORKSPACE,
        workspace_id="workspace-1",
        workspace_name="Research Workspace",
    )
    app_instance.open_chat_with_handoff = Mock()
    app = StudyDashboardTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        await pilot.click("#view-quizzes-btn")
        await pilot.pause(0.4)

        await pilot.click("#quiz-open-in-chat")
        await pilot.pause(0.2)

        payload = app_instance.open_chat_with_handoff.call_args.args[0]

        assert payload.runtime_backend == "server"
        assert payload.source_owner == "workspace"
        assert payload.source_selector_state == "workspace"
        assert payload.scope_type == "workspace"
        assert payload.workspace_id == "workspace-1"
        assert payload.backend_contracts == {
            "workspace_isolation": {"workspace_scope_id": "workspace-1"}
        }
