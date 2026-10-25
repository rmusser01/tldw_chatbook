"""TASK-36729: real Study Start events remain responsive and single-flight."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static
from textual.message import Message
from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp, BUNDLED_STYLESHEET
from tldw_chatbook.Study_Interop import QuizScopeService, ServerQuizService
from tldw_chatbook.UI.Screens.study_screen import StudyScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import PendingHandoffStore

QUESTION = {
    "id": 11,
    "quiz_id": 1,
    "question_type": "fill_blank",
    "question_text": "Two plus two",
    "correct_answer": "4",
    "points": 1,
    "order_index": 0,
}


class Client:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.cancelled = 0
        self.fail_once = False

    async def list_quizzes(self, **kwargs):
        return [
            {"id": 1, "name": "Audit quiz", "total_questions": 1, "workspace_id": None}
        ]

    async def list_quiz_questions(self, *args, **kwargs):
        return [dict(QUESTION)]

    async def list_quiz_attempts(self, **kwargs):
        return []

    async def start_quiz_attempt(self, *args, **kwargs):
        self.calls += 1
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self.fail_once:
            self.fail_once = False
            raise OSError("Definite failure before attempt creation")
        return {"id": 21, "quiz_id": 1, "questions": [dict(QUESTION)]}


class Study:
    async def list_decks(self, **kwargs):
        return []

    async def get_due_flashcards(self, **kwargs):
        return []


class AuditPing(Message):
    pass


class Host(ConsolidatedCSSApp):
    CSS_PATH = [BUNDLED_STYLESHEET]

    def __init__(self, client):
        super().__init__()
        self.pings = 0
        self.notices = []
        self.instance = SimpleNamespace(
            study_scope_service=Study(),
            study_quiz_scope_service=QuizScopeService(
                server_service=ServerQuizService(client)
            ),
            current_runtime_backend="server",
            runtime_backend=None,
            app_config={},
            pending_handoffs=PendingHandoffStore(),
            notify=lambda *args, **kwargs: None,
        )

    def notify(self, message, **kwargs):
        self.notices.append(str(message))
        return super().notify(message, **kwargs)

    def on_audit_ping(self):
        self.pings += 1

    def on_mount(self):
        self.push_screen(StudyScreen(self.instance))


async def until(predicate, seconds=5):
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(0.01)


async def settle(app, pilot):
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


@asynccontextmanager
async def release_on_exit(client):
    try:
        yield
    finally:
        client.release.set()


async def open_quizzes(app, pilot):
    await settle(app, pilot)
    screen = app.screen
    screen.query_one("#view-quizzes-btn", Button).press()
    await settle(app, pilot)
    assert screen.current_section == "quizzes"
    return screen, screen.study_window_widget


def start_button(screen, entry):
    return screen.query_one(
        "#quiz-start" if entry == "shell" else "#start-quiz-attempt-button", Button
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["shell", "inner"])
@private_profile_test
async def test_pending_start_delivers_dashboard_and_presents_result(request, entry):
    client = Client()
    app = Host(client)
    try:
        async with app.run_test(size=(180, 80)) as pilot, release_on_exit(client):
            screen, window = await open_quizzes(app, pilot)
            start = start_button(screen, entry)
            assert not start.disabled
            start.press()
            await asyncio.wait_for(client.entered.wait(), 5)
            dashboard = screen.query_one("#view-dashboard-btn", Button)
            assert not dashboard.disabled
            dashboard.press()
            app.post_message(AuditPing())
            await until(lambda: app.pings == 1)
            # The independent App pump is live while the real server await is held.
            try:
                await until(lambda: screen.current_section == "dashboard", seconds=1)
            finally:
                client.release.set()
            await settle(app, pilot)
            controller = window.quizzes_controller
            assert controller.current_attempt_id == "21"
            assert len(controller.current_attempt_questions) == 1
            assert (
                str(window.query_one("#quiz-attempt-question", Static).render())
                == "Two plus two"
            )
            assert screen.current_section == "dashboard"
            assert client.calls == 1 and client.cancelled == 0
            assert app._exception is None
    finally:
        client.release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first,second",
    [("shell", "shell"), ("inner", "inner"), ("shell", "inner"), ("inner", "shell")],
)
@private_profile_test
async def test_repeated_start_preserves_one_admitted_request(request, first, second):
    client = Client()
    app = Host(client)
    try:
        async with app.run_test(size=(180, 80)) as pilot, release_on_exit(client):
            screen, window = await open_quizzes(app, pilot)
            start_button(screen, first).press()
            await asyncio.wait_for(client.entered.wait(), 5)
            repeated = start_button(screen, second)
            # Button.press suppresses its active animation; wait for a real second gesture.
            await until(lambda: not repeated.has_class("-active"))
            assert not repeated.disabled
            repeated.press()
            app.post_message(AuditPing())
            await until(lambda: app.pings == 1)
            await asyncio.sleep(0.25)
            assert client.calls == 1 and client.cancelled == 0
            client.release.set()
            await settle(app, pilot)
            assert client.calls == 1 and client.cancelled == 0
            assert window.quizzes_controller.current_attempt_id == "21"
            assert app._exception is None
    finally:
        client.release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["shell", "inner"])
@pytest.mark.parametrize("failure", ["request", "policy"])
@private_profile_test
async def test_start_refusal_or_failure_allows_retry(request, entry, failure):
    client = Client()
    client.release.set()
    client.fail_once = failure == "request"
    app = Host(client)
    if failure == "policy":
        app.instance.require_ui_action_allowed = lambda **kwargs: SimpleNamespace(
            allowed=False
        )
    async with app.run_test(size=(180, 80)) as pilot, release_on_exit(client):
        screen, window = await open_quizzes(app, pilot)
        start = start_button(screen, entry)
        assert not start.disabled
        start.press()
        await settle(app, pilot)
        assert window.quizzes_controller.current_attempt_id is None
        assert client.calls == (1 if failure == "request" else 0)
        if failure == "request":
            assert "Failed to start quiz attempt." in app.notices
        else:
            app.instance.require_ui_action_allowed = lambda **kwargs: SimpleNamespace(
                allowed=True
            )
        assert not start.disabled
        await until(lambda: not start.has_class("-active"))
        start.press()
        await settle(app, pilot)
        assert window.quizzes_controller.current_attempt_id == "21"
        assert client.calls == (2 if failure == "request" else 1)
        assert client.cancelled == 0
        assert (
            str(window.query_one("#quiz-attempt-status", Static).render())
            == "Question 1 of 1."
        )
        assert app._exception is None
