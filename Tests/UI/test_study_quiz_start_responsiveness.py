"""TASK-36729: real Study Start events remain responsive and single-flight."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Static
from textual.message import Message
from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp, BUNDLED_STYLESHEET
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Study_Interop import (
    LocalQuizService,
    QuizScopeService,
    ServerQuizService,
)
from tldw_chatbook.UI.Screens.study_scope_models import (
    StudyScopeContext,
    StudyScopeType,
)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["shell", "inner"])
@pytest.mark.parametrize(
    "return_before_response", [False, True], ids=["return-after", "return-before"]
)
@private_profile_test
async def test_start_completion_respects_replaced_quiz_panel(
    request, entry, return_before_response
):
    client = Client()
    app = Host(client)
    async with app.run_test(size=(180, 80)) as pilot, release_on_exit(client):
        screen, window = await open_quizzes(app, pilot)
        original_answer = window.query_one("#quiz-answer-input", Input)
        start_button(screen, entry).press()
        await asyncio.wait_for(client.entered.wait(), 5)
        held_workers = set(app.workers)
        screen.query_one("#view-flashcards-btn", Button).press()
        await until(
            lambda: screen.current_section == "flashcards"
            and not original_answer.is_attached
        )
        app.post_message(AuditPing())
        await until(lambda: app.pings == 1)
        if return_before_response:
            screen.query_one("#view-quizzes-btn", Button).press()
            # Initialization workers complete independently of the held Start worker.
            await until(lambda: len(window.query("#quiz-answer-input")) == 1)
            await until(lambda: not start_button(screen, entry).disabled)
            await pilot.pause()
            for worker in list(app.workers):
                if worker not in held_workers:
                    await worker.wait()
            await pilot.pause()
            replacement_answer = window.query_one("#quiz-answer-input", Input)
            assert replacement_answer is not original_answer
            replacement_answer.value = "replacement draft"
            await pilot.pause()
            assert replacement_answer.value == "replacement draft"
            assert not client.release.is_set()
        client.release.set()
        await settle(app, pilot)
        assert app._exception is None
        assert client.calls == 1 and client.cancelled == 0
        if return_before_response:
            assert replacement_answer.value == "replacement draft"
        assert any("21" in message and "panel" in message for message in app.notices)
        if not return_before_response:
            assert screen.current_section == "flashcards"
            screen.query_one("#view-quizzes-btn", Button).press()
            await settle(app, pilot)
        assert window.quizzes_controller.current_attempt_id is None
        start = start_button(screen, entry)
        assert not start.disabled
        start.press()
        await settle(app, pilot)
        assert window.quizzes_controller.current_attempt_id == "21"
        assert client.calls == 2 and client.cancelled == 0
        assert (
            str(window.query_one("#quiz-attempt-question", Static).render())
            == "Two plus two"
        )
        assert app._exception is None


class ScopedClient(Client):
    """Hold only the external Start call; keep production scope/backend services."""

    def __init__(self):
        super().__init__()
        self.submissions = []

    async def list_quizzes(self, **kwargs):
        return [
            {
                "id": 1,
                "name": "Global server quiz",
                "total_questions": 1,
                "workspace_id": None,
            },
            {
                "id": 2,
                "name": "Workspace B quiz",
                "total_questions": 1,
                "workspace_id": "B",
            },
        ]

    async def list_quiz_questions(self, quiz_id, **kwargs):
        question = dict(QUESTION)
        if quiz_id == 2:
            question.update(id=12, quiz_id=2, question_text="Workspace B question")
        return [question]

    async def start_quiz_attempt(self, quiz_id):
        attempt = await super().start_quiz_attempt(quiz_id)
        attempt["quiz_id"] = quiz_id
        attempt["questions"] = await self.list_quiz_questions(quiz_id)
        return attempt

    async def submit_quiz_attempt(self, attempt_id, payload):
        self.submissions.append(attempt_id)
        return {"id": attempt_id, "quiz_id": 1, "score": 1, "total_possible": 1}


async def change_start_context(app, pilot, screen, window, transition):
    """Use the production scope/runtime observer while leaving Start pending."""
    held = set(app.workers)

    async def drain_new():
        await pilot.pause()
        for worker in list(app.workers):
            if worker not in held:
                await worker.wait()
        await pilot.pause()

    if transition == "workspace-B":
        await screen._apply_scope_context_and_refresh(
            StudyScopeContext(
                scope_type=StudyScopeType.WORKSPACE,
                workspace_id="B",
                workspace_name="Workspace B",
            ),
            study_window=window,
        )
        await drain_new()
    else:
        app.instance.current_runtime_backend = "local"
        await screen.handle_runtime_backend_changed("local")
        await drain_new()
        if transition == "runtime-away-back":
            app.instance.current_runtime_backend = "server"
            await screen.handle_runtime_backend_changed("server")
            await drain_new()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["shell", "inner"])
@pytest.mark.parametrize(
    "transition", ["runtime-local", "workspace-B", "runtime-away-back"]
)
@private_profile_test
async def test_start_completion_cannot_adopt_a_reset_context(
    request, entry, transition
):
    client = ScopedClient()
    db = CharactersRAGDB(db_path=":memory:", client_id="task-36899")
    local_quiz_id = db.create_quiz(name="Local only quiz")
    db.create_question(
        quiz_id=local_quiz_id,
        question_type="fill_blank",
        question_text="Local question",
        correct_answer="4",
    )
    app = Host(client)
    app.instance.study_quiz_scope_service = QuizScopeService(
        local_service=LocalQuizService(db),
        server_service=ServerQuizService(client),
    )
    try:
        async with app.run_test(size=(180, 80)) as pilot, release_on_exit(client):
            screen, window = await open_quizzes(app, pilot)
            answer = window.query_one("#quiz-answer-input", Input)
            start_button(screen, entry).press()
            await asyncio.wait_for(client.entered.wait(), 5)
            await change_start_context(app, pilot, screen, window, transition)
            assert answer.is_attached
            controller = window.quizzes_controller
            assert controller.current_attempt_id is None
            expected_label = (
                "Workspace B quiz"
                if transition == "workspace-B"
                else (
                    "Local only quiz"
                    if transition == "runtime-local"
                    else "Global server quiz"
                )
            )
            assert controller.selected_quiz_label() == expected_label
            answer.value = "new-context draft"
            await pilot.pause()
            status = str(window.query_one("#quiz-attempt-status", Static).render())
            question = str(window.query_one("#quiz-attempt-question", Static).render())
            client.release.set()
            await settle(app, pilot)
            assert controller.current_attempt_id is None
            assert answer.value == "new-context draft"
            assert (
                str(window.query_one("#quiz-attempt-status", Static).render()) == status
            )
            assert (
                str(window.query_one("#quiz-attempt-question", Static).render())
                == question
            )
            submit = window.query_one("#submit-quiz-answer-button", Button)
            assert submit.disabled
            submit.press()
            await settle(app, pilot)
            assert db.list_attempts(quiz_id=local_quiz_id) == {"items": [], "count": 0}
            assert client.submissions == []
            assert any("21" in notice and "context" in notice for notice in app.notices)
            # The settled old admission must not strand the new context.
            start = start_button(screen, entry)
            assert not start.disabled
            start.press()
            await settle(app, pilot)
            assert controller.current_attempt_id is not None
            assert str(window.query_one("#quiz-attempt-question", Static).render()) == (
                "Local question"
                if transition == "runtime-local"
                else (
                    "Workspace B question"
                    if transition == "workspace-B"
                    else "Two plus two"
                )
            )
            answer.value = "4"
            await pilot.pause()
            submit.press()
            await settle(app, pilot)
            assert (
                str(window.query_one("#quiz-attempt-status", Static).render())
                == "Score: 1 / 1"
            )
            if transition == "runtime-local":
                attempts = db.list_attempts(quiz_id=local_quiz_id)["items"]
                assert len(attempts) == 1 and attempts[0]["score"] == 1
                assert client.submissions == []
            else:
                assert client.submissions == [21]
            assert app._exception is None
    finally:
        client.release.set()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["runtime-local", "runtime-away-back"])
@private_profile_test
async def test_failed_start_reports_the_previous_context_and_allows_retry(
    request, transition
):
    client = ScopedClient()
    client.fail_once = True
    app = Host(client)
    db = CharactersRAGDB(db_path=":memory:", client_id="task-36899-failure")
    app.instance.study_quiz_scope_service = QuizScopeService(
        local_service=LocalQuizService(db), server_service=ServerQuizService(client)
    )
    request.addfinalizer(db.close)
    async with app.run_test(size=(180, 80)) as pilot, release_on_exit(client):
        screen, window = await open_quizzes(app, pilot)
        start_button(screen, "shell").press()
        await asyncio.wait_for(client.entered.wait(), 5)
        await change_start_context(app, pilot, screen, window, transition)
        answer = window.query_one("#quiz-answer-input", Input)
        answer.value = "new-context draft"
        status = str(window.query_one("#quiz-attempt-status", Static).render())
        client.release.set()
        await settle(app, pilot)
        assert answer.value == "new-context draft"
        assert window.quizzes_controller.current_attempt_id is None
        assert str(window.query_one("#quiz-attempt-status", Static).render()) == status
        assert any("previous Study context" in notice for notice in app.notices)
        if transition == "runtime-away-back":
            start_button(screen, "shell").press()
            await settle(app, pilot)
            assert window.quizzes_controller.current_attempt_id == "21"
        assert app._exception is None
