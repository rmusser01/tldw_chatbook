"""TASK-36930: final Submit may publish only into its original Study lifetime."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy

import pytest
from textual.widgets import Button, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_study_quiz_start_responsiveness import (
    Host,
    ScopedClient,
    change_start_context,
    open_quizzes,
    settle,
    start_button,
    until,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Study_Interop import (
    LocalQuizService,
    QuizScopeService,
    ServerQuizService,
)
from tldw_chatbook.UI.Screens.study_scope_models import StudyScopeContext


class SubmitClient(ScopedClient):
    """Control only remote response timing; retain production routing/schema."""

    def __init__(self, failure=False):
        super().__init__()
        self.release.set()
        self.submit_entered = asyncio.Event()
        self.submit_release = asyncio.Event()
        self.submit_completed = 0
        self.submit_cancelled = 0
        self.failure = failure
        self.payloads = []
        self.hold_history = False
        self.history_entered = asyncio.Event()
        self.history_release = asyncio.Event()

    async def submit_quiz_attempt(self, attempt_id, payload):
        self.payloads.append((attempt_id, payload.model_dump()))
        self.submit_entered.set()
        try:
            await self.submit_release.wait()
        except asyncio.CancelledError:
            self.submit_cancelled += 1
            raise
        if self.failure:
            raise OSError("Definite failure before remote submission")
        self.submit_completed += 1
        return {"id": attempt_id, "quiz_id": 1, "score": 1, "total_possible": 1}

    async def list_quiz_attempts(self, **kwargs):
        if self.hold_history:
            self.hold_history = False
            self.history_entered.set()
            await self.history_release.wait()
            return [{"id": 21, "quiz_id": 1, "score": 1, "total_possible": 1}]
        return []


@asynccontextmanager
async def release_on_exit(client):
    try:
        yield
    finally:
        # Release while the app is still mounted, including assertion failures.
        client.submit_release.set()
        client.history_release.set()
        client.release.set()


async def drain_new(app, pilot, held):
    await pilot.pause()
    for worker in list(app.workers):
        if worker not in held:
            await worker.wait()
    await pilot.pause()


def snapshot(window):
    controller = window.quizzes_controller
    answer = window.query_one("#quiz-answer-input", Input)
    history = window.query_one("#quiz-attempt-history-select", Select)
    return {
        "attempt": controller.current_attempt_id,
        "questions": deepcopy(controller.current_attempt_questions),
        "answers": deepcopy(controller.current_attempt_answers),
        "index": controller.current_question_index,
        "history": deepcopy(controller.current_attempt_history),
        "history_selection": history.value,
        "history_options": list(history._options),
        "question": str(window.query_one("#quiz-attempt-question", Static).render()),
        "status": str(window.query_one("#quiz-attempt-status", Static).render()),
        "summary": str(
            window.query_one("#quiz-attempt-history-summary", Static).render()
        ),
        "draft": answer.value,
        "selection": answer.selection,
        "answer_disabled": answer.disabled,
        "controls": {
            button.id: button.disabled
            for button in window.query(Button)
            if button.id and ("quiz" in button.id or "attempt" in button.id)
        },
    }


async def replace_context(app, pilot, screen, window, transition, held):
    controller = window.quizzes_controller
    if transition in {"runtime-local", "runtime-away-back", "workspace-B"}:
        await change_start_context(app, pilot, screen, window, transition)
    elif transition == "scope-away-back":
        await change_start_context(app, pilot, screen, window, "workspace-B")
        await screen._apply_scope_context_and_refresh(
            StudyScopeContext(), study_window=window
        )
        await drain_new(app, pilot, held)
    elif transition == "attempt-replaced":
        # A same-context reset and real Start can reuse the same server ID.
        controller.reset_attempt_panel("Explicitly retired prior attempt.")
    elif transition == "panel-replaced":
        old_answer = window.query_one("#quiz-answer-input", Input)
        screen.query_one("#view-flashcards-btn", Button).press()
        await until(lambda: not old_answer.is_attached)
        screen.query_one("#view-quizzes-btn", Button).press()
        await until(lambda: len(window.query("#quiz-answer-input")) == 1)
        await drain_new(app, pilot, held)
        assert window.query_one("#quiz-answer-input", Input) is not old_answer
        return
    start_button(screen, "inner").press()
    await drain_new(app, pilot, held)
    assert controller.current_attempt_id is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True], ids=["success", "failure"])
@pytest.mark.parametrize(
    "transition",
    [
        "runtime-local",
        "runtime-away-back",
        "workspace-B",
        "scope-away-back",
        "attempt-replaced",
        "panel-replaced",
    ],
)
@private_profile_test
async def test_obsolete_submit_preserves_current_attempt_and_exact_draft(
    request, failure, transition
):
    client = SubmitClient(failure)
    db = CharactersRAGDB(db_path=":memory:", client_id="task-36930")
    local_quiz = db.create_quiz(name="Local current quiz")
    db.create_question(
        quiz_id=local_quiz,
        question_type="fill_blank",
        question_text="Local current question",
        correct_answer="8",
    )
    app = Host(client)
    app.instance.study_quiz_scope_service = QuizScopeService(
        local_service=LocalQuizService(db), server_service=ServerQuizService(client)
    )
    try:
        async with app.run_test(size=(180, 80)) as pilot, release_on_exit(client):
            screen, window = await open_quizzes(app, pilot)
            start_button(screen, "shell").press()
            await settle(app, pilot)
            answer = window.query_one("#quiz-answer-input", Input)
            answer.value = "4"
            await pilot.pause()
            window.query_one("#submit-quiz-answer-button", Button).press()
            await asyncio.wait_for(client.submit_entered.wait(), 5)
            held = set(app.workers)
            await replace_context(app, pilot, screen, window, transition, held)
            answer = window.query_one("#quiz-answer-input", Input)
            answer.value = "  exact newer unsent answer  "
            answer.cursor_position = 8
            await pilot.pause()
            before = snapshot(window)
            client.submit_release.set()
            await settle(app, pilot)
            assert snapshot(window) == before
            assert len(client.payloads) == 1
            assert client.submit_cancelled == 0
            assert client.submit_completed == int(not failure)
            expected = "Failed to submit" if failure else "submitted"
            assert any(
                expected in notice
                and "previous Study context" in notice
                and "21" in notice
                for notice in app.notices
            )
            if transition == "runtime-local":
                records = db.list_attempts(quiz_id=local_quiz)["items"]
                assert len(records) == 1 and str(records[0]["id"]) == before["attempt"]
                stored = db.get_attempt(records[0]["id"], include_answers=True)
                assert stored["completed_at"] is None and stored["answers"] == []
                assert before["question"] == "Local current question"
                assert not before["controls"]["submit-quiz-answer-button"]
            assert app._exception is None
    finally:
        db.close()


@pytest.mark.asyncio
@private_profile_test
async def test_current_submit_completes_and_preserves_normal_history_refresh(request):
    client = SubmitClient()
    client.submit_release.set()
    client.hold_history = False
    app = Host(client)
    async with app.run_test(size=(180, 80)) as pilot, release_on_exit(client):
        screen, window = await open_quizzes(app, pilot)
        start_button(screen, "inner").press()
        await settle(app, pilot)
        window.query_one("#quiz-answer-input", Input).value = "4"
        await pilot.pause()
        window.query_one("#submit-quiz-answer-button", Button).press()
        await settle(app, pilot)
        assert window.quizzes_controller.current_attempt_id is None
        assert (
            str(window.query_one("#quiz-attempt-status", Static).render())
            == "Score: 1 / 1"
        )
        assert window.query_one("#quiz-answer-input", Input).value == ""
        assert len(client.payloads) == client.submit_completed == 1
        assert client.submit_cancelled == 0
        assert app._exception is None


@pytest.mark.asyncio
@private_profile_test
async def test_submit_history_completion_cannot_publish_after_runtime_reset(request):
    client = SubmitClient()
    client.submit_release.set()
    db = CharactersRAGDB(db_path=":memory:", client_id="task-36930-history")
    quiz = db.create_quiz(name="Local current quiz")
    db.create_question(
        quiz_id=quiz,
        question_type="fill_blank",
        question_text="Local",
        correct_answer="8",
    )
    app = Host(client)
    app.instance.study_quiz_scope_service = QuizScopeService(
        local_service=LocalQuizService(db), server_service=ServerQuizService(client)
    )
    try:
        async with app.run_test(size=(180, 80)) as pilot, release_on_exit(client):
            screen, window = await open_quizzes(app, pilot)
            start_button(screen, "inner").press()
            await settle(app, pilot)
            window.query_one("#quiz-answer-input", Input).value = "4"
            await pilot.pause()
            client.hold_history = True
            window.query_one("#submit-quiz-answer-button", Button).press()
            await asyncio.wait_for(client.history_entered.wait(), 5)
            held = set(app.workers)
            await replace_context(app, pilot, screen, window, "runtime-local", held)
            window.query_one("#quiz-answer-input", Input).value = "  local draft  "
            await pilot.pause()
            before = snapshot(window)
            client.history_release.set()
            await settle(app, pilot)
            assert snapshot(window) == before
            assert client.submit_completed == 1 and client.submit_cancelled == 0
            assert any(
                "21" in notice and "previous Study context" in notice
                for notice in app.notices
            )
            assert app._exception is None
    finally:
        db.close()
