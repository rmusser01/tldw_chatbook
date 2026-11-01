import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import pytest
from textual.widgets import Button, Input, Static
from Tests.private_profile import private_profile_test
from Tests.UI.test_study_quiz_start_responsiveness import (
    Host,
    Client,
    QUESTION,
    open_quizzes,
    start_button,
    settle,
    until,
    change_start_context,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Study_Interop import (
    QuizScopeService,
    LocalQuizService,
    ServerQuizService,
)

OUT = Path(__file__).parent


def receipt(name, data):
    (OUT / (name + ".json")).write_text(json.dumps(data, indent=2), encoding="utf-8")


class SubmitClient(Client):
    def __init__(self):
        super().__init__()
        self.release.set()
        self.submit_entered = asyncio.Event()
        self.submit_release = asyncio.Event()
        self.submissions = []
        self.submit_completed = 0
        self.submit_cancelled = 0

    async def submit_quiz_attempt(self, attempt_id, payload):
        self.submissions.append(
            {"attempt_id": attempt_id, "answers": payload.model_dump()}
        )
        self.submit_entered.set()
        try:
            await self.submit_release.wait()
        except asyncio.CancelledError:
            self.submit_cancelled += 1
            raise
        self.submit_completed += 1
        return {"id": attempt_id, "quiz_id": 1, "score": 1, "total_possible": 1}


@asynccontextmanager
async def release_submit(client):
    try:
        yield
    finally:
        client.release.set()
        client.submit_release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["ordinary", "runtime-local"])
@private_profile_test
async def test_pending_submit_observation(request, transition):
    client = SubmitClient()
    db = CharactersRAGDB(db_path=":memory:", client_id="audit-after3248-submit")
    local_id = db.create_quiz(name="Local current quiz")
    db.create_question(
        quiz_id=local_id,
        question_type="fill_blank",
        question_text="Local current question",
        correct_answer="8",
    )
    app = Host(client)
    app.instance.study_quiz_scope_service = QuizScopeService(
        local_service=LocalQuizService(db), server_service=ServerQuizService(client)
    )
    try:
        async with app.run_test(size=(180, 80)) as pilot, release_submit(client):
            screen, window = await open_quizzes(app, pilot)
            controller = window.quizzes_controller
            start_button(screen, "shell").press()
            await settle(app, pilot)
            assert controller.current_attempt_id == "21"
            answer = window.query_one("#quiz-answer-input", Input)
            answer.value = "4"
            await pilot.pause()
            submit = window.query_one("#submit-quiz-answer-button", Button)
            assert not submit.disabled
            submit.press()
            await asyncio.wait_for(client.submit_entered.wait(), 5)
            held = set(app.workers)

            async def drain_new():
                await pilot.pause()
                for worker in list(app.workers):
                    if worker not in held:
                        await worker.wait()
                await pilot.pause()

            before = None
            if transition == "runtime-local":
                await change_start_context(app, pilot, screen, window, "runtime-local")
                assert answer.is_attached and controller.current_attempt_id is None
                assert controller.selected_quiz_label() == "Local current quiz"
                start = start_button(screen, "inner")
                assert not start.disabled
                start.press()
                await drain_new()
                assert controller.current_attempt_id
                assert (
                    str(window.query_one("#quiz-attempt-question", Static).render())
                    == "Local current question"
                )
                answer.value = "local unsent answer"
                await pilot.pause()

            def snapshot():
                return {
                    "attempt_id": controller.current_attempt_id,
                    "question": str(
                        window.query_one("#quiz-attempt-question", Static).render()
                    ),
                    "status": str(
                        window.query_one("#quiz-attempt-status", Static).render()
                    ),
                    "draft": answer.value,
                    "mode": controller._current_mode(),
                    "scope_generation": controller._scope_generation,
                    "selected_quiz": controller.selected_quiz_label(),
                    "submit_enabled": not submit.disabled,
                }

            before = snapshot()
            client.submit_release.set()
            await settle(app, pilot)
            after = snapshot()
            records = db.list_attempts(quiz_id=local_id)["items"]
            local_records = [
                {
                    "id": str(x["id"]),
                    "completed_at": x["completed_at"],
                    "answers": db.get_attempt(x["id"], include_answers=True)["answers"],
                }
                for x in records
            ]
            receipt(
                "submit-" + transition,
                {
                    "transition": transition,
                    "before": before,
                    "after": after,
                    "client_submissions": client.submissions,
                    "server_complete": client.submit_completed,
                    "server_cancelled": client.submit_cancelled,
                    "local_records": local_records,
                    "notices": app.notices,
                    "app_exception": repr(app._exception),
                },
            )
            assert app._exception is None
            assert client.submit_completed == 1 and client.submit_cancelled == 0
            if transition == "ordinary":
                assert after["attempt_id"] is None and after["status"] == "Score: 1 / 1"
                assert after["draft"] == ""
            else:
                assert (
                    after["attempt_id"] == before["attempt_id"]
                ), "old server Submit discarded the current local attempt"
                assert (
                    after["draft"] == before["draft"]
                ), "old server Submit erased current local draft"
                assert after["question"] == before["question"]
                assert after["status"] == before["status"]
    finally:
        client.submit_release.set()
        client.release.set()
        db.close()
