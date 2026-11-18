"""Read-only mounted production Study event-queue audit."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input
from textual.message import Message
from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp, BUNDLED_STYLESHEET
from tldw_chatbook.Study_Interop import QuizScopeService, ServerQuizService
from tldw_chatbook.UI.Screens.study_screen import StudyScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import PendingHandoffStore

HERE = Path(__file__).parent
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
        await self.release.wait()
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


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("entry", ["shell", "inner"])
@private_profile_test
async def test_pending_start_section_event_delivery(request, entry):
    client = Client()
    app = Host(client)
    try:
        async with app.run_test(size=(180, 80)) as pilot:
            await settle(app, pilot)
            screen = app.screen
            screen.query_one("#view-quizzes-btn", Button).press()
            await settle(app, pilot)
            window = screen.study_window_widget
            assert screen.current_section == "quizzes"
            start = screen.query_one(
                "#quiz-start" if entry == "shell" else "#start-quiz-attempt-button",
                Button,
            )
            assert not start.disabled
            start.press()
            await asyncio.wait_for(client.entered.wait(), 5)
            # A queued Button.Pressed on the same real screen pump as shell Start.
            dashboard = screen.query_one("#view-dashboard-btn", Button)
            assert not dashboard.disabled
            dashboard.press()
            await asyncio.sleep(0.15)
            pending_section = screen.current_section
            app.post_message(AuditPing())
            await asyncio.sleep(0.05)
            app_pings_during_hold = app.pings
            assert app_pings_during_hold == 1
            # The direct Input assignment is diagnostic only; it is not a keyboard witness.
            name = window.query_one("#new-quiz-name-input", Input)
            name.focus()
            name.value = "later draft"
            await asyncio.sleep(0.05)
            child_draft = name.value
            client.release.set()
            await until(lambda: screen.current_section == "dashboard")
            await settle(app, pilot)
            receipt = {
                "entry": entry,
                "pending_section": pending_section,
                "after_release_section": screen.current_section,
                "client_calls": client.calls,
                "child_draft_during_hold": child_draft,
                "app_message_handled_during_hold": app_pings_during_hold,
                "held_at_real_server_api_await": True,
                "screen_type": type(screen).__name__,
                "window_type": type(window).__name__,
            }
            (HERE / (entry + ".json")).write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
            assert client.calls == 1
            assert pending_section == ("quizzes" if entry == "shell" else "dashboard")
    finally:
        client.release.set()
