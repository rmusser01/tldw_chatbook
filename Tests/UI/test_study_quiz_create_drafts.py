"""Pending Study creates acknowledge only the submitted authoring fields.

Controlled suspension models an awaited service response; released successes
use the real scope/local services and SQLite, not fabricated result records.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Study_Interop import LocalQuizService, QuizScopeService
from tldw_chatbook.UI.Study_Window import StudyWindow


class _HeldCreate:
    def __init__(self, delegate, verb):
        self.delegate = delegate
        self.verb = verb
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.fail = False

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    async def _create(self, verb, kwargs):
        if verb == self.verb:
            self.entered.set()
            await self.release.wait()
            if self.fail:
                raise OSError("Create backend unavailable")
        return await getattr(self.delegate, verb)(**kwargs)

    async def create_quiz(self, **kwargs):
        return await self._create("create_quiz", kwargs)

    async def create_question(self, **kwargs):
        return await self._create("create_question", kwargs)


class _StudyHost(ConsolidatedCSSApp):
    def __init__(self, scope):
        super().__init__()
        self.instance = SimpleNamespace(
            study_scope_service=None,
            study_quiz_scope_service=scope,
            current_runtime_backend="local",
            runtime_backend=None,
            app_config={},
            notify=lambda *args, **kwargs: None,
        )

    def compose(self):
        yield StudyWindow(self.instance)


async def _settle(app, pilot):
    async with asyncio.timeout(8):
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()


@asynccontextmanager
async def _mounted_form(verb):
    db = CharactersRAGDB(db_path=":memory:", client_id="study-create-drafts")
    quiz_id = db.create_quiz(name="Existing quiz")
    scope = _HeldCreate(QuizScopeService(local_service=LocalQuizService(db=db)), verb)
    app = _StudyHost(scope)
    try:
        async with app.run_test(size=(160, 60)) as pilot:
            await pilot.pause()
            app.query_one("#view-quizzes-btn", Button).press()
            await _settle(app, pilot)
            window = app.query_one(StudyWindow)
            if verb == "create_quiz":
                first = window.query_one("#new-quiz-name-input", Input)
                second = window.query_one("#new-quiz-description-input", Input)
                button = window.query_one("#create-quiz-button", Button)
            else:
                first = window.query_one("#quiz-question-text", TextArea)
                second = window.query_one("#quiz-correct-answer-input", Input)
                button = window.query_one("#create-quiz-question-button", Button)
            yield app, pilot, db, quiz_id, scope, first, second, button
    finally:
        scope.release.set()
        db.close()


def _seed(first, second):
    if isinstance(first, TextArea):
        first.load_text("  Submitted question  ")
        second.value = "  Submitted answer  "
    else:
        first.value = "  Submitted quiz  "
        second.value = "  Submitted description  "


def _draft(first, second):
    return (first.text if isinstance(first, TextArea) else first.value, second.value)


def _stored(db, quiz_id, verb):
    if verb == "create_quiz":
        return [
            (quiz["name"], quiz["description"])
            for quiz in db.list_quizzes()["items"]
            if quiz["id"] != quiz_id
        ]
    return [
        (question["question_text"], question["correct_answer"])
        for question in db.list_questions(quiz_id, include_answers=True)["items"]
    ]


async def _start(button, scope):
    scope.entered.clear()
    scope.release.clear()
    button.press()
    async with asyncio.timeout(8):
        await scope.entered.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["create_quiz", "create_question"])
@private_profile_test
async def test_create_preserves_later_raw_fields_and_editing_context(request, verb):
    async with _mounted_form(verb) as state:
        app, pilot, db, quiz_id, scope, first, second, button = state
        submitted = (
            ("Submitted quiz", "Submitted description")
            if verb == "create_quiz"
            else ("Submitted question", "Submitted answer")
        )
        # Each field independently, both together, and raw-only whitespace.
        for index, (edited, suffix) in enumerate(
            [((0,), "-new"), ((1,), "-new"), ((0, 1), "-new"), ((0,), " ")], 1
        ):
            _seed(first, second)
            await pilot.pause()
            await _start(button, scope)
            assert not first.disabled and not second.disabled
            for field_index in edited:
                field = (first, second)[field_index]
                field.focus()
                await pilot.press("end", *suffix)
            focused = (first, second)[edited[-1]]
            await pilot.press("shift+left")
            await pilot.pause()
            before = _draft(first, second)
            raw_submission = tuple(f"  {value}  " for value in submitted)
            assert before == tuple(
                value + suffix if i in edited else value
                for i, value in enumerate(raw_submission)
            )
            selection = focused.selection
            scope.release.set()
            await _settle(app, pilot)
            # Check actual records independently from current widget drafts.
            assert _stored(db, quiz_id, verb) == [submitted] * index
            expected = tuple(
                value if i in edited else "" for i, value in enumerate(before)
            )
            assert _draft(first, second) == expected
            assert app.focused is focused
            assert focused.selection == selection


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["create_quiz", "create_question"])
@private_profile_test
async def test_unchanged_create_clears_once_and_failed_create_can_retry(request, verb):
    async with _mounted_form(verb) as state:
        app, pilot, db, quiz_id, scope, first, second, button = state
        submitted = (
            ("Submitted quiz", "Submitted description")
            if verb == "create_quiz"
            else ("Submitted question", "Submitted answer")
        )
        _seed(first, second)
        await pilot.pause()
        await _start(button, scope)
        scope.release.set()
        await _settle(app, pilot)
        assert _stored(db, quiz_id, verb) == [submitted]
        assert _draft(first, second) == ("", "")

        _seed(first, second)
        await pilot.pause()
        scope.fail = True
        await _start(button, scope)
        first.focus()
        await pilot.press("end", *"-retry")
        await pilot.pause()
        retry_draft = _draft(first, second)
        selection = first.selection
        scope.release.set()
        await _settle(app, pilot)
        assert _stored(db, quiz_id, verb) == [submitted]
        assert _draft(first, second) == retry_draft
        assert app.focused is first
        assert first.selection == selection

        scope.fail = False
        await _start(button, scope)
        scope.release.set()
        await _settle(app, pilot)
        assert sorted(_stored(db, quiz_id, verb)) == sorted(
            [
                submitted,
                (retry_draft[0].strip(), retry_draft[1].strip()),
            ]
        )
        assert _draft(first, second) == ("", "")
