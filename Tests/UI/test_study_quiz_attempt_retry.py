"""Mounted quiz attempts retain one current answer across definite save failures.

The service seam fails before any write and otherwise delegates to real SQLite.
These cases catch final answers appended on each retry, stale edited answers,
and lost earlier answers or result/history publication on ordinary completion.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_study_quiz_create_drafts import _StudyHost, _settle
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Study_Interop import LocalQuizService, QuizScopeService
from tldw_chatbook.UI.Study_Window import StudyWindow


class _RecordedSubmission(LocalQuizService):
    def __init__(self, db, fail_once):
        super().__init__(db=db)
        self.fail_once = fail_once
        self.payloads = []

    def submit_attempt(self, attempt_id, *, answers):
        self.payloads.append([dict(answer) for answer in answers])
        if self.fail_once:
            self.fail_once = False
            raise OSError("Definite pre-write quiz submission failure")
        return super().submit_attempt(attempt_id, answers=answers)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question_count, initial_answer, retry_answer, expected_score",
    [
        pytest.param(1, "8", None, 1, id="ordinary-one-question"),
        pytest.param(2, "wrong", None, 1, id="ordinary-multi-question"),
        pytest.param(1, "8", "8", 1, id="retry-same-answer"),
        pytest.param(2, "wrong", "8", 2, id="retry-edited-correct"),
        pytest.param(2, "8", "wrong", 1, id="retry-edited-incorrect"),
    ],
)
@private_profile_test
async def test_attempt_submission_preserves_answers_grading_and_history(
    request, question_count, initial_answer, retry_answer, expected_score
):
    db = CharactersRAGDB(db_path=":memory:", client_id="study-attempt-retry")
    quiz_id = db.create_quiz(name="Retry practice")
    earlier_id = None
    if question_count == 2:
        earlier_id = db.create_question(
            quiz_id=quiz_id,
            question_type="fill_blank",
            question_text="Two plus two",
            correct_answer="4",
            points=1,
            order_index=0,
        )
    final_id = db.create_question(
        quiz_id=quiz_id,
        question_type="fill_blank",
        question_text="Four plus four",
        correct_answer="8",
        points=1,
        order_index=question_count - 1,
    )
    local = _RecordedSubmission(db, fail_once=retry_answer is not None)
    app = _StudyHost(QuizScopeService(local_service=local))
    try:
        async with app.run_test(size=(160, 60)) as pilot:
            await pilot.pause()
            app.query_one("#view-quizzes-btn", Button).press()
            await _settle(app, pilot)
            window = app.query_one(StudyWindow)
            controller = window.quizzes_controller
            start = window.query_one("#start-quiz-attempt-button", Button)
            assert not start.disabled
            start.press()
            await _settle(app, pilot)
            attempt_id = controller.current_attempt_id
            assert attempt_id
            answer = window.query_one("#quiz-answer-input", Input)
            submit = window.query_one("#submit-quiz-answer-button", Button)
            earlier_answers = []
            if earlier_id is not None:
                answer.focus()
                await pilot.press("4")
                assert not submit.disabled
                submit.press()
                await _settle(app, pilot)
                earlier_answers = [{"question_id": earlier_id, "user_answer": "4"}]
                assert answer.value == ""
                assert local.payloads == []
            question = window.query_one("#quiz-attempt-question", Static)
            status = window.query_one("#quiz-attempt-status", Static)
            assert str(question.render()) == "Four plus four"
            assert (
                str(status.render())
                == f"Question {question_count} of {question_count}."
            )
            answer.focus()
            await pilot.press(*initial_answer)
            assert not submit.disabled
            submit.press()
            await _settle(app, pilot)
            initial_payload = earlier_answers + [
                {"question_id": final_id, "user_answer": initial_answer}
            ]
            if retry_answer is not None:
                unsubmitted = db.get_attempt(attempt_id, include_answers=True)
                assert unsubmitted["completed_at"] is None
                assert unsubmitted["answers"] == []
                assert local.payloads == [initial_payload]
                assert answer.value == initial_answer
                assert controller.current_attempt_id == attempt_id
                assert str(question.render()) == "Four plus four"
                assert str(status.render()) == (
                    f"Question {question_count} of {question_count}."
                )
                assert not submit.disabled and not answer.disabled
                answer.focus()
                await pilot.press("home", "shift+end", *retry_answer)
                assert answer.value == retry_answer
                submit.press()
                await _settle(app, pilot)
            final_answer = initial_answer if retry_answer is None else retry_answer
            final_payload = earlier_answers + [
                {"question_id": final_id, "user_answer": final_answer}
            ]
            expected_payloads = (
                [initial_payload]
                if retry_answer is None
                else [initial_payload, final_payload]
            )
            assert local.payloads == expected_payloads
            stored = db.get_attempt(
                attempt_id, include_questions=True, include_answers=True
            )
            assert stored["completed_at"] is not None
            assert [
                {"question_id": item["question_id"], "user_answer": item["user_answer"]}
                for item in stored["answers"]
            ] == final_payload
            assert stored["score"] == expected_score
            assert stored["total_possible"] == question_count
            assert stored["score"] <= stored["total_possible"]
            assert [item["points_awarded"] for item in stored["answers"]] == (
                ([1] if earlier_id else []) + [int(final_answer == "8")]
            )
            assert str(status.render()) == f"Score: {expected_score} / {question_count}"
            assert str(question.render()) == "Attempt complete."
            assert answer.value == ""
            history = window.query_one("#quiz-attempt-history-select", Select)
            assert history.value == attempt_id
            load = window.query_one("#load-quiz-attempt-history-button", Button)
            assert not load.disabled
            load.press()
            await _settle(app, pilot)
            summary = str(
                window.query_one("#quiz-attempt-history-summary", Static).render()
            )
            assert f"Answer: {final_answer}" in summary
            assert summary.count("Four plus four") == 1
            if earlier_id is not None:
                assert "Answer: 4" in summary
                assert summary.count("Two plus two") == 1
            assert str(status.render()) == f"Score: {expected_score} / {question_count}"
    finally:
        db.close()
