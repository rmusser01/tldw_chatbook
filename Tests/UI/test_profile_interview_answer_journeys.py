"""Interview entry must stay readable and recoverable through keyboard submission."""

from dataclasses import replace

import pytest
from textual.widgets import Button, Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_profile_interview_screen import _Coordinator, _Host, _push
from tldw_chatbook.UI.Screens.profile_interview_screen import ProfileInterviewScreen


def _fixed(coordinator):
    coordinator.session = replace(
        coordinator.session,
        kind="workspace",
        mode="fixed",
        provider_label="Fixed local questionnaire",
        model_id=None,
        external_retention_notice="No external provider is used by the fixed questionnaire.",
        draft_is_memory_only=True,
    )
    return coordinator


def _paint(screen):
    return "\n".join(strip.text for strip in screen._compositor.render_strips())


async def _enter_answer(host, pilot, screen, value):
    answer = screen.query_one("#profile-interview-answer", Input)
    answer.focus()
    await pilot.pause()
    await pilot.press(*value)
    await pilot.wait_for_scheduled_animations()
    await pilot.pause()
    assert host.focused is answer
    return answer


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_question_and_typed_answer_paint_together(request, size, theme):
    coordinator = _fixed(_Coordinator())
    host = _Host()
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        screen = ProfileInterviewScreen(
            coordinator, kind="workspace", scope_id="workspace", mode="fixed"
        )
        await _push(host, screen)
        await _enter_answer(host, pilot, screen, "Accessible reader")
        text = _paint(screen)
        assert (
            coordinator.session.question.text in text,
            "Accessible reader" in text,
        ) == (True, True), text
        await pilot.press("tab")
        assert host.focused is screen.query_one("#profile-interview-submit", Button)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ValueError, RuntimeError])
@private_profile_test
async def test_failed_answer_retains_entry_and_retry_clears_only_after_acceptance(
    request, failure
):
    class FailsOnce(_Coordinator):
        failed = False

        def answer(self, session_id, answer):
            if not self.failed:
                self.failed = True
                self._record("rejected", session_id, answer)
                raise failure("private failure details")
            return super().answer(session_id, answer)

    coordinator = _fixed(FailsOnce())
    host = _Host()
    async with host.run_test(size=(80, 24)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator, kind="workspace", scope_id="workspace", mode="fixed"
        )
        await _push(host, screen)
        answer = await _enter_answer(host, pilot, screen, "Accessible reader")
        await pilot.press("tab", "enter")
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert answer.value == "Accessible reader"
        assert "private failure details" not in _paint(screen)
        answer.focus()
        await pilot.pause()
        await pilot.press("end", *" for everyone", "tab", "enter")
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert answer.value == ""
        assert [call for call in coordinator.calls if call[0] == "answer"] == [
            ("answer", "session-1", "Accessible reader for everyone")
        ]
        assert coordinator.session.question_attempts == 2


@pytest.mark.asyncio
@private_profile_test
async def test_closing_final_review_restores_review_action_without_writing(request):
    from tldw_profile_core import InterviewTurn

    from tldw_chatbook.Widgets.Settings_Widgets.personal_context_review_modal import (
        PersonalContextReviewModal,
    )

    class ReviewCoordinator(_Coordinator):
        def finish_early(self, session_id):
            self.session = replace(self.session, status="review", question=None)
            return super().finish_early(session_id)

    coordinator = _fixed(ReviewCoordinator())
    coordinator.session = replace(
        coordinator.session,
        turns=(InterviewTurn(question_id="question-1", answer="Accessible reader"),),
    )
    host = _Host()
    async with host.run_test(size=(80, 24)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator, kind="workspace", scope_id="workspace", mode="fixed"
        )
        await _push(host, screen)
        screen.query_one("#profile-interview-finish", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(host.screen, PersonalContextReviewModal)
        await pilot.press("escape")
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert host.screen is screen
        review = screen.query_one("#profile-interview-review", Button)
        assert review.display and not review.disabled
        review.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(host.screen, PersonalContextReviewModal)
        assert not host.results


@pytest.mark.asyncio
@private_profile_test
async def test_review_payload_value_remains_readable_while_focused(request):
    from Tests.UI.test_personal_context_review_modal import (
        _Coordinator as ReviewCoordinator,
    )
    from tldw_chatbook.Widgets.Settings_Widgets.personal_context_review_modal import (
        PersonalContextReviewModal,
    )

    coordinator = ReviewCoordinator()
    host = _Host()
    async with host.run_test(size=(80, 24)) as pilot:
        modal = PersonalContextReviewModal(
            coordinator, session_id="session-1", diff=coordinator.diff
        )
        await host.push_screen(modal)
        field = modal.query_one("#personal-context-review-value-0", Input)
        field.focus()
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        assert field.value
        assert field.value in _paint(modal)


@pytest.mark.asyncio
@private_profile_test
async def test_queued_submit_does_not_duplicate_a_pending_answer(request):
    import asyncio
    import threading

    class Delayed(_Coordinator):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def answer(self, session_id, answer):
            self._record("pending", session_id, answer)
            self.entered.set()
            assert self.release.wait(15)
            return super().answer(session_id, answer)

    coordinator = _fixed(Delayed())
    host = _Host()
    try:
        async with host.run_test(size=(80, 24)) as pilot:
            screen = ProfileInterviewScreen(
                coordinator, kind="workspace", scope_id="workspace", mode="fixed"
            )
            await _push(host, screen)
            answer = await _enter_answer(host, pilot, screen, "Accessible reader")
            button = screen.query_one("#profile-interview-submit", Button)
            await pilot.press("tab", "enter")
            async with asyncio.timeout(5):
                while not coordinator.entered.is_set():
                    await pilot.pause(0.02)
            # A press already queued before disabling must not submit twice.
            button.post_message(Button.Pressed(button))
            await pilot.pause()
            assert answer.disabled and answer.value == "Accessible reader"
            assert (
                len([call for call in coordinator.calls if call[0] == "pending"]) == 1
            )
            coordinator.release.set()
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert answer.value == "" and not answer.disabled
            assert len([call for call in coordinator.calls if call[0] == "answer"]) == 1
    finally:
        coordinator.release.set()


@pytest.mark.asyncio
@private_profile_test
async def test_review_return_preserves_applied_edit_without_rebuilding_answers(request):
    from Tests.Personal_Context.test_interview_coordinator import _coordinator
    from tldw_chatbook.Widgets.Settings_Widgets.personal_context_review_modal import (
        PersonalContextReviewModal,
    )

    coordinator = _coordinator()
    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")
    coordinator.answer(session.session_id, "Original answer")
    original_finish = coordinator.finish_early
    finishes = []

    def finish(session_id):
        finishes.append(session_id)
        return original_finish(session_id)

    coordinator.finish_early = finish
    host = _Host()
    async with host.run_test(size=(80, 24)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="fixed",
            session_id=session.session_id,
        )
        await _push(host, screen)
        screen.query_one("#profile-interview-finish", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(host.screen, PersonalContextReviewModal)
        modal = host.screen
        field = modal.query_one("#personal-context-review-value-0", Input)
        field.focus()
        await pilot.pause()
        await pilot.press("home", "shift+end", "backspace", *"Reviewed correction")
        modal.query_one("#personal-context-review-apply-0", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert (
            coordinator.review(session.session_id)
            .changes[0]
            .change.proposed_payload.value
            == "Reviewed correction"
        )

        def unavailable_resume(_session_id):
            raise RuntimeError("temporary draft read failure")

        coordinator.resume = unavailable_resume
        await pilot.press("escape")
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert host.screen is screen
        review = screen.query_one("#profile-interview-review", Button)
        assert review.display and not review.disabled
        review.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(host.screen, PersonalContextReviewModal)
        assert (
            coordinator.review(session.session_id)
            .changes[0]
            .change.proposed_payload.value
            == "Reviewed correction"
        )
        assert finishes == [session.session_id]
