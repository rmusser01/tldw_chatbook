from __future__ import annotations

from dataclasses import replace
import threading

import pytest
from textual.widgets import Button, Input, Static
from tldw_profile_core import (
    AgentVisibility,
    InterviewProposedChange,
    InterviewQuestion,
    InterviewTurn,
    PreferencePayload,
    ProfileControls,
    ProposalOperation,
    SemanticKey,
    SyncMode,
)

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Personal_Context.interview_coordinator import (
    InterviewCommitOutcomeUnknownError,
    InterviewSession,
)
from tldw_chatbook.Personal_Context.interview_draft_repository import (
    InterviewDraftExpiredError,
)
from tldw_chatbook.Personal_Context.interview_diff import (
    InterviewDiff,
    InterviewDiffChange,
)
from tldw_chatbook.UI.Screens.profile_interview_screen import (
    ProfileInterviewCancelModal,
    ProfileInterviewResult,
    ProfileInterviewScreen,
)


def _session_base() -> InterviewSession:
    return InterviewSession(
        session_id="session-1",
        kind="personal",
        scope_id="scope-global",
        mode="adaptive",
        provider_label="Configured Provider",
        model_id="model-profile",
        external_retention_notice=(
            "External retention is controlled by the selected provider."
        ),
        question_attempts=1,
        question=InterviewQuestion(
            question_id="question-1",
            topic="preferences",
            text="What response style do you prefer?",
        ),
        status="active",
        committed_record_ids=(),
    )


class _Coordinator:
    def __init__(self, session: InterviewSession | None = None) -> None:
        self.session = session or _session_base()
        self.diff = InterviewDiff(
            pack_id="personal-pack",
            audience="personal",
            changes=(),
        )
        self.calls: list[tuple] = []
        self.worker_threads: list[int] = []

    def _record(self, *call):
        self.calls.append(call)
        self.worker_threads.append(threading.get_ident())

    def start(self, *, kind, scope_id, mode):
        self._record("start", kind, scope_id, mode)
        self.session = replace(self.session, kind=kind, scope_id=scope_id, mode=mode)
        return self.session

    def resume(self, session_id):
        self._record("resume", session_id)
        return self.session

    def answer(self, session_id, answer):
        self._record("answer", session_id, answer)
        self.session = replace(
            self.session,
            question_attempts=2,
            question=InterviewQuestion(
                question_id="question-2",
                topic="identity",
                text="How should agents address you?",
            ),
        )
        return self.session

    def skip(self, session_id):
        self._record("skip", session_id)
        self.session = replace(self.session, question=None, status="complete")
        return self.session

    def retry(self, session_id):
        self._record("retry", session_id)
        self.session = replace(self.session, provider_error="provider_unavailable")
        return self.session

    def use_fixed_fallback(self, session_id):
        self._record("fallback", session_id)
        self.session = replace(
            self.session,
            mode="fixed",
            provider_label="Fixed local questionnaire",
            model_id=None,
            external_retention_notice=(
                "No external provider is used by the fixed questionnaire."
            ),
            provider_error=None,
        )
        return self.session

    def discard(self, session_id):
        self._record("discard", session_id)

    def finish_early(self, session_id):
        self._record("finish", session_id)
        return self.diff

    def review(self, session_id):
        self._record("review", session_id)
        return self.diff

    def retry_draft_cleanup(self, session_id):
        self._record("cleanup", session_id)
        return True


class _ExpiredCoordinator(_Coordinator):
    def resume(self, session_id):
        self._record("resume", session_id)
        raise InterviewDraftExpiredError("private draft detail")


class _BlockingStartCoordinator(_Coordinator):
    def __init__(self) -> None:
        super().__init__()
        self.start_entered = threading.Event()
        self.release_start = threading.Event()

    def start(self, *, kind, scope_id, mode):
        self._record("start", kind, scope_id, mode)
        self.start_entered.set()
        assert self.release_start.wait(5)
        return self.session


class _BlockingFailingStartCoordinator(_BlockingStartCoordinator):
    def start(self, *, kind, scope_id, mode):
        self._record("start", kind, scope_id, mode)
        self.start_entered.set()
        assert self.release_start.wait(5)
        raise RuntimeError("private provider failure detail")


class _FailingActionCoordinator(_Coordinator):
    def answer(self, session_id, answer):
        self._record("answer", session_id, answer)
        raise ValueError("secret or validation detail")

    def retry(self, session_id):
        self._record("retry", session_id)
        raise RuntimeError("transient provider detail")

    def review(self, session_id):
        self._record("review", session_id)
        raise ValueError("stale review detail")


class _UnknownCommitCoordinator(_Coordinator):
    def commit(self, session_id, **kwargs):
        self._record("commit", session_id, kwargs)
        raise InterviewCommitOutcomeUnknownError()


class _Host(ConsolidatedCSSApp):
    CSS_PATH = [str(BUNDLED_STYLESHEET)]

    def __init__(self) -> None:
        super().__init__()
        self.results: list[ProfileInterviewResult | None] = []


async def _push(host: _Host, screen: ProfileInterviewScreen) -> None:
    await host.push_screen(screen, callback=host.results.append)
    await host.workers.wait_for_complete()


@pytest.mark.asyncio
async def test_adaptive_disclosure_is_visible_before_answer_becomes_usable() -> None:
    coordinator = _Coordinator()
    host = _Host()
    main_thread = threading.get_ident()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="adaptive",
        )
        await _push(host, screen)
        await pilot.pause()

        assert "Configured Provider / model-profile" in str(
            screen.query_one("#profile-interview-provider", Static).renderable
        )
        assert "retention is controlled" in str(
            screen.query_one("#profile-interview-retention", Static).renderable
        )
        assert screen.query_one("#profile-interview-answer", Input).disabled is False
        assert coordinator.calls[0] == (
            "start",
            "personal",
            "scope-global",
            "adaptive",
        )
        assert coordinator.worker_threads[0] != main_thread


@pytest.mark.asyncio
async def test_memory_only_interview_is_disclosed_and_cannot_be_kept_for_resume() -> (
    None
):
    coordinator = _Coordinator(replace(_session_base(), draft_is_memory_only=True))
    host = _Host()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="fixed",
        )
        await _push(host, screen)
        state = str(
            screen.query_one("#profile-interview-state", Static).renderable
        ).lower()
        assert "memory only" in state
        assert "encrypted" not in state

        await pilot.press("escape")
        await pilot.pause()
        cancel = host.screen
        assert isinstance(cancel, ProfileInterviewCancelModal)
        cancel_copy = " ".join(str(item.renderable) for item in cancel.query(Static))
        assert "encrypted" not in cancel_copy.lower()
        assert "resume" not in cancel_copy.lower()
        assert not cancel.query("#profile-interview-cancel-keep")


@pytest.mark.asyncio
async def test_durable_interview_discloses_encrypted_resumable_draft() -> None:
    coordinator = _Coordinator(replace(_session_base(), draft_is_memory_only=False))
    host = _Host()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="fixed",
        )
        await _push(host, screen)
        assert (
            "encrypted"
            in str(
                screen.query_one("#profile-interview-state", Static).renderable
            ).lower()
        )
        await pilot.press("escape")
        await pilot.pause()
        assert host.screen.query("#profile-interview-cancel-keep")


@pytest.mark.asyncio
async def test_answer_skip_retry_and_fixed_fallback_use_coordinator_workers() -> None:
    coordinator = _Coordinator(
        replace(_session_base(), question=None, provider_error="invalid_question")
    )
    host = _Host()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="adaptive",
        )
        await _push(host, screen)
        await pilot.pause()
        assert "one valid question" in str(
            screen.query_one("#profile-interview-error", Static).renderable
        )
        await pilot.click("#profile-interview-retry")
        await host.workers.wait_for_complete()
        await pilot.click("#profile-interview-fixed-fallback")
        await host.workers.wait_for_complete()
        screen.apply_session(
            replace(coordinator.session, question=_session_base().question)
        )
        screen.query_one("#profile-interview-answer", Input).value = "concise"
        await pilot.pause()
        await pilot.click("#profile-interview-submit")
        await host.workers.wait_for_complete()
        await pilot.click("#profile-interview-skip")
        await host.workers.wait_for_complete()

        assert [call[0] for call in coordinator.calls] == [
            "start",
            "retry",
            "fallback",
            "answer",
            "skip",
        ]


@pytest.mark.asyncio
async def test_secret_answer_failure_restores_answer_controls() -> None:
    coordinator = _FailingActionCoordinator()
    host = _Host()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="adaptive",
        )
        await _push(host, screen)
        screen.query_one("#profile-interview-answer", Input).value = "secret"
        screen.query_one("#profile-interview-submit", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert ("answer", "session-1", "secret") in coordinator.calls
        assert screen.query_one("#profile-interview-submit", Button).disabled is False
        assert screen.query_one("#profile-interview-skip", Button).disabled is False
        error = str(screen.query_one("#profile-interview-error", Static).renderable)
        assert "not accepted" in error.lower()
        assert "secret or validation detail" not in error


@pytest.mark.asyncio
async def test_transient_retry_failure_restores_retry_controls() -> None:
    coordinator = _FailingActionCoordinator(
        replace(_session_base(), question=None, provider_error="provider_unavailable")
    )
    host = _Host()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="adaptive",
        )
        await _push(host, screen)
        screen.query_one("#profile-interview-retry", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert ("retry", "session-1") in coordinator.calls
        assert screen.query_one("#profile-interview-retry", Button).disabled is False
        assert (
            screen.query_one("#profile-interview-fixed-fallback", Button).disabled
            is False
        )
        error = str(screen.query_one("#profile-interview-error", Static).renderable)
        assert "action failed" in error.lower()
        assert "transient provider detail" not in error


@pytest.mark.asyncio
async def test_stale_review_failure_restores_review_control() -> None:
    coordinator = _FailingActionCoordinator(
        replace(_session_base(), question=None, status="review")
    )
    host = _Host()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="adaptive",
        )
        await _push(host, screen)
        screen.query_one("#profile-interview-review", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert ("review", "session-1") in coordinator.calls
        assert screen.query_one("#profile-interview-review", Button).disabled is False
        error = str(screen.query_one("#profile-interview-error", Static).renderable)
        assert "not accepted" in error.lower()
        assert "stale review detail" not in error


@pytest.mark.asyncio
async def test_provider_failure_copy_preserves_answers_and_counts_attempt() -> None:
    coordinator = _Coordinator(
        replace(_session_base(), question=None, provider_error="provider_unavailable")
    )
    host = _Host()

    async with host.run_test(size=(100, 30)):
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="adaptive",
        )
        await _push(host, screen)

        error = str(
            screen.query_one("#profile-interview-error", Static).renderable
        ).lower()
        assert "saved answers remain intact" in error
        assert "counts toward 20" in error


@pytest.mark.asyncio
async def test_escape_offers_keep_draft_and_discard_is_explicit() -> None:
    coordinator = _Coordinator()
    host = _Host()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="fixed",
        )
        await _push(host, screen)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(host.screen, ProfileInterviewCancelModal)
        await pilot.click("#profile-interview-cancel-discard")
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert ("discard", "session-1") in coordinator.calls
        assert host.results == [
            ProfileInterviewResult(
                status="discarded",
                committed_record_ids=(),
                runtime_enabled=None,
            )
        ]


@pytest.mark.asyncio
async def test_cancel_during_initial_start_waits_then_discards_created_draft() -> None:
    coordinator = _BlockingStartCoordinator()
    host = _Host()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="fixed",
        )
        await host.push_screen(screen, callback=host.results.append)
        assert coordinator.start_entered.wait(1)

        await pilot.press("escape")
        await pilot.pause()
        assert host.screen is screen
        assert host.results == []
        state_copy = str(
            screen.query_one("#profile-interview-state", Static).renderable
        )
        assert "cancell" in state_copy.lower()

        coordinator.release_start.set()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert ("discard", "session-1") in coordinator.calls
        assert host.results == [ProfileInterviewResult("discarded", (), None)]


@pytest.mark.asyncio
async def test_cancel_during_failing_initial_start_closes_without_a_draft() -> None:
    coordinator = _BlockingFailingStartCoordinator()
    host = _Host()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="adaptive",
        )
        await host.push_screen(screen, callback=host.results.append)
        assert coordinator.start_entered.wait(1)

        await pilot.press("escape")
        await pilot.pause()
        assert host.screen is screen

        coordinator.release_start.set()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert not any(call[0] == "discard" for call in coordinator.calls)
        assert host.results == [ProfileInterviewResult("cancelled", (), None)]


@pytest.mark.asyncio
async def test_escape_can_keep_the_encrypted_draft_without_discarding() -> None:
    coordinator = _Coordinator()
    host = _Host()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="fixed",
        )
        await _push(host, screen)
        await pilot.press("escape")
        await pilot.pause()
        await pilot.click("#profile-interview-cancel-keep")
        await pilot.pause()

        assert not any(call[0] == "discard" for call in coordinator.calls)
        assert host.results == [ProfileInterviewResult("saved", (), None)]


@pytest.mark.asyncio
async def test_finish_early_runs_in_worker_and_opens_persisted_review() -> None:
    session = replace(
        _session_base(),
        turns=(InterviewTurn(question_id="question-1", answer="concise"),),
        can_ask_another=False,
    )
    coordinator = _Coordinator(session)
    host = _Host()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="fixed",
        )
        await _push(host, screen)
        await pilot.press("f")
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert ("finish", "session-1") in coordinator.calls
        assert host.screen.__class__.__name__ == "PersonalContextReviewModal"


@pytest.mark.asyncio
async def test_live_unknown_commit_returns_unknown_result_through_parent() -> None:
    coordinator = _UnknownCommitCoordinator(
        replace(_session_base(), question=None, status="review")
    )
    coordinator.diff = InterviewDiff(
        pack_id="personal-pack",
        audience="personal",
        changes=(
            InterviewDiffChange(
                change_id="change-1",
                change=InterviewProposedChange(
                    operation=ProposalOperation.CREATE,
                    proposed_payload=PreferencePayload(
                        subject="response.detail",
                        polarity="like",
                        value="concise",
                    ),
                    controls=ProfileControls(
                        sync_mode=SyncMode.SYNCABLE,
                        agent_visibility=AgentVisibility.AGENT_VISIBLE,
                    ),
                    semantic_key=SemanticKey(
                        namespace="preference", subject="response.detail"
                    ),
                ),
            ),
        ),
    )
    host = _Host()

    async with host.run_test(size=(110, 34)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="fixed",
        )
        await _push(host, screen)
        screen.query_one("#profile-interview-review", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        modal = host.screen
        assert modal.__class__.__name__ == "PersonalContextReviewModal"

        modal.query_one("#personal-context-review-save-use", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        status = str(
            modal.query_one("#personal-context-review-status", Static).renderable
        ).lower()
        assert "outcome is unknown" in status
        assert host.results == []

        modal.query_one("#personal-context-review-close", Button).press()
        await pilot.pause()
        assert host.results == [ProfileInterviewResult("commit_unknown", (), None)]


@pytest.mark.asyncio
async def test_committed_resume_only_retries_cleanup_and_returns_committed_ids() -> (
    None
):
    coordinator = _Coordinator(
        replace(
            _session_base(),
            question=None,
            status="committed",
            committed_record_ids=("record-1",),
            runtime_requested=True,
            runtime_update_succeeded=True,
        )
    )
    host = _Host()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="fixed",
            session_id="session-1",
        )
        await _push(host, screen)
        await pilot.pause()

        assert screen.query_one("#profile-interview-answer", Input).disabled
        assert {button.id for button in screen.query(Button) if button.display} == {
            "profile-interview-cancel",
            "profile-interview-retry-cleanup",
        }
        await pilot.click("#profile-interview-retry-cleanup")
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert ("cleanup", "session-1") in coordinator.calls
        assert host.results == [
            ProfileInterviewResult("committed", ("record-1",), True)
        ]


@pytest.mark.asyncio
async def test_leaving_committed_resume_preserves_committed_outcome() -> None:
    coordinator = _Coordinator(
        replace(
            _session_base(),
            question=None,
            status="committed",
            committed_record_ids=("record-1",),
        )
    )
    host = _Host()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="fixed",
            session_id="session-1",
        )
        await _push(host, screen)
        await pilot.press("escape")
        await pilot.pause()

        assert not any(call[0] == "discard" for call in coordinator.calls)
        assert host.results == [
            ProfileInterviewResult("committed", ("record-1",), None)
        ]


@pytest.mark.asyncio
async def test_committed_resume_with_unknown_runtime_outcome_directs_to_settings() -> (
    None
):
    coordinator = _Coordinator(
        replace(
            _session_base(),
            question=None,
            status="committed",
            committed_record_ids=("record-1",),
            runtime_requested=True,
            runtime_update_succeeded=None,
        )
    )
    host = _Host()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="fixed",
            session_id="session-1",
        )
        await _push(host, screen)
        await pilot.pause()

        state_copy = str(
            screen.query_one("#profile-interview-state", Static).renderable
        )
        assert "unknown" in state_copy.lower()
        assert "settings" in state_copy.lower()
        await pilot.press("escape")
        await pilot.pause()
        assert host.results == [
            ProfileInterviewResult("committed", ("record-1",), None)
        ]


@pytest.mark.asyncio
async def test_cleanup_of_unknown_commit_preserves_unknown_result() -> None:
    coordinator = _Coordinator(
        replace(
            _session_base(),
            question=None,
            status="committing",
            committed_record_ids=(),
            runtime_requested=True,
            runtime_update_succeeded=None,
        )
    )
    host = _Host()

    async with host.run_test(size=(100, 30)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="fixed",
            session_id="session-1",
        )
        await _push(host, screen)
        screen.query_one("#profile-interview-retry-cleanup", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()

        assert ("cleanup", "session-1") in coordinator.calls
        assert host.results == [ProfileInterviewResult("commit_unknown", (), None)]


@pytest.mark.asyncio
async def test_expired_resume_is_content_safe_and_cannot_accept_answers() -> None:
    coordinator = _ExpiredCoordinator()
    host = _Host()

    async with host.run_test(size=(80, 24)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="fixed",
            session_id="session-expired",
        )
        await _push(host, screen)
        await pilot.pause()

        text = str(screen.query_one("#profile-interview-state", Static).renderable)
        assert "expired" in text.lower()
        assert "private draft detail" not in text
        assert screen.query_one("#profile-interview-answer", Input).disabled is True
        assert (
            screen.query_one("#profile-interview-retry-cleanup", Button).disabled
            is False
        )


@pytest.mark.asyncio
async def test_interview_has_only_safe_bindings_and_contains_actions_at_80x24() -> None:
    coordinator = _Coordinator()
    host = _Host()

    async with host.run_test(size=(80, 24)) as pilot:
        screen = ProfileInterviewScreen(
            coordinator,
            kind="personal",
            scope_id="scope-global",
            mode="fixed",
        )
        await _push(host, screen)
        await pilot.pause()

        keys = {binding.key for binding in screen.BINDINGS}
        assert keys == {"escape", "f"}
        assert screen.check_action("finish_early", ()) is False
        screen.apply_session(
            replace(
                coordinator.session,
                turns=(InterviewTurn(question_id="question-1", answer="concise"),),
            )
        )
        assert screen.check_action("finish_early", ()) is True
        await pilot.pause()
        content = screen.query_one("#profile-interview-shell")
        assert content.region.width <= 80
        assert content.region.height <= 24
        visible_buttons = [button for button in screen.query(Button) if button.display]
        assert {button.id for button in visible_buttons} == {
            "profile-interview-submit",
            "profile-interview-skip",
            "profile-interview-finish",
            "profile-interview-cancel",
        }
        for button in visible_buttons:
            assert button.region.x >= content.region.x
            assert button.region.right <= content.region.right
