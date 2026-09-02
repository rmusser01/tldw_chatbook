"""Keyboard-first Personal Context interview surface."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Literal, NamedTuple

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from ...Personal_Context.interview_coordinator import InterviewSession
from ...Personal_Context.interview_diff import InterviewDiff
from ...Personal_Context.interview_draft_repository import InterviewDraftExpiredError
from ...Widgets.modal_dismissal import SafeModalDismissMixin


_ERROR_COPY = {
    "invalid_question": (
        "The provider did not return one valid question. Retry or use the fixed "
        "local questionnaire; this attempt still counts toward 20."
    ),
    "provider_unavailable": (
        "The interview provider is unavailable. Retry or use the fixed local "
        "questionnaire. Your saved answers remain intact, and this failed attempt "
        "counts toward 20."
    ),
    "provider_selection_changed": (
        "The pinned provider or model changed. Use the fixed local questionnaire "
        "or reopen with the original selection."
    ),
}


class ProfileInterviewResult(NamedTuple):
    """Terminal result returned to a Settings/setup launch point."""

    status: Literal["committed", "commit_unknown", "saved", "discarded", "cancelled"]
    committed_record_ids: tuple[str, ...]
    runtime_enabled: bool | None


class ProfileInterviewCancelModal(
    SafeModalDismissMixin, ModalScreen[Literal["keep", "discard"] | None]
):
    """Make autosave versus crypto-shredding explicit before closing."""

    SAFE_MODAL_CONTENT = "#profile-interview-cancel-modal"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Continue", show=False)]

    def __init__(self, *, memory_only: bool = False) -> None:
        super().__init__()
        self._memory_only = memory_only

    def compose(self) -> ComposeResult:
        with Vertical(id="profile-interview-cancel-modal"):
            yield Static("Leave interview", classes="profile-interview-title")
            yield Static(
                (
                    "This interview exists only in memory. Discard it or continue; "
                    "it cannot be kept after Chatbook closes."
                    if self._memory_only
                    else "Keep the encrypted draft to resume later, or discard it "
                    "and destroy its draft key."
                ),
                classes="profile-interview-copy",
            )
            with Horizontal(classes="profile-interview-cancel-actions"):
                yield Button(
                    "Continue interview", id="profile-interview-cancel-continue"
                )
                if not self._memory_only:
                    yield Button("Keep draft", id="profile-interview-cancel-keep")
                yield Button(
                    "Discard draft",
                    id="profile-interview-cancel-discard",
                    variant="error",
                )

    @on(Button.Pressed)
    async def handle_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "profile-interview-cancel-continue":
            await self.request_safe_cancel(source="button")
        elif event.button.id == "profile-interview-cancel-keep":
            self.dismiss_safe_once("keep")
        elif event.button.id == "profile-interview-cancel-discard":
            self.dismiss_safe_once("discard")


class ProfileInterviewScreen(
    SafeModalDismissMixin, ModalScreen[ProfileInterviewResult | None]
):
    """Thin Textual client over ``ProfileInterviewCoordinator``."""

    SAFE_MODAL_CONTENT = "#profile-interview-shell"
    BINDINGS = [
        Binding("escape", "request_safe_cancel", "Close"),
        Binding("f", "finish_early", "Finish"),
    ]

    def __init__(
        self,
        coordinator: Any,
        *,
        kind: str,
        scope_id: str,
        mode: str,
        session_id: str | None = None,
    ) -> None:
        super().__init__()
        self._coordinator = coordinator
        self._kind = kind
        self._scope_id = scope_id
        self._mode = mode
        self._session_id = session_id
        self._session: InterviewSession | None = None
        self._expired_or_cleanup_pending = False
        self._starting = session_id is None
        self._cancel_after_start = False
        self._busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="profile-interview-shell"):
            yield Static("Get to know you", classes="profile-interview-title")
            yield Static(
                "Loading pinned interview provider…",
                id="profile-interview-provider",
                classes="profile-interview-disclosure",
            )
            yield Static(
                "Answer entry stays unavailable until provider use is disclosed.",
                id="profile-interview-retention",
                classes="profile-interview-disclosure",
            )
            yield Static(
                "Loading interview…",
                id="profile-interview-state",
                classes="profile-interview-state",
            )
            yield Static(
                "Question 0 of 20",
                id="profile-interview-progress",
                classes="profile-interview-progress",
            )
            with VerticalScroll(id="profile-interview-body"):
                yield Static(
                    "",
                    id="profile-interview-question",
                    classes="profile-interview-question",
                )
                yield Input(
                    placeholder="Your answer",
                    id="profile-interview-answer",
                    disabled=True,
                )
                yield Static(
                    "",
                    id="profile-interview-error",
                    classes="profile-interview-error",
                )
            with Horizontal(classes="profile-interview-actions"):
                yield Button(
                    "Answer",
                    id="profile-interview-submit",
                    variant="primary",
                    disabled=True,
                )
                yield Button("Skip", id="profile-interview-skip", disabled=True)
                yield Button("Retry", id="profile-interview-retry", disabled=True)
                yield Button(
                    "Use fixed questions",
                    id="profile-interview-fixed-fallback",
                    disabled=True,
                )
                yield Button(
                    "Finish early",
                    id="profile-interview-finish",
                    disabled=True,
                )
                yield Button(
                    "Review changes",
                    id="profile-interview-review",
                    disabled=True,
                )
                yield Button("Cancel", id="profile-interview-cancel")
                yield Button(
                    "Retry cleanup",
                    id="profile-interview-retry-cleanup",
                    disabled=True,
                )

    def on_mount(self) -> None:
        super().on_mount()
        self._set_busy(True)
        operation = (
            partial(self._coordinator.resume, self._session_id)
            if self._session_id is not None
            else partial(
                self._coordinator.start,
                kind=self._kind,
                scope_id=self._scope_id,
                mode=self._mode,
            )
        )
        self._run_thread(operation, self._initial_load_succeeded)

    def _initial_load_succeeded(self, session: InterviewSession) -> None:
        self._starting = False
        if not self._cancel_after_start:
            self.apply_session(session)
            return
        self._session = session
        self._session_id = session.session_id
        # Let the start worker fully retire before scheduling the exclusive
        # discard worker; starting it inside the callback cancels its caller.
        self.call_after_refresh(self._discard_cancelled_start)

    def _discard_cancelled_start(self) -> None:
        if self._session_id is None:
            return
        self._set_busy(True)
        self._run_thread(
            partial(self._coordinator.discard, self._session_id),
            lambda _result: self.dismiss_safe_once(
                ProfileInterviewResult("discarded", (), None)
            ),
        )

    def check_action(
        self,
        action: str,
        parameters: tuple[object, ...],
    ) -> bool | None:
        """Advertise Finish only while a draft has answers to review."""

        if action == "finish_early":
            return bool(
                not self._busy
                and self._session is not None
                and self._session.status in {"active", "complete"}
                and self._session.turns
            )
        return super().check_action(action, parameters)

    def _run_thread(
        self,
        operation: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        self.run_worker(
            partial(self._thread_operation, operation, on_success),
            thread=True,
            exclusive=True,
            group="profile-interview-coordinator",
            exit_on_error=False,
        )

    def _thread_operation(
        self,
        operation: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        try:
            result = operation()
        except InterviewDraftExpiredError:
            self.app.call_from_thread(self._apply_expired)
        except (TypeError, ValueError):
            self.app.call_from_thread(
                self._apply_error,
                "That answer or interview action was not accepted. Check the entry and try again.",
            )
        except Exception:
            self.app.call_from_thread(
                self._apply_error,
                "The interview action failed. Your private draft details were not displayed.",
            )
        else:
            self.app.call_from_thread(on_success, result)

    def apply_session(self, session: InterviewSession) -> None:
        """Render one immutable coordinator view on the app thread."""

        self._session = session
        self._session_id = session.session_id
        provider = session.provider_label
        if session.model_id:
            provider = f"{provider} / {session.model_id}"
        self.query_one("#profile-interview-provider", Static).update(provider)
        self.query_one("#profile-interview-retention", Static).update(
            session.external_retention_notice
        )
        self.query_one("#profile-interview-progress", Static).update(
            f"Question {session.question_attempts} of 20"
        )
        question_text = session.question.text if session.question is not None else ""
        self.query_one("#profile-interview-question", Static).update(question_text)
        error = (
            _ERROR_COPY.get(
                session.provider_error,
                "The provider could not produce a usable question. Retry or use fixed questions.",
            )
            if session.provider_error
            else ""
        )
        self.query_one("#profile-interview-error", Static).update(error)
        active_copy = (
            "Interview active — raw answers are held in memory only and are lost "
            "when Chatbook closes."
            if session.draft_is_memory_only
            else "Interview active — raw answers stay in the encrypted local draft."
        )
        state_copy = {
            "active": active_copy,
            "complete": "Questions complete — review the proposed changes before saving.",
            "review": "Final review is ready.",
            "committing": (
                "Save outcome is unknown; verify records and agent use in Settings. "
                "Only draft cleanup may be retried."
            ),
        }.get(session.status, "Interview unavailable.")
        if session.status == "committed":
            state_copy = self._committed_state_copy(session)
        self.query_one("#profile-interview-state", Static).update(state_copy)
        self._set_busy(False)

        has_question = session.status == "active" and session.question is not None
        has_error = session.status == "active" and session.provider_error is not None
        self.query_one("#profile-interview-answer", Input).disabled = not has_question
        self.query_one("#profile-interview-submit", Button).disabled = not has_question
        self.query_one("#profile-interview-skip", Button).disabled = not has_question
        self.query_one("#profile-interview-retry", Button).disabled = not has_error
        self.query_one("#profile-interview-fixed-fallback", Button).disabled = not (
            has_error and session.mode == "adaptive"
        )
        can_finish = session.status in {"active", "complete"} and bool(session.turns)
        self.query_one("#profile-interview-finish", Button).disabled = not can_finish
        self.query_one("#profile-interview-review", Button).disabled = (
            session.status not in {"complete", "review"}
        )
        self.query_one("#profile-interview-retry-cleanup", Button).disabled = (
            session.status not in {"committed", "committing"}
        )
        self._show_actions_for(session)

    def _show_actions_for(self, session: InterviewSession) -> None:
        """Keep only currently meaningful actions in the persistent action bar."""

        visible = {"cancel"}
        if session.status == "active":
            if session.question is not None:
                visible.update({"submit", "skip"})
            if session.provider_error is not None:
                visible.add("retry")
                if session.mode == "adaptive":
                    visible.add("fixed-fallback")
            if session.turns:
                visible.add("finish")
        elif session.status in {"complete", "review"}:
            visible.add("review")
        elif session.status in {"committed", "committing"}:
            visible.add("retry-cleanup")
        for action in (
            "submit",
            "skip",
            "retry",
            "fixed-fallback",
            "finish",
            "review",
            "cancel",
            "retry-cleanup",
        ):
            self.query_one(f"#profile-interview-{action}", Button).display = (
                action in visible
            )

    def _set_busy(self, value: bool) -> None:
        self._busy = value
        if not self.is_mounted:
            return
        if value:
            self.query_one("#profile-interview-answer", Input).disabled = True
            for button_id in (
                "submit",
                "skip",
                "retry",
                "fixed-fallback",
                "finish",
                "review",
                "retry-cleanup",
            ):
                self.query_one(
                    f"#profile-interview-{button_id}", Button
                ).disabled = True

    def _apply_error(self, copy: str) -> None:
        if self._starting:
            self._starting = False
            if self._cancel_after_start:
                self._cancel_after_start = False
                self.dismiss_safe_once(ProfileInterviewResult("cancelled", (), None))
                return
        if self._session is None:
            self._set_busy(False)
        else:
            self.apply_session(self._session)
        self.query_one("#profile-interview-error", Static).update(copy)

    def _apply_expired(self) -> None:
        self._expired_or_cleanup_pending = True
        self._set_busy(False)
        self.query_one("#profile-interview-state", Static).update(
            "Draft expired or secure cleanup is pending. It cannot be resumed as an active interview."
        )
        self.query_one("#profile-interview-error", Static).update(
            "You may retry cleanup; no draft content is shown."
        )
        self.query_one("#profile-interview-retry-cleanup", Button).disabled = False
        for action in (
            "submit",
            "skip",
            "retry",
            "fixed-fallback",
            "finish",
            "review",
        ):
            self.query_one(f"#profile-interview-{action}", Button).display = False
        self.query_one("#profile-interview-cancel", Button).display = True
        self.query_one("#profile-interview-retry-cleanup", Button).display = True

    @on(Button.Pressed)
    async def handle_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith("profile-interview-"):
            return
        event.stop()
        if button_id == "profile-interview-submit":
            self._submit_answer()
        elif button_id == "profile-interview-skip":
            self._session_action(self._coordinator.skip)
        elif button_id == "profile-interview-retry":
            self._session_action(self._coordinator.retry)
        elif button_id == "profile-interview-fixed-fallback":
            self._session_action(self._coordinator.use_fixed_fallback)
        elif button_id in {"profile-interview-finish", "profile-interview-review"}:
            self.show_review()
        elif button_id == "profile-interview-cancel":
            await self.request_safe_cancel(source="button")
        elif button_id == "profile-interview-retry-cleanup":
            self._retry_cleanup()

    def _submit_answer(self) -> None:
        answer = self.query_one("#profile-interview-answer", Input).value.strip()
        if not answer:
            self._apply_error("Enter an answer, or choose Skip.")
            return
        self.query_one("#profile-interview-answer", Input).value = ""
        self._session_action(partial(self._coordinator.answer, answer=answer))

    def _session_action(self, operation: Callable[..., InterviewSession]) -> None:
        if self._session_id is None or self._busy:
            return
        self._set_busy(True)
        self._run_thread(partial(operation, self._session_id), self.apply_session)

    def show_review(self) -> None:
        if self._session_id is None or self._busy:
            return
        if self._session is not None and self._session.status == "review":
            operation = partial(self._coordinator.review, self._session_id)
        else:
            operation = partial(self._coordinator.finish_early, self._session_id)
        self._set_busy(True)
        self._run_thread(operation, self._open_review)

    def action_finish_early(self) -> None:
        self.show_review()

    def _open_review(self, diff: InterviewDiff) -> None:
        self._set_busy(False)
        from ...Widgets.Settings_Widgets.personal_context_review_modal import (
            PersonalContextReviewModal,
        )

        self.app.push_screen(
            PersonalContextReviewModal(
                self._coordinator,
                session_id=self._session_id or "",
                diff=diff,
            ),
            callback=self._review_closed,
        )

    def _review_closed(self, result: Any) -> None:
        if result is None:
            return
        from ...Widgets.Settings_Widgets.personal_context_review_modal import (
            ReviewCommitUnknownResult,
        )

        if isinstance(result, ReviewCommitUnknownResult):
            self.dismiss_safe_once(ProfileInterviewResult("commit_unknown", (), None))
            return
        receipt = result.receipt
        self.dismiss_safe_once(
            ProfileInterviewResult(
                status="committed",
                committed_record_ids=receipt.committed_record_ids,
                runtime_enabled=(
                    result.enable_runtime if receipt.runtime_update_succeeded else None
                ),
            )
        )

    @staticmethod
    def _committed_runtime_enabled(session: InterviewSession) -> bool | None:
        if session.runtime_update_succeeded is not True:
            return None
        return session.runtime_requested

    @classmethod
    def _committed_state_copy(cls, session: InterviewSession) -> str:
        runtime_enabled = cls._committed_runtime_enabled(session)
        if runtime_enabled is True:
            outcome = "Agent use was enabled."
        elif runtime_enabled is False:
            outcome = "Agent use was disabled."
        else:
            outcome = "Runtime outcome is unknown; verify it in Settings."
        return f"Records committed — {outcome} Only draft cleanup may be retried."

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        if self._starting and self._session_id is None:
            self._cancel_after_start = True
            self.query_one("#profile-interview-state", Static).update(
                "Cancellation pending — securely discarding the new draft."
            )
            return
        if self._cancel_after_start:
            if not self._busy:
                self._discard_cancelled_start()
            return
        if self._session_id is None or self._expired_or_cleanup_pending:
            self.dismiss_safe_once(ProfileInterviewResult("cancelled", (), None))
            return
        if self._session is not None and self._session.status == "committed":
            self.dismiss_safe_once(
                ProfileInterviewResult(
                    "committed",
                    self._session.committed_record_ids,
                    self._committed_runtime_enabled(self._session),
                )
            )
            return
        if self._session is not None and self._session.status == "committing":
            self.dismiss_safe_once(ProfileInterviewResult("commit_unknown", (), None))
            return
        # Defer until this screen's one-shot cancellation request has closed;
        # the nested modal then starts with its own dismissal generation.
        self.call_after_refresh(self._open_cancel_confirmation)

    def _open_cancel_confirmation(self) -> None:
        self.app.push_screen(
            ProfileInterviewCancelModal(
                memory_only=bool(
                    self._session is not None and self._session.draft_is_memory_only
                )
            ),
            callback=self._cancel_choice,
        )

    def _cancel_choice(self, choice: str | None) -> None:
        if choice == "keep":
            committed_ids = (
                self._session.committed_record_ids if self._session is not None else ()
            )
            self.dismiss_safe_once(ProfileInterviewResult("saved", committed_ids, None))
        elif choice == "discard" and self._session_id is not None:
            self._set_busy(True)
            self._run_thread(
                partial(self._coordinator.discard, self._session_id),
                lambda _result: self.dismiss_safe_once(
                    ProfileInterviewResult("discarded", (), None)
                ),
            )

    def _retry_cleanup(self) -> None:
        if self._session_id is None or self._busy:
            return
        self._set_busy(True)
        self._run_thread(
            partial(self._coordinator.retry_draft_cleanup, self._session_id),
            self._cleanup_succeeded,
        )

    def _cleanup_succeeded(self, _result: Any) -> None:
        committed_ids = (
            self._session.committed_record_ids if self._session is not None else ()
        )
        if self._session is not None and self._session.status == "committed":
            status = "committed"
        elif self._session is not None and self._session.status == "committing":
            status = "commit_unknown"
        else:
            status = "discarded"
        runtime_enabled = (
            self._committed_runtime_enabled(self._session)
            if self._session is not None and self._session.status == "committed"
            else None
        )
        self.dismiss_safe_once(
            ProfileInterviewResult(status, committed_ids, runtime_enabled)
        )
