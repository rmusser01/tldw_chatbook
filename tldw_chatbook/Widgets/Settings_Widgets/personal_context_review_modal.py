"""Structured final review for Personal Context interview changes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Select, Static
from tldw_profile_core import AgentVisibility, ProposalOperation, SyncMode

from ...Personal_Context.interview_coordinator import (
    InterviewCommitOutcomeUnknownError,
    InterviewCommitReceipt,
    InterviewReviewRewrite,
)
from ...Personal_Context.interview_diff import InterviewDiff, InterviewDiffChange
from ..modal_dismissal import SafeModalDismissMixin


@dataclass(frozen=True, slots=True)
class ReviewCommitResult:
    """Return the independent runtime request with its honest receipt."""

    receipt: InterviewCommitReceipt
    enable_runtime: bool


@dataclass(frozen=True, slots=True)
class ReviewCommitUnknownResult:
    """Return a terminal commit whose canonical outcome cannot be inferred."""


class PersonalContextReviewModal(
    SafeModalDismissMixin,
    ModalScreen[ReviewCommitResult | ReviewCommitUnknownResult | None],
):
    """Display, edit, select, and atomically commit one persisted review."""

    SAFE_MODAL_CONTENT = "#personal-context-review-modal"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Close")]
    _revision = reactive(0, recompose=True)

    def __init__(
        self,
        coordinator: Any,
        *,
        session_id: str,
        diff: InterviewDiff,
    ) -> None:
        super().__init__()
        self._coordinator = coordinator
        self._session_id = session_id
        self._diff = diff
        self._selected_ids = {item.change_id for item in diff.changes}
        self._receipt: InterviewCommitReceipt | None = None
        self._commit_unknown = False
        self._enable_runtime = False
        self._busy = False
        self._busy_control_state: dict[int, bool] = {}
        self._status_copy = (
            "Review each proposed change. Only checked rows will be saved."
        )

    @property
    def selected_change_ids(self) -> tuple[str, ...]:
        order = [item.change_id for item in self._diff.changes]
        return tuple(
            change_id for change_id in order if change_id in self._selected_ids
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="personal-context-review-modal"):
            yield Static("Review Personal Context", classes="profile-interview-title")
            yield Static(
                self._status_copy,
                id="personal-context-review-status",
                classes="profile-interview-state",
            )
            with VerticalScroll(id="personal-context-review-list"):
                if not self._diff.changes:
                    yield Static(
                        "No changes were proposed.",
                        classes="profile-interview-copy",
                    )
                for index, item in enumerate(self._diff.changes):
                    yield from self._compose_row(index, item)
            with Horizontal(classes="profile-interview-actions"):
                yield Button(
                    "Save only",
                    id="personal-context-review-save-only",
                    disabled=(
                        self._receipt is not None
                        or self._commit_unknown
                        or not self._diff.changes
                    ),
                )
                yield Button(
                    "Save and use with agents",
                    id="personal-context-review-save-use",
                    variant="primary",
                    disabled=(
                        self._receipt is not None
                        or self._commit_unknown
                        or not self._diff.changes
                    ),
                )
                yield Button(
                    "Retry cleanup",
                    id="personal-context-review-retry-cleanup",
                    disabled=not (
                        self._receipt is not None
                        and self._receipt.draft_cleanup_retry_required
                    )
                    and not self._commit_unknown,
                )
                yield Button("Close", id="personal-context-review-close")

    def _compose_row(self, index: int, item: InterviewDiffChange) -> ComposeResult:
        change = item.change
        payload = change.proposed_payload
        kind = payload.kind if payload is not None else "record"
        subject = getattr(payload, "subject", "") if payload is not None else ""
        value = (
            getattr(payload, "value", None)
            or getattr(payload, "outcome", None)
            or getattr(payload, "text", None)
            or ""
        )
        editable = payload is not None and change.operation in {
            ProposalOperation.CREATE,
            ProposalOperation.UPDATE,
        }
        with Vertical(classes="personal-context-review-row"):
            yield Checkbox(
                (
                    f"{change.operation.value.title()} · "
                    f"{kind.replace('_', ' ').title()} · {subject or 'record'}"
                ),
                value=item.change_id in self._selected_ids,
                id=f"personal-context-review-select-{index}",
            )
            if item.possible_private_duplicate:
                yield Static(
                    "Possible private duplicate — review without exposing the private record.",
                    id=f"personal-context-review-warning-{index}",
                    classes="personal-context-review-warning",
                )
            if editable:
                if kind != "legacy_unclassified":
                    yield Static("Subject", classes="settings-input-label")
                    yield Input(
                        value=str(subject),
                        id=f"personal-context-review-subject-{index}",
                    )
                yield Static("Value", classes="settings-input-label")
                yield Input(
                    value=str(value),
                    id=f"personal-context-review-value-{index}",
                )
                yield Static("Polarity", classes="settings-input-label")
                yield Select(
                    (("Like", "like"), ("Dislike", "dislike")),
                    value=getattr(payload, "polarity", "like"),
                    allow_blank=False,
                    disabled=kind != "preference",
                    id=f"personal-context-review-polarity-{index}",
                )
                yield Static("Syncability", classes="settings-input-label")
                yield Select(
                    (
                        ("Syncable", SyncMode.SYNCABLE.value),
                        ("Device only", SyncMode.DEVICE_ONLY.value),
                    ),
                    value=change.controls.sync_mode.value,
                    allow_blank=False,
                    id=f"personal-context-review-sync-{index}",
                )
                yield Static(
                    "An authorized home server can read syncable content. Device-only stays local.",
                    classes="settings-inline-guidance",
                )
                yield Static("Visibility", classes="settings-input-label")
                yield Select(
                    (
                        (
                            "Agent visible",
                            AgentVisibility.AGENT_VISIBLE.value,
                        ),
                        ("User only", AgentVisibility.USER_ONLY.value),
                    ),
                    value=change.controls.agent_visibility.value,
                    allow_blank=False,
                    id=f"personal-context-review-visibility-{index}",
                )
                yield Button(
                    "Apply edit",
                    id=f"personal-context-review-apply-{index}",
                    disabled=self._receipt is not None,
                )

    def on_mount(self) -> None:
        super().on_mount()

    @on(Checkbox.Changed)
    def handle_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if self._busy or self._receipt is not None:
            return
        checkbox_id = event.checkbox.id or ""
        if not checkbox_id.startswith("personal-context-review-select-"):
            return
        try:
            index = int(checkbox_id.rsplit("-", 1)[1])
            change_id = self._diff.changes[index].change_id
        except (IndexError, ValueError):
            return
        if event.value:
            self._selected_ids.add(change_id)
        else:
            self._selected_ids.discard(change_id)

    @on(Button.Pressed)
    async def handle_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith("personal-context-review-"):
            return
        event.stop()
        if self._commit_unknown and button_id not in {
            "personal-context-review-retry-cleanup",
            "personal-context-review-close",
        }:
            return
        if button_id.startswith("personal-context-review-apply-"):
            try:
                index = int(button_id.rsplit("-", 1)[1])
            except ValueError:
                return
            self._apply_edit(index)
        elif button_id == "personal-context-review-save-only":
            self._commit(enable_runtime=False)
        elif button_id == "personal-context-review-save-use":
            self._commit(enable_runtime=True)
        elif button_id == "personal-context-review-retry-cleanup":
            self._retry_cleanup()
        elif button_id == "personal-context-review-close":
            if self._busy:
                return
            if self._commit_unknown:
                self.dismiss_safe_once(ReviewCommitUnknownResult())
            elif self._receipt is None:
                await self.request_safe_cancel(source="button")
            else:
                self.dismiss_safe_once(
                    ReviewCommitResult(self._receipt, self._enable_runtime)
                )

    def _apply_edit(self, index: int) -> None:
        if self._busy or self._receipt is not None or self._commit_unknown:
            return
        try:
            item = self._diff.changes[index]
        except IndexError:
            return
        payload = item.change.proposed_payload
        controls = item.change.controls
        if payload is None or controls is None:
            return
        body = payload.model_dump(mode="json")
        if payload.kind == "legacy_unclassified":
            body["text"] = self.query_one(
                f"#personal-context-review-value-{index}", Input
            ).value
        else:
            body["subject"] = self.query_one(
                f"#personal-context-review-subject-{index}", Input
            ).value
            field = "outcome" if payload.kind == "goal" else "value"
            body[field] = self.query_one(
                f"#personal-context-review-value-{index}", Input
            ).value
        if payload.kind == "preference":
            body["polarity"] = str(
                self.query_one(
                    f"#personal-context-review-polarity-{index}", Select
                ).value
            )
        controls_body = {
            "sync_mode": str(
                self.query_one(f"#personal-context-review-sync-{index}", Select).value
            ),
            "agent_visibility": str(
                self.query_one(
                    f"#personal-context-review-visibility-{index}", Select
                ).value
            ),
        }
        was_selected = item.change_id in self._selected_ids
        self._set_busy(True)
        self._run_thread(
            partial(
                self._coordinator.rewrite_review_change,
                self._session_id,
                change_id=item.change_id,
                proposed_payload=body,
                controls=controls_body,
            ),
            partial(self._apply_rewrite, item.change_id, was_selected),
        )

    def _apply_rewrite(
        self,
        old_change_id: str,
        was_selected: bool,
        result: InterviewReviewRewrite,
    ) -> None:
        self._diff = result.diff
        self._selected_ids.discard(old_change_id)
        if was_selected:
            self._selected_ids.add(result.change_id)
        self._set_busy(False)
        self._revision += 1

    def _commit(self, *, enable_runtime: bool) -> None:
        if self._busy or self._receipt is not None or self._commit_unknown:
            return
        selections = self.selected_change_ids
        if not selections:
            self._set_status("Select at least one change before saving.")
            return
        self._enable_runtime = enable_runtime
        self._set_busy(True)
        self._run_thread(
            partial(
                self._coordinator.commit,
                self._session_id,
                selections=selections,
                enable_runtime=enable_runtime,
            ),
            self._commit_finished,
        )

    def _commit_finished(self, receipt: InterviewCommitReceipt) -> None:
        self._receipt = receipt
        self._set_busy(False)
        if receipt.runtime_update_succeeded and receipt.draft_cleanup_succeeded:
            self.dismiss_safe_once(ReviewCommitResult(receipt, self._enable_runtime))
            return
        parts = ["Selected records were saved."]
        if not receipt.runtime_update_succeeded:
            parts.append(
                "The requested runtime change is unconfirmed; verify it in Settings."
            )
        if not receipt.draft_cleanup_succeeded:
            parts.append("Draft cleanup is still pending and can be retried safely.")
        self._status_copy = " ".join(parts)
        self._revision += 1

    def _retry_cleanup(self) -> None:
        if self._busy or (self._receipt is None and not self._commit_unknown):
            return
        self._set_busy(True)
        self._run_thread(
            partial(self._coordinator.retry_draft_cleanup, self._session_id),
            self._cleanup_finished,
        )

    def _cleanup_finished(self, _result: Any) -> None:
        if self._commit_unknown:
            self._set_busy(False)
            self.dismiss_safe_once(ReviewCommitUnknownResult())
            return
        assert self._receipt is not None
        self._receipt = replace(
            self._receipt,
            draft_cleanup_succeeded=True,
            draft_cleanup_retry_required=False,
        )
        self._set_busy(False)
        if self._receipt.runtime_update_succeeded:
            self.dismiss_safe_once(
                ReviewCommitResult(self._receipt, self._enable_runtime)
            )
            return
        self._status_copy = (
            "Selected records were saved and draft cleanup finished. "
            "The requested runtime change is unconfirmed; verify it in Settings."
        )
        self._revision += 1

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        if self._busy:
            return
        if self._commit_unknown:
            self.dismiss_safe_once(ReviewCommitUnknownResult())
            return
        result = (
            None
            if self._receipt is None
            else ReviewCommitResult(self._receipt, self._enable_runtime)
        )
        self.dismiss_safe_once(result)

    def _run_thread(
        self,
        operation: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        self.run_worker(
            partial(self._thread_operation, operation, on_success),
            thread=True,
            exclusive=True,
            group="personal-context-review-coordinator",
            exit_on_error=False,
        )

    def _thread_operation(
        self,
        operation: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        try:
            result = operation()
        except InterviewCommitOutcomeUnknownError:
            self.app.call_from_thread(self._commit_outcome_unknown)
        except (TypeError, ValueError):
            copy = "The review edit is invalid or stale. Reload the interview review."
            self.app.call_from_thread(self._operation_failed, copy)
        except Exception:
            copy = "The review action failed. Private details were not displayed."
            self.app.call_from_thread(self._operation_failed, copy)
        else:
            self.app.call_from_thread(on_success, result)

    def _operation_failed(self, copy: str) -> None:
        self._set_busy(False)
        self._set_status(copy)

    def _commit_outcome_unknown(self) -> None:
        self._commit_unknown = True
        self._set_busy(False)
        for control in (
            *self.query(Checkbox),
            *self.query(Input),
            *self.query(Select),
        ):
            control.disabled = True
        for button in self.query(Button):
            safe = button.id in {
                "personal-context-review-retry-cleanup",
                "personal-context-review-close",
            }
            button.disabled = not safe
            button.display = safe
        self._set_status(
            "Commit outcome is unknown. Verify records and runtime in Settings; "
            "only draft cleanup may be retried safely."
        )

    def _set_status(self, copy: str) -> None:
        self._status_copy = copy
        if self.is_mounted:
            self.query_one("#personal-context-review-status", Static).update(copy)

    def _set_busy(self, value: bool) -> None:
        self._busy = value
        if not self.is_mounted:
            return
        controls = (
            *self.query(Checkbox),
            *self.query(Input),
            *self.query(Select),
            *self.query(Button),
        )
        if value:
            self._busy_control_state = {
                id(control): control.disabled for control in controls
            }
            for control in controls:
                control.disabled = True
            return
        for control in controls:
            control.disabled = self._busy_control_state.get(
                id(control), control.disabled
            )
        self._busy_control_state.clear()
        if self._receipt is None:
            return
        for control in (
            *self.query(Checkbox),
            *self.query(Input),
            *self.query(Select),
        ):
            control.disabled = True
        for button in self.query(Button):
            button.disabled = button.id not in {
                "personal-context-review-close",
                (
                    "personal-context-review-retry-cleanup"
                    if self._receipt.draft_cleanup_retry_required
                    else ""
                ),
            }
