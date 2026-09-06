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
from tldw_profile_core import (
    ActorType,
    AgentVisibility,
    ProfilePayload,
    ProfileProposal,
    ProfileRecord,
    ProposalOperation,
    SyncMode,
)

from ...Personal_Context.interview_coordinator import (
    InterviewCommitOutcomeUnknownError,
    InterviewCommitReceipt,
    InterviewReviewRewrite,
)
from ...Personal_Context.interview_diff import InterviewDiff, InterviewDiffChange
from ...Personal_Context.service import (
    ProfileConflictError,
    ProfileKeyCollisionError,
)
from ..modal_dismissal import SafeModalDismissMixin


@dataclass(frozen=True, slots=True)
class ReviewCommitResult:
    """Return the independent runtime request with its honest receipt."""

    receipt: InterviewCommitReceipt
    enable_runtime: bool


@dataclass(frozen=True, slots=True)
class ReviewCommitUnknownResult:
    """Return a terminal commit whose canonical outcome cannot be inferred."""


@dataclass(frozen=True, slots=True)
class ProposalReviewResult:
    """Return only the bounded outcome of a user-owned proposal review."""

    proposal_id: str
    state: str
    record_id: str | None


class PersonalContextProposalReviewModal(
    SafeModalDismissMixin,
    ModalScreen[ProposalReviewResult | None],
):
    """Review and resolve one pending agent proposal without exposing private peers."""

    SAFE_MODAL_CONTENT = "#personal-context-review-modal"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Close")]

    def __init__(
        self,
        proposal_service: Any,
        *,
        proposal: ProfileProposal,
        scope_label: str,
        target_record: ProfileRecord | None = None,
    ) -> None:
        super().__init__()
        self._proposal_service = proposal_service
        self._proposal = proposal
        self._scope_label = scope_label
        self._target_record = self._eligible_target(target_record)
        self._busy = False
        self._outcome_unknown = False
        self._control_state: dict[int, bool] = {}

    def compose(self) -> ComposeResult:
        payload = self._payload()
        target = self._target_record
        with Vertical(id="personal-context-review-modal"):
            yield Static("Review agent proposal", classes="profile-interview-title")
            yield Static(
                (
                    "Agent proposal · "
                    f"{self._scope_label} · {self._proposal.operation.value.title()}"
                ),
                id="personal-context-proposal-source",
                classes="profile-interview-state",
            )
            with VerticalScroll(id="personal-context-review-list"):
                yield Static(
                    "Agents cannot read user-only records. A similar private record may "
                    "exist; this review never reveals it.",
                    id="personal-context-proposal-private-warning",
                    classes="personal-context-review-warning",
                )
                if payload is not None:
                    kind = payload.kind.replace("_", " ").title()
                    yield Static(f"Kind: {kind}", classes="settings-inline-guidance")
                    if payload.kind != "legacy_unclassified":
                        yield Static("Subject", classes="settings-input-label")
                        yield Input(
                            value=str(getattr(payload, "subject", "")),
                            id="personal-context-proposal-subject",
                            classes="personal-context-proposal-field",
                        )
                    yield Static("Value", classes="settings-input-label")
                    yield Input(
                        value=str(self._payload_value(payload)),
                        id="personal-context-proposal-value",
                        classes="personal-context-proposal-field",
                    )
                    if payload.kind == "preference":
                        yield Static("Polarity", classes="settings-input-label")
                        yield Select(
                            (("Like", "like"), ("Dislike", "dislike")),
                            value=getattr(payload, "polarity", "like"),
                            allow_blank=False,
                            id="personal-context-proposal-polarity",
                            classes="personal-context-proposal-field",
                        )
                    assert self._proposal.proposed_record is not None
                    controls = self._proposal.proposed_record.controls
                    yield Static(
                        "Controls stay fixed during review: "
                        f"{controls.sync_mode.value.replace('_', ' ')} · "
                        f"{controls.agent_visibility.value.replace('_', ' ')}.",
                        classes="settings-inline-guidance",
                    )
                else:
                    if target is None:
                        yield Static(
                            "Target unavailable. Reload Settings before accepting; "
                            "you can still reject this proposal.",
                            classes="settings-inline-guidance",
                        )
                    else:
                        target_payload = target.payload
                        kind = target.kind.value.replace("_", " ").title()
                        subject = self._bounded_display(
                            getattr(target_payload, "subject", kind), limit=80
                        )
                        value = (
                            self._bounded_display(self._payload_value(target_payload))
                            if target_payload is not None
                            else ""
                        )
                        yield Static(f"Target kind: {kind}")
                        yield Static(f"Target subject: {subject}")
                        yield Static(f"Target value: {value}")
                        yield Static(
                            "This is the exact agent-visible record version referenced "
                            "by the proposal.",
                            classes="settings-inline-guidance",
                        )
                yield Static(
                    "",
                    id="personal-context-proposal-status",
                    classes="settings-inline-guidance",
                )
            with Horizontal(classes="profile-interview-actions"):
                yield Button(
                    "Accept",
                    id="personal-context-proposal-accept",
                    variant="primary",
                    disabled=payload is None and target is None,
                )
                yield Button(
                    "Accept edited",
                    id="personal-context-proposal-accept-edited",
                    disabled=payload is None,
                )
                yield Button(
                    "Reject",
                    id="personal-context-proposal-reject",
                    variant="error",
                )
                yield Button("Close", id="personal-context-proposal-close")

    def _payload(self) -> ProfilePayload | None:
        record = self._proposal.proposed_record
        return None if record is None else record.payload

    def _eligible_target(
        self, target_record: ProfileRecord | None
    ) -> ProfileRecord | None:
        if self._proposal.operation not in {
            ProposalOperation.ARCHIVE,
            ProposalOperation.PROMOTE,
        }:
            return None
        if target_record is None:
            return None
        if (
            target_record.record_id != self._proposal.target_record_id
            or target_record.version_id != self._proposal.base_version_id
            or target_record.scope_id != self._proposal.scope_id
            or target_record.controls.agent_visibility
            is not AgentVisibility.AGENT_VISIBLE
        ):
            return None
        return target_record

    @staticmethod
    def _payload_value(payload: ProfilePayload) -> str:
        return str(
            getattr(payload, "value", None)
            or getattr(payload, "outcome", None)
            or getattr(payload, "text", None)
            or ""
        )

    @staticmethod
    def _bounded_display(value: object, *, limit: int = 160) -> str:
        text = str(value)
        return text if len(text) <= limit else f"{text[: limit - 1]}…"

    def _edited_payload(self) -> ProfilePayload:
        payload = self._payload()
        if payload is None:
            raise ValueError("proposal_operation_is_not_editable")
        body = payload.model_dump(mode="python")
        if payload.kind == "legacy_unclassified":
            body["text"] = self.query_one(
                "#personal-context-proposal-value", Input
            ).value
        else:
            body["subject"] = self.query_one(
                "#personal-context-proposal-subject", Input
            ).value
            field = "outcome" if payload.kind == "goal" else "value"
            body[field] = self.query_one(
                "#personal-context-proposal-value", Input
            ).value
        if payload.kind == "preference":
            body["polarity"] = str(
                self.query_one("#personal-context-proposal-polarity", Select).value
            )
        return type(payload).model_validate(body)

    @on(Button.Pressed)
    async def handle_proposal_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith("personal-context-proposal-"):
            return
        event.stop()
        if self._busy:
            return
        if button_id == "personal-context-proposal-close":
            await self.request_safe_cancel(source="button")
        elif self._outcome_unknown:
            return
        elif button_id == "personal-context-proposal-reject":
            self._resolve("rejected")
        elif button_id == "personal-context-proposal-accept":
            if self._payload() is None and self._target_record is None:
                self._set_status(
                    "Target unavailable. Reload Settings before accepting."
                )
                return
            self._resolve("accepted")
        elif button_id == "personal-context-proposal-accept-edited":
            try:
                payload = self._edited_payload()
            except (TypeError, ValueError):
                self._set_status("The edited proposal content is invalid.")
                return
            self._resolve("accepted", edited_payload=payload)

    def _resolve(
        self,
        state: str,
        *,
        edited_payload: ProfilePayload | None = None,
    ) -> None:
        self._set_busy(True)
        if state == "rejected":
            operation = partial(
                self._proposal_service.reject,
                self._proposal.proposal_id,
            )
        else:
            operation = partial(
                self._proposal_service.accept,
                self._proposal.proposal_id,
                user_actor=ActorType.USER,
                edited_payload=edited_payload,
            )
        self.run_worker(
            partial(self._resolve_in_thread, state, operation),
            thread=True,
            exclusive=True,
            group="personal-context-proposal-review",
            exit_on_error=False,
        )

    def _resolve_in_thread(self, state: str, operation: Callable[[], Any]) -> None:
        try:
            result = operation()
        except (ProfileConflictError, ProfileKeyCollisionError):
            self.app.call_from_thread(
                self._known_failure,
                "Profile context changed. Close this review and reload proposals.",
            )
        except ValueError as exc:
            reason = str(exc)
            copy = (
                "This proposal expired. Close this review and reload proposals."
                if reason == "proposal_expired"
                else "This proposal is no longer available. Reload proposals."
            )
            self.app.call_from_thread(self._known_failure, copy)
        except Exception:
            self.app.call_from_thread(self._unknown_failure)
        else:
            record_id = result.record_id if isinstance(result, ProfileRecord) else None
            self.app.call_from_thread(self._resolved, state, record_id)

    def _known_failure(self, copy: str) -> None:
        self._set_busy(False)
        self._set_status(copy)

    def _unknown_failure(self) -> None:
        self._outcome_unknown = True
        self._set_busy(False)
        for control in (*self.query(Input), *self.query(Select)):
            control.disabled = True
        for button in self.query(Button):
            button.disabled = button.id != "personal-context-proposal-close"
        self._set_status(
            "The outcome could not be confirmed. Close and reload Settings before "
            "taking another action."
        )

    def _resolved(self, state: str, record_id: str | None) -> None:
        self._set_busy(False)
        self.dismiss_safe_once(
            ProposalReviewResult(
                proposal_id=self._proposal.proposal_id,
                state=state,
                record_id=record_id,
            )
        )

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        if not self._busy:
            self.dismiss_safe_once(None)

    def _set_status(self, copy: str) -> None:
        if self.is_mounted:
            self.query_one("#personal-context-proposal-status", Static).update(copy)

    def _set_busy(self, value: bool) -> None:
        self._busy = value
        if not self.is_mounted:
            return
        controls = (*self.query(Input), *self.query(Select), *self.query(Button))
        if value:
            self._control_state = {
                id(control): control.disabled for control in controls
            }
            for control in controls:
                control.disabled = True
            return
        for control in controls:
            control.disabled = self._control_state.get(id(control), control.disabled)
        self._control_state.clear()


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

    # No on_mount override: it did nothing but super().on_mount(), which the
    # dispatcher already invokes on SafeModalDismissMixin separately for
    # this Mount event (TASK-31822).

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
