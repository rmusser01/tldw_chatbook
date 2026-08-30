from __future__ import annotations

from dataclasses import replace
import threading

import pytest
from textual.widgets import Button, Checkbox, Input, Select, Static
from tldw_profile_core import (
    AgentVisibility,
    InterviewProposedChange,
    PreferencePayload,
    ProfileControls,
    ProposalOperation,
    SemanticKey,
    SyncMode,
)

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Personal_Context.interview_coordinator import (
    InterviewCommitReceipt,
    InterviewCommitOutcomeUnknownError,
    InterviewReviewRewrite,
)
from tldw_chatbook.Personal_Context.interview_diff import (
    InterviewDiff,
    InterviewDiffChange,
)
from tldw_chatbook.Widgets.Settings_Widgets.personal_context_review_modal import (
    PersonalContextReviewModal,
    ReviewCommitResult,
    ReviewCommitUnknownResult,
)


def _change(
    change_id: str,
    *,
    subject: str,
    value: str,
    private_duplicate: bool = False,
) -> InterviewDiffChange:
    return InterviewDiffChange(
        change_id=change_id,
        change=InterviewProposedChange(
            operation=ProposalOperation.CREATE,
            proposed_payload=PreferencePayload(
                subject=subject,
                polarity="like",
                value=value,
            ),
            controls=ProfileControls(
                sync_mode=SyncMode.SYNCABLE,
                agent_visibility=AgentVisibility.AGENT_VISIBLE,
            ),
            semantic_key=SemanticKey(namespace="preference", subject=subject),
        ),
        possible_private_duplicate=private_duplicate,
    )


def _diff() -> InterviewDiff:
    return InterviewDiff(
        pack_id="personal-pack",
        audience="personal",
        changes=(
            _change(
                "change-1",
                subject="response.detail",
                value="concise",
                private_duplicate=True,
            ),
            _change(
                "change-2",
                subject="response.tone",
                value="direct",
            ),
        ),
    )


class _Coordinator:
    def __init__(self) -> None:
        self.diff = _diff()
        self.rewrites: list[dict] = []
        self.commits: list[dict] = []
        self.cleanup_calls: list[str] = []
        self.worker_threads: list[int] = []
        self.receipt = InterviewCommitReceipt(
            committed_record_ids=("record-1",),
            runtime_update_succeeded=True,
            draft_cleanup_succeeded=True,
            draft_cleanup_retry_required=False,
        )

    def rewrite_review_change(self, session_id, **kwargs):
        self.worker_threads.append(threading.get_ident())
        self.rewrites.append({"session_id": session_id, **kwargs})
        old_id = kwargs["change_id"]
        selected = next(item for item in self.diff.changes if item.change_id == old_id)
        payload = PreferencePayload.model_validate(kwargs["proposed_payload"])
        controls = ProfileControls.model_validate(kwargs["controls"])
        rewritten_change = selected.change.model_copy(
            update={
                "proposed_payload": payload,
                "controls": controls,
                "semantic_key": SemanticKey(
                    namespace=payload.kind,
                    subject=payload.subject,
                ),
            }
        )
        replacement = replace(
            selected,
            change_id="change-rewritten",
            change=rewritten_change,
        )
        self.diff = replace(
            self.diff,
            changes=tuple(
                replacement if item.change_id == old_id else item
                for item in self.diff.changes
            ),
        )
        return InterviewReviewRewrite(
            diff=self.diff,
            change_id=replacement.change_id,
        )

    def commit(self, session_id, **kwargs):
        self.worker_threads.append(threading.get_ident())
        self.commits.append({"session_id": session_id, **kwargs})
        return self.receipt

    def retry_draft_cleanup(self, session_id):
        self.worker_threads.append(threading.get_ident())
        self.cleanup_calls.append(session_id)
        return True


class _BlockingCoordinator(_Coordinator):
    def __init__(self) -> None:
        super().__init__()
        self.commit_entered = threading.Event()
        self.release_commit = threading.Event()
        self.rewrite_entered = threading.Event()
        self.release_rewrite = threading.Event()

    def commit(self, session_id, **kwargs):
        self.worker_threads.append(threading.get_ident())
        self.commits.append({"session_id": session_id, **kwargs})
        self.commit_entered.set()
        assert self.release_commit.wait(5)
        return self.receipt

    def rewrite_review_change(self, session_id, **kwargs):
        self.rewrite_entered.set()
        assert self.release_rewrite.wait(5)
        return super().rewrite_review_change(session_id, **kwargs)


class _KnownCommitFailingCoordinator(_Coordinator):
    def commit(self, session_id, **kwargs):
        self.commits.append({"session_id": session_id, **kwargs})
        raise RuntimeError("PRIVATE_SERVICE_CANARY")


class _UnknownCommitCoordinator(_Coordinator):
    def commit(self, session_id, **kwargs):
        self.commits.append({"session_id": session_id, **kwargs})
        raise InterviewCommitOutcomeUnknownError()


class _Host(ConsolidatedCSSApp):
    CSS_PATH = [str(BUNDLED_STYLESHEET)]

    def __init__(self) -> None:
        super().__init__()
        self.results: list[ReviewCommitResult | ReviewCommitUnknownResult | None] = []


async def _push(host: _Host, modal: PersonalContextReviewModal) -> None:
    await host.push_screen(modal, callback=host.results.append)
    await host.workers.wait_for_complete()


@pytest.mark.asyncio
async def test_review_shows_every_change_privacy_controls_and_private_warning() -> None:
    coordinator = _Coordinator()
    host = _Host()

    async with host.run_test(size=(110, 34)) as pilot:
        modal = PersonalContextReviewModal(
            coordinator,
            session_id="session-1",
            diff=coordinator.diff,
        )
        await _push(host, modal)
        await pilot.pause()

        assert len(modal.query(".personal-context-review-row")) == 2
        assert len(modal.query(Checkbox)) == 2
        assert all(box.value for box in modal.query(Checkbox))
        assert (
            "possible private duplicate"
            in str(
                modal.query_one("#personal-context-review-warning-0", Static).renderable
            ).lower()
        )
        assert modal.query_one("#personal-context-review-sync-0", Select).value == (
            SyncMode.SYNCABLE.value
        )
        assert (
            modal.query_one("#personal-context-review-visibility-0", Select).value
            == AgentVisibility.AGENT_VISIBLE.value
        )


@pytest.mark.asyncio
async def test_review_edit_is_coordinator_validated_and_refreshes_selected_id() -> None:
    coordinator = _Coordinator()
    host = _Host()

    async with host.run_test(size=(110, 34)) as pilot:
        modal = PersonalContextReviewModal(
            coordinator,
            session_id="session-1",
            diff=coordinator.diff,
        )
        await _push(host, modal)
        modal.query_one(
            "#personal-context-review-subject-0", Input
        ).value = "response.length"
        modal.query_one("#personal-context-review-value-0", Input).value = "short"
        modal.query_one("#personal-context-review-polarity-0", Select).value = "dislike"
        modal.query_one(
            "#personal-context-review-sync-0", Select
        ).value = SyncMode.DEVICE_ONLY.value
        modal.query_one(
            "#personal-context-review-visibility-0", Select
        ).value = AgentVisibility.USER_ONLY.value
        modal.query_one("#personal-context-review-list").scroll_to_widget(
            modal.query_one("#personal-context-review-apply-0"), animate=False
        )
        await pilot.pause()
        await pilot.click("#personal-context-review-apply-0")
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert coordinator.rewrites[0]["change_id"] == "change-1"
        assert "change-rewritten" in modal.selected_change_ids
        assert "change-1" not in modal.selected_change_ids
        assert (
            modal.query_one("#personal-context-review-select-0", Checkbox).value is True
        )


@pytest.mark.asyncio
async def test_selection_is_frozen_while_review_rewrite_is_running() -> None:
    coordinator = _BlockingCoordinator()
    host = _Host()

    async with host.run_test(size=(110, 34)) as pilot:
        modal = PersonalContextReviewModal(
            coordinator,
            session_id="session-1",
            diff=coordinator.diff,
        )
        await _push(host, modal)
        apply_button = modal.query_one("#personal-context-review-apply-0", Button)
        modal.query_one("#personal-context-review-list").scroll_to_widget(
            apply_button, animate=False
        )
        await pilot.pause()
        await pilot.click(apply_button)
        assert coordinator.rewrite_entered.wait(1)

        checkbox = modal.query_one("#personal-context-review-select-0", Checkbox)
        assert checkbox.disabled is True
        checkbox.value = False
        await pilot.pause()
        assert modal.selected_change_ids == ("change-1", "change-2")

        coordinator.release_rewrite.set()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert modal.selected_change_ids == ("change-rewritten", "change-2")
        assert (
            modal.query_one("#personal-context-review-select-0", Checkbox).value is True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("button_id", "enable_runtime"),
    [
        ("personal-context-review-save-only", False),
        ("personal-context-review-save-use", True),
    ],
)
async def test_commit_uses_only_checked_rows_and_distinguishes_runtime_choice(
    button_id: str,
    enable_runtime: bool,
) -> None:
    coordinator = _Coordinator()
    host = _Host()
    main_thread = threading.get_ident()

    async with host.run_test(size=(110, 34)) as pilot:
        modal = PersonalContextReviewModal(
            coordinator,
            session_id="session-1",
            diff=coordinator.diff,
        )
        await _push(host, modal)
        modal.query_one("#personal-context-review-select-1", Checkbox).value = False
        await pilot.click(f"#{button_id}")
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert coordinator.commits == [
            {
                "session_id": "session-1",
                "selections": ("change-1",),
                "enable_runtime": enable_runtime,
            }
        ]
        assert coordinator.worker_threads[-1] != main_thread
        assert host.results == [
            ReviewCommitResult(
                receipt=coordinator.receipt,
                enable_runtime=enable_runtime,
            )
        ]


@pytest.mark.asyncio
async def test_partial_commit_reports_runtime_and_cleanup_independently_without_retrying_commit() -> (
    None
):
    coordinator = _Coordinator()
    coordinator.receipt = InterviewCommitReceipt(
        committed_record_ids=("record-1",),
        runtime_update_succeeded=False,
        draft_cleanup_succeeded=False,
        draft_cleanup_retry_required=True,
    )
    host = _Host()

    async with host.run_test(size=(110, 34)) as pilot:
        modal = PersonalContextReviewModal(
            coordinator,
            session_id="session-1",
            diff=coordinator.diff,
        )
        await _push(host, modal)
        await pilot.click("#personal-context-review-save-use")
        await host.workers.wait_for_complete()
        await pilot.pause()

        status = str(
            modal.query_one("#personal-context-review-status", Static).renderable
        )
        assert "saved" in status.lower()
        assert "runtime change is unconfirmed" in status.lower()
        assert "verify" in status.lower()
        assert "enabled" not in status.lower()
        assert "disabled" not in status.lower()
        assert "cleanup" in status.lower()
        assert modal.query_one("#personal-context-review-save-use", Button).disabled
        assert len(coordinator.commits) == 1

        await pilot.click("#personal-context-review-retry-cleanup")
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert coordinator.cleanup_calls == ["session-1"]
        assert len(coordinator.commits) == 1
        status = str(
            modal.query_one("#personal-context-review-status", Static).renderable
        )
        assert "runtime change is unconfirmed" in status.lower()
        assert "enabled" not in status.lower()
        assert "disabled" not in status.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "button_id",
    [
        "personal-context-review-save-only",
        "personal-context-review-save-use",
    ],
)
async def test_runtime_failure_copy_is_neutral_for_both_commit_modes(
    button_id: str,
) -> None:
    coordinator = _Coordinator()
    coordinator.receipt = InterviewCommitReceipt(
        committed_record_ids=("record-1",),
        runtime_update_succeeded=False,
        draft_cleanup_succeeded=False,
        draft_cleanup_retry_required=True,
    )
    host = _Host()

    async with host.run_test(size=(110, 34)) as pilot:
        modal = PersonalContextReviewModal(
            coordinator,
            session_id="session-1",
            diff=coordinator.diff,
        )
        await _push(host, modal)
        await pilot.click(f"#{button_id}")
        await host.workers.wait_for_complete()
        await pilot.pause()

        status = str(
            modal.query_one("#personal-context-review-status", Static).renderable
        ).lower()
        assert "requested runtime change is unconfirmed" in status
        assert "verify" in status
        assert "enabled" not in status
        assert "disabled" not in status


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_route", ["escape", "backdrop", "close"])
async def test_irreversible_commit_cannot_be_dismissed_while_busy(
    cancel_route: str,
) -> None:
    coordinator = _BlockingCoordinator()
    host = _Host()

    async with host.run_test(size=(110, 34)) as pilot:
        modal = PersonalContextReviewModal(
            coordinator,
            session_id="session-1",
            diff=coordinator.diff,
        )
        await _push(host, modal)
        await pilot.click("#personal-context-review-save-use")
        assert coordinator.commit_entered.wait(1)

        assert all(widget.disabled for widget in modal.query(Checkbox))
        assert modal.query_one("#personal-context-review-close", Button).disabled
        if cancel_route == "escape":
            await pilot.press("escape")
        elif cancel_route == "backdrop":
            await modal.request_safe_cancel(source="backdrop")
        else:
            await pilot.click("#personal-context-review-close")
        await pilot.pause()
        assert host.screen is modal
        assert host.results == []

        coordinator.release_commit.set()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert host.results == [
            ReviewCommitResult(coordinator.receipt, enable_runtime=True)
        ]


@pytest.mark.asyncio
async def test_known_commit_failure_restores_retryable_review() -> None:
    coordinator = _KnownCommitFailingCoordinator()
    host = _Host()

    async with host.run_test(size=(110, 34)) as pilot:
        modal = PersonalContextReviewModal(
            coordinator,
            session_id="session-1",
            diff=coordinator.diff,
        )
        await _push(host, modal)
        modal.query_one("#personal-context-review-save-use", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()

        status = str(
            modal.query_one("#personal-context-review-status", Static).renderable
        )
        assert "action failed" in status.lower()
        assert "PRIVATE_SERVICE_CANARY" not in status
        assert (
            modal.query_one("#personal-context-review-save-use", Button).disabled
            is False
        )
        assert all(checkbox.disabled is False for checkbox in modal.query(Checkbox))
        assert host.screen is modal


@pytest.mark.asyncio
async def test_unknown_commit_failure_becomes_terminal_live_state() -> None:
    coordinator = _UnknownCommitCoordinator()
    host = _Host()

    async with host.run_test(size=(110, 34)) as pilot:
        modal = PersonalContextReviewModal(
            coordinator,
            session_id="session-1",
            diff=coordinator.diff,
        )
        await _push(host, modal)
        modal.query_one("#personal-context-review-save-use", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()

        status = str(
            modal.query_one("#personal-context-review-status", Static).renderable
        ).lower()
        assert "outcome is unknown" in status
        assert "verify records and runtime in settings" in status
        assert "cleanup" in status
        assert all(control.disabled for control in modal.query(Checkbox))
        assert all(control.disabled for control in modal.query(Input))
        assert all(control.disabled for control in modal.query(Select))
        assert (
            modal.query_one("#personal-context-review-save-only", Button).display
            is False
        )
        assert (
            modal.query_one("#personal-context-review-save-use", Button).display
            is False
        )
        assert (
            modal.query_one("#personal-context-review-retry-cleanup", Button).disabled
            is False
        )
        assert (
            modal.query_one("#personal-context-review-close", Button).disabled is False
        )

        modal.query_one("#personal-context-review-close", Button).press()
        await pilot.pause()
        assert host.results == [ReviewCommitUnknownResult()]


@pytest.mark.asyncio
async def test_review_modal_contains_actions_at_80x24_and_escape_never_commits() -> (
    None
):
    coordinator = _Coordinator()
    host = _Host()

    async with host.run_test(size=(80, 24)) as pilot:
        modal = PersonalContextReviewModal(
            coordinator,
            session_id="session-1",
            diff=coordinator.diff,
        )
        await _push(host, modal)
        await pilot.pause()
        shell = modal.query_one("#personal-context-review-modal")
        assert shell.region.width <= 80
        assert shell.region.height <= 24
        for button_id in (
            "personal-context-review-save-only",
            "personal-context-review-save-use",
            "personal-context-review-retry-cleanup",
            "personal-context-review-close",
        ):
            button = modal.query_one(f"#{button_id}", Button)
            assert button.region.width > 0
            assert button.region.height > 0
            assert button.region.x >= shell.region.x
            assert button.region.right <= shell.region.right
            assert host.get_widget_at(*button.region.center)[0] is button
        await pilot.press("escape")
        await pilot.pause()
        assert coordinator.commits == []
