"""Workspace-state tests for guarded File Notes push rehydration."""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Label, TextArea

sys.modules.setdefault("parakeet_mlx", types.ModuleType("parakeet_mlx"))

from Tests.Notes.test_file_notes_git_push_service import (  # noqa: E402
    _publish_candidate_on_owner,
)
from Tests.UI.test_library_file_notes_git import (  # noqa: E402
    _FakeGitService,
    _DialogHarness,
    _PanelHarness,
    _RemountWorkspaceHarness,
    _WorkspaceHarness,
    _row,
    _status,
    _text,
)
from tldw_chatbook.Notes.file_notes_git_push import (  # noqa: E402
    PushAuthorizationProjection,
    PushCandidateProjection,
    PushDestinationProjection,
    PushDestinationPolicyResult,
    PushIncludedNote,
    PushReviewHandle,
    PushReviewProjection,
    PushRecoveryProjection,
    RemoteRefObservation,
    _push_destination_policy_result,
    push_outcome_copy,
    push_recovery_copy,
)
from tldw_chatbook.Notes.file_notes_git_service import (  # noqa: E402
    GitMutationAdmissionError,
    PushExecutionResult,
    PushPreflightResult,
    RetainedPushOperation,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica  # noqa: E402
from tldw_chatbook.Notes.file_notes_session_owner import (  # noqa: E402
    FileNotesSessionOwner,
    PushCandidateAvailability,
    SessionBinding,
    SessionChange,
)
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (  # noqa: E402
    LibraryFileNotesWorkspace,
)
from tldw_chatbook.Widgets.Library import (  # noqa: E402
    library_file_notes_git_panel as git_panel_module,
)

_PushResult = (
    PushDestinationPolicyResult
    | PushPreflightResult
    | PushExecutionResult
    | PushRecoveryProjection
)


def _push_availability_projection() -> PushCandidateAvailability:
    """Build one literal owner-projected push candidate for mounted UI tests."""
    candidate = PushCandidateProjection(
        local_branch_ref="refs/heads/feature/session-notes",
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
        subject="Publish exact session notes",
        included_notes=(
            PushIncludedNote(1, "folder/one.md"),
            PushIncludedNote(2, "two.md"),
        ),
    )
    return PushCandidateAvailability(
        generation=7,
        candidate=candidate,
        change_types=("Modified", "New"),
    )


def _push_destination_projection() -> PushDestinationProjection:
    """Return one exact sanitized HTTPS destination projection."""
    return PushDestinationProjection(
        "https",
        "push.example.test",
        443,
        "/team/notes.git",
        "refs/heads/session-notes",
    )


def _push_panel_review_projection():
    """Build one immutable review paired with retained note change types."""
    availability = _push_availability_projection()
    review = PushReviewProjection(
        availability.candidate,
        _push_destination_projection(),
        "origin",
    )
    return git_panel_module.PushPanelReviewProjection(
        review=review,
        availability=availability,
    )


class _PushPanelHarness(_PanelHarness):
    """Capture guarded-push presentation intents without service behavior."""

    def on_library_file_notes_git_panel_review_push_requested(
        self,
        message,
    ) -> None:
        self.messages.append(message)

    def on_library_file_notes_git_panel_push_operation_requested(
        self,
        message,
    ) -> None:
        self.messages.append(message)


class _PushGitService(_FakeGitService):
    """Deterministic retained-operation fake for workspace push state."""

    def __init__(self, owner, rows) -> None:
        super().__init__(owner, rows)
        self.push_review_calls: list[SessionBinding] = []
        self.push_query_calls: list[SessionBinding] = []
        self.push_calls: list[SessionBinding] = []
        self.cancel_push_calls: list[SessionBinding] = []
        self.authorize_and_check_calls: list[
            tuple[SessionBinding, RetainedPushOperation]
        ] = []
        self.recovery_authorization_calls: list[
            tuple[SessionBinding, RetainedPushOperation]
        ] = []
        self.recovery_operations: list[RetainedPushOperation] = []
        self.published_results: list[_PushResult] = []
        self.actual_child_started = asyncio.Event()
        self._push_operation: RetainedPushOperation | None = None
        self._push_child_signal: asyncio.Future[bool] | None = None
        self._push_operation_generation = 0
        self._planned_operations: list[
            tuple[str, _PushResult, asyncio.Event, bool]
        ] = []
        self.cancel_push_result = False
        self.push_query_errors: list[GitMutationAdmissionError] = []

    def plan_push_operation(
        self,
        kind: str,
        result: _PushResult,
        release: asyncio.Event,
        *,
        child_started: bool = False,
    ) -> None:
        """Queue the exact retained operation installed by the next start."""
        self._planned_operations.append((kind, result, release, child_started))

    def _start_planned_push_operation(
        self,
        binding: SessionBinding,
        expected_kind: str,
    ) -> asyncio.Task[_PushResult]:
        assert self._planned_operations
        kind, result, release, child_started = self._planned_operations.pop(0)
        assert kind == expected_kind
        operation = self.retain_push_operation(
            binding,
            kind,
            result,
            release,
            child_started=child_started,
        )
        return asyncio.create_task(operation.wait())

    def retain_push_operation(
        self,
        binding: SessionBinding,
        kind: str,
        result: _PushResult,
        release: asyncio.Event,
        *,
        child_started: bool = False,
        candidate: PushCandidateAvailability | None = None,
        failure: Exception | None = None,
        publish: Callable[[], None] | None = None,
    ) -> RetainedPushOperation:
        """Install one exact service-owned cycle with explicit settlement."""
        if candidate is None:
            candidate = self.owner.snapshot(binding).push_candidate
        assert candidate is not None
        child_signal: asyncio.Future[bool] | None = None
        if kind == "push":
            child_signal = asyncio.get_running_loop().create_future()
            if child_started:
                child_signal.set_result(True)
                self.actual_child_started.set()

        async def finish() -> _PushResult:
            try:
                await release.wait()
                if failure is not None:
                    raise failure
                if publish is not None:
                    publish()
                self.published_results.append(result)
                return result
            finally:
                if child_signal is not None and not child_signal.done():
                    child_signal.set_result(False)

        self._push_operation_generation += 1
        operation = RetainedPushOperation(
            binding,
            self._push_operation_generation,
            kind,  # type: ignore[arg-type]
            candidate,
            asyncio.create_task(finish()),
            child_signal,
        )
        self._push_operation = operation
        self._push_child_signal = child_signal
        return operation

    def mark_push_child_started(self) -> None:
        """Cross the fake's exact successful-child-spawn boundary."""
        signal = self._push_child_signal
        assert signal is not None and not signal.done()
        signal.set_result(True)
        self.actual_child_started.set()

    def retained_push_operation(
        self,
        binding: SessionBinding,
    ) -> RetainedPushOperation | None:
        operation = self._push_operation
        if operation is None or operation.binding != binding:
            return None
        return operation

    def start_push_review(
        self,
        binding: SessionBinding,
    ) -> asyncio.Task[PushDestinationPolicyResult]:
        self.push_review_calls.append(binding)
        return self._start_planned_push_operation(  # type: ignore[return-value]
            binding,
            "local_proof",
        )

    def authorize_and_check_push(
        self,
        binding: SessionBinding,
        operation: RetainedPushOperation,
    ) -> asyncio.Task[PushPreflightResult]:
        self.authorize_and_check_calls.append((binding, operation))
        return self._start_planned_push_operation(  # type: ignore[return-value]
            binding,
            "preflight",
        )

    def start_push(
        self,
        binding: SessionBinding,
        _handle: PushReviewHandle,
    ) -> asyncio.Task[PushExecutionResult]:
        self.push_calls.append(binding)
        return self._start_planned_push_operation(  # type: ignore[return-value]
            binding,
            "push",
        )

    def authorize_push_recovery(
        self,
        binding: SessionBinding,
        operation: RetainedPushOperation,
    ) -> bool:
        self.recovery_authorization_calls.append((binding, operation))
        return True

    def check_push_again(
        self,
        binding: SessionBinding,
        operation: RetainedPushOperation,
    ) -> asyncio.Task[PushRecoveryProjection]:
        self.push_query_calls.append(binding)
        self.recovery_operations.append(operation)
        if self.push_query_errors:
            raise self.push_query_errors.pop(0)
        return self._start_planned_push_operation(  # type: ignore[return-value]
            binding,
            "recovery",
        )

    def cancel_push(
        self,
        binding: SessionBinding,
        _operation: RetainedPushOperation,
    ) -> bool:
        self.cancel_push_calls.append(binding)
        return self.cancel_push_result


def _push_workspace_fixture(
    tmp_path: Path,
) -> tuple[
    FileNotesSessionOwner,
    SessionBinding,
    FileNotesReplica,
    _PushGitService,
    LibraryFileNotesWorkspace,
]:
    root = tmp_path / "notes"
    (root / "folder").mkdir(parents=True)
    (root / "folder" / "one.md").write_text("one", encoding="utf-8")
    (root / "two.md").write_text("two", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    assert owner.record_change(binding, SessionChange("modified", "folder/one.md"))
    assert owner.record_change(binding, SessionChange("modified", "two.md"))
    rows = (
        _row("unstaged", group_id=1, stage_action="stage"),
        _row("unstaged", group_id=2, stage_action="stage"),
    )
    service = _PushGitService(owner, rows)
    owner.attach_git_service(service)
    assert owner.publish_trust(binding, service.repository)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
        autosave_delay=10,
    )
    return owner, binding, replica, service, workspace


def _uncertain_push_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Build one mounted-ready retained uncertain push and owner recovery."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    candidate = owner.snapshot(binding).push_candidate
    assert candidate is not None
    destination = _push_destination_projection()
    recovery = push_recovery_copy(
        destination,
        RemoteRefObservation("parent", "a" * 40),
    )
    original_snapshot = FileNotesSessionOwner.snapshot

    def recovery_snapshot(
        current_owner: FileNotesSessionOwner,
        current_binding: SessionBinding,
    ):
        snapshot = original_snapshot(current_owner, current_binding)
        if current_owner is not owner or current_binding != binding:
            return snapshot
        return replace(
            snapshot,
            push_recovery=recovery,
            push_recovery_candidate=candidate,
            push_recovery_available=True,
        )

    monkeypatch.setattr(FileNotesSessionOwner, "snapshot", recovery_snapshot)
    settled = asyncio.Event()
    settled.set()
    operation = service.retain_push_operation(
        binding,
        "push",
        PushExecutionResult("uncertain", push_outcome_copy("uncertain")),
        settled,
        child_started=True,
    )
    return (
        owner,
        binding,
        replica,
        service,
        workspace,
        destination,
        operation,
    )


async def _until(pilot, predicate, message: str) -> None:
    """Drain deterministic Textual turns until one event-owned fact holds."""
    for _ in range(100):
        if predicate():
            return
        await pilot.pause()
    raise AssertionError(message)


def _assert_last_push_action(
    app: _PushPanelHarness,
    action: str,
    operation_id: int,
) -> None:
    message = app.messages[-1]
    assert isinstance(
        message,
        git_panel_module.LibraryFileNotesGitPanel.PushOperationRequested,
    )
    assert message.action == action
    assert message.operation_id == operation_id


def _assert_within_terminal(
    *widgets: Widget,
    size: tuple[int, int],
) -> None:
    """Keep a modal and its reachable controls within the compact viewport."""
    width, height = size
    for widget in widgets:
        assert widget.display
        assert 0 <= widget.region.x < width
        assert 0 <= widget.region.y < height
        assert widget.region.right <= width
        assert widget.region.bottom <= height


@pytest.mark.asyncio
async def test_push_panel_availability_is_a_separate_stable_list_action() -> None:
    """Coupling push availability to commit rows/status must break this test."""
    panel = git_panel_module.LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    availability = _push_availability_projection()

    async with _PanelHarness(panel).run_test() as pilot:
        panel.render_status(
            _status(_row("owned", group_id=1, unstage_eligible=True))
        )
        panel.render_push_availability(availability)
        await pilot.pause()

        review = panel.query_one("#file-notes-git-push-review", Button)
        commit_actions = panel.query_one("#file-notes-git-commit-actions")
        push_actions = panel.query_one("#file-notes-git-push-actions")
        siblings = tuple(commit_actions.parent.children)
        assert siblings.index(push_actions) == siblings.index(commit_actions) + 1
        assert str(review.label) == "Review push (1 commit)…"
        assert review.display
        assert not review.disabled

        panel.set_mutating(True, "Staging in progress…")
        assert review.display
        assert review.disabled
        panel.set_mutating(False)
        assert not review.disabled

        panel.mark_stale("Later session edits changed ordinary status rows.")
        await pilot.pause()
        assert review.display
        assert str(review.label) == "Review push (1 commit)…"


@pytest.mark.asyncio
async def test_push_panel_review_action_emits_exact_owner_projection() -> None:
    """Replacing the owner projection with row-derived state must fail."""
    panel = git_panel_module.LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    availability = _push_availability_projection()
    app = _PushPanelHarness(panel)

    async with app.run_test() as pilot:
        panel.render_push_availability(availability)
        panel.query_one("#file-notes-git-push-review", Button).press()
        await pilot.pause()

        assert len(app.messages) == 1
        intent = app.messages[0]
        assert intent.__class__.__name__ == "ReviewPushRequested"
        assert intent.availability is availability


@pytest.mark.asyncio
async def test_push_authorization_dialog_is_safe_explicit_and_has_endpoint_details(
) -> None:
    """A generic confirmation or hidden endpoint detail must fail this test."""
    availability = _push_availability_projection()
    destination = _push_destination_projection()
    dialog = git_panel_module.PushDestinationAuthorizationDialog(
        availability.candidate,
        PushAuthorizationProjection(destination),
    )
    app = _DialogHarness(dialog)

    async with app.run_test(size=(40, 20)) as pilot:
        await pilot.pause()
        cancel = dialog.query_one("#file-notes-push-auth-cancel", Button)
        details = dialog.query_one("#file-notes-push-auth-details", Button)
        confirm = dialog.query_one("#file-notes-push-auth-confirm", Button)
        auth_surface = dialog.query_one("#file-notes-push-auth-dialog", Widget)
        actions = dialog.query_one("#file-notes-push-auth-actions", Widget)
        _assert_within_terminal(
            dialog,
            auth_surface,
            actions,
            cancel,
            details,
            confirm,
            size=(40, 20),
        )
        assert cancel.has_focus
        copy = _text(dialog.query_one("#file-notes-push-auth-copy", Label))
        assert "Authorize configured destination" in _text(
            dialog.query_one("#file-notes-push-auth-title")
        )
        assert "push.example.test:443" in copy
        assert "/team/notes.git" in copy
        assert "Local branch: refs/heads/feature/session-notes" in copy
        assert "Full destination ref: refs/heads/session-notes" in copy
        assert "Transport: HTTPS" in copy
        assert "application process" in copy
        assert "existing SSH agent or an approved credential helper may run" in copy
        assert "Terminal prompts are disabled" in copy
        assert "checks the destination and does not push" in copy

        await pilot.press("tab")
        assert details.has_focus
        await pilot.press("shift+tab")
        assert cancel.has_focus
        await pilot.press("tab")
        assert details.has_focus
        await pilot.press("enter")
        await pilot.pause()
        endpoint_dialog = app.screen
        assert endpoint_dialog.__class__.__name__ == "PushEndpointDetailsDialog"
        endpoint_surface = endpoint_dialog.query_one(
            "#file-notes-push-endpoint-details-dialog",
            Widget,
        )
        endpoint_text = endpoint_dialog.query_one(
            "#file-notes-push-endpoint-details-text",
            TextArea,
        )
        endpoint_close = endpoint_dialog.query_one(
            "#file-notes-push-endpoint-details-close",
            Button,
        )
        _assert_within_terminal(
            endpoint_dialog,
            endpoint_surface,
            endpoint_text,
            endpoint_close,
            size=(40, 20),
        )
        assert endpoint_text.read_only
        assert endpoint_text.has_focus
        assert endpoint_text.text == "\n".join(
            f"{label}: {value}" for label, value in destination.selectable_details
        )

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is dialog
        assert details.has_focus
        await pilot.press("escape")
        await pilot.pause()
        assert app.result is False


@pytest.mark.parametrize(
    ("size", "scrolls"),
    (((40, 20), True), ((120, 40), False)),
)
@pytest.mark.asyncio
async def test_push_authorization_disclosure_is_bounded_and_keyboard_reachable(
    size: tuple[int, int],
    scrolls: bool,
) -> None:
    """The complete disclosure must wrap and remain keyboard reachable."""
    dialog = git_panel_module.PushDestinationAuthorizationDialog(
        _push_availability_projection().candidate,
        PushAuthorizationProjection(_push_destination_projection()),
    )
    app = _DialogHarness(dialog)

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        surface = dialog.query_one("#file-notes-push-auth-dialog", Widget)
        body = dialog.query_one(
            "#file-notes-push-auth-body",
            VerticalScroll,
        )
        copy = dialog.query_one("#file-notes-push-auth-copy", Label)
        actions = dialog.query_one("#file-notes-push-auth-actions", Widget)
        cancel = dialog.query_one("#file-notes-push-auth-cancel", Button)

        assert copy.region.x >= body.content_region.x
        assert copy.region.right <= body.content_region.right
        assert copy.region.width <= body.content_region.width
        assert body.region.x >= surface.content_region.x
        assert body.region.right <= surface.content_region.right
        assert body.region.bottom <= actions.region.y
        assert actions.region.bottom == surface.content_region.bottom
        _assert_within_terminal(
            surface,
            body,
            actions,
            size=size,
        )

        assert cancel.has_focus
        await pilot.press("shift+tab")
        assert body.has_focus
        prior_scroll = body.scroll_y
        await pilot.press("pagedown")
        await pilot.pause()
        assert body.has_focus
        assert (body.scroll_y > prior_scroll) is scrolls
        assert (body.max_scroll_y > 0) is scrolls
        if scrolls:
            await pilot.press("end")
            await pilot.pause()
            assert body.scroll_y == body.max_scroll_y
            await pilot.press("home")
            await pilot.pause()
            assert body.scroll_y == 0
        await pilot.press("tab")
        assert cancel.has_focus


@pytest.mark.parametrize(
    ("width", "stacked"),
    ((53, True), (56, True), (62, True), (63, False)),
)
@pytest.mark.asyncio
async def test_push_authorization_actions_use_their_real_available_width(
    width: int,
    stacked: bool,
) -> None:
    """The action layout boundary must follow the dialog's measured row."""
    dialog = git_panel_module.PushDestinationAuthorizationDialog(
        _push_availability_projection().candidate,
        PushAuthorizationProjection(_push_destination_projection()),
    )
    app = _DialogHarness(dialog)

    async with app.run_test(size=(width, 20)) as pilot:
        await pilot.pause()
        actions = dialog.query_one("#file-notes-push-auth-actions", Widget)
        buttons = tuple(actions.query(Button))
        assert actions.region.height == (3 if stacked else 1)
        for button in buttons:
            assert button.region.x >= actions.content_region.x
            assert button.region.right <= actions.content_region.right
        if stacked:
            assert all(
                button.region.width == actions.content_region.width
                for button in buttons
            )
        else:
            assert buttons[0].region.x == actions.content_region.x
            assert buttons[-1].region.right == actions.content_region.right


@pytest.mark.asyncio
async def test_push_authorization_dialog_affirmative_is_authorize_and_check(
) -> None:
    """A generic confirm action must not satisfy destination authorization."""
    dialog_type = git_panel_module.PushDestinationAuthorizationDialog
    dialog = dialog_type(
        _push_availability_projection().candidate,
        PushAuthorizationProjection(_push_destination_projection()),
    )
    app = _DialogHarness(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        confirm = dialog.query_one("#file-notes-push-auth-confirm", Button)
        assert str(confirm.label) == "Authorize and check"
        confirm.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.result is True


@pytest.mark.asyncio
async def test_push_authorization_dialog_window_close_declines() -> None:
    """Closing without an explicit result must never imply authorization."""
    dialog = git_panel_module.PushDestinationAuthorizationDialog(
        _push_availability_projection().candidate,
        PushAuthorizationProjection(_push_destination_projection()),
    )
    app = _DialogHarness(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        dialog.dismiss()
        await pilot.pause()
        assert app.result is False


@pytest.mark.asyncio
async def test_push_authorization_dialog_brackets_ipv6_endpoint_summary() -> None:
    """IPv6 endpoint copy must keep the SSH user, host, and port unambiguous."""
    destination = PushDestinationProjection(
        "ssh",
        "2001:db8::1",
        22,
        "/team/notes.git",
        "refs/heads/session-notes",
        "git",
    )
    dialog = git_panel_module.PushDestinationAuthorizationDialog(
        _push_availability_projection().candidate,
        PushAuthorizationProjection(destination),
    )

    async with _DialogHarness(dialog).run_test() as pilot:
        await pilot.pause()
        copy = _text(dialog.query_one("#file-notes-push-auth-copy", Label))
        assert (
            "Endpoint: ssh · git@[2001:db8::1]:22 · /team/notes.git"
            in copy
        )
        assert "strict snapshotted host trust" in copy
        assert "existing SSH agent only" in copy
        assert "identity files are disabled" in copy


@pytest.mark.asyncio
async def test_push_review_is_complete_immutable_and_keyboard_safe() -> None:
    """A row-derived or consequence-light final review must fail this test."""
    panel = git_panel_module.LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    app = _PushPanelHarness(panel)

    async with app.run_test(size=(80, 30)) as pilot:
        projection = _push_panel_review_projection()
        panel.render_push_review(projection, operation_id=41)
        await pilot.pause()

        assert panel.push_phase == "review"
        assert panel.commit_phase == "list"
        assert panel.query_one("#file-notes-git-push-workflow").display
        assert not panel.query_one("#file-notes-git-list-surface").display
        body = panel.query_one("#file-notes-git-push-body", VerticalScroll)
        assert body.styles.overflow_y == "auto"
        assert _text(panel.query_one("#file-notes-git-push-review-lead")) == (
            "Push 1 commit created from 2 session notes to origin/session-notes."
        )
        assert _text(panel.query_one("#file-notes-git-push-review-subject")) == (
            "Subject: Publish exact session notes"
        )
        assert _text(panel.query_one("#file-notes-git-push-review-candidate")) == (
            f"Candidate OID: {'d' * 40}"
        )
        assert _text(panel.query_one("#file-notes-git-push-review-transition")) == (
            f"Parent transition: {'a' * 40} → {'d' * 40}"
        )
        assert _text(panel.query_one("#file-notes-git-push-review-local-branch")) == (
            "Local branch: refs/heads/feature/session-notes"
        )
        assert _text(panel.query_one("#file-notes-git-push-review-remote")) == (
            "Configured remote: origin"
        )
        assert _text(panel.query_one("#file-notes-git-push-review-ref")) == (
            "Full destination ref: refs/heads/session-notes"
        )
        assert "push.example.test:443" in _text(
            panel.query_one("#file-notes-git-push-review-endpoint")
        )
        assert _text(panel.query_one("#file-notes-git-push-review-counts")) == (
            "Included changes: New 1 · Modified 1"
        )
        notes = panel.query_one("#file-notes-git-push-review-notes", TextArea)
        assert notes.read_only
        assert notes.text == "Modified: folder/one.md\nNew: two.md"
        assert _text(panel.query_one("#file-notes-git-push-review-lease")) == (
            f"Expected-parent lease: refs/heads/session-notes:{'a' * 40}"
        )
        assert _text(panel.query_one("#file-notes-git-push-review-transport")) == (
            "Secure transport: HTTPS with certificate verification; existing "
            "noninteractive authentication only; terminal prompts disabled"
        )
        assert _text(panel.query_one("#file-notes-git-push-review-local-hooks")) == (
            "Local pre-push hooks will not run"
        )
        assert _text(panel.query_one("#file-notes-git-push-review-remote-effects")) == (
            "Remote hooks, branch policy, CI, or mirrors may run"
        )
        assert _text(panel.query_one("#file-notes-git-push-review-later-edits")) == (
            "Later note edits remain local and are not added to this commit"
        )
        assert _text(panel.query_one("#file-notes-git-push-review-objects")) == (
            "Git publishes the reviewed commit and required Git objects; this "
            "list is provenance, not a separate note-transfer selection"
        )

        details = panel.query_one("#file-notes-git-push-review-details", Button)
        back = panel.query_one("#file-notes-git-push-back", Button)
        push = panel.query_one("#file-notes-git-push-confirm", Button)
        assert back.has_focus
        assert not push.has_focus
        assert tuple(button.id for button in back.parent.query(Button) if button.display) == (
            "file-notes-git-push-back",
            "file-notes-git-push-confirm",
        )

        details.press()
        await pilot.pause()
        _assert_last_push_action(app, "endpoint_details", 41)

        back.focus()
        await pilot.press("tab")
        assert push.has_focus
        await pilot.press("enter")
        await pilot.pause()
        _assert_last_push_action(app, "push_reviewed_commit", 41)


@pytest.mark.parametrize(
    (
        "phase",
        "copy",
        "detail",
        "control_id",
        "control_label",
        "action",
    ),
    (
        (
            "checking_candidate",
            "Checking push candidate…",
            "",
            "file-notes-git-push-cancel",
            "Cancel check",
            "cancel_check",
        ),
        (
            "checking_remote",
            "Checking remote before push…",
            "",
            "file-notes-git-push-cancel",
            "Cancel check",
            "cancel_check",
        ),
        (
            "checking_uncertain",
            "Checking uncertain outcome…",
            "This check does not push.",
            "file-notes-git-push-back-to-files",
            "Back to Files — check continues",
            "back_to_files",
        ),
        (
            "pushing",
            "Pushing 1 reviewed commit…",
            "Cancellation is unavailable after the network push starts.",
            "file-notes-git-push-back-to-files",
            "Back to Files — push continues",
            "back_to_files",
        ),
    ),
)
@pytest.mark.asyncio
async def test_push_panel_progress_is_compact_and_phase_safe(
    phase: git_panel_module.PushProgressPhase,
    copy: str,
    detail: str,
    control_id: str,
    control_label: str,
    action: str,
) -> None:
    """Every compact progress phase keeps a fixed safe exit and viewport."""
    panel = git_panel_module.LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    app = _PushPanelHarness(panel)

    async with app.run_test(size=(40, 20)) as pilot:
        operation_id = 51
        panel.render_push_progress(phase, operation_id=operation_id)
        await pilot.pause()
        assert panel.push_phase == phase
        assert _text(panel.query_one("#file-notes-git-push-progress-copy")) == (
            copy
        )
        assert _text(panel.query_one("#file-notes-git-push-progress-detail")) == (
            detail
        )
        body = panel.query_one("#file-notes-git-push-body", VerticalScroll)
        footer = panel.query_one("#file-notes-git-push-footer", Widget)
        control = panel.query_one(f"#{control_id}", Button)
        assert control.display
        assert str(control.label) == control_label
        assert control.has_focus
        assert body.region.x == footer.region.x == panel.content_region.x
        assert body.region.width == footer.region.width == panel.content_region.width
        assert body.region.bottom == footer.region.y
        assert footer.region.height == 1
        assert footer.region.bottom == panel.content_region.bottom
        assert control.region == footer.content_region

        body.focus()
        await pilot.press("pagedown")
        assert body.has_focus
        assert not app.messages

        control.focus()
        await pilot.press("enter")
        await pilot.pause()
        _assert_last_push_action(app, action, operation_id)
        await pilot.press("escape")
        await pilot.pause()
        _assert_last_push_action(app, action, operation_id)


@pytest.mark.parametrize(
    (
        "case",
        "title",
        "action",
        "action_enabled",
        "disabled_reason",
        "primary_id",
        "primary_label",
    ),
    (
        ("review", "", None, True, None, "file-notes-git-push-confirm", "Push 1 commit"),
        (
            "success",
            "Succeeded",
            "back_to_session",
            True,
            None,
            "file-notes-git-push-back-session",
            "Back to session",
        ),
        (
            "failed",
            "Failed with no update currently observed",
            "review_again",
            True,
            None,
            "file-notes-git-push-review-again",
            "Review again",
        ),
        (
            "uncertain-enabled",
            "Uncertain",
            "check_remote_again",
            True,
            None,
            "file-notes-git-push-check-remote",
            "Check remote again — no push",
        ),
        (
            "uncertain-disabled",
            "Uncertain",
            "check_remote_again",
            False,
            (
                "Owned push descendants are still settling; checking becomes "
                "available after every owned process ends."
            ),
            "file-notes-git-push-check-remote",
            "Check remote again — no push",
        ),
    ),
    ids=(
        "review",
        "success",
        "failed",
        "uncertain-enabled",
        "uncertain-disabled",
    ),
)
@pytest.mark.asyncio
async def test_push_panel_compact_review_and_result_matrix_is_keyboard_safe(
    case: str,
    title: str,
    action: str | None,
    action_enabled: bool,
    disabled_reason: str | None,
    primary_id: str,
    primary_label: str,
) -> None:
    """Compact review/results keep complete copy above a fixed safe footer."""
    panel = git_panel_module.LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    app = _PushPanelHarness(panel)
    operation_id = 58
    message = f"{case}: " + "complete selectable outcome copy; " * 16

    async with app.run_test(size=(40, 20)) as pilot:
        if case == "review":
            panel.render_push_review(
                _push_panel_review_projection(),
                operation_id=operation_id,
            )
        else:
            assert action is not None
            panel.render_push_result(
                git_panel_module.PushPanelResultProjection(
                    title=title,
                    message=message,
                    action=action,
                    action_enabled=action_enabled,
                    disabled_reason=disabled_reason,
                ),
                operation_id=operation_id,
            )
        await pilot.pause()

        body = panel.query_one("#file-notes-git-push-body", VerticalScroll)
        footer = panel.query_one("#file-notes-git-push-footer", Widget)
        back_id = (
            "file-notes-git-push-back"
            if case == "review"
            else "file-notes-git-push-back-session"
        )
        back = panel.query_one(f"#{back_id}", Button)
        primary = panel.query_one(f"#{primary_id}", Button)
        visible_buttons = tuple(
            button.id for button in footer.query(Button) if button.display
        )
        expected_buttons = (
            (back_id,)
            if primary_id == back_id
            else (back_id, primary_id)
        )
        assert visible_buttons == expected_buttons
        assert back.has_focus
        assert str(primary.label) == primary_label
        assert primary.region.width >= len(primary_label) + 2
        assert body.region.x == footer.region.x == panel.content_region.x
        assert body.region.width == footer.region.width == panel.content_region.width
        assert body.region.bottom == footer.region.y
        assert footer.region.bottom == panel.content_region.bottom
        assert footer.region.height == (
            2 if primary_id == "file-notes-git-push-check-remote" else 1
        )
        _assert_within_terminal(
            body,
            footer,
            back,
            primary,
            size=(40, 20),
        )

        if case == "review":
            notes = panel.query_one(
                "#file-notes-git-push-review-notes",
                TextArea,
            )
            details = panel.query_one(
                "#file-notes-git-push-review-details",
                Button,
            )
            body.focus()
            prior_scroll = body.scroll_y
            await pilot.press("pagedown")
            await pilot.pause()
            assert body.has_focus
            assert body.scroll_y > prior_scroll

            back.focus()
            await pilot.press("tab")
            assert primary.has_focus
            await pilot.press("shift+tab")
            assert back.has_focus
            await pilot.press("shift+tab")
            assert notes.has_focus
            await pilot.press("shift+tab")
            assert details.has_focus
            await pilot.press("enter")
            await pilot.pause()
            _assert_last_push_action(app, "endpoint_details", operation_id)
            back.focus()
            await pilot.press("tab", "enter")
            await pilot.pause()
            _assert_last_push_action(app, "push_reviewed_commit", operation_id)
            await pilot.press("escape")
            await pilot.pause()
            _assert_last_push_action(app, "back_from_review", operation_id)
            return

        result_title = panel.query_one(
            "#file-notes-git-push-result-title",
            Widget,
        )
        result_copy = panel.query_one(
            "#file-notes-git-push-result-copy",
            TextArea,
        )
        reason = panel.query_one("#file-notes-git-push-result-reason")
        assert _text(result_title) == title
        assert result_copy.read_only
        assert result_copy.text == message
        assert reason.display is (disabled_reason is not None)
        assert _text(reason) == (disabled_reason or "")

        await pilot.press("shift+tab")
        assert result_copy.has_focus
        await pilot.press("f7")
        assert result_copy.selected_text == message
        await pilot.press("pagedown")
        assert result_copy.has_focus
        await pilot.press("tab")
        assert back.has_focus

        if primary_id != back_id:
            await pilot.press("tab")
            if action_enabled:
                assert primary.has_focus
                await pilot.press("enter")
                await pilot.pause()
                assert action is not None
                _assert_last_push_action(app, action, operation_id)
            else:
                assert not primary.has_focus
                assert primary.disabled
                assert not app.messages
        else:
            await pilot.press("enter")
            await pilot.pause()
            _assert_last_push_action(app, "back_to_session", operation_id)

        await pilot.press("escape")
        await pilot.pause()
        _assert_last_push_action(app, "back_to_session", operation_id)


@pytest.mark.parametrize(
    ("width", "stacked"),
    ((47, True), (59, True), (60, False)),
)
@pytest.mark.asyncio
async def test_push_result_footer_uses_widest_equal_column_boundary(
    width: int,
    stacked: bool,
) -> None:
    """Unequal result labels must fit both equal columns before unstacking."""
    panel = git_panel_module.LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    app = _PushPanelHarness(panel)

    async with app.run_test(size=(width, 20)) as pilot:
        panel.render_push_result(
            git_panel_module.PushPanelResultProjection(
                title="Uncertain",
                message="The exact outcome remains uncertain.",
                action="check_remote_again",
            ),
            operation_id=59,
        )
        await pilot.pause()

        footer = panel.query_one("#file-notes-git-push-footer", Widget)
        buttons = tuple(
            button for button in footer.query(Button) if button.display
        )
        assert footer.region.height == (2 if stacked else 1)
        for button in buttons:
            required = len(str(button.label)) + button.styles.padding.width
            assert button.region.width >= required
            assert button.region.x >= footer.content_region.x
            assert button.region.right <= footer.content_region.right


@pytest.mark.asyncio
async def test_push_focus_repair_and_buffered_enter_cannot_cross_operation(
) -> None:
    """A stale checking callback must never focus or activate Push."""
    panel = git_panel_module.LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    app = _PushPanelHarness(panel)

    async with app.run_test() as pilot:
        panel.render_push_progress("checking_remote", operation_id=61)
        panel.render_push_review(
            _push_panel_review_projection(),
            operation_id=62,
        )
        await pilot.press("enter")
        await pilot.pause()

        assert panel.query_one("#file-notes-git-push-back", Button).has_focus
        assert not app.messages


@pytest.mark.asyncio
async def test_workspace_push_review_adopts_retained_operations_and_authorizes(
    tmp_path: Path,
) -> None:
    """Using returned waiters as identity or bypassing authorization must fail."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    candidate = owner.snapshot(binding).push_candidate
    assert candidate is not None
    destination = _push_destination_projection()
    local_release = asyncio.Event()
    preflight_release = asyncio.Event()
    local_result = _push_destination_policy_result("ready", destination)
    handle = object.__new__(PushReviewHandle)
    review = PushReviewProjection(candidate.candidate, destination, "origin")
    preflight_result = PushPreflightResult("review", handle, review)
    service.plan_push_operation("local_proof", local_result, local_release)
    service.plan_push_operation("preflight", preflight_result, preflight_release)

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace._rehydrate_git_presentation()
        review_button = workspace.query_one(
            "#file-notes-git-push-review",
            Button,
        )
        await _until(
            pilot,
            lambda: review_button.display,
            "owner push availability did not reach the list panel",
        )

        review_button.press()
        await _until(
            pilot,
            lambda: service.push_review_calls == [binding],
            "local push proof was not admitted",
        )
        local_operation = service.retained_push_operation(binding)
        assert local_operation is not None
        assert workspace._push_operation is local_operation
        assert workspace._git_panel_widget.push_phase == "checking_candidate"

        local_release.set()
        await _until(
            pilot,
            lambda: isinstance(
                workspace.app.screen,
                git_panel_module.PushDestinationAuthorizationDialog,
            ),
            "destination authorization did not open",
        )
        authorization = workspace.app.screen
        assert authorization.query_one(
            "#file-notes-push-auth-cancel",
            Button,
        ).has_focus
        authorization.query_one(
            "#file-notes-push-auth-confirm",
            Button,
        ).press()
        await _until(
            pilot,
            lambda: service.authorize_and_check_calls
            == [(binding, local_operation)],
            "authorization did not use the exact local-proof operation",
        )
        preflight_operation = service.retained_push_operation(binding)
        assert preflight_operation is not None
        assert preflight_operation is not local_operation
        assert workspace._push_operation is preflight_operation
        assert workspace._git_panel_widget.push_phase == "checking_remote"

        preflight_release.set()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "review",
            "immutable push review did not render",
        )
        assert workspace._push_review_handle is handle
        assert workspace._push_review_projection is not None

        workspace.query_one(
            "#file-notes-git-push-review-details",
            Button,
        ).press()
        await _until(
            pilot,
            lambda: isinstance(
                workspace.app.screen,
                git_panel_module.PushEndpointDetailsDialog,
            ),
            "review endpoint details did not open",
        )
        await pilot.press("escape")
        await pilot.pause()
        assert workspace.query_one(
            "#file-notes-git-push-review-details",
            Button,
        ).has_focus

        service.cancel_push_result = True
        workspace.query_one("#file-notes-git-push-back", Button).press()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "list",
            "Back did not discard the exact ready review",
        )
        assert service.cancel_push_calls == [binding]
        assert workspace._push_review_handle is None

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_push_keyboard_happy_path_reaches_succeeded_result(
    tmp_path: Path,
) -> None:
    """One real-keyboard journey must admit each exact operation only once."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    candidate = owner.snapshot(binding).push_candidate
    assert candidate is not None
    service.head = replace(
        service.head,
        branch=candidate.candidate.local_branch_ref,
        object_id=candidate.candidate.candidate_oid,
    )
    destination = _push_destination_projection()
    local_release = asyncio.Event()
    preflight_release = asyncio.Event()
    push_release = asyncio.Event()
    local_result = _push_destination_policy_result("ready", destination)
    handle = object.__new__(PushReviewHandle)
    review = PushReviewProjection(candidate.candidate, destination, "origin")
    preflight_result = PushPreflightResult("review", handle, review)
    push_result = PushExecutionResult(
        "succeeded",
        push_outcome_copy("succeeded"),
    )
    service.plan_push_operation("local_proof", local_result, local_release)
    service.plan_push_operation("preflight", preflight_result, preflight_release)
    service.plan_push_operation("push", push_result, push_release)

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        session_git = workspace.query_one(
            "#file-notes-session-changes",
            Button,
        )
        session_git.focus()
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: workspace._navigator_mode == "git",
            "keyboard did not open Prepare session",
        )
        review_button = workspace.query_one(
            "#file-notes-git-push-review",
            Button,
        )
        await _until(
            pilot,
            lambda: review_button.display,
            "Review push did not become visible",
        )
        review_button.focus()
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: service.push_review_calls == [binding],
            "keyboard did not start the local push proof",
        )
        local_operation = service.retained_push_operation(binding)
        assert local_operation is not None
        assert workspace._git_panel_widget.push_phase == "checking_candidate"

        local_release.set()
        await _until(
            pilot,
            lambda: isinstance(
                workspace.app.screen,
                git_panel_module.PushDestinationAuthorizationDialog,
            ),
            "destination authorization did not open",
        )
        authorization = workspace.app.screen
        cancel = authorization.query_one(
            "#file-notes-push-auth-cancel",
            Button,
        )
        details = authorization.query_one(
            "#file-notes-push-auth-details",
            Button,
        )
        confirm = authorization.query_one(
            "#file-notes-push-auth-confirm",
            Button,
        )
        assert cancel.has_focus
        await pilot.press("tab")
        assert details.has_focus
        await pilot.press("tab")
        assert confirm.has_focus
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: service.authorize_and_check_calls
            == [(binding, local_operation)],
            "authorization did not admit the exact preflight",
        )
        assert workspace._git_panel_widget.push_phase == "checking_remote"

        preflight_release.set()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "review",
            "immutable push review did not render",
        )
        back = workspace.query_one("#file-notes-git-push-back", Button)
        push = workspace.query_one("#file-notes-git-push-confirm", Button)
        assert back.has_focus
        await pilot.press("tab")
        assert push.has_focus
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: service.push_calls == [binding],
            "keyboard did not admit the reviewed push",
        )
        assert workspace._git_panel_widget.push_phase == "checking_remote"

        service.mark_push_child_started()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "pushing",
            "child start did not transition to pushing",
        )
        assert workspace.query_one(
            "#file-notes-git-push-back-to-files",
            Button,
        ).has_focus

        push_release.set()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "result",
            "successful push result did not render",
        )
        result_copy = workspace.query_one(
            "#file-notes-git-push-result-copy",
            TextArea,
        )
        result_back = workspace.query_one(
            "#file-notes-git-push-back-session",
            Button,
        )
        assert result_copy.text == push_result.outcome.message
        assert result_back.has_focus
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "list",
            "result Back did not return to the retained session",
        )

        assert service.push_review_calls == [binding]
        assert service.authorize_and_check_calls == [(binding, local_operation)]
        assert service.push_calls == [binding]
        assert service.push_query_calls == []
        assert service.cancel_push_calls == []

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.parametrize("cancel_settlement", ["blocked", "cancelled"])
@pytest.mark.asyncio
async def test_workspace_accepted_push_cancel_ignores_late_local_proof(
    tmp_path: Path,
    cancel_settlement: str,
) -> None:
    """An accepted active Cancel must retire every later observer callback."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    release = asyncio.Event()
    operation = service.retain_push_operation(
        binding,
        "local_proof",
        _push_destination_policy_result("blocked"),
        release,
    )
    service.cancel_push_result = True

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace._rehydrate_git_presentation()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase
            == "checking_candidate",
            "local proof did not render",
        )
        observer = workspace._push_observer_task
        operation_id = workspace._push_operation_id
        workspace.query_one("#file-notes-git-push-cancel", Button).press()
        await pilot.pause()

        assert service.cancel_push_calls == [binding]
        assert workspace._git_panel_widget.push_phase == "list"
        assert workspace._push_operation_id > operation_id
        assert workspace._push_observer_task is observer
        assert not workspace._push_operation_admitted

        if cancel_settlement == "cancelled":
            operation._settlement.cancel()
        else:
            release.set()
        assert observer is not None
        await observer
        assert workspace._git_panel_widget.push_phase == "list"
        assert workspace._push_result is None
        assert workspace._push_result_projection is None
        assert workspace._push_phase == "idle"

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_accepted_cancel_allows_new_candidate_review(
    tmp_path: Path,
) -> None:
    """A de-admitted candidate A must not leave candidate B unreviewable."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    candidate_a = owner.snapshot(binding).push_candidate
    assert candidate_a is not None
    release_a = asyncio.Event()
    operation_a = service.retain_push_operation(
        binding,
        "local_proof",
        _push_destination_policy_result("blocked"),
        release_a,
        candidate=candidate_a,
    )
    service.cancel_push_result = True

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace._rehydrate_git_presentation()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase
            == "checking_candidate",
            "candidate A proof did not render",
        )
        observer_a = workspace._push_observer_task
        latest_a = workspace._push_latest_service_operation_id
        workspace.query_one("#file-notes-git-push-cancel", Button).press()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "list",
            "accepted cancellation did not return to the list",
        )

        _publish_candidate_on_owner(
            owner,
            binding,
            service.repository,
            parent_oid="d" * 40,
            candidate_oid="e" * 40,
        )
        candidate_b = owner.snapshot(binding).push_candidate
        assert candidate_b is not None and candidate_b != candidate_a
        workspace._rehydrate_git_presentation()
        assert workspace._push_availability == candidate_b
        assert workspace._push_observer_task is observer_a
        assert workspace._push_operation is operation_a
        assert not workspace._push_operation_admitted

        release_b = asyncio.Event()
        service.plan_push_operation(
            "local_proof",
            _push_destination_policy_result("blocked"),
            release_b,
        )
        workspace.query_one(
            "#file-notes-git-push-review",
            Button,
        ).press()
        await _until(
            pilot,
            lambda: service.push_review_calls == [binding],
            "candidate B review was rejected by candidate A's stale key",
        )

        operation_b = service.retained_push_operation(binding)
        assert operation_b is not None and operation_b is not operation_a
        assert operation_b.candidate == candidate_b
        assert workspace._push_operation is operation_b
        assert workspace._push_operation_admitted
        assert workspace._push_observer_task is not observer_a
        assert workspace._push_latest_service_operation_id > latest_a

        release_a.set()
        assert observer_a is not None
        await observer_a
        assert workspace._push_operation is operation_b
        assert workspace._push_result is None
        assert workspace._git_panel_widget.push_phase == "checking_candidate"

        release_b.set()
        await operation_b.wait()

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_old_cancelled_replay_cannot_strand_new_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignored retained A must rekey B after an immediate review refusal."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    candidate_a = owner.snapshot(binding).push_candidate
    assert candidate_a is not None
    release_a = asyncio.Event()
    operation_a = service.retain_push_operation(
        binding,
        "local_proof",
        _push_destination_policy_result("blocked"),
        release_a,
        candidate=candidate_a,
    )
    service.cancel_push_result = True
    original_start_review = service.start_push_review

    def refuse_first_review(current_binding: SessionBinding):
        if not service.push_review_calls:
            service.push_review_calls.append(current_binding)
            raise GitMutationAdmissionError(
                "mutation_active",
                "raw-old-candidate-still-settling",
            )
        return original_start_review(current_binding)

    monkeypatch.setattr(service, "start_push_review", refuse_first_review)

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace._rehydrate_git_presentation()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase
            == "checking_candidate",
            "candidate A proof did not render",
        )
        observer_a = workspace._push_observer_task
        workspace.query_one("#file-notes-git-push-cancel", Button).press()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "list",
            "accepted cancellation did not return to the list",
        )

        workspace.query_one("#file-notes-git-push-review", Button).press()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "result",
            "immediate mutation_active refusal did not render",
        )
        assert service.push_review_calls == [binding]
        assert workspace._push_operation is None
        assert workspace._push_observer_task is None

        _publish_candidate_on_owner(
            owner,
            binding,
            service.repository,
            parent_oid="d" * 40,
            candidate_oid="e" * 40,
        )
        candidate_b = owner.snapshot(binding).push_candidate
        assert candidate_b is not None and candidate_b != candidate_a
        workspace._rehydrate_git_presentation()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "list",
            "ignored retained A did not yield candidate B's list state",
        )
        assert workspace._push_availability == candidate_b
        assert workspace._push_key is not None
        assert workspace._push_key.candidate == candidate_b.candidate
        assert workspace._push_observer_task is None

        release_b = asyncio.Event()
        service.plan_push_operation(
            "local_proof",
            _push_destination_policy_result("blocked"),
            release_b,
        )
        workspace.query_one("#file-notes-git-push-review", Button).press()
        await _until(
            pilot,
            lambda: service.push_review_calls == [binding, binding],
            "candidate B review was rejected by ignored candidate A",
        )
        operation_b = service.retained_push_operation(binding)
        assert operation_b is not None and operation_b is not operation_a
        assert operation_b.candidate == candidate_b
        observer_b = workspace._push_observer_task
        assert observer_b is not None and observer_b is not observer_a

        service._push_operation = operation_a
        workspace._rehydrate_git_presentation()
        assert workspace._push_operation is operation_b
        assert workspace._push_observer_task is observer_b
        assert workspace._git_panel_widget.push_phase == "checking_candidate"

        release_a.set()
        assert observer_a is not None
        await observer_a
        assert workspace._push_operation is operation_b
        assert workspace._push_result is None

        service._push_operation = operation_b
        release_b.set()
        await operation_b.wait()

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.parametrize(
    ("kind", "child_started", "expected_action"),
    [
        ("local_proof", False, "review_again"),
        ("preflight", False, "review_again"),
        ("push", True, "back_to_session"),
    ],
)
@pytest.mark.asyncio
async def test_workspace_push_observer_exception_renders_sanitized_attention(
    tmp_path: Path,
    kind: str,
    child_started: bool,
    expected_action: str,
) -> None:
    """An exact observer failure must replace progress without leaking text."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    release = asyncio.Event()
    operation = service.retain_push_operation(
        binding,
        kind,
        _push_destination_policy_result("blocked"),
        release,
        child_started=child_started,
        failure=RuntimeError("raw-secret-observer-failure"),
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace._rehydrate_git_presentation()
        release.set()
        observer = workspace._push_observer_task
        assert observer is not None
        await observer

        assert workspace._push_operation is operation
        assert workspace._git_panel_widget.push_phase == "result"
        projection = workspace._push_result_projection
        assert projection is not None
        assert projection.action == expected_action
        rendered = workspace.query_one(
            "#file-notes-git-push-result-copy",
            TextArea,
        ).text
        assert "raw-secret-observer-failure" not in rendered
        assert "retry" not in rendered.casefold()
        assert workspace._push_phase == "needs_attention"

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_push_review_admission_failure_has_safe_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-operation failure must be visible and Back must need no authority."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )

    def reject_review(current_binding: SessionBinding) -> None:
        service.push_review_calls.append(current_binding)
        raise GitMutationAdmissionError(
            "mutation_active",
            "raw-secret-review-admission-failure",
        )

    monkeypatch.setattr(service, "start_push_review", reject_review)

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace._rehydrate_git_presentation()
        review = workspace.query_one("#file-notes-git-push-review", Button)
        await _until(
            pilot,
            lambda: review.display,
            "push availability did not render",
        )

        review.press()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "result",
            "admission failure did not replace the list with a result",
        )

        assert service.push_review_calls == [binding]
        assert workspace._push_operation is None
        assert workspace._push_result_projection is not None
        assert workspace._push_result_projection.action == "back_to_session"
        rendered = workspace.query_one(
            "#file-notes-git-push-result-copy",
            TextArea,
        ).text
        assert "raw-secret-review-admission-failure" not in rendered
        assert "retry" not in rendered.casefold()
        assert workspace._push_phase == "needs_attention"

        back = workspace.query_one(
            "#file-notes-git-push-back-session",
            Button,
        )
        assert back.display
        assert back.has_focus
        back.press()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "list",
            "Back retained pre-operation authority that does not exist",
        )

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_push_child_boundary_hides_cancel_and_keeps_observer(
    tmp_path: Path,
) -> None:
    """UI inference or cancellation after the exact child start must fail."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    candidate = owner.snapshot(binding).push_candidate
    assert candidate is not None
    destination = _push_destination_projection()
    handle = object.__new__(PushReviewHandle)
    review = PushReviewProjection(candidate.candidate, destination, "origin")
    ready_release = asyncio.Event()
    ready_release.set()
    service.retain_push_operation(
        binding,
        "preflight",
        PushPreflightResult("review", handle, review),
        ready_release,
    )
    push_release = asyncio.Event()
    push_result = PushExecutionResult(
        "uncertain",
        push_outcome_copy("uncertain"),
    )
    service.plan_push_operation("push", push_result, push_release)

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        assert await workspace.open_path("folder/one.md")
        editor = workspace.query_one("#file-notes-editor", TextArea)
        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace._rehydrate_git_presentation()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "review",
            "retained ready review did not rehydrate",
        )

        workspace.query_one("#file-notes-git-push-confirm", Button).press()
        await _until(
            pilot,
            lambda: service.push_calls == [binding],
            "reviewed push was not admitted",
        )
        operation = service.retained_push_operation(binding)
        assert operation is not None and operation.kind == "push"
        observer = workspace._push_observer_task
        assert observer is not None
        assert workspace._push_operation is operation
        assert workspace._git_panel_widget.push_phase == "checking_remote"
        assert workspace.query_one("#file-notes-git-push-cancel", Button).display
        assert not editor.read_only
        assert workspace._editor_read_only_leases == {}

        service.mark_push_child_started()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "pushing",
            "actual child start did not render pushing",
        )
        assert not workspace.query_one(
            "#file-notes-git-push-cancel",
            Button,
        ).display
        back = workspace.query_one(
            "#file-notes-git-push-back-to-files",
            Button,
        )
        assert back.display
        assert not editor.read_only
        back.press()
        await _until(
            pilot,
            lambda: workspace._navigator_mode == "files",
            "Back to Files did not hide the admitted push",
        )
        assert service.cancel_push_calls == []
        assert workspace._push_observer_task is observer

        push_release.set()
        assert await operation.wait() is push_result
        await _until(
            pilot,
            lambda: workspace._push_result is push_result,
            "hidden push outcome did not publish",
        )
        assert workspace._push_phase == "needs_attention"
        assert workspace._push_observer_task is observer

        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace._rehydrate_git_presentation()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "result",
            "hidden result did not render on return",
        )
        assert service.push_calls == [binding]
        assert workspace._push_observer_task is observer
        assert workspace.query_one(
            "#file-notes-git-push-result-copy",
            TextArea,
        ).text == push_result.outcome.message
        check = workspace.query_one(
            "#file-notes-git-push-check-remote",
            Button,
        )
        assert check.disabled
        recovery = push_recovery_copy(
            destination,
            RemoteRefObservation("parent", "a" * 40),
        )
        workspace._rehydrate_push_state(
            service,
            binding,
            replace(
                owner.snapshot(binding),
                push_recovery=recovery,
                push_recovery_candidate=operation.candidate,
                push_recovery_available=True,
            ),
        )
        await pilot.pause()
        assert not check.disabled

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_push_failure_review_again_starts_fresh_local_proof(
    tmp_path: Path,
) -> None:
    """Review again must start a new proof, never reuse or retry the push."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    failed_release = asyncio.Event()
    failed_release.set()
    failed = PushExecutionResult(
        "failed_no_update_observed",
        push_outcome_copy("failed_no_update_observed"),
    )
    service.retain_push_operation(binding, "push", failed, failed_release)
    review_release = asyncio.Event()
    service.plan_push_operation(
        "local_proof",
        _push_destination_policy_result("blocked"),
        review_release,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace._rehydrate_git_presentation()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "result",
            "definite failure did not render",
        )

        workspace.query_one(
            "#file-notes-git-push-review-again",
            Button,
        ).press()
        await _until(
            pilot,
            lambda: service.push_review_calls == [binding],
            "Review again did not start a fresh local proof",
        )
        next_operation = service.retained_push_operation(binding)
        assert next_operation is not None
        assert next_operation.kind == "local_proof"
        assert workspace._push_operation is next_operation
        assert workspace._git_panel_widget.push_phase == "checking_candidate"
        assert service.push_calls == []

        review_release.set()
        await next_operation.wait()

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_push_recovery_authorizes_then_queries_retained_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed authorization_required must reopen auth before query-only recovery."""
    (
        owner,
        binding,
        replica,
        service,
        workspace,
        destination,
        original_operation,
    ) = _uncertain_push_workspace(
        tmp_path,
        monkeypatch,
    )
    service.push_query_errors.append(
        GitMutationAdmissionError(
            "authorization_required",
            "fresh destination authorization is required",
        )
    )
    recovery_release = asyncio.Event()
    recovered = push_recovery_copy(
        destination,
        RemoteRefObservation("candidate", "d" * 40),
    )
    service.plan_push_operation("recovery", recovered, recovery_release)

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace._rehydrate_git_presentation()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "result",
            "uncertain result did not render",
        )
        check = workspace.query_one(
            "#file-notes-git-push-check-remote",
            Button,
        )
        assert not check.disabled

        check.press()
        await _until(
            pilot,
            lambda: isinstance(
                workspace.app.screen,
                git_panel_module.PushDestinationAuthorizationDialog,
            ),
            "typed authorization_required did not open the destination dialog",
        )
        assert service.recovery_operations == [original_operation]
        workspace.app.screen.query_one(
            "#file-notes-push-auth-confirm",
            Button,
        ).press()
        await _until(
            pilot,
            lambda: len(service.push_query_calls) == 2,
            "authorized recovery did not start the query-only check",
        )
        assert service.recovery_authorization_calls == [
            (binding, original_operation)
        ]
        assert service.recovery_operations == [
            original_operation,
            original_operation,
        ]
        recovery_operation = service.retained_push_operation(binding)
        assert recovery_operation is not None
        assert recovery_operation.kind == "recovery"
        assert workspace._push_operation is recovery_operation
        assert workspace._git_panel_widget.push_phase == "checking_uncertain"
        assert service.push_calls == []
        assert not recovery_operation.child_started
        recovery_observer = workspace._push_observer_task
        assert recovery_observer is not None

        await pilot.press("escape")
        await _until(
            pilot,
            lambda: workspace._navigator_mode == "files",
            "Back to Files did not hide the query-only recovery",
        )
        assert service.cancel_push_calls == []
        assert workspace._push_operation is recovery_operation
        assert workspace._push_observer_task is recovery_observer
        assert service.push_query_calls == [binding, binding]
        session_git = workspace.query_one(
            "#file-notes-session-changes",
            Button,
        )
        assert "Push checking" in str(session_git.label)
        assert session_git.has_focus

        recovery_release.set()
        await _until(
            pilot,
            lambda: workspace._push_result is recovered,
            "query-only recovery result did not settle",
        )
        session_git.focus()
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "result",
            "settled query-only recovery did not rehydrate its result",
        )
        assert workspace._push_operation is recovery_operation
        assert workspace._push_observer_task is recovery_observer
        assert service.push_query_calls == [binding, binding]
        workspace.query_one(
            "#file-notes-git-push-back-session",
            Button,
        ).press()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "list",
            "Back to session did not return to the Session Git list",
        )

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.parametrize("size", ((40, 20), (120, 40)))
@pytest.mark.asyncio
async def test_workspace_reopens_original_uncertain_push_without_new_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: tuple[int, int],
) -> None:
    """Back, Files, and keyboard reopen must restore the uncertain push."""
    (
        owner,
        binding,
        replica,
        service,
        workspace,
        _destination,
        operation,
    ) = _uncertain_push_workspace(
        tmp_path,
        monkeypatch,
    )

    async with _WorkspaceHarness(workspace).run_test(size=size) as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace._rehydrate_git_presentation()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "result",
            "original uncertain push result did not render",
        )

        observer = workspace._push_observer_task
        result = workspace._push_result
        projection = workspace._push_result_projection
        assert observer is not None
        assert isinstance(result, PushExecutionResult)
        assert result.state == "uncertain"
        assert result.outcome is not None
        assert projection is not None
        assert projection.action == "check_remote_again"

        back_to_session = workspace.query_one(
            "#file-notes-git-push-back-session",
            Button,
        )
        back_to_session.focus()
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "list",
            "Back to session did not restore the Session Git list",
        )
        workspace._refresh_session_changes()
        assert workspace._git_panel_widget.push_phase == "list"
        assert workspace._push_result_projection is None
        workspace._rehydrate_git_presentation()
        assert workspace._git_panel_widget.push_phase == "list"
        assert workspace._push_result_projection is None

        snapshot = owner.snapshot(binding)
        recovery_candidate = snapshot.push_recovery_candidate
        assert snapshot.push_recovery is not None
        assert recovery_candidate is not None
        mismatched_candidate = replace(
            recovery_candidate,
            generation=recovery_candidate.generation + 1,
        )
        assert workspace._push_key_for_operation(
            operation
        ) != workspace._push_key_for_availability(
            binding,
            mismatched_candidate,
        )
        workspace._rehydrate_push_state(
            service,
            binding,
            replace(
                snapshot,
                push_recovery_candidate=mismatched_candidate,
            ),
        )
        assert workspace._push_operation is operation
        assert workspace._git_panel_widget.push_phase == "list"
        assert workspace._push_result_projection is None

        files = workspace.query_one("#file-notes-git-back", Button)
        files.focus()
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: workspace._navigator_mode == "files",
            "Files did not restore the notes navigator",
        )
        session_git = workspace.query_one(
            "#file-notes-session-changes",
            Button,
        )
        session_git.focus()
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "result",
            "keyboard reopen did not restore the original uncertain result",
        )

        assert workspace._push_operation is operation
        assert workspace._push_observer_task is observer
        assert workspace._push_result is result
        assert workspace._push_result_projection == projection
        assert workspace.query_one(
            "#file-notes-git-push-result-copy",
            TextArea,
        ).text == result.outcome.message
        assert workspace.query_one(
            "#file-notes-git-push-check-remote",
            Button,
        ).display
        assert service.push_review_calls == []
        assert service.push_query_calls == []
        assert service.push_calls == []
        assert service.cancel_push_calls == []

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.parametrize("failure_mode", ["authorization", "typed_admission"])
@pytest.mark.asyncio
async def test_workspace_push_recovery_failure_replaces_stale_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    """Recovery refusal must be visible, sanitized, and safely terminal."""
    (
        owner,
        binding,
        replica,
        service,
        workspace,
        _destination,
        original_operation,
    ) = _uncertain_push_workspace(
        tmp_path,
        monkeypatch,
    )
    if failure_mode == "authorization":
        service.push_query_errors.append(
            GitMutationAdmissionError(
                "authorization_required",
                "raw-secret-authorization-admission",
            )
        )

        def reject_authorization(
            current_binding: SessionBinding,
            operation: RetainedPushOperation,
        ) -> bool:
            service.recovery_authorization_calls.append(
                (current_binding, operation)
            )
            return False

        monkeypatch.setattr(
            service,
            "authorize_push_recovery",
            reject_authorization,
        )
    else:
        service.push_query_errors.append(
            GitMutationAdmissionError(
                "mutation_active",
                "raw-secret-recovery-admission",
            )
        )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace._rehydrate_git_presentation()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "result",
            "uncertain result did not render",
        )

        workspace.query_one(
            "#file-notes-git-push-check-remote",
            Button,
        ).press()
        if failure_mode == "authorization":
            await _until(
                pilot,
                lambda: isinstance(
                    workspace.app.screen,
                    git_panel_module.PushDestinationAuthorizationDialog,
                ),
                "authorization_required did not open the dialog",
            )
            workspace.app.screen.query_one(
                "#file-notes-push-auth-confirm",
                Button,
            ).press()

        await _until(
            pilot,
            lambda: (
                workspace._push_result_projection is not None
                and workspace._push_result_projection.title
                == "Remote check unavailable"
            ),
            "recovery failure left the stale Check action visible",
        )

        projection = workspace._push_result_projection
        assert projection is not None
        assert projection.action == "back_to_session"
        rendered = workspace.query_one(
            "#file-notes-git-push-result-copy",
            TextArea,
        ).text
        assert "raw-secret" not in rendered
        assert "retry" not in rendered.casefold()
        assert "externally" in rendered
        assert service.push_query_calls == [binding]
        assert service.recovery_operations == [original_operation]
        assert service.recovery_authorization_calls == (
            [(binding, original_operation)]
            if failure_mode == "authorization"
            else []
        )
        assert service.push_calls == []
        assert workspace._push_phase == "needs_attention"
        back = workspace.query_one(
            "#file-notes-git-push-back-session",
            Button,
        )
        assert back.display
        assert back.has_focus
        back.press()
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "list",
            "Back did not leave the recovery failure",
        )

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_rehydrates_push_candidate_without_commit_status(
    tmp_path,
) -> None:
    """Removing commit/status coupling must leave owner push availability."""
    (
        owner,
        binding,
        replica,
        git_service,
        workspace,
    ) = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        git_service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )

        assert workspace._commit_availability is None
        assert workspace._push_availability == owner.snapshot(binding).push_candidate
        assert git_service.review_calls == []
        assert git_service.commit_calls == []

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_edit_stales_rows_without_hiding_push_candidate(
    tmp_path: Path,
) -> None:
    """Coupling push to current rows must not erase an immutable candidate."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    release = asyncio.Event()
    operation = service.retain_push_operation(
        binding,
        "local_proof",
        _push_destination_policy_result("blocked"),
        release,
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        candidate = workspace._push_availability
        assert candidate is not None

        assert owner.record_change(
            binding,
            SessionChange("created", "later.md"),
        )
        workspace._refresh_session_changes()

        assert workspace._push_availability == candidate
        assert "Push checking" in str(
            workspace.query_one("#file-notes-session-changes").label
        )
        release.set()
        await operation.wait()

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_first_rehydrate_rejects_stale_retained_candidate(
    tmp_path: Path,
) -> None:
    """A retained phase for candidate A must not attach to newer candidate B."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    candidate_a = owner.snapshot(binding).push_candidate
    assert candidate_a is not None
    release = asyncio.Event()
    operation_a = service.retain_push_operation(
        binding,
        "local_proof",
        _push_destination_policy_result("blocked"),
        release,
        candidate=candidate_a,
    )
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="d" * 40,
        candidate_oid="e" * 40,
    )
    candidate_b = owner.snapshot(binding).push_candidate
    assert candidate_b is not None and candidate_b != candidate_a

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )

        assert workspace._push_availability == candidate_b
        assert workspace._push_observer_task is None
        assert workspace._push_phase == "idle"
        assert "Push checking" not in str(
            workspace.query_one("#file-notes-session-changes").label
        )

        assert workspace._rehydrate_git_presentation()
        assert workspace._push_observer_task is None
        release.set()
        await operation_a.wait()
        await pilot.pause()
        assert workspace._push_result is None
        assert workspace._push_phase == "idle"

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_push_callback_requires_exact_candidate_and_operation(
    tmp_path: Path,
) -> None:
    """An older same-binding settlement must not overwrite newer push state."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    release_a = asyncio.Event()
    operation_a = service.retain_push_operation(
        binding,
        "local_proof",
        _push_destination_policy_result("blocked"),
        release_a,
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        key = workspace._push_key
        assert key is not None
        operation_id_a = workspace._push_operation_id

        release_b = asyncio.Event()
        operation_b = service.retain_push_operation(
            binding,
            "preflight",
            PushPreflightResult("blocked"),
            release_b,
        )
        assert workspace._rehydrate_git_presentation()
        operation_id_b = workspace._push_operation_id
        assert operation_id_b > operation_id_a
        assert workspace._push_operation is operation_b

        release_a.set()
        await operation_a.wait()
        await pilot.pause()

        assert workspace._push_operation is operation_b
        assert workspace._push_phase == "checking"
        assert not workspace._push_operation_is_current(
            operation_a,
            key,
            operation_id_a,
        )
        assert not workspace._push_operation_is_current(
            operation_b,
            replace(key, generation=key.generation + 1),
            operation_id_b,
        )
        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace._rehydrate_git_presentation()
        assert workspace._git_panel_widget.push_phase == "checking_remote"
        snapshot = owner.snapshot(binding)
        assert snapshot.push_candidate is not None
        drifted = replace(
            snapshot,
            push_candidate=replace(
                snapshot.push_candidate,
                generation=snapshot.push_candidate.generation + 1,
            ),
        )
        observer_b = workspace._push_observer_task
        workspace._rehydrate_push_state(service, binding, drifted)
        assert workspace._push_operation is operation_b
        assert workspace._push_observer_task is observer_b
        assert workspace._push_phase == "idle"
        assert workspace._git_panel_widget.push_phase == "list"
        workspace._rehydrate_push_state(service, binding, drifted)
        assert workspace._push_observer_task is observer_b
        release_b.set()
        await operation_b.wait()
        await pilot.pause()
        assert workspace._push_phase == "idle"

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_binding_change_rejects_old_push_settlement(
    tmp_path: Path,
) -> None:
    """A real owner root generation change must retire old push UI state."""
    owner, binding_a, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding_a,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    release = asyncio.Event()
    operation_a = service.retain_push_operation(
        binding_a,
        "push",
        PushExecutionResult("uncertain", push_outcome_copy("uncertain")),
        release,
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        assert workspace._push_phase == "checking"

        other_root = tmp_path / "other-notes"
        other_root.mkdir()
        binding_b = owner.select_root(other_root)
        assert binding_b != binding_a
        workspace._refresh_session_changes()
        assert workspace._push_phase == "idle"
        assert not workspace._rehydrate_git_presentation()
        assert workspace._push_phase == "idle"
        assert "Push checking" not in str(
            workspace.query_one("#file-notes-session-changes").label
        )

        release.set()
        await operation_a.wait()
        await pilot.pause()
        assert workspace._push_result is None
        assert workspace._push_phase == "idle"
        assert "Push needs attention" not in str(
            workspace.query_one("#file-notes-session-changes").label
        )

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_candidate_aba_does_not_reattach_old_operation(
    tmp_path: Path,
) -> None:
    """Returning to candidate A's projection must not revive A's old phase."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    candidate_a = owner.snapshot(binding).push_candidate
    assert candidate_a is not None
    release_a = asyncio.Event()
    operation_a = service.retain_push_operation(
        binding,
        "local_proof",
        _push_destination_policy_result("blocked"),
        release_a,
        candidate=candidate_a,
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        observer_a = workspace._push_observer_task
        assert observer_a is not None

        _publish_candidate_on_owner(
            owner,
            binding,
            service.repository,
            parent_oid="d" * 40,
            candidate_oid="e" * 40,
        )
        assert workspace._rehydrate_git_presentation()
        assert workspace._push_phase == "idle"

        _publish_candidate_on_owner(
            owner,
            binding,
            service.repository,
            parent_oid="a" * 40,
            candidate_oid="d" * 40,
        )
        candidate_a_again = owner.snapshot(binding).push_candidate
        assert candidate_a_again is not None
        assert candidate_a_again.generation > candidate_a.generation
        aba_snapshot = replace(
            owner.snapshot(binding),
            push_candidate=replace(
                candidate_a_again,
                candidate=candidate_a.candidate,
            ),
        )
        workspace._rehydrate_push_state(service, binding, aba_snapshot)
        assert workspace._push_observer_task is observer_a
        assert workspace._push_phase == "idle"

        workspace._rehydrate_push_state(service, binding, aba_snapshot)
        release_a.set()
        await operation_a.wait()
        await pilot.pause()
        assert workspace._push_result is None
        assert workspace._push_phase == "idle"

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_operation_aba_keeps_newer_exact_observer(
    tmp_path: Path,
) -> None:
    """A service A→B→A replay must not replace B with older operation A."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    release_a = asyncio.Event()
    operation_a = service.retain_push_operation(
        binding,
        "local_proof",
        _push_destination_policy_result("blocked"),
        release_a,
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        release_b = asyncio.Event()
        operation_b = service.retain_push_operation(
            binding,
            "preflight",
            PushPreflightResult("blocked"),
            release_b,
        )
        assert workspace._rehydrate_git_presentation()
        observer_b = workspace._push_observer_task
        assert workspace._push_operation is operation_b

        service._push_operation = operation_a
        assert workspace._rehydrate_git_presentation()
        assert workspace._push_operation is operation_b
        assert workspace._push_observer_task is observer_b

        service._push_operation = operation_b
        assert workspace._rehydrate_git_presentation()
        service._push_operation = operation_a
        assert workspace._rehydrate_git_presentation()
        release_a.set()
        await operation_a.wait()
        await pilot.pause()
        assert workspace._push_operation is operation_b
        assert workspace._push_result is None
        assert workspace._push_phase == "checking"

        service._push_operation = operation_b
        release_b.set()
        await operation_b.wait()
        await _until(
            pilot,
            lambda: workspace._push_result is not None,
            "newer operation B did not publish",
        )

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_admitted_push_settles_after_candidate_clears(
    tmp_path: Path,
) -> None:
    """A successful push may clear its live candidate before its waiter resumes."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    release = asyncio.Event()
    result = PushExecutionResult("succeeded", push_outcome_copy("succeeded"))
    operation = service.retain_push_operation(
        binding,
        "push",
        result,
        release,
        child_started=True,
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        observer = workspace._push_observer_task
        assert observer is not None
        snapshot = owner.snapshot(binding)
        workspace._rehydrate_push_state(
            service,
            binding,
            replace(
                snapshot,
                push_candidate=None,
                push_candidate_generation=snapshot.push_candidate_generation + 1,
            ),
        )
        assert workspace._push_operation is operation
        assert workspace._push_operation_admitted
        assert workspace._push_observer_task is observer
        assert workspace._push_phase == "pushing"

        release.set()
        await operation.wait()
        await _until(
            pilot,
            lambda: workspace._push_result is result,
            "admitted push settlement was lost after candidate clear",
        )
        assert workspace._push_phase == "idle"

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_recovery_first_attach_uses_retained_candidate(
    tmp_path: Path,
) -> None:
    """Recovery may attach from retained evidence after live candidate removal."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    destination = PushDestinationProjection(
        "https",
        "push.example.test",
        443,
        "/team/notes.git",
        "refs/heads/main",
    )
    result = push_recovery_copy(
        destination,
        RemoteRefObservation("parent", "a" * 40),
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        release = asyncio.Event()
        operation = service.retain_push_operation(
            binding,
            "recovery",
            result,
            release,
        )
        snapshot = owner.snapshot(binding)
        workspace._rehydrate_push_state(
            service,
            binding,
            replace(
                snapshot,
                push_candidate=None,
                push_candidate_generation=snapshot.push_candidate_generation + 1,
                push_recovery=result,
                push_recovery_candidate=operation.candidate,
            ),
        )
        assert workspace._push_operation is operation
        assert workspace._push_operation_admitted
        assert workspace._push_operation_key is not None
        assert workspace._push_operation_key.candidate == operation.candidate.candidate
        assert workspace._push_phase == "checking"

        release.set()
        await operation.wait()
        await _until(
            pilot,
            lambda: workspace._push_result is result,
            "retained recovery settlement did not publish",
        )

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_recovery_first_attach_rejects_mismatched_candidate(
    tmp_path: Path,
) -> None:
    """Recovery B must not admit a retained recovery operation for candidate A."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    candidate_a = owner.snapshot(binding).push_candidate
    assert candidate_a is not None
    candidate_b = replace(candidate_a, generation=candidate_a.generation + 1)
    destination = PushDestinationProjection(
        "https",
        "push.example.test",
        443,
        "/team/notes.git",
        "refs/heads/main",
    )
    result = push_recovery_copy(
        destination,
        RemoteRefObservation("parent", "a" * 40),
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        release = asyncio.Event()
        operation_a = service.retain_push_operation(
            binding,
            "recovery",
            result,
            release,
            candidate=candidate_a,
        )
        snapshot = owner.snapshot(binding)
        workspace._rehydrate_push_state(
            service,
            binding,
            replace(
                snapshot,
                push_recovery=result,
                push_recovery_candidate=candidate_b,
            ),
        )
        assert workspace._push_operation is operation_a
        assert not workspace._push_operation_admitted
        assert workspace._push_observer_task is None
        assert workspace._push_phase == "needs_attention"

        release.set()
        await operation_a.wait()
        await pilot.pause()
        assert workspace._push_result is None

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.parametrize("size", ((40, 20), (120, 40)))
@pytest.mark.asyncio
async def test_workspace_rehydrate_hidden_settlement_without_duplicate_work(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    """Leave/remount must reuse the exact observer without launching Git again."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    release = asyncio.Event()
    result = PushExecutionResult("uncertain", push_outcome_copy("uncertain"))
    operation = service.retain_push_operation(
        binding,
        "push",
        result,
        release,
        child_started=True,
    )
    app = _RemountWorkspaceHarness(workspace)

    async with app.run_test(size=size) as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        assert await workspace.open_path("folder/one.md")
        editor = workspace.query_one("#file-notes-editor", TextArea)
        assert not editor.read_only
        assert workspace._editor_read_only_leases == {}
        if size == (40, 20):
            navigator_back = workspace.query_one("#file-notes-back", Button)
            assert navigator_back.display
            navigator_back.focus()
            await pilot.press("enter")
            assert workspace._narrow_view == "navigator"
            assert workspace.query_one("#file-notes-navigator").display
            assert not workspace.query_one("#file-notes-editor-pane").display

        observer = workspace._push_observer_task
        assert observer is not None
        assert "Pushing" in str(
            workspace.query_one("#file-notes-session-changes").label
        )
        session_git = workspace.query_one(
            "#file-notes-session-changes",
            Button,
        )
        session_git.focus()
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "pushing",
            "retained push did not open from Session Git",
        )
        back_to_files = workspace.query_one(
            "#file-notes-git-push-back-to-files",
            Button,
        )
        assert back_to_files.has_focus
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: workspace._navigator_mode == "files",
            "keyboard did not leave the retained push",
        )
        assert workspace._push_observer_task is observer
        assert workspace._push_operation is operation
        assert service.cancel_push_calls == []

        host = app.query_one("#remount-workspace-host")
        await host.remove_children()
        await _until(
            pilot,
            lambda: not workspace._active,
            "workspace did not unmount",
        )

        release.set()
        assert await operation.wait() is result
        await observer
        assert workspace._push_phase == "needs_attention"

        await host.mount(workspace)
        await _until(
            pilot,
            lambda: workspace._active,
            "workspace did not remount",
        )
        assert workspace._push_observer_task is observer
        assert workspace._push_operation is operation
        assert "Push needs attention" in str(
            workspace.query_one("#file-notes-session-changes").label
        )
        assert not editor.read_only
        assert workspace._editor_read_only_leases == {}
        action_status = workspace.query_one("#file-notes-action-status")
        assert result.outcome is not None
        assert result.outcome.message not in _text(action_status)
        if size == (40, 20):
            assert workspace._narrow_view == "navigator"
            assert workspace.query_one("#file-notes-navigator").display
            assert not workspace.query_one("#file-notes-editor-pane").display

        session_git = workspace.query_one(
            "#file-notes-session-changes",
            Button,
        )
        session_git.focus()
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: workspace._git_panel_widget.push_phase == "result",
            "hidden result did not render after keyboard reopen",
        )
        assert workspace._push_observer_task is observer
        assert workspace._push_operation is operation
        assert workspace.query_one(
            "#file-notes-git-push-result-copy",
            TextArea,
        ).text == result.outcome.message
        if size == (40, 20):
            assert workspace.query_one("#file-notes-navigator").display
            assert not workspace.query_one("#file-notes-editor-pane").display

        assert service.push_review_calls == []
        assert service.push_query_calls == []
        assert service.push_calls == []
        assert service.cancel_push_calls == []
        assert service.published_results == [result]

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_push_phases_keep_editor_editable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any accidental commit-lease reuse must fail every push phase."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )

    def reject_editor_lease(_binding: SessionBinding):
        raise AssertionError("push must not acquire _EditorReadOnlyLease")

    monkeypatch.setattr(workspace, "_acquire_editor_read_only", reject_editor_lease)
    destination = PushDestinationProjection(
        "https",
        "push.example.test",
        443,
        "/team/notes.git",
        "refs/heads/main",
    )
    phases: tuple[tuple[str, _PushResult], ...] = (
        ("local_proof", _push_destination_policy_result("blocked")),
        ("preflight", PushPreflightResult("blocked")),
        ("push", PushExecutionResult("uncertain", push_outcome_copy("uncertain"))),
        (
            "recovery",
            push_recovery_copy(
                destination,
                RemoteRefObservation("parent", "a" * 40),
            ),
        ),
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        assert await workspace.open_path("folder/one.md")
        editor = workspace.query_one("#file-notes-editor")
        assert not editor.read_only

        for kind, result in phases:
            release = asyncio.Event()
            operation = service.retain_push_operation(
                binding,
                kind,
                result,
                release,
                child_started=kind == "push",
            )
            if kind == "recovery":
                snapshot = owner.snapshot(binding)
                assert workspace._rehydrate_push_state(
                    service,
                    binding,
                    replace(
                        snapshot,
                        push_recovery=result,
                        push_recovery_candidate=operation.candidate,
                    ),
                )
            else:
                assert workspace._rehydrate_git_presentation()
            assert not editor.read_only
            assert workspace._editor_read_only_leases == {}
            release.set()
            await operation.wait()
            await _until(
                pilot,
                lambda: workspace._push_result is result,
                f"{kind} settlement did not publish",
            )
            assert not editor.read_only
            assert workspace._editor_read_only_leases == {}

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_session_git_indicator_shows_hidden_push_checking(
    tmp_path: Path,
) -> None:
    """Dropping hidden rehydration must lose the persistent checking state."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    release = asyncio.Event()
    operation = service.retain_push_operation(
        binding,
        "local_proof",
        _push_destination_policy_result("blocked"),
        release,
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )

        assert "Push checking" in str(
            workspace.query_one("#file-notes-session-changes").label
        )
        assert workspace._navigator_mode == "files"
        assert service.push_review_calls == []
        assert service.push_query_calls == []
        assert service.push_calls == []
        release.set()
        await operation.wait()

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_back_to_files_keeps_push_publication_and_attention(
    tmp_path: Path,
) -> None:
    """Cancelling hidden observers must not erase an in-flight push result."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    _publish_candidate_on_owner(
        owner,
        binding,
        service.repository,
        parent_oid="a" * 40,
        candidate_oid="d" * 40,
    )
    release = asyncio.Event()
    result = PushExecutionResult("uncertain", push_outcome_copy("uncertain"))

    def publish_hidden_owner_state() -> None:
        assert owner.record_change(
            binding,
            SessionChange("created", "published-after-hide.md"),
        )

    operation = service.retain_push_operation(
        binding,
        "push",
        result,
        release,
        publish=publish_hidden_owner_state,
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        assert "Push checking" in str(
            workspace.query_one("#file-notes-session-changes").label
        )

        service.mark_push_child_started()
        await service.actual_child_started.wait()
        await _until(
            pilot,
            lambda: "Pushing" in str(
                workspace.query_one("#file-notes-session-changes").label
            ),
            "actual child start did not publish Pushing",
        )

        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace.query_one("#file-notes-git-back").press()
        await _until(
            pilot,
            lambda: workspace._navigator_mode == "files",
            "Back did not return to Files",
        )
        await workspace._git_panel_widget.remove()
        release.set()
        assert await operation.wait() is result
        await _until(
            pilot,
            lambda: workspace._push_phase == "needs_attention",
            "hidden settlement did not publish attention",
        )

        assert service.published_results == [result]
        assert any(
            change.change.relative_path == "published-after-hide.md"
            for change in owner.snapshot(binding).changes
        )
        assert service.cancel_push_calls == []
        assert "Push needs attention" in str(
            workspace.query_one("#file-notes-session-changes").label
        )

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_commit_success_resyncs_candidate_without_starting_push(
    tmp_path: Path,
) -> None:
    """Omitting post-commit resync must hide the newly published candidate."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        assert workspace._push_availability is None
        assert workspace._navigator_mode == "files"

        _publish_candidate_on_owner(
            owner,
            binding,
            service.repository,
            parent_oid="a" * 40,
            candidate_oid="d" * 40,
        )
        workspace._clear_commit_draft_after_success()

        assert workspace._push_availability == owner.snapshot(binding).push_candidate
        assert workspace._navigator_mode == "files"
        assert service.push_review_calls == []
        assert service.push_query_calls == []
        assert service.push_calls == []

    await workspace.shutdown()
    owner.shutdown()
    replica.close()
