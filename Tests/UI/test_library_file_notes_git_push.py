"""Workspace-state tests for guarded File Notes push rehydration."""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

sys.modules.setdefault("parakeet_mlx", types.ModuleType("parakeet_mlx"))

from Tests.Notes.test_file_notes_git_push_service import (  # noqa: E402
    _publish_candidate_on_owner,
)
from Tests.UI.test_library_file_notes_git import (  # noqa: E402
    _FakeGitService,
    _RemountWorkspaceHarness,
    _WorkspaceHarness,
    _row,
)
from tldw_chatbook.Notes.file_notes_git_push import (  # noqa: E402
    PushDestinationProjection,
    PushDestinationPolicyResult,
    PushRecoveryProjection,
    RemoteRefObservation,
    _push_destination_policy_result,
    push_outcome_copy,
    push_recovery_copy,
)
from tldw_chatbook.Notes.file_notes_git_service import (  # noqa: E402
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

_PushResult = (
    PushDestinationPolicyResult
    | PushPreflightResult
    | PushExecutionResult
    | PushRecoveryProjection
)


class _PushGitService(_FakeGitService):
    """Deterministic retained-operation fake for workspace push state."""

    def __init__(self, owner, rows) -> None:
        super().__init__(owner, rows)
        self.push_review_calls: list[SessionBinding] = []
        self.push_query_calls: list[SessionBinding] = []
        self.push_calls: list[SessionBinding] = []
        self.cancel_push_calls: list[SessionBinding] = []
        self.published_results: list[_PushResult] = []
        self.actual_child_started = asyncio.Event()
        self._push_operation: RetainedPushOperation | None = None
        self._push_child_signal: asyncio.Future[bool] | None = None
        self._push_operation_generation = 0

    def retain_push_operation(
        self,
        binding: SessionBinding,
        kind: str,
        result: _PushResult,
        release: asyncio.Event,
        *,
        child_started: bool = False,
        candidate: PushCandidateAvailability | None = None,
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

    def start_push_review(self, binding: SessionBinding) -> None:
        self.push_review_calls.append(binding)

    def start_push(self, binding: SessionBinding) -> None:
        self.push_calls.append(binding)

    def check_push_again(self, binding: SessionBinding) -> None:
        self.push_query_calls.append(binding)

    def cancel_push(self, binding: SessionBinding, _operation) -> bool:
        self.cancel_push_calls.append(binding)
        return False


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


async def _until(pilot, predicate, message: str) -> None:
    """Drain deterministic Textual turns until one event-owned fact holds."""
    for _ in range(100):
        if predicate():
            return
        await pilot.pause()
    raise AssertionError(message)


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


@pytest.mark.asyncio
async def test_workspace_rehydrate_hidden_settlement_without_duplicate_work(
    tmp_path: Path,
) -> None:
    """Remount must reuse the exact observer instead of launching Git again."""
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

    async with app.run_test() as pilot:
        await _until(
            pilot,
            lambda: workspace.initialized,
            "workspace initialization did not settle",
        )
        observer = workspace._push_observer_task
        assert observer is not None
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
        assert service.push_review_calls == []
        assert service.push_query_calls == []
        assert service.push_calls == []

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
