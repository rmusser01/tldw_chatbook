from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

import tldw_chatbook.Notes.file_notes_git_service as git_service
from tldw_chatbook.Notes.file_notes_git_commit import CommitReviewResult
from tldw_chatbook.Notes.file_notes_git_service import (
    AsyncGitProcessRunner,
    FileNotesGitService,
    GitArg,
    GitCommandResult,
    GitRunCancelled,
    RetainedGitChildSettlement,
    RetainedGitChildToken,
)
from tldw_chatbook.Notes.file_notes_session_owner import (
    FileNotesSessionOwner,
    HeadIdentity,
    IndexBaseline,
    IndexEntry,
    SessionChange,
    StagingOwnership,
)


def _git(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test Author",
            "GIT_AUTHOR_EMAIL": "author@example.test",
            "GIT_COMMITTER_NAME": "Test Committer",
            "GIT_COMMITTER_EMAIL": "committer@example.test",
        },
    ).stdout


def _git_input(repository: Path, payload: bytes, *arguments: str) -> bytes:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        input=payload,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test Author",
            "GIT_AUTHOR_EMAIL": "author@example.test",
            "GIT_COMMITTER_NAME": "Test Committer",
            "GIT_COMMITTER_EMAIL": "committer@example.test",
        },
    ).stdout


def _init_repository(
    tmp_path: Path,
    *,
    note_path: str = "note.md",
) -> Path:
    repository = tmp_path / "notes"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "user@example.test")
    note = repository / note_path
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("baseline\n", encoding="utf-8")
    _git(repository, "add", note_path)
    _git(repository, "commit", "-q", "-m", "baseline")
    return repository


async def _stage_then_review(
    repository: Path,
    selected_root: Path,
    *,
    move: bool = False,
    recreate_source: bool = False,
) -> tuple[FileNotesGitService, object, CommitReviewResult]:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(selected_root)
    change = (
        SessionChange("moved", "note.md", "moved.md")
        if move
        else SessionChange("modified", "note.md")
    )
    assert owner.record_change(binding, change)
    service = FileNotesGitService(owner)
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert discovery.head is not None
    assert owner.publish_trust(binding, discovery.repository)

    note = selected_root / "note.md"
    if move:
        note.rename(selected_root / "moved.md")
    else:
        note.write_text("staged\n", encoding="utf-8")
    snapshot = owner.snapshot(binding)
    status = await service.start_status(binding, snapshot.changes)
    assert status.state == "ready"
    assert len(status.rows) == 1
    stage = await service.start_stage(binding, (status.rows[0].group_id,))
    assert stage.state == "success"
    if recreate_source:
        note.write_text("recreated source\n", encoding="utf-8")
    result = await service.start_commit_review(binding, "Review subject")
    return service, binding, result


class _RecordingRunner(AsyncGitProcessRunner):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[tuple[GitArg, ...], Mapping[str, str]]] = []

    async def run(
        self,
        argv: Sequence[GitArg],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
        stdout_limit: int | None = None,
        stderr_limit: int | None = None,
    ) -> GitCommandResult:
        self.calls.append((tuple(argv), dict(environment)))
        return await super().run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
        )


class _ControlledRetainedReviewRunner(_RecordingRunner):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.token = RetainedGitChildToken(git_service._RETAINED_CHILD_TOKEN_SECRET)
        self.exposed = asyncio.Event()
        self.terminal = asyncio.Event()
        self.settle_calls = 0
        self.claimed = False
        self.released = False
        self.shutdown_called = False
        self._intercepted = False

    def arm(self, mode: str) -> None:
        self.mode = mode
        self.exposed = asyncio.Event()
        self.terminal = asyncio.Event()
        self.claimed = False
        self.released = False
        self._intercepted = False

    async def run(
        self,
        argv: Sequence[GitArg],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
        stdout_limit: int | None = None,
        stderr_limit: int | None = None,
    ) -> GitCommandResult:
        if (
            self.mode != "passthrough"
            and "--no-replace-objects" in argv
            and not self._intercepted
        ):
            self._intercepted = True
            self.exposed.set()
            if self.mode == "cancelled_token":
                raise GitRunCancelled(retained_child=self.token)
            if self.mode == "cancelled_result":
                raise GitRunCancelled(
                    result=GitCommandResult(1, b"", b"terminal refusal")
                )
            if self.mode == "overflow":
                return GitCommandResult(
                    0,
                    f"{cwd}\n".encode(),
                    b"",
                    output_overflow=True,
                )
            return GitCommandResult(
                None,
                b"",
                b"",
                timed_out=True,
                termination_uncertain=True,
                retained_child=self.token,
            )
        return await super().run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
        )

    def read_retained_child(
        self,
        token: RetainedGitChildToken,
    ) -> RetainedGitChildSettlement:
        assert token is self.token
        if self.terminal.is_set():
            if self.shutdown_called:
                return RetainedGitChildSettlement(
                    "stop_requested",
                    returncode=-15,
                    stop_requested=True,
                )
            return RetainedGitChildSettlement("natural", returncode=1)
        return RetainedGitChildSettlement("alive")

    def claim_retained_child(self, token: RetainedGitChildToken) -> bool:
        assert token is self.token
        self.claimed = True
        return True

    async def settle_retained_child(
        self,
        token: RetainedGitChildToken,
        *,
        timeout: float = 0.0,
    ) -> RetainedGitChildSettlement:
        assert token is self.token
        self.settle_calls += 1
        if not self.terminal.is_set() and timeout > 0:
            try:
                await asyncio.wait_for(self.terminal.wait(), timeout)
            except TimeoutError:
                pass
        return self.read_retained_child(token)

    def release_retained_child(self, token: RetainedGitChildToken) -> bool:
        assert token is self.token
        if not self.terminal.is_set():
            return False
        self.released = True
        return True

    def shutdown(self):
        self.shutdown_called = True
        self.terminal.set()
        return super().shutdown()


async def _prepare_owned_review(
    repository: Path,
    *,
    stage_unrelated: bool = False,
    newer_owned_edit: bool = False,
    local_marker: str | None = None,
    replacement_reference: bool = False,
    promisor_repository: bool = False,
    partial_repository: bool = False,
    local_promisor_marker: bool = False,
    sparse_repository: bool = False,
    detach_head: bool = False,
    unsupported_index_state: str | None = None,
    owned_case: str = "modified",
    recreate_owned_source: bool = False,
    ignore_recreated_source: bool = False,
    unrelated_unstaged: bool = False,
    include_no_op_group: bool = False,
    runner: AsyncGitProcessRunner | None = None,
    environment: Mapping[str, str] | None = None,
    service_capture: list[tuple[FileNotesGitService, object]] | None = None,
) -> tuple[FileNotesGitService, object, CommitReviewResult]:
    if include_no_op_group:
        (repository / "no-op.md").write_text("unchanged\n", encoding="utf-8")
        _git(repository, "add", "no-op.md")
        _git(repository, "commit", "-q", "-m", "add no-op fixture")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository)
    change = (
        SessionChange("moved", "note.md", "moved.md")
        if owned_case == "move"
        else SessionChange(
            "deleted" if owned_case == "deletion" else "modified",
            "note.md",
        )
    )
    assert owner.record_change(binding, change)
    if include_no_op_group:
        assert owner.record_change(
            binding,
            SessionChange("modified", "no-op.md"),
        )
    service = FileNotesGitService(owner, runner=runner, environment=environment)
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert discovery.head is not None
    assert discovery.head.kind == "attached"
    assert owner.publish_trust(binding, discovery.repository)

    baseline_line = _git(repository, "ls-files", "-s", "--", "note.md").decode()
    baseline_mode, baseline_oid, _stage_and_path = baseline_line.split(" ", 2)
    baseline = IndexEntry("note.md", baseline_mode, baseline_oid)
    if owned_case == "deletion":
        (repository / "note.md").unlink()
        _git(repository, "add", "--all", "--", "note.md")
        staged: IndexEntry | None = None
        endpoints = ("note.md",)
        move_edges: tuple[tuple[str, str], ...] = ()
        current_path = "note.md"
        original_baselines = {"note.md": IndexBaseline(baseline)}
        post_stage_entries = {"note.md": None}
    elif owned_case == "move":
        (repository / "note.md").rename(repository / "moved.md")
        _git(repository, "add", "--all", "--", "note.md", "moved.md")
        staged_line = _git(
            repository,
            "ls-files",
            "-s",
            "--",
            "moved.md",
        ).decode()
        staged_mode, staged_oid, _stage_and_path = staged_line.split(" ", 2)
        staged = IndexEntry("moved.md", staged_mode, staged_oid)
        endpoints = ("note.md", "moved.md")
        move_edges = (("note.md", "moved.md"),)
        current_path = "moved.md"
        original_baselines = {
            "note.md": IndexBaseline(baseline),
            "moved.md": IndexBaseline(None),
        }
        post_stage_entries = {"note.md": None, "moved.md": staged}
    elif owned_case == "no_op":
        staged = baseline
        endpoints = ("note.md",)
        move_edges = ()
        current_path = "note.md"
        original_baselines = {"note.md": IndexBaseline(baseline)}
        post_stage_entries = {"note.md": baseline}
    else:
        (repository / "note.md").write_text("staged\n", encoding="utf-8")
        _git(repository, "add", "note.md")
        staged_line = _git(repository, "ls-files", "-s", "--", "note.md").decode()
        staged_mode, staged_oid, _stage_and_path = staged_line.split(" ", 2)
        staged = IndexEntry("note.md", staged_mode, staged_oid)
        endpoints = ("note.md",)
        move_edges = ()
        current_path = "note.md"
        original_baselines = {"note.md": IndexBaseline(baseline)}
        post_stage_entries = {"note.md": staged}
    if newer_owned_edit:
        (repository / current_path).write_text(
            "newer saved edit\n",
            encoding="utf-8",
        )
    if recreate_owned_source:
        if ignore_recreated_source:
            (repository / ".gitignore").write_text(
                "note.md\n",
                encoding="utf-8",
            )
        (repository / "note.md").write_text(
            "recreated newer source\n",
            encoding="utf-8",
        )
    if unrelated_unstaged:
        (repository / "unrelated-unstaged.md").write_text(
            "allowed\n",
            encoding="utf-8",
        )
    if stage_unrelated:
        (repository / "unrelated-secret.md").write_text(
            "unrelated\n",
            encoding="utf-8",
        )
        _git(repository, "add", "unrelated-secret.md")

    head = HeadIdentity.attached(
        discovery.head.branch or "",
        discovery.head.object_id or "",
    )
    ownership = StagingOwnership(
        repository=discovery.repository,
        head=head,
        approved_endpoint_topology=endpoints,
        approved_move_edges=move_edges,
        approved_current_path=current_path,
        original_baselines=original_baselines,
        post_stage_entries=post_stage_entries,
    )
    ownership_by_group = {1: ownership}
    if include_no_op_group:
        no_op_line = _git(
            repository,
            "ls-files",
            "-s",
            "--",
            "no-op.md",
        ).decode()
        no_op_mode, no_op_oid, _stage_and_path = no_op_line.split(" ", 2)
        no_op_entry = IndexEntry("no-op.md", no_op_mode, no_op_oid)
        ownership_by_group[2] = StagingOwnership(
            repository=discovery.repository,
            head=head,
            approved_endpoint_topology=("no-op.md",),
            approved_move_edges=(),
            approved_current_path="no-op.md",
            original_baselines={"no-op.md": IndexBaseline(no_op_entry)},
            post_stage_entries={"no-op.md": no_op_entry},
        )
    assert owner.publish_ownership(binding, ownership_by_group)
    if local_marker is not None:
        marker_path = repository / ".git" / local_marker
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("block\n", encoding="utf-8")
    if replacement_reference:
        tree = _git(repository, "rev-parse", "HEAD^{tree}").decode().strip()
        replacement = (
            _git(
                repository,
                "commit-tree",
                tree,
                "-p",
                head.object_id or "",
                "-m",
                "replacement",
            )
            .decode()
            .strip()
        )
        _git(repository, "replace", head.object_id or "", replacement)
    if promisor_repository:
        _git(repository, "config", "remote.origin.promisor", "true")
    if partial_repository:
        _git(repository, "config", "extensions.partialClone", "origin")
    if local_promisor_marker:
        marker = repository / ".git" / "objects" / "pack" / "local.promisor"
        marker.touch()
    if sparse_repository:
        _git(repository, "config", "core.sparseCheckout", "true")
    if detach_head:
        _git(repository, "checkout", "--detach", "-q")
    if unsupported_index_state == "intent":
        (repository / "intent.md").write_text("intent\n", encoding="utf-8")
        _git(repository, "add", "--intent-to-add", "intent.md")
    elif unsupported_index_state == "conflict":
        _git(repository, "update-index", "--force-remove", "note.md")
        _git_input(
            repository,
            (
                f"{baseline.mode} {baseline.object_id} 1\tnote.md\n"
                f"{baseline.mode} {baseline.object_id} 2\tnote.md\n"
                f"{staged.mode} {staged.object_id} 3\tnote.md\n"
            ).encode(),
            "update-index",
            "--index-info",
        )
    elif unsupported_index_state == "gitlink":
        _git(
            repository,
            "update-index",
            "--add",
            "--cacheinfo",
            "160000",
            head.object_id or "",
            "linked-repository",
        )
    elif unsupported_index_state == "semantic":
        _git(repository, "update-index", "--skip-worktree", "note.md")
    if service_capture is not None:
        service_capture.append((service, binding))
    result = await service.start_commit_review(binding, "Review subject", "Body")
    return service, binding, result


@pytest.mark.asyncio
async def test_commit_review_attached_repository_returns_sanitized_projection(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)

    service, _binding, result = await _prepare_owned_review(repository)

    assert result.state == "ready"
    assert result.handle is not None
    assert result.projection is not None
    assert result.projection.message == "Review subject\n\nBody\n"
    assert result.projection.included_note_count == 1
    assert result.projection.included_notes[0].display_text == "note.md"
    assert result.projection.branch.startswith("refs/heads/")
    assert result.projection.hooks_bypassed is True
    assert result.projection.unsigned is True
    assert "unrelated" not in repr(result)
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_review_blocked_message_supersedes_ready_capability(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)

    service, binding, first = await _prepare_owned_review(repository)
    assert first.state == "ready"
    assert len(service._commit_review_snapshots) == 1

    second = await service.start_commit_review(binding, "invalid\nsubject")

    assert second.state == "blocked"
    assert second.handle is None
    assert second.projection is None
    assert service._commit_review_snapshots == {}
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_review_admission_refusal_preserves_ready_capability(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)

    service, binding, first = await _prepare_owned_review(repository)
    assert first.state == "ready"
    ready_snapshots = dict(service._commit_review_snapshots)
    admission = service._owner.admit_mutation(binding)
    assert admission.lease is not None
    try:
        with pytest.raises(
            git_service.GitMutationAdmissionError,
            match="admission was refused",
        ):
            service.start_commit_review(binding, "Second review")
        assert service._commit_review_snapshots == ready_snapshots
    finally:
        admission.lease.release()

    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_review_retained_cancellation_supersedes_ready_capability(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledRetainedReviewRunner("passthrough")

    service, binding, first = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert first.state == "ready"
    assert len(service._commit_review_snapshots) == 1

    runner.arm("cancelled_token")
    second_waiter = service.start_commit_review(binding, "Second review")
    await asyncio.wait_for(runner.exposed.wait(), 1.0)
    try:
        assert service._commit_review_snapshots == {}
    finally:
        runner.terminal.set()
        second = await asyncio.wait_for(second_waiter, 1.0)

    assert second.state == "cancelled"
    assert second.handle is None
    assert second.projection is None
    assert service._commit_review_snapshots == {}
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_review_proof_output_overflow_blocks_without_payload(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledRetainedReviewRunner("overflow")

    service, _binding, result = await _prepare_owned_review(
        repository,
        runner=runner,
    )

    assert result.state == "blocked"
    assert result.handle is None
    assert result.projection is None
    assert result.message == "Repository identity changed."
    assert str(repository) not in repr(result)
    assert service._commit_review_snapshots == {}
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_review_runs_security_proof_in_required_order(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _RecordingRunner()

    service, _binding, result = await _prepare_owned_review(
        repository,
        runner=runner,
    )

    assert result.state == "ready"
    review_calls = [
        (argv, environment)
        for argv, environment in runner.calls
        if "--no-replace-objects" in argv
    ]
    commands = [argv for argv, _environment in review_calls]
    assert [argv[2] for argv in commands[:3]] == [
        "rev-parse",
        "rev-parse",
        "rev-parse",
    ]
    positions = {
        command: next(index for index, argv in enumerate(commands) if command in argv)
        for command in (
            "config",
            "symbolic-ref",
            "ls-files",
            "diff-index",
            "write-tree",
            "status",
        )
    }
    assert positions["config"] < positions["symbolic-ref"]
    assert positions["symbolic-ref"] < positions["ls-files"]
    assert positions["ls-files"] < positions["diff-index"]
    assert positions["diff-index"] < positions["write-tree"]
    assert positions["write-tree"] < positions["status"]
    identity_positions = [index for index, argv in enumerate(commands) if "var" in argv]
    assert identity_positions
    assert positions["status"] < min(identity_positions)
    assert all(
        environment["GIT_NO_LAZY_FETCH"] == "1" for _argv, environment in review_calls
    )
    assert not any("commit" in argv for argv in commands)
    await service.shutdown()


@pytest.mark.asyncio
async def test_complete_commit_proof_blocks_unrelated_staged_without_disclosure(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)

    service, binding, result = await _prepare_owned_review(
        repository,
        stage_unrelated=True,
    )

    assert result.state == "blocked"
    assert result.handle is None
    assert result.projection is None
    assert "unrelated-secret" not in repr(result)
    assert "unrelated-secret" not in repr(service._owner.snapshot(binding))
    await service.shutdown()


@pytest.mark.asyncio
async def test_complete_commit_proof_blocks_newer_included_worktree_edit(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)

    service, _binding, result = await _prepare_owned_review(
        repository,
        newer_owned_edit=True,
    )

    assert result.state == "blocked"
    assert result.handle is None
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", ["MERGE_HEAD", "index.lock", "info/grafts"])
async def test_commit_review_local_blockers_run_before_object_proof(
    tmp_path: Path,
    marker: str,
) -> None:
    repository = _init_repository(tmp_path)

    runner = _RecordingRunner()
    service, _binding, result = await _prepare_owned_review(
        repository,
        local_marker=marker,
        runner=runner,
    )

    assert result.state == "blocked"
    review_commands = [
        argv for argv, _environment in runner.calls if "--no-replace-objects" in argv
    ]
    assert not any(
        command in argv
        for argv in review_commands
        for command in ("symbolic-ref", "ls-files", "diff-index", "write-tree", "var")
    )
    assert not any("commit" in argv for argv in review_commands)
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_review_blocks_replacement_reference(tmp_path: Path) -> None:
    repository = _init_repository(tmp_path)

    service, _binding, result = await _prepare_owned_review(
        repository,
        replacement_reference=True,
    )

    assert result.state == "blocked"
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_review_blocks_promisor_repository_without_fetch(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    sentinel = tmp_path / "network-was-used"
    _git(
        repository,
        "config",
        "remote.origin.uploadpack",
        f"touch {sentinel}",
    )
    runner = _RecordingRunner()

    service, _binding, result = await _prepare_owned_review(
        repository,
        promisor_repository=True,
        runner=runner,
    )

    assert result.state == "blocked"
    assert not sentinel.exists()
    review_commands = [
        argv for argv, _environment in runner.calls if "--no-replace-objects" in argv
    ]
    assert not any(
        command in argv
        for argv in review_commands
        for command in ("symbolic-ref", "ls-files", "diff-index", "write-tree", "var")
    )
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "repository_flag",
    ["partial_repository", "local_promisor_marker", "sparse_repository"],
)
async def test_commit_review_blocks_unsupported_local_repository_formats(
    tmp_path: Path,
    repository_flag: str,
) -> None:
    repository = _init_repository(tmp_path)

    service, _binding, result = await _prepare_owned_review(
        repository,
        **{repository_flag: True},
    )

    assert result.state == "blocked"
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_review_blocks_detached_head(tmp_path: Path) -> None:
    repository = _init_repository(tmp_path)

    service, _binding, result = await _prepare_owned_review(
        repository,
        detach_head=True,
    )

    assert result.state == "blocked"
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_review_blocks_unborn_head(tmp_path: Path) -> None:
    repository = tmp_path / "unborn"
    repository.mkdir()
    _git(repository, "init", "-q")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository)
    assert owner.record_change(binding, SessionChange("created", "note.md"))
    service = FileNotesGitService(owner)
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert discovery.head is not None
    assert discovery.head.kind == "unborn"
    assert owner.publish_trust(binding, discovery.repository)

    result = await service.start_commit_review(binding, "Subject")

    assert result.state == "blocked"
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_review_blocks_missing_identity(tmp_path: Path) -> None:
    repository = _init_repository(tmp_path)
    _git(repository, "config", "--unset", "user.name")
    _git(repository, "config", "--unset", "user.email")
    _git(repository, "config", "user.useConfigOnly", "true")
    isolated_home = tmp_path / "isolated-home"
    isolated_home.mkdir()

    service, _binding, result = await _prepare_owned_review(
        repository,
        environment={
            "PATH": os.environ["PATH"],
            "HOME": str(isolated_home),
        },
    )

    assert result.state == "blocked"
    assert result.message is not None
    assert "user.name" in result.message
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsupported_index_state",
    ["intent", "conflict", "gitlink", "semantic"],
)
async def test_complete_commit_proof_blocks_unsupported_index_states(
    tmp_path: Path,
    unsupported_index_state: str,
) -> None:
    repository = _init_repository(tmp_path)

    service, _binding, result = await _prepare_owned_review(
        repository,
        unsupported_index_state=unsupported_index_state,
    )

    assert result.state == "blocked"
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("owned_case", ["deletion", "move"])
async def test_complete_commit_proof_blocks_recreated_owned_source(
    tmp_path: Path,
    owned_case: str,
) -> None:
    repository = _init_repository(tmp_path)

    service, _binding, result = await _prepare_owned_review(
        repository,
        owned_case=owned_case,
        recreate_owned_source=True,
    )

    assert result.state == "blocked"
    assert result.handle is None
    await service.shutdown()


@pytest.mark.asyncio
async def test_complete_commit_proof_blocks_ignored_recreated_deletion(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)

    service, _binding, result = await _prepare_owned_review(
        repository,
        owned_case="deletion",
        recreate_owned_source=True,
        ignore_recreated_source=True,
    )

    assert result.state == "blocked"
    assert result.handle is None
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("nested_root", [False, True])
async def test_commit_review_after_real_stage_supports_selected_root_mapping(
    tmp_path: Path,
    nested_root: bool,
) -> None:
    note_path = "notes/note.md" if nested_root else "note.md"
    repository = _init_repository(tmp_path, note_path=note_path)
    selected_root = repository / "notes" if nested_root else repository

    service, _binding, result = await _stage_then_review(
        repository,
        selected_root,
    )

    assert result.state == "ready"
    assert result.projection is not None
    assert result.projection.included_note_count == 1
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("recreate_source", [False, True])
async def test_commit_review_nested_real_stage_move_freshness(
    tmp_path: Path,
    recreate_source: bool,
) -> None:
    repository = _init_repository(tmp_path, note_path="notes/note.md")
    selected_root = repository / "notes"

    service, _binding, result = await _stage_then_review(
        repository,
        selected_root,
        move=True,
        recreate_source=recreate_source,
    )

    assert result.state == ("blocked" if recreate_source else "ready")
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("collision_state", ["modified", "untracked"])
async def test_commit_review_nested_root_name_collision_is_unrelated(
    tmp_path: Path,
    collision_state: str,
) -> None:
    repository = _init_repository(tmp_path, note_path="notes/note.md")
    root_note = repository / "note.md"
    if collision_state == "modified":
        root_note.write_text("root baseline\n", encoding="utf-8")
        _git(repository, "add", "note.md")
        _git(repository, "commit", "-q", "-m", "root baseline")
    root_note.write_text("unrelated root change\n", encoding="utf-8")

    service, _binding, result = await _stage_then_review(
        repository,
        repository / "notes",
    )

    assert result.state == "ready"
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_review_nested_chained_move_recreation_blocks(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path, note_path="notes/note.md")
    selected_root = repository / "notes"
    owner = FileNotesSessionOwner()
    binding = owner.select_root(selected_root)
    assert owner.record_change(
        binding,
        SessionChange("moved", "note.md", "intermediate.md"),
    )
    assert owner.record_change(
        binding,
        SessionChange("moved", "intermediate.md", "moved.md"),
    )
    service = FileNotesGitService(owner)
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)

    (selected_root / "note.md").rename(selected_root / "intermediate.md")
    (selected_root / "intermediate.md").rename(selected_root / "moved.md")
    snapshot = owner.snapshot(binding)
    status = await service.start_status(binding, snapshot.changes)
    assert status.state == "ready"
    assert len(status.rows) == 1
    stage = await service.start_stage(binding, (status.rows[0].group_id,))
    assert stage.state == "success"
    (selected_root / "intermediate.md").write_text(
        "recreated intermediate\n",
        encoding="utf-8",
    )

    result = await service.start_commit_review(binding, "Review subject")

    assert result.state == "blocked"
    await service.shutdown()


@pytest.mark.asyncio
async def test_complete_commit_proof_accepts_exact_move_topology(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)

    service, _binding, result = await _prepare_owned_review(
        repository,
        owned_case="move",
    )

    assert result.state == "ready"
    assert result.projection is not None
    assert result.projection.included_note_count == 1
    assert result.projection.included_notes[0].display_text == ("note.md → moved.md")
    await service.shutdown()


@pytest.mark.asyncio
async def test_complete_commit_proof_allows_unrelated_unstaged_path(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)

    service, _binding, result = await _prepare_owned_review(
        repository,
        unrelated_unstaged=True,
    )

    assert result.state == "ready"
    assert "unrelated-unstaged" not in repr(result)
    await service.shutdown()


@pytest.mark.asyncio
async def test_complete_commit_proof_excludes_no_op_ownership_group(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)

    service, _binding, result = await _prepare_owned_review(
        repository,
        include_no_op_group=True,
    )

    assert result.state == "ready"
    assert result.projection is not None
    assert result.projection.included_note_count == 1
    assert tuple(note.display_text for note in result.projection.included_notes) == (
        "note.md",
    )
    await service.shutdown()


@pytest.mark.asyncio
async def test_complete_commit_proof_no_op_ownership_cannot_authorize_empty_commit(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)

    service, _binding, result = await _prepare_owned_review(
        repository,
        owned_case="no_op",
    )

    assert result.state == "blocked"
    assert result.handle is None
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["uncertain_result", "cancelled_token"])
async def test_commit_review_retained_proof_child_holds_mutation_until_drained(
    tmp_path: Path,
    mode: str,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledRetainedReviewRunner(mode)
    captured: list[tuple[FileNotesGitService, object]] = []
    preparation = asyncio.create_task(
        _prepare_owned_review(
            repository,
            runner=runner,
            service_capture=captured,
        )
    )
    await asyncio.wait_for(runner.exposed.wait(), 1.0)
    service, binding = captured[0]

    blocked_admission = service._owner.admit_mutation(binding)
    assert blocked_admission.reason == "mutation_active"
    assert not preparation.done()

    runner.terminal.set()
    _service, _binding, result = await asyncio.wait_for(preparation, 1.0)

    assert result.state in {"blocked", "cancelled"}
    assert runner.claimed is True
    assert runner.settle_calls >= 1
    assert runner.released is True
    admitted = service._owner.admit_mutation(binding)
    assert admitted.lease is not None
    admitted.lease.release()
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_review_terminal_cancellation_result_is_preserved(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledRetainedReviewRunner("cancelled_result")

    service, _binding, result = await _prepare_owned_review(
        repository,
        runner=runner,
    )

    assert result.state == "blocked"
    assert runner.settle_calls == 0
    assert runner.released is False
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_review_caller_cancellation_keeps_proof_child_owned(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledRetainedReviewRunner("uncertain_result")
    captured: list[tuple[FileNotesGitService, object]] = []
    preparation = asyncio.create_task(
        _prepare_owned_review(
            repository,
            runner=runner,
            service_capture=captured,
        )
    )
    await asyncio.wait_for(runner.exposed.wait(), 1.0)
    service, binding = captured[0]

    preparation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await preparation

    assert service._owner.admit_mutation(binding).reason == "mutation_active"
    cycle = service._commit_review_cycle
    assert cycle is not None
    runner.terminal.set()
    result = await asyncio.wait_for(asyncio.shield(cycle), 1.0)

    assert result.state == "blocked"
    assert runner.released is True
    admitted = service._owner.admit_mutation(binding)
    assert admitted.lease is not None
    admitted.lease.release()
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_review_shutdown_stops_and_drains_proof_child(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledRetainedReviewRunner("uncertain_result")
    captured: list[tuple[FileNotesGitService, object]] = []
    preparation = asyncio.create_task(
        _prepare_owned_review(
            repository,
            runner=runner,
            service_capture=captured,
        )
    )
    await asyncio.wait_for(runner.exposed.wait(), 1.0)
    service, binding = captured[0]

    await asyncio.wait_for(service.shutdown(), 1.0)
    _service, _binding, result = await asyncio.wait_for(preparation, 1.0)

    assert result.state == "blocked"
    assert runner.released is True
    assert runner.shutdown_called is True
    assert service._owner.mutation_active(binding) is False
