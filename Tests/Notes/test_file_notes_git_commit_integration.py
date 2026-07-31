from __future__ import annotations

import asyncio
import os
import shlex
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

import tldw_chatbook.Notes.file_notes_git_service as git_service
from tldw_chatbook.Notes.file_notes_git_commit import (
    CommitReviewHandle,
    CommitReviewResult,
    GitIdentity,
    parse_raw_commit_object,
)
from tldw_chatbook.Notes.file_notes_git_service import (
    AsyncGitProcessRunner,
    FileNotesGitService,
    GitArg,
    GitCommandResult,
    GitRunCancelled,
    RetainedGitChildSettlement,
    RetainedGitChildToken,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.file_notes_session_owner import (
    CommitPublicationResult,
    FileNotesSessionOwner,
    HeadIdentity,
    IndexBaseline,
    IndexEntry,
    SessionChange,
    StagingOwnership,
)

_ASYNC_SETTLE_TIMEOUT = 10.0


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
        on_spawn: Callable[[], None] | None = None,
        cancel_before_spawn: bool = False,
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
            on_spawn=on_spawn,
            cancel_before_spawn=cancel_before_spawn,
        )


class _NoNetworkRunner(_RecordingRunner):
    """Fail the test if the guarded flow launches a network-capable Git child."""

    _ALLOWED_GUARDED_REVIEW_COMMANDS = frozenset(
        {
            ("rev-parse", "--path-format=absolute", "--show-toplevel"),
            ("rev-parse", "--absolute-git-dir"),
            ("rev-parse", "--path-format=absolute", "--git-common-dir"),
            ("config", "--includes", "--local", "--null", "--list"),
            ("config", "--includes", "--worktree", "--null", "--list"),
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self._guarded_review_armed = False

    def arm(self) -> None:
        """Start fail-closed inspection immediately before guarded review."""
        self.calls.clear()
        self._guarded_review_armed = True

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
        on_spawn: Callable[[], None] | None = None,
        cancel_before_spawn: bool = False,
    ) -> GitCommandResult:
        decoded = tuple(os.fsdecode(value) for value in argv)
        if self._guarded_review_armed:
            assert environment.get("GIT_NO_LAZY_FETCH") == "1", (
                "guarded review command lacks no-lazy-fetch isolation"
            )
            assert len(decoded) >= 3 and decoded[1] == "--no-replace-objects", (
                "guarded review command lacks replacement-ref isolation"
            )
            assert decoded[2:] in self._ALLOWED_GUARDED_REVIEW_COMMANDS, (
                f"guarded review command is not local-only: {decoded[2:]!r}"
            )
        return await super().run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
            on_spawn=on_spawn,
            cancel_before_spawn=cancel_before_spawn,
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
        on_spawn: Callable[[], None] | None = None,
        cancel_before_spawn: bool = False,
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
            on_spawn=on_spawn,
            cancel_before_spawn=cancel_before_spawn,
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


class _ControlledCommitRunner(_RecordingRunner):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.token = RetainedGitChildToken(git_service._RETAINED_CHILD_TOKEN_SECRET)
        self.commit_admitted = asyncio.Event()
        self.commit_started = asyncio.Event()
        self.release_commit = asyncio.Event()
        self.commit_calls = 0
        self.hooks_directory: Path | None = None
        self.claimed = False
        self.terminal = mode == "retained_natural_failure"
        self.released = False
        self.shutdown_called = False

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
        on_spawn: Callable[[], None] | None = None,
        cancel_before_spawn: bool = False,
    ) -> GitCommandResult:
        if not any(
            isinstance(value, str) and value.startswith("core.hooksPath=")
            for value in argv
        ):
            return await super().run(
                argv,
                cwd=cwd,
                environment=environment,
                stdin=stdin,
                timeout=timeout,
                stdout_limit=stdout_limit,
                stderr_limit=stderr_limit,
                on_spawn=on_spawn,
                cancel_before_spawn=cancel_before_spawn,
            )
        self.calls.append((tuple(argv), dict(environment)))
        self.commit_calls += 1
        assert cancel_before_spawn
        self.commit_admitted.set()
        hooks_argument = next(
            value
            for value in argv
            if isinstance(value, str) and value.startswith("core.hooksPath=")
        )
        self.hooks_directory = Path(hooks_argument.split("=", 1)[1])
        assert self.hooks_directory.is_dir()
        assert not any(self.hooks_directory.iterdir())
        assert stat.S_IMODE(self.hooks_directory.stat().st_mode) == 0o700
        assert self.hooks_directory.stat().st_dev == Path(cwd).stat().st_dev
        assert not self.hooks_directory.is_relative_to(Path(cwd))
        def mark_spawned() -> None:
            if on_spawn is not None:
                on_spawn()
            self.commit_started.set()

        uses_real_child = self.mode in {
            "pause_then_commit",
            "commit_then_branch_drift",
            "commit_then_index_drift",
            "commit_then_worktree_edit",
        }
        if not uses_real_child:
            mark_spawned()
        if self.mode == "failure":
            return GitCommandResult(1, b"", b"commit refused")
        if self.mode == "zero_without_commit":
            return GitCommandResult(0, b"", b"")
        if self.mode == "signal":
            return GitCommandResult(-9, b"", b"terminated by signal")
        if self.mode in {"terminal_stop", "terminal_force"}:
            return GitCommandResult(
                -9 if self.mode == "terminal_force" else -15,
                b"",
                b"commit child was stopped",
                termination_uncertain=True,
                stop_requested=True,
                force_stopped=self.mode == "terminal_force",
            )
        if self.mode == "oserror":
            raise OSError("commit launch outcome is unknown")
        if self.mode in {
            "uncertain",
            "uncertain_signal",
            "retained_natural_failure",
            "uncertain_confirmed_shutdown",
            "uncertain_unconfirmed_shutdown",
        }:
            return GitCommandResult(
                None,
                b"",
                b"",
                timed_out=True,
                termination_uncertain=True,
                retained_child=self.token,
            )
        if self.mode == "pause":
            await self.release_commit.wait()
            return GitCommandResult(1, b"", b"commit refused")
        if self.mode == "pause_zero_without_commit":
            await self.release_commit.wait()
            return GitCommandResult(0, b"", b"")
        if self.mode == "pause_then_commit":
            await self.release_commit.wait()
            return await AsyncGitProcessRunner.run(
                self,
                argv,
                cwd=cwd,
                environment=environment,
                stdin=stdin,
                timeout=timeout,
                stdout_limit=stdout_limit,
                stderr_limit=stderr_limit,
                on_spawn=mark_spawned,
                cancel_before_spawn=cancel_before_spawn,
            )
        if self.mode == "shutdown_stop":
            await self.release_commit.wait()
            return GitCommandResult(
                -15,
                b"",
                b"stopped during shutdown",
                termination_uncertain=True,
                retained_child=self.token,
                stop_requested=True,
            )
        if self.mode in {
            "commit_then_branch_drift",
            "commit_then_index_drift",
            "commit_then_worktree_edit",
        }:
            result = await AsyncGitProcessRunner.run(
                self,
                argv,
                cwd=cwd,
                environment=environment,
                stdin=stdin,
                timeout=timeout,
                stdout_limit=stdout_limit,
                stderr_limit=stderr_limit,
                on_spawn=mark_spawned,
                cancel_before_spawn=cancel_before_spawn,
            )
            if self.mode == "commit_then_branch_drift":
                _git(Path(cwd), "checkout", "-q", "-b", "unexpected")
            elif self.mode == "commit_then_index_drift":
                (Path(cwd) / "unexpected.md").write_text(
                    "unexpected staged content\n",
                    encoding="utf-8",
                )
                _git(Path(cwd), "add", "unexpected.md")
            else:
                (Path(cwd) / "note.md").write_text(
                    "newer worktree edit\n",
                    encoding="utf-8",
                )
            return result
        raise AssertionError(f"Unsupported controlled commit mode: {self.mode}")

    def read_retained_child(
        self,
        token: RetainedGitChildToken,
    ) -> RetainedGitChildSettlement:
        assert token is self.token
        if self.mode == "shutdown_stop" and self.terminal:
            return RetainedGitChildSettlement(
                "stop_requested",
                returncode=-15,
                stop_requested=True,
            )
        if self.terminal:
            return RetainedGitChildSettlement(
                "natural",
                returncode=-9 if self.mode == "uncertain_signal" else 1,
            )
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
        del timeout
        return self.read_retained_child(token)

    def release_retained_child(self, token: RetainedGitChildToken) -> bool:
        assert token is self.token
        if not self.terminal:
            return False
        self.released = True
        return True

    def shutdown(self):
        self.shutdown_called = True
        if self.mode in {
            "uncertain_confirmed_shutdown",
            "uncertain_unconfirmed_shutdown",
        }:
            confirmed = self.mode == "uncertain_confirmed_shutdown"
            self.terminal = confirmed

            async def settle_shutdown() -> bool:
                await asyncio.sleep(0)
                return confirmed

            return git_service._RetainedSettlement(
                asyncio.create_task(settle_shutdown())
            )
        if self.mode == "shutdown_stop":
            self.terminal = True
            self.release_commit.set()
        return super().shutdown()


async def _prepare_owned_review(
    repository: Path,
    *,
    stage_unrelated: bool = False,
    newer_owned_edit: bool = False,
    local_marker: str | None = None,
    replacement_reference: bool = False,
    promisor_repository: bool = False,
    worktree_promisor_repository: bool = False,
    included_promisor_repository: bool = False,
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
    pre_review_state: dict[str, bytes] | None = None,
    arm_runner_before_review: bool = False,
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
    if worktree_promisor_repository:
        _git(repository, "config", "extensions.worktreeConfig", "true")
        _git(
            repository,
            "config",
            "--worktree",
            "remote.origin.promisor",
            "true",
        )
    if included_promisor_repository:
        include_path = repository / ".git" / "promisor.inc"
        include_path.write_text(
            '[remote "origin"]\n\tpromisor\n',
            encoding="utf-8",
        )
        _git(repository, "config", "--local", "include.path", "promisor.inc")
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
    elif unsupported_index_state == "assume":
        _git(repository, "update-index", "--assume-unchanged", "note.md")
    if service_capture is not None:
        service_capture.append((service, binding))
    if pre_review_state is not None:
        pre_review_state["head"] = _git(repository, "rev-parse", "HEAD")
        pre_review_state["index"] = _git(
            repository,
            "ls-files",
            "-z",
            "--stage",
            "-v",
        )
    if arm_runner_before_review:
        assert isinstance(runner, _NoNetworkRunner)
        runner.arm()
    result = await service.start_commit_review(binding, "Review subject", "Body")
    return service, binding, result


async def _prepare_uncertain_commit_recovery(
    repository: Path,
    *,
    mode: str,
) -> tuple[
    FileNotesGitService,
    object,
    CommitReviewResult,
    _ControlledCommitRunner,
]:
    """Create one exact quarantined commit attempt for recovery tests."""
    runner = _ControlledCommitRunner(mode)
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None
    outcome = await service.start_commit(binding, review.handle)
    assert outcome.state == "uncertain"
    assert runner.commit_calls == 1
    return service, binding, review, runner


async def _stage_changes_then_review(
    repository: Path,
    changes: Sequence[SessionChange],
    *,
    runner: AsyncGitProcessRunner | None = None,
    environment: Mapping[str, str] | None = None,
    reset_runner_before_review: bool = False,
) -> tuple[FileNotesGitService, object, CommitReviewResult]:
    """Stage real session changes and create one guarded commit review."""
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository)
    for change in changes:
        assert owner.record_change(binding, change)
    service = FileNotesGitService(owner, runner=runner, environment=environment)
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert discovery.head is not None
    assert discovery.head.kind == "attached"
    assert owner.publish_trust(binding, discovery.repository)
    status = await service.start_status(binding, owner.snapshot(binding).changes)
    assert status.state == "ready"
    group_ids = tuple(row.group_id for row in status.rows)
    assert group_ids
    staged = await service.start_stage(binding, group_ids)
    assert staged.state == "success"
    assert set(staged.staged_group_ids) == set(group_ids)
    if reset_runner_before_review:
        assert isinstance(runner, _RecordingRunner)
        runner.calls.clear()
    review = await service.start_commit_review(
        binding,
        "Repository matrix",
        "Exact staged session state",
    )
    return service, binding, review


def _commit_reviewed_index(
    repository: Path,
    review: CommitReviewResult,
) -> str:
    """Create the exact reviewed commit outside Chatbook for recovery proof."""
    projection = review.projection
    assert projection is not None
    subprocess.run(
        (
            "git",
            "--no-replace-objects",
            "-c",
            "commit.gpgSign=false",
            "commit",
            "--no-gpg-sign",
            "--cleanup=verbatim",
            "-F",
            "-",
        ),
        cwd=repository,
        input=projection.message.encode("utf-8"),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": projection.author.name,
            "GIT_AUTHOR_EMAIL": projection.author.email,
            "GIT_COMMITTER_NAME": projection.committer.name,
            "GIT_COMMITTER_EMAIL": projection.committer.email,
        },
    )
    return _git(repository, "rev-parse", "HEAD").decode().strip()


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
async def test_candidate_publication_review_snapshot_retains_provenance_seed(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)

    service, _binding, result = await _prepare_owned_review(repository)

    assert result.handle is not None
    snapshot = service._commit_review_snapshots[result.handle._token]
    seed = snapshot.candidate_seed
    assert not hasattr(seed, "guarded_commit_capture")
    assert (
        seed._guarded_commit_identity
        is snapshot.capture._guarded_commit_identity
    )
    assert seed.subject == "Review subject"
    assert tuple(note.display_text for note in seed.included_notes) == ("note.md",)
    assert seed.change_types == ("Modified",)
    assert "staged\n" not in repr(seed)
    await service.shutdown()


@pytest.mark.asyncio
async def test_candidate_publication_immediate_success_uses_owner_locked_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, review = await _prepare_owned_review(repository)
    assert review.handle is not None
    review_snapshot = service._commit_review_snapshots[review.handle._token]
    seed = review_snapshot.candidate_seed
    original_publish = FileNotesSessionOwner.publish_commit_outcome
    observed: list[tuple[object, object]] = []

    def observe_publication(self, lease, capture, publication):
        result = original_publish(self, lease, capture, publication)
        if result.published and publication.state == "succeeded":
            observed.append(
                (
                    publication.candidate_seed,
                    self.snapshot(capture.binding).push_candidate,
                )
            )
        return result

    monkeypatch.setattr(
        FileNotesSessionOwner,
        "publish_commit_outcome",
        observe_publication,
    )

    outcome = await service.start_commit(binding, review.handle)

    assert outcome.state == "succeeded"
    assert len(observed) == 1
    assert observed[0][0] is seed
    assert observed[0][1] == service._owner.snapshot(binding).push_candidate
    availability = service._owner.snapshot(binding).push_candidate
    assert availability is not None
    assert availability.candidate.parent_oid == review_snapshot.capture.head.object_id
    assert availability.candidate.candidate_oid == outcome.commit_object_id
    assert availability.candidate.subject == "Review subject"
    assert tuple(
        note.display_text for note in availability.candidate.included_notes
    ) == ("note.md",)
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_review_allows_linked_worktree_without_worktree_config(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    linked = tmp_path / "linked"
    _git(
        repository,
        "worktree",
        "add",
        "-q",
        "-b",
        "linked-review",
        str(linked),
        "HEAD",
    )

    service, _binding, result = await _prepare_owned_review(linked)

    assert result.state == "ready"
    await service.shutdown()


@pytest.mark.asyncio
async def test_armed_no_network_runner_rejects_unprotected_object_resolution(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _NoNetworkRunner()
    runner.arm()

    with pytest.raises(AssertionError, match="guarded review command"):
        await runner.run(
            ("git", "cat-file", "commit", "HEAD"),
            cwd=str(repository),
            environment={"GIT_NO_LAZY_FETCH": "1"},
        )


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
    await asyncio.wait_for(runner.exposed.wait(), _ASYNC_SETTLE_TIMEOUT)
    try:
        assert service._commit_review_snapshots == {}
    finally:
        runner.terminal.set()
        second = await asyncio.wait_for(second_waiter, _ASYNC_SETTLE_TIMEOUT)

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
    runner = _RecordingRunner()
    pre_review_state: dict[str, bytes] = {}

    service, binding, result = await _prepare_owned_review(
        repository,
        stage_unrelated=True,
        runner=runner,
        pre_review_state=pre_review_state,
    )

    assert result.state == "blocked"
    assert result.handle is None
    assert result.projection is None
    assert result.message == (
        "The complete staged state does not exactly match this session. "
        "If Git has unrelated staged changes, commit or unstage them outside "
        "Chatbook; then Refresh and review this session again."
    )
    assert "unrelated-secret" not in repr(result)
    assert "unrelated-secret" not in repr(service._owner.snapshot(binding))
    assert "unrelated-secret" not in repr(service._commit_review_snapshots)
    assert _git(repository, "rev-parse", "HEAD") == pre_review_state["head"]
    assert _git(repository, "ls-files", "-z", "--stage", "-v") == (
        pre_review_state["index"]
    )
    assert not any(
        any(
            isinstance(value, str) and value.startswith("core.hooksPath=")
            for value in argv
        )
        for argv, _environment in runner.calls
    )
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
async def test_commit_review_uses_trusted_clean_filter_for_worktree_freshness(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    filter_executable = repository / ".git" / "chatbook-matrix-clean"
    filter_executable.write_text(
        "#!/bin/sh\nsed 's/^stamp:.*/stamp: normalized/'\n",
        encoding="utf-8",
    )
    filter_executable.chmod(0o755)
    _git(
        repository,
        "config",
        "filter.chatbook-matrix.clean",
        shlex.quote(str(filter_executable)),
    )
    _git(repository, "config", "filter.chatbook-matrix.required", "true")
    (repository / ".gitattributes").write_text(
        "note.md filter=chatbook-matrix\n",
        encoding="utf-8",
    )
    _git(repository, "add", ".gitattributes")
    _git(repository, "commit", "-q", "-m", "configure clean filter")
    worktree_bytes = b"session content\nstamp: worktree-only\n"
    (repository / "note.md").write_bytes(worktree_bytes)

    service, binding, review = await _stage_changes_then_review(
        repository,
        (SessionChange("modified", "note.md"),),
    )

    assert review.state == "ready"
    assert review.handle is not None
    staged_bytes = _git(repository, "cat-file", "blob", ":note.md")
    assert staged_bytes == b"session content\nstamp: normalized\n"
    assert staged_bytes != worktree_bytes
    outcome = await service.start_commit(binding, review.handle)
    assert outcome.state == "succeeded"
    assert (repository / "note.md").read_bytes() == worktree_bytes
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "marker",
    [
        "MERGE_HEAD",
        "sequencer",
        "BISECT_START",
        "index.lock",
        "refs/heads/session.lock",
        "info/grafts",
    ],
)
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
@pytest.mark.parametrize(
    "repository_flag",
    [
        "promisor_repository",
        "worktree_promisor_repository",
        "included_promisor_repository",
    ],
)
async def test_commit_review_blocks_promisor_repository_without_fetch(
    tmp_path: Path,
    repository_flag: str,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _NoNetworkRunner()

    service, _binding, result = await _prepare_owned_review(
        repository,
        runner=runner,
        arm_runner_before_review=True,
        **{repository_flag: True},
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
    runner = _NoNetworkRunner()

    service, _binding, result = await _prepare_owned_review(
        repository,
        runner=runner,
        arm_runner_before_review=True,
        **{repository_flag: True},
    )

    assert result.state == "blocked"
    assert runner.calls
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
    ["intent", "conflict", "gitlink", "semantic", "assume"],
)
async def test_complete_commit_proof_blocks_unsupported_index_states(
    tmp_path: Path,
    unsupported_index_state: str,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _RecordingRunner()
    pre_review_state: dict[str, bytes] = {}

    service, _binding, result = await _prepare_owned_review(
        repository,
        unsupported_index_state=unsupported_index_state,
        runner=runner,
        pre_review_state=pre_review_state,
    )

    assert result.state == "blocked"
    assert result.handle is None
    assert _git(repository, "rev-parse", "HEAD") == pre_review_state["head"]
    assert _git(repository, "ls-files", "-z", "--stage", "-v") == (
        pre_review_state["index"]
    )
    assert not any(
        any(
            isinstance(value, str) and value.startswith("core.hooksPath=")
            for value in argv
        )
        for argv, _environment in runner.calls
    )
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
    await asyncio.wait_for(runner.exposed.wait(), _ASYNC_SETTLE_TIMEOUT)
    service, binding = captured[0]

    blocked_admission = service._owner.admit_mutation(binding)
    assert blocked_admission.reason == "mutation_active"
    assert not preparation.done()

    runner.terminal.set()
    _service, _binding, result = await asyncio.wait_for(
        preparation, _ASYNC_SETTLE_TIMEOUT
    )

    assert result.state in {"blocked", "cancelled"}
    assert runner.claimed is True
    assert runner.settle_calls >= 1
    assert runner.released is True
    admitted = service._owner.admit_mutation(binding)
    assert admitted.lease is not None
    admitted.lease.release()
    await service.shutdown()


@pytest.mark.asyncio
async def test_explicit_review_cancel_retains_proof_child_and_mutation_lease(
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
    await asyncio.wait_for(runner.exposed.wait(), _ASYNC_SETTLE_TIMEOUT)
    service, binding = captured[0]

    assert service.cancel_commit(binding)
    await asyncio.sleep(0)
    assert service.cancel_commit(binding)
    await asyncio.sleep(0)

    assert not preparation.done()
    assert service._owner.mutation_active(binding)
    assert runner.claimed is True
    assert runner.released is False

    runner.terminal.set()
    _service, _binding, result = await asyncio.wait_for(
        preparation, _ASYNC_SETTLE_TIMEOUT
    )

    assert result.state == "cancelled"
    assert runner.released is True
    assert service._owner.mutation_active(binding) is False
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
    await asyncio.wait_for(runner.exposed.wait(), _ASYNC_SETTLE_TIMEOUT)
    service, binding = captured[0]

    preparation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await preparation

    assert service._owner.admit_mutation(binding).reason == "mutation_active"
    cycle = service._commit_review_cycle
    assert cycle is not None
    runner.terminal.set()
    result = await asyncio.wait_for(asyncio.shield(cycle), _ASYNC_SETTLE_TIMEOUT)

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
    await asyncio.wait_for(runner.exposed.wait(), _ASYNC_SETTLE_TIMEOUT)
    service, binding = captured[0]

    await asyncio.wait_for(service.shutdown(), _ASYNC_SETTLE_TIMEOUT)
    _service, _binding, result = await asyncio.wait_for(
        preparation, _ASYNC_SETTLE_TIMEOUT
    )

    assert result.state == "blocked"
    assert runner.released is True
    assert runner.shutdown_called is True
    assert service._owner.mutation_active(binding) is False


@pytest.mark.asyncio
async def test_service_shutdown_joins_active_commit_before_return(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("shutdown_stop")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None
    waiter = service.start_commit(binding, review.handle)
    await asyncio.wait_for(runner.commit_started.wait(), _ASYNC_SETTLE_TIMEOUT)
    hooks_directory = runner.hooks_directory
    assert hooks_directory is not None and hooks_directory.is_dir()

    await asyncio.wait_for(service.shutdown(), _ASYNC_SETTLE_TIMEOUT)
    joined_at_return = waiter.done()
    outcome = await asyncio.wait_for(waiter, _ASYNC_SETTLE_TIMEOUT)

    assert runner.shutdown_called
    assert runner.released
    assert joined_at_return
    assert outcome.state == "uncertain"
    assert not service._owner.mutation_active(binding)
    assert service._owner.snapshot(binding).commit_recovery is not None
    assert not hooks_directory.exists()


@pytest.mark.asyncio
async def test_owner_shutdown_joins_commit_before_closing_publication(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("shutdown_stop")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None
    service._owner.attach_git_service(service)
    waiter = service.start_commit(binding, review.handle)
    await asyncio.wait_for(runner.commit_started.wait(), _ASYNC_SETTLE_TIMEOUT)
    hooks_directory = runner.hooks_directory
    assert hooks_directory is not None and hooks_directory.is_dir()

    await asyncio.wait_for(service._owner.shutdown_async(), _ASYNC_SETTLE_TIMEOUT)
    joined_at_return = waiter.done()
    outcome = await asyncio.wait_for(waiter, _ASYNC_SETTLE_TIMEOUT)

    assert joined_at_return
    assert runner.released
    assert outcome.state == "uncertain"
    assert service._uncertain_commit is not None
    assert service._uncertain_commit.recovery_capability is not None
    assert service._uncertain_commit.mutation_lease is None
    assert not service._owner.mutation_active(binding)
    assert not hooks_directory.exists()


@pytest.mark.asyncio
async def test_service_shutdown_releases_completed_unpublished_commit_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("zero_without_commit")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None
    original_publish = FileNotesSessionOwner.publish_commit_outcome

    def refuse_uncertainty(self, lease, capture, publication):
        if publication.state == "uncertain":
            return CommitPublicationResult(published=False)
        return original_publish(self, lease, capture, publication)

    monkeypatch.setattr(
        FileNotesSessionOwner,
        "publish_commit_outcome",
        refuse_uncertainty,
    )
    outcome = await service.start_commit(binding, review.handle)
    assert outcome.state == "uncertain"
    assert service._uncertain_commit is not None
    assert service._uncertain_commit.mutation_lease is not None
    assert service._owner.mutation_active(binding)

    await service.shutdown()

    assert service._uncertain_commit.mutation_lease is None
    assert not service._owner.mutation_active(binding)


@pytest.mark.asyncio
async def test_owner_shutdown_releases_completed_unpublished_commit_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("zero_without_commit")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None
    service._owner.attach_git_service(service)
    original_publish = FileNotesSessionOwner.publish_commit_outcome

    def refuse_uncertainty(self, lease, capture, publication):
        if publication.state == "uncertain":
            return CommitPublicationResult(published=False)
        return original_publish(self, lease, capture, publication)

    monkeypatch.setattr(
        FileNotesSessionOwner,
        "publish_commit_outcome",
        refuse_uncertainty,
    )
    outcome = await service.start_commit(binding, review.handle)
    assert outcome.state == "uncertain"
    assert service._uncertain_commit is not None
    assert service._uncertain_commit.mutation_lease is not None
    assert service._owner.mutation_active(binding)

    await service._owner.shutdown_async()

    assert service._uncertain_commit.mutation_lease is None
    assert not service._owner.mutation_active(binding)


@pytest.mark.asyncio
async def test_repeated_shutdown_preserves_unconfirmed_commit_hooks(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("uncertain_unconfirmed_shutdown")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None
    outcome = await service.start_commit(binding, review.handle)
    assert outcome.state == "uncertain"
    hooks_directory = runner.hooks_directory
    assert hooks_directory is not None and hooks_directory.is_dir()
    evidence = service._uncertain_commit
    assert evidence is not None
    assert evidence.retained_child is runner.token

    try:
        await service.shutdown()
        await service.shutdown()
        await service.shutdown()

        evidence = service._uncertain_commit
        assert evidence is not None
        assert evidence.retained_child is runner.token
        assert evidence.hooks_directory == hooks_directory
        assert runner.read_retained_child(runner.token).state == "alive"
        assert not runner.released
        assert hooks_directory.is_dir()
    finally:
        git_service._remove_private_hooks_directory(hooks_directory)


@pytest.mark.asyncio
async def test_repeated_shutdown_retries_confirmed_commit_hooks_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("uncertain_confirmed_shutdown")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None
    outcome = await service.start_commit(binding, review.handle)
    assert outcome.state == "uncertain"
    hooks_directory = runner.hooks_directory
    assert hooks_directory is not None and hooks_directory.is_dir()
    real_rmdir = Path.rmdir
    attempts = 0

    def fail_first_hooks_rmdir(path: Path) -> None:
        nonlocal attempts
        if path == hooks_directory:
            attempts += 1
            if attempts == 1:
                raise OSError("injected first cleanup failure")
        real_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", fail_first_hooks_rmdir)

    await service.shutdown()

    assert runner.released
    assert attempts == 1
    assert hooks_directory.is_dir()
    evidence = service._uncertain_commit
    assert evidence is not None
    assert evidence.retained_child is None
    assert evidence.hooks_directory == hooks_directory

    await service.shutdown()

    assert attempts == 2
    assert not hooks_directory.exists()
    assert service._uncertain_commit.hooks_directory is None


@pytest.mark.asyncio
async def test_commit_confirmation_consumes_handle_once_and_revalidates_message(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("failure")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None

    outcome = await service.start_commit(
        binding,
        review.handle,
        subject="Different subject",
    )

    assert outcome.state == "blocked"
    assert runner.commit_calls == 0
    with pytest.raises(
        git_service.GitMutationAdmissionError,
        match="capability",
    ):
        service.start_commit(binding, review.handle)
    await service.shutdown()


@pytest.mark.asyncio
async def test_malformed_commit_handle_is_rejected_before_mutation_admission(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("failure")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None
    retained_snapshots = dict(service._commit_review_snapshots)

    with pytest.raises(git_service.GitMutationAdmissionError) as refusal:
        service.start_commit(binding, object())  # type: ignore[arg-type]

    assert refusal.value.reason == "invalid_capability"
    assert not service._owner.mutation_active(binding)
    assert service._commit_review_snapshots == retained_snapshots
    outcome = await service.start_commit(binding, review.handle)
    assert outcome.state == "failed_unchanged"
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "drift",
    ["generation", "repository", "branch", "index", "worktree", "identity"],
)
async def test_commit_confirmation_rejects_every_review_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("failure")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None
    if drift == "generation":
        assert service._owner.record_change(
            binding,
            SessionChange("modified", "later.md"),
        )
    elif drift == "repository":
        trusted = service._owner.snapshot(binding).trusted_repository
        assert trusted is not None
        assert service._owner.publish_trust(
            binding,
            replace(
                trusted,
                worktree_identity=replace(
                    trusted.worktree_identity,
                    inode=(trusted.worktree_identity.inode or 0) + 1,
                ),
            ),
        )
    elif drift == "branch":
        _git(repository, "checkout", "-q", "-b", "other")
    elif drift == "index":
        (repository / "note.md").write_text("different staged\n", encoding="utf-8")
        _git(repository, "add", "note.md")
    elif drift == "worktree":
        (repository / "note.md").write_text("newer saved\n", encoding="utf-8")
    else:
        _git(repository, "config", "user.name", "Changed Identity")

    outcome = await service.start_commit(binding, review.handle)

    assert outcome.state == "blocked"
    assert runner.commit_calls == 0
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_confirmation_cancels_before_child_start(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledRetainedReviewRunner("passthrough")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None
    runner.arm("cancelled_token")

    waiter = service.start_commit(binding, review.handle)
    await asyncio.wait_for(runner.exposed.wait(), _ASYNC_SETTLE_TIMEOUT)
    assert service.cancel_commit(binding) is True
    runner.terminal.set()
    outcome = await asyncio.wait_for(waiter, _ASYNC_SETTLE_TIMEOUT)

    assert outcome.state == "cancelled"
    assert not any("commit" in argv for argv, _environment in runner.calls)
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_confirmation_cancel_refuses_after_child_begins(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("pause")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None

    waiter = service.start_commit(binding, review.handle)
    await asyncio.wait_for(runner.commit_started.wait(), _ASYNC_SETTLE_TIMEOUT)
    hooks_directory = runner.hooks_directory
    assert hooks_directory is not None and hooks_directory.is_dir()
    assert service.cancel_commit(binding) is False
    runner.release_commit.set()
    outcome = await asyncio.wait_for(waiter, _ASYNC_SETTLE_TIMEOUT)

    assert outcome.state == "failed_unchanged"
    assert runner.commit_calls == 1
    assert not hooks_directory.exists()
    await service.shutdown()


@pytest.mark.asyncio
async def test_hooks_directory_lives_through_child_and_is_removed_with_rmdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("pause")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None
    removed: list[Path] = []
    real_rmdir = Path.rmdir

    def recording_rmdir(path: Path) -> None:
        removed.append(path)
        real_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", recording_rmdir)
    waiter = service.start_commit(binding, review.handle)
    await asyncio.wait_for(runner.commit_started.wait(), _ASYNC_SETTLE_TIMEOUT)
    hooks_directory = runner.hooks_directory
    assert hooks_directory is not None and hooks_directory.is_dir()
    assert removed == []
    runner.release_commit.set()
    await asyncio.wait_for(waiter, _ASYNC_SETTLE_TIMEOUT)

    assert removed == [hooks_directory]
    assert not hooks_directory.exists()
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["chmod", "stat", "iterdir"])
async def test_failed_hooks_creation_removes_or_tracks_every_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, review = await _prepare_owned_review(repository)
    assert review.handle is not None
    created: list[Path] = []
    failed: set[Path] = set()
    real_mkdtemp = git_service.tempfile.mkdtemp
    real_chmod = Path.chmod
    real_stat = Path.stat
    real_iterdir = Path.iterdir
    real_rmdir = Path.rmdir

    def recording_mkdtemp(*args, **kwargs):
        directory = Path(real_mkdtemp(*args, **kwargs))
        created.append(directory)
        return str(directory)

    def fail_chmod_once(path: Path, *args, **kwargs):
        if path.name.startswith(".chatbook-hooks-") and path not in failed:
            failed.add(path)
            raise OSError("injected chmod failure")
        return real_chmod(path, *args, **kwargs)

    def fail_hook_stat(path: Path, *args, **kwargs):
        if path.name.startswith(".chatbook-hooks-"):
            failed.add(path)
            raise OSError("injected stat failure")
        return real_stat(path, *args, **kwargs)

    def fail_iterdir_once(path: Path):
        if path.name.startswith(".chatbook-hooks-") and path not in failed:
            failed.add(path)
            raise OSError("injected iterdir failure")
        return real_iterdir(path)

    def path_is_present(path: Path) -> bool:
        try:
            real_stat(path)
        except FileNotFoundError:
            return False
        return True

    monkeypatch.setattr(git_service.tempfile, "mkdtemp", recording_mkdtemp)
    if operation == "chmod":
        monkeypatch.setattr(Path, "chmod", fail_chmod_once)
    elif operation == "stat":
        monkeypatch.setattr(Path, "stat", fail_hook_stat)
    else:
        monkeypatch.setattr(Path, "iterdir", fail_iterdir_once)

    try:
        outcome = await service.start_commit(binding, review.handle)

        assert outcome.state == "blocked"
        assert created
        assert set(created) == failed
        pending = service._pending_hooks_cleanup
        assert all(
            not path_is_present(directory) or directory in pending
            for directory in created
        )
        monkeypatch.undo()
        await service.shutdown()
        assert service._pending_hooks_cleanup == set()
        assert all(not directory.exists() for directory in created)
    finally:
        for directory in created:
            try:
                real_rmdir(directory)
            except FileNotFoundError:
                pass


@pytest.mark.asyncio
async def test_failed_hooks_rmdir_retries_on_repeated_shutdown_without_recursion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, review = await _prepare_owned_review(repository)
    assert review.handle is not None
    created: list[Path] = []
    attempts: dict[Path, int] = {}
    real_mkdtemp = git_service.tempfile.mkdtemp
    real_iterdir = Path.iterdir
    real_rmdir = Path.rmdir

    def recording_mkdtemp(*args, **kwargs):
        directory = Path(real_mkdtemp(*args, **kwargs)).resolve()
        created.append(directory)
        return str(directory)

    def report_synthetic_entry(path: Path):
        if path.name.startswith(".chatbook-hooks-"):
            return iter((path / "synthetic",))
        return real_iterdir(path)

    def fail_twice_then_rmdir(path: Path) -> None:
        if not path.name.startswith(".chatbook-hooks-"):
            real_rmdir(path)
            return
        attempts[path] = attempts.get(path, 0) + 1
        if attempts[path] <= 2:
            raise OSError("injected rmdir failure")
        real_rmdir(path)

    def forbid_recursive_delete(*args, **kwargs):
        raise AssertionError(f"recursive deletion is forbidden: {args!r} {kwargs!r}")

    monkeypatch.setattr(git_service.tempfile, "mkdtemp", recording_mkdtemp)
    monkeypatch.setattr(Path, "iterdir", report_synthetic_entry)
    monkeypatch.setattr(Path, "rmdir", fail_twice_then_rmdir)
    monkeypatch.setattr(git_service.shutil, "rmtree", forbid_recursive_delete)

    try:
        outcome = await service.start_commit(binding, review.handle)

        assert outcome.state == "blocked"
        assert created
        assert service._pending_hooks_cleanup == set(created)
        assert all(directory.exists() for directory in created)

        await service.shutdown()
        assert service._pending_hooks_cleanup == set(created)
        assert all(directory.exists() for directory in created)

        await service.shutdown()
        assert service._pending_hooks_cleanup == set()
        assert all(not directory.exists() for directory in created)
    finally:
        monkeypatch.undo()
        for directory in created:
            try:
                real_rmdir(directory)
            except FileNotFoundError:
                pass


@pytest.mark.asyncio
async def test_hooks_creation_rejects_nonsticky_shared_parent_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, _review = await _prepare_owned_review(repository)
    repository_identity = service._owner.snapshot(binding).trusted_repository
    assert repository_identity is not None
    unsafe_temp = tmp_path / "apparently-private"
    unsafe_temp.mkdir()
    unsafe_temp.chmod(0o700)
    original_mode = stat.S_IMODE(tmp_path.stat().st_mode)
    tmp_path.chmod(0o777)
    mkdtemp_calls: list[str] = []

    def unsafe_mkdtemp_was_used(*args, **kwargs):
        mkdtemp_calls.append(str(kwargs.get("dir")))
        raise AssertionError("unsafe hooks parent reached mkdtemp")

    monkeypatch.setattr(
        git_service.tempfile,
        "gettempdir",
        lambda: str(unsafe_temp),
    )
    monkeypatch.setattr(
        git_service.tempfile,
        "mkdtemp",
        unsafe_mkdtemp_was_used,
    )
    try:
        with pytest.raises(OSError, match="private hooks"):
            git_service._create_private_hooks_directory(
                repository_identity,
                set(),
            )
        assert mkdtemp_calls == []
    finally:
        tmp_path.chmod(original_mode)
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_mode", [0o700, 0o1777])
async def test_hooks_creation_accepts_safe_owner_or_sticky_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_mode: int,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, _review = await _prepare_owned_review(repository)
    repository_identity = service._owner.snapshot(binding).trusted_repository
    assert repository_identity is not None
    safe_temp = tmp_path / f"safe-{parent_mode:o}"
    safe_temp.mkdir()
    safe_temp.chmod(parent_mode)
    monkeypatch.setattr(
        git_service.tempfile,
        "gettempdir",
        lambda: str(safe_temp),
    )
    pending: set[Path] = set()

    directory = git_service._create_private_hooks_directory(
        repository_identity,
        pending,
    )

    assert directory.parent == safe_temp.resolve()
    assert directory in pending
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert git_service._remove_private_hooks_directory(directory)
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_outcome_uses_immediately_known_natural_retained_result(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("retained_natural_failure")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None

    outcome = await service.start_commit(binding, review.handle)

    assert outcome.state == "failed_unchanged"
    assert runner.claimed is True
    assert runner.hooks_directory is not None
    assert not runner.hooks_directory.exists()
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_outcome_uncertain_child_retains_hooks_directory(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("uncertain")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None

    outcome = await service.start_commit(binding, review.handle)

    assert outcome.state == "uncertain"
    assert runner.claimed is True
    assert runner.hooks_directory is not None
    assert runner.hooks_directory.is_dir()
    assert service._owner.snapshot(binding).commit_recovery is not None
    assert service._uncertain_commit is not None
    assert service._uncertain_commit.recovery_capability is not None


@pytest.mark.asyncio
async def test_commit_outcome_zero_without_branch_change_is_uncertain(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("zero_without_commit")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None

    outcome = await service.start_commit(binding, review.handle)

    assert outcome.state == "uncertain"
    assert runner.commit_calls == 1
    assert service._owner.snapshot(binding).commit_recovery is not None
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["pause_then_commit", "pause", "pause_zero_without_commit"],
)
async def test_commit_outcome_authority_drift_always_falls_back_to_quarantine(
    tmp_path: Path,
    mode: str,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner(mode)
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None
    old_head = _git(repository, "rev-parse", "HEAD").decode().strip()

    waiter = service.start_commit(binding, review.handle)
    await asyncio.wait_for(runner.commit_admitted.wait(), _ASYNC_SETTLE_TIMEOUT)
    assert service._owner.record_change(
        binding,
        SessionChange("modified", "later.md"),
    )
    runner.release_commit.set()
    outcome = await asyncio.wait_for(waiter, _ASYNC_SETTLE_TIMEOUT)

    assert outcome.state == "uncertain"
    if mode == "pause_then_commit":
        assert _git(repository, "rev-parse", "HEAD").decode().strip() != old_head
    else:
        assert _git(repository, "rev-parse", "HEAD").decode().strip() == old_head
    snapshot = service._owner.snapshot(binding)
    assert [change.sequence for change in snapshot.changes] == [1, 2]
    assert snapshot.commit_recovery is not None
    assert dict(snapshot.staging_ownership) == {}
    assert service._uncertain_commit is not None
    assert service._uncertain_commit.recovery_capability is not None
    assert service._owner.admit_mutation(binding).reason == "recovery_required"
    assert runner.commit_calls == 1
    await service.shutdown()


@pytest.mark.asyncio
async def test_active_commit_refuses_root_rebind_and_publishes_original_session(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("pause_then_commit")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None

    waiter = service.start_commit(binding, review.handle)
    await asyncio.wait_for(runner.commit_admitted.wait(), _ASYNC_SETTLE_TIMEOUT)
    with pytest.raises(RuntimeError, match="Git mutation is in progress"):
        service._owner.select_root(tmp_path / "other")
    assert service._owner.current_binding() == binding
    runner.release_commit.set()
    outcome = await asyncio.wait_for(waiter, _ASYNC_SETTLE_TIMEOUT)

    assert outcome.state == "succeeded"
    assert service._owner.current_binding() == binding
    snapshot = service._owner.snapshot(binding)
    assert snapshot.changes == ()
    assert snapshot.commit_recovery is None
    assert snapshot.git_status is not None
    assert snapshot.git_status.rows == ()
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_outcome_runner_oserror_is_uncertain_not_natural_failure(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("oserror")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None

    outcome = await service.start_commit(binding, review.handle)

    assert outcome.state == "uncertain"
    assert service._owner.snapshot(binding).commit_recovery is not None
    assert service._uncertain_commit is not None
    assert service._uncertain_commit.recovery_capability is not None
    assert runner.hooks_directory is not None
    assert runner.hooks_directory.is_dir()


@pytest.mark.asyncio
async def test_live_unpublished_uncertainty_retains_exact_mutation_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("zero_without_commit")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None
    original_publish = FileNotesSessionOwner.publish_commit_outcome

    def refuse_uncertainty(self, lease, capture, publication):
        if publication.state == "uncertain":
            return CommitPublicationResult(published=False)
        return original_publish(self, lease, capture, publication)

    monkeypatch.setattr(
        FileNotesSessionOwner,
        "publish_commit_outcome",
        refuse_uncertainty,
    )

    outcome = await service.start_commit(binding, review.handle)

    assert outcome.state == "uncertain"
    assert service._uncertain_commit is not None
    assert service._uncertain_commit.recovery_capability is None
    retained_lease = service._uncertain_commit.mutation_lease
    assert retained_lease is not None
    assert retained_lease._binding == binding
    assert service._owner.mutation_active(binding)
    assert service._owner.admit_mutation(binding).reason == "mutation_active"
    assert service._owner.select_root(repository) == binding
    with pytest.raises(RuntimeError, match="Git mutation is in progress"):
        service._owner.select_root(tmp_path / "other")
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["commit_then_branch_drift", "commit_then_index_drift"],
)
async def test_commit_outcome_unexpected_branch_or_index_movement_is_uncertain(
    tmp_path: Path,
    mode: str,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner(mode)
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None

    outcome = await service.start_commit(binding, review.handle)

    assert outcome.state == "uncertain"
    assert service._owner.snapshot(binding).commit_recovery is not None
    assert runner.commit_calls == 1
    if mode == "commit_then_branch_drift":
        assert _git(repository, "symbolic-ref", "HEAD").decode().strip() == (
            "refs/heads/unexpected"
        )
    else:
        assert _git(repository, "diff", "--cached", "--name-only").decode() == (
            "unexpected.md\n"
        )
    await service.shutdown()


@pytest.mark.asyncio
async def test_guarded_commit_retains_newer_post_commit_worktree_edit(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("commit_then_worktree_edit")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None

    outcome = await service.start_commit(binding, review.handle)

    assert outcome.state == "succeeded"
    assert outcome.commit_object_id is not None
    assert _git(repository, "show", f"{outcome.commit_object_id}:note.md") == (
        b"staged\n"
    )
    assert (repository / "note.md").read_bytes() == b"newer worktree edit\n"
    assert _git(repository, "diff-index", "--cached", "HEAD", "--") == b""
    snapshot = service._owner.snapshot(binding)
    assert [change.sequence for change in snapshot.changes] == [1]
    assert dict(snapshot.staging_ownership) == {}
    assert snapshot.git_status is not None
    assert tuple(
        (row.group_id, row.state) for row in snapshot.git_status.rows
    ) == ((1, "unstaged"),)
    assert snapshot.push_candidate is not None
    assert snapshot.push_candidate.candidate.candidate_oid == outcome.commit_object_id
    assert snapshot.push_candidate.candidate.included_notes[0].display_text == (
        "note.md"
    )
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contradiction",
    [
        "missing_commit",
        "tree",
        "message",
        "author",
        "committer",
        "signature",
    ],
)
async def test_commit_outcome_raw_commit_contradictions_are_uncertain(
    tmp_path: Path,
    contradiction: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, review = await _prepare_owned_review(repository)
    assert review.handle is not None
    original_postflight = service._read_commit_postflight

    async def contradictory_postflight(*args, **kwargs):
        postflight = await original_postflight(*args, **kwargs)
        raw = postflight.raw_commit
        assert raw is not None
        if contradiction == "missing_commit":
            return replace(postflight, raw_commit=None)
        if contradiction == "tree":
            raw = replace(raw, tree_object_id="f" * 40)
        elif contradiction == "message":
            raw = replace(raw, message=b"Different message\n")
        elif contradiction == "author":
            raw = replace(
                raw,
                author=GitIdentity("Different Author", "author@example.test"),
            )
        elif contradiction == "committer":
            raw = replace(
                raw,
                committer=GitIdentity(
                    "Different Committer",
                    "committer@example.test",
                ),
            )
        else:
            raw = replace(raw, signature_headers=("gpgsig",))
        return replace(postflight, raw_commit=raw)

    monkeypatch.setattr(
        service,
        "_read_commit_postflight",
        contradictory_postflight,
    )

    outcome = await service.start_commit(binding, review.handle)

    assert outcome.state == "uncertain"
    assert service._owner.snapshot(binding).commit_recovery is not None
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_check_again_refuses_live_exact_child_without_new_commit(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, _review, runner = (
        await _prepare_uncertain_commit_recovery(
            repository,
            mode="uncertain",
        )
    )
    evidence = service._uncertain_commit
    assert evidence is not None
    hooks_directory = evidence.hooks_directory
    assert hooks_directory is not None and hooks_directory.is_dir()

    checked = await service.check_commit_again(binding)

    assert checked.state == "uncertain"
    assert runner.commit_calls == 1
    assert service._uncertain_commit is not None
    assert service._uncertain_commit.retained_child is runner.token
    assert hooks_directory.is_dir()
    projection = service._owner.snapshot(binding).commit_recovery
    assert projection is not None
    assert projection.can_check_again is False

    runner.terminal = True
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_recovery_evidence_drops_captured_staging_ownership(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    service, _binding, _review, runner = (
        await _prepare_uncertain_commit_recovery(
            repository,
            mode="uncertain",
        )
    )

    evidence = service._uncertain_commit

    assert evidence is not None
    assert not hasattr(evidence, "snapshot")
    assert not hasattr(evidence, "capture")
    assert "note.md" not in repr(evidence)
    runner.terminal = True
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocker", ["index.lock", "MERGE_HEAD"])
async def test_commit_check_again_waits_for_lock_or_special_state(
    tmp_path: Path,
    blocker: str,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, _review, runner = (
        await _prepare_uncertain_commit_recovery(
            repository,
            mode="uncertain",
        )
    )
    runner.terminal = True
    blocker_path = repository / ".git" / blocker
    blocker_path.touch()

    blocked = await service.check_commit_again(binding)

    assert blocked.state == "uncertain"
    assert runner.commit_calls == 1
    assert runner.released is True
    assert service._uncertain_commit is not None
    assert service._uncertain_commit.retained_child is None
    assert not runner.hooks_directory.exists()
    blocked_snapshot = service._owner.snapshot(binding)
    assert dict(blocked_snapshot.staging_ownership) == {}
    assert blocked_snapshot.commit_recovery is not None
    assert blocked_snapshot.commit_recovery.can_check_again is False

    blocker_path.unlink()
    recovered = await service.check_commit_again(binding)

    assert recovered.state == "failed_unchanged"
    assert runner.commit_calls == 1
    snapshot = service._owner.snapshot(binding)
    assert snapshot.commit_recovery is None
    assert snapshot.staging_ownership
    assert service._uncertain_commit is None
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_check_again_converges_to_exact_delayed_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, review, runner = await _prepare_uncertain_commit_recovery(
        repository,
        mode="uncertain",
    )
    evidence = service._uncertain_commit
    assert evidence is not None
    seed = evidence.proof.candidate_seed
    assert seed.subject == "Review subject"
    assert tuple(note.display_text for note in seed.included_notes) == ("note.md",)
    original_publish = FileNotesSessionOwner.publish_commit_outcome
    observed: list[tuple[object, object]] = []

    def observe_publication(self, lease, capture, publication):
        result = original_publish(self, lease, capture, publication)
        if result.published and publication.state == "succeeded":
            observed.append(
                (
                    publication.candidate_seed,
                    self.snapshot(capture.binding).push_candidate,
                )
            )
        return result

    monkeypatch.setattr(
        FileNotesSessionOwner,
        "publish_commit_outcome",
        observe_publication,
    )
    new_head = _commit_reviewed_index(repository, review)
    runner.terminal = True

    recovered = await service.check_commit_again(binding)

    assert recovered.state == "succeeded"
    assert recovered.commit_object_id == new_head
    assert recovered.committed_note_count == 1
    assert runner.commit_calls == 1
    snapshot = service._owner.snapshot(binding)
    assert snapshot.commit_recovery is None
    assert snapshot.changes == ()
    assert dict(snapshot.staging_ownership) == {}
    assert service._uncertain_commit is None
    assert len(observed) == 1
    assert observed[0][0] is seed
    assert observed[0][1] == snapshot.push_candidate
    assert snapshot.push_candidate is not None
    assert review.projection is not None
    assert snapshot.push_candidate.candidate.candidate_oid == new_head
    assert (
        snapshot.push_candidate.candidate.parent_oid
        == review.projection.old_commit
    )
    assert snapshot.push_candidate.candidate.subject == "Review subject"
    assert snapshot.push_candidate.change_types == ("Modified",)
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_check_again_keeps_unchanged_state_without_natural_failure(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, _review, runner = (
        await _prepare_uncertain_commit_recovery(
            repository,
            mode="zero_without_commit",
        )
    )
    initial_projection = service._owner.snapshot(binding).commit_recovery
    assert initial_projection is not None
    assert initial_projection.can_check_again is True
    initial_generation = service._owner.snapshot(
        binding
    ).git_authority_generation

    first = await service.check_commit_again(binding)
    first_generation = service._owner.snapshot(
        binding
    ).git_authority_generation
    second = await service.check_commit_again(binding)
    second_generation = service._owner.snapshot(
        binding
    ).git_authority_generation

    assert first.state == second.state == "uncertain"
    assert first == second
    assert first_generation == second_generation == initial_generation
    assert runner.commit_calls == 1
    assert dict(service._owner.snapshot(binding).staging_ownership) == {}
    projection = service._owner.snapshot(binding).commit_recovery
    assert projection is not None
    assert projection.can_check_again is True
    assert service._uncertain_commit is not None
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["signal", "uncertain_signal"])
async def test_commit_check_again_does_not_restore_after_signal_exit(
    tmp_path: Path,
    mode: str,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, _review, runner = (
        await _prepare_uncertain_commit_recovery(
            repository,
            mode=mode,
        )
    )
    if mode == "uncertain_signal":
        runner.terminal = True

    recovered = await service.check_commit_again(binding)

    assert recovered.state == "uncertain"
    assert runner.commit_calls == 1
    snapshot = service._owner.snapshot(binding)
    assert snapshot.commit_recovery is not None
    assert dict(snapshot.staging_ownership) == {}
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["terminal_stop", "terminal_force"])
async def test_commit_check_again_accepts_terminal_stopped_result_without_token(
    tmp_path: Path,
    mode: str,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, _review, runner = (
        await _prepare_uncertain_commit_recovery(
            repository,
            mode=mode,
        )
    )
    hooks_directory = runner.hooks_directory
    evidence = service._uncertain_commit

    assert hooks_directory is not None
    assert evidence is not None
    assert evidence.retained_child is None
    assert evidence.termination_known is True
    assert evidence.known_normal_returncode is None
    assert evidence.hooks_directory is None
    assert not hooks_directory.exists()
    projection = service._owner.snapshot(binding).commit_recovery
    assert projection is not None
    assert projection.can_check_again is True

    recovered = await service.check_commit_again(binding)

    assert recovered.state == "uncertain"
    assert runner.commit_calls == 1
    snapshot = service._owner.snapshot(binding)
    assert snapshot.commit_recovery is not None
    assert dict(snapshot.staging_ownership) == {}
    assert service._uncertain_commit is not None
    assert service._uncertain_commit.known_normal_returncode is None
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_check_again_requires_fresh_status_before_restoring_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, _review, runner = (
        await _prepare_uncertain_commit_recovery(
            repository,
            mode="uncertain",
        )
    )
    runner.terminal = True
    original_query = service._query_status

    async def unavailable_status(*args, **kwargs):
        status = await original_query(*args, **kwargs)
        return replace(status, state="error", message="injected status failure")

    monkeypatch.setattr(service, "_query_status", unavailable_status)

    recovered = await service.check_commit_again(binding)

    assert recovered.state == "uncertain"
    assert runner.commit_calls == 1
    snapshot = service._owner.snapshot(binding)
    assert snapshot.commit_recovery is not None
    assert dict(snapshot.staging_ownership) == {}
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_check_again_keeps_repository_differing_from_both_states(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, _review, runner = (
        await _prepare_uncertain_commit_recovery(
            repository,
            mode="uncertain",
        )
    )
    runner.terminal = True
    (repository / "note.md").write_text("different staged state\n", encoding="utf-8")
    _git(repository, "add", "note.md")

    first = await service.check_commit_again(binding)
    second = await service.check_commit_again(binding)

    assert first.state == second.state == "uncertain"
    assert runner.commit_calls == 1
    snapshot = service._owner.snapshot(binding)
    assert snapshot.commit_recovery is not None
    assert snapshot.commit_recovery.can_check_again is True
    assert dict(snapshot.staging_ownership) == {}
    assert service._uncertain_commit is not None
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_recovery_rebinding_discards_terminal_exact_evidence(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, _review, runner = (
        await _prepare_uncertain_commit_recovery(
            repository,
            mode="zero_without_commit",
        )
    )
    rebound = service._owner.select_root(tmp_path / "other")

    with pytest.raises(git_service.GitMutationAdmissionError) as refusal:
        service.check_commit_again(binding)

    assert refusal.value.reason == "stale_binding"
    assert service._owner.snapshot(rebound).commit_recovery is None
    assert service._uncertain_commit is None
    assert runner.commit_calls == 1
    await service.shutdown()


@pytest.mark.asyncio
async def test_live_commit_child_rebind_discards_proof_and_blocks_new_mutations(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    service, _binding, _review, runner = (
        await _prepare_uncertain_commit_recovery(
            repository,
            mode="uncertain",
        )
    )
    evidence = service._uncertain_commit
    assert evidence is not None
    hooks_directory = evidence.hooks_directory
    assert hooks_directory is not None and hooks_directory.is_dir()
    rebound_parent = tmp_path / "rebound"
    rebound_parent.mkdir()
    rebound_repository = _init_repository(rebound_parent)

    rebound = service._owner.select_root(rebound_repository)
    discovery = await service.discover(rebound)

    assert discovery.state == "ready"
    assert discovery.repository is not None
    assert service._uncertain_commit is None
    orphaned = service._orphaned_commit
    assert orphaned is not None
    assert not hasattr(orphaned, "proof")
    assert "Review subject" not in repr(orphaned)
    assert str(repository) not in repr(orphaned)
    assert hooks_directory.is_dir()
    assert runner.released is False

    assert service._owner.publish_trust(rebound, discovery.repository)
    assert service._owner.record_change(
        rebound,
        SessionChange("modified", "note.md"),
    )
    (rebound_repository / "note.md").write_text(
        "rebound edit\n",
        encoding="utf-8",
    )
    status = await service.start_status(
        rebound,
        service._owner.snapshot(rebound).changes,
    )
    assert status.state == "ready"
    assert len(status.rows) == 1

    actions = (
        lambda: service.start_stage(rebound, (status.rows[0].group_id,)),
        lambda: service.start_unstage(rebound, (status.rows[0].group_id,)),
        lambda: service.start_commit_review(rebound, "New root commit"),
        lambda: service.start_commit(
            rebound,
            CommitReviewHandle(object()),
        ),
    )
    for action in actions:
        with pytest.raises(git_service.GitMutationAdmissionError) as refusal:
            action()
        assert refusal.value.reason == "mutation_active"

    assert runner.commit_calls == 1
    runner.terminal = True
    await service.discover(rebound)

    assert runner.released is True
    assert not hooks_directory.exists()
    assert service._orphaned_commit is None
    staged = await service.start_stage(
        rebound,
        (status.rows[0].group_id,),
    )
    assert staged.state == "success"
    await service.shutdown()


@pytest.mark.asyncio
async def test_terminal_commit_child_rebind_settles_without_stale_check(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    service, _binding, _review, runner = (
        await _prepare_uncertain_commit_recovery(
            repository,
            mode="uncertain",
        )
    )
    evidence = service._uncertain_commit
    assert evidence is not None
    hooks_directory = evidence.hooks_directory
    assert hooks_directory is not None and hooks_directory.is_dir()
    runner.terminal = True
    rebound_parent = tmp_path / "rebound"
    rebound_parent.mkdir()
    rebound_repository = _init_repository(rebound_parent)

    rebound = service._owner.select_root(rebound_repository)
    discovery = await service.discover(rebound)

    assert discovery.state == "ready"
    assert discovery.repository is not None
    assert service._uncertain_commit is None
    assert service._orphaned_commit is None
    assert runner.released is True
    assert not hooks_directory.exists()
    assert runner.commit_calls == 1

    assert service._owner.publish_trust(rebound, discovery.repository)
    assert service._owner.record_change(
        rebound,
        SessionChange("modified", "note.md"),
    )
    (rebound_repository / "note.md").write_text(
        "rebound edit\n",
        encoding="utf-8",
    )
    status = await service.start_status(
        rebound,
        service._owner.snapshot(rebound).changes,
    )
    assert status.state == "ready"
    staged = await service.start_stage(
        rebound,
        (status.rows[0].group_id,),
    )
    assert staged.state == "success"
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_check_again_caller_cancellation_keeps_recovery_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, _review, runner = (
        await _prepare_uncertain_commit_recovery(
            repository,
            mode="zero_without_commit",
        )
    )
    started = asyncio.Event()
    release = asyncio.Event()
    original_postflight = service._read_commit_postflight

    async def delayed_postflight(*args, **kwargs):
        started.set()
        await release.wait()
        return await original_postflight(*args, **kwargs)

    monkeypatch.setattr(
        service,
        "_read_commit_postflight",
        delayed_postflight,
    )
    waiter = service.check_commit_again(binding)
    await asyncio.wait_for(started.wait(), _ASYNC_SETTLE_TIMEOUT)

    waiter.cancel("panel unmounted")
    with pytest.raises(asyncio.CancelledError):
        await waiter

    cycle = service._commit_recovery_cycle
    assert cycle is not None and not cycle.done()
    assert service._owner.mutation_active(binding)
    release.set()
    outcome = await asyncio.wait_for(asyncio.shield(cycle), _ASYNC_SETTLE_TIMEOUT)

    assert outcome.state == "uncertain"
    assert runner.commit_calls == 1
    assert not service._owner.mutation_active(binding)
    assert service._owner.snapshot(binding).commit_recovery is not None
    await service.shutdown()


@pytest.mark.asyncio
async def test_commit_recovery_process_exit_discards_quarantine_after_settlement(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, _review, runner = (
        await _prepare_uncertain_commit_recovery(
            repository,
            mode="uncertain",
        )
    )
    service._owner.attach_git_service(service)
    runner.terminal = True

    await service._owner.shutdown_async()

    assert runner.commit_calls == 1
    assert runner.released is True
    assert service._owner.snapshot(binding).commit_recovery is None
    assert dict(service._owner.snapshot(binding).staging_ownership) == {}
    assert not service._owner.mutation_active(binding)
    assert not runner.hooks_directory.exists()


@pytest.mark.asyncio
async def test_retained_commit_shutdown_publishes_before_hooks_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("shutdown_stop")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None
    events: list[str] = []
    original_publish = FileNotesSessionOwner.publish_commit_outcome
    original_rmdir = Path.rmdir

    def recording_publish(self, lease, capture, publication):
        events.append(f"publish:{publication.state}")
        return original_publish(self, lease, capture, publication)

    def recording_rmdir(path: Path) -> None:
        if path.name.startswith(".chatbook-hooks-"):
            events.append("hooks-cleanup")
        original_rmdir(path)

    monkeypatch.setattr(
        FileNotesSessionOwner,
        "publish_commit_outcome",
        recording_publish,
    )
    monkeypatch.setattr(Path, "rmdir", recording_rmdir)
    waiter = service.start_commit(binding, review.handle)
    await asyncio.wait_for(runner.commit_started.wait(), _ASYNC_SETTLE_TIMEOUT)

    await service.shutdown()
    outcome = await waiter

    assert outcome.state == "uncertain"
    assert runner.commit_calls == 1
    assert events.index("publish:uncertain") < events.index("hooks-cleanup")


@pytest.mark.asyncio
async def test_success_uses_one_status_snapshot_for_retirement_and_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, review = await _prepare_owned_review(
        repository,
        include_no_op_group=True,
    )
    assert review.handle is not None
    original_query = service._query_status
    queried_changes = []
    loop = asyncio.get_running_loop()

    async def inject_edit_after_query(
        query_binding,
        changes,
        query_repository,
        *,
        publish_ownership_changes=True,
    ):
        queried_changes.append(changes)
        status = await original_query(
            query_binding,
            changes,
            query_repository,
            publish_ownership_changes=publish_ownership_changes,
        )
        rows = tuple(
            replace(row, state="unstaged")
            if row.group_id == 2
            else row
            for row in status.rows
        )
        if len(queried_changes) == 1:
            loop.call_soon(
                service._owner.record_change,
                binding,
                SessionChange("modified", "later.md"),
            )
        return replace(status, rows=rows)

    monkeypatch.setattr(service, "_query_status", inject_edit_after_query)
    original_publish = FileNotesSessionOwner.publish_commit_outcome
    publications = []

    def record_publication(self, lease, capture, publication):
        publications.append(publication)
        return original_publish(self, lease, capture, publication)

    monkeypatch.setattr(
        FileNotesSessionOwner,
        "publish_commit_outcome",
        record_publication,
    )

    outcome = await service.start_commit(binding, review.handle)
    await asyncio.sleep(0)

    assert outcome.state == "succeeded"
    assert len(queried_changes) == 1
    assert [change.sequence for change in queried_changes[0]] == [1, 2]
    assert len(publications) == 1
    publication = publications[0]
    assert publication.state == "succeeded"
    assert publication.retired_sequence_ids == (1,)
    assert publication.divergent_sequence_ids == (2,)
    assert publication.refreshed_status is not None
    assert tuple(
        (row.group_id, row.state)
        for row in publication.refreshed_status.rows
    ) == ((2, "unstaged"),)
    assert [
        change.sequence
        for change in service._owner.snapshot(binding).changes
    ] == [2, 3]
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change_kind", "relative_path", "expected_change_type"),
    [
        ("created", "created.md", "New"),
        ("modified", "note.md", "Modified"),
        ("deleted", "note.md", "Deleted"),
        ("mode", "note.md", "Modified"),
    ],
)
async def test_guarded_commit_applies_basic_file_shapes_to_exact_complete_tree(
    tmp_path: Path,
    change_kind: str,
    relative_path: str,
    expected_change_type: str,
) -> None:
    repository = _init_repository(tmp_path)
    (repository / "untouched.md").write_text("untouched\n", encoding="utf-8")
    _git(repository, "add", "untouched.md")
    _git(repository, "commit", "-q", "-m", "add complete-tree fixture")
    target = repository / relative_path
    if change_kind == "created":
        target.write_text("created\n", encoding="utf-8")
        change = SessionChange("created", relative_path)
    elif change_kind == "deleted":
        target.unlink()
        change = SessionChange("deleted", relative_path)
    elif change_kind == "mode":
        if os.name != "posix":
            pytest.skip("POSIX executable-bit contract")
        target.chmod(0o755)
        change = SessionChange("modified", relative_path)
    else:
        target.write_text("modified\n", encoding="utf-8")
        change = SessionChange("modified", relative_path)

    service, binding, review = await _stage_changes_then_review(
        repository,
        (change,),
    )
    assert review.state == "ready"
    assert review.handle is not None
    assert review.projection is not None
    assert review.projection.included_notes[0].change_type == expected_change_type
    old_head = _git(repository, "rev-parse", "HEAD").decode().strip()
    old_branch = _git(repository, "symbolic-ref", "HEAD")
    expected_tree = _git(repository, "write-tree").decode().strip()
    expected_listing = _git(repository, "ls-tree", "-r", "-z", expected_tree)

    outcome = await service.start_commit(binding, review.handle)

    assert outcome.state == "succeeded"
    assert outcome.commit_object_id is not None
    new_head = _git(repository, "rev-parse", "HEAD").decode().strip()
    raw_commit = parse_raw_commit_object(
        _git(repository, "cat-file", "commit", new_head)
    )
    assert new_head == outcome.commit_object_id
    assert raw_commit.parent_object_id == old_head
    assert raw_commit.tree_object_id == expected_tree
    assert _git(repository, "ls-tree", "-r", "-z", new_head) == expected_listing
    assert _git(repository, "symbolic-ref", "HEAD") == old_branch
    assert _git(repository, "diff-index", "--cached", new_head, "--") == b""
    assert (repository / "untouched.md").read_bytes() == b"untouched\n"
    if change_kind == "mode":
        tree_entry = _git(repository, "ls-tree", new_head, relative_path)
        assert tree_entry.startswith(b"100755 ")
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("move_case", ["grouped", "chained"])
async def test_guarded_commit_applies_grouped_and_chained_moves(
    tmp_path: Path,
    move_case: str,
) -> None:
    repository = _init_repository(tmp_path)
    source = repository / "note.md"
    if move_case == "grouped":
        destination = repository / "moved.md"
        source.rename(destination)
        destination.write_text("moved and edited\n", encoding="utf-8")
        changes = (
            SessionChange("moved", "note.md", "moved.md"),
            SessionChange("modified", "moved.md"),
        )
        absent_paths = ("note.md",)
    else:
        intermediate = repository / "intermediate.md"
        destination = repository / "final.md"
        source.rename(intermediate)
        intermediate.rename(destination)
        changes = (
            SessionChange("moved", "note.md", "intermediate.md"),
            SessionChange("moved", "intermediate.md", "final.md"),
        )
        absent_paths = ("note.md", "intermediate.md")

    service, binding, review = await _stage_changes_then_review(
        repository,
        changes,
    )
    assert review.state == "ready"
    assert review.handle is not None
    assert review.projection is not None
    assert review.projection.included_note_count == 1
    assert review.projection.included_notes[0].change_type == "Moved"
    expected_tree = _git(repository, "write-tree").decode().strip()

    outcome = await service.start_commit(binding, review.handle)

    assert outcome.state == "succeeded"
    assert outcome.committed_note_count == 1
    assert destination.read_bytes() == (
        b"moved and edited\n" if move_case == "grouped" else b"baseline\n"
    )
    assert all(not (repository / path).exists() for path in absent_paths)
    assert _git(repository, "rev-parse", "HEAD^{tree}").decode().strip() == (
        expected_tree
    )
    assert _git(repository, "diff-index", "--cached", "HEAD", "--") == b""
    await service.shutdown()


@pytest.mark.asyncio
async def test_guarded_commit_ignores_ambient_author_and_committer_dates(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _RecordingRunner()
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
        environment={
            **os.environ,
            "GIT_AUTHOR_DATE": "1000000000 +0000",
            "GIT_COMMITTER_DATE": "1000000001 +0000",
        },
    )
    assert review.state == "ready"
    assert review.handle is not None

    outcome = await service.start_commit(binding, review.handle)

    assert outcome.state == "succeeded"
    assert outcome.commit_object_id is not None
    raw_commit = _git(
        repository,
        "cat-file",
        "commit",
        outcome.commit_object_id,
    )
    assert b" 1000000000 +0000\n" not in raw_commit
    assert b" 1000000001 +0000\n" not in raw_commit
    assert all(
        "GIT_AUTHOR_DATE" not in environment
        and "GIT_COMMITTER_DATE" not in environment
        for _argv, environment in runner.calls
        if "--no-replace-objects" in _argv
    )
    await service.shutdown()


@pytest.mark.asyncio
async def test_guarded_commit_leaves_note_bytes_and_sqlite_recovery_rows_unchanged(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    service, binding, review = await _prepare_owned_review(repository)
    assert review.handle is not None
    note_bytes = (repository / "note.md").read_bytes()
    root = str(repository.resolve())
    replica = FileNotesReplica(tmp_path / "file-notes.sqlite3")
    replica.upsert_file(
        root,
        "note.md",
        note_bytes,
        content_hash="a" * 64,
        decoded_text=note_bytes.decode("utf-8"),
        size=len(note_bytes),
        mtime_ns=1,
    )
    replica.protect(root, "note.md")
    assert replica.checkpoint(
        root,
        "note.md",
        b"baseline\n",
        content_hash="b" * 64,
        session_key="matrix-session",
        created_at="2026-07-29T00:00:00Z",
    )
    tombstone_bytes = b"deleted recovery bytes\n"
    replica.upsert_file(
        root,
        "deleted.md",
        tombstone_bytes,
        content_hash="c" * 64,
        decoded_text=tombstone_bytes.decode("utf-8"),
        size=len(tombstone_bytes),
        mtime_ns=2,
    )
    replica.prepare_deletion(
        root,
        "deleted.md",
        tombstone_bytes,
        content_hash="c" * 64,
        decoded_text=tombstone_bytes.decode("utf-8"),
        deleted_at="2026-07-29T00:01:00Z",
        created_at="2026-07-29T00:01:00Z",
    )

    def recovery_rows() -> tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]:
        with replica._lock:
            return tuple(
                (
                    table,
                    tuple(
                        tuple(row)
                        for row in replica._connection.execute(
                            f"SELECT * FROM {table} ORDER BY rowid"
                        )
                    ),
                )
                for table in ("files", "revisions", "protected_paths")
            )

    rows_before = recovery_rows()
    try:
        outcome = await service.start_commit(binding, review.handle)

        assert outcome.state == "succeeded"
        assert (repository / "note.md").read_bytes() == note_bytes
        assert recovery_rows() == rows_before
        assert replica.get_bytes(root, "note.md") == note_bytes
        assert replica.get_restore_bytes(root, "deleted.md") == tombstone_bytes
        assert replica.list_deleted(root) == ["deleted.md"]
        assert replica.is_protected(root, "note.md")
    finally:
        replica.close()
        await service.shutdown()


@pytest.mark.asyncio
async def test_guarded_commit_git_process_count_is_constant_for_1000_notes(
    tmp_path: Path,
) -> None:
    async def run_representative_session(
        repository_name: str,
        note_count: int,
    ) -> int:
        repository = tmp_path / repository_name
        repository.mkdir()
        _git(repository, "init", "-q")
        _git(repository, "config", "user.name", "Test User")
        _git(repository, "config", "user.email", "user@example.test")
        for index in range(note_count):
            (repository / f"note-{index:04d}.md").write_text(
                f"baseline {index}\n",
                encoding="utf-8",
            )
        _git(repository, "add", "--all")
        _git(repository, "commit", "-q", "-m", "bulk baseline")

        (repository / "note-0000.md").write_text(
            "representative modification\n",
            encoding="utf-8",
        )
        (repository / "note-0001.md").unlink()
        (repository / "note-0002.md").rename(repository / "moved.md")
        (repository / "created.md").write_text(
            "representative creation\n",
            encoding="utf-8",
        )
        runner = _RecordingRunner()
        service, binding, review = await _stage_changes_then_review(
            repository,
            (
                SessionChange("modified", "note-0000.md"),
                SessionChange("deleted", "note-0001.md"),
                SessionChange("moved", "note-0002.md", "moved.md"),
                SessionChange("created", "created.md"),
            ),
            runner=runner,
            reset_runner_before_review=True,
        )
        assert review.state == "ready"
        assert review.handle is not None
        assert review.projection is not None
        assert review.projection.included_note_count == 4
        outcome = await service.start_commit(binding, review.handle)
        assert outcome.state == "succeeded"
        assert outcome.committed_note_count == 4
        protected_call_count = sum(
            "--no-replace-objects" in argv for argv, _environment in runner.calls
        )
        assert protected_call_count < len(runner.calls)
        await service.shutdown()
        return len(runner.calls)

    small_count = await run_representative_session("small", 4)
    large_count = await run_representative_session("large", 1_000)

    assert small_count == large_count
    assert small_count <= 64


@pytest.mark.asyncio
async def test_guarded_commit_success_proves_exact_unsigned_commit(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    sentinel = tmp_path / "unowned-helper-ran"
    hook = repository / ".git" / "hooks" / "pre-commit"
    hook.write_text(f"#!/bin/sh\ntouch '{sentinel}'\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    signer = tmp_path / "signer"
    signer.write_text(f"#!/bin/sh\ntouch '{sentinel}'\nexit 1\n", encoding="utf-8")
    signer.chmod(0o755)
    fsmonitor = tmp_path / "fsmonitor"
    fsmonitor.write_text(
        f"#!/bin/sh\ntouch '{sentinel}'\nexit 1\n",
        encoding="utf-8",
    )
    fsmonitor.chmod(0o755)
    unrelated = repository / "unrelated.txt"
    unrelated.write_bytes(b"unrelated bytes\x00stay")
    old_head = _git(repository, "rev-parse", "HEAD").decode().strip()
    runner = _RecordingRunner()
    service, binding, review = await _prepare_owned_review(
        repository,
        unrelated_unstaged=True,
        runner=runner,
    )
    assert review.handle is not None
    assert review.projection is not None
    _git(repository, "config", "commit.gpgSign", "true")
    _git(repository, "config", "gpg.program", str(signer))
    _git(repository, "config", "core.fsmonitor", str(fsmonitor))

    outcome = await service.start_commit(binding, review.handle)

    assert outcome.state == "succeeded"
    assert outcome.commit_object_id is not None
    assert outcome.message == (
        f"Committed 1 session notes as {outcome.commit_object_id[:12]}; "
        "unrelated changes untouched."
    )
    assert outcome.qualification == (
        "No unrelated staged content was committed; "
        "Chatbook selected no unrelated worktree paths."
    )
    assert not sentinel.exists()
    commit_calls = [
        argv
        for argv, _environment in runner.calls
        if any(
            isinstance(value, str) and value.startswith("core.hooksPath=")
            for value in argv
        )
    ]
    assert len(commit_calls) == 1
    assert "maintenance.auto=false" in commit_calls[0]
    assert "gc.auto=0" in commit_calls[0]
    assert "core.fsmonitor=false" in commit_calls[0]
    assert "commit.gpgSign=false" in commit_calls[0]
    new_head = _git(repository, "rev-parse", "HEAD").decode().strip()
    assert new_head == outcome.commit_object_id
    raw = parse_raw_commit_object(_git(repository, "cat-file", "commit", new_head))
    assert raw.parent_object_id == old_head
    assert raw.tree_object_id == _git(repository, "write-tree").decode().strip()
    assert raw.message == b"Review subject\n\nBody\n"
    assert raw.author == review.projection.author
    assert raw.committer == review.projection.committer
    assert raw.has_signature is False
    assert _git(repository, "diff-index", "--cached", new_head, "--") == b""
    assert unrelated.read_bytes() == b"unrelated bytes\x00stay"
    snapshot = service._owner.snapshot(binding)
    assert snapshot.changes == ()
    assert dict(snapshot.staging_ownership) == {}
    assert snapshot.git_status is not None
    assert snapshot.git_status.head is not None
    assert snapshot.git_status.head.object_id == new_head
    assert snapshot.git_status.rows == ()
    await service.shutdown()


@pytest.mark.asyncio
async def test_guarded_commit_failure_proves_branch_and_index_unchanged(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path)
    runner = _ControlledCommitRunner("failure")
    service, binding, review = await _prepare_owned_review(
        repository,
        runner=runner,
    )
    assert review.handle is not None
    old_head = _git(repository, "rev-parse", "HEAD").decode().strip()
    old_index = _git(repository, "ls-files", "-z", "--stage", "-v")

    outcome = await service.start_commit(binding, review.handle)

    assert outcome.state == "failed_unchanged"
    assert runner.commit_calls == 1
    assert _git(repository, "rev-parse", "HEAD").decode().strip() == old_head
    assert _git(repository, "ls-files", "-z", "--stage", "-v") == old_index
    assert service._owner.snapshot(binding).staging_ownership
    await service.shutdown()
