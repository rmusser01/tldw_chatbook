from __future__ import annotations

import os
import shlex
import shutil
import stat
import subprocess
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from tldw_chatbook.Notes.file_notes_git_service import (
    AsyncGitProcessRunner,
    FileNotesGitService,
    GitCommandResult,
    GitStatusAdmissionError,
    coalesce_session_changes,
    parse_index_entries_z,
)
from tldw_chatbook.Notes.file_notes_session_owner import (
    FileNotesSessionOwner,
    HeadIdentity,
    IndexBaseline,
    IndexEntry,
    SequencedSessionChange,
    SessionChange,
    StagingOwnership,
)


@dataclass(frozen=True, slots=True)
class _Repository:
    git: str
    path: Path
    environment: dict[str, str]
    service_environment: dict[str, str]

    def run(
        self,
        *arguments: str,
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [self.git, *arguments],
            cwd=cwd or self.path,
            env=self.environment,
            check=check,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


class _RecordingRunner:
    """Delegate to the real runner while retaining the exact command trace."""

    def __init__(self) -> None:
        self.delegate = AsyncGitProcessRunner()
        self.calls: list[tuple[str | bytes, ...]] = []
        self.stdins: list[bytes | None] = []

    async def run(
        self,
        argv: Sequence[str | bytes],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        self.calls.append(tuple(argv))
        self.stdins.append(stdin)
        return await self.delegate.run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
        )

    def shutdown(self) -> Awaitable[bool]:
        return self.delegate.shutdown()


def _disposable_repository(
    tmp_path: Path,
    *,
    name: str = "repository",
    commit: bool = True,
) -> _Repository:
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is not installed")

    private_home = tmp_path / "private-home"
    private_home.mkdir()
    private_global_config = private_home / "global.gitconfig"
    private_global_config.write_text("", encoding="utf-8")
    path = tmp_path / name
    path.mkdir()
    environment = {
        **os.environ,
        "HOME": str(private_home),
        "XDG_CONFIG_HOME": str(private_home / "xdg"),
        "GIT_CONFIG_GLOBAL": str(private_global_config),
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    service_environment = {
        key: value
        for key, value in environment.items()
        if key not in {
            "GIT_CONFIG_GLOBAL",
            "GIT_CONFIG_SYSTEM",
            "GIT_CONFIG_NOSYSTEM",
        }
    }
    repository = _Repository(
        git=git,
        path=path,
        environment=environment,
        service_environment=service_environment,
    )
    repository.run("init", "--initial-branch=main")
    repository.run("config", "--local", "user.name", "Chatbook Test")
    repository.run(
        "config",
        "--local",
        "user.email",
        "chatbook@example.invalid",
    )
    if commit:
        (path / "tracked.md").write_text("initial\n", encoding="utf-8")
        repository.run("add", "--", "tracked.md")
        repository.run("commit", "-m", "initial")
    return repository


@pytest.mark.asyncio
@pytest.mark.parametrize("notes_subdirectory", [None, "notes"])
async def test_discover_supports_repo_equal_to_or_above_notes_root(
    tmp_path: Path,
    notes_subdirectory: str | None,
) -> None:
    repository = _disposable_repository(tmp_path)
    notes_root = repository.path
    if notes_subdirectory is not None:
        notes_root = repository.path / notes_subdirectory
        notes_root.mkdir()
    owner = FileNotesSessionOwner()
    binding = owner.select_root(notes_root)
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )

    result = await service.discover(binding)

    assert result.state == "ready"
    assert result.repository is not None
    assert result.repository.worktree_root == str(repository.path.resolve())
    assert result.head is not None
    assert result.head.kind == "attached"
    assert result.head.branch == "refs/heads/main"
    assert owner.snapshot(binding).trusted_repository is None
    await service.shutdown()


@pytest.mark.asyncio
async def test_discover_linked_worktree_has_distinct_git_and_common_dirs(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    linked = tmp_path / "linked"
    repository.run("worktree", "add", "--detach", str(linked), "HEAD")
    notes_root = linked / "notes"
    notes_root.mkdir()
    owner = FileNotesSessionOwner()
    binding = owner.select_root(notes_root)
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )

    result = await service.discover(binding)

    assert result.state == "ready"
    assert result.repository is not None
    assert result.repository.worktree_root == str(linked.resolve())
    assert result.repository.git_dir != result.repository.git_common_dir
    assert result.head == HeadIdentity.detached(
        repository.run("rev-parse", "HEAD").stdout.decode("ascii").strip()
    )
    await service.shutdown()


@pytest.mark.asyncio
async def test_discover_reports_explicit_unborn_head(tmp_path: Path) -> None:
    repository = _disposable_repository(tmp_path, commit=False)
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )

    result = await service.discover(binding)

    assert result.state == "ready"
    assert result.head == HeadIdentity.unborn("refs/heads/main")
    await service.shutdown()


@pytest.mark.asyncio
async def test_discover_non_repository_is_unavailable(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is not installed")
    root = tmp_path / "notes"
    root.mkdir()
    private_home = tmp_path / "private-home"
    private_home.mkdir()
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    service = FileNotesGitService(
        owner,
        git_executable=git,
        environment={
            "HOME": str(private_home),
            "PATH": os.environ.get("PATH", ""),
        },
    )

    result = await service.discover(binding)

    assert result.state == "not_repository"
    assert result.repository is None
    await service.shutdown()


@pytest.mark.asyncio
async def test_revalidate_rejects_replaced_git_directory_identity(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    original_git_dir = Path(discovery.repository.git_dir)
    replaced_git_dir = repository.path / ".git-replaced"
    original_git_dir.rename(replaced_git_dir)
    original_git_dir.mkdir()

    assert not await service.revalidate_repository(
        binding,
        discovery.repository,
    )
    snapshot = owner.snapshot(binding)
    assert snapshot.trusted_repository is None
    assert snapshot.git_status is None
    assert not snapshot.staging_ownership
    await service.shutdown()


@pytest.mark.asyncio
async def test_status_rejects_linked_worktree_gitdir_rebinding(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    linked = tmp_path / "linked"
    replacement = tmp_path / "replacement"
    repository.run("worktree", "add", "--detach", str(linked), "HEAD")
    repository.run("worktree", "add", "--detach", str(replacement), "HEAD")
    (linked / "tracked.md").write_text("linked change\n", encoding="utf-8")
    runner = _RecordingRunner()
    owner = FileNotesSessionOwner()
    binding = owner.select_root(linked)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert discovery.head is not None
    assert owner.publish_trust(binding, discovery.repository)
    assert owner.publish_ownership(
        binding,
        {
            1: StagingOwnership(
                repository=discovery.repository,
                head=discovery.head,
                approved_endpoint_topology=("tracked.md",),
                approved_move_edges=(),
                approved_current_path="tracked.md",
                original_baselines={
                    "tracked.md": IndexBaseline(entry=None),
                },
                post_stage_entries={"tracked.md": None},
            )
        },
    )

    replacement_git_file = replacement / ".git"
    replacement_git_dir = Path(
        replacement_git_file.read_text(encoding="utf-8")
        .removeprefix("gitdir: ")
        .strip()
    )
    (linked / ".git").write_text(
        f"gitdir: {replacement_git_dir}\n",
        encoding="utf-8",
    )
    (replacement_git_dir / "gitdir").write_text(
        f"{linked / '.git'}\n",
        encoding="utf-8",
    )
    trusted_git_dir = Path(discovery.repository.git_dir)
    assert trusted_git_dir.is_dir()
    call_boundary = len(runner.calls)

    status = await service.start_status(
        binding,
        (_change(1, "modified", "tracked.md"),),
    )

    assert status.state == "stale"
    assert status.message is not None
    assert "identity changed" in status.message.lower()
    snapshot = owner.snapshot(binding)
    assert snapshot.trusted_repository is None
    assert snapshot.git_status is None
    assert not snapshot.staging_ownership
    status_calls = runner.calls[call_boundary:]
    assert status_calls
    assert all(
        len(call) > 1 and os.fsdecode(call[1]) == "rev-parse"
        for call in status_calls
    )
    await service.shutdown()


@pytest.mark.asyncio
async def test_status_classifies_real_merge_conflict_locally(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    repository.run("checkout", "-b", "conflict-side")
    (repository.path / "tracked.md").write_text(
        "side change\n",
        encoding="utf-8",
    )
    repository.run("add", "--", "tracked.md")
    repository.run("commit", "-m", "side change")
    repository.run("checkout", "main")
    (repository.path / "tracked.md").write_text(
        "main change\n",
        encoding="utf-8",
    )
    repository.run("add", "--", "tracked.md")
    repository.run("commit", "-m", "main change")
    merge = repository.run("merge", "conflict-side", check=False)
    assert merge.returncode != 0
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)

    status = await service.start_status(
        binding,
        (_change(1, "modified", "tracked.md"),),
    )

    assert status.state == "ready"
    assert len(status.rows) == 1
    assert status.rows[0].state == "conflict"
    assert status.rows[0].disabled_reason == "Git conflict"
    await service.shutdown()


def _change(
    sequence: int,
    action: str,
    relative_path: str,
    destination_path: str | None = None,
) -> SequencedSessionChange:
    return SequencedSessionChange(
        sequence=sequence,
        change=SessionChange(  # type: ignore[arg-type]
            action=action,
            relative_path=relative_path,
            destination_path=destination_path,
        ),
    )


def _index_entries(repository: _Repository) -> dict[str, IndexEntry]:
    return {
        entry.path: entry
        for entry in parse_index_entries_z(
            repository.run(
                "ls-files",
                "-z",
                "--stage",
                "-v",
                "--",
            ).stdout
        )
    }


@pytest.mark.asyncio
async def test_stage_bulk_exact_paths_preserves_unrelated_index_and_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenReplica:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"Stage touched the replica boundary: {name}")

    repository = _disposable_repository(tmp_path)
    created = repository.path / "created.md"
    created.write_text("created\n", encoding="utf-8")
    tracked = repository.path / "tracked.md"
    tracked.write_text("modified\n", encoding="utf-8")
    deleted = repository.path / "deleted.md"
    deleted.write_text("delete me\n", encoding="utf-8")
    unrelated = repository.path / "unrelated.md"
    unrelated.write_text("baseline\n", encoding="utf-8")
    repository.run("add", "--", "deleted.md", "unrelated.md")
    repository.run("commit", "-m", "more baseline")
    deleted.unlink()
    unrelated.write_text("externally staged\n", encoding="utf-8")
    repository.run("add", "--", "unrelated.md")
    unrelated_index_before = repository.run(
        "ls-files",
        "--stage",
        "--",
        "unrelated.md",
    ).stdout

    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    changes = (
        _change(1, "modified", "tracked.md"),
        _change(2, "created", "created.md"),
        _change(3, "deleted", "deleted.md"),
    )
    for item in changes:
        assert owner.record_change(binding, item.change)
    runner = _RecordingRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    service._replica = ForbiddenReplica()  # type: ignore[attr-defined]
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    changes_before = owner.snapshot(binding).changes
    record_calls: list[tuple[object, ...]] = []

    def forbidden_record_change(*args: object) -> bool:
        record_calls.append(args)
        raise AssertionError("Stage wrote File Notes session history")

    monkeypatch.setattr(
        FileNotesSessionOwner,
        "record_change",
        forbidden_record_change,
    )

    result = await service.start_stage(binding, (1, 2, 3))

    assert result.state == "success"
    assert result.staged_group_ids == (1, 2, 3)
    assert repository.run(
        "diff",
        "--cached",
        "--name-only",
        "--",
        "tracked.md",
        "created.md",
        "deleted.md",
    ).stdout.splitlines() == [b"created.md", b"deleted.md", b"tracked.md"]
    assert repository.run(
        "ls-files",
        "--stage",
        "--",
        "unrelated.md",
    ).stdout == unrelated_index_before
    stage_calls = [
        call
        for call in runner.calls
        if "add" in tuple(os.fsdecode(argument) for argument in call)
    ]
    assert len(stage_calls) == 1
    assert tuple(os.fsdecode(argument) for argument in stage_calls[0][:7]) == (
        repository.git,
        "--literal-pathspecs",
        "-c",
        "add.ignoreErrors=false",
        "add",
        "--all",
        "--",
    )
    snapshot = owner.snapshot(binding)
    assert snapshot.changes == changes_before
    assert not record_calls
    assert snapshot.git_status is None
    assert set(snapshot.staging_ownership) == {1, 2, 3}
    await service.shutdown()


@pytest.mark.asyncio
async def test_stage_update_retains_original_baseline_and_expands_owned_content(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "tracked.md"),
    )
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    original_entry = repository.run(
        "ls-files",
        "--stage",
        "--",
        "tracked.md",
    ).stdout

    (repository.path / "tracked.md").write_text("first\n", encoding="utf-8")
    first = await service.start_stage(binding, (1,))
    assert first.state == "success"
    first_ownership = owner.snapshot(binding).staging_ownership[1]

    (repository.path / "tracked.md").write_text("second\n", encoding="utf-8")
    second = await service.start_stage(binding, (1,))

    assert second.state == "success"
    second_ownership = owner.snapshot(binding).staging_ownership[1]
    assert (
        second_ownership.original_baselines
        == first_ownership.original_baselines
    )
    assert second_ownership.post_stage_entries != first_ownership.post_stage_entries
    baseline = second_ownership.original_baselines["tracked.md"].entry
    assert baseline is not None
    assert (
        f"{baseline.mode} {baseline.object_id} {baseline.stage}\ttracked.md\n".encode()
        == original_entry
    )
    refreshed = await service.start_status(
        binding,
        owner.snapshot(binding).changes,
    )
    assert refreshed.rows[0].state == "owned"
    assert refreshed.rows[0].unstage_eligible
    await service.shutdown()


@pytest.mark.asyncio
async def test_unstage_restores_saved_baseline_and_keeps_newer_worktree_edits(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    tracked = repository.path / "tracked.md"
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "tracked.md"),
    )
    runner = _RecordingRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)

    tracked.write_text("staged by Chatbook\n", encoding="utf-8")
    assert (await service.start_stage(binding, (1,))).state == "success"
    tracked.write_text("newer unstaged edit\n", encoding="utf-8")
    call_boundary = len(runner.calls)

    result = await service.start_unstage(binding, (1,))

    assert result.action == "unstage"
    assert result.state == "success"
    assert result.unstaged_group_ids == (1,)
    assert repository.run(
        "diff",
        "--cached",
        "--",
        "tracked.md",
    ).stdout == b""
    assert b"newer unstaged edit" in repository.run(
        "diff",
        "--",
        "tracked.md",
    ).stdout
    assert 1 not in owner.snapshot(binding).staging_ownership
    action_calls = runner.calls[call_boundary:]
    update_calls = [
        call
        for call in action_calls
        if "update-index" in tuple(os.fsdecode(argument) for argument in call)
    ]
    assert len(update_calls) == 1
    assert not {
        "checkout",
        "restore",
        "reset",
        "read-tree",
    }.intersection(
        os.fsdecode(argument)
        for call in action_calls
        for argument in call
    )
    await service.shutdown()


@pytest.mark.asyncio
async def test_unstage_execution_preserves_exact_filename_bytes_in_stdin(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    raw_name = b"tab\tand-newline\n.md"
    relative_path = os.fsdecode(raw_name)
    assert os.fsencode(relative_path) == raw_name
    note = repository.path / relative_path
    note.write_bytes(b"session bytes\n")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("created", relative_path),
    )
    runner = _RecordingRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    assert (await service.start_stage(binding, (1,))).state == "success"
    stdin_boundary = len(runner.stdins)

    result = await service.start_unstage(binding, (1,))

    assert result.state == "success"
    payloads = [
        payload
        for payload in runner.stdins[stdin_boundary:]
        if payload is not None
    ]
    assert payloads == [
        b"0 " + b"0" * 40 + b"\t" + raw_name + b"\0",
    ]
    assert payloads[0].count(b"\0") == 1
    assert note.read_bytes() == b"session bytes\n"
    assert repository.run(
        "ls-files",
        "-z",
        "--",
        relative_path,
    ).stdout == b""
    await service.shutdown()


@pytest.mark.asyncio
async def test_stage_preflight_blocks_partially_staged_same_path(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    tracked = repository.path / "tracked.md"
    tracked.write_text("staged version\n", encoding="utf-8")
    repository.run("add", "--", "tracked.md")
    index_before = repository.run(
        "ls-files",
        "--stage",
        "--",
        "tracked.md",
    ).stdout
    tracked.write_text("newer worktree version\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "tracked.md"),
    )
    runner = _RecordingRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)

    result = await service.start_stage(binding, (1,))

    assert result.state == "blocked"
    assert result.blocked_group_ids == (1,)
    assert not owner.snapshot(binding).staging_ownership
    assert repository.run(
        "ls-files",
        "--stage",
        "--",
        "tracked.md",
    ).stdout == index_before
    assert not any(
        "add" in tuple(os.fsdecode(argument) for argument in call)
        for call in runner.calls
    )
    await service.shutdown()


@pytest.mark.asyncio
async def test_stage_nonzero_result_claims_no_ownership(
    tmp_path: Path,
) -> None:
    class FailingAddRunner(_RecordingRunner):
        async def run(
            self,
            argv: Sequence[str | bytes],
            *,
            cwd: str,
            environment: Mapping[str, str],
            stdin: bytes | None = None,
            timeout: float | None = None,
        ) -> GitCommandResult:
            text = tuple(os.fsdecode(argument) for argument in argv)
            if "add" in text:
                self.calls.append(tuple(argv))
                return GitCommandResult(1, b"", b"index refused")
            return await super().run(
                argv,
                cwd=cwd,
                environment=environment,
                stdin=stdin,
                timeout=timeout,
            )

    repository = _disposable_repository(tmp_path)
    (repository.path / "tracked.md").write_text("changed\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "tracked.md"),
    )
    runner = FailingAddRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)

    result = await service.start_stage(binding, (1,))

    assert result.state == "error"
    assert not result.staged_group_ids
    assert not owner.snapshot(binding).staging_ownership
    assert repository.run(
        "diff",
        "--cached",
        "--name-only",
        "--",
        "tracked.md",
    ).stdout == b""
    await service.shutdown()


@pytest.mark.asyncio
async def test_stage_with_index_lock_reports_busy_without_mutating_or_taking_ownership(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    tracked = repository.path / "tracked.md"
    tracked.write_text("changed while locked\n", encoding="utf-8")
    worktree_before = tracked.read_bytes()
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "tracked.md"),
    )
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    index = Path(discovery.repository.git_dir) / "index"
    index_before = index.read_bytes()
    index_lock = Path(discovery.repository.git_dir) / "index.lock"
    lock_contents = b"external Git owns this lock\n"
    index_lock.write_bytes(lock_contents)

    try:
        result = await service.start_stage(binding, (1,))

        assert result.state == "error"
        assert result.message == "Git index busy; retry"
        assert not result.staged_group_ids
        assert not owner.snapshot(binding).staging_ownership
        assert tracked.read_bytes() == worktree_before
        assert index.read_bytes() == index_before
        assert index_lock.read_bytes() == lock_contents
    finally:
        await service.shutdown()
    assert index_lock.read_bytes() == lock_contents


@pytest.mark.asyncio
async def test_stage_update_revokes_ownership_after_external_index_change(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    tracked = repository.path / "tracked.md"
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "tracked.md"),
    )
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    tracked.write_text("owned\n", encoding="utf-8")
    assert (await service.start_stage(binding, (1,))).state == "success"
    assert 1 in owner.snapshot(binding).staging_ownership

    tracked.write_text("external index\n", encoding="utf-8")
    repository.run("add", "--", "tracked.md")
    tracked.write_text("newer worktree\n", encoding="utf-8")
    result = await service.start_stage(binding, (1,))

    assert result.state == "blocked"
    assert 1 not in owner.snapshot(binding).staging_ownership
    await service.shutdown()


@pytest.mark.asyncio
async def test_stage_update_blocks_move_to_newly_ignored_destination_before_add(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    source = repository.path / "tracked.md"
    ignored = repository.path / "ignored.md"
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "tracked.md"),
    )
    source.write_text("owned stage\n", encoding="utf-8")
    runner = _RecordingRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    assert (await service.start_stage(binding, (1,))).state == "success"
    assert 1 in owner.snapshot(binding).staging_ownership
    add_calls_before = sum(
        "add" in tuple(os.fsdecode(argument) for argument in call)
        for call in runner.calls
    )

    (repository.path / ".gitignore").write_text(
        "ignored.md\n",
        encoding="utf-8",
    )
    source.rename(ignored)
    assert owner.record_change(
        binding,
        SessionChange("moved", "tracked.md", "ignored.md"),
    )

    result = await service.start_stage(binding, (1,))

    assert result.state == "blocked"
    assert result.blocked_group_ids == (1,)
    assert sum(
        "add" in tuple(os.fsdecode(argument) for argument in call)
        for call in runner.calls
    ) == add_calls_before
    assert 1 not in owner.snapshot(binding).staging_ownership
    assert repository.run(
        "diff",
        "--cached",
        "--name-only",
        "--",
        "tracked.md",
        "ignored.md",
    ).stdout == b"tracked.md\n"
    await service.shutdown()


@pytest.mark.asyncio
async def test_stage_supports_mode_change_and_grouped_move(tmp_path: Path) -> None:
    repository = _disposable_repository(tmp_path)
    script = repository.path / "script.sh"
    moved = repository.path / "before.md"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    moved.write_text("move\n", encoding="utf-8")
    repository.run("add", "--", "script.sh", "before.md")
    repository.run("commit", "-m", "stage cases")
    script.chmod(0o755)
    moved.rename(repository.path / "after.md")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "script.sh"),
    )
    assert owner.record_change(
        binding,
        SessionChange("moved", "before.md", "after.md"),
    )
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)

    result = await service.start_stage(binding, (1, 2))

    assert result.state == "success"
    assert result.staged_group_ids == (1, 2)
    assert repository.run(
        "diff",
        "--cached",
        "--name-status",
        "--",
        "script.sh",
        "before.md",
        "after.md",
    ).stdout.splitlines() == [b"R100\tbefore.md\tafter.md", b"M\tscript.sh"]
    ownership = owner.snapshot(binding).staging_ownership
    assert set(ownership[2].post_stage_entries) == {"before.md", "after.md"}
    await service.shutdown()


@pytest.mark.asyncio
async def test_status_maps_repo_above_notes_and_supports_weird_filenames(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    notes_root = repository.path / "notes"
    notes_root.mkdir()
    requested_paths = (
        "-leading.md",
        "line\nbreak\tand snowman \N{SNOWMAN}.md",
    )
    for relative_path in requested_paths:
        (notes_root / relative_path).write_text("initial\n", encoding="utf-8")
    unrelated_note = notes_root / "unrelated.md"
    unrelated_note.write_text("initial\n", encoding="utf-8")
    unrelated_repo_file = repository.path / "outside.md"
    unrelated_repo_file.write_text("initial\n", encoding="utf-8")
    repository.run("add", "--", "notes", "outside.md")
    repository.run("commit", "-m", "notes")
    for relative_path in requested_paths:
        (notes_root / relative_path).write_text("changed\n", encoding="utf-8")
    unrelated_note.write_text("unrelated change\n", encoding="utf-8")
    unrelated_repo_file.write_text("unrelated change\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(notes_root)
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)

    status = await service.start_status(
        binding,
        tuple(
            _change(index, "modified", relative_path)
            for index, relative_path in enumerate(requested_paths, start=1)
        ),
    )

    assert status.state == "ready"
    assert status.repository == discovery.repository
    assert status.head is not None
    assert {row.group.current_path for row in status.rows} == set(
        requested_paths
    )
    assert {row.state for row in status.rows} == {"unstaged"}
    assert all(
        "unrelated" not in row.group.current_path
        and "outside" not in row.group.current_path
        for row in status.rows
    )
    await service.shutdown()


@pytest.mark.asyncio
async def test_configured_clean_filter_requires_trust_and_stage_preserves_worktree(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    filter_executable = repository.path / ".git" / "chatbook-clean-filter"
    sentinel = tmp_path / "filter-invocations.log"
    filter_executable.write_text(
        '#!/bin/sh\nprintf "invoked\\\\n" >> "$1"\ncat\n',
        encoding="utf-8",
    )
    filter_executable.chmod(
        filter_executable.stat().st_mode | stat.S_IXUSR,
    )
    repository.run(
        "config",
        "--local",
        "filter.chatbook-test.clean",
        f"{shlex.quote(str(filter_executable))} {shlex.quote(str(sentinel))}",
    )
    repository.run(
        "config",
        "--local",
        "filter.chatbook-test.required",
        "true",
    )
    (repository.path / ".gitattributes").write_text(
        "tracked.md filter=chatbook-test\n",
        encoding="utf-8",
    )
    repository.run("add", "--", ".gitattributes")
    repository.run("commit", "-m", "configure repository-local clean filter")
    sentinel.unlink(missing_ok=True)

    tracked = repository.path / "tracked.md"
    tracked.write_text("filter-visible edit\n", encoding="utf-8")
    worktree_before = tracked.read_bytes()
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "tracked.md"),
    )
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )

    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert not sentinel.exists()
    with pytest.raises(
        GitStatusAdmissionError,
        match="Repository trust is required before Git status",
    ):
        await service.start_status(
            binding,
            owner.snapshot(binding).changes,
        )
    assert not sentinel.exists()

    assert owner.publish_trust(binding, discovery.repository)
    status = await service.start_status(
        binding,
        owner.snapshot(binding).changes,
    )
    assert status.state == "ready"
    assert status.rows[0].stage_action == "stage"
    assert tracked.read_bytes() == worktree_before

    sentinel.unlink(missing_ok=True)
    assert not sentinel.exists()
    staged = await service.start_stage(binding, (1,))

    assert staged.state == "success"
    assert sentinel.read_text(encoding="utf-8").splitlines()
    assert tracked.read_bytes() == worktree_before
    await service.shutdown()


@pytest.mark.asyncio
async def test_status_reports_matching_ignored_session_path(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    (repository.path / ".gitignore").write_text(
        "ignored.md\n",
        encoding="utf-8",
    )
    repository.run("add", "--", ".gitignore")
    repository.run("commit", "-m", "ignore")
    (repository.path / "ignored.md").write_text("ignored\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)

    status = await service.start_status(
        binding,
        (_change(1, "created", "ignored.md"),),
    )

    assert status.state == "ready"
    assert len(status.rows) == 1
    assert status.rows[0].state == "ignored"
    await service.shutdown()


@pytest.mark.asyncio
async def test_status_fails_closed_for_active_sparse_checkout(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    repository.run("sparse-checkout", "init", "--cone", "--sparse-index")
    (repository.path / "tracked.md").write_text("changed\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)

    status = await service.start_status(
        binding,
        (_change(1, "modified", "tracked.md"),),
    )

    assert status.state == "unavailable"
    assert status.message is not None
    assert "sparse" in status.message.lower()
    await service.shutdown()


@pytest.mark.asyncio
async def test_status_blocks_endpoint_beneath_nested_worktree(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    nested = repository.path / "nested"
    nested.mkdir()
    repository.run("init", "--initial-branch=main", cwd=nested)
    repository.run("config", "--local", "user.name", "Nested", cwd=nested)
    repository.run(
        "config",
        "--local",
        "user.email",
        "nested@example.invalid",
        cwd=nested,
    )
    (nested / "note.md").write_text("nested\n", encoding="utf-8")
    repository.run("add", "--", "note.md", cwd=nested)
    repository.run("commit", "-m", "nested", cwd=nested)
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)

    status = await service.start_status(
        binding,
        (_change(1, "modified", "nested/note.md"),),
    )

    assert status.state == "ready"
    assert len(status.rows) == 1
    assert status.rows[0].state == "nested_repository"
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocked_case",
    ["ignored", "nested", "sparse", "unsafe", "semantic", "closure"],
)
async def test_stage_preflight_refusals_start_no_add_child(
    tmp_path: Path,
    blocked_case: str,
) -> None:
    repository = _disposable_repository(tmp_path)
    action = "modified"
    relative_path = "tracked.md"
    if blocked_case == "ignored":
        (repository.path / ".gitignore").write_text(
            "ignored.md\n",
            encoding="utf-8",
        )
        repository.run("add", "--", ".gitignore")
        repository.run("commit", "-m", "ignore")
        relative_path = "ignored.md"
        action = "created"
        (repository.path / relative_path).write_text("ignored\n", encoding="utf-8")
    elif blocked_case == "nested":
        nested = repository.path / "nested"
        nested.mkdir()
        repository.run("init", "--initial-branch=main", cwd=nested)
        (nested / "note.md").write_text("nested\n", encoding="utf-8")
        relative_path = "nested/note.md"
    elif blocked_case == "sparse":
        repository.run("sparse-checkout", "init", "--cone", "--sparse-index")
        (repository.path / relative_path).write_text("changed\n", encoding="utf-8")
    elif blocked_case == "unsafe":
        relative_path = "directory"
        action = "created"
        (repository.path / relative_path).mkdir()
    elif blocked_case == "semantic":
        repository.run(
            "update-index",
            "--assume-unchanged",
            "--",
            relative_path,
        )
        (repository.path / relative_path).write_text("changed\n", encoding="utf-8")
    else:
        ancestor = repository.path / "collision"
        ancestor.write_text("tracked ancestor\n", encoding="utf-8")
        repository.run("add", "--", "collision")
        repository.run("commit", "-m", "collision")
        ancestor.unlink()
        ancestor.mkdir()
        (ancestor / "child.md").write_text("child\n", encoding="utf-8")
        relative_path = "collision/child.md"
        action = "created"

    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange(action, relative_path),  # type: ignore[arg-type]
    )
    runner = _RecordingRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)

    result = await service.start_stage(binding, (1,))

    assert result.state == "blocked"
    assert result.blocked_group_ids == (1,)
    assert not owner.snapshot(binding).staging_ownership
    assert not any(
        "add" in tuple(os.fsdecode(argument) for argument in call)
        for call in runner.calls
    )
    await service.shutdown()


@pytest.mark.asyncio
async def test_stage_bulk_stages_only_eligible_and_reports_clean_and_blocked(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    clean = repository.path / "clean.md"
    clean.write_text("clean\n", encoding="utf-8")
    (repository.path / ".gitignore").write_text("ignored.md\n", encoding="utf-8")
    repository.run("add", "--", "clean.md", ".gitignore")
    repository.run("commit", "-m", "bulk baseline")
    (repository.path / "tracked.md").write_text("changed\n", encoding="utf-8")
    (repository.path / "ignored.md").write_text("ignored\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    for change in (
        SessionChange("modified", "tracked.md"),
        SessionChange("created", "ignored.md"),
        SessionChange("modified", "clean.md"),
    ):
        assert owner.record_change(binding, change)
    runner = _RecordingRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)

    result = await service.start_stage(binding, (1, 2, 3))

    assert result.state == "success"
    assert result.staged_group_ids == (1,)
    assert result.blocked_group_ids == (2,)
    assert result.clean_group_ids == (3,)
    add_calls = [
        call
        for call in runner.calls
        if "add" in tuple(os.fsdecode(argument) for argument in call)
    ]
    assert len(add_calls) == 1
    boundary = add_calls[0].index("--")
    assert tuple(add_calls[0][boundary + 1 :]) == (b"tracked.md",)
    await service.shutdown()


@pytest.mark.asyncio
async def test_stage_update_expands_chained_move_topology_without_noop_ownership(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    notes_root = repository.path / "notes"
    notes_root.mkdir()
    source_name = "-before[1].md"
    transient_name = "middle*.md"
    final_name = "final?.md"
    source = notes_root / source_name
    source.write_text("baseline\n", encoding="utf-8")
    repository.run("add", "--", f"notes/{source_name}")
    repository.run("commit", "-m", "move baseline")
    repository.run("config", "--local", "add.ignoreErrors", "true")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(notes_root)
    assert owner.record_change(
        binding,
        SessionChange("modified", source_name),
    )
    source.write_text("first stage\n", encoding="utf-8")
    runner = _RecordingRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    assert (await service.start_stage(binding, (1,))).state == "success"
    first = owner.snapshot(binding).staging_ownership[1]

    transient = notes_root / transient_name
    final = notes_root / final_name
    source.rename(transient)
    assert owner.record_change(
        binding,
        SessionChange("moved", source_name, transient_name),
    )
    transient.rename(final)
    assert owner.record_change(
        binding,
        SessionChange("moved", transient_name, final_name),
    )

    result = await service.start_stage(binding, (1,))

    assert result.state == "success"
    updated = owner.snapshot(binding).staging_ownership[1]
    assert updated.approved_endpoint_topology == (
        source_name,
        transient_name,
        final_name,
    )
    assert (
        updated.original_baselines[f"notes/{source_name}"]
        == first.original_baselines[f"notes/{source_name}"]
    )
    assert updated.original_baselines[f"notes/{final_name}"].entry is None
    assert f"notes/{transient_name}" not in updated.original_baselines
    assert f"notes/{transient_name}" not in updated.post_stage_entries
    add_calls = [
        tuple(os.fsdecode(argument) for argument in call)
        for call in runner.calls
        if "add" in tuple(os.fsdecode(argument) for argument in call)
    ]
    assert add_calls[-1][:7] == (
        repository.git,
        "--literal-pathspecs",
        "-c",
        "add.ignoreErrors=false",
        "add",
        "--all",
        "--",
    )
    assert set(add_calls[-1][7:]) == {
        f"notes/{source_name}",
        f"notes/{final_name}",
    }
    await service.shutdown()


@pytest.mark.asyncio
async def test_stage_supports_restored_session_change(tmp_path: Path) -> None:
    repository = _disposable_repository(tmp_path)
    (repository.path / "tracked.md").write_text(
        "restored version\n",
        encoding="utf-8",
    )
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("restored", "tracked.md"),
    )
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)

    result = await service.start_stage(binding, (1,))

    assert result.state == "success"
    assert result.staged_group_ids == (1,)
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("head_kind", ["detached", "unborn"])
async def test_unstage_uses_saved_baseline_for_detached_and_unborn_head(
    tmp_path: Path,
    head_kind: str,
) -> None:
    repository = _disposable_repository(
        tmp_path,
        commit=head_kind != "unborn",
    )
    if head_kind == "detached":
        repository.run("checkout", "--detach")
        relative_path = "tracked.md"
        action = "modified"
    else:
        relative_path = "created.md"
        action = "created"
    note = repository.path / relative_path
    note.write_text("session change\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(binding, SessionChange(action, relative_path))
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert discovery.head is not None
    assert discovery.head.kind == head_kind
    assert owner.publish_trust(binding, discovery.repository)
    assert (await service.start_stage(binding, (1,))).state == "success"

    result = await service.start_unstage(binding, (1,))

    assert result.state == "success"
    assert result.unstaged_group_ids == (1,)
    assert repository.run("diff", "--cached").stdout == b""
    assert note.read_text(encoding="utf-8") == "session change\n"
    await service.shutdown()


@pytest.mark.asyncio
async def test_unstage_bulk_restores_modify_create_delete_restore_mode_and_move(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    deleted = repository.path / "deleted.md"
    restored = repository.path / "restored.md"
    script = repository.path / "script.sh"
    before = repository.path / "before.md"
    for path, body in (
        (deleted, "delete baseline\n"),
        (restored, "restore baseline\n"),
        (script, "#!/bin/sh\n"),
        (before, "move baseline\n"),
    ):
        path.write_text(body, encoding="utf-8")
    repository.run(
        "add",
        "--",
        "deleted.md",
        "restored.md",
        "script.sh",
        "before.md",
    )
    repository.run("commit", "-m", "unstage matrix baseline")

    (repository.path / "tracked.md").write_text(
        "modified\n",
        encoding="utf-8",
    )
    (repository.path / "created.md").write_text("created\n", encoding="utf-8")
    deleted.unlink()
    restored.write_text("restored session version\n", encoding="utf-8")
    script.chmod(0o755)
    before.rename(repository.path / "after.md")

    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    changes = (
        SessionChange("modified", "tracked.md"),
        SessionChange("created", "created.md"),
        SessionChange("deleted", "deleted.md"),
        SessionChange("restored", "restored.md"),
        SessionChange("modified", "script.sh"),
        SessionChange("moved", "before.md", "after.md"),
    )
    for change in changes:
        assert owner.record_change(binding, change)
    runner = _RecordingRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    requested = tuple(range(1, 7))
    assert (
        await service.start_stage(binding, requested)
    ).staged_group_ids == requested
    assert repository.run("diff", "--cached").stdout

    result = await service.start_unstage(binding, requested)

    assert result.state == "success"
    assert result.unstaged_group_ids == requested
    assert repository.run("diff", "--cached").stdout == b""
    assert not owner.snapshot(binding).staging_ownership
    assert (repository.path / "created.md").is_file()
    assert not deleted.exists()
    assert (repository.path / "after.md").is_file()
    assert not before.exists()
    assert stat.S_IMODE(script.stat().st_mode) == 0o755
    update_calls = [
        call
        for call in runner.calls
        if "update-index" in tuple(os.fsdecode(argument) for argument in call)
    ]
    assert len(update_calls) == 1
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["file_to_directory", "directory_to_file"])
async def test_unstage_reverses_owned_file_directory_replacement_exactly(
    tmp_path: Path,
    direction: str,
) -> None:
    repository = _disposable_repository(tmp_path)
    tree = repository.path / "tree"
    child = tree / "owned.md"
    if direction == "file_to_directory":
        tree.write_text("file baseline\n", encoding="utf-8")
        repository.run("add", "--", "tree")
        repository.run("commit", "-m", "file baseline")
        baseline_entries = _index_entries(repository)
        tree.unlink()
        tree.mkdir()
        child.write_text("directory replacement\n", encoding="utf-8")
        change = SessionChange("moved", "tree", "tree/owned.md")
        baseline_path = "tree"
        replacement_path = "tree/owned.md"
    else:
        tree.mkdir()
        child.write_text("directory baseline\n", encoding="utf-8")
        repository.run("add", "--", "tree/owned.md")
        repository.run("commit", "-m", "directory baseline")
        baseline_entries = _index_entries(repository)
        child.unlink()
        tree.rmdir()
        tree.write_text("file replacement\n", encoding="utf-8")
        change = SessionChange("moved", "tree/owned.md", "tree")
        baseline_path = "tree/owned.md"
        replacement_path = "tree"
    repository.run("add", "--all", "--", "tree")
    post_entries = _index_entries(repository)

    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(binding, change)
    runner = _RecordingRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert discovery.head is not None
    assert owner.publish_trust(binding, discovery.repository)
    group = coalesce_session_changes(owner.snapshot(binding).changes)[0]
    assert owner.publish_ownership(
        binding,
        {
            1: StagingOwnership(
                repository=discovery.repository,
                head=discovery.head,
                approved_endpoint_topology=group.endpoints,
                approved_move_edges=group.move_edges,
                approved_current_path=group.current_path,
                original_baselines={
                    baseline_path: IndexBaseline(
                        baseline_entries[baseline_path],
                    ),
                },
                post_stage_entries={
                    baseline_path: None,
                    replacement_path: post_entries[replacement_path],
                },
            )
        },
    )
    stdin_boundary = len(runner.stdins)

    result = await service.start_unstage(binding, (1,))

    assert result.state == "success"
    assert repository.run("diff", "--cached").stdout == b""
    payloads = [
        payload
        for payload in runner.stdins[stdin_boundary:]
        if payload is not None
    ]
    assert len(payloads) == 1
    assert payloads[0].startswith(b"0 " + b"0" * 40 + b"\t")
    if direction == "file_to_directory":
        assert tree.is_dir()
        assert child.read_text(encoding="utf-8") == "directory replacement\n"
    else:
        assert tree.is_file()
        assert tree.read_text(encoding="utf-8") == "file replacement\n"
    await service.shutdown()


@pytest.mark.asyncio
async def test_unstage_blocks_unexpected_external_replacement_closure_before_stdin(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    tree = repository.path / "tree"
    tree.write_text("file baseline\n", encoding="utf-8")
    repository.run("add", "--", "tree")
    repository.run("commit", "-m", "file baseline")
    baseline_entries = _index_entries(repository)
    tree.unlink()
    tree.mkdir()
    owned_child = tree / "owned.md"
    owned_child.write_text("owned replacement\n", encoding="utf-8")
    repository.run("add", "--all", "--", "tree")
    owned_post_entries = _index_entries(repository)
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("moved", "tree", "tree/owned.md"),
    )
    runner = _RecordingRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert discovery.head is not None
    assert owner.publish_trust(binding, discovery.repository)
    group = coalesce_session_changes(owner.snapshot(binding).changes)[0]
    assert owner.publish_ownership(
        binding,
        {
            1: StagingOwnership(
                repository=discovery.repository,
                head=discovery.head,
                approved_endpoint_topology=group.endpoints,
                approved_move_edges=group.move_edges,
                approved_current_path=group.current_path,
                original_baselines={
                    "tree": IndexBaseline(
                        baseline_entries["tree"],
                    ),
                },
                post_stage_entries={
                    "tree": None,
                    "tree/owned.md": owned_post_entries["tree/owned.md"],
                },
            )
        },
    )
    external = tree / "external.md"
    external.write_text("external index entry\n", encoding="utf-8")
    repository.run("add", "--", "tree/external.md")
    external_index_before = repository.run(
        "ls-files",
        "--stage",
        "--",
        "tree/external.md",
    ).stdout
    call_boundary = len(runner.calls)
    stdin_boundary = len(runner.stdins)

    result = await service.start_unstage(binding, (1,))

    assert result.state == "blocked"
    assert result.blocked_group_ids == (1,)
    assert not any(
        "update-index" in tuple(os.fsdecode(argument) for argument in call)
        for call in runner.calls[call_boundary:]
    )
    assert all(payload is None for payload in runner.stdins[stdin_boundary:])
    assert repository.run(
        "ls-files",
        "--stage",
        "--",
        "tree/external.md",
    ).stdout == external_index_before
    assert 1 in owner.snapshot(binding).staging_ownership
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("external_change", ["index", "semantic", "head"])
async def test_unstage_revokes_external_index_semantic_or_head_change(
    tmp_path: Path,
    external_change: str,
) -> None:
    repository = _disposable_repository(tmp_path)
    tracked = repository.path / "tracked.md"
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "tracked.md"),
    )
    runner = _RecordingRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    tracked.write_text("owned stage\n", encoding="utf-8")
    assert (await service.start_stage(binding, (1,))).state == "success"
    if external_change == "index":
        tracked.write_text("external staged\n", encoding="utf-8")
        repository.run("add", "--", "tracked.md")
    elif external_change == "semantic":
        repository.run("update-index", "--assume-unchanged", "tracked.md")
    else:
        repository.run("commit", "-m", "external head change")
    index_before = repository.run(
        "ls-files",
        "--stage",
        "-v",
        "--",
        "tracked.md",
    ).stdout
    call_boundary = len(runner.calls)
    stdin_boundary = len(runner.stdins)

    result = await service.start_unstage(binding, (1,))

    assert result.state == "blocked"
    assert not owner.snapshot(binding).staging_ownership
    assert not any(
        "update-index" in tuple(os.fsdecode(argument) for argument in call)
        for call in runner.calls[call_boundary:]
    )
    assert all(payload is None for payload in runner.stdins[stdin_boundary:])
    assert repository.run(
        "ls-files",
        "--stage",
        "-v",
        "--",
        "tracked.md",
    ).stdout == index_before
    await service.shutdown()


@pytest.mark.asyncio
async def test_unstage_raw_conflict_stages_block_before_stdin_and_revoke(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    tracked = repository.path / "tracked.md"
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "tracked.md"),
    )
    runner = _RecordingRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    tracked.write_text("owned stage\n", encoding="utf-8")
    assert (await service.start_stage(binding, (1,))).state == "success"
    owned_entry = owner.snapshot(binding).staging_ownership[1].post_stage_entries[
        "tracked.md"
    ]
    baseline_entry = owner.snapshot(binding).staging_ownership[
        1
    ].original_baselines["tracked.md"].entry
    assert owned_entry is not None
    assert baseline_entry is not None
    conflict_payload = (
        f"0 {'0' * 40}\ttracked.md\n"
        f"100644 {baseline_entry.object_id} 1\ttracked.md\n"
        f"100644 {owned_entry.object_id} 2\ttracked.md\n"
        f"100644 {baseline_entry.object_id} 3\ttracked.md\n"
    ).encode()
    subprocess.run(
        [repository.git, "update-index", "--index-info"],
        cwd=repository.path,
        env=repository.environment,
        input=conflict_payload,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    conflict_before = repository.run(
        "ls-files",
        "--stage",
        "--",
        "tracked.md",
    ).stdout
    call_boundary = len(runner.calls)
    stdin_boundary = len(runner.stdins)

    result = await service.start_unstage(binding, (1,))

    assert result.state == "blocked"
    assert not owner.snapshot(binding).staging_ownership
    assert not any(
        "update-index" in tuple(os.fsdecode(argument) for argument in call)
        for call in runner.calls[call_boundary:]
    )
    assert all(payload is None for payload in runner.stdins[stdin_boundary:])
    assert repository.run(
        "ls-files",
        "--stage",
        "--",
        "tracked.md",
    ).stdout == conflict_before
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("relation", "baseline_path", "conflict_path"),
    (
        ("ancestor", "tree/owned.md", "tree"),
        ("descendant", "tree", "tree/conflict.md"),
        ("exact", "tree", "tree"),
    ),
)
async def test_unstage_conflict_replacement_closure_blocks_before_stdin(
    tmp_path: Path,
    relation: str,
    baseline_path: str,
    conflict_path: str,
) -> None:
    repository = _disposable_repository(tmp_path)
    baseline = repository.path / baseline_path
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(f"{relation} baseline\n", encoding="utf-8")
    repository.run("add", "--", baseline_path)
    repository.run("commit", "-m", f"{relation} baseline")
    baseline_entries = _index_entries(repository)
    repository.run("rm", "--cached", "--", baseline_path)

    if relation == "ancestor":
        baseline.unlink()
        baseline.parent.rmdir()
        (repository.path / conflict_path).write_text(
            "ancestor conflict\n",
            encoding="utf-8",
        )
    elif relation == "descendant":
        baseline.unlink()
        baseline.mkdir()
        conflict = repository.path / conflict_path
        conflict.write_text("descendant conflict\n", encoding="utf-8")

    conflict_payload = (
        f"100644 {baseline_entries[baseline_path].object_id} 1\t{conflict_path}\n"
        f"100644 {baseline_entries['tracked.md'].object_id} 2\t{conflict_path}\n"
        f"100644 {baseline_entries[baseline_path].object_id} 3\t{conflict_path}\n"
    ).encode()
    subprocess.run(
        [repository.git, "update-index", "--index-info"],
        cwd=repository.path,
        env=repository.environment,
        input=conflict_payload,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    conflict_before = repository.run(
        "ls-files",
        "--stage",
        "--",
        conflict_path,
    ).stdout
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("deleted", baseline_path),
    )
    runner = _RecordingRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert discovery.head is not None
    assert owner.publish_trust(binding, discovery.repository)
    group = coalesce_session_changes(owner.snapshot(binding).changes)[0]
    assert owner.publish_ownership(
        binding,
        {
            1: StagingOwnership(
                repository=discovery.repository,
                head=discovery.head,
                approved_endpoint_topology=group.endpoints,
                approved_move_edges=group.move_edges,
                approved_current_path=group.current_path,
                original_baselines={
                    baseline_path: IndexBaseline(
                        baseline_entries[baseline_path],
                    ),
                },
                post_stage_entries={baseline_path: None},
            )
        },
    )
    call_boundary = len(runner.calls)
    stdin_boundary = len(runner.stdins)

    result = await service.start_unstage(binding, (1,))

    assert result.state == "blocked"
    assert result.blocked_group_ids == (1,)
    assert not any(
        "update-index" in tuple(os.fsdecode(argument) for argument in call)
        for call in runner.calls[call_boundary:]
    )
    assert all(payload is None for payload in runner.stdins[stdin_boundary:])
    assert repository.run(
        "ls-files",
        "--stage",
        "--",
        conflict_path,
    ).stdout == conflict_before
    await service.shutdown()


@pytest.mark.asyncio
async def test_unstage_preserves_unrelated_preexisting_conflict_stages(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    unrelated = repository.path / "unrelated.md"
    unrelated.write_text("unrelated baseline\n", encoding="utf-8")
    repository.run("add", "--", "unrelated.md")
    repository.run("commit", "-m", "unrelated baseline")
    baseline_entries = _index_entries(repository)
    conflict_payload = (
        f"0 {'0' * 40}\tunrelated.md\n"
        f"100644 {baseline_entries['unrelated.md'].object_id} 1\tunrelated.md\n"
        f"100644 {baseline_entries['tracked.md'].object_id} 2\tunrelated.md\n"
        f"100644 {baseline_entries['unrelated.md'].object_id} 3\tunrelated.md\n"
    ).encode()
    subprocess.run(
        [repository.git, "update-index", "--index-info"],
        cwd=repository.path,
        env=repository.environment,
        input=conflict_payload,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    conflict_before = repository.run(
        "ls-files",
        "--stage",
        "--",
        "unrelated.md",
    ).stdout
    tracked = repository.path / "tracked.md"
    tracked.write_text("owned stage\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "tracked.md"),
    )
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    assert (await service.start_stage(binding, (1,))).state == "success"

    result = await service.start_unstage(binding, (1,))

    assert result.state == "success"
    assert result.unstaged_group_ids == (1,)
    assert repository.run(
        "ls-files",
        "--stage",
        "--",
        "unrelated.md",
    ).stdout == conflict_before
    assert repository.run(
        "diff",
        "--cached",
        "--",
        "tracked.md",
    ).stdout == b""
    await service.shutdown()


@pytest.mark.asyncio
async def test_unstage_topology_change_requires_stage_update_without_revocation(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    tracked = repository.path / "tracked.md"
    tracked.write_text("owned stage\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "tracked.md"),
    )
    runner = _RecordingRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    assert (await service.start_stage(binding, (1,))).state == "success"
    tracked.rename(repository.path / "renamed.md")
    assert owner.record_change(
        binding,
        SessionChange("moved", "tracked.md", "renamed.md"),
    )
    call_boundary = len(runner.calls)

    result = await service.start_unstage(binding, (1,))

    assert result.state == "blocked"
    assert 1 in owner.snapshot(binding).staging_ownership
    assert not any(
        "update-index" in tuple(os.fsdecode(argument) for argument in call)
        for call in runner.calls[call_boundary:]
    )
    status = await service.start_status(
        binding,
        owner.snapshot(binding).changes,
    )
    assert status.rows[0].state == "owned_topology_changed"
    assert status.rows[0].stage_action == "stage_update"
    assert not status.rows[0].unstage_eligible
    await service.shutdown()


@pytest.mark.asyncio
async def test_stage_update_then_unstage_restores_earliest_saved_baseline(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    tracked = repository.path / "tracked.md"
    baseline = repository.run(
        "ls-files",
        "--stage",
        "--",
        "tracked.md",
    ).stdout
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "tracked.md"),
    )
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    tracked.write_text("first stage\n", encoding="utf-8")
    assert (await service.start_stage(binding, (1,))).state == "success"
    tracked.write_text("stage update\n", encoding="utf-8")
    assert (await service.start_stage(binding, (1,))).state == "success"

    result = await service.start_unstage(binding, (1,))

    assert result.state == "success"
    assert repository.run(
        "ls-files",
        "--stage",
        "--",
        "tracked.md",
    ).stdout == baseline
    assert tracked.read_text(encoding="utf-8") == "stage update\n"
    await service.shutdown()


@pytest.mark.asyncio
async def test_selected_and_bulk_unstage_include_only_valid_owned_groups(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    second = repository.path / "second.md"
    second.write_text("second baseline\n", encoding="utf-8")
    repository.run("add", "--", "second.md")
    repository.run("commit", "-m", "second baseline")
    (repository.path / "tracked.md").write_text("first change\n", encoding="utf-8")
    second.write_text("second change\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "tracked.md"),
    )
    assert owner.record_change(
        binding,
        SessionChange("modified", "second.md"),
    )
    service = FileNotesGitService(
        owner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    assert (await service.start_stage(binding, (1, 2))).state == "success"

    selected = await service.start_unstage(binding, (1,))

    assert selected.unstaged_group_ids == (1,)
    assert set(owner.snapshot(binding).staging_ownership) == {2}
    assert repository.run(
        "diff",
        "--cached",
        "--name-only",
    ).stdout == b"second.md\n"

    bulk = await service.start_unstage(binding, (1, 2))

    assert bulk.state == "success"
    assert bulk.unstaged_group_ids == (2,)
    assert bulk.blocked_group_ids == (1,)
    assert repository.run("diff", "--cached").stdout == b""
    assert not owner.snapshot(binding).staging_ownership
    await service.shutdown()


class _PostflightUnstageRaceRunner(_RecordingRunner):
    def __init__(self, repository: _Repository, path: Path) -> None:
        super().__init__()
        self.repository = repository
        self.path = path
        self.raced = False

    async def run(
        self,
        argv: Sequence[str | bytes],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        result = await super().run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
        )
        text = tuple(os.fsdecode(argument) for argument in argv)
        if "update-index" in text and stdin is not None and not self.raced:
            self.raced = True
            self.repository.run("add", "--", self.path.name)
        return result


@pytest.mark.asyncio
async def test_unstage_postflight_mismatch_revokes_without_claiming_success(
    tmp_path: Path,
) -> None:
    repository = _disposable_repository(tmp_path)
    tracked = repository.path / "tracked.md"
    second = repository.path / "second.md"
    second.write_text("second baseline\n", encoding="utf-8")
    repository.run("add", "--", "second.md")
    repository.run("commit", "-m", "second baseline")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(repository.path)
    assert owner.record_change(
        binding,
        SessionChange("modified", "tracked.md"),
    )
    assert owner.record_change(
        binding,
        SessionChange("modified", "second.md"),
    )
    runner = _PostflightUnstageRaceRunner(repository, tracked)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=repository.git,
        environment=repository.service_environment,
    )
    discovery = await service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    tracked.write_text("owned stage\n", encoding="utf-8")
    second.write_text("second owned stage\n", encoding="utf-8")
    assert (await service.start_stage(binding, (1, 2))).state == "success"
    tracked.write_text("external race\n", encoding="utf-8")

    result = await service.start_unstage(binding, (1,))

    assert result.state == "uncertain"
    assert result.unstaged_group_ids == ()
    assert set(owner.snapshot(binding).staging_ownership) == {2}
    assert set(repository.run(
        "diff",
        "--cached",
        "--name-only",
    ).stdout.splitlines()) == {b"second.md", b"tracked.md"}
    await service.shutdown()
