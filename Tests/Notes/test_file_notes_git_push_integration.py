from __future__ import annotations

import asyncio
import os
import shutil
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

import tldw_chatbook.Notes.file_notes_git_network as git_network
import tldw_chatbook.Notes.file_notes_git_push as push_contracts
from tldw_chatbook.Notes.file_notes_git_service import (
    AsyncGitProcessRunner,
    FileNotesGitService,
    GitArg,
    GitCommandResult,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.file_notes_session_owner import (
    CommitPublication,
    FileNotesSessionOwner,
    FileSystemIdentity,
    HeadIdentity,
    IndexBaseline,
    IndexEntry,
    PushIncludedNote,
    RepositoryIdentity,
    SessionBinding,
    SessionChange,
    StagingOwnership,
)


BRANCH_REF = "refs/heads/main"


def _git(
    repository: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
    environment: Mapping[str, str] | None = None,
) -> bytes:
    result = subprocess.run(
        (shutil.which("git") or "git", "-C", str(repository), *arguments),
        input=input_bytes,
        env=None if environment is None else dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, (
        f"git {arguments!r} failed: {result.stderr.decode(errors='replace')}"
    )
    return result.stdout


def _git_dir(git_dir: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        (shutil.which("git") or "git", f"--git-dir={git_dir}", *arguments),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, (
        f"git-dir {arguments!r} failed: {result.stderr.decode(errors='replace')}"
    )
    return result.stdout


def _identity(path: Path) -> FileSystemIdentity:
    metadata = path.stat()
    return FileSystemIdentity(metadata.st_dev, metadata.st_ino)


def _init_source_and_destination(
    tmp_path: Path,
    *,
    candidate_count: int = 1,
) -> tuple[Path, Path, tuple[str, ...]]:
    source = tmp_path / "notes"
    destination = tmp_path / "destination.git"
    _git(tmp_path, "init", "--initial-branch=main", str(source))
    _git(source, "config", "user.name", "Chatbook Test")
    _git(source, "config", "user.email", "chatbook@example.test")
    note = source / "note.md"
    note.write_text("parent\n", encoding="utf-8")
    _git(source, "add", "note.md")
    _git(source, "commit", "-m", "Parent")
    parent = _git(source, "rev-parse", "HEAD").decode().strip()
    _git(tmp_path, "init", "--bare", str(destination))
    _git(source, "remote", "add", "origin", str(destination))
    _git(source, "push", "--set-upstream", "origin", "main")
    commits = [parent]
    for number in range(1, candidate_count + 1):
        note.write_text(f"candidate {number}\n", encoding="utf-8")
        _git(source, "add", "note.md")
        _git(source, "commit", "-m", f"Candidate {number}")
        commits.append(_git(source, "rev-parse", "HEAD").decode().strip())
    _git_dir(destination, "update-ref", "refs/heads/untouched", parent)
    _git_dir(destination, "tag", "untouched-tag", parent)
    return source, destination, tuple(commits)


def _owner_for_candidate(
    source: Path,
    parent_oid: str,
    candidate_oid: str,
) -> tuple[FileNotesSessionOwner, SessionBinding]:
    git_dir = source / ".git"
    repository = RepositoryIdentity(
        worktree_root=str(source.resolve()),
        git_dir=str(git_dir.resolve()),
        git_common_dir=str(git_dir.resolve()),
        worktree_identity=_identity(source),
        git_dir_identity=_identity(git_dir),
        git_common_dir_identity=_identity(git_dir),
    )
    owner = FileNotesSessionOwner()
    binding = owner.select_root(source)
    assert owner.record_change(binding, SessionChange("modified", "note.md"))
    assert owner.publish_trust(binding, repository)
    parent_blob = _git(source, "rev-parse", f"{parent_oid}:note.md").decode().strip()
    candidate_blob = _git(
        source,
        "rev-parse",
        f"{candidate_oid}:note.md",
    ).decode().strip()
    old_head = HeadIdentity.attached(BRANCH_REF, parent_oid)
    ownership = StagingOwnership(
        repository=repository,
        head=old_head,
        approved_endpoint_topology=("note.md",),
        approved_move_edges=(),
        approved_current_path="note.md",
        original_baselines={
            "note.md": IndexBaseline(
                IndexEntry("note.md", "100644", parent_blob)
            )
        },
        post_stage_entries={
            "note.md": IndexEntry("note.md", "100644", candidate_blob)
        },
    )
    assert owner.publish_ownership(binding, {1: ownership})
    lease = owner.try_acquire_mutation(binding)
    assert lease is not None
    reviewed = owner._capture_commit_authority_after_review(
        lease,
        binding=binding,
        authority_generation=owner.snapshot(binding).git_authority_generation,
        repository=repository,
        head=old_head,
        group_sequence_ids={1: (1,)},
        subject="Guarded candidate",
        included_notes=(PushIncludedNote(1, "note.md"),),
        change_types=("Modified",),
    )
    assert reviewed is not None
    capture = owner._recapture_commit_authority(
        lease,
        prior_capture=reviewed,
    )
    assert capture is not None
    published = owner.publish_commit_outcome(
        lease,
        capture,
        CommitPublication(
            "succeeded",
            new_head=HeadIdentity.attached(BRANCH_REF, candidate_oid),
            retired_sequence_ids=(1,),
            candidate_seed=capture._candidate_seed,
        ),
    )
    assert published.published
    lease.release()
    return owner, binding


class _RecordingRunner(AsyncGitProcessRunner):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
        self.results: list[GitCommandResult] = []

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
        owned_process_tree: bool = False,
    ) -> GitCommandResult:
        self.calls.append(
            (
                tuple(os.fsdecode(argument) for argument in argv),
                {
                    "cwd": cwd,
                    "environment": dict(environment),
                    "stdin": stdin,
                    "timeout": timeout,
                    "owned_process_tree": owned_process_tree,
                },
            )
        )
        result = await super().run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
            on_spawn=on_spawn,
            cancel_before_spawn=cancel_before_spawn,
            owned_process_tree=owned_process_tree,
        )
        self.results.append(result)
        return result


class _Barrier:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self) -> None:
        self.entered.set()
        await self.release.wait()


def _service(
    owner: FileNotesSessionOwner,
    runner: _RecordingRunner,
    *,
    barrier: _Barrier | None = None,
    environment: Mapping[str, str] | None = None,
) -> FileNotesGitService:
    executable = shutil.which("git")
    assert executable is not None
    exec_path = subprocess.run(
        (executable, "--exec-path"),
        env={"LC_ALL": "C", "PATH": os.defpath},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.decode().strip()
    binding = owner.current_binding()
    assert binding is not None
    temporary_parent = Path(binding.root_key).parent / "network-contexts"
    temporary_parent.mkdir(mode=0o700, exist_ok=True)
    network_environment = {} if environment is None else environment
    return FileNotesGitService(
        owner,
        runner=runner,
        git_executable=executable,
        environment=network_environment,
        transport_admission=(
            push_contracts._local_bare_transport_admission_for_tests()
        ),
        network_context_factory=git_network.NetworkContextFactory(
            environment=network_environment,
            temporary_parent=temporary_parent,
            git_executable=executable,
            git_exec_path=exec_path,
        ),
        before_push_spawn=barrier,
    )


async def _review(
    service: FileNotesGitService,
    binding: SessionBinding,
):
    assert (await service.start_push_review(binding)).state == "ready"
    local_operation = service.retained_push_operation(binding)
    assert local_operation is not None
    reviewed = await service.authorize_and_check_push(binding, local_operation)
    assert reviewed.state == "review"
    assert reviewed.handle is not None
    return reviewed


def _refs(repository: Path, *, bare: bool = False) -> bytes:
    if bare:
        return _git_dir(
            repository,
            "for-each-ref",
            "--format=%(refname) %(objectname)",
        )
    return _git(
        repository,
        "for-each-ref",
        "--format=%(refname) %(objectname)",
    )


def _ref_map(repository: Path, *, bare: bool = False) -> dict[str, str]:
    return {
        ref.decode("utf-8"): object_id.decode("ascii")
        for line in _refs(repository, bare=bare).splitlines()
        for ref, object_id in (line.split(b" ", 1),)
    }


def _replica_rows(
    replica: FileNotesReplica,
) -> tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]:
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


@pytest.mark.skipif(os.name != "posix", reason="guarded push is POSIX-only")
@pytest.mark.asyncio
async def test_exact_push_compare_and_swap_updates_only_reviewed_remote_ref(
    tmp_path: Path,
) -> None:
    source, destination, (parent_oid, candidate_oid) = (
        _init_source_and_destination(tmp_path)
    )
    owner, binding = _owner_for_candidate(source, parent_oid, candidate_oid)
    runner = _RecordingRunner()
    service = _service(owner, runner)
    reviewed = await _review(service, binding)
    local_refs = _refs(source)
    local_head = _git(source, "symbolic-ref", "HEAD")
    local_index = _git(source, "ls-files", "--stage", "-v", "-z")
    local_config = (source / ".git" / "config").read_bytes()
    note = source / "note.md"
    worktree = (note.read_bytes(), stat.S_IMODE(note.stat().st_mode))
    remote_before = _ref_map(destination, bare=True)
    replica = FileNotesReplica(tmp_path / "file-notes.sqlite3")
    replica.upsert_file(
        str(source.resolve()),
        "note.md",
        note.read_bytes(),
        content_hash="a" * 64,
        decoded_text=note.read_text(encoding="utf-8"),
        size=note.stat().st_size,
        mtime_ns=1,
    )
    assert replica.checkpoint(
        str(source.resolve()),
        "note.md",
        b"parent\n",
        content_hash="b" * 64,
        session_key="push-integration",
        created_at="2026-07-30T00:00:00Z",
    )
    replica.protect(str(source.resolve()), "note.md")
    replica_rows = _replica_rows(replica)
    try:
        result = await service.start_push(binding, reviewed.handle)

        assert result.state == "succeeded"
        assert _git_dir(destination, "rev-parse", BRANCH_REF).decode().strip() == (
            candidate_oid
        )
        assert _git_dir(
            destination,
            "rev-parse",
            "refs/heads/untouched",
        ).decode().strip() == parent_oid
        assert _git_dir(
            destination,
            "rev-parse",
            "refs/tags/untouched-tag",
        ).decode().strip() == parent_oid
        expected_remote = dict(remote_before)
        expected_remote[BRANCH_REF] = candidate_oid
        assert _ref_map(destination, bare=True) == expected_remote
        assert _refs(source) == local_refs
        assert _git(source, "symbolic-ref", "HEAD") == local_head
        assert _git(source, "ls-files", "--stage", "-v", "-z") == local_index
        assert (source / ".git" / "config").read_bytes() == local_config
        assert (note.read_bytes(), stat.S_IMODE(note.stat().st_mode)) == worktree
        assert _replica_rows(replica) == replica_rows
        push_calls = [argv for argv, _kwargs in runner.calls if "push" in argv]
        assert len(push_calls) == 1
        assert push_calls[0][-1] == f"{candidate_oid}:{BRANCH_REF}"
        assert f"--force-with-lease={BRANCH_REF}:{parent_oid}" in push_calls[0]
    finally:
        replica.close()
        await service.shutdown()


@pytest.mark.skipif(os.name != "posix", reason="guarded push is POSIX-only")
@pytest.mark.asyncio
@pytest.mark.parametrize("race", ["destination_delete", "divergent"])
async def test_exact_push_compare_and_swap_never_recreates_or_overwrites_race(
    tmp_path: Path,
    race: str,
) -> None:
    source, destination, (parent_oid, candidate_oid) = (
        _init_source_and_destination(tmp_path)
    )
    owner, binding = _owner_for_candidate(source, parent_oid, candidate_oid)
    barrier = _Barrier()
    runner = _RecordingRunner()
    service = _service(owner, runner, barrier=barrier)
    reviewed = await _review(service, binding)
    waiter = service.start_push(binding, reviewed.handle)
    await asyncio.wait_for(barrier.entered.wait(), timeout=2)
    if race == "destination_delete":
        _git_dir(destination, "update-ref", "-d", BRANCH_REF)
        expected = None
    else:
        tree_oid = _git(source, "rev-parse", f"{parent_oid}^{{tree}}").decode().strip()
        divergent_oid = _git(
            source,
            "commit-tree",
            tree_oid,
            input_bytes=b"divergent\n",
            environment={
                **os.environ,
                "GIT_AUTHOR_NAME": "Other",
                "GIT_AUTHOR_EMAIL": "other@example.test",
                "GIT_COMMITTER_NAME": "Other",
                "GIT_COMMITTER_EMAIL": "other@example.test",
            },
        ).decode().strip()
        _git(source, "push", str(destination), f"{divergent_oid}:refs/heads/race")
        _git_dir(destination, "update-ref", BRANCH_REF, divergent_oid)
        expected = divergent_oid
    barrier.release.set()

    result = await asyncio.wait_for(waiter, timeout=5)

    observed = subprocess.run(
        (
            shutil.which("git") or "git",
            f"--git-dir={destination}",
            "rev-parse",
            "--verify",
            BRANCH_REF,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.state == "uncertain"
    if expected is None:
        assert observed.returncode != 0
    else:
        assert observed.stdout.decode().strip() == expected
    assert len([argv for argv, _kwargs in runner.calls if "push" in argv]) == 1
    assert owner.snapshot(binding).push_candidate is not None
    await service.shutdown()


@pytest.mark.skipif(os.name != "posix", reason="guarded push is POSIX-only")
@pytest.mark.asyncio
async def test_second_guarded_commit_does_not_range_push_from_older_remote(
    tmp_path: Path,
) -> None:
    source, destination, commits = _init_source_and_destination(
        tmp_path,
        candidate_count=2,
    )
    remote_parent, first_candidate, second_candidate = commits
    owner, binding = _owner_for_candidate(
        source,
        first_candidate,
        second_candidate,
    )
    runner = _RecordingRunner()
    service = _service(owner, runner)
    assert (await service.start_push_review(binding)).state == "ready"
    operation = service.retained_push_operation(binding)
    assert operation is not None

    blocked = await service.authorize_and_check_push(binding, operation)

    assert blocked.state == "blocked"
    assert _git_dir(destination, "rev-parse", BRANCH_REF).decode().strip() == (
        remote_parent
    )
    assert not any("push" in argv for argv, _kwargs in runner.calls)
    assert owner.snapshot(binding).push_candidate is not None
    await service.shutdown()


@pytest.mark.skipif(os.name != "posix", reason="guarded push is POSIX-only")
@pytest.mark.asyncio
async def test_exact_push_uses_frozen_context_during_concurrent_edit_and_config_drift(
    tmp_path: Path,
) -> None:
    source, destination, (parent_oid, candidate_oid) = (
        _init_source_and_destination(tmp_path)
    )
    decoy = tmp_path / "decoy.git"
    _git(tmp_path, "init", "--bare", str(decoy))
    _git(source, "config", "extensions.worktreeConfig", "true")
    owner, binding = _owner_for_candidate(source, parent_oid, candidate_oid)
    global_config = tmp_path / "global.gitconfig"
    system_config = tmp_path / "system.gitconfig"
    global_config.write_bytes(b"")
    system_config.write_bytes(b"")
    environment = {
        "PATH": os.defpath,
        "GIT_CONFIG_GLOBAL": str(global_config),
        "GIT_CONFIG_SYSTEM": str(system_config),
    }
    barrier = _Barrier()
    runner = _RecordingRunner()
    service = _service(
        owner,
        runner,
        barrier=barrier,
        environment=environment,
    )
    reviewed = await _review(service, binding)
    replica = FileNotesReplica(tmp_path / "file-notes.sqlite3")
    waiter = service.start_push(binding, reviewed.handle)
    await asyncio.wait_for(barrier.entered.wait(), timeout=2)
    marker = tmp_path / "helper-ran"
    _git(source, "config", "remote.origin.pushurl", str(decoy))
    _git(source, "config", "credential.helper", f"!touch {marker}")
    _git(source, "config", "core.sshCommand", f"touch {marker}")
    _git(
        source,
        "config",
        "--worktree",
        "remote.origin.pushurl",
        str(decoy),
    )
    _git(
        source,
        "config",
        "--worktree",
        "credential.helper",
        f"!touch {marker}",
    )
    _git(
        source,
        "config",
        "--worktree",
        "core.sshCommand",
        f"touch {marker}",
    )
    assert (source / ".git" / "config.worktree").is_file()
    global_config.write_text(
        f"[url \"{decoy}\"]\n\tinsteadOf = {destination}\n",
        encoding="utf-8",
    )
    system_config.write_text(
        f"[credential]\n\thelper = !touch {marker}\n",
        encoding="utf-8",
    )
    concurrent_bytes = b"edited while push was retained\n"
    (source / "note.md").write_bytes(concurrent_bytes)
    assert owner.record_change(
        binding,
        SessionChange("modified", "note.md"),
    )
    replica.upsert_file(
        str(source.resolve()),
        "note.md",
        concurrent_bytes,
        content_hash="c" * 64,
        decoded_text=concurrent_bytes.decode(),
        size=len(concurrent_bytes),
        mtime_ns=2,
    )
    barrier.release.set()
    try:
        result = await asyncio.wait_for(waiter, timeout=5)

        assert result.state == "succeeded"
        assert _git_dir(destination, "rev-parse", BRANCH_REF).decode().strip() == (
            candidate_oid
        )
        assert not _refs(decoy, bare=True)
        assert not marker.exists()
        assert (source / "note.md").read_bytes() == concurrent_bytes
        assert replica.get_bytes(str(source.resolve()), "note.md") == concurrent_bytes
        assert owner.snapshot(binding).push_candidate is None
    finally:
        replica.close()
        await service.shutdown()
