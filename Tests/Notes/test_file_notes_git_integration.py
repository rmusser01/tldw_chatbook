from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from tldw_chatbook.Notes.file_notes_git_service import FileNotesGitService
from tldw_chatbook.Notes.file_notes_session_owner import (
    FileNotesSessionOwner,
    HeadIdentity,
    SequencedSessionChange,
    SessionChange,
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
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [self.git, *arguments],
            cwd=cwd or self.path,
            env=self.environment,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


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
    service.shutdown()


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
    service.shutdown()


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
    service.shutdown()


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
    service.shutdown()


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

    assert not service.revalidate_repository(
        binding,
        discovery.repository,
    )
    snapshot = owner.snapshot(binding)
    assert snapshot.trusted_repository is None
    assert snapshot.git_status is None
    assert not snapshot.staging_ownership
    service.shutdown()


def _change(
    sequence: int,
    action: str,
    relative_path: str,
) -> SequencedSessionChange:
    return SequencedSessionChange(
        sequence=sequence,
        change=SessionChange(  # type: ignore[arg-type]
            action=action,
            relative_path=relative_path,
        ),
    )


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
    service.shutdown()


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
    service.shutdown()


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
    service.shutdown()


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
    service.shutdown()
