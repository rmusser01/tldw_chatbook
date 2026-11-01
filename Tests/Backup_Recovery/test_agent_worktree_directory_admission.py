"""Fixed worktree-directory admission stays independent of its parent DB scope."""

import pytest

from Tests.Agents.test_fleet_runtime import git_repo as git_repo
from Tests.Backup_Recovery.test_participant_lifetimes import local_root as local_root
from tldw_chatbook import config
from tldw_chatbook.Agents import agent_worktree
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.participants import _core_operation
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Utils.private_paths import PrivatePathError, secure_private_directory

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def database(tmp_path, local_root):
    owner = AgentRunsDB(tmp_path / "runs.db")
    connection = owner._held_connection()
    connection.execute("BEGIN")
    try:
        yield owner, connection
    finally:
        connection.rollback()
        owner.close()
        assert not owner._maintenance_participant.connections


def test_real_worktree_root_preparation_restores_exact_borrowed_database_scope(
    database, tmp_path
):
    owner, connection = database
    with _core_operation(owner):
        original = storage._operation_local.operation
        base = agent_worktree._worktrees_base()
        assert base == config.get_user_data_dir() / "agent_worktrees"
        assert base.is_dir()
        assert base.stat().st_mode & 0o777 == 0o700
        assert storage._operation_local.operation is original
        assert original in storage._operations
        assert original.lease in storage._live_leases
        with pytest.raises(
            bootstrap.RecoveryRequired, match="operation_path_outside_scope"
        ):
            secure_private_directory(tmp_path / "unrelated", create=True)
        assert not (tmp_path / "unrelated").exists()
    assert owner._held_connection() is connection
    assert connection.in_transaction
    assert connection.execute("SELECT 1").fetchone()[0] == 1


@pytest.mark.parametrize("create_worktree", [False, True])
def test_native_worktree_directory_failure_restores_parent_scope(
    database, tmp_path, git_repo, create_worktree
):
    base = agent_worktree._worktrees_base()
    saved = base.with_name("agent_worktrees-before-symlink-control")
    base.rename(saved)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    base.symlink_to(outside, target_is_directory=True)
    try:
        with _core_operation(database[0]):
            original = storage._operation_local.operation
            if create_worktree:
                result = agent_worktree.create_agent_worktree(git_repo, "symlink-child")
                assert isinstance(result, agent_worktree.WorktreeRefusal)
                assert result.reason_code == "storage_unavailable"
            else:
                with pytest.raises(PrivatePathError):
                    agent_worktree._worktrees_base()
            assert storage._operation_local.operation is original
            storage._check_operation(original, original.path)
            assert original.lease in storage._live_leases
    finally:
        base.unlink()
        saved.rename(base)
    assert tuple(outside.iterdir()) == ()


@pytest.mark.parametrize("create_worktree", [False, True])
def test_pause_before_native_worktree_directory_admission_restores_parent(
    database, git_repo, monkeypatch, create_worktree
):
    base = config.get_user_data_dir() / "agent_worktrees"
    native_acquire = storage.acquire_storage
    pauses = []

    def pause_then_acquire(path=None, *, related_paths=()):
        if path == base:
            assert storage._operation_local.operation is None
            pauses.append(storage._begin_local_pause())
        return native_acquire(path, related_paths=related_paths)

    monkeypatch.setattr(storage, "acquire_storage", pause_then_acquire)
    with _core_operation(database[0]):
        original = storage._operation_local.operation
        try:
            if create_worktree:
                result = agent_worktree.create_agent_worktree(git_repo, "paused-child")
                assert isinstance(result, agent_worktree.WorktreeRefusal)
                assert result.reason_code == "storage_unavailable"
            else:
                with pytest.raises(bootstrap.RecoveryRequired):
                    agent_worktree._worktrees_base()
        finally:
            for pause in reversed(pauses):
                pause.resume()
        assert len(pauses) == 1
        assert storage._operation_local.operation is original
        storage._check_operation(original, original.path)
        assert original.lease in storage._live_leases
    assert database[1].in_transaction
    assert not (git_repo / "child.txt").exists()
