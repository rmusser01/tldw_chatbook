"""Real SQLite evidence for bounded send reads and absent-state invalidation."""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import tldw_chatbook.Personal_Context.repository as repository_module
from tldw_chatbook.Chat.console_chat_controller import _compose_profile_tool_provider
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.runtime_policy import PersonalContextAuthorityError
from tldw_chatbook.Personal_Context.service import (
    PersonalContextService,
    ProfileOperationalState,
)


@pytest.fixture
def profile(tmp_path, memory_protector):
    repository = PersonalContextRepository(
        tmp_path / "context.db", key_protector=memory_protector
    )
    return repository, PersonalContextService(repository)


@pytest.fixture
def opens(monkeypatch):
    connections = []
    original = repository_module.connect_private_sqlite

    def track(*args, **kwargs):
        connection = original(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(repository_module, "connect_private_sqlite", track)
    return connections


def compose(service, workspace_id=None):
    return _compose_profile_tool_provider(
        service,
        workspace_id=workspace_id,
        ephemeral=False,
        run_id="run",
        session_id="session",
        current_user_message=None,
        kill_switch=lambda: False,
    )


@pytest.mark.parametrize("workspace", [False, True])
def test_configured_composition_uses_one_hardened_open(profile, opens, workspace):
    repository, service = profile
    service.create_profile()
    service.set_runtime_enabled(True)
    if workspace:
        service.create_workspace_scope("workspace", "Project")
    opens.clear()

    assert compose(service, "workspace" if workspace else None) is not None
    assert len(opens) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opens[0].execute("SELECT 1")


@pytest.mark.parametrize("wal", [False, True])
def test_absent_composition_reuses_only_unchanged_negative_status(profile, opens, wal):
    repository, service = profile
    if wal:
        with sqlite3.connect(repository.db_path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
        connection.close()
    assert compose(service) is None
    assert opens
    opens.clear()
    assert compose(service) is None
    assert service.status().state is ProfileOperationalState.ABSENT
    assert opens == []
    service.create_profile()
    service.set_runtime_enabled(True)
    opens.clear()
    assert compose(service) is not None
    assert len(opens) == 1


def test_absent_status_invalidates_on_external_wal_setup(
    profile, opens, memory_protector
):
    repository, service = profile
    external = PersonalContextRepository(
        repository.db_path, key_protector=memory_protector
    )
    with sqlite3.connect(repository.db_path) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        assert service.status().state is ProfileOperationalState.ABSENT
        assert service.status().state is ProfileOperationalState.ABSENT
        opens.clear()
        assert service.status().state is ProfileOperationalState.ABSENT
        assert opens == []
        # Pin an older reader so setup commits remain exclusively in WAL.
        writer.execute("BEGIN")
        writer.execute("SELECT * FROM profile_meta").fetchall()
        prior_db = repository.db_path.stat()
        other_service = PersonalContextService(external)
        other_service.create_profile()
        other_service.set_runtime_enabled(True)
        assert repository.db_path.stat().st_mtime_ns == prior_db.st_mtime_ns
        opens.clear()
        assert service.status().state is ProfileOperationalState.READY
        assert opens


def test_nested_operations_close_after_failure_and_next_operation_is_fresh(
    profile, opens
):
    repository, service = profile
    service.create_profile()
    opens.clear()
    with pytest.raises(RuntimeError, match="failure"):
        with repository.operation():
            repository.get_manifest()
            with repository.operation():
                repository.list_scopes()
            raise RuntimeError("failure")
    assert len(opens) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opens[0].execute("SELECT 1")
    with repository.operation():
        assert repository.get_manifest() is not None
    assert len(opens) == 2


def test_operations_are_thread_local(profile, opens):
    repository, service = profile
    service.create_profile()
    opens.clear()
    barrier = threading.Barrier(2)

    def read():
        with repository.operation():
            first = repository.get_manifest()
            barrier.wait(timeout=5)
            assert repository.get_manifest() == first

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(read) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)
    assert len(opens) == 2


@pytest.mark.parametrize("replacement", ["file", "symlink", "permissions", "parent"])
def test_scoped_read_rejects_mid_operation_path_authority_change(profile, replacement):
    repository, service = profile
    service.create_profile()
    with repository.operation():
        repository.get_manifest()
        path = repository.db_path
        if replacement in {"file", "symlink"}:
            saved = path.with_suffix(".saved")
            path.rename(saved)
            if replacement == "file":
                path.write_bytes(saved.read_bytes())
                path.chmod(0o600)
            else:
                path.symlink_to(saved)
        elif replacement == "permissions":
            path.chmod(0o644)
        else:
            path.parent.chmod(0o777)
        with pytest.raises((OSError, repository_module.ProfileIntegrityError)):
            repository.get_manifest()


def test_authorized_absent_view_still_fails_closed_without_connects(profile, opens):
    repository, service = profile
    with pytest.raises(PersonalContextAuthorityError):
        service.authorized_context_view()
    opens.clear()
    with pytest.raises(PersonalContextAuthorityError):
        service.authorized_context_view()
    assert opens == []


def test_failed_setup_does_not_retain_absent_cache(profile, opens, monkeypatch):
    repository, service = profile
    assert service.status().state is ProfileOperationalState.ABSENT

    def fail(*args, **kwargs):
        raise RuntimeError("setup failure")

    with monkeypatch.context() as patch:
        patch.setattr(repository, "create_profile_with_global_scope", fail)
        with pytest.raises(RuntimeError, match="setup failure"):
            service.create_profile()
    opens.clear()
    assert service.status().state is ProfileOperationalState.ABSENT
    assert opens
    service.create_profile()
    assert service.status().profile_present


def test_storage_error_discards_cached_absence(profile, opens, monkeypatch):
    repository, service = profile
    assert service.status().state is ProfileOperationalState.ABSENT
    original = repository.storage_signature

    def fail():
        raise OSError("metadata unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(repository, "storage_signature", fail)
        assert compose(service) is None
    opens.clear()
    assert service.status().state is ProfileOperationalState.ABSENT
    assert opens
    assert repository.storage_signature() == original()


def test_same_path_replacement_invalidates_absence_and_fresh_operation(profile, opens):
    repository, service = profile
    assert service.status().state is ProfileOperationalState.ABSENT
    copied = repository.db_path.with_suffix(".replacement")
    copied.write_bytes(repository.db_path.read_bytes())
    copied.chmod(0o600)
    copied.replace(repository.db_path)
    opens.clear()
    assert service.status().state is ProfileOperationalState.ABSENT
    assert opens
    service.create_profile()
    service.set_runtime_enabled(True)
    assert compose(service) is not None
    repository.db_path.rename(copied)
    repository.db_path.symlink_to(copied)
    opens.clear()
    assert compose(service) is None
    assert opens == []


def test_ready_authority_is_never_cached_and_locked_facade_never_connects(
    profile, opens
):
    repository, service = profile
    service.create_profile()
    service.set_runtime_enabled(True)
    assert compose(service) is not None
    service.set_runtime_enabled(False)
    opens.clear()
    assert compose(service) is None
    assert opens
    locked = PersonalContextService.locked(profile_present=True)
    opens.clear()
    assert compose(locked) is None
    assert compose(locked) is None
    assert opens == []
    # Unlock installs a fresh unlocked service over repository/key custody.
    service.set_runtime_enabled(True)
    assert compose(PersonalContextService(repository)) is not None


def test_connection_setup_failure_closes_and_does_not_poison_next_operation(
    profile, opens, monkeypatch
):
    repository, service = profile
    service.create_profile()
    service.set_runtime_enabled(True)
    original = repository.storage_signature
    calls = 0

    def fail_post_open_validation():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("path changed during open")
        return original()

    opens.clear()
    with monkeypatch.context() as patch:
        patch.setattr(repository, "storage_signature", fail_post_open_validation)
        with pytest.raises(OSError, match="path changed"):
            with repository.operation():
                repository.get_manifest()
    assert len(opens) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opens[0].execute("SELECT 1")
    assert compose(service) is not None
