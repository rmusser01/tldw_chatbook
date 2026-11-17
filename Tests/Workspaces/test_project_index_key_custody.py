"""Exact SQLite, pin and storage-hold joins used by the fixed key worker."""

import sqlite3
import asyncio
import threading
import sys
from types import MethodType

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook import config
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
from tldw_chatbook.Workspaces.project_index import runtime as index
from tldw_chatbook.Workspaces.project_index import key_process
from tldw_chatbook.Workspaces.project_index.models import ProjectIndexRetirementUnproven
from tldw_chatbook.MCP.permission_process import ProtocolError


@private_profile_test
async def test_failed_pipe_closes_still_contain_real_owner_and_keep_failures(
    tmp_path, request
):
    from Tests.Workspaces.test_project_index_key_process import managed_index, prepare

    assertion = None
    with pytest.raises(ProtocolError, match="key_outcome_unresolved"):
        async with managed_index(tmp_path) as rig:
            index._KEY_PROCESS_ENABLED = True
            owner = None
            originals = None
            attempts = []
            failures = [OSError("stdin close failed"), OSError("stdout close failed")]

            class FailedClose:
                def __init__(self, original, name, failure):
                    self.original, self.name, self.failure = original, name, failure

                def __getattr__(self, name):
                    return getattr(self.original, name)

                def close(self):
                    attempts.append(self.name)
                    self.original.close()
                    raise self.failure

            try:
                await prepare(rig)
                owner = rig.app._project_index_key_owner
                assert owner.selected == 1 and owner.receipts[-1]["retired"]
                originals = owner.process.stdin, owner.process.stdout
                owner.process.stdin = FailedClose(originals[0], "stdin", failures[0])
                owner.process.stdout = FailedClose(originals[1], "stdout", failures[1])
                primary = OSError("original operation failure")
                record = ("failed-command", primary)
                try:
                    await asyncio.to_thread(owner._unresolve, record)
                except BaseException:
                    pass  # Assertions below inspect physical state before cleanup.
                assert owner.process.poll() is not None
                assert not owner.monitor.is_alive()
                assert attempts == ["stdin", "stdout"]
                assert owner.process.stdin.closed and owner.process.stdout.closed
                assert owner.retired and owner.unresolved is record
                assert owner.unresolved[-1] is primary
                retained = [error for _resource, error in owner._retirement_errors]
                assert retained == failures
                assert all(
                    actual is expected
                    for actual, expected in zip(retained, failures, strict=True)
                )
                assert owner.containment_failure.exceptions == tuple(failures)
                assert owner in key_process._owners
                key_process.participant._maintenance_close_admission()
                assert not key_process.participant.drained()
                with pytest.raises(ProtocolError, match="key_outcome_unresolved"):
                    owner.close()
            except BaseException as error:
                assertion = error
            finally:
                if originals is not None:
                    owner.process.stdin, owner.process.stdout = originals
                    if not owner.retired:
                        owner._retire_process()
                index._KEY_PROCESS_ENABLED = False
    if assertion is not None:
        raise assertion


def native_service(tmp_path):
    database = WorkspaceDB(config.get_workspaces_db_path())
    registry = LocalWorkspaceRegistryService(database)
    registry.create_workspace(workspace_id="custody", name="Project")
    root = tmp_path / "project"
    root.mkdir()
    binding = registry.add_folder_binding("custody", root)
    database.close()
    service = index._compose_project_index_service(registry)
    return database, service, ("custody", binding.binding_id, root)


@private_profile_test
async def test_configuration_failure_retains_exact_pre_cache_connection(
    tmp_path, request, monkeypatch
):
    from tldw_chatbook.Workspaces.project_index.key_worker import _failure

    database, service, arguments = native_service(tmp_path)
    failure = sqlite3.OperationalError("private SQL and path must not cross wire")

    def configure(*args, **kwargs):
        raise failure

    monkeypatch.setattr(WorkspaceDB, "_configure_connection", configure)
    resources = key_process._KeyResources()
    resources.service = service
    try:
        with pytest.raises(sqlite3.OperationalError) as caught:
            with resources:
                index._project_index_key_owned(service, *arguments)
        assert caught.value is failure and resources.failure is failure
        assert getattr(database._thread_local, "conn", None) is None
        assert len(resources.connections) == 1
        owner, connection, participant = resources.connections[0]
        assert owner is database and participant is database._maintenance_participant
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            sqlite3.Connection.in_transaction.__get__(connection)
        assert resources.retirement()["connections"] == 1
        decoded = key_process._decode_failure(_failure(failure))
        assert type(decoded) is sqlite3.OperationalError
        assert "private SQL" not in str(decoded)
    finally:
        database.close()


@pytest.mark.parametrize("raising", [False, True])
@private_profile_test
async def test_nonretiring_pin_keeps_exact_failed_resources(
    tmp_path, request, monkeypatch, raising
):
    database, service, arguments = native_service(tmp_path)
    resources = key_process._KeyResources()
    resources.service = service
    original = index._DataAuthorityPin.close
    failure = OSError("injected pin close failure")

    def fail_close(pin):
        if raising:
            raise failure

    monkeypatch.setattr(index._DataAuthorityPin, "close", fail_close)
    try:
        if raising:
            with pytest.raises(OSError) as caught:
                with resources:
                    index._project_index_key_owned(service, *arguments)
            assert caught.value is failure
        else:
            with resources:
                index._project_index_key_owned(service, *arguments)
        with pytest.raises(ProtocolError, match="key_native_retirement_unproven"):
            resources.retirement()
        assert resources in key_process._unresolved_resources
        assert len(resources.pins) == 1 and not resources.pins[0].closed
        assert resources.pins[0].factory is service._factory
    finally:
        # Explicitly settle this exact injected pin; keep the original failed record.
        for pin in resources.pins:
            original(pin)
        database.close()


@pytest.mark.parametrize("raising", [False, True])
@private_profile_test
async def test_nonretiring_connection_keeps_native_and_core_identity(
    tmp_path, request, monkeypatch, raising
):
    database, service, arguments = native_service(tmp_path)
    resources = key_process._KeyResources()
    resources.service = service
    configure = WorkspaceDB._configure_connection
    failure = OSError("injected native close failure")

    def configured(database, connection, **kwargs):
        result = configure(database, connection, **kwargs)

        def close():
            if raising:
                raise failure

        connection.close = close
        return result

    monkeypatch.setattr(WorkspaceDB, "_configure_connection", configured)
    try:
        with pytest.raises(ProjectIndexRetirementUnproven):
            with resources:
                index._project_index_key_owned(service, *arguments)
        with pytest.raises(ProtocolError, match="key_native_retirement_unproven"):
            resources.retirement()
        assert resources in key_process._unresolved_resources
        assert len(resources.connections) == 1
        owner, connection, participant = resources.connections[0]
        assert owner is database
        assert connection in participant.connections
        assert sqlite3.Connection.in_transaction.__get__(connection) is False
        assert any(operation.retirement_unproven for operation in service._operations)
        if raising:
            assert any(
                record[-1] is failure
                for record in getattr(resources, "close_failures", ())
            )
    finally:
        for _owner, connection, _participant in resources.connections:
            type(connection).close(connection)
        database.close()


@private_profile_test
async def test_nonraising_retired_hold_failure_cannot_hide_behind_empty_counts(
    tmp_path, request
):
    from Tests.Workspaces.test_project_index_key_process import trace_functions
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    database, service, arguments = native_service(tmp_path)
    # Use real shutdown to retire startup before testing an ordinary last holder.
    # No participant registry is manually cleared or repaired.
    assert storage._live_leases == set(storage._startups.values())
    storage._shutdown()
    resources = key_process._KeyResources()
    resources.service = service
    failed_holds = []
    failure = OSError("injected native hold exit failure")

    def observe(function, event, result):
        if event == "return":
            hold = sys._getframe(2).f_locals["self"]
            assert hold.error is None
            hold.error = failure
            failed_holds.append(hold)

    try:
        with trace_functions((storage._Hold._run,), observe), resources:
            index._project_index_key_owned(service, *arguments)
        assert failed_holds
        hold = failed_holds[0]
        assert not hold.thread.is_alive() and hold.count == 0
        assert hold.error is failure and hold in storage._retiring_holds
        with pytest.raises(ProtocolError, match="native_retirement_unproven"):
            resources.retirement()
        assert resources in key_process._unresolved_resources
        assert resources.command.leases[0][0]._key is None
    finally:
        database.close()


@private_profile_test
async def test_instance_connection_override_uses_parent_before_child_effects(
    tmp_path, request
):
    from Tests.Workspaces.test_project_index_key_process import managed_index, prepare

    async with managed_index(tmp_path) as rig:
        database = rig.app.workspace_registry_service.db
        original = database._open_connection
        calls = []

        def custom(actual, **kwargs):
            assert actual is database
            calls.append(True)
            return original(**kwargs)

        database._open_connection = MethodType(custom, database)
        index._KEY_PROCESS_ENABLED = True
        try:
            await prepare(rig)
            assert calls
            assert not hasattr(rig.app, "_project_index_key_owner")
        finally:
            index._KEY_PROCESS_ENABLED = False
            del database._open_connection


@private_profile_test
async def test_app_close_fences_adoption_while_retaining_parent_tail(
    tmp_path, request, monkeypatch
):
    from Tests.Workspaces.test_project_index_key_process import (
        managed_index,
        prepare,
        wait_event,
    )
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    async with managed_index(tmp_path) as rig:
        entered, release = threading.Event(), threading.Event()

        def status(service, key):
            entered.set()
            assert release.wait(60)
            return None

        monkeypatch.setattr(index, "project_index_status_for_console", status)
        index._KEY_PROCESS_ENABLED = True
        pending = asyncio.create_task(prepare(rig))
        closing = None
        try:
            await wait_event(entered)
            owner = rig.app._project_index_key_owner
            closing = asyncio.create_task(index.close_project_index_service(rig.app))
            async with asyncio.timeout(30):
                while not getattr(rig.app, "_project_index_key_closed", False):
                    await asyncio.sleep(0.01)
            assert not closing.done()
            assert len(rig.runtime._preparation_reads) == 1
            release.set()
            with pytest.raises(
                RecoveryRequired, match="console_snapshot_owner_changed"
            ):
                await pending
            await closing
            assert owner.retired and owner.process.poll() == 0
        finally:
            release.set()
            await asyncio.gather(
                pending, *([closing] if closing else []), return_exceptions=True
            )
            index._KEY_PROCESS_ENABLED = False


@private_profile_test
async def test_child_initialization_rejects_outdated_schema_without_migration(
    tmp_path, request
):
    from Tests.Workspaces.test_project_index_key_process import managed_index, prepare

    assertion = None
    with pytest.raises(ProtocolError, match="key_outcome_unresolved"):
        async with managed_index(tmp_path) as rig:
            database = rig.app.workspace_registry_service.db
            with database.transaction() as connection:
                connection.execute("DELETE FROM schema_version")
                connection.execute("INSERT INTO schema_version(version) VALUES (1)")
            index._KEY_PROCESS_ENABLED = True
            try:
                service, scopes = await prepare(rig)
                assert service is None and not scopes
                owner = rig.app._project_index_key_owner
                assert owner.selected == 0 and owner.unresolved is not None
                assert owner.retired and owner.process.poll() is not None
                assert database.get_schema_version() == 1
                assert not rig.cache.exists()
            except BaseException as error:
                assertion = error
            finally:
                index._KEY_PROCESS_ENABLED = False
    if assertion is not None:
        raise assertion
