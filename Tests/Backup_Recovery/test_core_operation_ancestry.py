"""Finite multi-repository work reuses only its exact live operation ancestors."""

import asyncio
import threading
import time

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.participants import core_operation
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB


def test_nested_repository_returns_to_its_issued_ancestor(tmp_path, monkeypatch):
    first = WorkspaceDB(tmp_path / "first.sqlite")
    second = WorkspaceDB(tmp_path / "second.sqlite")
    admissions = []
    original = storage.acquire_storage

    def counted(path=None, **kwargs):
        admissions.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(storage, "acquire_storage", counted)
    try:
        with core_operation(first):
            first_operation = storage._operation_local.operation
            with core_operation(second):
                second_operation = storage._operation_local.operation
                with core_operation(first):
                    assert storage._operation_local.operation is first_operation
                    with first.transaction() as connection:
                        assert connection.execute("SELECT 17").fetchone()[0] == 17
                    with core_operation(second):
                        assert storage._operation_local.operation is second_operation
                assert storage._operation_local.operation is second_operation
            assert storage._operation_local.operation is first_operation
        assert storage._operation_local.operation is None
        assert admissions == [first.db_path, second.db_path]
    finally:
        second.close()
        first.close()


def test_same_file_other_repository_does_not_borrow_ancestor(tmp_path, monkeypatch):
    first = WorkspaceDB(tmp_path / "first.sqlite")
    other = WorkspaceDB(first.db_path)
    second = WorkspaceDB(tmp_path / "second.sqlite")
    admissions = []
    original = storage.acquire_storage

    def counted(path=None, **kwargs):
        admissions.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(storage, "acquire_storage", counted)
    try:
        with core_operation(first):
            first_operation = storage._operation_local.operation
            with core_operation(second):
                with core_operation(other):
                    assert storage._operation_local.operation is not first_operation
                    assert (
                        storage._operation_local.operation.participant.repository()
                        is other
                    )
        assert admissions == [first.db_path, second.db_path, other.db_path]
    finally:
        second.close()
        other.close()
        first.close()


def test_paused_batch_finishes_ancestor_but_refuses_new_repository(tmp_path):
    first = WorkspaceDB(tmp_path / "first.sqlite")
    second = WorkspaceDB(tmp_path / "second.sqlite")
    third = WorkspaceDB(tmp_path / "third.sqlite")
    pause = None
    try:
        with core_operation(first):
            first_operation = storage._operation_local.operation
            with core_operation(second):
                pause = storage._begin_local_pause()
                with core_operation(first):
                    assert storage._operation_local.operation is first_operation
                    with first.transaction() as connection:
                        assert connection.execute("SELECT 23").fetchone()[0] == 23
                with pytest.raises(
                    bootstrap.RecoveryRequired, match="storage_locally_paused"
                ):
                    with core_operation(third):
                        pytest.fail("new participant entered a paused batch")
                assert not pause.drain(time.monotonic() + 0.01)
        assert not pause.drain(time.monotonic() + 0.01)
    finally:
        third.close()
        second.close()
        first.close()
        if pause is not None:
            try:
                assert pause.drain(time.monotonic() + 1)
            finally:
                pause.resume()


def test_borrowed_ancestor_restores_dynamic_owner_after_exception(tmp_path):
    first = WorkspaceDB(tmp_path / "first.sqlite")
    second = WorkspaceDB(tmp_path / "second.sqlite")
    try:
        with core_operation(first), core_operation(second):
            previous = storage._operation_local.operation
            ancestry = storage._operation_local.ancestry
            with pytest.raises(ValueError, match="callback failed"):
                with core_operation(first):
                    raise ValueError("callback failed")
            assert storage._operation_local.operation is previous
            assert storage._operation_local.ancestry is ancestry
        assert storage._operation_local.operation is None
        assert storage._operation_local.ancestry == ()
    finally:
        second.close()
        first.close()


@pytest.mark.parametrize("changed_field", ["db_path", "_read_only"])
def test_ancestor_rechecks_current_repository_identity(tmp_path, changed_field):
    first = WorkspaceDB(tmp_path / "first.sqlite")
    second = WorkspaceDB(tmp_path / "second.sqlite")
    had_field = hasattr(first, changed_field)
    original = getattr(first, changed_field, False)
    try:
        with core_operation(first), core_operation(second):
            try:
                setattr(
                    first,
                    changed_field,
                    second.db_path if changed_field == "db_path" else True,
                )
                with pytest.raises(
                    bootstrap.RecoveryRequired,
                    match="operation_(path_outside_scope|provenance_invalid)",
                ):
                    with first._maintenance_participant.operation():
                        pytest.fail("changed repository borrowed an ancestor")
            finally:
                if had_field:
                    setattr(first, changed_field, original)
                else:
                    delattr(first, changed_field)
    finally:
        second.close()
        first.close()


def test_suspended_discovery_does_not_recover_ancestor_from_stack(tmp_path):
    first = WorkspaceDB(tmp_path / "first.sqlite")
    pause = None
    try:
        with core_operation(first):
            previous = storage._operation_local.operation
            pause = storage._begin_local_pause()
            storage._operation_local.operation = None
            try:
                with pytest.raises(
                    bootstrap.RecoveryRequired, match="storage_locally_paused"
                ):
                    with core_operation(first):
                        pytest.fail("suspended operation granted descendant authority")
            finally:
                storage._operation_local.operation = previous
    finally:
        first.close()
        if pause is not None:
            pause.resume()


@pytest.mark.asyncio
async def test_ancestor_cannot_be_borrowed_by_another_task(tmp_path):
    first = WorkspaceDB(tmp_path / "first.sqlite")
    second = WorkspaceDB(tmp_path / "second.sqlite")

    async def foreign():
        with pytest.raises(
            bootstrap.RecoveryRequired, match="operation_provenance_invalid"
        ):
            with core_operation(first):
                pytest.fail("foreign task borrowed an ancestor")

    try:
        with core_operation(first), core_operation(second):
            await asyncio.create_task(foreign())
    finally:
        second.close()
        first.close()


def test_copied_ancestry_cannot_be_borrowed_by_another_thread(tmp_path):
    first = WorkspaceDB(tmp_path / "first.sqlite")
    second = WorkspaceDB(tmp_path / "second.sqlite")
    results = []

    def foreign(previous, ancestry):
        storage._operation_local.operation = previous
        storage._operation_local.ancestry = ancestry
        try:
            with core_operation(first):
                results.append("borrowed")
        except bootstrap.RecoveryRequired as error:
            results.append(str(error))
        finally:
            storage._operation_local.operation = None
            storage._operation_local.ancestry = ()

    try:
        with core_operation(first), core_operation(second):
            worker = threading.Thread(
                target=foreign,
                args=(
                    storage._operation_local.operation,
                    storage._operation_local.ancestry,
                ),
            )
            worker.start()
            worker.join(5)
            assert not worker.is_alive()
            assert results == ["operation_provenance_invalid"]
    finally:
        second.close()
        first.close()


def test_retired_ancestor_cannot_be_replayed_from_discovery(tmp_path):
    first = WorkspaceDB(tmp_path / "first.sqlite")
    second = WorkspaceDB(tmp_path / "second.sqlite")
    try:
        with core_operation(first):
            retired = storage._operation_local.operation
        assert retired not in storage._operations
        with core_operation(second):
            previous = storage._operation_local.operation
            saved = storage._operation_local.ancestry
            storage._operation_local.ancestry = (retired, previous)
            try:
                with pytest.raises(
                    bootstrap.RecoveryRequired, match="operation_provenance_invalid"
                ):
                    with core_operation(first):
                        pytest.fail("retired operation was replayed")
            finally:
                storage._operation_local.ancestry = saved
    finally:
        second.close()
        first.close()
