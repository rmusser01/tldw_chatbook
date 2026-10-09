"""Controller durable callbacks retain exact finite native repository ownership."""

import asyncio
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_console_metadata_batching import (
    observe_worker_connections,
)
from Tests.Backup_Recovery.test_finite_db_counted_interval import (
    live_operations,
    observe_admissions,
    read_value,
)
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import local_root as local_root  # noqa: PLC0414
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.DB.base_db import operation_owned_connection
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.usefixtures("local_root")


def controller_for(database):
    controller = object.__new__(ConsoleChatController)
    controller.store = SimpleNamespace(persistence=SimpleNamespace(db=database))
    return controller


@pytest.fixture
def database(tmp_path):
    owner = CharactersRAGDB(tmp_path / "controller.sqlite", "controller-finite")
    owner.close()
    try:
        yield owner
    finally:
        owner.close()


@pytest.mark.asyncio
async def test_controller_callback_counts_once_keeps_sql_and_closes_after_interval(
    database, monkeypatch
):
    controller = controller_for(database)
    admissions = observe_admissions(monkeypatch)
    connections, statements = observe_worker_connections(
        monkeypatch, database, "get_connection"
    )
    closes = []
    actual_close = database.close_connection

    def observed_close():
        closes.append(tuple(live_operations(database)))
        actual_close()

    monkeypatch.setattr(database, "close_connection", observed_close)

    def body(prefix, *, suffix):
        return prefix + tuple(read_value(database, n) for n in (11, 22, 33)) + suffix

    assert await asyncio.to_thread(
        controller._run_owned_chat_db_operation, body, (7,), suffix=(8,)
    ) == (7, 11, 22, 33, 8)
    ordinary = [
        lease
        for path, operation, lease in admissions
        if path == database.db_path and operation is None
    ]
    descendants = [
        lease
        for path, operation, lease in admissions
        if path == database.db_path and operation is not None
    ]
    assert (
        len(ordinary) == 1
    ), "one durable callback must share one complete ordinary admission"
    assert (
        len(descendants) == 1
    ), "the native connection retains its separate descendant lease"
    assert [sql for _, sql in statements if "finite_probe" in sql] == [
        "SELECT 11 AS finite_probe",
        "SELECT 22 AS finite_probe",
        "SELECT 33 AS finite_probe",
    ]
    assert len(connections) == 1
    assert closes == [()], "the callback interval must exit before native close"
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")
    assert not worker_leases(database)
    assert not live_operations(database)
    assert all(lease not in storage._live_leases for _, _, lease in admissions)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_controller_callback_accepted_before_pause_survives_awaiter_cancel(
    database, tmp_path, cancel
):
    controller = controller_for(database)
    other = CharactersRAGDB(tmp_path / "other.sqlite", "independent")
    other.close()
    entered, release, finished = (threading.Event() for _ in range(3))
    results = []

    def body():
        try:
            first = read_value(database, 11)
            entered.set()
            assert release.wait(10)
            results.append((first, read_value(database, 22)))
            with operation_owned_connection(other):
                with pytest.raises(
                    bootstrap.RecoveryRequired, match="storage_locally_paused"
                ):
                    other.get_connection()
            return "complete"
        finally:
            finished.set()

    pending = asyncio.create_task(controller._run_durable_db_call(body))
    pause = None
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        assert live_operations(
            database
        ), "the complete callback remains counted between SQL reads"
        pause = storage._begin_local_pause()
        assert worker_leases(database)
        assert not pause.drain(time.monotonic() + 0.02)
        if cancel:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert live_operations(database)
        assert worker_leases(database)
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            await controller._run_durable_db_call(
                lambda: pytest.fail("fresh callback reached its body during pause")
            )
        release.set()
        if not cancel:
            assert await pending == "complete"
        assert await asyncio.to_thread(finished.wait, 10)
        for _ in range(200):
            if not live_operations(database) and not worker_leases(database):
                break
            await asyncio.sleep(0.005)
        assert results == [(11, 22)]
        assert not live_operations(database)
        assert not worker_leases(database)
        assert pause.drain(time.monotonic() + 1)
    finally:
        release.set()
        if pause is not None:
            pause.resume()
        await asyncio.gather(pending, return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, 10)
        other.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "same_path", [False, True], ids=("different-file", "same-file")
)
async def test_controller_callback_other_receiver_admits_independently(
    database, tmp_path, monkeypatch, same_path
):
    other = CharactersRAGDB(
        database.db_path if same_path else tmp_path / "independent.sqlite",
        "other-receiver",
    )
    other.close()
    admissions = observe_admissions(monkeypatch)
    controller = controller_for(database)
    original_scopes = []

    def body():
        original = storage._operation_local.operation
        assert original.participant.repository() is database
        assert read_value(database, 11) == 11
        with operation_owned_connection(other):
            native = other.get_connection()
            assert native.execute("SELECT 22 AS finite_probe").fetchone()[0] == 22
        original_scopes.append(storage._operation_local.operation)
        return read_value(database, 33)

    try:
        assert await controller._run_durable_db_call(body) == 33
        ordinary = [path for path, operation, _ in admissions if operation is None]
        assert ordinary == [database.db_path, other.db_path]
        assert original_scopes[0].participant.repository() is database
        assert not worker_leases(database)
        assert not worker_leases(other)
        assert not live_operations(database)
        assert not live_operations(other)
    finally:
        other.close()


@pytest.mark.parametrize("shape", ["subclass", "memory"])
def test_controller_unqualified_or_memory_receiver_preserves_borrowed_handle(
    tmp_path, monkeypatch, shape
):
    owner_type = (
        type("UninstalledNotes", (CharactersRAGDB,), {})
        if shape == "subclass"
        else CharactersRAGDB
    )
    owner = owner_type(
        ":memory:" if shape == "memory" else tmp_path / "uninstalled.sqlite",
        "compatibility",
    )
    controller = controller_for(owner)
    borrowed = owner.get_connection()
    admissions = observe_admissions(monkeypatch)
    try:
        borrowed.execute("BEGIN")
        assert (
            controller._run_owned_chat_db_operation(lambda: owner.get_connection())
            is borrowed
        )
        assert borrowed.in_transaction
        assert borrowed.execute("SELECT 43").fetchone()[0] == 43
        assert not live_operations(owner)
        assert all(operation is None for _, operation, _ in admissions)
    finally:
        borrowed.rollback()
        owner.close()


def test_controller_custom_receiver_preserves_callback_args_result_and_thread():
    owner = SimpleNamespace(is_memory_db=False)
    controller = controller_for(owner)
    origin = threading.current_thread()
    result = controller._run_owned_chat_db_operation(
        lambda prefix, *, suffix: (threading.current_thread(), prefix + suffix),
        "custom",
        suffix="-result",
    )
    assert result == (origin, "custom-result")


@pytest.mark.asyncio
async def test_controller_callback_retarget_refuses_fresh_sql(database, tmp_path):
    controller = controller_for(database)
    original = database.db_path
    replacement = tmp_path / "replacement.sqlite"

    def body():
        assert read_value(database, 11) == 11
        database.db_path = replacement
        try:
            with pytest.raises(
                (ValueError, bootstrap.RecoveryRequired),
                match="repository_participant_not_installed|operation_path_outside_scope",
            ):
                read_value(database, 22)
        finally:
            database.db_path = original
        return read_value(database, 33)

    assert await controller._run_durable_db_call(body) == 33
    assert not replacement.exists()
    assert not worker_leases(database)
    assert not live_operations(database)
