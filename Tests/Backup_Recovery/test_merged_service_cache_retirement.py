"""Upstream held service caches retain the backup borrower lifetime contract."""

import sqlite3
import threading
import time

import pytest

from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.participants import (
    _core_operation,
    _retire_current_thread_caches,
)
from Tests.Backup_Recovery.test_participant_lifetimes import local_root as _local_root

local_root = _local_root


@pytest.fixture(params=["writing", "research", "events", "notes_device"])
def held_service(request, tmp_path, local_root):
    path = tmp_path / "held.sqlite"
    if request.param == "writing":
        from tldw_chatbook.Writing_Interop.local_writing_service import LocalWritingService

        owner = LocalWritingService(path)
        getter = owner._connect
    elif request.param == "research":
        from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService

        owner = LocalResearchService(path)
        getter = owner._connect
    elif request.param == "events":
        from tldw_chatbook.Notifications.event_state_repository import EventStateRepository

        owner = EventStateRepository(path)
        getter = owner._held_connection
    else:
        from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore

        owner = NotesDeviceStateStore(path)
        getter = owner._get_connection
    try:
        yield owner, getter
    finally:
        owner.close()


def test_current_thread_cache_retires_and_reopens(held_service):
    _owner, get = held_service
    connection = get()
    pause = storage._begin_local_pause()
    try:
        _retire_current_thread_caches(pause)
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
        assert pause.drain(time.monotonic() + 1)
    finally:
        pause.resume()
    reopened = get()
    assert reopened is not connection
    assert reopened.execute("SELECT 1").fetchone()[0] == 1


@pytest.mark.parametrize("borrower", ["transaction", "operation"])
def test_admitted_borrower_defers_cache_retirement(held_service, borrower):
    owner, get = held_service
    connection = get()
    operation = _core_operation(owner) if borrower == "operation" else None
    if operation is not None:
        operation.__enter__()
    else:
        connection.execute("BEGIN")
    pause = storage._begin_local_pause()
    try:
        _retire_current_thread_caches(pause)
        assert connection.execute("SELECT 1").fetchone()[0] == 1
        assert not pause.drain(time.monotonic())
        if operation is not None:
            operation.__exit__(None, None, None)
            operation = None
        else:
            connection.rollback()
        _retire_current_thread_caches(pause)
        assert pause.drain(time.monotonic() + 1)
    finally:
        if operation is not None:
            operation.__exit__(None, None, None)
        pause.resume()


def test_foreign_thread_cache_remains_owned_by_its_worker(held_service):
    _owner, get = held_service
    entered, release = threading.Event(), threading.Event()
    failures = []

    def worker():
        try:
            connection = get()
            entered.set()
            assert release.wait(5)
            assert connection.execute("SELECT 1").fetchone()[0] == 1
            connection.close()
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    assert entered.wait(5)
    pause = storage._begin_local_pause()
    try:
        _retire_current_thread_caches(pause)
        assert not pause.drain(time.monotonic())
        release.set()
        thread.join(5)
        assert not failures
        assert not thread.is_alive()
        assert pause.drain(time.monotonic() + 1)
    finally:
        release.set()
        thread.join(5)
        pause.resume()
