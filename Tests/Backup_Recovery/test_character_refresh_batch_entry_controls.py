"""Root-run entry faults; installed guards and Native callbacks stay original."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import CodeType, FunctionType

import pytest

from Tests.Backup_Recovery.test_character_refresh_finite_batch import (
    _original_calls,
    _source_hashes,
)
from Tests.Backup_Recovery.test_console_presentation_cadence import _character, _display
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.DB import base_db
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules import character_context

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.usefixtures("local_root")]


def _body_fault(function, calls):
    """Mutate only the two new unguarded methods, not a custody callback."""
    assert function.__name__ in {"_read_recent_scope_sync", "_read_refresh_scope_sync"}
    assert type(function) is FunctionType and function.__closure__ is None
    clone = FunctionType(
        function.__code__,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    namespace = function.__globals__
    names = ("_character_entry_original", "_character_entry_calls")
    assert all(name not in namespace for name in names)
    namespace[names[0]], namespace[names[1]] = clone, calls
    exec(
        "def _character_entry_body(self, *args, **kwargs):\n"
        "    _character_entry_calls.append(type(self).__name__)\n"
        "    return _character_entry_original(self, *args, **kwargs)\n",
        namespace,
    )
    original_code = function.__code__
    function.__code__ = namespace.pop("_character_entry_body").__code__

    def restore():
        function.__code__ = original_code
        for name in names:
            namespace.pop(name)

    return restore


@contextmanager
def _entry_hold(database, controller, boundary, executor, observed):
    """Hold only an exact qualified queue/native/inner entry, with original code."""
    selector = character_context._stock_character_refresh_readers
    admission = storage._acquire_storage
    recent = character_context.ConsoleCharacterContextController._read_recent_scope_sync
    originals = tuple((fn, fn.__code__) for fn in (selector, admission, recent))
    invoke = tuple(
        code
        for code in base_db.run_owned_db_call.__code__.co_consts
        if type(code) is CodeType and code.co_name == "invoke"
    )
    assert len(invoke) == 1
    old_main, old_thread = sys.getprofile(), threading.getprofile()
    current = threading.current_thread()
    entered, release = threading.Event(), threading.Event()
    facts = dict(
        entered=entered,
        release=release,
        readers=None,
        worker=None,
        admission=None,
        blocker=None,
        native_before=None,
        inner_entered=False,
    )

    def occupy_worker():
        facts["worker"] = threading.current_thread()
        entered.set()
        assert release.wait(10), "test-owned queue hold was not released"

    def matching_invoke(frame):
        parent = frame.f_back
        while parent is not None:
            if parent.f_code is invoke[0]:
                return parent.f_locals.get("database") is database
            parent = parent.f_back
        return False

    def hold_admission(lease):
        assert threading.current_thread() is not current
        assert type(lease) is storage.StorageLease and lease._key is not None
        with storage._lock:
            assert lease in storage._live_leases and lease._key is not None
        facts["worker"], facts["admission"] = threading.current_thread(), lease
        entered.set()
        assert release.wait(10), "test-owned original admission was not released"

    def observe(frame, event, argument):
        delegate = old_main if threading.current_thread() is current else old_thread
        if delegate is not None:
            delegate(frame, event, argument)
        if (
            event == "return"
            and frame.f_code is originals[0][1]
            and frame.f_locals.get("controller") is controller
            and argument is not None
            and facts["readers"] is None
            and controller.state.loading
        ):
            assert threading.current_thread() is current
            # Ignore the resident precheck classification; hold the actual
            # refresh selection after its generation increment. No DB work
            # precedes this selected refresh when the fingerprint is absent.
            assert observed["callbacks"] == 0 and observed["pairs"] == 0
            assert not observed["connections"]
            assert not worker_leases(database)
            facts["readers"] = argument
            facts["native_before"] = observed["native_opens"]
            if boundary == "queued":
                facts["blocker"] = executor.submit(occupy_worker)
                assert entered.wait(10), "the actual executor was not occupied"
        elif (
            boundary == "admission"
            and not entered.is_set()
            and event == "return"
            and frame.f_code is originals[1][1]
            and frame.f_locals.get("path") == database.db_path
            and facts["readers"] is not None
            and observed["callbacks"] == 1
            and observed["pairs"] == 0
            and matching_invoke(frame)
            and type(argument) is storage.StorageLease
        ):
            if os.name == "nt":
                assert observed["native_opens"] > facts["native_before"]
            hold_admission(argument)
        elif (
            boundary == "inner_entry"
            and not entered.is_set()
            and event == "call"
            and frame.f_code is originals[2][1]
            and frame.f_locals.get("self") is controller
            and frame.f_locals.get("readers") is facts["readers"]
        ):
            assert observed["callbacks"] == 1 and observed["pairs"] == 0
            assert matching_invoke(frame)
            operation = getattr(storage._operation_local, "operation", None)
            assert operation is not None
            assert operation.participant is database._maintenance_participant
            facts["inner_entered"] = True
            hold_admission(operation.lease)

    threading.setprofile_all_threads(observe)
    try:
        yield facts
    finally:
        release.set()
        if facts["blocker"] is not None:
            facts["blocker"].result(timeout=10)
        threading.setprofile_all_threads(old_thread)
        sys.setprofile(old_main)
        # The recent method is the intentional fault in two controls; its body
        # is restored by the caller after the actual worker has retired.
        assert selector.__code__ is originals[0][1]
        assert admission.__code__ is originals[1][1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("boundary", "method"),
    [
        ("queued", "_read_recent_scope_sync"),
        ("queued", "_read_refresh_scope_sync"),
        ("admission", "_read_recent_scope_sync"),
        ("admission", "_read_refresh_scope_sync"),
        ("inner_entry", "_read_refresh_scope_sync"),
    ],
)
async def test_character_entry_body_drift_refuses_before_changed_body(
    tmp_path, boundary, method
):
    database = CharactersRAGDB(tmp_path / "entry.sqlite", "entry")
    controller, screen, *_ = _character(database)
    loop = asyncio.get_running_loop()
    previous_executor = loop._default_executor
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="character-entry")
    loop.set_default_executor(executor)
    function = vars(character_context.ConsoleCharacterContextController)[method]
    original_code, before = function.__code__, _source_hashes()
    calls, restore, task = [], None, None
    try:
        with _original_calls(database) as observed:
            with _entry_hold(
                database, controller, boundary, executor, observed
            ) as facts:
                task = asyncio.create_task(_display(controller, screen))
                try:
                    async with asyncio.timeout(10):
                        while not facts["entered"].is_set():
                            await asyncio.sleep(0.01)
                    assert facts["readers"] is not None and not task.done()
                    assert observed["pairs"] == 0 and observed["recent"] == 0
                    assert not observed["connections"]
                    if boundary == "queued":
                        assert observed["callbacks"] == 0
                        assert facts["admission"] is None
                        assert not worker_leases(database)
                    else:
                        assert observed["callbacks"] == 1
                        with storage._lock:
                            assert facts["admission"] in storage._live_leases
                            assert facts["admission"]._key is not None
                        assert facts["inner_entered"] == (boundary == "inner_entry")
                    restore = _body_fault(function, calls)
                    facts["release"].set()
                    assert await task
                finally:
                    facts["release"].set()
                    await asyncio.gather(task, return_exceptions=True)
        # Source refusal occurs before any metadata/SQLite handle is opened;
        # the actual admission lease (when entered) still has to retire below.
        assert not observed["connections"]
        assert not worker_leases(database)
        if facts["admission"] is not None:
            with storage._lock:
                assert facts["admission"]._key is None
                assert facts["admission"] not in storage._live_leases
        assert observed["callbacks"] == 1 and observed["pairs"] == 0
        assert observed["recent"] == 0
        assert controller._presentation_scope_key is None
        assert controller.state.scope_fingerprint is None
        assert before == _source_hashes()
        # Old candidate reaches every original boundary/cleanup assertion above
        # then fails only because the custom same-function body actually ran.
        assert not calls, "changed Character entry body executed before source refusal"
    finally:
        if restore is not None:
            restore()
        assert function.__code__ is original_code
        executor.shutdown(wait=True)
        if previous_executor is None:
            loop._default_executor = None
        else:
            loop.set_default_executor(previous_executor)
        database.close()
