"""Additional root-run controls; no guarded callback or native source is replaced."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import CodeType, FunctionType

import pytest

from Tests.Backup_Recovery.test_character_refresh_finite_batch import (
    _closed,
    _original_calls,
    _retired,
    _source_hashes,
)
from Tests.Backup_Recovery.test_console_presentation_cadence import _character, _display
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationNavigationService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB import base_db
from tldw_chatbook.UI.Console_Modules import character_context

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.usefixtures("local_root")]


@contextmanager
def _uninstalled_calls(database, worker):
    """Memory/subclass owners have no installed core participant to inspect."""
    controller = character_context.ConsoleCharacterContextController
    functions = (
        base_db.run_owned_db_call,
        controller._read_database_scope_metadata_pair,
        controller._load_recent_sync,
    )
    originals = tuple((function, function.__code__) for function in functions)
    invoke = tuple(
        code
        for code in originals[0][1].co_consts
        if type(code) is CodeType and code.co_name == "invoke"
    )
    assert len(invoke) == 1
    codes = {
        invoke[0]: "callbacks",
        originals[1][1]: "pairs",
        originals[2][1]: "recent",
    }
    stats = dict(callbacks=0, pairs=0, recent=0)
    previous_thread, previous_main = threading.getprofile(), sys.getprofile()

    def observe(frame, event, _argument):
        name = codes.get(frame.f_code)
        if event == "call" and name and frame.f_locals.get("database") is database:
            assert threading.current_thread() is worker
            stats[name] += 1

    threading.setprofile_all_threads(observe)
    try:
        yield stats
    finally:
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous_main)
        assert all(function.__code__ is code for function, code in originals)


def _body_fault(function, calls):
    """Change only an unguarded original body, retaining its actual function."""
    original = FunctionType(
        function.__code__,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    original.__kwdefaults__ = function.__kwdefaults__
    namespace = function.__globals__
    assert "_character_batch_test_original" not in namespace
    assert "_character_batch_test_calls" not in namespace
    namespace["_character_batch_test_original"] = original
    namespace["_character_batch_test_calls"] = calls
    exec(
        "def _character_batch_test_body(self, *args, **kwargs):\n"
        "    _character_batch_test_calls.append(type(self).__name__)\n"
        "    return _character_batch_test_original(self, *args, **kwargs)\n",
        namespace,
    )
    old = function.__code__
    function.__code__ = namespace.pop("_character_batch_test_body").__code__

    def restore():
        function.__code__ = old
        namespace.pop("_character_batch_test_original")
        namespace.pop("_character_batch_test_calls")

    return restore


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ["metadata", "recent"])
async def test_character_batch_midbody_change_refuses_before_next_stock_reader(
    tmp_path, reader
):
    database = CharactersRAGDB(tmp_path / "body.sqlite", "body")
    controller, screen, *_ = _character(database)
    body_calls = []
    restore = None
    pair_count = 0
    function = (
        CharactersRAGDB.get_local_authority_id
        if reader == "metadata"
        else CharacterConversationNavigationService.recent_groups
    )
    before = _source_hashes()

    def barrier(_frame, name, event, _argument):
        nonlocal restore, pair_count
        if name == "pairs" and event == "call":
            pair_count += 1
        if (
            name == "pairs"
            and event == "return"
            and pair_count == 1
            and restore is None
        ):
            restore = _body_fault(function, body_calls)

    try:
        with _original_calls(database, barrier=barrier) as observed:
            assert await _display(controller, screen)
        _retired(database, observed)
        assert restore is not None, "original inner pair boundary was not reached"
        assert not body_calls, "a changed original body ran after stock qualification"
        assert observed["callbacks"] == 1 and observed["pairs"] == 1
        assert (
            observed["recent"] == 0
        ), "groups were read after reader provenance changed"
        assert controller._presentation_scope_key is None
        assert controller.state.scope_fingerprint is None
        assert before == _source_hashes()
    finally:
        if restore is not None:
            restore()
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ["metadata", "recent"])
async def test_character_preinstalled_body_change_retains_original_custom_route(
    tmp_path, reader
):
    database = CharactersRAGDB(tmp_path / "custom-body.sqlite", "custom-body")
    controller, screen, *_ = _character(database)
    function = (
        CharactersRAGDB.get_local_authority_id
        if reader == "metadata"
        else CharacterConversationNavigationService.recent_groups
    )
    calls = []
    before = _source_hashes()
    restore = _body_fault(function, calls)
    try:
        with _original_calls(database) as observed:
            assert await _display(controller, screen)
        _retired(database, observed)
        assert calls, "the declared custom body was suppressed"
        assert observed["callbacks"] == 4
        assert observed["pairs"] == 3 and observed["recent"] == 1
        assert controller.state.scope_fingerprint is not None
        assert before == _source_hashes()
    finally:
        restore()
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["groups", "final_metadata"])
async def test_character_batch_real_sql_failure_keeps_final_pair_error_semantics(
    tmp_path, failure
):
    database = CharactersRAGDB(tmp_path / "sql-error.sqlite", "sql-error")
    controller, screen, *_ = _character(database)
    pairs, denials = 0, []
    authorizer_connection = None
    before = _source_hashes()

    def authorize(action, table, _column, _db_name, _trigger):
        refused = action == sqlite3.SQLITE_READ and table == (
            "conversations"
            if failure == "groups"
            else "character_conversation_search_revision"
        )
        if refused:
            denials.append(table)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def barrier(frame, name, event, _argument):
        nonlocal pairs, authorizer_connection
        if name == "pairs" and event == "call":
            pairs += 1
        install = event == "call" and (
            failure == "groups"
            and name == "recent"
            or failure == "final_metadata"
            and name == "pairs"
            and pairs in (2, 4, 6)
        )
        if install:
            # Open through the unchanged original getter on this exact worker.
            # The authorizer rejects an actual read, never changes a reader body.
            assert frame.f_locals.get("database") is database
            authorizer_connection = database.get_connection()
            authorizer_connection.set_authorizer(authorize)
        retire = event == "return" and (
            failure == "groups"
            and name == "recent"
            or failure == "final_metadata"
            and name == "pairs"
        )
        if retire and authorizer_connection is not None:
            authorizer_connection.set_authorizer(None)
            authorizer_connection = None

    try:
        with _original_calls(database, barrier=barrier) as observed:
            assert await _display(controller, screen)
        _retired(database, observed)
        assert denials, "original SQL fault boundary was not exercised"
        assert authorizer_connection is None
        if failure == "groups":
            assert observed["callbacks"] == 1
            assert observed["pairs"] == 2 and observed["recent"] == 1
            assert (
                controller.state.error == "Could not load local character chats · Retry"
            )
        else:
            assert len(denials) == 3
            assert observed["callbacks"] == 3
            assert observed["pairs"] == 6 and observed["recent"] == 3
            assert controller.state.error == "Local character chats changed · Retry"
        assert controller.state.scope_fingerprint is None
        assert (
            not controller.state.loading and controller._presentation_scope_key is None
        )
        assert before == _source_hashes()
    finally:
        # The finite workers retired their handles; never close from this loop.
        database.close()


@pytest.mark.asyncio
async def test_character_batch_actual_revision_change_retries_both_original_pairs(
    tmp_path,
):
    database = CharactersRAGDB(tmp_path / "revision.sqlite", "revision")
    controller, screen, *_ = _character(database)
    old = database.get_character_conversation_search_revision()
    pairs, changed = 0, False
    before = _source_hashes()

    def barrier(frame, name, event, _argument):
        nonlocal pairs, changed
        if name == "pairs" and event == "call":
            pairs += 1
        if name == "pairs" and event == "return" and pairs == 1 and not changed:
            assert frame.f_locals.get("database") is database
            assert (
                database.increment_character_conversation_search_revision() == old + 1
            )
            changed = True

    try:
        with _original_calls(database, barrier=barrier) as observed:
            assert await _display(controller, screen)
        _retired(database, observed)
        assert changed and observed["callbacks"] == 2
        assert observed["pairs"] == 4 and observed["recent"] == 2
        assert controller.state.scope_fingerprint.data_revision == old + 1
        assert controller.state.data_revision == old + 1
        assert not controller.state.loading and not controller.state.error
        assert before == _source_hashes()
    finally:
        database.close()


@pytest.mark.asyncio
async def test_character_batch_original_postpair_owner_error_propagates_after_retirement(
    tmp_path,
):
    database = CharactersRAGDB(tmp_path / "ambient-error.sqlite", "ambient-error")
    controller, screen, *_ = _character(database)
    original_accessor = controller._current_character_accessor
    fail = [False]
    pair_count = 0
    before = _source_hashes()

    def declared_accessor():
        if fail[0]:
            fail[0] = False
            raise TypeError("test-owned original ambient error")
        return original_accessor()

    controller._current_character_accessor = declared_accessor

    def barrier(_frame, name, event, _argument):
        nonlocal pair_count
        if name == "pairs" and event == "call":
            pair_count += 1
        if name == "pairs" and event == "return" and pair_count == 1:
            fail[0] = True

    try:
        with _original_calls(database, barrier=barrier) as observed:
            with pytest.raises(TypeError, match="test-owned original ambient error"):
                await _display(controller, screen)
        _retired(database, observed)
        assert pair_count == 1 and observed["callbacks"] == 1
        assert observed["recent"] == 0 and not fail[0]
        assert controller._presentation_scope_key is None
        assert controller.state.scope_fingerprint is None
        assert before == _source_hashes()
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["memory", "subclass"])
async def test_character_nonstandard_database_keeps_declared_worker_lifetime(
    tmp_path, route
):
    class DeclaredCharactersDatabase(CharactersRAGDB):
        pass

    loop = asyncio.get_running_loop()
    old_executor = loop._default_executor
    executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="character-custom-db"
    )
    loop.set_default_executor(executor)
    database = connection = None
    before = _source_hashes()
    try:
        # Keep the actual memory schema and custom receiver on its owning worker.
        constructor = (
            CharactersRAGDB if route == "memory" else DeclaredCharactersDatabase
        )
        path = ":memory:" if route == "memory" else tmp_path / "subclass.sqlite"
        database = await asyncio.to_thread(constructor, path, "custom-db")
        connection = await asyncio.to_thread(database.get_connection)
        worker = await asyncio.to_thread(threading.current_thread)
        controller, screen, *_ = _character(database)
        with _uninstalled_calls(database, worker) as observed:
            assert await _display(controller, screen)
        assert observed["callbacks"] == 4
        assert observed["pairs"] == 3 and observed["recent"] == 1
        assert not _closed(connection), "custom/memory connection was newly adopted"
        assert controller.state.scope_fingerprint is not None
        assert before == _source_hashes()
        await asyncio.to_thread(database.close)
        assert _closed(connection)
    finally:
        if database is not None:
            await asyncio.to_thread(database.close)
        executor.shutdown(wait=True)
        if old_executor is None:
            loop._default_executor = None
        else:
            loop.set_default_executor(old_executor)
