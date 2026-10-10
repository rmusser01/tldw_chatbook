"""Evidence-only causal controls; root owns installation and Native execution."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import CodeType

import pytest

from Tests.Backup_Recovery.test_console_presentation_cadence import _character, _display
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from tldw_chatbook.Backup_Recovery import participants
from tldw_chatbook import config
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationNavigationService,
)
from tldw_chatbook.DB import base_db, private_sqlite
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules import character_context

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.usefixtures("local_root")]


def _source_hashes():
    return {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in (
            ("base_db", base_db),
            ("private_sqlite", private_sqlite),
            ("participants", participants),
            ("character_context", character_context),
        )
    }


def _closed(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        return True
    return False


@contextmanager
def _original_calls(database, *, barrier=None):
    """Count actual finite callbacks and observe their held real connections."""
    controller_type = character_context.ConsoleCharacterContextController
    invoke_codes = tuple(
        item
        for item in base_db.run_owned_db_call.__code__.co_consts
        if type(item) is CodeType and item.co_name == "invoke"
    )
    assert len(invoke_codes) == 1
    pair = controller_type._read_database_scope_metadata_pair
    recent = controller_type._load_recent_sync
    prepare = private_sqlite.prepare_in_helper
    originals = (
        (base_db.run_owned_db_call, base_db.run_owned_db_call.__code__),
        (pair, pair.__code__),
        (recent, recent.__code__),
        (prepare, prepare.__code__),
    )
    codes = {
        invoke_codes[0]: "callbacks",
        pair.__code__: "pairs",
        recent.__code__: "recent",
    }
    codes[prepare.__code__] = "helpers"
    if os.name == "nt":
        from tldw_chatbook.Utils import windows_files

        native = windows_files._native()
        opener = native.open_handle.__func__
        codes[opener.__code__] = "native_opens"
        originals += ((opener, opener.__code__),)
    stats = {
        "callbacks": 0,
        "pairs": 0,
        "recent": 0,
        "helpers": 0,
        "native_opens": 0,
        "connections": [],
        "leases": [],
        "reader_threads": [],
    }
    old_thread, old_main = threading.getprofile(), sys.getprofile()
    current = threading.current_thread()

    def record_connection():
        participant = database._maintenance_participant
        for connection, lease in tuple(participant.connections.items()):
            if lease.resource_thread is not threading.current_thread():
                continue
            if not any(item is connection for item in stats["connections"]):
                stats["connections"].append(connection)
                stats["leases"].append(lease)

    def observe(frame, event, argument):
        name = codes.get(frame.f_code)
        if name is None or event not in {"call", "return"}:
            return
        if name == "callbacks" and frame.f_locals.get("database") is not database:
            return
        if (
            name in {"pairs", "recent"}
            and frame.f_locals.get("database") is not database
        ):
            return
        if event == "call":
            stats[name] += 1
            if name in {"pairs", "recent"}:
                assert threading.current_thread() is not current
                stats["reader_threads"].append(threading.current_thread())
                record_connection()
            if barrier is not None:
                barrier(frame, name, event, argument)
        elif name in {"pairs", "recent"}:
            record_connection()
            if barrier is not None:
                barrier(frame, name, event, argument)

    threading.setprofile_all_threads(observe)
    try:
        yield stats
    finally:
        threading.setprofile_all_threads(old_thread)
        sys.setprofile(old_main)
        assert all(function.__code__ is code for function, code in originals)


def _retired(database, stats):
    assert stats["connections"], "reader never opened a genuine worker SQLite handle"
    assert all(_closed(item) for item in stats["connections"])
    assert not worker_leases(database)


def _receipt(stats, elapsed):
    return {
        key: value
        for key, value in stats.items()
        if key not in {"connections", "leases", "reader_threads"}
    } | {
        "seconds": elapsed,
        "physical_worker_handles_closed": len(stats["connections"]),
    }


@pytest.mark.asyncio
async def test_changed_character_display_batches_only_inner_original_reads(
    tmp_path, record_property
):
    database = CharactersRAGDB(tmp_path / "character-display.sqlite", "finite-display")
    controller, screen, *_ = _character(database)
    before = _source_hashes()
    try:
        with _original_calls(database) as live:
            started = time.monotonic()
            await controller.refresh_if_scope_changed(force=True)
            live_elapsed = time.monotonic() - started
        _retired(database, live)
        expected = controller.state
        assert live["pairs"] == 3 and live["recent"] == 1
        assert (
            live["callbacks"] == 4
        ), "direct/action positive stopped using its original fresh scopes"
        controller.invalidate_scope()
        with _original_calls(database) as displayed:
            started = time.monotonic()
            assert await _display(controller, screen)
            display_elapsed = time.monotonic() - started
        _retired(database, displayed)
        assert controller.state == expected
        assert displayed["recent"] == 1
        assert before == _source_hashes()
        record_property("direct_original_receipt", _receipt(live, live_elapsed))
        record_property("changed_display_receipt", _receipt(displayed, display_elapsed))
        # Baseline must reach every original read and physical retirement above,
        # then fail this single intended causal callback-count assertion.
        assert (
            displayed["callbacks"] == 1
        ), "changed display still prepares a redundant outer scope callback"
        assert displayed["pairs"] == 2
        assert len(displayed["connections"]) == 1
        if os.name == "nt":
            assert 0 < displayed["native_opens"] < live["native_opens"]
        else:
            assert live["helpers"] == 4 and displayed["helpers"] == 1
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("member", ["database", "character", "conversation"])
async def test_character_resident_identity_change_uses_only_original_refresh(
    tmp_path, record_property, member
):
    first = CharactersRAGDB(tmp_path / "resident.sqlite", "resident-display")
    controller, screen, active, current, conversation = _character(first)
    second = None
    before = _source_hashes()
    try:
        character_id = first.add_character_card({"name": "Resident character"})
        conversation_id = first.add_conversation({"title": "Resident conversation"})
        assert character_id is not None and conversation_id
        assert await _display(controller, screen)
        previous = controller.state.scope_fingerprint
        assert previous is not None
        if member == "database":
            second = CharactersRAGDB(tmp_path / "replacement.sqlite", "replacement")
            active[0] = second
            screen.app_instance.chachanotes_db = second
        elif member == "character":
            current[0] = (character_id, "Resident character")
        else:
            conversation[0] = conversation_id
        database = active[0]
        with _original_calls(database) as observed:
            started = time.monotonic()
            assert await _display(controller, screen)
            elapsed = time.monotonic() - started
        _retired(database, observed)
        fingerprint = controller.state.scope_fingerprint
        assert fingerprint is not None and fingerprint != previous
        assert fingerprint.database_identity == id(database)
        assert fingerprint.current_character_id == (
            current[0][0] if current[0] else None
        )
        assert fingerprint.open_conversation_id == (conversation[0] or "")
        assert (
            fingerprint.data_revision
            == database.get_character_conversation_search_revision()
        )
        assert not controller.state.loading and not controller.state.error
        assert observed["recent"] == 1 and not worker_leases(first)
        assert before == _source_hashes()
        record_property("resident_change_receipt", _receipt(observed, elapsed))
        assert (
            observed["callbacks"] == 1
        ), "resident mismatch repeated the outer precheck"
        assert observed["pairs"] == 2 and len(observed["connections"]) == 1
    finally:
        first.close()
        if second is not None:
            second.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("revision_changed", [False, True], ids=["same", "revision"])
async def test_character_same_ambient_expiry_retains_fresh_metadata_probe(
    tmp_path, revision_changed
):
    database = CharactersRAGDB(tmp_path / "same-ambient.sqlite", "same-ambient")
    controller, screen, active, current, conversation = _character(database)
    before = _source_hashes()
    try:
        assert await _display(controller, screen)
        previous = controller.state.scope_fingerprint
        owner = controller._presentation_owner_key(screen)
        assert previous is not None
        revision = previous.data_revision
        if revision_changed:
            revision = database.increment_character_conversation_search_revision()
            assert revision > previous.data_revision
        controller._presentation_scope_at -= (
            character_context._CHARACTER_PRESENTATION_TTL_SECONDS + 0.01
        )
        assert controller._presentation_owner_key(screen) == owner
        assert active[0] is database and current[0] is None and conversation[0] is None
        with _original_calls(database) as observed:
            assert await _display(controller, screen) is revision_changed
        _retired(database, observed)
        assert controller.state.scope_fingerprint.data_revision == revision
        assert controller.state.scope_fingerprint.database_identity == id(database)
        assert not controller.state.loading and not controller.state.error
        assert observed["callbacks"] == (2 if revision_changed else 1)
        assert observed["pairs"] == (3 if revision_changed else 1)
        assert observed["recent"] == int(revision_changed)
        assert before == _source_hashes()
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "member",
    [
        "generation",
        "session",
        "session_same_id",
        "workspace",
        "settings",
        "app_config",
        "database",
        "character",
        "conversation",
        "config_generation",
    ],
)
async def test_character_inner_batch_rejects_changed_owner_after_final_original_pair(
    tmp_path, member
):
    first = CharactersRAGDB(tmp_path / "first-display.sqlite", "first-display")
    second = CharactersRAGDB(tmp_path / "second-display.sqlite", "second-display")
    controller, screen, active, current, conversation = _character(first)
    store = screen._console_chat_store
    entered, release = threading.Event(), threading.Event()
    changed = False
    pair_count = 0
    late_final_publications = []
    original_publish = character_context.ConsoleCharacterContextController._publish
    previous_main, previous_thread = sys.getprofile(), threading.getprofile()
    body_task = None
    before = _source_hashes()

    def barrier(_frame, name, event, _argument):
        nonlocal pair_count
        if name == "pairs" and event == "call":
            pair_count += 1
        if name == "pairs" and event == "return" and pair_count == 2:
            entered.set()
            assert release.wait(10), "test-owned final original pair was not released"

    def publication(frame, event, argument):
        # The read observer remains original and is installed below; only this
        # exact original publication frame is additionally observed on the loop.
        if event == "call" and frame.f_code is original_publish.__code__:
            state = frame.f_locals.get("state")
            if (
                changed
                and frame.f_locals.get("self") is controller
                and state is not None
                and not state.loading
                and state.scope_fingerprint is not None
            ):
                late_final_publications.append(state.scope_fingerprint)
        if selected_observer is not None:
            selected_observer(frame, event, argument)

    selected_observer = None
    try:
        with _original_calls(first, barrier=barrier) as observed:
            selected_observer = sys.getprofile()
            sys.setprofile(publication)
            body_task = asyncio.create_task(_display(controller, screen))
            assert await asyncio.to_thread(entered.wait, 10)
            assert worker_leases(first)
            owner = next(
                item for item in store.sessions() if item.id == store.active_session_id
            )
            if member == "generation":
                controller.invalidate_scope()
            elif member == "session":
                store.create_session(
                    session_id="other", workspace_id="other", ephemeral=True
                )
            elif member == "session_same_id":
                store._sessions[owner.id] = replace(owner)
            elif member == "workspace":
                owner.workspace_id = "replacement-workspace"
            elif member == "settings":
                store.replace_session_settings(
                    owner.id, replace(owner.settings, model="replacement-model")
                )
            elif member == "app_config":
                screen.app_instance.app_config = dict(screen.app_instance.app_config)
            elif member == "database":
                active[0] = second
                screen.app_instance.chachanotes_db = second
            elif member == "character":
                current[0] = (9, "Replacement")
            elif member == "conversation":
                conversation[0] = "replacement-conversation"
            elif member == "config_generation":
                assert config.save_setting_to_cli_config(
                    "console", "character_batch_control", "changed"
                )
            changed = True
            release.set()
            await body_task
            sys.setprofile(selected_observer)
        _retired(first, observed)
        assert (
            not late_final_publications
        ), "an old joined display read published after owner drift"
        assert controller._presentation_scope_key is None
        assert before == _source_hashes()
        assert not worker_leases(second)
    finally:
        release.set()
        if body_task is not None:
            await asyncio.gather(body_task, return_exceptions=True)
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous_main)
        first.close()
        second.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route", ["custom_service", "borrowed_recent", "borrowed_pair"]
)
async def test_character_custom_and_borrowed_callbacks_keep_original_route(
    tmp_path, route
):
    database = CharactersRAGDB(tmp_path / "custom-display.sqlite", "custom-display")
    controller, screen, *_ = _character(database)
    other, *_ = _character(database)
    custom_calls = []
    if route == "custom_service":

        def declared_factory(actual_database, *, current_character=None):
            assert actual_database is database
            custom_calls.append((actual_database, current_character))
            return CharacterConversationNavigationService(
                actual_database, current_character=current_character
            )

        controller._service_factory = declared_factory
    elif route == "borrowed_recent":
        controller._load_recent_sync = other._load_recent_sync
    else:
        controller._read_database_scope_metadata_pair = (
            other._read_database_scope_metadata_pair
        )
    before = _source_hashes()
    try:
        with _original_calls(database) as observed:
            assert await _display(controller, screen)
        _retired(database, observed)
        assert (
            observed["callbacks"] == 4
        ), "custom or foreign receiver was grouped under a stock-only interval"
        assert observed["pairs"] == 3 and observed["recent"] == 1
        if route == "custom_service":
            assert len(custom_calls) == 1
        assert controller.state.scope_fingerprint is not None
        assert before == _source_hashes()
    finally:
        database.close()


@pytest.mark.asyncio
async def test_character_inner_batch_preserves_real_same_worker_borrowed_connection(
    tmp_path,
):
    database = CharactersRAGDB(tmp_path / "borrowed-display.sqlite", "borrowed-display")
    controller, screen, *_ = _character(database)
    loop = asyncio.get_running_loop()
    previous_executor = loop._default_executor
    owned_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="character-borrow-control"
    )
    loop.set_default_executor(owned_executor)
    connection = None
    before = _source_hashes()
    try:
        connection = await asyncio.to_thread(database.get_connection)
        assert not _closed(connection)
        held = worker_leases(database)
        assert len(held) == 1
        with _original_calls(database) as observed:
            assert await _display(controller, screen)
        assert observed["callbacks"] == 1
        assert observed["pairs"] == 2 and observed["recent"] == 1
        assert (
            len(observed["connections"]) == 1
            and observed["connections"][0] is connection
        )
        assert worker_leases(database) == held and not _closed(connection)
        await asyncio.to_thread(database.close_connection)
        assert _closed(connection) and not worker_leases(database)
        assert before == _source_hashes()
    finally:
        if connection is not None:
            await asyncio.to_thread(database.close_connection)
            assert _closed(connection)
        # Retire only the test-created executor after its exact DB owner closes.
        owned_executor.shutdown(wait=True)
        if previous_executor is not None:
            loop.set_default_executor(previous_executor)
        else:
            loop._default_executor = None
        database.close()


@pytest.mark.asyncio
async def test_character_batch_cancellation_keeps_actual_native_callback_owned(
    tmp_path,
):
    database = CharactersRAGDB(tmp_path / "cancel-display.sqlite", "finite-display")
    controller, screen, *_ = _character(database)
    entered, release = threading.Event(), threading.Event()
    held = False

    def barrier(_frame, name, event, _argument):
        nonlocal held
        if name == "recent" and event == "return" and not held:
            held = True
            entered.set()
            assert release.wait(10), "test-owned original reader hold was not released"

    task = None
    before = _source_hashes()
    try:
        with _original_calls(database, barrier=barrier) as observed:
            task = asyncio.create_task(_display(controller, screen))
            assert await asyncio.to_thread(entered.wait, 10)
            assert worker_leases(
                database
            ), "held original read has no genuine native owner"
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert controller._presentation_scope_lock.locked()
            assert worker_leases(database)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        _retired(database, observed)
        assert controller._presentation_scope_key is None
        assert before == _source_hashes()
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        database.close()
