"""Known display changes need the original refresh, not an extra change probe."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from Tests.Backup_Recovery.test_character_presentation_cadence import (
    _character,
    _display,
)
from Tests.Backup_Recovery.test_character_refresh_finite_batch import (
    _closed,
    _original_calls,
    _retired,
    _source_hashes,
)
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414 - pytest fixture re-export
)
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationNavigationService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.usefixtures("local_root")]


async def _changed_display(tmp_path):
    db = CharactersRAGDB(tmp_path / "known-change.sqlite", "known-change")
    controller, screen, active, current, conversation = _character(db)
    try:
        old = db.add_conversation({"title": "Before first persistence"})
        new = db.add_conversation({"title": "New persisted chat"})
        assert old and new and old != new
        conversation[0] = old
        await controller.refresh()
        assert controller.state.scope_fingerprint.open_conversation_id == old
        controller._run_active = lambda: True
        conversation[0] = new
        return db, controller, screen, active, current, conversation
    except BaseException:
        db.close()
        raise


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route,callbacks,pairs,recent",
    [
        ("display", 1, 2, 1),
        ("direct", 4, 3, 1),
        ("forced", 4, 3, 1),
        ("custom", 4, 3, 1),
        ("custom_reader", 4, 3, 1),
        ("custom_db", 4, 3, 1),
        ("unchanged", 1, 1, 0),
        # TASK-33620.15.6: an invalidated stock display decides in its batch.
        ("invalidated", 1, 2, 1),
    ],
)
async def test_known_conversation_refresh_preserves_native_checks_and_fallbacks(
    tmp_path, route, callbacks, pairs, recent, record_property
):
    db, controller, screen, _, _, conversation = await _changed_display(tmp_path)
    before = _source_hashes()
    try:
        if route == "unchanged":
            conversation[0] = controller.state.scope_fingerprint.open_conversation_id
        elif route == "invalidated":
            controller.invalidate_scope()
        elif route == "custom":
            controller._service_factory = lambda *args, **kwargs: (
                CharacterConversationNavigationService(*args, **kwargs)
            )
        if route == "custom_reader":
            original = controller._capture_scope
            controller._capture_scope = lambda: original()
        elif route == "custom_db":
            original = db.get_character_conversation_search_revision
            db.get_character_conversation_search_revision = lambda: original()
        with _original_calls(db) as observed:
            if route in {"direct", "forced"}:
                assert await controller.refresh_if_scope_changed(
                    force=route == "forced"
                )
            else:
                changed = await _display(controller, screen)
                assert changed is (route != "unchanged")
        _retired(db, observed)
        assert not controller.state.loading and not controller.state.error
        assert (
            controller.state.scope_fingerprint.open_conversation_id == conversation[0]
        )
        assert (
            controller.state.data_revision
            == db.get_character_conversation_search_revision()
        )
        assert observed["recent"] == recent
        assert before == _source_hashes()
        record_property(
            "native_read_receipt",
            {k: observed[k] for k in ("callbacks", "pairs", "recent", "helpers")},
        )
        # Baseline reaches all native/retirement assertions, then fails here.
        assert observed["callbacks"] == callbacks, (
            "known changed display repeated a preliminary native probe"
        )
        assert observed["pairs"] == pairs
        assert len(observed["connections"]) == callbacks
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "generation",
        "conversation",
        "session",
        "settings",
        "app_config",
        "source",
        "cancel",
    ],
)
async def test_known_change_retains_final_pair_refusal_and_physical_retirement(
    tmp_path, change
):
    db, controller, screen, _, _, conversation = await _changed_display(tmp_path)
    entered, release = threading.Event(), threading.Event()
    recent_returned = False
    task = None
    before = _source_hashes()
    original_scope = controller.state.scope_fingerprint

    def barrier(frame, name, event, argument):
        nonlocal recent_returned
        if name == "recent" and event == "return":
            recent_returned = True
        if (
            name == "pairs"
            and event == "return"
            and recent_returned
            and not entered.is_set()
        ):
            entered.set()
            assert release.wait(10), "test-owned final original pair not released"

    try:
        with _original_calls(db, barrier=barrier) as observed:
            task = asyncio.create_task(_display(controller, screen))
            assert await asyncio.to_thread(entered.wait, 10)
            assert not task.done() and worker_leases(db)
            assert any(
                not _closed(connection) for connection in observed["connections"]
            )
            store = screen._console_chat_store
            if change == "generation":
                controller.invalidate_scope()
            elif change == "conversation":
                conversation[0] = "later-conversation"
            elif change == "session":
                store.create_session(
                    session_id="replacement", workspace_id="workspace", ephemeral=True
                )
            elif change == "settings":
                owner = next(
                    item
                    for item in store.sessions()
                    if item.id == store.active_session_id
                )
                store.replace_session_settings(
                    owner.id, replace(owner.settings, model="changed-model")
                )
            elif change == "app_config":
                screen.app_instance.app_config = dict(screen.app_instance.app_config)
            elif change == "source":
                controller._load_recent_sync = lambda *args, **kwargs: ((), ())
            else:
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done() and worker_leases(db)
            release.set()
            if change == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await task
        _retired(db, observed)
        assert controller.state.scope_fingerprint in (original_scope, None)
        assert before == _source_hashes()
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        db.close()


@pytest.mark.asyncio
async def test_known_change_retries_after_actual_metadata_write_before_final_pair(
    tmp_path,
):
    db, controller, screen, _, _, _ = await _changed_display(tmp_path)
    entered, release = threading.Event(), threading.Event()
    task = None
    held = False
    before = _source_hashes()
    revision = db.get_character_conversation_search_revision()

    def barrier(frame, name, event, argument):
        nonlocal held
        if name == "recent" and event == "return" and not held:
            held = True
            entered.set()
            assert release.wait(10), "test-owned original row read not released"

    try:
        with _original_calls(db, barrier=barrier) as observed:
            task = asyncio.create_task(_display(controller, screen))
            assert await asyncio.to_thread(entered.wait, 10)
            assert worker_leases(db)
            changed = db.increment_character_conversation_search_revision()
            assert changed > revision
            release.set()
            assert await task
        _retired(db, observed)
        assert observed["recent"] == 2, (
            "changed metadata was accepted without a fresh retry"
        )
        assert controller.state.scope_fingerprint.data_revision == changed
        assert not controller.state.error and not controller.state.loading
        assert before == _source_hashes()
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        db.close()


@pytest.mark.asyncio
async def test_known_change_keeps_exact_same_worker_borrowed_connection(tmp_path):
    db, controller, screen, *_ = await _changed_display(tmp_path)
    loop = asyncio.get_running_loop()
    previous = loop._default_executor
    executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="known-change-borrower"
    )
    loop.set_default_executor(executor)
    connection = None
    before = _source_hashes()
    try:
        connection = await asyncio.to_thread(db.get_connection)
        held = worker_leases(db)
        assert len(held) == 1 and not _closed(connection)
        with _original_calls(db) as observed:
            assert await _display(controller, screen)
        assert observed["connections"] == [connection]
        assert worker_leases(db) == held and not _closed(connection)
        await asyncio.to_thread(db.close_connection)
        assert _closed(connection) and not worker_leases(db)
        assert before == _source_hashes()
        assert observed["callbacks"] == 1, (
            "known changed borrower repeated a preliminary native probe"
        )
        assert observed["pairs"] == 2 and observed["recent"] == 1
    finally:
        if connection is not None:
            await asyncio.to_thread(db.close_connection)
            assert _closed(connection)
        executor.shutdown(wait=True)
        if previous is None:
            loop._default_executor = None
        else:
            loop.set_default_executor(previous)
        db.close()
