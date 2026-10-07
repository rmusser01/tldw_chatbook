"""Retained Character picker reads own their actual worker connection."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import sys
import threading

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "borrowed"),
    [
        ("normal", False),
        ("cancel", False),
        ("repeated_cancel", False),
        ("normal", True),
        ("cancel", True),
    ],
)
@private_profile_test
async def test_exact_picker_fetch_retires_only_its_owned_native_work(
    request, action, borrowed
):
    from Tests.UI.test_console_character_controller import _controller
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        ResolvedLocalCharacterKey,
    )
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Widgets.Console.console_character_picker_modal import (
        ConsoleCharacterChoice,
    )

    database = CharactersRAGDB(config.get_user_data_dir() / "picker.sqlite", "picker")
    participant = _repository_participant(database)
    character_id = database.add_character_card({"name": "Original"})
    key = ResolvedLocalCharacterKey(database.get_local_authority_id(), character_id)
    database.close()
    store = ConsoleChatStore()
    controller = _controller(
        character_db_accessor=lambda: database,
        ensure_chat_store=lambda: store,
        default_session_settings=lambda: ConsoleSessionSettings(provider="synthetic"),
    )
    entered, release = threading.Event(), threading.Event()
    captured = {}
    callback = CharactersRAGDB.get_character_card_by_id
    code = callback.__code__
    loop = asyncio.get_running_loop()
    task = None
    owner_finished_while_held = False
    after_product = None

    def closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            return True
        return False

    def hold_return(observed_code, offset, result):
        frame = sys._getframe(1)
        if (
            captured
            or observed_code is not code
            or frame.f_locals.get("self") is not database
        ):
            return
        assert result["name"] == "Original"
        connection = database._local.conn
        current = threading.current_thread()
        assert current is not threading.main_thread()
        with storage._lock:
            lease = participant.connections[connection]
            assert lease in storage._live_leases and lease.resource_thread is current
        captured.update(connection=connection, lease=lease, thread=current)
        entered.set()
        assert release.wait(20), "test did not release the original picker read"

    def snapshot_and_cleanup():
        if not captured:
            database.close()
            return None
        assert threading.current_thread() is captured["thread"]
        connection = captured["connection"]
        with storage._lock:
            result = {
                "closed": closed(connection),
                "registered": connection in participant.connections,
                "lease_live": captured["lease"] in storage._live_leases,
                "operations": sum(
                    op.participant is participant for op in storage._operations
                ),
            }
        assert result["operations"] == 0
        if not result["closed"]:
            assert database._local.conn is connection
            if borrowed:
                assert sqlite3.Connection.in_transaction.__get__(connection)
                connection.rollback()
            database.close()  # Exact test-owned resource on its original worker.
        assert closed(connection)
        with storage._lock:
            assert connection not in participant.connections
            assert captured["lease"] not in storage._live_leases
        return result

    monitoring = sys.monitoring
    tool = next(value for value in range(1, 6) if monitoring.get_tool(value) is None)
    monitoring.use_tool_id(tool, "character-picker-native-lifetime")
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="picker-owner"
    ) as executor:
        loop.set_default_executor(executor)
        try:
            if borrowed:

                def borrow():
                    connection = database.get_connection()
                    connection.execute("BEGIN")
                    return connection

                borrowed_connection = await loop.run_in_executor(executor, borrow)
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, hold_return)
            monitoring.set_local_events(tool, code, monitoring.events.PY_RETURN)
            task = asyncio.create_task(
                controller._apply_console_character_choice_async(
                    ConsoleCharacterChoice(character_id, "Original", "new"),
                    expected_key=key,
                    required_database=database,
                    commit_is_current=lambda: True,
                )
            )

            async def wait_until_held():
                while not entered.is_set():
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_until_held(), 15)
            assert captured and not closed(captured["connection"])
            if borrowed:
                assert captured["connection"] is borrowed_connection
            if action != "normal":
                task.cancel()
                await asyncio.sleep(0)
                if action == "repeated_cancel":
                    task.cancel()
                done, _ = await asyncio.wait({task}, timeout=0.1)
                owner_finished_while_held = bool(done)
            assert store.active_session_id is None
        finally:
            release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            # The executor queue fence waits for the original read even on the
            # broken bare-to_thread cancellation route. Record before cleanup.
            after_product = await loop.run_in_executor(executor, snapshot_and_cleanup)
            monitoring.set_local_events(tool, code, 0)
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
            assert monitoring.get_events(tool) == 0
            monitoring.free_tool_id(tool)
            database.close()
    assert CharactersRAGDB.get_character_card_by_id is callback
    assert callback.__code__ is code
    assert (
        not owner_finished_while_held
    ), "picker returned while its native read was live"
    assert after_product is not None and after_product["operations"] == 0
    assert after_product["closed"] is not borrowed
    assert after_product["registered"] is borrowed
    assert after_product["lease_live"] is borrowed
    if action == "normal":
        assert task.exception() is None
        assert (
            store.switch_session(store.active_session_id).character_name == "Original"
        )
    else:
        assert task.cancelled() and store.active_session_id is None
