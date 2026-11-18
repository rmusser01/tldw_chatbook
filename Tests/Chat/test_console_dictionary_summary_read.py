"""Original summary jobs retain native cleanup separately from their result."""

import asyncio
import copy
import sys
import threading

import pytest

from Tests.private_profile import private_profile_test


def _sources():
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery.chat_source_participants import (
        build_dictionary_service,
    )
    from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import (
        ChatDictionaryScopeService,
    )
    from tldw_chatbook.app_services import open_chachanotes_db

    config.load_settings()
    database = open_chachanotes_db(
        config.get_chachanotes_db_path(), client_id=config.CLI_APP_CLIENT_ID
    )
    local = build_dictionary_service(database)
    return database, ChatDictionaryScopeService(
        local_service=local, server_service=None
    )


async def _read(receipt, service, character_id=2147483647, *, world=False):
    from tldw_chatbook.Chat.console_preparation_reads import (
        preparation_read_for,
        run_preparation_read,
    )

    def original():
        handoff = preparation_read_for(service)
        if world:
            return receipt.read_world_books(
                service.local_service.db, None, handoff=handoff
            )
        return receipt.read(service, None, character_id, handoff=handoff)

    return await run_preparation_read(
        original,
        creator=service,
        session_id=None,
        reads=set(),
        require_current=lambda: None,
        source=service,
    )


@private_profile_test
@pytest.mark.asyncio
async def test_original_dictionary_and_world_summary_receipts_and_failure(request):
    from tldw_chatbook.Chat.console_dictionary_summary_read import (
        DictionarySummaryRead,
        dictionary_summary_retired,
    )

    database, service = _sources()
    try:
        receipt = DictionarySummaryRead(service)
        result = await _read(receipt, service)
        assert result == {"dictionaries": [], "source": "local"}
        assert dictionary_summary_retired(receipt)
        assert not dictionary_summary_retired(copy.copy(receipt))
        assert (
            receipt.creator_observation.creator[1]
            is not receipt.worker_observation.creator[1]
        )
        assert receipt.producer.done() and not receipt.producer.cancelled()
        assert receipt.job._state == "closed"
        assert receipt.worker_observation.connections
        failed = DictionarySummaryRead(service)
        with pytest.raises(ValueError):
            await _read(failed, service, "not-an-integer")
        assert dictionary_summary_retired(failed)
        assert failed.job is not None and failed.worker_observation.closed
        world = DictionarySummaryRead(service)
        assert await _read(world, service, world=True) == {
            "world_books": [],
            "source": "local",
        }
        assert dictionary_summary_retired(world)
        with pytest.raises(RuntimeError, match="already issued"):
            await _read(receipt, service)
    finally:
        database.close()


@private_profile_test
@pytest.mark.asyncio
async def test_dictionary_close_uncertainty_stays_negative_after_cleanup(request):
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.Chat.console_dictionary_summary_read import (
        DictionarySummaryRead,
        dictionary_summary_retired,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database, service = _sources()
    receipt = DictionarySummaryRead(service)
    injected = []

    def fail_original_close(frame):
        if frame.f_locals.get("self") is database and not injected:
            injected.append(True)
            raise OSError("dictionary close failure")

    try:
        with _observe(
            CharactersRAGDB.close_connection.__code__,
            sys.monitoring.events.PY_START,
            fail_original_close,
        ):
            with pytest.raises(OSError, match="dictionary close failure"):
                await _read(receipt, service)
        assert injected and receipt.worker_observation.connections
        assert not dictionary_summary_retired(receipt)
        # The original quiescence owner closes retained same-file handles after
        # all real producers have joined; it does not rewrite the failed receipt.
        assert database.close_idle_connections()
        assert receipt.worker_observation.confirmed()
        assert not dictionary_summary_retired(receipt)
    finally:
        database.close_idle_connections()
        database.close()


@private_profile_test
@pytest.mark.asyncio
async def test_original_dictionary_worker_tail_is_joined_before_outer_read(request):
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.Backup_Recovery.dictionary_source_job import _DictionaryJob
    from tldw_chatbook.Chat.console_dictionary_summary_read import (
        DictionarySummaryRead,
        dictionary_summary_retired,
    )

    database, service = _sources()
    receipt = DictionarySummaryRead(service)
    entered, release = threading.Event(), threading.Event()

    def hold_original_tail(frame):
        if frame.f_locals.get("self") is receipt.job:
            entered.set()
            assert release.wait(10), "Original dictionary tail was not released"

    task = None
    try:
        with _observe(
            _DictionaryJob._work.__code__,
            sys.monitoring.events.PY_RETURN,
            hold_original_tail,
        ):
            task = asyncio.create_task(_read(receipt, service))
            async with asyncio.timeout(10):
                while not entered.is_set():
                    await asyncio.sleep(0.01)
            assert not task.done() and not receipt.producer.done()
            assert not dictionary_summary_retired(receipt)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert dictionary_summary_retired(receipt)
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        database.close()
