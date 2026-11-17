"""Run display polls consume completed observations until an input changes."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_character_presentation_cadence import (
    _actual_calls,
    _character,
    _display,
)
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules import character_context

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.usefixtures("local_root")]


@pytest.mark.asyncio
async def test_overlapping_run_display_polls_share_one_completed_native_check(
    tmp_path, monkeypatch
):
    database = CharactersRAGDB(tmp_path / "overlapping-display.db", "overlapping")
    controller, screen, *_ = _character(database)
    controller._run_active = lambda: True
    monkeypatch.setattr(
        character_context, "time", SimpleNamespace(monotonic=lambda: 100.0)
    )
    entered, release = threading.Event(), threading.Event()
    tasks = []

    def hold_first_pair(_frame, name):
        if name == "pairs" and not entered.is_set():
            entered.set()
            assert release.wait(10), "test-owned native read was not released"

    try:
        await controller.refresh()
        with _actual_calls(barrier=hold_first_pair) as observed:
            tasks.append(asyncio.create_task(_display(controller, screen)))
            assert await asyncio.to_thread(entered.wait, 10)
            tasks.extend(
                asyncio.create_task(_display(controller, screen)) for _ in range(2)
            )
            await asyncio.sleep(0)
            release.set()
            await asyncio.gather(*tasks)
        assert observed["pairs"] == 1
        assert not worker_leases(database)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_slow_completed_run_display_does_not_expire_its_own_observation(
    tmp_path, monkeypatch
):
    database = CharactersRAGDB(tmp_path / "slow-display.db", "slow-display")
    controller, screen, *_ = _character(database)
    controller._run_active = lambda: True
    clock = [100.0]
    monkeypatch.setattr(
        character_context, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )

    def elapsed_during_read(_frame, name):
        if name == "pairs":
            clock[0] += character_context.SCOPE_RECHECK_DURING_RUN_SECONDS + 0.5

    try:
        await controller.refresh()
        with _actual_calls(barrier=elapsed_during_read) as initial:
            await _display(controller, screen)
        assert initial["pairs"] == 1
        state = controller.state
        with _actual_calls() as following:
            for _ in range(6):
                assert await _display(controller, screen) is False
        assert following["pairs"] == following["opens"] == 0
        assert controller.state is state
        assert not worker_leases(database)
    finally:
        database.close()


@pytest.mark.asyncio
async def test_new_payload_revision_refreshes_run_display_before_cadence_expires(
    tmp_path, monkeypatch
):
    database = CharactersRAGDB(tmp_path / "payload-display.db", "payload-display")
    controller, screen, *_ = _character(database)
    controller._run_active = lambda: True
    monkeypatch.setattr(
        character_context, "time", SimpleNamespace(monotonic=lambda: 100.0)
    )
    try:
        await controller.refresh()
        await _display(controller, screen)
        revision = database.increment_character_conversation_search_revision()
        screen._console_chat_store.append_message(
            "session", role=ConsoleMessageRole.USER, content="A new saved turn"
        )
        with _actual_calls() as changed:
            assert await _display(controller, screen) is True
        assert changed["pairs"] == 2 and changed["recent"] == 1
        assert controller.state.data_revision == revision
        assert not worker_leases(database)
    finally:
        database.close()


@pytest.mark.asyncio
async def test_run_end_still_reconciles_external_changes_before_idle_memo(
    tmp_path, monkeypatch
):
    database = CharactersRAGDB(tmp_path / "settled-display.db", "settled-display")
    controller, screen, *_ = _character(database)
    running = [False]
    controller._run_active = lambda: running[0]
    monkeypatch.setattr(
        character_context, "time", SimpleNamespace(monotonic=lambda: 100.0)
    )
    try:
        await controller.refresh()
        await _display(controller, screen)
        running[0] = True
        await _display(controller, screen)
        revision = database.increment_character_conversation_search_revision()
        running[0] = False
        with _actual_calls() as settled:
            assert await _display(controller, screen) is True
        assert settled["pairs"] == 2 and settled["recent"] == 1
        assert controller.state.data_revision == revision
        assert not worker_leases(database)
    finally:
        database.close()


@pytest.mark.asyncio
async def test_run_display_reconciles_external_revision_after_completed_cadence(
    tmp_path, monkeypatch
):
    database = CharactersRAGDB(tmp_path / "external-display.db", "external-display")
    controller, screen, *_ = _character(database)
    controller._run_active = lambda: True
    clock = [100.0]
    monkeypatch.setattr(
        character_context, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    try:
        await controller.refresh()
        await _display(controller, screen)
        revision = database.increment_character_conversation_search_revision()
        with _actual_calls() as current:
            assert await _display(controller, screen) is False
        assert current["pairs"] == 0
        clock[0] += 1.01
        with _actual_calls() as expired:
            assert await _display(controller, screen) is True
        assert expired["pairs"] == 2 and expired["recent"] == 1
        assert controller.state.data_revision == revision
        assert not worker_leases(database)
    finally:
        database.close()
