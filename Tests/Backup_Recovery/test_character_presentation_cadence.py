"""Original PR3023 Character display cadence and current ambient ownership."""

import asyncio
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from Tests.UI.test_console_character_context import _controller
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationNavigationService,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.DB import private_sqlite
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.character_context import (
    ConsoleCharacterContextController,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.usefixtures("local_root")]


@contextmanager
def _actual_calls(*, barrier=None):
    """Observe original code objects; never replace guarded/native identities."""
    codes = {
        ConsoleCharacterContextController._read_database_scope_metadata_pair.__code__: "pairs",
        AgentRunsDB.count_subagents_by_conversation.__code__: "counts",
        ConsoleCharacterContextController._load_recent_sync.__code__: "recent",
        CharactersRAGDB.get_character_conversation_search_revision.__code__: "revision",
        private_sqlite.prepare_in_helper.__code__: "helpers",
    }
    if os.name == "nt":
        from tldw_chatbook.Utils import windows_files

        codes[windows_files._native().open_handle.__func__.__code__] = "opens"
    calls = {
        "pairs": 0,
        "counts": 0,
        "helpers": 0,
        "opens": 0,
        "recent": 0,
        "revision": 0,
    }
    previous_thread, previous_main = threading.getprofile(), sys.getprofile()

    def observe(frame, event, _argument):
        if event == "call" and frame.f_code in codes:
            name = codes[frame.f_code]
            calls[name] += 1
            if barrier is not None:
                barrier(frame, name)
        elif event == "return" and frame.f_code in codes and barrier is not None:
            barrier(frame, codes[frame.f_code] + "_return")

    threading.setprofile_all_threads(observe)
    try:
        yield calls
    finally:
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous_main)


def _character(database):
    active, current, conversation = [database], [None], [None]
    store = ConsoleChatStore()
    store.create_session(
        session_id="session",
        workspace_id="workspace",
        ephemeral=True,
        settings=ConsoleSessionSettings(provider="synthetic", model="synthetic"),
    )
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(chachanotes_db=database, app_config={}),
        _console_chat_store=store,
        _console_chat_tearing_down=False,
    )
    controller = _controller(
        database_accessor=lambda: active[0],
        current_character_accessor=lambda: current[0],
        open_conversation_accessor=lambda: conversation[0],
        service_factory=CharacterConversationNavigationService,
    )
    return controller, screen, active, current, conversation


async def _display(controller, screen):
    # The baseline follows the actual prior native screen behavior, rather
    # than failing because the new facade name is absent.
    facade = getattr(controller, "refresh_presentation_if_scope_changed", None)
    if facade is None:
        return await controller.refresh_if_scope_changed()
    return await facade(screen)


@pytest.mark.asyncio
async def test_real_character_display_does_not_repeat_unchanged_metadata(
    tmp_path, record_property
):
    database = CharactersRAGDB(tmp_path / "character.db", "cadence")
    controller, screen, *_ = _character(database)
    try:
        await controller.refresh()
        with _actual_calls() as legacy:
            started = time.monotonic()
            for _ in range(6):
                await controller.refresh_if_scope_changed()
            legacy_elapsed = time.monotonic() - started
        assert legacy["pairs"] == 6, (
            "fresh legacy positive control stopped observing native metadata"
        )
        with _actual_calls() as displayed:
            started = time.monotonic()
            for _ in range(6):
                await _display(controller, screen)
            display_elapsed = time.monotonic() - started
        record_property("legacy_receipt", {**legacy, "seconds": legacy_elapsed})
        record_property("display_receipt", {**displayed, "seconds": display_elapsed})
        assert displayed["pairs"] == 1, (
            "unchanged general display repeated finite metadata callbacks"
        )
        if os.name == "nt":
            assert 0 < displayed["opens"] < legacy["opens"]
        else:
            assert legacy["helpers"] == 6 and displayed["helpers"] == 1
        assert not worker_leases(database)
    finally:
        database.close()


@pytest.mark.asyncio
async def test_character_display_expiry_and_live_scope_remain_fresh(tmp_path):
    database = CharactersRAGDB(tmp_path / "expiry.db", "expiry")
    controller, screen, *_ = _character(database)
    try:
        await controller.refresh()
        await _display(controller, screen)
        with _actual_calls() as warm:
            await _display(controller, screen)
        assert warm["pairs"] == 0
        controller._presentation_scope_at -= 2.01
        with _actual_calls() as expired:
            await _display(controller, screen)
        assert expired["pairs"] == 1
        with _actual_calls() as actions:
            first = await controller._capture_scope()
            assert await controller._scope_is_current(first)
            await controller.refresh_if_scope_changed()
        assert actions["pairs"] == 3, "direct action scope used display memo"
        revision = database.increment_character_conversation_search_revision()
        snapshot = await controller._capture_scope()
        assert snapshot.fingerprint.data_revision == revision
        assert not worker_leases(database)
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "member",
    [
        "session",
        "session_same_id",
        "workspace",
        "settings",
        "app",
        "database",
        "character",
        "conversation",
        "generation",
        "source",
    ],
)
async def test_character_changed_local_owner_is_observed_before_ttl(
    tmp_path, monkeypatch, member
):
    database = CharactersRAGDB(tmp_path / "owner.db", "owner")
    second = CharactersRAGDB(tmp_path / "new-owner.db", "new-owner")
    controller, screen, active, current, conversation = _character(database)
    store = screen._console_chat_store
    try:
        await controller.refresh()
        await _display(controller, screen)
        old = controller._presentation_scope_key
        if member == "session":
            store.create_session(
                session_id="later",
                ephemeral=True,
                settings=ConsoleSessionSettings(
                    provider="synthetic", model="synthetic"
                ),
            )
        elif member == "session_same_id":
            store._sessions["session"] = replace(store.sessions()[0])
        elif member == "workspace":
            store.sessions()[0].workspace_id = "later"
        elif member == "settings":
            store.sessions()[0].conversation_binding_revision += 1
        elif member == "app":
            screen.app_instance.app_config = {}
        elif member == "database":
            active[0] = second
            screen.app_instance.chachanotes_db = second
        elif member == "character":
            current[0] = (1, "Default Assistant")
        elif member == "conversation":
            conversation[0] = "later"
        elif member == "generation":
            controller.invalidate_scope()
        else:
            monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "later.toml"))
        with _actual_calls() as changed:
            await _display(controller, screen)
        assert changed["pairs"] >= 1, "changed local owner waited for display TTL"
        assert controller._presentation_scope_key != old
        assert not worker_leases(database) and not worker_leases(second)
    finally:
        database.close()
        second.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("member", ["workspace", "database", "generation", "teardown"])
async def test_pending_character_read_cannot_memo_or_publish_for_changed_owner(
    tmp_path, member
):
    database = CharactersRAGDB(tmp_path / "pending.db", "pending")
    second = CharactersRAGDB(tmp_path / "replacement.db", "replacement")
    controller, screen, active, *_ = _character(database)
    entered, release = threading.Event(), threading.Event()
    blocked = False

    def barrier(_frame, name):
        nonlocal blocked
        if name == "pairs" and not blocked:
            blocked = True
            entered.set()
            assert release.wait(10)

    pending = None
    try:
        await controller.refresh()
        prior = controller.state
        with _actual_calls(barrier=barrier):
            pending = asyncio.create_task(_display(controller, screen))
            assert await asyncio.to_thread(entered.wait, 10)
            if member == "workspace":
                screen._console_chat_store.sessions()[0].workspace_id = "later"
            elif member == "database":
                active[0] = second
                screen.app_instance.chachanotes_db = second
            elif member == "generation":
                controller.invalidate_scope()
                prior = controller.state
            else:
                screen._console_chat_tearing_down = True
            release.set()
            await pending
        assert controller._presentation_scope_key is None
        assert controller.state is prior, (
            "retired display read published after owner changed"
        )
        assert not worker_leases(database) and not worker_leases(second)
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        database.close()
        second.close()


@pytest.mark.asyncio
async def test_character_waiter_recaptures_owner_after_lock_acquisition(tmp_path):
    database = CharactersRAGDB(tmp_path / "waiters.db", "waiters")
    controller, screen, *_ = _character(database)
    entered, release = threading.Event(), threading.Event()
    blocked = False

    def barrier(_frame, name):
        nonlocal blocked
        if name == "pairs" and not blocked:
            blocked = True
            entered.set()
            assert release.wait(10)

    first = second = None
    try:
        await controller.refresh()
        with _actual_calls(barrier=barrier) as calls:
            first = asyncio.create_task(_display(controller, screen))
            assert await asyncio.to_thread(entered.wait, 10)
            second = asyncio.create_task(_display(controller, screen))
            await asyncio.sleep(0)
            screen._console_chat_store.sessions()[0].workspace_id = "later"
            release.set()
            await asyncio.gather(first, second)
        assert calls["pairs"] == 2, "new owner waiter reused the retired read"
        assert controller._presentation_scope_key == controller._presentation_owner_key(
            screen
        )
        assert not worker_leases(database)
    finally:
        release.set()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None),
            return_exceptions=True,
        )
        database.close()


@pytest.mark.asyncio
async def test_cancelled_character_display_drains_and_does_not_memo(tmp_path):
    database = CharactersRAGDB(tmp_path / "cancel-display.db", "cancel")
    controller, screen, *_ = _character(database)
    entered, release = threading.Event(), threading.Event()
    blocked = False

    def barrier(_frame, name):
        nonlocal blocked
        if name == "revision_return" and not blocked:
            blocked = True
            entered.set()
            assert release.wait(10)

    pending = None
    try:
        await controller.refresh()
        with _actual_calls(barrier=barrier):
            pending = asyncio.create_task(_display(controller, screen))
            assert await asyncio.to_thread(entered.wait, 10)
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done() and worker_leases(database)
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done() and controller._presentation_scope_lock.locked()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert controller._presentation_scope_key is None and not worker_leases(
            database
        )
        with _actual_calls() as retry:
            await _display(controller, screen)
        assert retry["pairs"] == 1
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_actual_metadata_failure_is_not_memoized(tmp_path):
    database = CharactersRAGDB(tmp_path / "failure.db", "failure")
    controller, screen, *_ = _character(database)
    try:
        await controller.refresh()
        with database.transaction() as cursor:
            cursor.execute(
                "UPDATE character_conversation_search_revision SET data_revision = ? WHERE singleton_id = 1",
                ("invalid",),
            )
        with _actual_calls() as failed:
            await _display(controller, screen)
            await _display(controller, screen)
        assert failed["pairs"] >= 4
        assert controller._presentation_scope_key is None and controller.state.error
        with database.transaction() as cursor:
            cursor.execute(
                "UPDATE character_conversation_search_revision SET data_revision = ? WHERE singleton_id = 1",
                (1,),
            )
        await _display(controller, screen)
        assert not controller.state.error and not worker_leases(database)
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("member", ["workspace", "generation", "teardown"])
async def test_display_triggered_refresh_cannot_publish_after_native_owner_change(
    tmp_path, member
):
    database = CharactersRAGDB(tmp_path / "refresh-publication.db", "publication")
    controller, screen, *_ = _character(database)
    entered, release = threading.Event(), threading.Event()
    blocked = False

    def barrier(_frame, name):
        nonlocal blocked
        if name == "recent" and not blocked:
            blocked = True
            entered.set()
            assert release.wait(10)

    pending = None
    try:
        with _actual_calls(barrier=barrier):
            pending = asyncio.create_task(_display(controller, screen))
            assert await asyncio.to_thread(entered.wait, 10)
            if member == "workspace":
                screen._console_chat_store.sessions()[0].workspace_id = "later"
            elif member == "generation":
                controller.invalidate_scope()
            else:
                screen._console_chat_tearing_down = True
            prior = controller.state
            release.set()
            await pending
        assert controller.state is prior, (
            "display-owned refresh published for a replaced ambient owner"
        )
        assert controller._presentation_scope_key is None
        assert not worker_leases(database)
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["cancelled", "superseded", "successful"])
async def test_direct_refresh_outcome_cannot_certify_an_old_display(tmp_path, outcome):
    database = CharactersRAGDB(tmp_path / "direct-outcome.db", "outcome")
    controller, screen, *_ = _character(database)
    entered, release = threading.Event(), threading.Event()
    blocked = False
    pending = None

    def barrier(_frame, name):
        nonlocal blocked
        if name == "recent_return" and not blocked:
            blocked = True
            entered.set()
            assert release.wait(10)

    try:
        await controller.refresh()
        await _display(controller, screen)
        with _actual_calls(barrier=barrier):
            pending = asyncio.create_task(controller.refresh())
            assert await asyncio.to_thread(entered.wait, 10)
            if outcome == "cancelled":
                pending.cancel()
                await asyncio.sleep(0)
                assert not pending.done() and worker_leases(database)
            elif outcome == "superseded":
                controller.invalidate_scope()
            release.set()
            result = await asyncio.gather(pending, return_exceptions=True)
        assert not worker_leases(database)
        if outcome == "cancelled":
            assert isinstance(result[0], asyncio.CancelledError)
        elif outcome == "successful":
            assert controller._successful_refresh_generation == controller._generation
        else:
            assert controller._successful_refresh_generation is None
        await _display(controller, screen)
        assert controller._presentation_scope_key is None or (
            controller._successful_refresh_generation == controller._generation
            and not controller.state.loading
            and not controller.state.error
        ), "old fingerprint certified a refresh that never successfully published"
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reader",
    ["_load_recent_sync", "_load_unavailable_details_sync", "refresh_if_scope_changed"],
)
@pytest.mark.parametrize("mutation", ["instance", "code", "defaults"])
async def test_warm_display_checks_actual_custom_reader(
    tmp_path, monkeypatch, reader, mutation
):
    database = CharactersRAGDB(tmp_path / "custom-display.db", "custom")
    controller, screen, *_ = _character(database)
    try:
        await controller.refresh()
        await _display(controller, screen)
        original = getattr(controller, reader)
        calls = []
        if reader == "refresh_if_scope_changed":

            async def replacement():
                calls.append(True)
                return await original()
        else:

            def replacement(*args, **kwargs):
                return original(*args, **kwargs)

        if mutation == "instance":
            monkeypatch.setattr(controller, reader, replacement)
        elif mutation == "code":
            monkeypatch.setattr(
                original.__func__, "__code__", original.__func__.__code__.replace()
            )
        else:
            monkeypatch.setattr(original.__func__, "__defaults__", (None,))
        with _actual_calls() as actual:
            await _display(controller, screen)
        assert actual["pairs"] == 1, "changed producer incorrectly reused stock memo"
        if reader == "refresh_if_scope_changed" and mutation == "instance":
            assert calls == [True], "custom no-argument callback was bypassed"
        assert not worker_leases(database)
    finally:
        database.close()


@pytest.mark.asyncio
async def test_direct_error_does_not_establish_display_freshness(tmp_path):
    database = CharactersRAGDB(tmp_path / "direct-error.db", "error")
    controller, screen, *_ = _character(database)
    try:
        await controller.refresh()
        await _display(controller, screen)
        with database.transaction() as cursor:
            cursor.execute(
                "UPDATE character_conversation_search_revision SET data_revision = 'invalid' WHERE singleton_id = 1"
            )
        await controller.refresh()
        assert (
            controller.state.error and controller._successful_refresh_generation is None
        )
        with _actual_calls() as reads:
            await _display(controller, screen)
        assert reads["pairs"] > 0
        assert controller._presentation_scope_key is None
        assert not worker_leases(database)
    finally:
        database.close()


@pytest.mark.asyncio
async def test_waiting_display_retains_current_run_scope_owner(tmp_path):
    database = CharactersRAGDB(tmp_path / "run-transition.db", "run")
    controller, screen, *_ = _character(database)
    running = [False]
    controller._run_active = lambda: running[0]
    pending = None
    try:
        await controller.refresh()
        await _display(controller, screen)
        async with controller._presentation_scope_lock:
            pending = asyncio.create_task(_display(controller, screen))
            await asyncio.sleep(0)
            assert not pending.done()
            running[0] = True
            assert controller._scope_checked is None
        with _actual_calls() as reads:
            await pending
        assert reads["pairs"] == 1, "queued display bypassed current run cache owner"
        assert controller._scope_checked is not None
        with _actual_calls() as warm:
            await _display(controller, screen)
        assert warm["pairs"] == 0
        assert not worker_leases(database)
    finally:
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["entry", "queued", "held"])
@pytest.mark.parametrize("closure", ["closing", "closed", "runtime_shutdown"])
async def test_actual_character_screen_closure_refuses_display(
    tmp_path, boundary, closure
):
    from textual.app import App
    from textual.screen import Screen

    from tldw_chatbook.app_lifecycle import LifecycleMixin
    from tldw_chatbook.UI.Screens.chat_screen import _console_screen_is_torn_down

    database = CharactersRAGDB(tmp_path / "screen-closure.db", "closure")
    controller, witness, *_ = _character(database)
    app = App()
    app.chachanotes_db = database
    app.app_config = {}
    app._console_runtime_shutdown_task = None
    entered, release = threading.Event(), threading.Event()
    pending = None
    blocked = False

    def barrier(_frame, name):
        nonlocal blocked
        if name == "recent_return" and not blocked:
            blocked = True
            entered.set()
            assert release.wait(10)

    try:
        async with app.run_test():
            screen = Screen()
            screen.app_instance = app
            screen._console_chat_store = witness._console_chat_store
            await app.push_screen(screen)
            assert not _console_screen_is_torn_down(screen)
            if boundary in {"entry", "queued"}:
                await controller.refresh()
            with _actual_calls(
                barrier=barrier if boundary == "held" else None
            ) as observed:
                if boundary == "held":
                    pending = asyncio.create_task(_display(controller, screen))
                    assert await asyncio.to_thread(entered.wait, 10)
                    assert worker_leases(database)
                if boundary == "queued":
                    await controller._presentation_scope_lock.acquire()
                    pending = asyncio.create_task(_display(controller, screen))
                    await asyncio.sleep(0)
                    assert not pending.done()
                prior = controller.state
                if closure == "closing":
                    await screen._close_messages(wait=False)
                    assert screen._closing
                elif closure == "closed":
                    await app.pop_screen()
                    assert screen._closed
                else:
                    await LifecycleMixin._shutdown_console_runtime(app)
                    assert isinstance(app._console_runtime_shutdown_task, asyncio.Task)
                    assert app._console_runtime_shutdown_task.done()
                assert _console_screen_is_torn_down(screen)
                assert not getattr(screen, "_console_chat_tearing_down", False)
                release.set()
                if boundary == "queued":
                    controller._presentation_scope_lock.release()
                if pending is None:
                    await _display(controller, screen)
                else:
                    await pending
            assert not worker_leases(database)
            assert controller.state is prior, (
                "closed actual screen published old display"
            )
            assert controller._presentation_scope_key is None
            if boundary in {"entry", "queued"}:
                assert observed["pairs"] == observed["recent"] == 0
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_foreign_character_presentation_keeps_no_argument_callback():
    from tldw_chatbook.UI.Console_Modules.character_context import (
        character_presentation_work,
    )

    events = []
    result = object()

    async def declared():
        events.append("called")
        return result

    receiver = SimpleNamespace(refresh_if_scope_changed=declared)
    assert await character_presentation_work(receiver, object()) is result
    assert events == ["called"]
