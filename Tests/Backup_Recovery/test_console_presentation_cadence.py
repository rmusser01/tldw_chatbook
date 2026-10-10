"""Real finite display reads keep bounded cadence and exact ambient ownership."""

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
from Tests.UI.test_console_refresh_read_batching import _agent, _finish
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationNavigationService,
)
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
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
        controller.state = replace(
            controller.state, scope_fingerprint=await controller._fingerprint()
        )
        with _actual_calls() as legacy:
            started = time.monotonic()
            for _ in range(6):
                await controller.refresh_if_scope_changed()
            legacy_elapsed = time.monotonic() - started
        assert (
            legacy["pairs"] == 6
        ), "fresh legacy positive control stopped observing native metadata"
        with _actual_calls() as displayed:
            started = time.monotonic()
            for _ in range(6):
                await _display(controller, screen)
            display_elapsed = time.monotonic() - started
        record_property("legacy_receipt", {**legacy, "seconds": legacy_elapsed})
        record_property("display_receipt", {**displayed, "seconds": display_elapsed})
        assert (
            displayed["pairs"] == 1
        ), "unchanged general display repeated finite metadata callbacks"
        if os.name == "nt":
            assert 0 < displayed["opens"] < legacy["opens"]
        else:
            assert legacy["helpers"] == 6 and displayed["helpers"] == 1
        assert not worker_leases(database)
    finally:
        database.close()


@pytest.mark.asyncio
async def test_active_count_display_uses_existing_two_second_interval(
    tmp_path, record_property
):
    database = AgentRunsDB(tmp_path / "counts.db", "cadence")
    primary = database.create_run(conversation_id="conv", agent_kind="primary")
    database.create_run(
        conversation_id="conv", agent_kind="subagent", parent_run_id=primary
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=database, store=None, provider_gateway=None
    )
    agent, tasks = _agent(bridge)
    rows = (SimpleNamespace(conversation_id="conv"),)
    key = frozenset({"conv"})
    try:
        with _actual_calls() as calls:
            started = time.monotonic()
            for _ in range(6):
                # Model actual active ticks 0.3s apart without sleeping six
                # times: successful native reads remain within the existing 2s TTL.
                if key in agent._console_subagent_counts_read:
                    agent._console_subagent_counts_read[key]["at"] = (
                        time.monotonic() - 0.3
                    )
                agent._console_subagent_counts_for_rows(bridge, rows)
                await _finish(tasks)
            elapsed = time.monotonic() - started
        record_property("active_count_receipt", {**calls, "seconds": elapsed})
        assert (
            calls["counts"] == 1
        ), "active unchanged badge still expires at 0.2 seconds"
        assert agent._console_subagent_counts_for_rows(bridge, rows) == {"conv": 1}
        with _actual_calls() as live:
            assert bridge.subagent_counts(["conv"]) == {"conv": 1}
        assert live["counts"] == 1, "live query reused display data"
        assert not worker_leases(database)
    finally:
        await _finish(tasks)
        database.close()


@pytest.mark.asyncio
async def test_count_receiver_change_invalidates_before_display_ttl(tmp_path):
    first = AgentRunsDB(tmp_path / "first-counts.db", "first")
    second = AgentRunsDB(tmp_path / "second-counts.db", "second")
    primary = first.create_run(conversation_id="conv", agent_kind="primary")
    first.create_run(
        conversation_id="conv", agent_kind="subagent", parent_run_id=primary
    )
    bridge = ConsoleAgentBridge(agent_runs_db=first, store=None, provider_gateway=None)
    agent, tasks = _agent(bridge)
    rows = (SimpleNamespace(conversation_id="conv"),)
    try:
        agent._console_subagent_counts_for_rows(bridge, rows)
        await _finish(tasks)
        assert agent._console_subagent_counts_for_rows(bridge, rows) == {"conv": 1}
        bridge._db = second
        assert (
            agent._console_subagent_counts_for_rows(bridge, rows) == {}
        ), "display reused old receiver under a new exact DB owner"
        await _finish(tasks)
        assert agent._console_subagent_counts_for_rows(bridge, rows) == {}
    finally:
        await _finish(tasks)
        first.close()
        second.close()


@pytest.mark.asyncio
async def test_character_display_expiry_and_live_scope_remain_fresh(tmp_path):
    database = CharactersRAGDB(tmp_path / "expiry.db", "expiry")
    controller, screen, *_ = _character(database)
    try:
        controller.state = replace(
            controller.state, scope_fingerprint=await controller._fingerprint()
        )
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
        controller.state = replace(
            controller.state, scope_fingerprint=await controller._fingerprint()
        )
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
        controller.state = replace(
            controller.state, scope_fingerprint=await controller._fingerprint()
        )
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
        assert (
            controller.state is prior
        ), "retired display read published after owner changed"
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
        controller.state = replace(
            controller.state, scope_fingerprint=await controller._fingerprint()
        )
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
        controller.state = replace(
            controller.state, scope_fingerprint=await controller._fingerprint()
        )
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
        controller.state = replace(
            controller.state, scope_fingerprint=await controller._fingerprint()
        )
        with database.transaction() as cursor:
            cursor.execute(
                "UPDATE character_conversation_search_revision SET data_revision = ? WHERE singleton_id = 1",
                ("invalid",),
            )
        with _actual_calls() as failed:
            await _display(controller, screen)
            assert failed["pairs"] == 2  # Fresh comparison, then actual refresh.
            assert controller._presentation_scope_key is None and controller.state.error
            await _display(controller, screen)
        # The failed refresh cleared its fingerprint: the next attempt goes
        # straight to another real refresh, rather than memoizing the failure.
        assert failed["pairs"] == 3
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
@pytest.mark.parametrize("mutation", ["receiver", "child", "run", "authority"])
async def test_pending_counts_capture_receiver_and_refuse_changed_owner(
    tmp_path, mutation
):
    first = AgentRunsDB(tmp_path / "pending-counts.db", "counts")
    second = AgentRunsDB(tmp_path / "new-counts.db", "counts-new")
    primary = first.create_run(conversation_id="conv", agent_kind="primary")
    first.create_run(
        conversation_id="conv", agent_kind="subagent", parent_run_id=primary
    )
    bridge = ConsoleAgentBridge(agent_runs_db=first, store=None, provider_gateway=None)
    agent, tasks = _agent(bridge)
    rows = (SimpleNamespace(conversation_id="conv"),)
    entered, release = threading.Event(), threading.Event()
    receivers = []
    blocked = False

    def barrier(frame, name):
        nonlocal blocked
        if name == "counts":
            receivers.append(frame.f_locals["self"])
            if not blocked:
                blocked = True
                entered.set()
                assert release.wait(10)

    try:
        with _actual_calls(barrier=barrier) as calls:
            agent._console_subagent_counts_for_rows(bridge, rows)
            state = agent._console_subagent_counts_read[frozenset({"conv"})]
            assert await asyncio.to_thread(entered.wait, 10)
            if mutation == "receiver":
                bridge._db = second
            elif mutation == "child":
                from tldw_chatbook.Chat.console_agent_bridge import (
                    AgentLiveSnapshot,
                    SubAgentSummary,
                )

                bridge._publish_live(
                    "conv",
                    "turn",
                    AgentLiveSnapshot(
                        status="running",
                        subagents=(SubAgentSummary("new", "done", "new"),),
                    ),
                    primary=True,
                )
            elif mutation == "run":
                bridge._live_primary_runs["conv"] = "new-run"
            else:
                agent.app_instance.chachanotes_db = object()
            release.set()
            await _finish(tasks)
            if mutation == "run":
                # The stock query counts conversation subagent rows, so a
                # primary-only run change preserves this exact count input.
                assert state["values"] == {"conv": 1} and state["at"] > 0
                assert agent._console_subagent_counts_for_rows(bridge, rows) == {
                    "conv": 1
                }
                await _finish(tasks)
        assert calls["counts"] == 1
        assert receivers == [
            first
        ], "pending query redirected to a replacement receiver"
        if mutation != "run":
            assert state["values"] == {} and state["at"] == 0
        assert not worker_leases(first) and not worker_leases(second)
    finally:
        release.set()
        await _finish(tasks)
        first.close()
        second.close()


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
        assert (
            controller.state is prior
        ), "display-owned refresh published for a replaced ambient owner"
        assert controller._presentation_scope_key is None
        assert not worker_leases(database)
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_cancelled_actual_count_read_drains_on_its_receiver_and_retries(tmp_path):
    database = AgentRunsDB(tmp_path / "cancel-native-counts.db", "cancel-counts")
    primary = database.create_run(conversation_id="conv", agent_kind="primary")
    database.create_run(
        conversation_id="conv", agent_kind="subagent", parent_run_id=primary
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=database, store=None, provider_gateway=None
    )
    agent, tasks = _agent(bridge)
    rows = (SimpleNamespace(conversation_id="conv"),)
    entered, release = threading.Event(), threading.Event()
    blocked = False

    def barrier(_frame, name):
        nonlocal blocked
        if name == "counts_return" and not blocked:
            blocked = True
            entered.set()
            assert release.wait(10)

    try:
        with _actual_calls(barrier=barrier):
            agent._console_subagent_counts_for_rows(bridge, rows)
            state = agent._console_subagent_counts_read[frozenset({"conv"})]
            assert await asyncio.to_thread(entered.wait, 10)
            tasks[0].cancel()
            await asyncio.sleep(0)
            assert not tasks[0].done()
            assert state["values"] == {} and state["at"] == 0
            assert worker_leases(
                database
            ), "cancelled awaiter prematurely closed native worker handle"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            for _ in range(1000):
                if not worker_leases(database):
                    break
                await asyncio.sleep(0.01)
            assert not worker_leases(database)
        agent._console_subagent_counts_for_rows(bridge, rows)
        await _finish(tasks[1:])
        assert agent._console_subagent_counts_for_rows(bridge, rows) == {"conv": 1}
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_custom_file_bridge_preserves_its_count_callback_contract(tmp_path):
    class CustomBridge(ConsoleAgentBridge):
        def subagent_counts(self, conversation_ids):
            counts = super().subagent_counts(conversation_ids)
            return {key: value + 10 for key, value in counts.items()}

    database = AgentRunsDB(tmp_path / "custom-counts.db", "custom-counts")
    primary = database.create_run(conversation_id="conv", agent_kind="primary")
    database.create_run(
        conversation_id="conv", agent_kind="subagent", parent_run_id=primary
    )
    bridge = CustomBridge(agent_runs_db=database, store=None, provider_gateway=None)
    agent, tasks = _agent(bridge)
    rows = (SimpleNamespace(conversation_id="conv"),)
    try:
        with _actual_calls() as calls:
            agent._console_subagent_counts_for_rows(bridge, rows)
            await _finish(tasks)
        assert calls["counts"] == 1 and not worker_leases(database)
        assert agent._console_subagent_counts_for_rows(bridge, rows) == {
            "conv": 11
        }, "qualified DB type bypassed custom bridge semantics"
        assert database.count_subagents_by_conversation(["conv"]) == {"conv": 1}
    finally:
        await _finish(tasks)
        database.close()
