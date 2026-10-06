"""Painted saved-chat pending counts and explicit durable report inspection."""

import pytest
from textual.widgets import SelectionList

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_console_agent_progress import _modal_type
from Tests.UI.test_console_fleet_targets import _painted_text
from Tests.UI.test_console_workspace_tree import _TreeHarness
from tldw_chatbook.Agents.fleet_messages import (
    MessageError,
    MessageIdentity,
    MessageStore,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Console.console_workspace_tree import ConsoleWorkspaceTree
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
)
from tldw_chatbook.Workspaces.workspace_tree_state import build_workspace_tree_state


@private_profile_test
async def test_reopened_saved_progress_paints_count_and_explicit_read_discard(
    tmp_path, request, monkeypatch
):
    import asyncio

    db = CharactersRAGDB(tmp_path / "chat.sqlite", "progress-ui")
    runs = AgentRunsDB(tmp_path / "runs.sqlite", client_id="progress-ui")
    try:
        native = ConsoleChatStore(persistence=ChatPersistenceService(db))
        temporary = native.create_session(
            ephemeral=True,
            settings=ConsoleSessionSettings(provider="openai", model="test"),
        )
        messages = MessageStore()
        native.register_progress_message_store(messages)
        inbox = messages.open_inbox(native.progress_owner_id(temporary.id))
        message_id = inbox.sender(
            MessageIdentity("child", "child-run", "parent", "chain", "Researcher")
        ).send("PRIVATE-SAVED-PROGRESS-BODY")
        conversation_id = native.promote_ephemeral_session(temporary.id)
        native.close_session(temporary.id)
        restored = native.restore_persisted_session(
            title="Saved chat",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=[],
        )
        bridge = ConsoleAgentBridge(
            agent_runs_db=runs, store=native, provider_gateway=object()
        )
        current = bridge.message_store.get_inbox(native.progress_owner_id(restored.id))
        row = ConsoleConversationBrowserInputRow(
            row_key=conversation_id,
            conversation_id=conversation_id,
            native_session_id=restored.id,
            title="Saved chat",
            scope_type="workspace",
            workspace_id="named",
            workspace_label="Named",
        )

        def projection():
            return build_workspace_tree_state(
                workspaces=[("named", "Named")],
                rows=[row],
                progress_counts=bridge.progress_counts(),
            )

        tree = ConsoleWorkspaceTree()
        monkeypatch.setattr(
            _TreeHarness, "CSS_PATH", [str(path) for path in APP_STYLESHEETS]
        )
        host = _TreeHarness(tree)
        async with host.run_test(size=(100, 40)) as pilot:
            tree.sync_projection(projection(), expanded_workspace_ids={"named"})
            await pilot.pause()
            assert "Progress: 1" in _painted_text(host, tree)
            assert "PRIVATE-SAVED" not in _painted_text(host, tree)
            assert "PRIVATE-SAVED" not in repr(projection())
            modal = _modal_type()(
                conversation_id=native.progress_owner_id(restored.id),
                load=current.snapshot,
                discard=current.discard,
            )
            await host.push_screen(modal)
            await pilot.pause()
            listing = modal.query_one(SelectionList)
            assert listing.option_count == 1
            listing.select(message_id)
            listing.highlighted = 0
            await pilot.pause()
            assert "PRIVATE-SAVED-PROGRESS-BODY" in _painted_text(
                host, modal.query_one("#agent-progress-body")
            )
            await pilot.click("#agent-progress-discard")
            await pilot.pause()
            await asyncio.wait_for(host.workers.wait_for_complete(), 5)
            assert "Discarded 1" in str(
                modal.query_one("#agent-progress-status").renderable
            )
            assert current.snapshot() == ()
            assert (
                db.get_connection()
                .execute("SELECT count(*) FROM fleet_progress_messages")
                .fetchone()[0]
                == 0
            )
            await pilot.click("#agent-progress-close")
            tree.sync_projection(projection(), expanded_workspace_ids={"named"})
            await pilot.pause()
            assert "Progress:" not in _painted_text(host, tree)
    finally:
        runs.close()
        db.close()


@private_profile_test
async def test_explicit_progress_retry_loads_deferred_saved_reports_after_capacity_frees(
    tmp_path, request, monkeypatch
):
    from types import SimpleNamespace

    from tldw_chatbook.DB.fleet_progress_repository import FleetProgressRepository
    from tldw_chatbook.UI.Console_Modules.agent import ConsoleAgentController

    db = CharactersRAGDB(tmp_path / "chat.sqlite", "progress-ui")
    runs = AgentRunsDB(tmp_path / "runs.sqlite", client_id="progress-ui")
    try:
        persistence = ChatPersistenceService(db)
        conversation_id = persistence.create_conversation(conversation_title="Saved")
        history = MessageStore()
        saved = history.open_inbox(
            "historic",
            repository=FleetProgressRepository(db),
            saved_conversation_id=conversation_id,
        )
        saved.sender(
            MessageIdentity("child", "run", "parent", None, "Researcher")
        ).send("PRIVATE-DEFERRED-REPORT")
        history.close()
        native = ConsoleChatStore(persistence=persistence)
        session = native.restore_persisted_session(
            title="Saved",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=[],
        )
        native.switch_session(session.id)
        runtime = MessageStore()
        for queue in range(8):
            blocker = runtime.open_inbox(f"blocker-{queue}")
            for child in range(4):
                sender = blocker.sender(
                    MessageIdentity(
                        f"h-{queue}-{child}",
                        f"r-{queue}-{child}",
                        "parent",
                        None,
                        "Other",
                    )
                )
                for report in range(8):
                    sender.send(f"memory report {report}")
        bridge = ConsoleAgentBridge(
            agent_runs_db=runs,
            store=native,
            provider_gateway=object(),
            message_store=runtime,
        )
        assert bridge.progress_counts() == {session.id: 1}
        loads = []
        original_load = FleetProgressRepository.load

        def counted_load(repository, target):
            assert not native._progress_identity_lock._is_owned()
            loads.append(target)
            return original_load(repository, target)

        monkeypatch.setattr(FleetProgressRepository, "load", counted_load)
        monkeypatch.setattr(
            _TreeHarness, "CSS_PATH", [str(path) for path in APP_STYLESHEETS]
        )
        host = _TreeHarness(ConsoleWorkspaceTree())
        async with host.run_test(size=(100, 40)) as pilot:
            controller = ConsoleAgentController.__new__(ConsoleAgentController)
            controller._screen = host.screen
            controller._chat_controller_accessor = lambda: SimpleNamespace(store=native)
            monkeypatch.setattr(
                controller, "_ensure_console_agent_bridge", lambda: bridge
            )
            controller.open_fleet_progress()
            await pilot.pause(0.8)
            modal = host.screen
            assert (
                "capacity"
                in str(modal.query_one("#agent-progress-count").renderable).lower()
            )
            assert "No queued progress" not in _painted_text(host, modal)
            assert modal.query_one(SelectionList).option_count == 0
            assert loads == [conversation_id]
            await pilot.pause(0.7)
            assert loads == [conversation_id]  # Timer observes memory, never SQLite.
            await pilot.click("#agent-progress-close")
            runtime.close_inbox("blocker-0")
            controller.open_fleet_progress()
            await pilot.pause(0.8)
            modal = host.screen
            assert modal.query_one(SelectionList).option_count == 1
            assert "PRIVATE-DEFERRED-REPORT" in _painted_text(
                host, modal.query_one("#agent-progress-body")
            )
            assert loads == [conversation_id, conversation_id]
            assert (
                len(runtime.get_inbox(native.progress_owner_id(session.id)).snapshot())
                == 1
            )
    finally:
        runs.close()
        db.close()


@pytest.mark.parametrize(
    "close_while_waiting",
    [
        False,
        True,
        "native_session",
        "cancelled_close",
        "dispose",
        "cancelled_dispose",
        "replacement",
        "native_rebind",
        "replacement_dispose",
        "first_fleet_dispose",
        "saved_prepare_dispose",
        "bind_sender_dispose",
        "live_sender_close",
        "saved_hydration_dispose",
        "cancelled_hydration_dispose",
        "physical_cancelled_hydration_dispose",
        "restore_rollback_dispose",
    ],
)
@private_profile_test
async def test_durable_discard_contention_keeps_mounted_reads_and_heartbeat_responsive(
    tmp_path, request, monkeypatch, close_while_waiting
):
    import asyncio
    import sqlite3
    import threading
    import time
    from types import SimpleNamespace

    from textual.widgets import Button

    from tldw_chatbook.DB.fleet_progress_repository import FleetProgressRepository
    from tldw_chatbook.UI.Console_Modules.agent import ConsoleAgentController

    db = CharactersRAGDB(tmp_path / "chat.sqlite", "progress-ui")
    runs = AgentRunsDB(tmp_path / "runs.sqlite", client_id="progress-ui")
    entered = threading.Event()
    ready = threading.Event()
    released = threading.Event()
    release_requested = threading.Event()
    removals = []
    writer = None
    hydration_modes = {
        "saved_hydration_dispose",
        "cancelled_hydration_dispose",
        "physical_cancelled_hydration_dispose",
    }
    try:
        native = ConsoleChatStore(persistence=ChatPersistenceService(db))
        saved_id = native.persistence.create_conversation(conversation_title="Saved")
        session = native.restore_persisted_session(
            title="Saved",
            workspace_id=None,
            persisted_conversation_id=saved_id,
            all_nodes=[],
        )
        native.switch_session(session.id)
        bridge = ConsoleAgentBridge(
            agent_runs_db=runs, store=native, provider_gateway=object()
        )
        messages = bridge.message_store
        owner = native.progress_owner_id(session.id)
        inbox = messages.get_inbox(owner)
        sender = inbox.sender(
            MessageIdentity("child", "run", "parent", None, "Researcher")
        )
        reader = inbox.reader("primary", chain_id=None, automatic=False)
        first = sender.send("PRIVATE-CONTENDED-REPORT")
        second = sender.send("Retain this second report")
        runtime = None
        native_controller = None
        if isinstance(close_while_waiting, str):
            from Tests.Chat.test_console_chat_controller import StreamingGateway
            from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
            from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

            native_controller = ConsoleChatController(
                store=native,
                provider_gateway=StreamingGateway(),
                agent_bridge=bridge,
            )
            runtime = ConsoleRuntime(app=None)
            runtime.set_chat_store(native)
            runtime.set_agent_bridge(bridge)
            runtime.set_chat_controller(native_controller)
        target_session = None
        target_owner = None
        fleet = None
        child_cancel = threading.Event()
        if close_while_waiting in {
            "first_fleet_dispose",
            "saved_prepare_dispose",
            *hydration_modes,
            "restore_rollback_dispose",
        }:
            if close_while_waiting == "first_fleet_dispose":
                target_session = native.create_session(ephemeral=True)
            else:
                target_saved = native.persistence.create_conversation(
                    conversation_title="Other saved"
                )
                if close_while_waiting not in hydration_modes:
                    target_session = native.restore_persisted_session(
                        title="Other saved",
                        workspace_id=None,
                        persisted_conversation_id=target_saved,
                        all_nodes=[],
                    )
            if target_session is not None:
                target_owner = native.progress_owner_id(target_session.id)
                if close_while_waiting != "restore_rollback_dispose":
                    messages.close_inbox(target_owner)
            native.switch_session(session.id)
        if close_while_waiting in {"bind_sender_dispose", "live_sender_close"}:
            from tldw_chatbook.Agents.agent_service import AgentService

            fleet = bridge._conversation_fleet_coordinator(
                saved_id, progress_owner_id=owner
            )
            child = fleet.reserve(task="Live child", agent="child")
            fleet.attach_run(child.handle_id, "live-child-run")
            if close_while_waiting == "live_sender_close":
                child_sender = fleet.bind_progress_sender(
                    child.handle_id, parent_run_id="primary", chain_id=None
                )
                assert child_sender is not None
                service = AgentService.__new__(AgentService)
                service._fleet = fleet
                service._fleet_cancels = {child.handle_id: child_cancel}
                service._revoke_approvals = None
                bridge._fleet_services[saved_id] = service
        observations = []
        prepared_finished = threading.Event()
        if close_while_waiting == "physical_cancelled_hydration_dispose":
            receipts = []
            original_submit = native._stream_persistence_executor.submit
            original_wrap = asyncio.wrap_future
            original_prepare = native.prepare_progress_inbox

            def observed_submit(callback, *args, **kwargs):
                receipt = original_submit(callback, *args, **kwargs)
                receipts.append(receipt)
                return receipt

            def observed_wrap(receipt, **kwargs):
                observation = original_wrap(receipt, **kwargs)
                if any(receipt is expected for expected in receipts):
                    observations.append(observation)
                return observation

            def observed_prepare(target, **kwargs):
                try:
                    return original_prepare(target, **kwargs)
                finally:
                    if target != session.id:
                        prepared_finished.set()

            monkeypatch.setattr(
                native._stream_persistence_executor, "submit", observed_submit
            )
            monkeypatch.setattr(asyncio, "wrap_future", observed_wrap)
            monkeypatch.setattr(native, "prepare_progress_inbox", observed_prepare)
        opening = threading.Event()
        original_open = messages.open_inbox

        def observed_open(conversation, **kwargs):
            if conversation == target_owner or (
                close_while_waiting in hydration_modes and conversation != owner
            ):
                opening.set()
            return original_open(conversation, **kwargs)

        monkeypatch.setattr(messages, "open_inbox", observed_open)
        original_sender = inbox.sender

        def observed_sender(identity):
            if identity.run_id == "live-child-run":
                opening.set()
            return original_sender(identity)

        monkeypatch.setattr(inbox, "sender", observed_sender)
        rebind_entered = threading.Event()
        original_registration = native.register_progress_message_store

        def observed_registration(message_store, **kwargs):
            if message_store is not messages:
                rebind_entered.set()
            return original_registration(message_store, **kwargs)

        monkeypatch.setattr(
            native, "register_progress_message_store", observed_registration
        )
        original_remove = FleetProgressRepository.remove

        def contested_remove(repository, conversation, ids):
            removals.append(threading.get_ident())
            repository.db.get_connection().execute("PRAGMA busy_timeout = 3000")
            entered.set()
            return original_remove(repository, conversation, ids)

        monkeypatch.setattr(FleetProgressRepository, "remove", contested_remove)

        def hold_external_writer():
            connection = sqlite3.connect(tmp_path / "chat.sqlite", isolation_level=None)
            try:
                connection.execute("BEGIN IMMEDIATE")
                ready.set()
                entered.wait(5)
                release_requested.wait(1.2)
                connection.rollback()
                released.set()
            finally:
                connection.close()

        writer = threading.Thread(target=hold_external_writer)
        writer.start()
        assert await asyncio.to_thread(ready.wait, 5)
        monkeypatch.setattr(
            _TreeHarness, "CSS_PATH", [str(path) for path in APP_STYLESHEETS]
        )
        host = _TreeHarness(ConsoleWorkspaceTree())
        async with host.run_test(size=(100, 40)) as pilot:
            controller = ConsoleAgentController.__new__(ConsoleAgentController)
            controller._screen = host.screen
            controller._chat_controller_accessor = lambda: SimpleNamespace(store=native)
            monkeypatch.setattr(
                controller, "_ensure_console_agent_bridge", lambda: bridge
            )
            controller.open_fleet_progress()
            await pilot.pause(0.4)
            modal = host.screen
            listing = modal.query_one(SelectionList)
            listing.select(first)
            await pilot.pause()
            observed = []

            def heartbeat():
                if entered.is_set() and not released.is_set():
                    started = time.monotonic()
                    counts = bridge.progress_counts()
                    native.progress_owner_id(session.id)
                    current = messages.get_inbox(owner)
                    snapshot = current.snapshot() if current is not None else ()
                    observed.append(
                        (
                            time.monotonic() - started,
                            counts,
                            tuple(m.message_id for m in snapshot),
                            not released.is_set(),
                        )
                    )

            timer = host.set_interval(0.05, heartbeat)
            try:
                await pilot.click("#agent-progress-discard")
                assert await asyncio.to_thread(entered.wait, 0.5)
                await pilot.pause(0.05)
                assert not released.is_set(), (
                    "Discard blocked the UI until SQLite released its writer"
                )
                modal.post_message(
                    Button.Pressed(modal.query_one("#agent-progress-discard", Button))
                )
                closing = None
                if close_while_waiting:
                    await pilot.click("#agent-progress-close")
                if runtime is not None:
                    revision = native_controller.lifecycle_impact(
                        session_id=session.id
                    ).revision
                    cancelled_close = close_while_waiting in {
                        "cancelled_close",
                        "cancelled_dispose",
                    }
                    replacement = None
                    if close_while_waiting == "restore_rollback_dispose":
                        native.rollback_restored_session(
                            target_session.id,
                            expected_session=target_session,
                            prior_active_session_id=session.id,
                        )
                        assert not released.is_set(), (
                            "Restore rollback blocked the UI behind SQL"
                        )
                        assert native.active_session_id == session.id
                        assert native.progress_owner_id(target_session.id) is None
                        runtime.begin_dispose()
                        closing = asyncio.create_task(
                            runtime.dispose(timeout_seconds=0.05)
                        )
                    elif close_while_waiting in hydration_modes:
                        from tldw_chatbook.Chat.console_conversation_hydration import (
                            hydrate_console_session,
                        )

                        hydrating = asyncio.create_task(
                            hydrate_console_session(
                                app=SimpleNamespace(chachanotes_db=db),
                                store=native,
                                conversation_id=target_saved,
                                tree={
                                    "conversation": {
                                        "id": target_saved,
                                        "title": "Other saved",
                                    }
                                },
                                settings=ConsoleSessionSettings(
                                    provider="openai", model="test"
                                ),
                            )
                        )
                        assert await asyncio.to_thread(opening.wait, 0.5)
                        assert not released.is_set(), (
                            "Saved hydration blocked the UI behind SQL"
                        )
                        if (
                            close_while_waiting
                            == "physical_cancelled_hydration_dispose"
                        ):
                            assert observations
                            observations[0].cancel()
                            await pilot.pause(0.05)
                            assert not hydrating.done()
                            assert not prepared_finished.is_set()
                        if close_while_waiting == "cancelled_hydration_dispose":
                            hydrating.cancel()
                            await pilot.pause(0.05)
                            assert not hydrating.done()
                        runtime.begin_dispose()
                        closing = asyncio.gather(
                            hydrating,
                            runtime.dispose(timeout_seconds=0.05),
                            return_exceptions=True,
                        )
                    elif close_while_waiting in {
                        "first_fleet_dispose",
                        "saved_prepare_dispose",
                        "bind_sender_dispose",
                    }:

                        def prepare_capability():
                            if close_while_waiting == "first_fleet_dispose":
                                return bridge._conversation_fleet_coordinator(
                                    target_session.id, progress_owner_id=target_owner
                                )
                            if close_while_waiting == "saved_prepare_dispose":
                                return native.prepare_progress_inbox(
                                    target_session.id, message_store=messages
                                )
                            return fleet.bind_progress_sender(
                                child.handle_id, parent_run_id="primary", chain_id=None
                            )

                        preparing = asyncio.create_task(
                            asyncio.to_thread(prepare_capability)
                        )
                        assert await asyncio.to_thread(opening.wait, 0.5)
                        runtime.begin_dispose()
                        if fleet is not None:
                            fleet.finish(child.handle_id, status="cancelled")
                            bridge._notify_fleet_activity(saved_id)
                        closing = asyncio.gather(
                            preparing,
                            runtime.dispose(timeout_seconds=0.05),
                            return_exceptions=True,
                        )
                    elif close_while_waiting == "native_rebind":
                        replacement = ConsoleAgentBridge(
                            agent_runs_db=runs, store=native, provider_gateway=object()
                        )
                        closing = asyncio.create_task(
                            asyncio.to_thread(lambda: replacement.message_store)
                        )
                    elif close_while_waiting in {"replacement", "replacement_dispose"}:
                        replacement = ConsoleAgentBridge(
                            agent_runs_db=runs, store=native, provider_gateway=object()
                        )
                        runtime.set_agent_bridge(replacement)
                        registration = asyncio.create_task(
                            asyncio.to_thread(lambda: replacement.message_store)
                        )
                        if close_while_waiting == "replacement_dispose":
                            assert await asyncio.to_thread(rebind_entered.wait, 0.5)
                            runtime.begin_dispose()
                            closing = asyncio.gather(
                                registration,
                                runtime.dispose(timeout_seconds=0.05),
                                *runtime._progress_cleanup_tasks,
                                return_exceptions=True,
                            )
                        else:
                            closing = asyncio.gather(
                                registration, *runtime._progress_cleanup_tasks
                            )
                    else:
                        closing = asyncio.create_task(
                            runtime.dispose(
                                timeout_seconds=0.05 if cancelled_close else 3
                            )
                            if close_while_waiting in {"dispose", "cancelled_dispose"}
                            else runtime.close_session(
                                session.id,
                                expected_revision=revision,
                                timeout_seconds=0.05
                                if cancelled_close
                                or close_while_waiting == "live_sender_close"
                                else 3,
                            )
                        )
                        if cancelled_close:
                            await asyncio.sleep(0.01)
                            closing.cancel()
                    await pilot.pause(0.1)
                    assert not released.is_set(), (
                        "Native session close blocked the UI behind durable discard"
                    )
                    if close_while_waiting in {
                        "native_session",
                        "cancelled_close",
                        "live_sender_close",
                    }:
                        assert native.progress_owner_id(session.id) is None
                    if close_while_waiting == "live_sender_close":
                        assert child_cancel.is_set()
                        fleet.finish(child.handle_id, status="cancelled")
                        bridge._notify_fleet_activity(saved_id)
                    if close_while_waiting == "cancelled_close":
                        assert bridge._progress_close_drains
                    if close_while_waiting == "cancelled_dispose":
                        assert runtime._progress_cleanup_tasks
                        assert not bridge._progress_shutdown_task.done()
                    assert bridge.progress_counts() == {}
                await pilot.pause(1.3)
                if closing is not None:
                    if close_while_waiting in {"cancelled_close", "cancelled_dispose"}:
                        with pytest.raises(asyncio.CancelledError):
                            await closing
                    else:
                        result = await closing
                        if close_while_waiting in {
                            "cancelled_hydration_dispose",
                            "physical_cancelled_hydration_dispose",
                        }:
                            assert isinstance(result[0], asyncio.CancelledError)
                            assert result[1] is None
                        if close_while_waiting == "replacement_dispose":
                            assert isinstance(result[0], MessageError)
                            assert result[0].code == "unavailable"
                            assert all(value is None for value in result[1:])
                            with pytest.raises(MessageError, match="unavailable"):
                                _ = replacement.message_store
                    for operation in (
                        lambda: sender.send("late report"),
                        reader.collect,
                    ):
                        with pytest.raises(MessageError, match="unavailable"):
                            await asyncio.to_thread(operation)
                    assert not bridge._progress_close_drains
                assert len([item for item in observed if item[3]]) >= 3
                assert max(item[0] for item in observed) < 0.25
                assert all(
                    (item[1] == {session.id: 2} and item[2] == (first, second))
                    or (item[1] == {} and item[2] == ())
                    if runtime is not None
                    else item[1] == {session.id: 2} and item[2] == (first, second)
                    for item in observed
                    if item[3]
                )
                assert removals == [removals[0]]
                assert removals[0] != threading.get_ident()
                if runtime is None:
                    assert bridge.progress_counts() == {session.id: 1}
                    assert [message.message_id for message in inbox.snapshot()] == [
                        second
                    ]
                elif close_while_waiting in {
                    "dispose",
                    "cancelled_dispose",
                    "replacement_dispose",
                    "first_fleet_dispose",
                    "saved_prepare_dispose",
                    "bind_sender_dispose",
                    *hydration_modes,
                    "restore_rollback_dispose",
                }:
                    assert runtime._disposed
                    assert messages.get_inbox(owner) is None
                    assert not runtime._progress_cleanup_tasks
                elif close_while_waiting in {"replacement", "native_rebind"}:
                    runtime.set_agent_bridge(replacement)
                    current = replacement.message_store.get_inbox(owner)
                    assert [message.message_id for message in current.snapshot()] == [
                        second
                    ]
                else:
                    assert native.sessions() == []
                    restored = native.restore_persisted_session(
                        title="Restored after close",
                        workspace_id=None,
                        persisted_conversation_id=saved_id,
                        all_nodes=[],
                    )
                    current = messages.get_inbox(native.progress_owner_id(restored.id))
                    assert [message.message_id for message in current.snapshot()] == [
                        second
                    ]
                assert [
                    row[0]
                    for row in db.get_connection().execute(
                        "SELECT message_id FROM fleet_progress_messages ORDER BY sequence"
                    )
                ] == [second]
                if not close_while_waiting:
                    assert modal.query_one(SelectionList).option_count == 1
                    assert "Discarded 1" in str(
                        modal.query_one("#agent-progress-status").renderable
                    )
                assert host._exception is None
            finally:
                timer.stop()
    finally:
        release_requested.set()
        if writer is not None:
            await asyncio.to_thread(writer.join, 5)
        runs.close()
        db.close()


@pytest.mark.parametrize("operation", ["prepare", "discard"])
@pytest.mark.parametrize("borrowed", [False, True])
@private_profile_test
async def test_progress_modal_workers_retire_only_new_db_caches(
    tmp_path, request, monkeypatch, operation, borrowed
):
    import asyncio
    import threading
    from time import monotonic

    from tldw_chatbook.DB.fleet_progress_repository import FleetProgressRepository

    db = CharactersRAGDB(tmp_path / "modal-cache.sqlite", "progress-ui")
    persistence = ChatPersistenceService(db)
    saved_id = persistence.create_conversation(conversation_title="Saved")
    repository = FleetProgressRepository(db)
    messages = MessageStore()
    inbox = messages.open_inbox(
        "exact-owner", repository=repository, saved_conversation_id=saved_id
    )
    message_id = inbox.sender(
        MessageIdentity("child", "child-run", "parent", None, "Researcher")
    ).send("PRIVATE-MODAL-CACHE-REPORT")
    main_connection = db.get_connection()
    observed = []
    finished = threading.Event()
    modal = _modal_type()(
        conversation_id="exact-owner",
        load=inbox.snapshot,
        discard=inbox.discard,
        prepare=(lambda: repository.load(saved_id)) if operation == "prepare" else None,
    )
    name = "_prepare_inbox" if operation == "prepare" else "_discard_reports"
    original = getattr(modal, name)

    def observe_worker(*args):
        existing = db.get_connection() if borrowed else None
        try:
            original(*args)
            observed.append((existing, getattr(db._local, "conn", None)))
        finally:
            # Retire the harness-owned borrowed handle, or the leaked RED handle.
            db.close_connection()
            finished.set()

    monkeypatch.setattr(modal, name, observe_worker)
    monkeypatch.setattr(_TreeHarness, "CSS_PATH", [str(p) for p in APP_STYLESHEETS])
    host = _TreeHarness(ConsoleWorkspaceTree())
    try:
        async with host.run_test(size=(100, 40)) as pilot:
            await host.push_screen(modal)
            if operation == "discard":
                await pilot.pause()
                modal.query_one(SelectionList).select(message_id)
                await pilot.pause()
                await pilot.click("#agent-progress-discard")
            assert await asyncio.to_thread(finished.wait, 2)
            assert len(observed) == 1
            existing, cached = observed[0]
            assert cached is existing, (
                "finite modal SQL worker retained its new DB cache"
            )
            assert main_connection.execute("SELECT 1").fetchone()[0] == 1
            assert [m.message_id for m in inbox.snapshot()] == (
                [message_id] if operation == "prepare" else []
            )
            assert db.get_connection().execute(
                "SELECT count(*) FROM fleet_progress_messages"
            ).fetchone()[0] == (1 if operation == "prepare" else 0)
            assert not modal._discarding
            await pilot.click("#agent-progress-close")
        messages.close()
        db.close()
        assert not db._maintenance_participant.connections
        participant = db._maintenance_participant
        participant.close_admission()
        try:
            assert participant.drain(monotonic() + 0.05)
        finally:
            participant.resume()
    finally:
        messages.close()
        db.close()
