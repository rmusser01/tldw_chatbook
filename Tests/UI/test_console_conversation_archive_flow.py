"""Mounted Console recovery controls with isolated, real conversation storage."""

import pytest
from textual.widgets import Button, Input

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_conversation_recovery_flow import wait_until
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.Console_Modules.archive import (
    archive_current_conversation,
    consume_conversation_resume,
)
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    ConsoleConversationResumeIntent,
    HandoffChannel,
)
from tldw_chatbook.Widgets.Console.console_workspace_switcher_modal import (
    WorkspaceArchiveReceiptModal,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (160, 44)])
async def test_console_archive_undo_then_resume_reuses_original_and_full_search(
    size, tmp_path
):
    from tldw_chatbook.config import save_setting_to_cli_config

    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app()
    from tldw_chatbook.Chat.chat_conversation_scope_service import (
        ChatConversationScopeService,
    )
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "console-archive.db", client_id="archive-ui")
    app.chachanotes_db = db
    service = app.local_chat_conversation_service = ChatConversationService(db)
    app.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=service, server_service=None
    )
    cid = service.create_conversation(title="Archive recovery example")
    app.chachanotes_db.add_message(
        {
            "conversation_id": cid,
            "sender": "user",
            "content": "Retained original transcript",
        }
    )
    unrelated = service.create_conversation(title="Unrelated saved discussion")
    db.add_message(
        {"conversation_id": unrelated, "sender": "user", "content": "Different topic"}
    )
    app.workspace_registry_service.create_workspace(
        workspace_id="named-workspace", name="Named workspace"
    )
    app.workspace_registry_service.set_active_workspace("named-workspace")
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    host = app
    async with host.run_test(size=size) as pilot:
        await wait_until(
            pilot,
            lambda: (
                getattr(app, "_initial_screen_pushed", False)
                and type(app.screen).__name__ != "Screen"
            ),
            lambda: (
                getattr(app, "_initial_screen_pushed", False),
                type(app.screen).__name__,
            ),
        )
        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await wait_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        console = host.screen
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_CONVERSATION_RESUME,
            ConsoleConversationResumeIntent(cid),
        )
        await consume_conversation_resume(console)
        from tldw_chatbook.Chat.console_chat_models import CONSOLE_GLOBAL_WORKSPACE_ID

        resumed = next(
            item
            for item in console._ensure_console_chat_store().sessions()
            if item.persisted_conversation_id == cid
        )
        assert resumed.workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID
        original = console._ensure_console_chat_store().active_session_id
        await archive_current_conversation(console)
        await wait_until(
            pilot,
            lambda: (
                isinstance(host.screen, WorkspaceArchiveReceiptModal)
                and bool(host.screen.query("#workspace-archive-undo"))
            ),
        )
        assert service.get_conversation_metadata(cid)["archived"] is True
        assert isinstance(host.screen, WorkspaceArchiveReceiptModal)
        from tldw_chatbook.Chat.conversation_archive_actions import (
            conversation_send_refusal,
        )

        assert "archived" in await conversation_send_refusal(app, cid)
        host.screen.query_one("#workspace-archive-undo", Button).press()
        await wait_until(
            pilot,
            lambda: (
                service.get_conversation_metadata(cid)["archived"] is False
                and host.screen is console
            ),
            lambda: (
                service.get_conversation_metadata(cid),
                type(host.screen).__name__,
            ),
        )
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_CONVERSATION_RESUME,
            ConsoleConversationResumeIntent(cid),
        )
        await consume_conversation_resume(console)
        assert console._ensure_console_chat_store().active_session_id == original
        assert (
            sum(
                s.persisted_conversation_id == cid
                for s in console._ensure_console_chat_store().sessions()
            )
            == 1
        )
        await console.action_open_console_session_switcher()
        await wait_until(
            pilot, lambda: bool(host.screen.query("#console-switcher-query"))
        )
        query = "Retained original transcript"
        assert (
            query.lower() not in service.get_conversation_metadata(cid)["title"].lower()
        )
        host.screen.query_one("#console-switcher-query", Input).value = query
        await wait_until(
            pilot, lambda: bool(host.screen.query("#console-switcher-full-search"))
        )
        host.save_screenshot(
            filename=f"console-search-{size[0]}x{size[1]}.svg", path=str(tmp_path)
        )
        host.screen.query_one("#console-switcher-full-search", Button).press()
        await wait_until(
            pilot,
            lambda: isinstance(host.screen, LibraryScreen),
            lambda: type(host.screen).__name__,
        )
        library = host.screen
        await wait_until(
            pilot,
            lambda: (
                library._conversations_state.requested_query == query
                and not library._conversations_state.loading
                and library._conversations_state.freshness == "fresh"
                and bool(library.query("#library-conversations-filter"))
                and library._conversations_state.reader_state.loaded_actions_eligible
            ),
            lambda: (
                library._conversations_state.requested_query,
                library._conversations_state.total,
                library._conversations_state.error,
                library._library_lookup_error,
                bool(library.query("#library-conversations-canvas")),
                library._conversations_state.reader_state,
            ),
        )
        assert library._conversation_recovery().scope == "all"
        assert library.query_one("#library-conversations-filter", Input).value == query
        assert library._conversations_state.total == 1
        assert [row["id"] for row in library._conversations_state.page_records] == [cid]
        await wait_until(
            pilot,
            lambda: library._conversations_state.reader_state.loaded_actions_eligible,
            lambda: library._conversations_state.reader_state,
        )
        reader = library._conversations_state.reader_state
        assert reader.loaded_id == cid
        assert "Retained original transcript" in reader.messages[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace_collision", [False, True])
async def test_restore_resume_then_send_retains_original_history_and_unrelated_draft(
    tmp_path, workspace_collision, monkeypatch
):
    from Tests.UI.test_console_native_chat_flow import (
        CapturingGateway,
        _configure_native_ready_console,
    )
    from tldw_chatbook.Chat.chat_conversation_scope_service import (
        ChatConversationScopeService,
    )
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules.archive import request_conversation_resume
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
    from tldw_chatbook.Widgets.Console.console_workspace_switcher_modal import (
        ConsoleWorkspaceRenameModal,
    )

    db = CharactersRAGDB(tmp_path / "resume-send.db", client_id="resume-send-ui")
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.chachanotes_db = db
    local = app.local_chat_conversation_service = ChatConversationService(db)
    app.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=local, server_service=None
    )
    gateway = CapturingGateway(chunks=("Continued original conversation.",))
    app.console_provider_gateway_factory = lambda: gateway
    registry = app.workspace_registry_service
    if workspace_collision:
        registry.create_workspace(workspace_id="old-workspace", name="Research")
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    # Seed a saved Console chat through its real persistence path, including
    # the durable Library policy that current Console sends validate.
    seed_store = ConsoleChatStore(
        persistence=ChatPersistenceService(db, workspace_registry=registry)
    )
    seed_session = seed_store.create_session(
        title="Original research",
        workspace_id="old-workspace" if workspace_collision else None,
    )
    seed_store.append_message(
        seed_session.id,
        role=ConsoleMessageRole.USER,
        content="Original question",
        persist=True,
    )
    original_answer = seed_store.append_message(
        seed_session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Original answer",
        persist=True,
    )
    cid = seed_session.persisted_conversation_id
    assistant_id = original_answer.persisted_message_id
    assert cid is not None and assistant_id is not None
    metadata = local.get_conversation_metadata(cid)
    db.update_conversation(
        cid, {"active_leaf_message_id": assistant_id}, metadata["version"]
    )
    local.set_conversations_archived(
        [cid],
        archived=True,
        expected_versions={cid: local.get_conversation_metadata(cid)["version"]},
    )
    if workspace_collision:
        registry.archive_workspace("old-workspace")
        registry.create_workspace(workspace_id="new-workspace", name="Research")
    navigation = []
    host = ConsoleHarness(app)
    try:
        async with host.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.4)
            console = host.screen
            store = console._ensure_console_chat_store()
            keeper = store.active_session_id
            console._console_composer_or_none().load_draft("Unrelated unfinished draft")
            await console._sync_native_console_chat_ui()
            assert store.session_draft(keeper) == "Unrelated unfinished draft"
            monkeypatch.setattr(app, "push_screen", host.push_screen)
            monkeypatch.setattr(app, "run_worker", host.run_worker)
            monkeypatch.setattr(
                app, "post_message", lambda message: navigation.append(message)
            )
            await request_conversation_resume(app, cid)
            await pilot.pause(0.2)
            dialog = host.screen
            assert isinstance(dialog, ConfirmationDialog)
            if workspace_collision:
                assert "whole workspace" in dialog.message
            dialog.query_one("#confirm-button", Button).press()
            if workspace_collision:
                for _ in range(80):
                    await pilot.pause(0.03)
                    if isinstance(
                        host.screen, ConsoleWorkspaceRenameModal
                    ) and host.screen.query("#console-workspace-rename-save"):
                        break
                assert isinstance(host.screen, ConsoleWorkspaceRenameModal)
                host.screen.query_one(
                    "#console-workspace-rename-input", Input
                ).value = "Recovered research"
                host.screen.query_one("#console-workspace-rename-save", Button).press()
            for _ in range(100):
                await pilot.pause(0.03)
                if any(isinstance(message, NavigateToScreen) for message in navigation):
                    break
            assert any(isinstance(message, NavigateToScreen) for message in navigation)
            assert local.get_conversation_metadata(cid)["archived"] is False
            await consume_conversation_resume(console)
            await pilot.pause(0.3)
            resumed = next(
                session
                for session in store.sessions()
                if session.persisted_conversation_id == cid
            )
            assert store.active_session_id == resumed.id
            assert store.session_draft(keeper) == "Unrelated unfinished draft"
            if workspace_collision:
                assert (
                    registry.get_workspace("old-workspace").name == "Recovered research"
                )
                assert registry.get_workspace("new-workspace").name == "Research"
                assert resumed.workspace_id == "old-workspace"
            # A restore by another writer can leave an old archived projection.
            # The synchronous admission gate must allow the durable send check.
            app._conversation_archive_states[cid] = True
            assert console._console_send_blocked_reason() == ""
            await console._submit_console_native_draft(
                "Continue this original conversation", resumed.id
            )
            assert app._conversation_archive_states[cid] is False
            assert gateway.sent_messages, (
                console._console_chat_controller.run_state_for(resumed.id),
                console._console_send_blocked_reason(),
            )
            sent_content = str(gateway.sent_messages[-1])
            assert (
                "Original question" in sent_content
                and "Original answer" in sent_content
            )
            assert "Continue this original conversation" in sent_content
            stored = local.get_library_conversation_messages(cid, message_limit=10)
            bodies = [message["text"] for message in stored["messages"]]
            assert bodies == [
                "Original question",
                "Original answer",
                "Continue this original conversation",
                "Continued original conversation.",
            ]
            assert store.session_draft(keeper) == "Unrelated unfinished draft"
    finally:
        db.close_connection()
