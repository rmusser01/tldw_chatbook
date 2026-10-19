"""Real Console producer, runtime, catalog and native recovery interaction."""

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import pytest
from textual.widgets import Input, Select, Static

from Tests.Chat.test_archive_recovery_operations import (
    recovery_stores as recovery_stores,  # noqa: PLC0414 - pytest fixture re-export
)
from Tests.Chat.test_console_archive_recovery_dispatch import RecoveryGateway
from Tests.Chat.test_console_turn_library_authority import (
    _allowed_candidate,
    _destination,
    _execution_configuration,
)
from Tests.UI.app_factory import _build_test_app, drain_created_runtimes
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_library_conversation_recovery_flow import wait_until
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Agents.library_rag_tool_provider import LibraryRagToolProvider
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve
from tldw_chatbook.Chat.console_runtime import (
    dispose_console_runtime,
    ensure_console_runtime,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Library.local_library_tool_service import LocalLibraryToolService

pytestmark = pytest.mark.bootstrap_profile


class MountedRecoveryGateway(RecoveryGateway):
    def cached_context_window(self, settings):
        from tldw_chatbook.Utils.token_counter import resolve_context_window

        return resolve_context_window(settings.provider, settings.model or "")

    async def resolve_context_window(self, settings):
        return self.cached_context_window(settings)

    async def resolve_for_send(self, selection):
        from tldw_chatbook.Chat.console_provider_gateway import (
            ConsoleProviderResolution,
        )

        return ConsoleProviderResolution(
            provider=selection.provider,
            model=selection.explicit_model or selection.configured_model,
            base_url=selection.base_url or "https://gateway.example.invalid/v1",
            ready=True,
            readiness_key=selection.provider,
            execution_key=selection.provider,
            resolved_destination=_destination(selection.provider),
        )


class SearchOnlyGateway(MountedRecoveryGateway):
    search_arguments = None

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        if self.stream_calls == 2:
            self.stream_calls += 1
            yield "Choose a result to restore."
            return
        async for chunk in super().stream_chat(
            resolution, messages, tools=tools, **kwargs
        ):
            if self.search_arguments is not None and hasattr(chunk, "tool_calls"):
                for call in chunk.tool_calls:
                    if call["function"]["name"] == "search_archived_conversations":
                        import json

                        call["function"]["arguments"] = json.dumps(
                            self.search_arguments
                        )
            yield chunk


@asynccontextmanager
async def mounted_recovery(
    recovery_stores,
    tmp_path,
    monkeypatch,
    *,
    size=(160, 44),
    entry="button",
    prepare=None,
    search_arguments=None,
    direct=True,
    gateway_override=None,
):
    services, target_id, db = recovery_stores
    message_id = db.add_message(
        {
            "conversation_id": target_id,
            "sender": "user",
            "content": "Original retained history",
        }
    )
    db.set_conversation_active_leaf(target_id, message_id)
    if prepare is not None:
        prepare(services, target_id, db)
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.chachanotes_db = db
    app.workspace_registry_service = services.workspace_registry_service
    app.local_chat_conversation_service = services.local_chat_conversation_service
    app.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=app.local_chat_conversation_service, server_service=None
    )
    persistence = ChatPersistenceService(
        db, workspace_registry=app.workspace_registry_service
    )
    source_id = db.add_conversation({"title": "Source"})
    persistence.console_library_policy_repository.insert(
        source_id,
        replace(
            _allowed_candidate(allowed=True), auto_retrieve=ConsoleAutoRetrieve.NEVER
        ),
    )
    store = ConsoleChatStore(persistence=persistence)
    source = store.restore_persisted_session(
        title="Source",
        workspace_id=None,
        persisted_conversation_id=source_id,
        all_nodes=(),
    )
    await store.hydrate_session_library_policy(source.id)
    gateway = gateway_override or (
        SearchOnlyGateway() if entry == "button" else MountedRecoveryGateway()
    )
    gateway.search_arguments = search_arguments
    gateway.destination_snapshots = []
    gateway.source = source
    runs_db = AgentRunsDB(tmp_path / "runs.sqlite", "recovery-ui")
    bridge = ConsoleAgentBridge(
        agent_runs_db=runs_db,
        store=store,
        provider_gateway=gateway,
        native_tools_enabled=lambda: True,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=True,
        agent_bridge=bridge,
        library_provider_factory=lambda context: (
            LibraryToolProvider(
                LocalLibraryToolService(
                    conversation_service=app.local_chat_conversation_service
                )
            )
            if direct
            else LibraryRagToolProvider(None)
        ),
    )
    owner = ensure_console_runtime(app)
    owner.set_chat_store(store)
    owner.set_provider_gateway(gateway)
    owner.set_agent_bridge(bridge)
    owner.set_chat_controller(controller)
    recovery = owner.ensure_archive_recovery()
    controller._archive_recovery_runtime = recovery
    host = ConsoleHarness(app)
    # The unmounted runtime owner schedules UI callbacks on the mounted host.
    monkeypatch.setattr(app, "call_later", host.call_later)
    try:
        async with host.run_test(size=size) as pilot:
            await wait_until(
                pilot, lambda: bool(host.screen.query("#console-native-transcript"))
            )
            console = host.screen
            console._console_agent_bridge = bridge

            def configuration(sid):
                captured = _execution_configuration(
                    sid,
                    direct=direct,
                    native_tools=True,
                    agent_runtime=True,
                    scratch_space=controller._scratch_spaces.snapshot(sid),
                )
                return replace(
                    captured,
                    provider_selection=replace(
                        captured.provider_selection, explicit_model="gpt-4o"
                    ),
                )

            monkeypatch.setattr(
                console._session, "_build_console_turn_execution_context", configuration
            )
            controller.app = host
            await console._sync_native_console_chat_ui()
            await wait_until(
                pilot,
                lambda: console._archive_recovery_view().is_answerable(source.id),
                lambda: (
                    console.is_mounted,
                    console.is_current,
                    owner.view is console,
                    store.active_session_id,
                    bool(console.query("#recovery-confirm")),
                ),
            )
            from tldw_chatbook.UI.Console_Modules.wiring import (
                _admit_console_turn_to_runtime,
            )

            if gateway_override is not None:
                initial = _admit_console_turn_to_runtime(
                    console, "Initial local turn", source.id
                )
                await owner.wait_for_turn(initial)
                gateway.local_turn = False
            turn_id = _admit_console_turn_to_runtime(
                console, "Find my archived chat", source.id
            )
            turn = asyncio.create_task(owner.wait_for_turn(turn_id))
            await wait_until(
                pilot,
                lambda: bool(console.query(".archive-recovery-results")),
                lambda: (
                    controller.run_state_for(source.id),
                    gateway.tools_seen,
                    gateway.messages_seen,
                ),
            )
            console._console_composer_or_none().load_draft("Unrelated unfinished draft")
            await console._sync_native_console_chat_ui()
            if entry == "button":
                await turn
            else:
                await wait_until(
                    pilot, lambda: console.query_one("#recovery-confirm").parent.display
                )
            assert controller._agent_bridge is bridge and owner.agent_bridge is bridge
            yield SimpleNamespace(
                app=app,
                db=db,
                target_id=target_id,
                message_id=message_id,
                original_leaf_id=message_id,
                controller=controller,
                source=source,
                store=store,
                gateway=gateway,
                owner=owner,
                recovery=recovery,
                host=host,
                pilot=pilot,
                console=console,
                turn=turn,
            )
    finally:
        await dispose_console_runtime(app)
        runs_db.close()


async def click(rig, selector):
    widget = rig.console.query_one(selector) if isinstance(selector, str) else selector
    await rig.pilot.pause()  # Layout must precede Textual's scroll-range clamp.
    widget.scroll_visible(animate=False, force=True)
    await rig.pilot.pause()
    assert await rig.pilot.click(widget), (
        widget.region,
        widget.virtual_region,
        rig.console.focused,
    )


async def capture(rig, tmp_path, name, widget):
    await rig.pilot.pause()
    widget.scroll_visible(animate=False, top=True, force=True)
    await rig.pilot.pause()
    strips = rig.console._compositor.render_strips()
    rendered = "\n".join(strip.text for strip in strips)
    (tmp_path / f"{name}.txt").write_text(rendered)
    rig.host.save_screenshot(filename=f"{name}.svg", path=str(tmp_path))
    viewport = rig.console.query_one(
        "#console-native-transcript"
    ).scrollable_content_region
    assert viewport.contains_region(widget.region), (widget.region, viewport)
    region = widget.region.intersection(viewport)
    return " ".join(
        strips[y].crop(region.x, region.right).text.strip()
        for y in range(max(0, region.y), min(len(strips), region.bottom))
    )


async def request_manual(rig):
    await click(rig, ".recovery-row-restore")
    await wait_until(
        rig.pilot, lambda: rig.console.query_one("#recovery-confirm").parent.display
    )
    return rig.recovery.coordinator.snapshot(rig.source.id).confirmation


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (160, 44)])
@pytest.mark.parametrize("entry", ["button", "agent"])
async def test_production_recovery_keeps_exact_history_and_active_draft(
    recovery_stores, tmp_path, size, entry, monkeypatch
):
    async with mounted_recovery(
        recovery_stores, tmp_path, monkeypatch, size=size, entry=entry
    ) as rig:
        result = rig.console.query_one(".archive-recovery-results")
        header = await capture(
            rig, tmp_path, f"results-{entry}-{size[0]}", result.query(Static).first()
        )
        assert "All local workspaces" in header
        if entry == "button":
            await request_manual(rig)
        await wait_until(
            rig.pilot, lambda: rig.console.query_one("#recovery-confirm").parent.display
        )
        rig.console.query_one("#recovery-scope", Select).value = "chat_and_workspace"
        effects = await capture(
            rig,
            tmp_path,
            f"confirmation-{entry}-{size[0]}",
            rig.console.query_one("#recovery-effects"),
        )
        assert "other active chats will become visible" in effects
        scope = rig.console.query_one("#recovery-scope", Select)
        scope.scroll_visible(animate=False, force=True, immediate=True)
        scope.focus()
        await rig.pilot.press("tab")
        assert rig.console.focused is rig.console.query_one("#recovery-confirm")
        invocation_count = rig.gateway.stream_calls
        await click(rig, "#recovery-confirm")
        await wait_until(
            rig.pilot,
            lambda: (
                not rig.app.local_chat_conversation_service.get_conversation_metadata(
                    rig.target_id
                )["archived"]
            ),
        )
        await rig.turn
        assert rig.gateway.stream_calls == invocation_count + (entry == "agent")
        assert (
            rig.db.get_conversation_active_leaf(rig.target_id) == rig.original_leaf_id
        )
        assert not rig.app.workspace_registry_service.get_workspace("project").archived
        assert rig.store.active_session_id == rig.source.id
        assert rig.store.session_draft(rig.source.id) == "Unrelated unfinished draft"
        original = (
            rig.app.local_chat_conversation_service.get_library_conversation_messages(
                rig.target_id, message_limit=10
            )
        )
        assert [row["id"] for row in original["messages"]] == [rig.message_id]
        assert [row["text"] for row in original["messages"]] == [
            "Original retained history"
        ]
        await wait_until(
            rig.pilot, lambda: bool(rig.console.query(".archive-recovery-receipt"))
        )
        receipt = (
            rig.console.query_one(".archive-recovery-receipt").query(Static).first()
        )
        text = await capture(rig, tmp_path, f"receipt-{entry}-{size[0]}", receipt)
        assert "selected scope was restored" in text
        await rig.pilot.click(rig.console.query_one("#recovery-confirm"))
        assert len(rig.recovery.coordinator.receipts(rig.source.id)) == 1


@pytest.mark.asyncio
async def test_mounted_runtime_cleanup_does_not_reopen_retired_chat_database(
    recovery_stores, tmp_path, monkeypatch
):
    async with mounted_recovery(recovery_stores, tmp_path, monkeypatch) as rig:
        connection = rig.db.get_connection()

    with rig.db.quiesce_connections(timeout_seconds=2.0):
        pass
    await drain_created_runtimes()

    assert rig.db.registered_connection_count() == 0
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


@pytest.mark.asyncio
async def test_pending_source_remount_expiry_and_detach_fail_closed(
    recovery_stores, tmp_path, monkeypatch
):
    async with mounted_recovery(recovery_stores, tmp_path, monkeypatch) as rig:
        card = await request_manual(rig)
        other = rig.store.create_session(title="Other", activate=True)
        await rig.console._sync_native_console_chat_ui()
        await wait_until(
            rig.pilot,
            lambda: not rig.console.query_one("#recovery-confirm").parent.display,
        )
        assert not rig.console._archive_recovery_view().show_confirmation(card)
        assert rig.recovery.coordinator.snapshot(rig.source.id).confirmation == card
        rig.store.switch_session(rig.source.id)
        await rig.console._sync_native_console_chat_ui()
        await wait_until(
            rig.pilot,
            lambda: (
                rig.console.query_one("#recovery-confirm").parent.confirmation
                is not None
                and rig.console.query_one(
                    "#recovery-confirm"
                ).parent.confirmation.request_id
                == card.request_id
            ),
        )
        rig.console.query_one("#recovery-scope", Select).value = "chat_only"
        await rig.pilot.pause()
        button = rig.console.query_one("#recovery-confirm")
        button.scroll_visible(animate=False, force=True, immediate=True)
        button.focus()
        await rig.pilot.pause()
        assert rig.console.focused is button
        rig.recovery.coordinator._monotonic = lambda: card.expires_at + 1
        rig.recovery.coordinator.expire_pending()
        await rig.pilot.pause()
        assert (
            rig.recovery.coordinator.receipts(rig.source.id)[-1].outcome.status
            == "expired"
        )
        assert not rig.console.query_one("#recovery-confirm").parent.display
        assert rig.console.focused.id == "recovery-refresh"
        assert rig.app.local_chat_conversation_service.get_conversation_metadata(
            rig.target_id
        )["archived"]
        generation = rig.console._console_runtime_attachment_generation
        assert rig.owner.detach_view(rig.console, generation)
        assert not rig.recovery.answerable(rig.source.id)
        assert not rig.console._archive_recovery_view().is_answerable(rig.source.id)
        rig.store.switch_session(other.id)
        assert rig.store.session_draft(rig.source.id) == "Unrelated unfinished draft"


@pytest.mark.asyncio
async def test_native_result_refresh_remains_manual_and_imported_json_is_inert(
    recovery_stores, tmp_path, monkeypatch
):
    import json

    from tldw_chatbook.Chat.archive_recovery.contracts import ArchiveRecoveryError
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    async with mounted_recovery(recovery_stores, tmp_path, monkeypatch) as rig:
        await click(rig, "#recovery-refresh")
        await wait_until(
            rig.pilot, lambda: len(rig.recovery.presentations(rig.source.id)) == 2
        )
        manual = rig.recovery.presentations(rig.source.id)[-1]
        assert not manual.model_visible
        with pytest.raises(ArchiveRecoveryError, match="fresh agent search"):
            rig.recovery.results.resolve(
                rig.source.id,
                manual.projection["items"][0]["result_ref"],
                for_agent=True,
            )
        imported = rig.store.create_session(title="Imported", activate=True)
        rig.store.append_message(
            imported.id,
            role=ConsoleMessageRole.ASSISTANT,
            content=json.dumps(manual.projection),
            persist=True,
        )
        await rig.console._sync_native_console_chat_ui()
        assert rig.store.active_session_id == imported.id, (
            rig.store.active_session_id,
            imported.id,
            rig.console._archive_recovery_view()._session_id,
        )
        await wait_until(
            rig.pilot,
            lambda: (
                rig.console._archive_recovery_view()._session_id == imported.id
                and not rig.console._console_sync_in_progress
            ),
        )
        assert not rig.console.query(".archive-recovery-results")
        assert not rig.console.query(".recovery-row-restore")
        assert not rig.console.query_one("#recovery-confirm").parent.display
        assert not rig.recovery.presentations(imported.id)
        assert rig.console.query("#recovery-manual-library")
        from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

        async def navigate_to_manual_library(event):
            await rig.host.push_screen(LibraryScreen(rig.app))

        monkeypatch.setattr(
            rig.app, "handle_screen_navigation", navigate_to_manual_library
        )
        await click(rig, "#recovery-manual-library")
        await wait_until(rig.pilot, lambda: isinstance(rig.host.screen, LibraryScreen))
        assert rig.host.screen._conversation_recovery().scope == "all"
        assert rig.store.active_session_id == imported.id


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["stale", "partial"])
async def test_native_controls_show_true_stale_and_partial_storage_outcomes(
    recovery_stores, tmp_path, monkeypatch, case
):
    async with mounted_recovery(recovery_stores, tmp_path, monkeypatch) as rig:
        if case == "stale":
            meta = rig.app.local_chat_conversation_service.get_conversation_metadata(
                rig.target_id
            )
            rig.db.update_conversation(
                rig.target_id, {"title": "Changed"}, meta["version"]
            )
            await click(rig, ".recovery-row-restore")
        else:
            await request_manual(rig)
            with rig.db.transaction() as cursor:
                cursor.execute(
                    "CREATE TRIGGER task6_chat_restore_failure BEFORE UPDATE OF archived ON conversations WHEN NEW.archived = 0 BEGIN SELECT RAISE(ABORT, 'task6 injected write failure'); END"
                )
            rig.console.query_one(
                "#recovery-scope", Select
            ).value = "chat_and_workspace"
            await click(rig, "#recovery-confirm")
        await wait_until(
            rig.pilot, lambda: bool(rig.recovery.coordinator.receipts(rig.source.id))
        )
        receipt = rig.recovery.coordinator.receipts(rig.source.id)[-1]
        assert receipt.outcome.status == case
        assert rig.app.local_chat_conversation_service.get_conversation_metadata(
            rig.target_id
        )["archived"]
        if case == "partial":
            assert (
                receipt.outcome.workspace_changed and not receipt.outcome.chat_changed
            )
            assert not rig.app.workspace_registry_service.get_workspace(
                "project"
            ).archived
        assert rig.store.active_session_id == rig.source.id
        await wait_until(
            rig.pilot, lambda: bool(rig.console.query(".recovery-receipt-refresh"))
        )
        assert rig.store.session_draft(rig.source.id) == "Unrelated unfinished draft"


@pytest.mark.asyncio
async def test_restore_as_requires_fresh_final_name_confirmation(
    recovery_stores, tmp_path, monkeypatch
):
    from tldw_chatbook.Widgets.Console.console_workspace_switcher_modal import (
        ConsoleWorkspaceRenameModal,
    )

    async with mounted_recovery(recovery_stores, tmp_path, monkeypatch) as rig:
        card = await request_manual(rig)
        await click(rig, "#recovery-rename")
        await wait_until(
            rig.pilot, lambda: isinstance(rig.host.screen, ConsoleWorkspaceRenameModal)
        )
        modal = rig.host.screen
        modal.query_one(
            "#console-workspace-rename-input", Input
        ).value = "Recovered project"
        assert await rig.pilot.click("#console-workspace-rename-save")
        await wait_until(
            rig.pilot,
            lambda: (
                rig.recovery.coordinator.snapshot(rig.source.id).confirmation.request_id
                != card.request_id
            ),
        )
        revised = rig.recovery.coordinator.snapshot(rig.source.id).confirmation
        assert revised.replacement_name == "Recovered project"
        assert "Final workspace name: Recovered project" in str(
            rig.console.query_one("#recovery-target").content
        )
        await rig.recovery.coordinator.decide(
            card.session_id, card.request_id, allow=True, scope="chat_and_workspace"
        )
        assert rig.app.workspace_registry_service.get_workspace("project").archived
        rig.console.query_one("#recovery-scope", Select).value = "chat_and_workspace"
        await click(rig, "#recovery-confirm")
        await wait_until(
            rig.pilot,
            lambda: (
                not rig.app.workspace_registry_service.get_workspace("project").archived
            ),
        )
        assert (
            rig.app.workspace_registry_service.get_workspace("project").name
            == "Recovered project"
        )
        assert rig.store.active_session_id == rig.source.id


@pytest.mark.asyncio
async def test_preview_hidden_chat_and_explicit_open_preserve_original_branch(
    recovery_stores, tmp_path, monkeypatch
):
    from tldw_chatbook.UI.Console_Modules.archive import consume_conversation_resume
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    async with mounted_recovery(recovery_stores, tmp_path, monkeypatch) as rig:

        async def navigate_to_reader(event):
            assert event.screen_name == "library"
            await rig.host.push_screen(LibraryScreen(rig.app))

        monkeypatch.setattr(rig.app, "handle_screen_navigation", navigate_to_reader)
        await click(rig, ".recovery-row-preview")
        await wait_until(rig.pilot, lambda: isinstance(rig.host.screen, LibraryScreen))
        library = rig.host.screen
        await wait_until(
            rig.pilot,
            lambda: (
                library._conversations_state.reader_state.loaded_id == rig.target_id
            ),
        )
        assert (
            library._conversations_state.reader_state.messages[0].text
            == "Original retained history"
        )
        assert rig.app.local_chat_conversation_service.get_conversation_metadata(
            rig.target_id
        )["archived"]
        assert rig.app.workspace_registry_service.get_workspace("project").archived
        assert rig.store.active_session_id == rig.source.id
        assert rig.store.session_draft(rig.source.id) == "Unrelated unfinished draft"
        from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

        assert not rig.app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_LIVE_WORK
        )
        assert not rig.app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_CONVERSATION_RESUME
        )
        await rig.host.pop_screen()
        await rig.console._sync_native_console_chat_ui()
        inactive = rig.db.add_message(
            {
                "conversation_id": rig.target_id,
                "sender": "assistant",
                "content": "Inactive branch",
                "parent_message_id": rig.message_id,
            }
        )
        rig.db.set_conversation_active_leaf(rig.target_id, rig.message_id)
        await click(rig, "#recovery-refresh")
        await wait_until(
            rig.pilot, lambda: len(rig.console.query(".archive-recovery-results")) == 2
        )
        fresh = rig.console.query(".archive-recovery-results").last()
        await click(rig, fresh.query_one(".recovery-row-restore"))
        await wait_until(
            rig.pilot, lambda: rig.console.query_one("#recovery-confirm").parent.display
        )
        rig.console.query_one("#recovery-scope", Select).value = "chat_and_workspace"
        await click(rig, "#recovery-confirm")
        await wait_until(
            rig.pilot, lambda: bool(rig.console.query(".recovery-receipt-open"))
        )
        assert rig.db.get_conversation_active_leaf(rig.target_id) == rig.message_id
        assert rig.store.active_session_id == rig.source.id
        navigation = []
        monkeypatch.setattr(rig.app, "post_message", navigation.append)
        await click(rig, ".recovery-receipt-open")
        await wait_until(
            rig.pilot,
            lambda: any(isinstance(event, NavigateToScreen) for event in navigation),
        )
        await consume_conversation_resume(rig.console)
        resumed = next(
            session
            for session in rig.store.sessions()
            if session.persisted_conversation_id == rig.target_id
        )
        assert rig.store.active_session_id == resumed.id
        assert rig.db.get_conversation_active_leaf(rig.target_id) == rig.message_id
        assert {
            node.persisted_message_id
            for node in rig.store.all_messages_for_session(resumed.id)
        } == {rig.message_id, inactive}
        assert rig.store.session_draft(rig.source.id) == "Unrelated unfinished draft"
        assert rig.gateway.stream_calls == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["expired", "stale", "source_switch"])
async def test_restore_as_modal_refusal_remains_recoverable(
    recovery_stores, tmp_path, monkeypatch, change
):
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.Widgets.Console.console_workspace_switcher_modal import (
        ConsoleWorkspaceRenameModal,
    )

    async with mounted_recovery(recovery_stores, tmp_path, monkeypatch) as rig:
        card = await request_manual(rig)
        feedback = []
        notify = rig.console.notify

        def observe_feedback(message, **kwargs):
            feedback.append((message, kwargs))
            notify(message, **kwargs)

        monkeypatch.setattr(rig.console, "notify", observe_feedback)
        navigation = []
        monkeypatch.setattr(rig.app, "post_message", navigation.append)
        await click(rig, "#recovery-rename")
        await wait_until(
            rig.pilot, lambda: isinstance(rig.host.screen, ConsoleWorkspaceRenameModal)
        )
        modal = rig.host.screen
        if change == "expired":
            rig.recovery.coordinator._monotonic = lambda: card.expires_at + 1
            rig.recovery.coordinator.expire_pending()
        elif change == "stale":
            meta = rig.app.local_chat_conversation_service.get_conversation_metadata(
                rig.target_id
            )
            rig.db.update_conversation(
                rig.target_id, {"title": "Changed"}, meta["version"]
            )
        else:
            other = rig.store.create_session(title="Other", activate=True)
        modal.query_one(
            "#console-workspace-rename-input", Input
        ).value = "Recovered project"
        assert await rig.pilot.click("#console-workspace-rename-save")
        await wait_until(rig.pilot, lambda: rig.host.screen is rig.console)
        if change == "source_switch":
            await rig.pilot.pause()
            assert not feedback
            assert rig.store.active_session_id == other.id
            assert (
                rig.recovery.coordinator.snapshot(rig.source.id).confirmation.request_id
                == card.request_id
            )
        else:
            await wait_until(rig.pilot, lambda: bool(feedback))
            assert "Refresh recovery before choosing a name" in feedback[-1][0]
            assert feedback[-1][1]["severity"] == "warning"
            await wait_until(
                rig.pilot, lambda: bool(rig.console.query(".archive-recovery-receipt"))
            )
            receipt = rig.recovery.coordinator.receipts(rig.source.id)[-1]
            assert receipt.outcome.status == change
            assert not rig.console.query_one("#recovery-confirm").parent.display
            assert rig.store.active_session_id == rig.source.id
        assert rig.host.is_running
        workspace = rig.app.workspace_registry_service.get_workspace("project")
        assert workspace.archived and workspace.name == "Project"
        assert rig.app.local_chat_conversation_service.get_conversation_metadata(
            rig.target_id
        )["archived"]
        assert rig.store.session_draft(rig.source.id) == "Unrelated unfinished draft"
        assert not any(isinstance(event, NavigateToScreen) for event in navigation)
        assert not rig.app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_CONVERSATION_RESUME
        )
        assert not rig.app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_LIVE_WORK
        )
        assert rig.gateway.stream_calls == 3


@pytest.mark.asyncio
async def test_compact_confirmation_focus_shows_exact_target(
    recovery_stores, tmp_path, monkeypatch
):
    async with mounted_recovery(
        recovery_stores, tmp_path, monkeypatch, size=(100, 30)
    ) as rig:
        await request_manual(rig)
        scope = rig.console.query_one("#recovery-scope", Select)
        scope.value = "chat_and_workspace"
        scope.scroll_visible(animate=False, force=True, immediate=True)
        scope.focus()
        await rig.pilot.press("tab")
        button = rig.console.query_one("#recovery-confirm")
        assert rig.console.focused is button
        visible = await capture(rig, tmp_path, "compact-confirm-target", button)
        assert "Original" in visible and "Confirm restore" in visible
        assert rig.console.focused is button
        await rig.pilot.press("enter")
        await wait_until(
            rig.pilot,
            lambda: (
                not rig.app.local_chat_conversation_service.get_conversation_metadata(
                    rig.target_id
                )["archived"]
            ),
        )
        assert rig.store.active_session_id == rig.source.id
        assert rig.store.session_draft(rig.source.id) == "Unrelated unfinished draft"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chat_archived,workspace_archived,scope",
    [
        (True, False, "chat_only"),
        (False, True, "workspace_only"),
        (True, True, "chat_and_workspace"),
    ],
)
async def test_native_rag_only_lifecycle_matrix_without_index(
    recovery_stores, tmp_path, monkeypatch, chat_archived, workspace_archived, scope
):
    active_id = None

    def prepare(services, target_id, db):
        nonlocal active_id
        if not chat_archived:
            meta = db.get_conversation_by_id(target_id)
            db.set_conversations_archived(
                [target_id],
                archived=False,
                expected_versions={target_id: meta["version"]},
            )
        if not workspace_archived:
            services.workspace_registry_service.unarchive_workspace("project")
        active_id = db.add_conversation({"title": "Already active in Default"})

    async with mounted_recovery(
        recovery_stores, tmp_path, monkeypatch, prepare=prepare, direct=False
    ) as rig:
        page = rig.recovery.presentations(rig.source.id)[0].projection
        assert active_id not in [row["conversation_id"] for row in page["items"]]
        row = next(
            row for row in page["items"] if row["conversation_id"] == rig.target_id
        )
        assert row["conversation_archived"] == chat_archived
        assert row["workspace"]["archived"] == workspace_archived
        card = await request_manual(rig)
        assert scope in card.allowed_scopes
        rig.console.query_one("#recovery-scope", Select).value = scope
        sends = rig.gateway.stream_calls
        await click(rig, "#recovery-confirm")
        await wait_until(
            rig.pilot, lambda: bool(rig.recovery.coordinator.receipts(rig.source.id))
        )
        outcome = rig.recovery.coordinator.receipts(rig.source.id)[-1].outcome
        assert outcome.status == "restored" and outcome.chat_available
        assert not rig.db.get_conversation_by_id(rig.target_id)["archived"]
        assert not rig.app.workspace_registry_service.get_workspace("project").archived
        assert rig.store.active_session_id == rig.source.id
        assert rig.store.session_draft(rig.source.id) == "Unrelated unfinished draft"
        assert (
            rig.db.get_conversation_active_leaf(rig.target_id) == rig.original_leaf_id
        )
        assert rig.gateway.stream_calls == sends


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True, "workspace"])
async def test_native_date_search_labels_monday_chat_friday_workspace_and_unknown(
    recovery_stores, tmp_path, monkeypatch, unknown
):
    def prepare(services, target_id, db):
        with db.transaction() as conn:
            conn.execute(
                "UPDATE conversations SET archived_at = ? WHERE id = ?",
                (None if unknown is True else "2026-09-21T10:00:00+00:00", target_id),
            )
        with services.workspace_registry_service.db.transaction() as conn:
            conn.execute(
                "UPDATE workspace_records SET archived_at = ? WHERE workspace_id = ?",
                (
                    None if unknown == "workspace" else "2026-09-25T10:00:00+00:00",
                    "project",
                ),
            )

    arguments = {
        "archived_after": "2026-09-21T00:00:00+00:00",
        "archived_before": "2026-09-22T00:00:00+00:00",
        "include_unknown_archive_times": bool(unknown),
    }
    async with mounted_recovery(
        recovery_stores,
        tmp_path,
        monkeypatch,
        prepare=prepare,
        search_arguments=arguments,
    ) as rig:
        row = rig.recovery.presentations(rig.source.id)[0].projection["items"][0]
        assert row["conversation_id"] == rig.target_id
        assert row["archive_match_basis"] == ([] if unknown is True else ["chat"])
        assert row["chronology_incomplete"] == bool(unknown)
        text = " ".join(
            str(widget.content)
            for widget in rig.console.query(".archive-recovery-results Static")
        )
        assert "2026-09-21" in text
        assert ("last modified, then chat identity" in text) == (unknown is True)
        if unknown == "workspace":
            assert "unknown" in text.lower() and "may match" not in text.lower()
        else:
            assert "2026-09-25" in text
        if unknown is True:
            assert "unknown" in text.lower() and "may match" in text.lower()
        else:
            assert datetime.fromisoformat(
                row["conversation_archived_at"]
            ) == datetime.fromisoformat("2026-09-21T10:00:00+00:00")
        assert rig.db.get_conversation_by_id(rig.target_id)["archived"]


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["system", "tool"])
async def test_native_recovery_cannot_discover_excluded_role_only_needle(
    recovery_stores, tmp_path, monkeypatch, role
):
    def prepare(services, target_id, db):
        db.add_message(
            {
                "conversation_id": target_id,
                "sender": role,
                "role": role,
                "content": "excluded_archive_unique_needle",
            }
        )

    async with mounted_recovery(
        recovery_stores,
        tmp_path,
        monkeypatch,
        prepare=prepare,
        search_arguments={"query": "excluded_archive_unique_needle"},
    ) as rig:
        assert rig.recovery.presentations(rig.source.id)[0].projection["items"] == []
        assert not rig.console.query(".recovery-row-restore")
        assert rig.db.get_conversation_by_id(rig.target_id)["archived"]
        assert all(
            "excluded_archive_unique_needle" not in str(messages)
            for messages in rig.gateway.messages_seen
        )


@pytest.mark.asyncio
async def test_native_stale_result_after_rehoming_does_not_restore_either_owner(
    recovery_stores, tmp_path, monkeypatch
):
    async with mounted_recovery(recovery_stores, tmp_path, monkeypatch) as rig:
        rig.app.workspace_registry_service.create_workspace(
            workspace_id="other", name="Other"
        )
        metadata = rig.db.get_conversation_by_id(rig.target_id)
        rig.db.update_conversation(
            rig.target_id, {"workspace_id": "other"}, metadata["version"]
        )
        await click(rig, ".recovery-row-restore")
        await wait_until(
            rig.pilot, lambda: bool(rig.recovery.coordinator.receipts(rig.source.id))
        )
        assert (
            rig.recovery.coordinator.receipts(rig.source.id)[-1].outcome.status
            == "stale"
        )
        assert rig.db.get_conversation_by_id(rig.target_id)["workspace_id"] == "other"
        assert rig.db.get_conversation_by_id(rig.target_id)["archived"]
        assert rig.app.workspace_registry_service.get_workspace("project").archived
        assert not rig.app.workspace_registry_service.get_workspace("other").archived
        assert rig.store.active_session_id == rig.source.id
        assert rig.store.session_draft(rig.source.id) == "Unrelated unfinished draft"


@pytest.mark.asyncio
async def test_native_cancel_after_workspace_commit_reports_partial_and_skips_chat(
    recovery_stores, tmp_path, monkeypatch
):
    import threading

    async with mounted_recovery(recovery_stores, tmp_path, monkeypatch) as rig:
        card = await request_manual(rig)
        loop = asyncio.get_running_loop()
        committed = asyncio.Event()
        release = threading.Event()
        original = rig.app.workspace_registry_service.unarchive_workspace

        def pause_after_commit(*args, **kwargs):
            result = original(*args, **kwargs)
            loop.call_soon_threadsafe(committed.set)
            assert release.wait(10), "test must release the committed workspace writer"
            return result

        monkeypatch.setattr(
            rig.app.workspace_registry_service,
            "unarchive_workspace",
            pause_after_commit,
        )
        waiter = asyncio.create_task(
            rig.recovery.coordinator.wait_result(card.operation_id)
        )
        rig.console.query_one("#recovery-scope", Select).value = "chat_and_workspace"
        await click(rig, "#recovery-confirm")
        try:
            await asyncio.wait_for(committed.wait(), 5)
            assert not rig.app.workspace_registry_service.get_workspace(
                "project"
            ).archived
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        finally:
            release.set()
        await wait_until(
            rig.pilot, lambda: bool(rig.recovery.coordinator.receipts(rig.source.id))
        )
        outcome = rig.recovery.coordinator.receipts(rig.source.id)[-1].outcome
        assert outcome.status == "partial"
        assert (
            outcome.workspace_changed
            and not outcome.chat_changed
            and not outcome.chat_available
        )
        assert rig.db.get_conversation_by_id(rig.target_id)["archived"]
        assert rig.store.active_session_id == rig.source.id
        assert rig.store.session_draft(rig.source.id) == "Unrelated unfinished draft"
        assert rig.gateway.stream_calls == 3


class DestinationTransitionGateway(SearchOnlyGateway):
    local_turn = True

    async def resolve_for_send(self, selection):
        from tldw_chatbook.Chat.console_dispatch_checkpoint import ConsoleEgressClass

        resolution = await super().resolve_for_send(selection)
        if self.local_turn:
            return replace(
                resolution,
                resolved_destination=replace(
                    resolution.resolved_destination,
                    endpoint_identity="http://127.0.0.1:9099",
                    egress_class=ConsoleEgressClass.ON_DEVICE,
                ),
            )
        return resolution

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        self.destination_snapshots.append(self.source.library_destination_runtime)
        if self.local_turn:
            yield "Initial local response."
            return
        async for chunk in super().stream_chat(
            resolution, messages, tools=tools, **kwargs
        ):
            yield chunk


@pytest.mark.asyncio
async def test_current_native_recovery_discloses_external_destination_transition(
    recovery_stores, tmp_path, monkeypatch
):
    from tldw_chatbook.Chat.console_dispatch_checkpoint import ConsoleEgressClass

    gateway = DestinationTransitionGateway()
    async with mounted_recovery(
        recovery_stores, tmp_path, monkeypatch, gateway_override=gateway
    ) as rig:
        first, external = gateway.destination_snapshots[:2]
        assert first.resolved_destination.egress_class is ConsoleEgressClass.ON_DEVICE
        assert first.disclosure is None
        assert external.resolved_destination.egress_class is ConsoleEgressClass.UNKNOWN
        assert external.disclosure is not None
        assert (
            external.disclosure.previous_resolved_identity[3]
            is ConsoleEgressClass.ON_DEVICE
        )
        assert external.disclosure.resolved_destination == external.resolved_destination
        assert external.owner_attempt_id and external.owner_message_id
        assert rig.source.library_destination_runtime.disclosure is None


@pytest.mark.asyncio
async def test_native_rows_number_immutable_page_and_label_inactive_match(
    recovery_stores, tmp_path, monkeypatch
):
    branch_id = None

    def prepare(services, target_id, db):
        nonlocal branch_id
        original_leaf = db.get_conversation_active_leaf(target_id)
        branch_id = db.add_message(
            {
                "conversation_id": target_id,
                "sender": "assistant",
                "content": "branch_discovery_needle",
                "parent_message_id": original_leaf,
            }
        )
        db.set_conversation_active_leaf(target_id, original_leaf)
        other = db.add_conversation({"title": "Other branch_discovery_needle"})
        db.set_conversations_archived(
            [other], archived=True, expected_versions={other: 1}
        )

    async with mounted_recovery(
        recovery_stores,
        tmp_path,
        monkeypatch,
        prepare=prepare,
        search_arguments={"query": "branch_discovery_needle"},
    ) as rig:
        page = rig.recovery.presentations(rig.source.id)[0].projection
        assert len(page["items"]) == 2
        rows = rig.console.query(".archive-recovery-row")
        target_index = next(
            i
            for i, row in enumerate(page["items"])
            if row["conversation_id"] == rig.target_id
        )
        target = page["items"][target_index]
        assert target["inactive_branch"] and target["match_message_id"] == branch_id
        assert (
            rig.db.get_conversation_active_leaf(rig.target_id) == rig.original_leaf_id
        )
        for index, row in enumerate(page["items"]):
            assert str(rows[index].query(Static).first().content).startswith(
                f"{index + 1}. {row['title']}"
            )
        text = " ".join(
            str(widget.content) for widget in rows[target_index].query(Static)
        )
        assert "Match found in inactive branch" in text
        sends = rig.gateway.stream_calls
        await click(rig, f"#recovery-row-{target_index}-restore")
        await wait_until(
            rig.pilot, lambda: rig.console.query_one("#recovery-confirm").parent.display
        )
        rig.console.query_one("#recovery-scope", Select).value = "chat_and_workspace"
        await click(rig, "#recovery-confirm")
        await wait_until(
            rig.pilot, lambda: bool(rig.recovery.coordinator.receipts(rig.source.id))
        )
        assert not rig.db.get_conversation_by_id(rig.target_id)["archived"]
        assert (
            rig.db.get_conversation_active_leaf(rig.target_id) == rig.original_leaf_id
        )
        original = (
            rig.app.local_chat_conversation_service.get_library_conversation_messages(
                rig.target_id, message_limit=10
            )
        )
        assert {row["id"] for row in original["messages"]} == {
            rig.message_id,
            branch_id,
        }
        assert rig.store.active_session_id == rig.source.id
        assert rig.store.session_draft(rig.source.id) == "Unrelated unfinished draft"
        assert rig.gateway.stream_calls == sends


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["restore", "open"])
@pytest.mark.parametrize("key", ["enter", "space"])
async def test_focused_native_recovery_actions_activate_from_keyboard(
    recovery_stores, tmp_path, monkeypatch, action, key
):
    from tldw_chatbook.UI.Console_Modules.archive import consume_conversation_resume
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    async with mounted_recovery(recovery_stores, tmp_path, monkeypatch) as rig:
        navigation = []
        monkeypatch.setattr(rig.app, "post_message", navigation.append)
        if action == "open":
            await request_manual(rig)
            rig.console.query_one(
                "#recovery-scope", Select
            ).value = "chat_and_workspace"
            await click(rig, "#recovery-confirm")
            await wait_until(
                rig.pilot, lambda: bool(rig.console.query(".recovery-receipt-open"))
            )
        selector = (
            ".recovery-row-restore" if action == "restore" else ".recovery-receipt-open"
        )
        button = rig.console.query_one(selector)
        button.scroll_visible(animate=False, force=True, immediate=True)
        button.focus()
        await rig.pilot.pause()
        assert rig.console.focused is button
        sends = rig.gateway.stream_calls
        button.disabled = True
        # Disabled controls blur; exercise their refusal with an actual click
        # instead of accidentally sending a key to the focused composer.
        await rig.pilot.click(button)
        await rig.pilot.pause()
        if action == "restore":
            assert not rig.recovery.coordinator.snapshot(rig.source.id)
        else:
            assert not any(isinstance(event, NavigateToScreen) for event in navigation)
        button.disabled = False
        button.focus()
        await rig.pilot.pause()
        assert rig.console.focused is button
        await rig.pilot.press(key)
        if action == "restore":
            await wait_until(
                rig.pilot,
                lambda: rig.console.query_one("#recovery-confirm").parent.display,
            )
            assert (
                rig.recovery.coordinator.snapshot(
                    rig.source.id
                ).confirmation.target.conversation_id
                == rig.target_id
            )
            assert rig.db.get_conversation_by_id(rig.target_id)["archived"]
            assert rig.store.active_session_id == rig.source.id
        else:
            await wait_until(
                rig.pilot,
                lambda: any(
                    isinstance(event, NavigateToScreen) for event in navigation
                ),
            )
            await consume_conversation_resume(rig.console)
            active = next(
                session
                for session in rig.store.sessions()
                if session.id == rig.store.active_session_id
            )
            assert active.persisted_conversation_id == rig.target_id
            assert (
                rig.db.get_conversation_active_leaf(rig.target_id)
                == rig.original_leaf_id
            )
        assert rig.store.session_draft(rig.source.id) == "Unrelated unfinished draft"
        assert rig.gateway.stream_calls == sends


@pytest.mark.asyncio
async def test_retained_native_recovery_resource_owners_plateau_and_close(
    tmp_path, monkeypatch
):
    """Measure one retained native owner before any test-fixture GC teardown."""
    import json
    import os
    import shutil
    import sqlite3
    import subprocess
    import sys
    import threading
    from concurrent.futures import ThreadPoolExecutor

    lsof = shutil.which("lsof")
    if lsof is None:
        pytest.skip("exact OS database descriptor qualification requires lsof")
    loop = asyncio.get_running_loop()
    previous_executor = loop._default_executor
    worker = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="recovery-owner-probe"
    )
    fixture = None
    worker_threads = set()
    main_thread = threading.get_ident()
    handles = {name: {} for name in ("chat", "workspace", "runs")}
    handle_threads = {}
    try:
        loop.set_default_executor(worker)

        async def on_each_worker(callback):
            barrier = threading.Barrier(2, timeout=5)

            def synchronized():
                barrier.wait()
                return callback()

            return await asyncio.gather(
                loop.run_in_executor(worker, synchronized),
                loop.run_in_executor(worker, synchronized),
            )

        worker_threads = set(await on_each_worker(threading.get_ident))
        fixture = recovery_stores.__wrapped__(tmp_path)
        services, target_id, db = next(fixture)
        workspace_db = services.workspace_registry_service.db
        threads = {name: set() for name in handles}
        samples = []
        runs_instances = {}

        def observe(name, original):
            def recorded(*args, **kwargs):
                connection = original(*args, **kwargs)
                if name == "runs" and os.path.realpath(
                    args[0].db_path_str
                ) != os.path.realpath(tmp_path / "runs.sqlite"):
                    return connection
                if name == "runs":
                    runs_instances[id(args[0])] = args[0]
                handles[name][id(connection)] = connection
                handle_threads[id(connection)] = threading.get_ident()
                threads[name].add(threading.get_ident())
                return connection

            return recorded

        monkeypatch.setattr(db, "get_connection", observe("chat", db.get_connection))
        monkeypatch.setattr(
            workspace_db,
            "_held_connection",
            observe("workspace", workspace_db._held_connection),
        )
        monkeypatch.setattr(
            AgentRunsDB,
            "_held_connection",
            observe("runs", AgentRunsDB._held_connection),
        )
        db.get_connection()
        workspace_db._held_connection()
        owned_paths = {
            name: os.path.realpath(tmp_path / filename)
            for name, filename in (
                ("chat", "chat.sqlite"),
                ("workspace", "workspace.sqlite"),
                ("runs", "runs.sqlite"),
            )
        }

        async def sample(phase):
            # lsof gives actual OS descriptors and exact database file ownership;
            # retain only these disposable paths, never unrelated process filenames.
            output = (
                await asyncio.to_thread(
                    subprocess.run,
                    [lsof, "-a", "-p", str(os.getpid()), "-Fn"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            ).stdout
            owned_fds = {name: [] for name in handles}
            descriptor = None
            for line in output.splitlines():
                if line.startswith("f"):
                    descriptor = line[1:]
                elif line.startswith("n") and descriptor and descriptor[0].isdigit():
                    path = os.path.realpath(line[1:])
                    for name, owned in owned_paths.items():
                        if path in (owned, owned + "-wal", owned + "-shm"):
                            owned_fds[name].append(descriptor)

            def live_handles_on_current_thread():
                counts = dict.fromkeys(handles, 0)
                for name, connections in handles.items():
                    for connection in connections.values():
                        if handle_threads[id(connection)] != threading.get_ident():
                            continue
                        try:
                            connection.execute("SELECT 1")
                        except sqlite3.ProgrammingError as exc:
                            assert "closed" in str(exc)
                        else:
                            counts[name] += 1
                return counts

            assert set(handle_threads.values()) <= {main_thread, *worker_threads}
            main_live = live_handles_on_current_thread()
            worker_live = await on_each_worker(live_handles_on_current_thread)
            open_handles = {
                name: main_live[name] + sum(row[name] for row in worker_live)
                for name in handles
            }
            record = {
                "phase": phase,
                "os_fd_total": len(
                    os.listdir(
                        "/dev/fd" if sys.platform == "darwin" else "/proc/self/fd"
                    )
                ),
                "owned_os_fds": {name: len(fds) for name, fds in owned_fds.items()},
                "live_handles": open_handles,
                "threads_seen": {name: len(ids) for name, ids in threads.items()},
                "chat_registered": db.registered_connection_count(),
            }
            samples.append(record)
            return record

        await sample("stores_created")
        async with mounted_recovery(
            (services, target_id, db), tmp_path, monkeypatch
        ) as rig:
            owner = rig.owner
            runtime = rig.recovery
            controller = rig.controller
            for cycle in range(5):
                if cycle:
                    metadata = services.local_chat_conversation_service.get_conversation_metadata(
                        target_id
                    )
                    db.set_conversations_archived(
                        [target_id],
                        archived=True,
                        expected_versions={target_id: metadata["version"]},
                    )
                    services.workspace_registry_service.archive_workspace("project")
                    await runtime.search(rig.source.id, {}, model_visible=False)
                    await rig.pilot.pause()
                result = list(rig.console.query(".archive-recovery-results"))[-1]
                await click(rig, result.query_one(".recovery-row-restore"))
                await wait_until(
                    rig.pilot,
                    lambda: (
                        runtime.coordinator.snapshot(rig.source.id).confirmation
                        is not None
                    ),
                )
                operation = runtime.coordinator.snapshot(rig.source.id)
                rig.console.query_one(
                    "#recovery-scope", Select
                ).value = "chat_and_workspace"
                await click(rig, "#recovery-confirm")
                outcome = await runtime.coordinator.wait_result(operation.operation_id)
                assert outcome.status == "restored"
                assert not services.local_chat_conversation_service.get_conversation_metadata(
                    target_id
                )["archived"]
                assert not services.workspace_registry_service.get_workspace(
                    "project"
                ).archived
                await rig.pilot.pause()
                assert (
                    not any(runtime._searches.values())
                    and not runtime.coordinator._tasks
                )
                await sample(f"settled_cycle_{cycle + 1}")
            stable = samples[-3:]
            assert all(
                {
                    key: row[key]
                    for key in (
                        "os_fd_total",
                        "owned_os_fds",
                        "live_handles",
                        "chat_registered",
                    )
                }
                == {
                    key: stable[0][key]
                    for key in (
                        "os_fd_total",
                        "owned_os_fds",
                        "live_handles",
                        "chat_registered",
                    )
                }
                for row in stable
            ), samples
            assert stable[-1]["live_handles"]["chat"] >= 2
            assert stable[-1]["live_handles"]["workspace"] >= 2
            await runtime.close_session(rig.source.id)
            assert not runtime.presentations(rig.source.id)
            assert (
                not any(runtime._searches.values()) and not runtime.coordinator._tasks
            )
            await sample("source_closed")
        # These are the same retained owners, now joined/disposed by the helper.
        assert runtime._shutdown_task.done()
        assert controller is owner.chat_controller or owner.chat_controller is None
        await sample("runtime_disposed")
        fixture.close()
        current_thread_closed = await sample("current_thread_stores_closed")
        assert current_thread_closed["chat_registered"] == 0

        def close_owned_worker_stores():
            assert threading.get_ident() in worker_threads
            workspace_db.close()
            for runs_db in runs_instances.values():
                runs_db.close()

        # The fixture externally owns these stores. Its current-thread close is
        # not an application-runtime disposal guarantee for worker-held handles.
        # This probe owns both workers, so close each store on its creating thread.
        await on_each_worker(close_owned_worker_stores)
        closed = await sample("all_owned_store_handles_closed")
        assert closed["live_handles"] == dict.fromkeys(handles, 0), samples
        assert closed["owned_os_fds"] == dict.fromkeys(handles, 0), samples
    finally:
        try:
            try:
                if fixture is not None:
                    fixture.close()
            finally:
                # Close each observed handle on its exact creator, even after a
                # partial setup or fixture cleanup failure, before executor/GC.
                def close_handles_on_creator():
                    for connections in handles.values():
                        for connection in connections.values():
                            if handle_threads[id(connection)] == threading.get_ident():
                                connection.close()

                try:
                    if worker_threads:
                        await on_each_worker(close_handles_on_creator)
                finally:
                    close_handles_on_creator()
        finally:
            loop._default_executor = previous_executor
            worker.shutdown(wait=True)
    evidence = {
        "samples": samples,
        "cycles": 5,
        "executor_workers": 2,
        "observer_retains_handles": True,
        "gc_requested": False,
    }
    (tmp_path / "resource-owners.json").write_text(json.dumps(evidence, indent=2))
    print("RECOVERY_RESOURCE_OWNERS " + json.dumps(evidence))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["unavailable", "incomplete", "source_switch"])
async def test_receipt_refresh_refusal_keeps_native_source_safe(
    recovery_stores, tmp_path, monkeypatch, failure
):
    import threading

    from tldw_chatbook.Chat.archive_recovery.search_service import ArchiveSearchService

    async with mounted_recovery(
        recovery_stores,
        tmp_path,
        monkeypatch,
        search_arguments={"workspace_id": "project"},
    ) as rig:
        # Wait for the setup turn's rendered receipt before changing visibility.
        marks = rig.owner._console_local_marks_service()
        await wait_until(rig.pilot, lambda: marks.list_console_unseen_marks() == ())
        rig.db.update_conversation(
            rig.target_id,
            {"title": "Changed"},
            rig.db.get_conversation_by_id(rig.target_id)["version"],
        )
        await click(rig, ".recovery-row-restore")
        await wait_until(
            rig.pilot, lambda: bool(rig.console.query(".recovery-receipt-refresh"))
        )
        feedback = []
        original_notify = rig.console.notify

        def notify(message, **kwargs):
            feedback.append((message, kwargs))
            original_notify(message, **kwargs)

        monkeypatch.setattr(rig.console, "notify", notify)
        sends = rig.gateway.stream_calls
        navigation = []
        monkeypatch.setattr(rig.app, "post_message", navigation.append)
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        if failure == "unavailable":
            with rig.app.workspace_registry_service.db.transaction() as conn:
                conn.execute(
                    "DELETE FROM workspace_records WHERE workspace_id = ?", ("project",)
                )
        else:
            original_search = ArchiveSearchService.search

            def delayed_budget(self, *args, **kwargs):
                if failure == "source_switch":
                    loop.call_soon_threadsafe(entered.set)
                    assert release.wait(10)
                # Real search boundary, deterministic expired budget, no fake result.
                ticks = iter([0.0, 6.0])
                self.monotonic = lambda: next(ticks, 6.0)
                return original_search(self, *args, **kwargs)

            monkeypatch.setattr(ArchiveSearchService, "search", delayed_budget)
        try:
            await click(rig, ".recovery-receipt-refresh")
            if failure == "source_switch":
                await asyncio.wait_for(entered.wait(), 5)
                other = rig.store.create_session(title="Other", activate=True)
                await rig.console._sync_native_console_chat_ui()
                release.set()
                await wait_until(
                    rig.pilot, lambda: not any(rig.recovery._searches.values())
                )
            await rig.pilot.pause()
            assert rig.host.is_running
            if failure == "source_switch":
                assert not feedback
                assert rig.store.active_session_id == other.id
                assert not rig.console.query(".archive-recovery-receipt")
                assert not rig.console.query(".archive-recovery-results")
            else:
                assert len(feedback) == 1 and feedback[0][1]["severity"] == "warning"
                assert bool(rig.console.query(".recovery-receipt-refresh"))
            assert rig.db.get_conversation_by_id(rig.target_id)["archived"]
            assert (
                rig.store.session_draft(rig.source.id) == "Unrelated unfinished draft"
            )
            assert rig.gateway.stream_calls == sends
            assert not navigation
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_state", ["background", "detached", "closed"])
async def test_admitted_restore_delivers_app_completion_when_source_cannot_receive(
    recovery_stores, tmp_path, monkeypatch, source_state
):
    import threading

    async with mounted_recovery(recovery_stores, tmp_path, monkeypatch) as rig:
        # Wait for the setup turn's rendered receipt before changing visibility.
        marks = rig.owner._console_local_marks_service()
        await wait_until(rig.pilot, lambda: marks.list_console_unseen_marks() == ())
        card = await request_manual(rig)
        sends = rig.gateway.stream_calls
        feedback = []
        original_notify = rig.app.notify

        def notify(message, **kwargs):
            feedback.append((message, kwargs))
            original_notify(message, **kwargs)

        monkeypatch.setattr(rig.app, "notify", notify)
        navigation = []
        monkeypatch.setattr(rig.app, "post_message", navigation.append)
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        original = rig.app.workspace_registry_service.unarchive_workspace

        def gated(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(15)
            return original(*args, **kwargs)

        monkeypatch.setattr(
            rig.app.workspace_registry_service, "unarchive_workspace", gated
        )
        rig.console.query_one("#recovery-scope", Select).value = "chat_and_workspace"
        closing = None
        try:
            await click(rig, "#recovery-confirm")
            await asyncio.wait_for(entered.wait(), 5)
            if source_state == "detached":
                assert rig.owner.detach_view(
                    rig.console, rig.console._console_runtime_attachment_generation
                )
                assert rig.recovery.on_settled is None
            elif source_state == "background":
                other = rig.store.create_session(title="Other", activate=True)
                await rig.console._sync_native_console_chat_ui()
            else:
                closing = asyncio.create_task(
                    rig.owner.close_session(
                        rig.source.id,
                        expected_revision=rig.controller.lifecycle_impact(
                            session_id=rig.source.id
                        ).revision,
                    )
                )
                async with asyncio.timeout(5):
                    while rig.source.id not in rig.recovery._closed_sessions:
                        await asyncio.sleep(0)
                assert not closing.done()
                assert rig.recovery.coordinator.snapshot(rig.source.id) is not None
            release.set()
            if closing:
                await closing
                await rig.console._sync_native_console_chat_ui()
            else:
                await rig.recovery.coordinator.wait_result(card.operation_id)
            await rig.pilot.pause()
            assert not rig.app.workspace_registry_service.get_workspace(
                "project"
            ).archived
            assert bool(rig.db.get_conversation_by_id(rig.target_id)["archived"]) == (
                source_state == "closed"
            )
            assert len(feedback) == 1
            text = str(feedback[0][0])
            assert "Workspace: restored" in text
            assert (
                "Chat: unchanged" if source_state == "closed" else "Chat: restored"
            ) in text
            from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

            assert rig.gateway.stream_calls == sends
            assert not any(isinstance(event, NavigateToScreen) for event in navigation)
            assert not any(
                session.persisted_conversation_id == rig.target_id
                for session in rig.store.sessions()
            )
            if source_state == "closed":
                assert not any(
                    session.id == rig.source.id for session in rig.store.sessions()
                )
                assert not rig.recovery.coordinator.receipts(rig.source.id)
                await wait_until(
                    rig.pilot,
                    lambda: not rig.console.query(".archive-recovery-receipt"),
                    lambda: (
                        rig.store.active_session_id,
                        rig.console._archive_recovery_view()._session_id,
                    ),
                )
            else:
                if source_state == "detached":
                    rig.console._console_runtime_attachment_generation = (
                        rig.owner.attach_view(rig.console)
                    )
                else:
                    assert rig.store.active_session_id == other.id
                    assert not rig.console.query(".archive-recovery-receipt")
                    rig.store.switch_session(rig.source.id)
                await rig.console._sync_native_console_chat_ui()
                await rig.console._sync_native_console_chat_ui()
                await rig.recovery.coordinator.decide(
                    rig.source.id,
                    card.request_id,
                    allow=True,
                    scope="chat_and_workspace",
                )
                assert len(feedback) == 1
                assert (
                    rig.store.session_draft(rig.source.id)
                    == "Unrelated unfinished draft"
                )
        finally:
            release.set()
            if closing:
                await closing


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["setup", "cleanup"])
async def test_resource_probe_restores_executor_after_partial_failure(
    tmp_path, monkeypatch, fault
):
    import subprocess

    loop = asyncio.get_running_loop()
    previous = loop._default_executor
    original_fixture = recovery_stores.__wrapped__
    fixtures = []

    def broken_fixture(path):
        if fault == "setup":
            raise RuntimeError("injected setup failure")
        fixture = original_fixture(path)
        fixtures.append(fixture)
        try:
            yield next(fixture)
        finally:
            fixture.close()
            raise RuntimeError("injected cleanup failure")

    def fail_sample(*args, **kwargs):
        raise OSError("injected sample failure")

    monkeypatch.setattr(recovery_stores, "__wrapped__", broken_fixture)
    monkeypatch.setattr(subprocess, "run", fail_sample)
    try:
        with pytest.raises(RuntimeError, match=f"injected {fault} failure"):
            await test_retained_native_recovery_resource_owners_plateau_and_close(
                tmp_path, monkeypatch
            )
        assert loop._default_executor is previous
    finally:
        # Keep this regression's own RED run from contaminating following tests.
        leaked = loop._default_executor
        loop._default_executor = previous
        if leaked is not previous:
            leaked.shutdown(wait=True)
        for fixture in fixtures:
            fixture.close()


@pytest.mark.asyncio
async def test_long_current_title_has_bounded_usable_native_confirmation(
    recovery_stores, tmp_path, monkeypatch
):
    title = "  " + "Long title " * 100 + "  "

    def prepare(services, cid, db):
        db.update_conversation(
            cid, {"title": title}, db.get_conversation_by_id(cid)["version"]
        )

    async with mounted_recovery(
        recovery_stores, tmp_path, monkeypatch, size=(100, 30), prepare=prepare
    ) as rig:
        card = await request_manual(rig)
        assert card.target.title == title.strip()
        label = str(rig.console.query_one("#recovery-confirm").label)
        assert label.startswith("Confirm restore · Long title")
        assert label.endswith("…") and len(label) < 200
        rig.console.query_one("#recovery-scope", Select).value = "chat_and_workspace"
        scope = rig.console.query_one("#recovery-scope", Select)
        scope.scroll_visible(animate=False, force=True, immediate=True)
        scope.focus()
        await rig.pilot.press("tab")
        button = rig.console.query_one("#recovery-confirm")
        assert rig.console.focused is button
        rendered = await capture(rig, tmp_path, "confirmation-long-title-100", button)
        assert (
            "Confirm restore" in rendered
            and "Long title" in rendered
            and "…" in rendered
        )
        await rig.pilot.press("tab")
        assert rig.console.focused.id == "recovery-cancel"
        await rig.pilot.press("shift+tab", "enter")
        await wait_until(
            rig.pilot, lambda: bool(rig.recovery.coordinator.receipts(rig.source.id))
        )
        assert (
            rig.recovery.coordinator.receipts(rig.source.id)[-1].outcome.status
            == "restored"
        )
        assert (
            rig.store.active_session_id == rig.source.id
            and rig.gateway.stream_calls == 3
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["enter", "send-button"])
async def test_stock_received_send_keeps_exact_recovery_user_proof(route, monkeypatch):
    from Tests.UI.test_console_hook_review_send_freeze import _until
    from Tests.UI.test_console_received_intent_feedback import (
        _held_received_record,
        _received_console_case,
        _send,
    )

    async with _received_console_case(monkeypatch, f"archive-proof-{route}") as case:
        _send(case, route)
        record = await _held_received_record(case)
        proof = record.received_intent.recovery_user_origin
        recovery = case.runtime.ensure_archive_recovery()
        assert proof is not None and proof.session_id == case.session.id
        assert recovery._proofs.get(id(proof)) is proof

        case.probe.release.set()
        assert await _until(lambda: record.request is not None, 15)
        assert record.request.configuration.recovery_user_origin is proof
