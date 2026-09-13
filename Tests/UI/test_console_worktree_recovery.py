"""Mounted exact-round agent worktree decisions."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from Tests.Agents.test_agent_worktree_confirmed_recovery import work as _work

work = _work

from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards


class CardApp(App):
    def compose(self) -> ComposeResult:
        yield ChatTaskCards(id="cards")

    def on_mount(self):
        self.decisions = []

    def on_chat_task_cards_worktree_decided(self, event):
        self.decisions.append((event.request_id, event.allow))


def payload(round_id="round-1"):
    return {
        "request_id": round_id,
        "action": "apply",
        "source": "/tmp/[bold]file[/bold]\x1b\nname",
        "destination": "/tmp/repo",
        "diffstat": " [red]file[/red] | 1 +",
    }


@pytest.mark.asyncio
async def test_card_plain_text_and_immutable_old_button():
    app = CardApp()
    async with app.run_test(size=(80, 30)) as pilot:
        cards = app.query_one(ChatTaskCards)
        state = TaskResumeState()
        state.pending_worktree_merge = payload()
        cards.sync_state(state)
        await pilot.pause()
        assert cards.display
        card = cards.query_one("#chat-worktree-confirm-card")
        assert card.display
        assert "[bold]file[/bold]" in str(
            card.query_one("#worktree-source", Static).render()
        )
        assert "\\x1b" in str(card.query_one("#worktree-source", Static).render())
        old = card.query_one(".worktree-allow", Button)
        card.post_message(Button.Pressed(old))
        await pilot.pause()
        assert app.decisions == [("round-1", True)]
        cards.sync_state(state)
        await pilot.pause()
        assert old.disabled
        state.pending_worktree_merge = payload("round-2")
        cards.sync_state(state)
        await pilot.pause()
        card.post_message(Button.Pressed(old))
        await pilot.pause()
        assert app.decisions == [("round-1", True)]
        card.query_one(".worktree-deny", Button).press()
        await pilot.pause()
        assert app.decisions[-1] == ("round-2", False)


def test_worktree_payload_is_never_serialized_or_restored():
    state = TaskResumeState()
    state.pending_worktree_merge = payload()
    assert "pending_worktree_merge" not in state.to_dict()
    assert (
        TaskResumeState.from_dict(
            {"pending_worktree_merge": payload()}
        ).pending_worktree_merge
        is None
    )


def test_recovery_row_copy_preserves_uncertainty():
    import importlib.util

    assert importlib.util.find_spec(
        "tldw_chatbook.Widgets.Chat_Widgets.worktree_recovery_dialog"
    )
    from tldw_chatbook.Widgets.Chat_Widgets.worktree_recovery_dialog import row_status

    assert row_status(
        {"writer_state": "held", "run_status": "done", "mutation_state": "unresolved"}
    ) == (False, "Completion has not been confirmed; work is retained.")
    assert (
        "manual review"
        in row_status(
            {
                "writer_state": "drained",
                "run_status": "done",
                "mutation_state": "applying",
            }
        )[1]
    )
    assert (
        "baseline checkout retained"
        in row_status(
            {
                "writer_state": "drained",
                "run_status": "done",
                "mutation_state": "discarded_cleanup_pending",
            }
        )[1]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action, change",
    [
        ("apply", "normal"),
        ("discard", "normal"),
        ("apply", "deny"),
        ("apply", "source_changed"),
        ("apply", "authority_changed"),
        ("apply", "adapter"),
        ("apply", "adapter_stale"),
        ("apply", "adapter_close"),
    ],
)
async def test_reopened_record_manual_action_uses_real_controller_card_and_git(
    work, action, change
):
    import asyncio
    from types import SimpleNamespace

    from Tests.Chat.test_console_agent_project_instructions import _BindingRegistry
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )
    from tldw_chatbook.Chat.console_worktree_recovery import ConsoleWorktreeRecovery
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

    db, authority, child, _records = work
    (child.worktree_path / "a.txt").write_text("child changes\n")
    db.close()
    reopened = AgentRunsDB(db.db_path_str)
    store = ConsoleChatStore()
    session = store.create_session(title="Recovery", workspace_id="workspace")
    session.persisted_conversation_id = "chat"
    state = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="binding",
        working_folder_locator_fingerprint="f" * 64,
        project_instruction_notice_key=None,
    )
    store.set_session_project_instruction_state(session.id, state)
    binding = WorkspaceRuntimeBinding(
        workspace_id="workspace",
        binding_id="binding",
        binding_kind="local-filesystem",
        label="Repository",
        locator=str(authority.root),
        status="ready",
        metadata={"access": "rw"},
    )
    # The recorded fingerprint is the real current locator fingerprint.
    from tldw_chatbook.Chat.console_project_instructions import (
        fingerprint_canonical_locator,
    )

    fingerprint = fingerprint_canonical_locator(str(authority.root))
    from dataclasses import replace

    store.set_session_project_instruction_state(
        session.id, replace(state, working_folder_locator_fingerprint=fingerprint)
    )
    with reopened.transaction() as conn:
        conn.execute(
            "UPDATE agent_worktrees SET locator_fingerprint=? WHERE run_id=?",
            (fingerprint, child.run_id),
        )
    controller = ConsoleChatController(store=store, provider_gateway=object())
    capacity = RuntimeCapacity()
    helper = ConsoleWorktreeRecovery(
        controller, SimpleNamespace(runs_db=reopened, runtime_capacity=capacity)
    )

    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    notifications = []

    class RecoveryApp(CardApp):
        action_recover_agent_work = ChatScreen.action_recover_agent_work

        def _console_runtime(self):
            return runtime

        def notify(self, message, **kwargs):
            notifications.append(message)

        def console_view_hooks(self):
            return {
                "set_pending_worktree_merge": lambda value: self.query_one(
                    ChatTaskCards
                ).sync_state(TaskResumeState(pending_worktree_merge=value))
            }

        workspace_registry_service = _BindingRegistry([binding])
        unified_mcp_service = SimpleNamespace(get_kill_switch=lambda: False)

        def on_chat_task_cards_worktree_decided(self, event):
            controller.resolve_pending_worktree_merge(
                event.allow, request_id=event.request_id
            )

    app = RecoveryApp()
    runtime = ConsoleRuntime(app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    runtime._worktree_recovery = helper
    try:
        async with app.run_test(size=(100, 35)) as pilot:
            controller.app = app
            generation = runtime.attach_view(app)
            runtime.finish_view_reconciliation(app, generation)
            controller.set_pending_worktree_merge = lambda value: app.query_one(
                ChatTaskCards
            ).sync_state(TaskResumeState(pending_worktree_merge=value))
            page = await helper.list_work(session.id)
            assert [r["run_id"] for r in page.rows] == [child.run_id]
            assert page.rows[0]["run_status"] == "done"
            if change.startswith("adapter"):
                from tldw_chatbook.Widgets.Chat_Widgets.worktree_recovery_dialog import (
                    _RecoveryButton,
                )

                await app.action_recover_agent_work()
                await pilot.pause()
                if change == "adapter_close":
                    draining, release = asyncio.Event(), asyncio.Event()

                    async def await_fleet(_conversation_id):
                        draining.set()
                        await release.wait()
                        return True

                    runtime._agent_bridge = SimpleNamespace(
                        await_fleet_terminal=await_fleet
                    )
                    close = asyncio.create_task(
                        runtime.close_session(
                            session.id,
                            expected_revision=controller.lifecycle_impact(
                                session_id=session.id
                            ).revision,
                        )
                    )
                    try:
                        await asyncio.wait_for(draining.wait(), 3)
                        app.screen.query(_RecoveryButton).first().press()
                        await pilot.pause(0.2)
                        assert not helper.operations
                        assert not helper.receipts
                        assert not controller.pending_worktree_merge_ids()
                        assert not app.query(".worktree-allow")
                        assert (authority.root / "a.txt").read_text() == "base\n"
                    finally:
                        release.set()
                        await close
                    return
                app.screen.query(_RecoveryButton).first().press()
                for _ in range(100):
                    if session.id in helper.operations:
                        break
                    await pilot.pause(0.02)
                assert session.id in helper.operations, (notifications, helper.receipts)
                waiter = helper.operations[session.id].task
            else:
                waiter = asyncio.create_task(
                    helper.start(session.id, child.run_id, action)
                )
            for _ in range(100):
                if app.query(".worktree-allow"):
                    break
                await pilot.pause(0.02)
            assert controller.pending_worktree_merge_ids()
            assert app.query(".worktree-allow")
            await pilot.pause()
            if change == "source_changed":
                (child.worktree_path / "a.txt").write_text("new source after preview\n")
            if change == "authority_changed":
                store.set_session_project_instruction_state(
                    session.id, replace(state, working_folder_binding_id=None)
                )
            if change == "adapter_stale":
                other = store.create_session(title="Other", workspace_id="workspace")
                other.persisted_conversation_id = "other-chat"
                store.switch_session(other.id)
            app.query_one(
                ".worktree-deny" if change == "deny" else ".worktree-allow", Button
            ).press()
            outcome = await asyncio.wait_for(waiter, 10)
            if change in {"normal", "adapter", "adapter_stale"}:
                assert outcome.state == (
                    "applied" if action == "apply" else "discarded_cleanup_pending"
                )
                assert (authority.root / "a.txt").read_text() == (
                    "child changes\n" if action == "apply" else "base\n"
                )
                if change != "adapter_stale":
                    assert (
                        await helper.start(session.id, child.run_id, action)
                    ).reason_code
            else:
                assert outcome.reason_code
                assert (authority.root / "a.txt").read_text() == "base\n"
            assert child.worktree_path.exists()
            if change == "adapter_stale":
                await pilot.pause()
                assert notifications == []
                assert helper.receipts["chat"] is outcome
                assert "other-chat" not in helper.receipts

    finally:
        await helper.close()
        controller.begin_shutdown()
        capacity.close()
        reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("allow", [True, False])
async def test_controller_bridge_turn_child_write_visible_confirmation(
    work, monkeypatch, allow
):
    import asyncio
    from types import SimpleNamespace

    import tldw_chatbook.Chat.console_agent_bridge as bridge_module
    from Tests.Agents.conftest import join_fleet_children, pin_agent_settings
    from Tests.Agents.test_agent_service import FleetChat, fence
    from Tests.Agents.test_fleet_runtime import _fs_local_provider
    from Tests.console_provider_doubles import (
        persisted_console_store,
        provider_resolution,
    )
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
        fingerprint_canonical_locator,
    )
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

    db, authority, _old_child, _records = work
    pin_agent_settings(
        monkeypatch,
        max_live_subagents=3,
        max_steps=40,
        max_model_turns=40,
        subagents_outlive_turn=False,
    )
    services = []
    real_service = bridge_module.AgentService

    class ObservedService(real_service):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            services.append(self)

        def run_turn(self, *args, **kwargs):
            assert kwargs.get("request_worktree_merge_confirm") is not None, (
                "missing real worktree hook"
            )
            return super().run_turn(*args, **kwargs)

    monkeypatch.setattr(bridge_module, "AgentService", ObservedService)

    def merge_reply():
        service = services[-1]
        join_fleet_children(service)
        handle_id = next(iter(service._agent_worktrees))
        return fence("merge_agent_worktree", {"handle_id": handle_id})

    script = FleetChat(
        [
            fence(
                "spawn_subagent", {"task": "write child file", "isolation": "worktree"}
            ),
            fence("wait_agents", {}),
            merge_reply,
            merge_reply,
            "finished",
        ],
        {
            "write child file": [
                fence("load_tools", {"ids": ["local:fs_write"]}),
                fence("fs_write", {"path": "new-child.txt", "content": "from child"}),
                fence("fs_write", {"path": "new-child.txt", "content": "from child"}),
                "child done",
            ]
        },
    )

    class Gateway:
        async def resolve_for_send(self, selection):
            return provider_resolution(
                provider="llama_cpp", model="test-model", execution_key="llama_cpp"
            )

        async def stream_chat(self, resolution, messages, **kwargs):
            yield script(messages_payload=messages)["choices"][0]["message"]["content"]

    from pathlib import Path

    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

    workspace_db = WorkspaceDB(
        Path(db.db_path_str).parent / "workspaces.db", client_id="flow"
    )
    registry = LocalWorkspaceRegistryService(workspace_db)
    registry.create_workspace(workspace_id="workspace", name="Work")
    store = persisted_console_store(
        db_path=Path(db.db_path_str).parent / "chat.db", workspace_registry=registry
    )
    session = store.create_session(title="Child confirmation", workspace_id="workspace")
    binding = WorkspaceRuntimeBinding(
        workspace_id="workspace",
        binding_id="binding",
        binding_kind="local-filesystem",
        label="Repository",
        locator=str(authority.root),
        status="ready",
        metadata={"access": "rw"},
    )
    registry.save_runtime_binding(binding)
    store.set_session_project_instruction_state(
        session.id,
        ProjectInstructionControlState(
            project_instructions_enabled=True,
            working_folder_binding_id="binding",
            working_folder_locator_fingerprint=fingerprint_canonical_locator(
                str(authority.root)
            ),
            project_instruction_notice_key=None,
        ),
    )
    gateway = Gateway()
    capacity = RuntimeCapacity()
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=gateway,
        native_tools_enabled=lambda: False,
        runtime_capacity=capacity,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
        provider="llama_cpp",
        model="test-model",
    )

    async def providers(**kwargs):
        return None, None, _fs_local_provider(authority.root), None

    monkeypatch.setattr(controller, "_compose_agent_request_providers", providers)

    class TurnApp(CardApp):
        workspace_registry_service = registry
        unified_mcp_service = SimpleNamespace(get_kill_switch=lambda: False)

        def console_view_hooks(self):
            return {
                "set_pending_worktree_merge": lambda value: self.query_one(
                    ChatTaskCards
                ).sync_state(TaskResumeState(pending_worktree_merge=value))
            }

        def on_chat_task_cards_worktree_decided(self, event):
            controller.resolve_pending_worktree_merge(
                event.allow, request_id=event.request_id
            )

    controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    app = TurnApp()
    runtime = ConsoleRuntime(app)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    runtime.set_agent_bridge(bridge)
    submit = None
    try:
        async with app.run_test(size=(100, 35)) as pilot:
            generation = runtime.attach_view(app)
            runtime.finish_view_reconciliation(app, generation)
            submit = asyncio.create_task(
                controller.submit_draft(
                    "Please delegate the edit", session_id=session.id
                )
            )
            for _ in range(200):
                if app.query(".worktree-allow"):
                    break
                await pilot.pause(0.03)
            if not controller.pending_worktree_merge_ids():
                print(
                    "FLOW_MESSAGES",
                    script.parent_calls[-1]["messages_payload"]
                    if script.parent_calls
                    else [],
                )
            assert controller.pending_worktree_merge_ids(), (
                submit.result() if submit.done() else "no card",
                script.parent_calls[-1] if script.parent_calls else None,
            )
            await pilot.pause()
            card = app.query_one("#chat-worktree-confirm-card")
            assert card.display
            card.query_one(
                ".worktree-allow" if allow else ".worktree-deny", Button
            ).press()
            result = await asyncio.wait_for(submit, 15)
            assert result.accepted
            for service in services:
                join_fleet_children(service)
            for _ in range(100):
                if not controller.pending_worktree_merge_ids():
                    break
                await pilot.pause(0.03)
            await pilot.pause(0.3)
            created = next(iter(services[-1]._agent_worktrees.values()))
            assert (created.worktree_path / "new-child.txt").read_text() == "from child"
            assert (authority.root / "new-child.txt").exists() is allow
    finally:
        await controller.shutdown()
        if submit is not None and not submit.done():
            await asyncio.wait_for(submit, 5)
        for service in services:
            join_fleet_children(service)
        bridge.close_all_progress()
        capacity.close()
        db.close()
        workspace_db.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", [True, False])
@pytest.mark.parametrize("inbox_state", ["unopened", "empty", "queued"])
@pytest.mark.parametrize("fleet_size", [1, 3])
async def test_controller_previews_match_live_worktree_schemas(
    work, monkeypatch, surface, inbox_state, fleet_size
):
    import asyncio
    from types import SimpleNamespace

    import tldw_chatbook.Chat.console_agent_bridge as bridge_module
    from Tests.Agents.conftest import join_fleet_children, pin_agent_settings
    from Tests.Agents.test_fleet_runtime import _fs_local_provider
    from Tests.console_provider_doubles import (
        persisted_console_store,
        provider_resolution,
    )
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
        fingerprint_canonical_locator,
    )
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

    db, authority, _old_child, _records = work
    pin_agent_settings(
        monkeypatch,
        max_live_subagents=fleet_size,
        max_steps=40,
        max_model_turns=40,
        subagents_outlive_turn=False,
        run_log_enabled=False,
    )
    services = []
    real_service = bridge_module.AgentService

    class ObservedService(real_service):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            services.append(self)

        def _build_model_request(self, *args, **kwargs):
            request = super()._build_model_request(*args, **kwargs)
            actual_requests.append(request)
            return request

    actual_requests = []
    plans = []
    real_plan = bridge_module.build_console_first_request_plan

    def observe_plan(**kwargs):
        plan = real_plan(**kwargs)
        plans.append((kwargs["worktree_merge_enabled"], plan))
        return plan

    monkeypatch.setattr(bridge_module, "AgentService", ObservedService)
    monkeypatch.setattr(bridge_module, "build_console_first_request_plan", observe_plan)

    class Gateway:
        async def resolve_for_send(self, selection):
            return provider_resolution(
                provider="openai", model="gpt-4o-mini", execution_key="openai"
            )

        async def stream_chat(self, resolution, messages, **kwargs):
            yield "finished"

    from pathlib import Path

    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

    workspace_db = WorkspaceDB(
        Path(db.db_path_str).parent / "workspaces.db", client_id="flow"
    )
    registry = LocalWorkspaceRegistryService(workspace_db)
    registry.create_workspace(workspace_id="workspace", name="Work")
    store = persisted_console_store(
        db_path=Path(db.db_path_str).parent / "chat.db", workspace_registry=registry
    )
    session = store.create_session(title="Child confirmation", workspace_id="workspace")
    binding = WorkspaceRuntimeBinding(
        workspace_id="workspace",
        binding_id="binding",
        binding_kind="local-filesystem",
        label="Repository",
        locator=str(authority.root),
        status="ready",
        metadata={"access": "rw"},
    )
    registry.save_runtime_binding(binding)
    store.set_session_project_instruction_state(
        session.id,
        ProjectInstructionControlState(
            project_instructions_enabled=True,
            working_folder_binding_id="binding",
            working_folder_locator_fingerprint=fingerprint_canonical_locator(
                str(authority.root)
            ),
            project_instruction_notice_key=None,
        ),
    )
    gateway = Gateway()
    capacity = RuntimeCapacity()
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=gateway,
        native_tools_enabled=lambda: True,
        runtime_capacity=capacity,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
        provider="openai",
        model="gpt-4o-mini",
    )

    async def providers(**kwargs):
        return None, None, _fs_local_provider(authority.root), None

    monkeypatch.setattr(controller, "_compose_agent_request_providers", providers)

    class TurnApp(CardApp):
        workspace_registry_service = registry
        unified_mcp_service = SimpleNamespace(get_kill_switch=lambda: False)

        def console_view_hooks(self):
            return {
                "set_pending_worktree_merge": lambda value: self.query_one(
                    ChatTaskCards
                ).sync_state(TaskResumeState(pending_worktree_merge=value))
            }

        def on_chat_task_cards_worktree_decided(self, event):
            controller.resolve_pending_worktree_merge(
                event.allow, request_id=event.request_id
            )

    controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    app = TurnApp()
    from Tests.Chat.test_console_raw_shell_revocation import _PermissionService

    app.unified_mcp_service = _PermissionService()
    app.raw_cli_runtime = SimpleNamespace(
        permitted=True,
        armed=inbox_state == "queued",
        set_model_authority_revoker=lambda callback: None,
        execute=lambda *args, **kwargs: pytest.fail("preview must never execute shell"),
    )
    runtime = ConsoleRuntime(app)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    runtime.set_agent_bridge(bridge)
    if inbox_state != "unopened":
        from tldw_chatbook.Agents.fleet_messages import MessageIdentity

        owner_id = store.progress_owner_id(session.id)
        inbox = bridge.message_store.open_inbox(owner_id)
        if inbox_state == "queued":
            inbox.sender(MessageIdentity("h", "r", "parent", None, "reader")).send(
                "retained progress"
            )
    submit = None
    try:
        async with app.run_test(size=(100, 35)):
            generation = runtime.attach_view(app)
            runtime.finish_view_reconciliation(app, generation)
            if not surface:
                controller.set_pending_worktree_merge = None
            # A generic retained decision target never substitutes for this hook.
            monkeypatch.setattr(
                controller._interrupt_host,
                "has_retained_decision_target",
                lambda sid: True,
            )
            from Tests.Chat.test_console_personal_context_snapshot import (
                _ProfileContextBuilder,
            )

            async def builder(_service=None):
                return _ProfileContextBuilder()

            monkeypatch.setattr(controller, "_personal_context_builder", builder)
            before_inbox = bridge._session_progress_inbox(session.id)
            before_messages = before_inbox.snapshot() if before_inbox else ()
            await controller.build_context_snapshot(
                draft="question", session_id=session.id
            )
            assert bridge._session_progress_inbox(session.id) is before_inbox
            assert (before_inbox.snapshot() if before_inbox else ()) == before_messages
            assert len(plans) == 2, "both real controller-to-bridge previews must plan"
            preview_plans = tuple(plans)
            assert len(actual_requests) == 2, (
                "project preview must build an actual request"
            )
            preview_request = actual_requests[-1]
            preview_count = len(actual_requests)
            result = await controller.submit_draft("question", session_id=session.id)
            assert result.accepted
            assert len(plans) == 3
            assert len(actual_requests) > preview_count
            live_request = actual_requests[-1]
            assert preview_request.tools == live_request.tools
            for enabled, plan in preview_plans:
                assert enabled is surface
                assert (
                    plan.schemas.runtime_schemas == plans[2][1].schemas.runtime_schemas
                )
                assert plan.schemas.active_schemas == plans[2][1].schemas.active_schemas
            names = {tool["function"]["name"] for tool in live_request.tools}
            assert ("shell_exec" in names) is (inbox_state == "queued")
            assert "virtual_cli" in names
            assert "spawn_subagent" in names
            assert ("merge_agent_worktree" in names) is (surface and fleet_size > 1)
            assert ("discard_agent_worktree" in names) is (surface and fleet_size > 1)
    finally:
        await controller.shutdown()
        if submit is not None and not submit.done():
            await asyncio.wait_for(submit, 5)
        for service in services:
            join_fleet_children(service)
        bridge.close_all_progress()
        capacity.close()
        db.close()
        workspace_db.close()
        store.persistence.db.close_connection()
