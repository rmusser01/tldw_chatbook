"""Rendered, real-owner coverage of Console progress inspection and discard."""

from importlib.util import find_spec
from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import Button, SelectionList

from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Agents.fleet_messages import MessageIdentity, MessageStore


class StyledConsoleHarness(ConsoleHarness):
    """Use the production component rules as well as consolidated widget CSS."""

    CSS_PATH = str(
        Path(__file__).resolve().parents[2] / "tldw_chatbook/css/tldw_cli_modular.tcss"
    )

    def __init__(self, app_instance):
        from Tests.UI.test_console_native_chat_flow import (
            _configure_native_ready_console,
        )

        _configure_native_ready_console(app_instance, model="sample-model")
        super().__init__(app_instance)


async def _show_button(pilot, host, button):
    """Settle rail recomposition before claiming a scroll target is painted."""
    for _ in range(4):
        button.scroll_visible(animate=False, force=True)
        await pilot.pause(0.2)
        if (
            0 <= button.region.y < host.size.height
            and 0 <= button.region.x + 1 < host.size.width
            and host.get_widget_at(button.region.x + 1, button.region.y)[0] is button
        ):
            return
    raise AssertionError(f"Button is not painted at {button.region}")


def _queue(store=None, conversation_id="conv-A"):
    store = store or MessageStore()
    inbox = store.open_inbox(conversation_id)
    sender = inbox.sender(
        MessageIdentity("child-1", "run-1", "parent-1", "chain-1", "Researcher")
    )
    return store, inbox, sender


def _modal_type():
    module = "tldw_chatbook.Widgets.Console.console_agent_progress_modal"
    assert find_spec(module) is not None, "Console progress inspection modal is missing"
    from tldw_chatbook.Widgets.Console.console_agent_progress_modal import (
        ConsoleAgentProgressModal,
    )

    return ConsoleAgentProgressModal


class ProgressHarness(App):
    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, modal):
        super().__init__()
        self.modal = modal

    def on_mount(self):
        self.push_screen(self.modal)


@pytest.mark.asyncio
async def test_inspect_select_concurrent_arrival_discard_preserves_exact_ids_and_lifetime(
    tmp_path,
):
    from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs

    modal_type = _modal_type()
    store, inbox, sender = _queue()
    first = sender.send("[bold]literal finding[/bold]\nsecond line\tindent\rcarriage")
    second = sender.send("Another finding")
    modal = modal_type(
        conversation_id="conv-A", load=inbox.snapshot, discard=inbox.discard
    )
    host = ProgressHarness(modal)
    marks = _attach_real_dbs(host, tmp_path)
    marks.set_mark("conv-A", marks.FLEET_UNSEEN)
    attention_before = marks.get_mark("conv-A", marks.FLEET_UNSEEN)
    async with host.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        assert len(inbox.snapshot()) == 2
        listing = modal.query_one(SelectionList)
        listing.select(first)
        third = sender.send("Concurrent arrival")
        await pilot.pause()
        await pilot.click("#agent-progress-discard")
        await pilot.pause()
        assert [m.message_id for m in inbox.snapshot()] == [second, third]
        assert listing.option_count == 2
        assert "Discarded 1" in str(
            modal.query_one("#agent-progress-status").renderable
        )
        # Discard releases pending capacity but does not refund accepted sends.
        assert inbox._senders["child-1"].accepted_count == 3
        assert store.pending_counts() == {"conv-A": 2}
        assert marks.get_mark("conv-A", marks.FLEET_UNSEEN) == attention_before
        assert modal.query_one(Button).is_mounted


@pytest.mark.asyncio
async def test_unsaved_mode_off_progress_entry_and_captured_owner_survive_navigation(
    tmp_path,
):
    from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
    from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector

    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    host = StyledConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        controller = console._console_chat_controller
        session_id = controller.store.active_session_id
        assert console._character._current_console_rail_conversation_id() is None
        bridge = console._ensure_console_agent_bridge()
        owner_id = controller.store.progress_owner_id(session_id)
        store, _inbox, sender = _queue(bridge.message_store, owner_id)
        selected = sender.send("PRIVATE-BODY-PROGRESS")
        console._set_console_rail_preference(
            section_updates={"agent": True}, notify_on_failure=False
        )
        console._sync_console_agent_section()
        await pilot.pause(0.6)
        assert list(console.query("#console-agent-progress")), (
            "Agent section has no progress action"
        )
        button = console.query_one("#console-agent-progress", Button)
        await _show_button(pilot, host, button)
        assert "1 queued" in str(button.label)
        assert "PRIVATE-BODY-PROGRESS" not in str(
            console._agent._console_agent_fleet_section_state()
        )
        await pilot.click("#console-agent-progress")
        await pilot.pause()
        modal = host.screen
        assert isinstance(modal, _modal_type())
        assert modal.conversation_id == owner_id
        modal.query_one(SelectionList).select(selected)
        store.close_inbox(owner_id)
        _, replacement, new_sender = _queue(store, owner_id)
        new_id = new_sender.send("Replacement private body")
        await pilot.click("#agent-progress-discard")
        await pilot.pause(0.6)
        assert [m.message_id for m in replacement.snapshot()] == [new_id]
        assert modal.query_one(SelectionList).option_count == 0
        assert "unavailable" in str(
            modal.query_one("#agent-progress-status").renderable
        )
        await pilot.click("#agent-progress-close")
        await pilot.pause()
        console._agent.open_fleet_progress()
        await pilot.pause()
        assert host.screen.query_one(SelectionList).option_count == 1
        reader = replacement.reader("primary", chain_id=None, automatic=False)
        assert reader.collect().collected_count == 1
        await pilot.pause(0.7)
        assert host.screen.query_one(SelectionList).option_count == 0
        assert "0 queued" in str(
            console.query_one("#console-agent-progress", Button).label
        )


def test_navigation_keeps_progress_separate_from_historical_counts_for_unsaved_sessions():
    from dataclasses import replace

    from tldw_chatbook.Workspaces.conversation_browser_state import (
        ConsoleConversationBrowserInputRow,
        build_console_conversation_browser_state,
    )

    row = ConsoleConversationBrowserInputRow(
        row_key="native:draft-1",
        conversation_id=None,
        native_session_id="draft-1",
        title="Draft",
        scope_type="global",
        workspace_id=None,
        workspace_label="Chats",
    )
    state = build_console_conversation_browser_state(
        rows=[
            row,
            replace(
                row,
                row_key="saved-1",
                conversation_id="saved-1",
                native_session_id="native-saved-1",
            ),
        ],
        active_workspace_id=None,
        progress_counts={"draft-1": 3, "native-saved-1": 4},
        subagent_counts={"draft-1": 9, "saved-1": 7},
    )
    rows = {item.row_key: item for item in state.sections[-1].rows}
    assert (
        rows["native:draft-1"].progress_count,
        rows["native:draft-1"].subagent_count,
    ) == (3, 0)
    assert (rows["saved-1"].progress_count, rows["saved-1"].subagent_count) == (4, 7)


@pytest.mark.asyncio
async def test_model_collection_refreshes_open_modal_without_discard_or_wake():
    store, inbox, sender = _queue()
    first = sender.send("[bold]Literal markup[/bold]\tTab\rReturn")
    modal = _modal_type()(
        conversation_id="conv-A", load=inbox.snapshot, discard=inbox.discard
    )
    host = ProgressHarness(modal)
    async with host.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        listing = modal.query_one(SelectionList)
        assert modal.query_one("#agent-progress-body").renderable.plain.endswith(
            "[bold]Literal markup[/bold]\\tTab\\rReturn"
        )
        assert not modal.query_one("#agent-progress-body").renderable.spans
        listing.select(first)
        reader = inbox.reader("primary", chain_id=None, automatic=False)
        assert reader.collect().collected_count == 1
        await pilot.pause(0.7)
        assert listing.option_count == 0
        assert modal.query_one("#agent-progress-discard", Button).disabled
        assert store.pending_counts() == {}
        sender.send("Still live")
        await pilot.pause(0.7)
        assert listing.option_count == 1
        timer = modal._poll_timer
        await pilot.click("#agent-progress-close")
        await pilot.pause()
        assert timer._task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(180, 48), (100, 36)])
async def test_console_progress_navigation_pruned_access_and_actual_paint(
    tmp_path, size
):
    from pathlib import Path

    from Tests.UI.test_console_fleet_discoverability import _compositor_text
    from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
    from Tests.UI.test_console_parallel_runs import _assert_painted_at_own_region
    from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
    from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator

    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    host = StyledConsoleHarness(app)
    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        controller = console._console_chat_controller
        session_a = controller.store.active_session_id
        bridge = console._ensure_console_agent_bridge()
        console._console_config()["agent_runtime"] = False
        controller.update_agent_runtime(enabled=False, bridge=bridge)
        inbox = bridge.message_store.open_inbox(
            controller.store.progress_owner_id(session_a)
        )
        console._set_console_rail_preference(
            left_open=True,
            section_updates={"agent": True},
            notify_on_failure=False,
        )
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.6)
        assert "0 queued" in str(
            console.query_one("#console-agent-progress", Button).label
        )
        fleet = FleetCoordinator(max_live=2, clock=lambda: 1000.0, message_inbox=inbox)
        handle = fleet.reserve("Synthetic sample research", "Researcher (sample)")
        fleet.attach_run(handle.handle_id, "sample-run")
        sender = fleet.bind_progress_sender(
            handle.handle_id, parent_run_id="sample-primary", chain_id=None
        )
        sender.send(
            "Sample report: the parser expects UTF-8 input.\n[bold]This is literal report text.[/bold]"
        )
        sender.send(
            "Sample report: validation is complete; the supervisor may relay the finding."
        )
        fleet.finish(handle.handle_id, "done", result="Sample final result")
        assert fleet.prune_terminal() == 1
        assert fleet.snapshot() == []
        await pilot.pause(0.6)
        button = console.query_one("#console-agent-progress", Button)
        await _show_button(pilot, host, button)
        assert "2 queued" in str(button.label)
        _assert_painted_at_own_region(host, button)
        evidence = (
            Path(__file__).resolve().parents[2]
            / ".superpowers/sdd/2026-09-10-scoped-agent-messaging/task-3-screenshots"
        )
        evidence.mkdir(parents=True, exist_ok=True)
        dimensions = f"{size[0]}x{size[1]}"
        host.save_screenshot(
            f"console-progress-entry-{dimensions}.svg", path=str(evidence)
        )
        console._agent.open_fleet_progress()
        await pilot.pause()
        modal = host.screen
        assert modal.query_one(SelectionList).option_count == 2
        modal.query_one(SelectionList).select(inbox.snapshot()[0].message_id)
        await pilot.pause()
        paint = _compositor_text(host.export_screenshot())
        assert "Queued progress (this session)" in paint
        assert "Sample report:" in paint
        assert "Discard selected (1)" in paint
        host.save_screenshot(
            f"console-progress-modal-{dimensions}.svg", path=str(evidence)
        )
        await pilot.click("#agent-progress-close")
        await pilot.pause()
        # Navigation paints the pending count separately; the bodies stay out of its DTO.
        console._set_console_rail_preference(
            section_updates={"conversations": True}, notify_on_failure=False
        )
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        state = console._workspace._build_console_workspace_context_state().conversation_browser
        rows = [row for section in state.sections for row in section.rows]
        row_a = next(row for row in rows if row.native_session_id == session_a)
        assert row_a.progress_count == 2
        assert "Sample report" not in repr(state)
        conversation_buttons = [
            button
            for button in console.query_one("#console-workspace-context").query(Button)
            if getattr(button, "row_key", None) == row_a.row_key
            and "Progress:" in str(button.label)
        ]
        assert conversation_buttons
        await _show_button(pilot, host, conversation_buttons[0])
        host.save_screenshot(
            f"console-progress-navigation-{dimensions}.svg", path=str(evidence)
        )
        region = conversation_buttons[0].region
        strips = host.screen._compositor.render_strips()
        row_paint = "\n".join(
            strip.crop(region.x, region.right).text
            for strip in strips[region.y : region.bottom]
        )
        assert "Progress: 2" in row_paint
        session_b = controller.new_session().id
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.6)
        assert console._agent.progress_state()[0] == 0
        _, inbox_b, sender_b = _queue(
            bridge.message_store, controller.store.progress_owner_id(session_b)
        )
        sender_b.send("Other conversation")
        controller.switch_session(session_a)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.6)
        assert console._agent.progress_state()[0] == 2
        console._agent.open_fleet_progress()
        await pilot.pause()
        modal = host.screen
        modal.query_one(SelectionList).select(inbox.snapshot()[0].message_id)
        controller.store.switch_session(session_b)
        # A saved stale view refuses even before its polling timer observes the change.
        await pilot.click("#agent-progress-discard")
        await pilot.pause(0.6)
        assert len(inbox.snapshot()) == 2
        assert len(inbox_b.snapshot()) == 1
        assert modal.query_one(SelectionList).option_count == 0


@pytest.mark.asyncio
async def test_real_live_report_save_preserves_open_selection_and_navigation(
    tmp_path, monkeypatch
):
    import asyncio
    import threading

    from Tests.Agents.conftest import pin_agent_settings
    from Tests.Chat.test_console_agent_swap import (
        SUBAGENT_PROMPT_PREFIX,
        _fence,
        _Gateway,
    )
    from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
    from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
    from tldw_chatbook.Agents.fleet_messages import MessageError
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )

    pin_agent_settings(monkeypatch, max_live_subagents=3, run_log_enabled=False)
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    host = StyledConsoleHarness(app)
    entered, release = threading.Event(), threading.Event()
    gateway = _Gateway(
        [
            [_fence("spawn_subagent", {"task": "report twice"})],
            [_fence("wait_agents", {})],
            ["done"],
        ],
        [
            [_fence("report_to_supervisor", {"message": "selected before Save"})],
            [_fence("report_to_supervisor", {"message": "arrival after Save"})],
            ["child done"],
        ],
    )
    original_stream = gateway.stream_chat

    async def gated_stream(resolution, messages, **kwargs):
        if (
            str(messages[0].get("content", "")).startswith(SUBAGENT_PROMPT_PREFIX)
            and gateway.child_calls == 1
        ):
            entered.set()
            assert await asyncio.to_thread(release.wait, 20)
        async for chunk in original_stream(resolution, messages, **kwargs):
            yield chunk

    monkeypatch.setattr(gateway, "stream_chat", gated_stream)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        controller = console._console_chat_controller
        store = controller.store
        session = store.create_session(
            ephemeral=True,
            settings=store.session_settings(store.active_session_id),
            project_instruction_state=ProjectInstructionControlState.legacy_disabled(),
        )
        bridge = console._ensure_console_agent_bridge()
        bridge._gateway = gateway
        controller.provider_gateway = gateway
        console._console_config()["agent_runtime"] = True
        controller.update_agent_runtime(enabled=True, bridge=bridge)
        task = asyncio.create_task(controller.submit_draft("Report findings"))
        try:
            assert await asyncio.to_thread(entered.wait, 10), (
                task.result() if task.done() else "submit still pending",
                gateway.calls,
            )
            owner = store.progress_owner_id(session.id)
            inbox = bridge.message_store.get_inbox(owner)
            original = inbox.snapshot()[0]
            console._agent.open_fleet_progress()
            await pilot.pause()
            modal = host.screen
            listing = modal.query_one(SelectionList)
            listing.select(original.message_id)
            # Upstream admission forbids Save while a turn owns this session.
            with pytest.raises(RuntimeError, match="pending turn"):
                await asyncio.to_thread(store.promote_ephemeral_session, session.id)
            assert console._agent.progress_state()[0] == 1
            release.set()
            assert (await task).accepted
            assert await asyncio.to_thread(store.promote_ephemeral_session, session.id)
            await pilot.pause(0.7)
            assert listing.option_count == 2
            assert listing.selected == [original.message_id]
            await pilot.click("#agent-progress-discard")
            await pilot.pause()
            assert [message.body for message in inbox.snapshot()] == [
                "arrival after Save"
            ]
            assert console._agent.progress_state() == (1, {session.id: 1})
            state = console._workspace._build_console_workspace_context_state().conversation_browser
            rows = [row for section in state.sections for row in section.rows]
            assert (
                next(
                    row for row in rows if row.native_session_id == session.id
                ).progress_count
                == 1
            )
            stale_load, stale_discard = modal._load, modal._discard
            impact = controller.lifecycle_impact(session_id=session.id)
            ticket = controller.begin_session_close(
                session.id, expected_revision=impact.revision
            )
            controller.finalize_session_close(ticket)
            with pytest.raises(MessageError):
                stale_load()
            await pilot.pause()
            replacement = store.create_session(session_id=session.id, ephemeral=True)
            assert store.progress_owner_id(replacement.id) != owner
            with pytest.raises(MessageError):
                stale_load()
            with pytest.raises(MessageError):
                stale_discard([original.message_id])
            assert bridge.message_store.pending_counts() == {}
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
