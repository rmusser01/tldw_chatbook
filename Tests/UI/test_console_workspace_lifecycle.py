"""Workspace rename/archive lifecycle through the switcher modal (TASK-714)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from textual.css.query import QueryError
from textual.widgets import Button, Input

from Tests.UI.background_signals import wait_for_signal
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Console import (
    ConsoleWorkspaceRenameModal,
    ConsoleWorkspaceSwitcherModal,
)
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID


async def _open_switcher(host, pilot):
    await pilot.press("alt+w")
    await pilot.pause(0.2)
    modal = host.screen_stack[-1]
    assert isinstance(modal, ConsoleWorkspaceSwitcherModal)
    return modal


@pytest.mark.asyncio
async def test_rename_flow_renames_workspace() -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-a", name="Workspace 1")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        notifications: list[str] = []
        app.notify = lambda message, **kwargs: notifications.append(str(message))

        modal = await _open_switcher(host, pilot)
        rename_buttons = [
            button
            for button in modal.query(".console-workspace-switcher-lifecycle")
            if str(button.label) == "Rename"
        ]
        assert rename_buttons, "expected a Rename button for the non-default row"
        rename_buttons[0].press()
        await pilot.pause(0.3)

        rename_modal = host.screen_stack[-1]
        assert isinstance(rename_modal, ConsoleWorkspaceRenameModal)
        name_input = rename_modal.query_one("#console-workspace-rename-input", Input)
        name_input.value = "Client A"
        rename_modal.query_one("#console-workspace-rename-save", Button).press()
        await pilot.pause(0.4)

        record = registry.get_workspace("ws-a")
        assert record is not None and record.name == "Client A"
        assert any("Client A" in message for message in notifications)


@pytest.mark.asyncio
async def test_archive_flow_confirms_and_falls_back_to_default() -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-a", name="Workspace 1")
    registry.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        notifications: list[str] = []
        app.notify = lambda message, **kwargs: notifications.append(str(message))

        modal = await _open_switcher(host, pilot)
        archive_buttons = [
            button
            for button in modal.query(".console-workspace-switcher-lifecycle")
            if str(button.label) == "Archive"
        ]
        assert archive_buttons, "expected an Archive button for the non-default row"
        archive_buttons[0].press()
        await pilot.pause(0.3)

        confirm = host.screen_stack[-1]
        assert isinstance(confirm, ConfirmationDialog)
        confirm.query_one("#confirm-button", Button).press()
        await pilot.pause(0.4)

        record = registry.get_workspace("ws-a")
        assert record is not None and record.archived is True
        active = registry.get_active_workspace()
        assert active is not None
        assert active.workspace_id == DEFAULT_WORKSPACE_ID
        assert any("stay saved" in message for message in notifications)


@pytest.mark.asyncio
async def test_default_workspace_has_no_lifecycle_buttons() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        modal = await _open_switcher(host, pilot)
        assert not list(modal.query(".console-workspace-switcher-lifecycle")), (
            "the built-in Default workspace must not offer rename/archive"
        )


@pytest.mark.asyncio
async def test_default_row_labeled_everyday_chats() -> None:
    """ADR-027 (TASK-723): the switcher annotates Default so it agrees with
    the browser filing Default-workspace conversations under Chats."""
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-a", name="Workspace 1")
    registry.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        modal = await _open_switcher(host, pilot)
        default_buttons = [
            button
            for button in modal.query(".console-workspace-switcher-option")
            if "Default" in str(getattr(button, "label", ""))
        ]
        assert default_buttons, "expected the Default row as a switch option"
        assert "everyday chats" in str(default_buttons[0].label)

        console = host.screen_stack[-2]
        state = console._workspace._build_console_workspace_context_state()
        assert [node.workspace_id for node in state.workspace_tree] == ["ws-a"]
        assert DEFAULT_WORKSPACE_ID not in {
            node.workspace_id for node in state.workspace_tree
        }


@pytest.mark.asyncio
async def test_owner_tokens_fall_back_while_rail_widgets_are_unmounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._workspace

        def query_unmounted(*_args, **_kwargs):
            raise QueryError("rail widget is not mounted")

        with monkeypatch.context() as patch:
            patch.setattr(
                console,
                "_console_conversation_browser_owner",
                None,
                raising=False,
            )
            patch.setattr(console, "query_one", query_unmounted)

            assert controller._workspace_tree_owner_token() is None
            assert controller._flat_conversation_owner_token() is console


@pytest.mark.asyncio
async def test_mounted_page_request_publishes_loading_and_suppresses_duplicate() -> (
    None
):
    # This case owns the manual request lifecycle.  Explicitly persist no
    # expanded workspaces so the startup expansion loader does not claim the
    # same lane before the test installs its controlled async service.
    app = _build_test_app(
        config_overrides={
            "console": {
                "conversation_browser": {"expanded_workspace_ids": []},
            },
        }
    )
    app.workspace_registry_service.create_workspace(
        workspace_id="ws-a", name="Workspace 1"
    )
    started = asyncio.Event()
    release = asyncio.Event()
    service_calls: list[dict[str, object]] = []

    async def list_conversations(**kwargs):
        service_calls.append(kwargs)
        if not (
            kwargs.get("scope_type") == "workspace"
            and kwargs.get("workspace_id") == "ws-a"
        ):
            return {"items": [], "pagination": {"total": 0}}
        started.set()
        await release.wait()
        return {
            "items": [{"id": "page-row", "title": "Page row", "state": "saved"}],
            "pagination": {"total": 1},
        }

    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._workspace
        published = []
        original_sync = console._sync_console_workspace_context

        def capture_sync():
            published.append(controller.workspace_tree_projection())
            return original_sync()

        controller._sync_workspace_context_fn = capture_sync
        app.local_chat_conversation_service = None
        app.chat_conversation_scope_service = SimpleNamespace(
            list_conversations=list_conversations,
            local_service=None,
        )
        controller._invalidate_console_persisted_rows_cache()
        controller.request_workspace_tree_page("ws-a", 0)
        await wait_for_signal(started, what="workspace page request start")

        assert published
        loading_node = {node.workspace_id: node for node in published[-1]}["ws-a"]
        assert loading_node.loading is True

        controller.request_workspace_tree_page("ws-a", 0)
        await pilot.pause(0.1)
        page_calls = [
            call
            for call in service_calls
            if call.get("scope_type") == "workspace"
            and call.get("workspace_id") == "ws-a"
        ]
        assert len(page_calls) == 1

        release.set()
        await pilot.pause(0.3)
        final_state = controller._build_console_workspace_context_state()
        final_node = {node.workspace_id: node for node in final_state.workspace_tree}[
            "ws-a"
        ]
        assert final_node.loading is False
        assert [row.conversation_id for row in final_node.conversations] == ["page-row"]
        assert any(call not in page_calls for call in service_calls)
        assert controller._console_persisted_rows_cache is not None
        assert all(
            row.conversation_id != "page-row"
            for row in controller._console_persisted_rows_cache[0]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (160, 44)])
async def test_switcher_shows_archived_and_restores_as_without_switching(
    size, tmp_path
) -> None:
    from textual.widgets import Checkbox

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-old", name="Client A")
    registry.archive_workspace("ws-old")
    registry.create_workspace(workspace_id="ws-new", name="Client A")
    active_id = registry.get_active_workspace().workspace_id
    host = ConsoleHarness(app)
    async with host.run_test(size=size) as pilot:
        await pilot.pause(0.2)
        notices = []
        app.notify = lambda message, **kwargs: notices.append(str(message))
        modal = await _open_switcher(host, pilot)
        assert "(archived)" not in host.export_screenshot()
        modal.query_one("#console-workspace-show-archived", Checkbox).focus()
        await pilot.press("space")
        await pilot.pause(0.3)
        assert "(archived)" in host.export_screenshot()
        host.save_screenshot(
            filename=f"workspace-switcher-{size[0]}.svg", path=str(tmp_path)
        )
        button = next(b for b in modal.query(Button) if str(b.label) == "Restore as…")
        button.focus()
        await pilot.press("enter")
        await pilot.pause(0.2)
        rename = host.screen
        rename.query_one("#console-workspace-rename-input", Input).value = "Recovered A"
        rename.query_one("#console-workspace-rename-save", Button).press()
        await pilot.pause(0.4)
        assert registry.get_workspace("ws-old").name == "Recovered A"
        assert not registry.get_workspace("ws-old").archived
        assert registry.get_active_workspace().workspace_id == active_id
        assert any("Restored Recovered A" in message for message in notices)



@pytest.mark.asyncio
async def test_archive_receipt_undo_restores_without_activation() -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-receipt", name="Receipt")
    registry.set_active_workspace("ws-receipt")
    host = ConsoleHarness(app)
    async with host.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        modal = await _open_switcher(host, pilot)
        next(b for b in modal.query(Button) if str(b.label) == "Archive").press()
        await pilot.pause(0.2)
        host.screen.query_one("#confirm-button", Button).press()
        await pilot.pause(0.4)
        assert registry.get_workspace("ws-receipt").archived
        assert "View archived" in host.export_screenshot().replace("&#160;", " ")
        host.screen.query_one("#workspace-archive-undo", Button).press()
        await pilot.pause(0.4)
        assert not registry.get_workspace("ws-receipt").archived
        assert registry.get_active_workspace().workspace_id == DEFAULT_WORKSPACE_ID



@pytest.mark.asyncio
async def test_workspace_archive_rechecks_draft_after_confirmation() -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-draft", name="Draft")
    registry.set_active_workspace("ws-draft")
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen
        store = console._ensure_console_chat_store()
        modal = await _open_switcher(host, pilot)
        next(b for b in modal.query(Button) if str(b.label) == "Archive").press()
        await pilot.pause(0.2)
        session = next(s for s in store.sessions() if s.workspace_id == "ws-draft")
        store.set_session_draft(session.id, "Keep this draft")
        host.screen.query_one("#confirm-button", Button).press()
        await pilot.pause(0.3)
        assert not registry.get_workspace("ws-draft").archived
        assert store.session_draft(session.id) == "Keep this draft"
