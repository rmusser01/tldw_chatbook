"""Production-shaped integration contracts for Console Workspace Files."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.Chat.console_fleet_attention import set_fleet_unseen_completion
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_workspace_files_modal import (
    ConsoleWorkspaceFilesModal,
    WorkspaceFilesAttention,
)


ROOT = Path(__file__).resolve().parents[2]


class _StyledConsoleHarness(ConsoleHarness):
    """Use the exact shipped CSS stack, not widget default CSS."""

    CSS_PATH = str(ROOT / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss")


def _install_named_workspace_with_folder(service, workspace_id: str, name: str, root: Path) -> None:
    service.create_workspace(workspace_id=workspace_id, name=name)
    root.mkdir()
    (root / "safe.txt").write_text("safe file", encoding="utf-8")
    service.add_folder_binding(workspace_id, root)
    service.link_membership(
        workspace_id,
        item_type="conversation",
        item_id=f"conversation-{workspace_id}",
        role="workspace-thread",
        title=f"Conversation for {name}",
    )


async def _wait_for_files_modal(host, pilot) -> ConsoleWorkspaceFilesModal:
    for _ in range(100):
        screen = host.screen_stack[-1]
        if isinstance(screen, ConsoleWorkspaceFilesModal):
            return screen
        await pilot.pause(0.02)
    raise AssertionError("Workspace Files modal did not open")


def _console_fingerprint(console, app) -> tuple[object, ...]:
    """Capture all Console state this non-activating surface must preserve."""
    store = console._ensure_console_chat_store()
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    context = console._workspace._build_console_workspace_context_state()
    selections = tuple(
        (row.row_key, row.selected)
        for section in (context.conversation_browser.sections if context.conversation_browser else ())
        for group in section.groups
        for row in group.rows
    )
    active = app.workspace_registry_service.get_active_workspace()
    return (
        active.workspace_id if active else None,
        store.active_session_id,
        console._current_console_conversation_id(),
        composer.draft_text,
        composer._pending_attachment_label,
        console._task_resume_state.pending_approval,
        console._pending_console_launch_context,
        selections,
    )


async def _open_navigate_and_dismiss(host, pilot, button: Button) -> ConsoleWorkspaceFilesModal:
    button.focus()
    button.press()
    modal = await _wait_for_files_modal(host, pilot)
    await pilot.click("#console-workspace-files-details")
    await pilot.pause()
    await pilot.click("#console-workspace-files-back")
    await pilot.pause()
    return modal


@pytest.mark.asyncio
async def test_default_files_request_is_typed_and_preserves_console_state() -> None:
    """The Default action remains focusable and never infers identity from copy."""
    app = _build_test_app()
    host = _StyledConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-files-open")
        button = console.query_one("#console-workspace-files-open", Button)
        before = app.workspace_registry_service.get_active_workspace().workspace_id
        assert button.disabled is False
        assert button.workspace_id == before
        button.press()
        await pilot.pause()
        assert app.workspace_registry_service.get_active_workspace().workspace_id == before


@pytest.mark.asyncio
async def test_files_action_refuses_below_minimum_without_context_mutation() -> None:
    app = _build_test_app()
    host = _StyledConsoleHarness(app)
    async with host.run_test(size=(79, 23)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-files-open")
        before = app.workspace_registry_service.get_active_workspace().workspace_id
        console.query_one("#console-workspace-files-open", Button).press()
        await pilot.pause()
        assert app.workspace_registry_service.get_active_workspace().workspace_id == before


@pytest.mark.asyncio
async def test_two_named_workspace_files_routes_preserve_complete_console_fingerprint(
    tmp_path: Path,
) -> None:
    """Both real Files entry points inspect without activating or mutating Console."""
    app = _build_test_app()
    service = app.workspace_registry_service
    _install_named_workspace_with_folder(
        service, "ws-active", "Active named workspace", tmp_path / "active"
    )
    _install_named_workspace_with_folder(
        service, "ws-other", "Other named workspace", tmp_path / "other"
    )
    service.set_active_workspace("ws-active")
    host = _StyledConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-files-open")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("draft remains exactly here")
        composer.set_pending_attachment_label("evidence.txt · 1 KB")
        console._set_console_pending_approval(
            {"approval_id": "approval-immutable", "body": "approval body stays local"}
        )
        before = _console_fingerprint(console, app)

        active_button = console.query_one("#console-workspace-files-open", Button)
        assert active_button.workspace_id == "ws-active"
        active_modal = await _open_navigate_and_dismiss(host, pilot, active_button)
        assert active_modal.inspected_workspace_id == "ws-active"
        assert app.workspace_registry_service.get_active_workspace().workspace_id == "ws-active"
        assert _console_fingerprint(console, app) == before

        group_button = next(
            button
            for button in console.query(".console-workspace-group-files")
            if getattr(button, "workspace_id", None) == "ws-other"
        )
        grouped_modal = await _open_navigate_and_dismiss(host, pilot, group_button)
        assert grouped_modal.inspected_workspace_id == "ws-other"
        assert app.workspace_registry_service.get_active_workspace().workspace_id == "ws-active"
        assert _console_fingerprint(console, app) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["active", "grouped"])
async def test_typed_stale_files_routes_open_pinned_empty_recovery_without_activation(
    tmp_path: Path, route: str
) -> None:
    """A render-time available expectation distinguishes stale from no-folder clicks."""
    app = _build_test_app()
    service = app.workspace_registry_service
    _install_named_workspace_with_folder(
        service, "ws-active", "Active named workspace", tmp_path / "active"
    )
    _install_named_workspace_with_folder(
        service, "ws-other", "Other named workspace", tmp_path / "other"
    )
    service.set_active_workspace("ws-active")
    host = _StyledConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-files-open")
        button = (
            console.query_one("#console-workspace-files-open", Button)
            if route == "active"
            else next(
                candidate
                for candidate in console.query(".console-workspace-group-files")
                if getattr(candidate, "workspace_id", None) == "ws-other"
            )
        )
        requested_id = button.workspace_id
        assert button.workspace_files_expected_available is True
        binding = service.list_folder_bindings(requested_id)[0]
        service.remove_runtime_binding(binding.binding_id)
        before = _console_fingerprint(console, app)

        button.press()
        modal = await _wait_for_files_modal(host, pilot)
        assert modal.inspected_workspace_id == requested_id
        assert modal.state.selected_binding_id is None
        assert modal.state.status_copy == "No local folders are attached. Add one in Settings."
        await pilot.click("#console-workspace-files-back")
        await pilot.pause()
        assert app.workspace_registry_service.get_active_workspace().workspace_id == "ws-active"
        assert _console_fingerprint(console, app) == before


@pytest.mark.asyncio
async def test_chat_screen_sync_publishes_private_monotonic_attention_without_resolution(
    tmp_path: Path,
) -> None:
    """Actual approval/run/fleet producers update the mounted modal generically."""
    app = _build_test_app()
    service = app.workspace_registry_service
    app.conversation_local_marks_service = ConversationLocalMarksService(
        CharactersRAGDB(str(tmp_path / "marks.sqlite"), client_id="workspace-files")
    )
    _install_named_workspace_with_folder(
        service, "ws-active", "Active named workspace", tmp_path / "active"
    )
    service.set_active_workspace("ws-active")
    host = _StyledConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-files-open")
        console.query_one("#console-workspace-files-open", Button).press()
        modal = await _wait_for_files_modal(host, pilot)
        approval = {
            "approval_id": "approval-private",
            "body": "never render this body",
            "path": "/private/secret.txt",
            "tool_args": {"danger": "never render"},
            "error": "never render this error",
        }
        console._set_console_pending_approval(approval)
        controller = console._ensure_console_chat_controller()
        controller._set_run_state(ConsoleRunState.blocked("raw blocked error must stay private"))
        set_fleet_unseen_completion(app, "conversation-fleet-private")
        console._sync_console_workspace_context()
        await pilot.pause()

        attention = modal._attention
        assert attention.pending_approval_count == 1
        assert attention.has_blocked_activity is True
        assert attention.has_new_activity is True
        generation = console._workspace._workspace_files_attention_generation
        assert modal.update_attention(
            WorkspaceFilesAttention("body /private tool_args error"), generation
        ) is False
        visible = str(modal.query_one("#console-workspace-files-attention").renderable)
        assert visible == "Console needs attention · 1 approval waiting"
        assert all(
            forbidden not in visible
            for forbidden in ("body", "/private", "tool_args", "error", "secret")
        )
        assert console._task_resume_state.pending_approval is approval
        await pilot.click("#console-workspace-files-back")
