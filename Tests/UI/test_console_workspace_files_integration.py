"""Production-shaped integration contracts for Console Workspace Files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from textual.css.query import NoMatches
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunState
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


def _scratch_tree_fingerprint(root: Path) -> tuple[tuple[str, str, str], ...]:
    """Return a recursive, content-safe fingerprint without file bodies."""
    if not root.exists():
        return ()
    entries: list[tuple[str, str, str]] = []
    for item in sorted(root.rglob("*")):
        relative = str(item.relative_to(root))
        if item.is_symlink():
            entries.append((relative, "symlink", os.readlink(item)))
        elif item.is_dir():
            entries.append((relative, "directory", ""))
        elif item.is_file():
            entries.append(
                (relative, "file", hashlib.sha256(item.read_bytes()).hexdigest())
            )
        else:
            entries.append((relative, "other", ""))
    return tuple(entries)


def _workspace_roots_fingerprint(
    active_root: Path, other_root: Path
) -> tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...]:
    """Fingerprint both named workspace roots before and after a read-only visit."""
    return (
        ("active", _scratch_tree_fingerprint(active_root)),
        ("other", _scratch_tree_fingerprint(other_root)),
    )


def _redirected_scratch_fingerprint(app) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """Fingerprint only the per-test profile/config/data and registry database roots."""
    roots = {
        "home": Path(os.environ["HOME"]),
        "config": Path(os.environ["XDG_CONFIG_HOME"]),
        "data": Path(os.environ["XDG_DATA_HOME"]),
        "registry": app.local_workspace_db.db_path.parent,
    }
    return tuple(
        (name, _scratch_tree_fingerprint(root)) for name, root in sorted(roots.items())
    )


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
            try:
                screen.query_one("#console-workspace-files-back", Button)
            except NoMatches:
                pass
            else:
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


async def _seed_ready_console_transcript(console) -> None:
    """Avoid the unrelated first-run setup overlay in real click evidence."""
    store = console._ensure_console_chat_store()
    session = store.ensure_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Task 26042 scratch setup message",
    )
    await console._sync_native_console_chat_ui()
    console._sync_console_transcript_guidance()


async def _open_navigate_and_dismiss(host, pilot, button: Button) -> ConsoleWorkspaceFilesModal:
    button.focus()
    button.press()
    modal = await _wait_for_files_modal(host, pilot)
    await pilot.click("#console-workspace-files-details")
    await pilot.pause()
    await pilot.click("#console-workspace-files-back")
    await pilot.pause()
    return modal


async def _wait_for_available_files_button(
    console, pilot, *, workspace_id: str, grouped: bool
) -> Button:
    """Wait for the real off-loop binding snapshot to enable one typed route."""
    for _ in range(100):
        buttons = (
            console.query(".console-workspace-group-files")
            if grouped
            else console.query("#console-workspace-files-open")
        )
        button = next(
            (
                candidate
                for candidate in buttons
                if getattr(candidate, "workspace_id", None) == workspace_id
                and getattr(candidate, "workspace_files_expected_available", False)
            ),
            None,
        )
        if button is not None:
            return button
        await pilot.pause(0.02)
    raise AssertionError(f"Files route for {workspace_id!r} did not become available")


async def _reveal_and_click_files_button(console, pilot, button: Button) -> None:
    """Scroll the real rail until a Files control is genuinely mouse-hittable."""
    button = console.query_one(f"#{button.id}", Button)
    rail_body = console.query_one("#console-left-rail-body")
    rail_body.scroll_to_widget(button, animate=False)
    await pilot.pause()
    button = console.query_one(f"#{button.id}", Button)
    widget_at_center, _ = console.screen.get_widget_at(
        button.region.x + button.region.width // 2,
        button.region.y + button.region.height // 2,
    )
    assert widget_at_center is button
    assert await pilot.click(button)


async def _ensure_workspace_rail_open(console, pilot) -> None:
    """Use the real collapsed-rail control when the host starts it closed."""
    left_rail = console.query_one("#console-left-rail")
    if left_rail.region.width:
        return
    opener = console.query_one("#console-context-rail-open", Button)
    assert opener.region.width > 0
    assert await pilot.click(opener)
    for _ in range(100):
        left_rail = console.query_one("#console-left-rail")
        if left_rail.region.width:
            return
        await pilot.pause(0.02)
    raise AssertionError("Context rail did not open through its visible control")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((80, 24), (100, 30), (120, 40), (160, 50)))
async def test_shipped_css_four_size_routes_keep_read_only_console_fingerprint(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """Actual clicks cover both routes, escaped filtering, paging, and safe return."""
    app = _build_test_app(
        config_overrides={
            "chat_defaults": {"provider": "openai", "model": "gpt-test"},
            "api_settings": {"openai": {"api_key": "task-26042-scratch-key"}},
        }
    )
    service = app.workspace_registry_service
    active_root = tmp_path / "active-root"
    other_root = tmp_path / "other-root"
    _install_named_workspace_with_folder(service, "ws-active", "Active [workspace]", active_root)
    _install_named_workspace_with_folder(service, "ws-other", "Other [workspace]", other_root)
    hostile_name = "literal[needle]\n\u202e.txt"
    (active_root / hostile_name).write_text("hostile label only", encoding="utf-8")
    large_payload = "é" * 100_000 + "🙂" * 100_001
    (active_root / "large.txt").write_text(large_payload, encoding="utf-8")
    (active_root / "nested").mkdir()
    (active_root / "nested" / "unchanged.txt").write_text("nested active", encoding="utf-8")
    (other_root / "nested").mkdir()
    (other_root / "nested" / "unchanged.txt").write_text("nested other", encoding="utf-8")
    service.set_active_workspace("ws-active")
    initial_bindings = service.list_runtime_bindings("ws-active")
    initial_workspace_trees = _workspace_roots_fingerprint(active_root, other_root)
    host = _StyledConsoleHarness(app)

    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-files-open")
        await _seed_ready_console_transcript(console)
        await pilot.pause()
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("unchanged scratch draft")
        composer.set_pending_attachment_label("scratch-attachment.txt · 1 KB")
        console._set_console_pending_approval({"approval_id": "scratch-approval", "body": "private"})
        if size[0] >= 84:
            await _ensure_workspace_rail_open(console, pilot)
            active_button = await _wait_for_available_files_button(
                console, pilot, workspace_id="ws-active", grouped=False
            )
        else:
            active_button = None
        before = _console_fingerprint(console, app)
        scratch_before = _redirected_scratch_fingerprint(app)

        if size[0] < 84:
            # The established Console single-pane contract suppresses rails
            # and their handles below 84 columns to keep the transcript usable.
            # Exercise the same typed admission seam directly, then prove the
            # full-screen compact modal rather than pretending a hidden rail is
            # a live input route.
            assert console.query_one("#console-left-rail").display is False
            assert console.query_one("#console-context-rail-handle").display is False
            await console._workspace.request_workspace_files(
                "ws-active", expected_available=True
            )
        else:
            assert active_button is not None
            await _reveal_and_click_files_button(console, pilot, active_button)
        modal = await _wait_for_files_modal(host, pilot)
        assert modal.inspected_workspace_id == "ws-active"
        assert modal.query_one("#console-workspace-files-back", Button).can_focus
        assert modal.query_one("#console-workspace-files-details", Button).can_focus
        assert modal.region.width <= size[0]
        assert modal.region.height <= size[1]

        large_button = None
        for _ in range(100):
            large_button = next(
                (
                    entry
                    for entry in modal.query(".console-workspace-files-entry")
                    if "large.txt" in str(entry.label)
                ),
                None,
            )
            if large_button is not None:
                break
            await pilot.pause(0.02)
        assert large_button is not None
        await pilot.click(large_button)
        for _ in range(100):
            if modal.state.file_result is not None:
                break
            await pilot.pause(0.02)
        assert modal.state.file_result is not None
        assert modal.state.file_result.character_range == (0, 100_000)
        await pilot.click("#console-workspace-files-next")
        for _ in range(100):
            if modal.state.file_result and modal.state.file_result.character_range == (100_000, 200_000):
                break
            await pilot.pause(0.02)
        assert modal.state.file_result is not None
        assert modal.state.file_result.character_range == (100_000, 200_000)
        assert modal.state.file_result.text.startswith("🙂")
        assert modal.state.file_result.next_page_offset == 200_000

        await pilot.click("#console-workspace-files-filter")
        await pilot.press("l", "i", "t", "e", "r", "a", "l", "[")
        await pilot.press("enter")
        for _ in range(100):
            if modal.state.filter_result is not None:
                break
            await pilot.pause(0.02)
        assert modal.state.filter_result is not None
        assert modal.state.filter_result.matches
        assert "\\x1b" not in str(modal.state.filter_result.matches[0].display_path)
        assert "\\n" in str(modal.state.filter_result.matches[0].display_path)
        assert str(active_root) not in str(modal.state.filter_result.matches[0].display_path)
        await pilot.click("#console-workspace-files-back")
        await pilot.pause()
        assert modal.owned_lane_count == 0
        assert _console_fingerprint(console, app) == before

        if size[0] >= 84:
            grouped_button = await _wait_for_available_files_button(
                console, pilot, workspace_id="ws-other", grouped=True
            )
            await _reveal_and_click_files_button(console, pilot, grouped_button)
            grouped_modal = await _wait_for_files_modal(host, pilot)
            assert grouped_modal.inspected_workspace_id == "ws-other"
            await pilot.click("#console-workspace-files-back")
            await pilot.pause()

        assert _console_fingerprint(console, app) == before
        assert _redirected_scratch_fingerprint(app) == scratch_before
        assert service.list_runtime_bindings("ws-active") == initial_bindings
        assert _workspace_roots_fingerprint(active_root, other_root) == initial_workspace_trees


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
