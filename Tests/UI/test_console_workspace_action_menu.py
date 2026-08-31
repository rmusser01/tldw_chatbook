"""The Workspaces-tree row action menus, driven through the real Console.

TASK-25710. TASK-23200 gave the grouped browser's conversation rows an
asterisk menu and TASK-25709 made it dismissable everywhere; this suite pins
the same pattern on the Workspaces tree: workspace rows open the workspace
action menu, chat rows open the shared conversation menu, the pointer
affordance is the row's trailing asterisk, and the ``m`` binding is the
keyboard equivalent.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.UI.test_console_left_rail import (
    _click_rail_toggle,
    make_console_pilot,
)
from Tests.UI.test_console_workspace_tree_cursor_layout import (
    _console_with_probe_tree,
)
from tldw_chatbook.Widgets.Console.console_conversation_action_menu import (
    ConsoleConversationActionMenu,
)
from tldw_chatbook.Widgets.Console.console_workspace_action_menu import (
    ConsoleWorkspaceActionMenu,
)
from tldw_chatbook.Widgets.Console.console_workspace_tree import (
    WorkspaceTreeMenuRequested,
)


def _stub_registry():
    """A registry resolving the two synthetic seeded workspaces."""

    records = {
        "ws-alpha": SimpleNamespace(
            workspace_id="ws-alpha", name="Workspace Alpha", active=True
        ),
        "ws-beta": SimpleNamespace(
            workspace_id="ws-beta", name="Workspace Beta", active=False
        ),
    }
    return SimpleNamespace(
        get_workspace=lambda ws_id: records.get(ws_id)
    )


def _request_menu(console, *, kind: str, **kwargs) -> None:
    """Drive the screen's tree-menu seam the way the tree's message would."""
    payload = {
        "kind": kind,
        "workspace_id": "ws-beta",
        "screen_x": 4,
        "screen_y": 8,
    }
    payload.update(kwargs)
    console.on_workspace_tree_menu_requested(WorkspaceTreeMenuRequested(**payload))


@pytest.mark.asyncio
async def test_tree_rows_paint_a_trailing_asterisk_affordance() -> None:
    """Workspace and chat rows carry the opener; auxiliary rows do not."""
    async with make_console_pilot(size=(160, 44), production_styles=True) as pilot:
        host = pilot.app
        console, _rail, tree = await _console_with_probe_tree(host, pilot)

        workspace_key = tree.workspace_nodes["ws-alpha"].data.key
        conversation_key = tree.conversation_nodes["conv-a0"].data.key
        assert workspace_key in tree._menu_zones, "workspace row lost its asterisk"
        assert conversation_key in tree._menu_zones, "chat row lost its asterisk"

        start, end = tree._menu_zones[conversation_key]
        assert start < end, "empty asterisk zone"
        # The affordance press is recognized inside the zone and not outside.
        tree._pressed_x = start
        assert tree._pressed_menu_affordance(
            tree.conversation_nodes["conv-a0"].data
        )
        tree._pressed_x = end
        assert not tree._pressed_menu_affordance(
            tree.conversation_nodes["conv-a0"].data
        )


@pytest.mark.asyncio
async def test_workspace_menu_opens_with_the_five_approved_entries(
    monkeypatch,
) -> None:
    async with make_console_pilot(size=(160, 44), production_styles=True) as pilot:
        console = pilot.app.screen
        console.app_instance.workspace_registry_service = _stub_registry()

        _request_menu(console, kind="workspace")
        await pilot.pause(0.4)

        menu = console.query_one(ConsoleWorkspaceActionMenu)
        labels = [str(button.label).strip() for button in menu.query(Button)]
        assert labels == [
            "Activate",
            "New chat",
            "Rename…",
            "RAG scope…",
            "More ▸",
        ]
        # The stub registry reports ws-beta inactive, so Activate is live and
        # RAG scope states its precondition instead of being silently dead.
        actions = {
            getattr(b, "workspace_action_id", ""): b for b in menu.query(Button)
        }
        assert actions["activate"].disabled is False
        assert actions["rag-scope"].disabled is True
        assert actions["rag-scope"].tooltip


@pytest.mark.asyncio
async def test_workspace_menu_pages_and_escapes_like_the_conversation_menu(
    monkeypatch,
) -> None:
    async with make_console_pilot(size=(160, 44), production_styles=True) as pilot:
        console = pilot.app.screen
        console.app_instance.workspace_registry_service = _stub_registry()

        _request_menu(console, kind="workspace")
        await pilot.pause(0.4)

        more = next(
            b
            for b in console.query_one(ConsoleWorkspaceActionMenu).query(Button)
            if getattr(b, "workspace_action_id", "") == "page:more"
        )
        more.press()
        await pilot.pause(0.5)
        menu = console.query_one(ConsoleWorkspaceActionMenu)
        assert menu.page == "more"
        assert [
            getattr(b, "workspace_action_id", "") for b in menu.query(Button)
        ] == ["page:root", "archive"]

        await pilot.press("escape")
        await pilot.pause(0.5)
        assert console.query_one(ConsoleWorkspaceActionMenu).page == "root"
        await pilot.press("escape")
        await pilot.pause(0.5)
        assert not console.query(ConsoleWorkspaceActionMenu)


@pytest.mark.asyncio
async def test_tree_chat_row_opens_the_shared_conversation_menu() -> None:
    async with make_console_pilot(size=(160, 44), production_styles=True) as pilot:
        console = pilot.app.screen

        _request_menu(
            console, kind="conversation", conversation_id="conv-a0"
        )
        await pilot.pause(0.4)

        menu = console.query_one(ConsoleConversationActionMenu)
        labels = [str(button.label).strip() for button in menu.query(Button)]
        assert labels[:2] == ["Favourite", "Change status ▸"]
        assert menu.target.conversation_id == "conv-a0"


@pytest.mark.asyncio
async def test_m_binding_opens_the_menu_for_the_cursor_workspace_row(
    monkeypatch,
) -> None:
    async with make_console_pilot(size=(160, 44), production_styles=True) as pilot:
        host = pilot.app
        console, rail, tree = await _console_with_probe_tree(host, pilot)
        console.app_instance.workspace_registry_service = _stub_registry()

        # The default rail state collapses the Workspaces section, which
        # leaves the tree a 0x0 region and the keyboard anchor meaningless
        # (the section-header click a real user made first is what opens
        # it); click that toggle exactly as the rail suite's helper does.
        await _click_rail_toggle(pilot, "workspace")
        await pilot.pause(0.4)
        assert tree.region, "workspace section stayed collapsed"

        # Walk the cursor onto a workspace row.
        tree.focus()
        for _ in range(12):
            node = tree.cursor_node
            if node is not None and node.data and node.data.kind == "workspace":
                break
            tree.action_cursor_up()
        node = tree.cursor_node
        assert node is not None and node.data.kind == "workspace"

        await pilot.press("m")
        await pilot.pause(0.5)
        assert console.query(ConsoleWorkspaceActionMenu), (
            "the m binding did not open the workspace menu"
        )


@pytest.mark.asyncio
async def test_workspace_menu_dismisses_on_outside_click_and_stranded_escape(
    monkeypatch,
) -> None:
    async with make_console_pilot(size=(160, 44), production_styles=True) as pilot:
        console = pilot.app.screen
        console.app_instance.workspace_registry_service = _stub_registry()

        _request_menu(console, kind="workspace")
        await pilot.pause(0.4)
        assert console.query(ConsoleWorkspaceActionMenu)

        await pilot.click("#console-native-composer")
        await pilot.pause(0.3)
        assert not console.query(ConsoleWorkspaceActionMenu), (
            "an outside click left the workspace menu open"
        )

        _request_menu(console, kind="workspace")
        await pilot.pause(0.4)
        composer = console.query_one("#console-native-composer")
        console.set_focus(composer)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause(0.3)
        assert not console.query(ConsoleWorkspaceActionMenu), (
            "stranded Escape left the workspace menu open"
        )
        assert pilot.app.focused is composer


@pytest.mark.asyncio
async def test_workspace_actions_route_through_the_existing_seams(
    monkeypatch,
) -> None:
    async with make_console_pilot(size=(160, 44), production_styles=True) as pilot:
        console = pilot.app.screen
        calls: list[tuple] = []
        monkeypatch.setattr(
            console._workspace,
            "activate_workspace_id",
            lambda ws_id: calls.append(("activate", ws_id)),
        )
        monkeypatch.setattr(
            console._workspace,
            "_open_console_workspace_rename",
            lambda ws_id: calls.append(("rename", ws_id)),
        )
        monkeypatch.setattr(
            console._workspace,
            "_confirm_console_workspace_archive",
            lambda ws_id: calls.append(("archive", ws_id)),
        )
        async def _fake_scope_picker():
            calls.append(("rag-scope", None))

        monkeypatch.setattr(
            console._workspace,
            "_open_console_workspace_scope_picker",
            _fake_scope_picker,
        )

        async def _fake_create():
            calls.append(("new-chat", None))

        monkeypatch.setattr(
            console._session,
            "_create_native_console_session_from_active_context",
            _fake_create,
        )

        from tldw_chatbook.Chat.console_workspace_actions import (
            ACTION_ACTIVATE,
            ACTION_ARCHIVE,
            ACTION_NEW_CHAT,
            ACTION_RAG_SCOPE,
            ACTION_RENAME,
            WorkspaceMenuTarget,
        )
        from tldw_chatbook.Widgets.Console.console_workspace_action_menu import (
            WorkspaceActionChosen,
        )

        target = WorkspaceMenuTarget(
            workspace_id="ws-beta", name="Workspace Beta", is_active=False
        )
        for action in (
            ACTION_ACTIVATE,
            ACTION_NEW_CHAT,
            ACTION_RENAME,
            ACTION_RAG_SCOPE,
            ACTION_ARCHIVE,
        ):
            console.on_workspace_action_chosen(
                WorkspaceActionChosen(action, target)
            )
        await pilot.pause(0.8)

        routed = {name for name, _ in calls}
        assert routed == {"activate", "new-chat", "rename", "rag-scope", "archive"}
        assert ("activate", "ws-beta") in calls
        assert ("rename", "ws-beta") in calls
        assert ("archive", "ws-beta") in calls


@pytest.mark.asyncio
async def test_contextual_star_button_and_selection_line_are_retired() -> None:
    """The single Star control is gone; row menus own the actions now."""
    async with make_console_pilot(size=(160, 44), production_styles=True) as pilot:
        console = pilot.app.screen
        await _console_with_probe_tree(pilot.app, pilot)

        assert not console.query("#console-workspace-tree-star")
        assert not console.query("#console-workspace-tree-selection-context")
        assert not console.query("#console-workspace-context-action-row")
