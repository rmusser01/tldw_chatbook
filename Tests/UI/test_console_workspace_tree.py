"""Native Textual Tree contracts for the Console Workspaces section."""

from __future__ import annotations

from dataclasses import replace

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Input, Tree

import tldw_chatbook.Widgets.Console.console_workspace_tree as tree_module
from tldw_chatbook.Widgets.Console.console_workspace_tree import (
    ConsoleWorkspaceTree,
    WorkspaceTreeFocusRecoveryRequested,
    WorkspaceTreeConversationSelected,
    WorkspaceTreeNodeData,
    WorkspaceTreeStarRequested,
    WorkspaceTreeWorkspaceSelected,
)
from tldw_chatbook.Workspaces.workspace_tree_state import (
    WorkspaceTreeConversation,
    WorkspaceTreeWorkspace,
)


def _workspace(
    workspace_id: str,
    label: str,
    *conversations: tuple[str, str],
    loading: bool = False,
    error: str = "",
    next_cursor: str | None = None,
) -> WorkspaceTreeWorkspace:
    return WorkspaceTreeWorkspace(
        workspace_id=workspace_id,
        label=label,
        conversations=tuple(
            WorkspaceTreeConversation(
                conversation_id=conversation_id,
                title=title,
                starred=False,
                updated_sort=index,
                selected=False,
                run_marker="",
            )
            for index, (conversation_id, title) in enumerate(conversations)
        ),
        loading=loading,
        error=error,
        next_cursor=next_cursor,
    )


class _TreeHarness(App[None]):
    CSS = """
    Screen { layout: vertical; }
    ConsoleWorkspaceTree { height: 8; width: 32; }
    Input { height: 1; }
    """

    def __init__(self, tree: ConsoleWorkspaceTree) -> None:
        super().__init__()
        self.workspace_tree = tree
        self.messages: list[object] = []

    def compose(self) -> ComposeResult:
        yield Input(id="workspace-search")
        yield self.workspace_tree

    def on_workspace_tree_workspace_selected(
        self, event: WorkspaceTreeWorkspaceSelected
    ) -> None:
        self.messages.append(event)

    def on_workspace_tree_conversation_selected(
        self, event: WorkspaceTreeConversationSelected
    ) -> None:
        self.messages.append(event)

    def on_workspace_tree_star_requested(
        self, event: WorkspaceTreeStarRequested
    ) -> None:
        self.messages.append(event)

    def on_workspace_tree_focus_recovery_requested(
        self, event: WorkspaceTreeFocusRecoveryRequested
    ) -> None:
        self.messages.append(event)


def _tree() -> ConsoleWorkspaceTree:
    tree = ConsoleWorkspaceTree(id="console-workspace-tree")
    tree.sync_projection(
        (
            _workspace("w1", "One", ("c1", "First"), ("c2", "Second")),
            _workspace("w2", "Two", ("c3", "Third")),
        ),
        expanded_workspace_ids={"w1"},
    )
    return tree


def test_native_configuration_and_literal_unicode_labels() -> None:
    tree = ConsoleWorkspaceTree()
    raw_workspace = "[bold]研究 👩🏽‍💻\nignored"
    raw_conversation = "[red]会話 🧪\nignored"
    tree.sync_projection(
        (_workspace("w", raw_workspace, ("c", raw_conversation)),),
        expanded_workspace_ids={"w"},
    )

    assert isinstance(tree, Tree)
    assert tree.show_root is False
    assert tree.auto_expand is False
    assert tree.guide_depth == 2
    assert tree.root.is_expanded is True
    assert tree.root.allow_expand is False
    assert tree.ICON_NODE and tree.ICON_NODE_EXPANDED
    assert all(len(glyphs) == 4 for glyphs in tree.LINES.values())
    assert tree.workspace_nodes["w"].label.plain == "[bold]研究 👩🏽‍💻"
    assert tree.conversation_nodes["c"].label.plain.endswith("[red]会話 🧪")
    assert isinstance(tree.workspace_nodes["w"].label, Text)
    assert tree.workspace_nodes["w"].data.raw_label == raw_workspace


def test_ascii_mode_uses_only_ascii_tree_icons_and_guides(monkeypatch) -> None:
    monkeypatch.setattr(tree_module, "ascii_glyph_mode", lambda: True)
    tree = ConsoleWorkspaceTree()

    assert tree.ICON_NODE == "> "
    assert tree.ICON_NODE_EXPANDED == "v "
    assert all(
        glyph.isascii() for guide_set in tree.LINES.values() for glyph in guide_set
    )


@pytest.mark.asyncio
async def test_enter_selects_without_expanding_and_space_toggles() -> None:
    tree = _tree()
    app = _TreeHarness(tree)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        workspace = tree.workspace_nodes["w2"]
        tree.move_cursor(workspace)
        tree.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert workspace.is_collapsed
        assert isinstance(app.messages[-1], WorkspaceTreeWorkspaceSelected)
        assert app.messages[-1].workspace_id == "w2"

        await pilot.press("space")
        await pilot.pause()
        assert workspace.is_expanded


@pytest.mark.asyncio
async def test_native_pointer_disclosure_toggles_and_label_selects() -> None:
    tree = _tree()
    app = _TreeHarness(tree)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        workspace = tree.workspace_nodes["w2"]

        assert workspace.is_collapsed
        assert await pilot.click(tree, offset=(0, 3))
        await pilot.pause()
        assert workspace.is_expanded

        assert await pilot.click(tree, offset=(4, 3))
        await pilot.pause()
        assert isinstance(app.messages[-1], WorkspaceTreeWorkspaceSelected)
        assert app.messages[-1].workspace_id == "w2"


@pytest.mark.asyncio
async def test_long_literal_label_renders_on_one_ellipsized_physical_row() -> None:
    tree = ConsoleWorkspaceTree()
    raw = "[bold]" + "界🙂" * 30 + "\nsecond physical row"
    tree.sync_projection(
        (_workspace("w", raw),),
        expanded_workspace_ids=set(),
    )
    app = _TreeHarness(tree)
    async with app.run_test(size=(24, 12)) as pilot:
        await pilot.pause()
        rendered = tree.render_line(0).text

        assert "[bold]" in rendered
        assert "second physical row" not in rendered
        assert "…" in rendered


@pytest.mark.asyncio
async def test_status_nodes_are_inert_and_tree_tabs_when_not_overflowing() -> None:
    tree = ConsoleWorkspaceTree()
    tree.sync_projection(
        (_workspace("w", "One", loading=True, error="offline", next_cursor="p2"),),
        expanded_workspace_ids={"w"},
    )
    app = _TreeHarness(tree)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert tree.max_scroll_y == 0
        assert tree.can_focus is True
        tree.focus()
        status = next(iter(tree.auxiliary_nodes.values()))
        tree.move_cursor(status)
        await pilot.press("enter", "space", "s")
        await pilot.pause()
        assert app.messages == []


@pytest.mark.asyncio
async def test_plain_and_shift_navigation_never_reaches_hidden_root_or_none(
    monkeypatch,
) -> None:
    tree = _tree()
    app = _TreeHarness(tree)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        first = tree.workspace_nodes["w1"]
        leaf = tree.conversation_nodes["c1"]
        tree.move_cursor(first)
        tree.focus()
        await pilot.press("left", "shift+left", "shift+up")
        assert tree.cursor_node is first
        requested_nodes = []
        original_move_cursor = tree.move_cursor

        def record_move(node, *args, **kwargs):
            requested_nodes.append(node)
            return original_move_cursor(node, *args, **kwargs)

        monkeypatch.setattr(tree, "move_cursor", record_move)
        tree.action_cursor_parent()
        tree.action_cursor_previous_sibling()
        assert tree.cursor_node is first
        assert tree.root not in requested_nodes

        first.expand()
        await pilot.pause()
        tree.move_cursor(leaf)
        await pilot.press("right")
        assert tree.cursor_node is leaf

        last = tree.workspace_nodes["w2"]
        tree.move_cursor(last)
        await pilot.press("shift+right", "shift+down")
        assert tree.cursor_node is last


@pytest.mark.asyncio
async def test_right_on_expanded_workspace_moves_to_first_selectable_child() -> None:
    tree = _tree()
    app = _TreeHarness(tree)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        workspace = tree.workspace_nodes["w1"]
        tree.move_cursor(workspace)
        tree.focus()

        await pilot.press("right")

        assert tree.cursor_node is tree.conversation_nodes["c1"]


@pytest.mark.asyncio
async def test_removal_cursor_fallback_uses_logical_neighbors_before_hidden_root() -> (
    None
):
    tree = _tree()
    app = _TreeHarness(tree)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        tree.move_cursor(tree.conversation_nodes["c1"])

        tree.sync_projection(
            (
                _workspace("w1", "One", ("c2", "Second")),
                _workspace("w2", "Two", ("c3", "Third")),
            ),
            expanded_workspace_ids={"w1"},
        )
        assert tree.cursor_node is tree.conversation_nodes["c2"]

        tree.sync_projection(
            (
                _workspace("w1", "One"),
                _workspace("w2", "Two", ("c3", "Third")),
            ),
            expanded_workspace_ids={"w1"},
        )
        assert tree.cursor_node is tree.workspace_nodes["w1"]

        tree.sync_projection(
            (_workspace("w2", "Two", ("c3", "Third")),),
            expanded_workspace_ids=set(),
        )
        assert tree.cursor_node is tree.workspace_nodes["w2"]
        assert tree.cursor_node is not tree.root


@pytest.mark.asyncio
async def test_collapsing_workspace_moves_descendant_cursor_to_workspace() -> None:
    tree = _tree()
    app = _TreeHarness(tree)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        tree.move_cursor(tree.conversation_nodes["c2"])
        tree.focus()
        tree.workspace_nodes["w1"].collapse()
        await pilot.pause()
        assert tree.cursor_node is tree.workspace_nodes["w1"]
        assert tree.workspace_nodes["w1"].is_collapsed


@pytest.mark.asyncio
async def test_contextual_star_only_posts_for_focused_conversation() -> None:
    tree = _tree()
    app = _TreeHarness(tree)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        tree.move_cursor(tree.conversation_nodes["c1"])
        tree.focus()
        await pilot.press("s")
        await pilot.pause()
        assert isinstance(app.messages[-1], WorkspaceTreeStarRequested)
        assert app.messages[-1].conversation_id == "c1"

        message_count = len(app.messages)
        app.query_one("#workspace-search", Input).focus()
        await pilot.press("s")
        await pilot.pause()
        assert len(app.messages) == message_count


@pytest.mark.asyncio
async def test_keyed_sync_preserves_identity_cursor_and_registration_across_move() -> (
    None
):
    tree = _tree()
    app = _TreeHarness(tree)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        node = tree.conversation_nodes["c1"]
        node_id = node.id
        tree.move_cursor(node)

        tree.sync_projection(
            (
                _workspace("w2", "Two", ("c1", "Renamed"), ("c3", "Third")),
                _workspace("w1", "One", ("c2", "Second")),
            ),
            expanded_workspace_ids={"w1", "w2"},
        )
        await pilot.pause()

        assert tree.conversation_nodes["c1"] is node
        assert node.parent is tree.workspace_nodes["w2"]
        assert node.label.plain.endswith("Renamed")
        assert tree.get_node_by_id(node_id) is node
        assert tree.cursor_node is node


def test_private_move_fails_closed_when_textual_shape_is_not_exact(monkeypatch) -> None:
    tree = _tree()
    node = tree.conversation_nodes["c1"]
    monkeypatch.delattr(tree, "_tree_nodes")

    with pytest.raises(RuntimeError, match="Textual 8.2.8"):
        tree._move_node_preserving_identity(
            node,
            tree.workspace_nodes["w2"],
            0,
        )


def test_private_move_fails_closed_on_textual_version_mismatch_without_mutation(
    monkeypatch,
) -> None:
    tree = _tree()
    node = tree.conversation_nodes["c1"]
    original_parent = node.parent
    original_children = tuple(original_parent.children)
    invalidations: list[None] = []
    monkeypatch.setattr(tree_module.textual, "__version__", "8.2.9")
    monkeypatch.setattr(tree, "_invalidate", lambda: invalidations.append(None))

    with pytest.raises(RuntimeError, match="Textual 8.2.8"):
        tree._move_node_preserving_identity(node, tree.workspace_nodes["w2"], 0)

    assert node.parent is original_parent
    assert tuple(original_parent.children) == original_children
    assert tree.get_node_by_id(node.id) is node
    assert invalidations == []


def test_private_cross_parent_move_preserves_identity_with_one_invalidation(
    monkeypatch,
) -> None:
    tree = _tree()
    node = tree.conversation_nodes["c1"]
    node_id = node.id
    invalidations: list[None] = []
    monkeypatch.setattr(tree, "_invalidate", lambda: invalidations.append(None))

    tree._move_node_preserving_identity(node, tree.workspace_nodes["w2"], 0)

    assert node is tree.conversation_nodes["c1"]
    assert node.parent is tree.workspace_nodes["w2"]
    assert tree.get_node_by_id(node_id) is node
    assert invalidations == [None]


def test_private_same_parent_move_treats_index_as_final_position(monkeypatch) -> None:
    tree = _tree()
    parent = tree.workspace_nodes["w1"]
    first = tree.conversation_nodes["c1"]
    second = parent.children[1]
    invalidations: list[None] = []
    monkeypatch.setattr(tree, "_invalidate", lambda: invalidations.append(None))

    tree._move_node_preserving_identity(first, parent, 1)

    assert tuple(parent.children[:2]) == (second, first)
    assert invalidations == [None]


@pytest.mark.asyncio
async def test_removing_final_selectable_node_requests_owning_section_recovery() -> (
    None
):
    tree = ConsoleWorkspaceTree()
    app = _TreeHarness(tree)
    async with app.run_test(size=(60, 20)) as pilot:
        tree.sync_projection((_workspace("w1", "One"),), expanded_workspace_ids=set())
        await pilot.pause()
        tree.move_cursor(tree.workspace_nodes["w1"])
        tree.focus()

        tree.sync_projection((), expanded_workspace_ids=set())
        await pilot.pause()

        assert any(
            isinstance(message, WorkspaceTreeFocusRecoveryRequested)
            for message in app.messages
        )
        assert tree.can_focus is False


def test_search_expansion_snapshot_restores_exactly_without_persistence_messages() -> (
    None
):
    tree = _tree()
    persisted: list[frozenset[str]] = []
    tree.expansion_preferences_changed = persisted.append

    tree.set_search_active(True, forced_workspace_ids={"w2"})
    tree.workspace_nodes["w1"].collapse()
    tree.workspace_nodes["w2"].expand()
    tree._record_expansion_gesture(tree.workspace_nodes["w1"])
    tree.set_search_active(False)

    assert tree.workspace_nodes["w1"].is_expanded
    assert tree.workspace_nodes["w2"].is_collapsed
    assert persisted == []


def test_search_clear_restores_old_snapshot_and_seeds_new_workspaces_expanded() -> None:
    tree = _tree()
    tree.set_search_active(True, forced_workspace_ids={"w2"})
    tree.sync_projection(
        (
            _workspace("w1", "One", ("c1", "First")),
            _workspace("w2", "Two", ("c3", "Third")),
            _workspace("w3", "New", ("c4", "Fourth")),
        ),
        expanded_workspace_ids={"w1"},
    )

    tree.set_search_active(False)

    assert tree.workspace_nodes["w1"].is_expanded
    assert tree.workspace_nodes["w2"].is_collapsed
    assert tree.workspace_nodes["w3"].is_expanded
    assert "w3" in tree.preferred_expanded_workspace_ids


@pytest.mark.asyncio
async def test_passive_keyed_sync_preserves_local_scroll_offset_and_active_marker() -> (
    None
):
    tree = ConsoleWorkspaceTree()
    conversations = tuple((f"c{index}", f"Conversation {index}") for index in range(30))
    tree.sync_projection(
        (_workspace("w", "Workspace", *conversations),),
        expanded_workspace_ids={"w"},
    )
    app = _TreeHarness(tree)
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        tree.scroll_y = 8
        await pilot.pause()
        offset = tree.scroll_y
        active = _workspace("w", "Renamed", *reversed(conversations))
        active = replace(active, active=True)
        tree.sync_projection((active,), expanded_workspace_ids={"w"})
        await pilot.pause()

        assert tree.scroll_y == offset
        assert tree.workspace_nodes["w"].label.plain.startswith("● ")


def test_true_deletion_uses_public_removal_and_drops_registration() -> None:
    tree = _tree()
    node = tree.conversation_nodes["c1"]
    node_id = node.id
    tree.sync_projection(
        (_workspace("w1", "One", ("c2", "Second")),),
        expanded_workspace_ids={"w1"},
    )

    assert "c1" not in tree.conversation_nodes
    with pytest.raises(Exception):
        tree.get_node_by_id(node_id)


def test_tree_disables_focus_only_when_it_has_no_actionable_nodes() -> None:
    tree = ConsoleWorkspaceTree()
    tree.sync_projection((), expanded_workspace_ids=set())
    assert tree.can_focus is False
    tree.sync_projection(
        (_workspace("w", "One"),),
        expanded_workspace_ids=set(),
    )
    assert tree.can_focus is True


def test_cursor_and_hover_publish_full_raw_tooltip() -> None:
    tree = ConsoleWorkspaceTree()
    raw = "A very long [bold] literal workspace identity"
    tree.sync_projection(
        (_workspace("w", raw),),
        expanded_workspace_ids=set(),
    )
    node = tree.workspace_nodes["w"]
    node._line = 0
    tree.cursor_line = 0
    tree.watch_cursor_line(-1, 0)
    assert tree.tooltip == raw


def test_node_data_kinds_are_explicit() -> None:
    data = WorkspaceTreeNodeData.workspace("w", "Work")
    assert data.kind == "workspace"
    assert data.workspace_id == "w"
    assert data.selectable is True
