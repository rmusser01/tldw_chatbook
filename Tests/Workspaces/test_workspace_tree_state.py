"""Pure ownership and ordering contracts for the Console workspace Tree."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
    build_console_conversation_browser_state,
)
from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID
from tldw_chatbook.Workspaces.workspace_tree_state import (
    WorkspaceTreeConversation,
    WorkspaceTreeWorkspace,
    build_workspace_tree_state,
    update_workspace_tree_conversation,
)


def _row(
    conversation_id: str,
    *,
    workspace_id: str | None,
    title: str,
    starred: bool = False,
    updated_sort: str = "",
    selected: bool = False,
    run_marker: str = "",
    scope_type: str = "workspace",
) -> ConsoleConversationBrowserInputRow:
    return ConsoleConversationBrowserInputRow(
        row_key=f"row:{conversation_id}:{workspace_id}",
        conversation_id=conversation_id,
        native_session_id=None,
        title=title,
        scope_type=scope_type,
        workspace_id=workspace_id,
        workspace_label=str(workspace_id or "Chats"),
        starred=starred,
        updated_sort=updated_sort,
        selected=selected,
        run_marker=run_marker,
    )


def test_public_tree_shapes_are_frozen_slotted_values() -> None:
    conversation = WorkspaceTreeConversation("c1", "Title", False, "", False, "")
    workspace = WorkspaceTreeWorkspace("w1", "Label", (conversation,), None)

    assert not hasattr(conversation, "__dict__")
    assert not hasattr(workspace, "__dict__")
    with pytest.raises(FrozenInstanceError):
        conversation.title = "Changed"  # type: ignore[misc]


def test_named_and_default_unassigned_rows_have_exclusive_owners() -> None:
    rows = (
        _row("named", workspace_id="w1", title="Named"),
        _row("default", workspace_id=DEFAULT_WORKSPACE_ID, title="Default"),
        _row("unassigned", workspace_id=None, title="Unassigned", scope_type="global"),
    )

    tree = build_workspace_tree_state(
        workspaces=(("w1", "Workspace One"), (DEFAULT_WORKSPACE_ID, "Default")),
        rows=rows,
    )
    flat = build_console_conversation_browser_state(
        rows=rows,
        active_workspace_id="w1",
    )

    assert [
        (node.workspace_id, [row.conversation_id for row in node.conversations])
        for node in tree
    ] == [
        ("w1", ["named"]),
    ]
    assert [section.section_id for section in flat.sections] == ["chats"]
    assert [row.conversation_id for row in flat.sections[0].rows] == [
        "default",
        "unassigned",
    ]


def test_named_workspace_keeps_unsaved_native_session_by_stable_row_key() -> None:
    row = ConsoleConversationBrowserInputRow(
        row_key="native:session-7",
        conversation_id=None,
        native_session_id="session-7",
        title="Unsaved roleplay",
        scope_type="workspace",
        workspace_id="w1",
        workspace_label="Workspace One",
        star_enabled=False,
        source_kind="native",
    )

    tree = build_workspace_tree_state(
        workspaces=(("w1", "Workspace One"),),
        rows=(row,),
    )

    assert [conversation.conversation_id for conversation in tree[0].conversations] == [
        "native:session-7"
    ]
    assert tree[0].conversations[0].star_enabled is False


def test_tree_sorts_starred_first_then_recency_and_preserves_markers() -> None:
    tree = build_workspace_tree_state(
        workspaces=(("w1", "Workspace"),),
        rows=(
            _row("new", workspace_id="w1", title="Newest", updated_sort="2026-08-22"),
            _row(
                "star-old",
                workspace_id="w1",
                title="Star old",
                starred=True,
                updated_sort="2026-08-01",
                selected=True,
                run_marker="◆",
            ),
            _row(
                "star-new",
                workspace_id="w1",
                title="Star new",
                starred=True,
                updated_sort="2026-08-20",
            ),
            _row("old", workspace_id="w1", title="Old", updated_sort="2026-08-02"),
        ),
    )

    conversations = tree[0].conversations
    assert [row.conversation_id for row in conversations] == [
        "star-new",
        "star-old",
        "new",
        "old",
    ]
    assert conversations[1].selected is True
    assert conversations[1].run_marker == "◆"
    assert len({row.conversation_id for row in conversations}) == 4


def test_boundary_dedupes_conversation_id_once_and_movement_is_atomic() -> None:
    duplicate_rows = (
        _row("move", workspace_id="w1", title="First owner"),
        _row("move", workspace_id="w2", title="Duplicate owner"),
    )
    before = build_workspace_tree_state(
        workspaces=(("w2", "Two"), ("w1", "One")),
        rows=duplicate_rows,
    )
    after = build_workspace_tree_state(
        workspaces=(("w2", "Two"), ("w1", "One")),
        rows=(_row("move", workspace_id="w2", title="Moved"),),
    )

    assert _owners(before, "move") == ["w1"]
    assert _owners(after, "move") == ["w2"]


@pytest.mark.parametrize("flat_first", (False, True))
def test_duplicate_id_across_named_and_flat_owners_uses_first_boundary_owner(
    flat_first: bool,
) -> None:
    named = _row("same", workspace_id="w1", title="Named")
    flat = _row("same", workspace_id=DEFAULT_WORKSPACE_ID, title="Flat")
    rows = (flat, named) if flat_first else (named, flat)

    tree = build_workspace_tree_state(workspaces=(("w1", "One"),), rows=rows)
    flat_state = build_console_conversation_browser_state(
        rows=rows, active_workspace_id=None
    )
    tree_ids = {row.conversation_id for node in tree for row in node.conversations}
    flat_ids = {
        row.conversation_id for section in flat_state.sections for row in section.rows
    }

    assert tree_ids.isdisjoint(flat_ids)
    assert (tree_ids, flat_ids) == (
        (set(), {"same"}) if flat_first else ({"same"}, set())
    )


def test_literal_labels_and_deterministic_ties_are_preserved() -> None:
    workspaces = (("w-b", "[bold]你好[/bold] 🧭"), ("w-a", "[bold]你好[/bold] 🧭"))
    rows = (
        _row("c-b", workspace_id="w-a", title="[red]same[/red]", updated_sort="x"),
        _row("c-a", workspace_id="w-a", title="[red]same[/red]", updated_sort="x"),
    )

    first = build_workspace_tree_state(workspaces=workspaces, rows=rows)
    second = build_workspace_tree_state(workspaces=reversed(workspaces), rows=rows)

    assert first == second
    assert [node.workspace_id for node in first] == ["w-a", "w-b"]
    assert first[0].label == "[bold]你好[/bold] 🧭"
    assert [row.conversation_id for row in first[0].conversations] == ["c-a", "c-b"]
    assert first[0].conversations[0].title == "[red]same[/red]"


def test_search_matches_literal_workspace_label_and_excludes_nonmatches() -> None:
    tree = build_workspace_tree_state(
        workspaces=(
            ("literal", "[bold]工程[/bold] 🧭"),
            ("other", "Unrelated workspace"),
        ),
        rows=(),
        query="工程",
    )

    assert [(node.workspace_id, node.label) for node in tree] == [
        ("literal", "[bold]工程[/bold] 🧭")
    ]


def test_next_cursor_is_scoped_to_its_workspace() -> None:
    tree = build_workspace_tree_state(
        workspaces=(("w1", "One"), ("w2", "Two")),
        rows=(),
        next_cursors={"w1": 75, "w2": None},
    )

    assert [(node.workspace_id, node.next_cursor) for node in tree] == [
        ("w1", 75),
        ("w2", None),
    ]


def test_page_status_has_frozen_safe_defaults_and_scoped_values() -> None:
    tree = build_workspace_tree_state(
        workspaces=(("w1", "One"), ("w2", "Two")),
        rows=(),
        loading={"w1": True},
        errors={"w1": "Literal [error] 错误 🧭"},
        retry_cursors={"w1": 75},
        membership_unknown={"w1": True},
    )

    assert (
        tree[0].loading,
        tree[0].error,
        tree[0].retry_cursor,
        tree[0].membership_unknown,
    ) == (True, "Literal [error] 错误 🧭", 75, True)
    assert (
        tree[1].loading,
        tree[1].error,
        tree[1].retry_cursor,
        tree[1].membership_unknown,
    ) == (False, "", None, False)
    with pytest.raises(FrozenInstanceError):
        tree[0].loading = False


def test_marker_update_reuses_every_unrelated_workspace_projection() -> None:
    tree = build_workspace_tree_state(
        workspaces=(("w1", "One"), ("w2", "Two")),
        rows=(
            _row("c1", workspace_id="w1", title="One"),
            _row("c2", workspace_id="w2", title="Two"),
        ),
    )
    current = tree[0].conversations[0]

    updated = update_workspace_tree_conversation(
        tree,
        workspace_id="w1",
        conversation=WorkspaceTreeConversation(
            conversation_id=current.conversation_id,
            title=current.title,
            starred=current.starred,
            updated_sort=current.updated_sort,
            selected=True,
            run_marker="◆",
        ),
    )

    assert updated[0] is not tree[0]
    assert updated[1] is tree[1]
    assert updated[0].conversations[0].selected is True
    assert updated[0].conversations[0].run_marker == "◆"


def _owners(tree, conversation_id: str) -> list[str]:
    return [
        node.workspace_id
        for node in tree
        if any(row.conversation_id == conversation_id for row in node.conversations)
    ]
