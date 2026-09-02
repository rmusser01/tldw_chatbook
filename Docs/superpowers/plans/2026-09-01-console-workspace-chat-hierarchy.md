# Console Workspace and Chat Hierarchy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make workspace rows and their chats visually distinct through row-specific right-edge action markers and a four-cell child indent.

**Architecture:** Keep the change inside the existing native Textual `ConsoleWorkspaceTree` renderer. Render workspace `@` and conversation `*` affordances into a shared right-edge column, derive pointer hit zones from the final painted geometry, and retain the existing menu messages and keyboard binding.

**Tech Stack:** Python 3.11+, Textual 8.2.8, Rich `Text`, pytest/Textual pilot tests

**Spec:** `backlog/tasks/task-27137 - Clarify-Console-workspace-and-chat-hierarchy.md`

## Global Constraints

- Workspace rows use `@`; conversation rows keep `*` with the same visual style.
- Conversation rows sit four terminal cells beneath their owning workspace.
- Long labels ellipsize before the action column; neither action glyph may move or disappear.
- Pointer activation and the existing `m` binding continue to open the correct row menu.
- No new dependency, persistent state, key binding, or data-model change.
- ADR required: no. This is a renderer-level refinement of the existing Console tree and row-menu contract.

---

### Task 1: Align and distinguish Console tree rows

**Files:**
- Modify: `Tests/UI/test_console_workspace_tree.py`
- Modify: `Tests/UI/test_console_workspace_action_menu.py`
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_tree.py`
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_action_menu.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `backlog/tasks/task-27137 - Clarify-Console-workspace-and-chat-hierarchy.md`

**Interfaces:**
- Consumes: `ConsoleWorkspaceTree.render_label`, `WorkspaceTreeMenuRequested`, Textual tree guide geometry, and the existing `m` action.
- Produces: right-edge workspace `@` and conversation `*` affordances with absolute widget-cell hit zones; no public API changes.

- [x] **Step 1: Add failing tests for the approved row hierarchy**

Extend the real Textual tree harness to collect `WorkspaceTreeMenuRequested`, then add behavior tests equivalent to:

```python
@pytest.mark.asyncio
async def test_workspace_and_chat_action_markers_share_the_right_edge() -> None:
    tree = _tree()
    app = _TreeHarness(tree)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        edge = tree.scrollable_content_region.width
        assert tree._menu_zones["workspace:w1"] == (edge - 2, edge)
        assert tree._menu_zones["conversation:c1"] == (edge - 2, edge)
        assert tree.render_line(0).text[edge - 1] == "@"
        assert tree.render_line(1).text[edge - 1] == "*"
```

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("line", "kind"),
    [(0, "workspace"), (1, "conversation")],
)
async def test_right_edge_marker_opens_the_matching_row_menu(line, kind) -> None:
    tree = _tree()
    app = _TreeHarness(tree)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        edge = tree.scrollable_content_region.width - 1
        assert await pilot.click(tree, offset=(edge, line))
        await pilot.pause()
        request = next(
            message
            for message in app.messages
            if isinstance(message, WorkspaceTreeMenuRequested)
        )
        assert request.kind == kind
```

Change the native configuration assertion from `guide_depth == 2` to `guide_depth == 4`. Correct the pre-existing tooltip boundary fixtures to reserve the two-cell action column that TASK-25712 added; these fixtures currently fail on clean `dev` because they predate that column.

- [x] **Step 2: Run the new tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_console_workspace_tree.py::test_native_configuration_and_literal_unicode_labels \
  Tests/UI/test_console_workspace_tree.py::test_workspace_and_chat_action_markers_share_the_right_edge \
  Tests/UI/test_console_workspace_tree.py::test_right_edge_marker_opens_the_matching_row_menu -q
```

Expected: failures show `guide_depth` is still `2`, the workspace marker is still `*`, and short-row action zones end beside the label instead of at the viewport edge.

- [x] **Step 3: Implement the minimal renderer change**

Replace the shared affordance with a row-kind mapping and fill the available content cells before appending it:

```python
_MENU_AFFORDANCES = {
    "workspace": " @",
    "conversation": " *",
}

affordance = _MENU_AFFORDANCES.get(kind)
if affordance is not None and data is not None:
    content_budget = self._label_budget(node)
    label.truncate(content_budget, overflow="ellipsis")
    label.append(" " * max(0, content_budget - label.cell_len), base_style)
    label.append(affordance, base_style)
    end = self._visible_guide_cells(node) + label.cell_len
    self._menu_zones[data.key] = (end - len(affordance), end)
```

Set `guide_depth = 4`, share visible-guide measurement between the label budget and hit-zone calculation, and update comments/docstrings from “asterisk” to row-specific action affordances. Do not alter menu message payloads or keyboard actions.

- [x] **Step 4: Run the tree and menu tests and verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_console_workspace_tree.py \
  Tests/UI/test_console_workspace_action_menu.py -q
```

Expected: all targeted tests pass; the six clean-`dev` baseline failures are resolved by the new geometry and corrected action-column fixtures.

- [x] **Step 5: Run static and design checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/Widgets/Console/console_workspace_tree.py \
  Tests/UI/test_console_workspace_tree.py \
  Tests/UI/test_console_workspace_action_menu.py
```

Then run Impeccable's mechanical detector once over the changed UI targets and inspect `git diff --check`.

- [x] **Step 6: Close the task and commit the reviewable slice**

Mark all acceptance criteria complete, add concise implementation notes including the clean-`dev` baseline finding and ADR rationale, set TASK-27137 to Done, stage only the planned files, and commit:

```bash
git commit -m "feat(console): clarify workspace and chat hierarchy"
```
