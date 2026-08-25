# Console Rail Interaction and Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Console Tree activation deterministic, make one rail-disclosure layout follow users across workspaces by default, and keep the next send's authority visible in a pinned six-row Inspect summary.

**Architecture:** Keep the existing native Textual Tree, Console rail preference store, and atomic `ConsoleInspectorState` as the three authorities. Add only narrow adapters at their current presentation boundaries: press-key capture and two-stage activation in the Tree, scope-aware key selection/seeding in `ChatScreen`, and a pure pinned summary plus conditional group projection in the existing Inspector widget. Do not add a database, provider call, timer, secondary cache, or parallel reactive state owner.

**Tech Stack:** Python 3.12, Textual 8.2.8, Rich cell measurement, TOML-backed Console settings, pytest/Textual Pilot, Ruff, modular TCSS.

---

## Scope, boundaries, and file map

This plan implements AC #8–#10 of `TASK-20937.6`. The existing edge-rail geometry, 15/20/35 section ceilings, Character contain behavior, benchmark evidence, and cross-terminal closeout remain governed by the parent task and are regression gates, not redesign targets.

ADR required: yes.

ADR path: `backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md` (already amended by the approved design).

Reason: the work changes the long-lived Tree activation grammar and the persistence scope used by all Console rail disclosures. It does not introduce a new storage boundary or schema.

### Files with one clear responsibility

- Modify `tldw_chatbook/Widgets/Console/console_workspace_tree.py`: own press-time stable-key capture, selection versus activation, native click-chain validation, and truncation-aware Tree tooltips.
- Modify `tldw_chatbook/Widgets/Console/console_workspace_context.py`: render and patch the one-row full-label selection context without changing Star ownership.
- Modify `tldw_chatbook/UI/Console_Modules/left_rail.py`: relay Tree context changes and preserve the production rail gesture through allocation reconciliation.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: expose Tree grammar in F1, select/read/write the active rail preference scope, sync the pinned authority widget, persist More, and recover Inspector focus.
- Modify `tldw_chatbook/Chat/console_rail_state.py`: define scope values/keys, retain the shared key during pruning, and carry `inspector_more_open` in the existing disclosure payload.
- Modify `tldw_chatbook/config.py`: ship and normalize `console.rail_layout_scope = "global"`.
- Modify `tldw_chatbook/UI/Screens/settings_screen.py`: expose Global/Per workspace in canonical Console Behavior settings with search, guidance, draft, save, and revert behavior.
- Create `tldw_chatbook/Widgets/Console/console_send_authority_summary.py`: pure five-fact projection and fixed six-row widget synchronized from one `ConsoleInspectorState` snapshot.
- Modify `tldw_chatbook/Widgets/Console/console_run_inspector.py`: classify Tools/Approvals/Artifacts as promoted or More-owned, render one keyboard-complete More boundary, remove exact lower duplicates, and preserve focus across structural moves.
- Modify `tldw_chatbook/UI/Console_Modules/right_rail.py`: mount the authority summary outside the outer scroll owner and pass the initial scoped More preference.
- Modify `tldw_chatbook/Chat/console_display_state.py`: carry staged-source and pending-approval counts on the existing atomic Inspector snapshot.
- Modify `tldw_chatbook/Widgets/Console/__init__.py`: export the new authority-summary widget.
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss` and regenerate `tldw_chatbook/css/tldw_cli_modular.tcss`: enforce one-row ellipsis, six-row summary geometry, and distinct selected/active Tree cues.
- Modify `Docs/User_Guide/console/sessions-tabs-workspaces.md`: document selection/activation, full-label help, layout scope, pinned authority, and More.
- Modify `backlog/tasks/task-20937.6 - Verify-and-document-Console-edge-rails-and-workspace-ownership.md`: record implementation notes and check AC #8–#10 only after their focused evidence passes.

### Test ownership

- Modify `Tests/UI/test_console_workspace_tree.py`: isolated native Tree grammar, stable-key chain, keyboard behavior, and tooltip geometry.
- Modify `Tests/UI/test_console_workspace_context_rail.py`: selected-row context copy and one-row layout.
- Modify `Tests/UI/test_console_rail_reconciliation.py`: production stylesheet + outer-rail reflow regression and exact pressed-node identity.
- Modify `Tests/Chat/test_console_rail_state.py` and `Tests/Chat/test_console_rail_state_prune.py`: keys, coercion, serialization, collision resistance, and pruning.
- Modify `Tests/test_config_console_defaults.py`: global default and malformed-value fallback.
- Modify `Tests/UI/test_settings_console_rail_labels.py`: canonical settings search, keyboard draft, save, failure, and revert paths.
- Modify `Tests/UI/test_console_inspector_compact_access.py`: scope switch seeding, workspace round trips, inactive-record preservation, and responsive override non-mutation.
- Create `Tests/UI/test_console_send_authority_summary.py`: pure projection, atomic patching, exact row count, ellipsis, and incomplete-state behavior.
- Modify `Tests/UI/test_console_run_inspector.py`: conditional group ordering, More input grammar, persistence message, and focus recovery.
- Modify `Tests/UI/test_console_right_rail.py`: production boundary order, pinned placement, wide/narrow geometry, and atomic screen sync.

## Contract constants used throughout

Use these exact values and do not introduce synonyms:

```python
CONSOLE_RAIL_LAYOUT_SCOPE_GLOBAL = "global"
CONSOLE_RAIL_LAYOUT_SCOPE_WORKSPACE = "workspace"
CONSOLE_RAIL_SHARED_LAYOUT_SCOPE = "shared-layout-v1"
CONSOLE_INSPECTOR_MORE_DISCLOSURE_ID = "inspector_more"

CONSOLE_TREE_SELECTION_CONTEXT_ID = "console-workspace-tree-selection-context"
CONSOLE_AUTHORITY_SUMMARY_ID = "console-send-authority-summary"
CONSOLE_INSPECTOR_MORE_TOGGLE_ID = "console-inspector-more-toggle"
CONSOLE_INSPECTOR_MORE_BODY_ID = "console-inspector-more-body"
```

The user-visible labels are exactly `Global`, `Per workspace`, `What happens if I send now?`, `Where`, `Scope`, `Run`, `Sources`, `Approvals`, and `More`.

### Task 1: Make Tree selection and activation deterministic

**Files:**
- Modify: `Tests/UI/test_console_workspace_tree.py:162-218`
- Modify: `Tests/UI/test_console_rail_reconciliation.py:289-520`
- Modify: `Tests/UI/test_console_workspace_context_rail.py`
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_tree.py:166-230,645-815`
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py:1199-1406`
- Modify: `tldw_chatbook/UI/Console_Modules/left_rail.py:852-884,1967-1979`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:1400-1460,3136-3163`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss:3639-3670,5149-5185`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss` (generated)

- [ ] **Step 1: Replace immediate-click expectations with the two-stage grammar**

In `Tests/UI/test_console_workspace_tree.py`, replace `test_native_pointer_disclosure_toggles_and_label_selects` with focused tests that assert:

```python
async def test_single_label_click_selects_and_expands_without_activation():
    # Click the label of collapsed w2, not its disclosure glyph.
    assert workspace.is_collapsed
    assert await pilot.click(tree, offset=(4, 3))
    await pilot.pause()
    assert tree.cursor_node is workspace
    assert workspace.is_expanded
    assert not any(isinstance(item, WorkspaceTreeWorkspaceSelected) for item in app.messages)

async def test_double_click_activates_only_the_same_selected_stable_key():
    assert await pilot.click(tree, offset=workspace_offset, times=2)
    await pilot.pause()
    assert [item.workspace_id for item in app.messages
            if isinstance(item, WorkspaceTreeWorkspaceSelected)] == ["w2"]

async def test_enter_activates_and_space_left_right_only_change_disclosure():
    tree.move_cursor(workspace)
    await pilot.press("enter")
    assert isinstance(app.messages[-1], WorkspaceTreeWorkspaceSelected)
    message_count = len(app.messages)
    await pilot.press("space", "left", "right")
    assert len(app.messages) == message_count
```

Add equivalent conversation assertions and retain immediate action coverage for `load-more` and `retry` nodes.

- [ ] **Step 2: Add the production reflow RED test**

Use the existing complete-stylesheet production host in `Tests/UI/test_console_rail_reconciliation.py`. Start Workspaces inactive and its Tree overflowing, press the visible workspace label, allow `ConsoleLeftRail.on_mouse_down()` to activate/reveal the section, then release/click after the outer body moved.

Assert all of the following in one incident test:

```python
pressed_key = "workspace:workspace-1"
assert tree.cursor_node.data.key == pressed_key
assert app.workspace_requests == []
assert app.conversation_requests == []

# A native second click on the same stable key activates exactly once.
assert [event.workspace_id for event in app.workspace_requests] == ["workspace-1"]

# Reuse the old coordinate after forcing a different row beneath it.
# The chain must cancel, never activate the replacement conversation.
assert not any(event.conversation_id == "conversation-1"
               for event in app.conversation_requests)
```

The test must exercise Pilot pointer input; calling the event handler or posting a selection message directly is not evidence for this bug.

- [ ] **Step 3: Run the Tree RED tests**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_workspace_tree.py \
  Tests/UI/test_console_rail_reconciliation.py \
  -q
```

Expected: FAIL because a single label click still posts `WorkspaceTreeWorkspaceSelected`/`WorkspaceTreeConversationSelected`, and the reflow case activates the row now occupying the old coordinate.

- [ ] **Step 4: Capture the stable node at pointer press and split selection from activation**

In `ConsoleWorkspaceTree`, add only these transient fields:

```python
self._pressed_node_key: str | None = None
self._last_pointer_click_key: str | None = None
```

Add private helpers with these contracts:

```python
def _node_for_stable_key(self, key: str | None) -> TreeNode[WorkspaceTreeNodeData] | None:
    """Resolve a current node from the three existing keyed registries."""

def _select_node(self, node: TreeNode[WorkspaceTreeNodeData]) -> None:
    """Move the cursor; additionally expand a collapsed workspace label."""

def _activate_node(self, node: TreeNode[WorkspaceTreeNodeData]) -> None:
    """Post workspace/conversation activation or immediate auxiliary action."""
```

Override Textual's pinned 8.2.8 press/click seam, using `event.style.meta["line"]` only at press time. Resolve the later gesture exclusively through `_pressed_node_key`; never perform a second coordinate lookup after rail activation/reflow.

For a label click:

```python
node = self._node_for_stable_key(self._pressed_node_key)
if node is None:
    self._last_pointer_click_key = None
    return
self._select_node(node)
if event.chain == 2:
    data = node.data
    selected_key = self.cursor_node.data.key if self.cursor_node and self.cursor_node.data else None
    if data and data.key == self._last_pointer_click_key == selected_key:
        self._activate_node(node)
self._last_pointer_click_key = node.data.key if node.data else None
```

Disclosure-glyph clicks stay immediate disclosure gestures. `Tree.NodeSelected` becomes keyboard activation only: `action_select_cursor()` calls `_activate_node(self.cursor_node)`. `on_tree_node_selected()` must no longer be the pointer/business activation boundary. Do not add a timer; `events.Click.chain` is authoritative.

If the pressed node disappears, clear both transient keys, invoke the existing keyed focus-recovery behavior, and post no activation.

- [ ] **Step 5: Make selected and active states independently legible**

Keep the current `●` active-workspace and `›` active-conversation markers. In `render_label()`, add a selection marker/class driven by cursor identity rather than business `data.selected`; the selected marker must coexist with active markers and remain visible without color.

Add a test covering all four combinations: neither, selected-only, active-only, selected+active. Do not encode selection solely as background color.

- [ ] **Step 6: Add truncation-aware tooltip RED tests**

Replace the existing always-full-tooltip assertion with:

```python
assert short_tree.tooltip is None
assert long_tree.tooltip == raw_long_label

# After width growth makes the row fit, the tooltip clears.
tree.styles.width = 80
await pilot.pause()
assert tree.tooltip is None

# After reflow changes the hovered line's key, no old full label survives.
assert tree.tooltip != old_raw_label
```

Cover both cursor and hover, Unicode cell width, guide indentation, narrow overflow, and non-overflow.

- [ ] **Step 7: Implement the tooltip geometry from the same render budget**

Extract one helper used by both `render_label()` and `_update_tooltip()`:

```python
def _available_label_cells(self, node: TreeNode[WorkspaceTreeNodeData]) -> int:
    guide_cells = node.depth * 3
    return max(1, self.size.width - guide_cells)
```

Use `rich.cells.cell_len()` on the same untruncated visible label that `render_label()` builds (active/star/run/selection markers plus the first physical raw-label row) against that exact budget. A tooltip exists only when that visible label exceeds it, and its value remains the full literal raw label. Call `_update_tooltip()` after cursor/hover changes, `Resize`, projection sync, expansion/collapse, and outer reflow. If the hover line/key no longer resolves, clear the tooltip before recomputing from the cursor.

- [ ] **Step 8: Add the selected-row context and F1 RED tests**

In `Tests/UI/test_console_workspace_context_rail.py`, assert the workspace tray always reserves one physical row with:

```text
Selected: Research Lab · Enter open
```

and updates it in place for a conversation, an auxiliary action, and removal. Long values must ellipsize without growing beyond one row. The Star row remains independently conditional.

In the Console help test, focus the Tree, press F1, and assert the panel contains the complete unellipsized selected label plus all six gestures: single click, double-click, Enter, Space, Left, Right.

- [ ] **Step 9: Implement context copy without changing Star ownership**

Compose `#console-workspace-tree-selection-context` immediately after the compact Switch/New/RAG row and before the contextual Star row. Store every selectable cursor data object in `_workspace_tree_context_data`; compute markability separately with `_workspace_tree_context_is_markable()`.

Patch the Static text in `sync_workspace_tree_context()` and update the Star row exactly as today. The context Static uses `height: 1`, `text-wrap: nowrap`, and `text-overflow: ellipsis`; its `tooltip` is the complete literal copy only when its rendered cell length exceeds its measured width.

Build contextual F1 notes from the same stored cursor data so the help panel exposes the complete label even when the row or pointer tooltip is clipped.

- [ ] **Step 10: Regenerate CSS and run the focused Tree gate**

Run:

```bash
../../.venv/bin/python -B tldw_chatbook/css/build_css.py
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_workspace_tree.py \
  Tests/UI/test_console_workspace_context_rail.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_css_bundle_sync_guard.py \
  -q
```

Expected: PASS. Inspect the production test's captured geometry values to confirm the Workspaces content ceiling remains 20 rows and no second scroll owner was introduced.

- [ ] **Step 11: Commit the Tree slice**

```bash
git add \
  Tests/UI/test_console_workspace_tree.py \
  Tests/UI/test_console_workspace_context_rail.py \
  Tests/UI/test_console_rail_reconciliation.py \
  tldw_chatbook/Widgets/Console/console_workspace_tree.py \
  tldw_chatbook/Widgets/Console/console_workspace_context.py \
  tldw_chatbook/UI/Console_Modules/left_rail.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "fix(console): separate tree selection from activation"
```

### Task 2: Add global-by-default rail layout scope

**Files:**
- Modify: `Tests/Chat/test_console_rail_state.py:37-130,830-1005`
- Modify: `Tests/Chat/test_console_rail_state_prune.py`
- Modify: `Tests/UI/test_console_inspector_compact_access.py`
- Modify: `Tests/test_config_console_defaults.py:148-210`
- Modify: `Tests/UI/test_settings_console_rail_labels.py`
- Modify: `tldw_chatbook/Chat/console_rail_state.py:12-24,114-228,231-355`
- Modify: `tldw_chatbook/config.py:1555-1575,3067-3075`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py:795-807,1343-1375,4380-4510,6736-6749,12927-12963,17866-17880,18805-18825,21550-21555,22461-22483`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:10185-10347,10507-10536,10603-10669,10864-10923`

- [ ] **Step 1: Write pure key, default, serialization, and pruning RED tests**

Add assertions for these exact outcomes:

```python
assert normalize_console_rail_layout_scope(None) == "global"
assert normalize_console_rail_layout_scope("bogus") == "global"
assert normalize_console_rail_layout_scope("workspace") == "workspace"

global_key = build_console_rail_preference_key(
    workspace_id="Research Lab", layout_scope="global"
)
assert global_key.value == "console_rail_state:global:shared-layout-v1"
assert global_key.fallback_value is None

workspace_key = build_console_rail_preference_key(
    workspace_id="Research Lab", layout_scope="workspace"
)
assert workspace_key.value == "console_rail_state:Research_Lab:layout"
assert workspace_key.fallback_value == "console_rail_state:Research_Lab:global"

assert "console_rail_state:global:shared-layout-v1" not in collect_prunable_console_rail_keys(
    ["console_rail_state:global:shared-layout-v1"],
    live_scope_ids=set(),
)
```

Extend preference round-trip tests with `inspector_more_open`, defaulting to `False`, while proving scroll offsets, focus ids, search queries, selected node ids, and tooltip fields are ignored rather than serialized.

- [ ] **Step 2: Write mounted scope seeding and switch RED tests**

In `Tests/UI/test_console_inspector_compact_access.py`, use a mounted Console with two workspace ids and a monkeypatched persistence writer. Cover the precedence matrix:

1. Global missing + active workspace `:layout` exists → copy it once to shared.
2. Global missing + only active workspace legacy `:global` exists → copy it once to shared and retain legacy.
3. Global missing + no source → write defaults once.
4. Per workspace missing + legacy workspace record exists → seed from legacy before shared.
5. Per workspace missing + shared exists → seed from shared once.
6. Existing target record always wins and is never overwritten.
7. Switching modes/workspaces never deletes inactive records.
8. Compact-collapse resolution changes effective rendering only; the stored payload is byte-for-byte unchanged.

Use exact call counts and keys; a passing visible layout without persistence evidence is insufficient.

- [ ] **Step 3: Run the rail-state RED tests**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_rail_state.py \
  Tests/Chat/test_console_rail_state_prune.py \
  Tests/UI/test_console_inspector_compact_access.py \
  -q
```

Expected: FAIL because the current key is always `<workspace>:layout`, the pruner does not retain `shared-layout-v1`, and no cross-scope seed exists.

- [ ] **Step 4: Implement the pure scope contract**

In `console_rail_state.py`, add the constants from this plan and:

```python
def normalize_console_rail_layout_scope(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return (
        CONSOLE_RAIL_LAYOUT_SCOPE_WORKSPACE
        if normalized == CONSOLE_RAIL_LAYOUT_SCOPE_WORKSPACE
        else CONSOLE_RAIL_LAYOUT_SCOPE_GLOBAL
    )
```

Add `layout_scope` to `build_console_rail_preference_key()`. Global mode always returns the reserved shared key and ignores the workspace id for storage; workspace mode preserves the current sanitized workspace `:layout` key and read-only legacy fallback.

Add `inspector_more_open: bool = False` to `ConsoleRailPreferences` and `ConsoleRailState`, coercion, serialization, and `build_console_rail_state()`. Keep `CONSOLE_RAIL_SECTION_IDS` as the left-rail list; add a separate preference allowlist containing those ids plus `inspector_more` so left-rail synchronization never queries a nonexistent left section.

The pruner retains exactly `layout`, legacy `global`, and `shared-layout-v1` scopes.

- [ ] **Step 5: Replace destructive fallback migration with lossless one-time seeding**

In `ChatScreen`, replace `_migrate_console_rail_fallback_preferences()` with one helper that receives both the selected key and the active workspace key:

```python
def _ensure_console_rail_scope_seed(
    self,
    selected_key: ConsoleRailPreferenceKey,
    workspace_key: ConsoleRailPreferenceKey,
) -> Any:
    rail_state = self._console_rail_state_config()
    if selected_key.value in rail_state:
        return rail_state[selected_key.value]

    shared_key = build_console_rail_preference_key(layout_scope="global")
    candidates = (
        (workspace_key.value, workspace_key.fallback_value)
        if selected_key.scope_id == CONSOLE_RAIL_SHARED_LAYOUT_SCOPE
        else (workspace_key.fallback_value, shared_key.value)
    )
    source = next((rail_state[key] for key in candidates if key and key in rail_state), None)
    preferences = coerce_console_rail_preferences(source)
    serialized = serialize_console_rail_preferences(preferences)
    rail_state[selected_key.value] = serialized
    self._save_console_rail_preferences(selected_key.value, serialized, notify_on_failure=False)
    return serialized
```

Build `workspace_key` with `layout_scope="workspace"`, build `selected_key` with the normalized setting, and use the same helper in both `_build_console_rail_state()` and `_set_console_rail_preference()` so read and write cannot drift.

Do not pop or delete the legacy source. Do not overwrite an existing target. Do not seed during responsive rendering more than once; key presence is the one-time latch.

- [ ] **Step 6: Add and normalize the config default**

Add to the generated default TOML:

```toml
rail_layout_scope = "global"  # Share Console rail disclosure across workspaces; use "workspace" for per-workspace layouts
```

Normalize any non-string or unknown value to `global`; accept only `global` and `workspace`. Tests must load the generated default, a valid workspace override, mixed case/whitespace if existing config normalization permits it, and malformed values.

- [ ] **Step 7: Add canonical Settings RED tests**

Extend `Tests/UI/test_settings_console_rail_labels.py` with `RAIL_LAYOUT_SCOPE = "#settings-console-rail-layout-scope"`. Assert:

- the Select defaults to Global and appears immediately under `Rail presentation`;
- `/layout scope` search lands on it;
- focused guidance explains continuity, prior-record retention, exact save key, and that the change applies after save;
- keyboard selection stages `rail_layout_scope` without mutating runtime config;
- category Save writes exactly `{"console": {"rail_layout_scope": "workspace"}}` when it is the only dirty field;
- failed save retains the draft and Global stays active;
- Revert restores the loaded value along with other Console Behavior drafts.

- [ ] **Step 8: Implement the Settings field through existing draft plumbing**

Add the key to `CONSOLE_BEHAVIOR_CONSOLE_KEYS`, search aliases, loaded values, active field ids, guidance, and sync. Render:

```python
yield Static("Rail layout scope", classes="settings-input-label")
yield Select(
    (("Global", "global"), ("Per workspace", "workspace")),
    value=self._console_rail_layout_scope(),
    allow_blank=False,
    id="settings-console-rail-layout-scope",
)
yield Static(
    "Global keeps one arrangement everywhere. Per workspace restores and keeps each workspace's saved arrangement.",
    classes="settings-help-copy",
)
```

Handle `Select.Changed`, stage the normalized string, update draft status, and guard programmatic synchronization with a dedicated `_syncing_console_rail_layout_scope` flag. Use the generic category save worker; do not add a second writer.

- [ ] **Step 9: Run config/settings/scope gates**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_rail_state.py \
  Tests/Chat/test_console_rail_state_prune.py \
  Tests/test_config_console_defaults.py \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/UI/test_console_inspector_compact_access.py \
  -q
```

Expected: PASS with one persistence write per absent target, zero deletions on scope switches, and no responsive-write mutation.

- [ ] **Step 10: Commit the layout-scope slice**

```bash
git add \
  Tests/Chat/test_console_rail_state.py \
  Tests/Chat/test_console_rail_state_prune.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/test_config_console_defaults.py \
  tldw_chatbook/Chat/console_rail_state.py \
  tldw_chatbook/config.py \
  tldw_chatbook/UI/Screens/settings_screen.py \
  tldw_chatbook/UI/Screens/chat_screen.py
git commit -m "feat(console): add scoped rail layouts"
```

### Task 3: Pin next-send authority and conditionally disclose empty groups

**Files:**
- Create: `Tests/UI/test_console_send_authority_summary.py`
- Create: `tldw_chatbook/Widgets/Console/console_send_authority_summary.py`
- Modify: `Tests/UI/test_console_run_inspector.py`
- Modify: `Tests/UI/test_console_right_rail.py`
- Modify: `Tests/UI/test_console_inspector_compact_access.py`
- Modify: `tldw_chatbook/Widgets/Console/console_run_inspector.py:31-410`
- Modify: `tldw_chatbook/UI/Console_Modules/right_rail.py:221-305,1060-1170`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:11200-11281,11831-11908,17258-17282`
- Modify: `tldw_chatbook/Chat/console_display_state.py:1080-1233`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss:3740-3820`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss` (generated)

- [ ] **Step 1: Write pure authority-projection RED tests**

Create `Tests/UI/test_console_send_authority_summary.py` with a state factory covering:

- saved conversation in a named workspace;
- Default + temporary native session;
- one-shot prefill taking precedence in Scope;
- pinned prefill and narrowed `scope N items` detail;
- generating, pending approval, provider blocked, source blocked, and recovery-required precedence;
- zero and nonzero source/approval counts;
- resilient incomplete ownership.

Assert the pure projection is one immutable value with five fields and that no value comes from app, DB, provider, or another cache. Use exact copies such as:

```python
assert projection.where == "Research Lab › Research conversation 3"
assert projection.scope == "One-shot prefill · narrowed to 4 items"
assert projection.run == "Waiting for approval"
assert projection.sources == "2 staged"
assert projection.approvals == "1 pending · action required"
```

For missing ownership, assert `run == "Inspector data incomplete"` and use explicit unknown/none values for missing facts; never invent Ready.

Extend `ConsoleInspectorState` itself with `staged_source_count: int = 0`, `pending_approval_count: int = 0`, `scope_item_count: int | None = None`, and `ephemeral: bool = False`; populate them in `from_values()`. Pass the cached current staged-source count when `ChatScreen._build_console_inspector_state()` constructs the snapshot. This is additive data on the one existing atomic Inspector snapshot, not a second owner. The pinned summary must use these typed fields instead of parsing human-readable count, scope, or persistence strings.

- [ ] **Step 2: Write fixed geometry and atomic-sync RED tests**

Mount the new widget at wide and 34-column rail widths. Assert exactly six direct physical rows: heading plus five facts. Every fact must stay height 1, `nowrap`, and ellipsize. Then sync from state A to state B in one call and assert all five mounted Statics show B and none show A, with no recompose when the fixed structure is unchanged.

The summary is one keyboard focus stop, not five. When focused, contextual F1 must list all five complete literal values. Each fact row gets a pointer tooltip only when its own rendered value is actually truncated; widening the rail or syncing a shorter value clears it. Add long Unicode Where/Scope tests that prove the painted row ellipsizes while both truncated-only tooltip and focused F1 expose the complete value.

Expected IDs:

```python
(
    "console-send-authority-heading",
    "console-send-authority-where",
    "console-send-authority-scope",
    "console-send-authority-run",
    "console-send-authority-sources",
    "console-send-authority-approvals",
)
```

- [ ] **Step 3: Run the authority-summary RED tests**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_send_authority_summary.py \
  -q
```

Expected: FAIL because the module/widget does not exist.

- [ ] **Step 4: Implement the pure projection and fixed widget**

Create:

```python
@dataclass(frozen=True, slots=True)
class ConsoleSendAuthorityProjection:
    where: str
    scope: str
    run: str
    sources: str
    approvals: str

def project_console_send_authority(
    state: ConsoleInspectorState,
    *,
    ownership_policy: InspectorOwnershipPolicy = InspectorOwnershipPolicy.RESILIENT,
) -> ConsoleSendAuthorityProjection:
    owned = classify_inspector_content(state, ownership_policy)
    rows = {entry.row.label: entry.row for entry in owned.rows}

    workspace = str(rows.get("Workspace").value).strip() if rows.get("Workspace") else "Default"
    conversation_row = rows.get("Selected conversation")
    conversation = (
        str(conversation_row.value).strip()
        if conversation_row and str(conversation_row.value).strip() not in {"", "No active conversation"}
        else ("Temporary conversation" if state.ephemeral else "No active conversation")
    )

    scope_parts: list[str] = []
    if rows.get("Prefill (next send only)"):
        scope_parts.append("One-shot prefill")
    elif rows.get("Prefill (pinned)"):
        scope_parts.append("Pinned prefill")
    if state.scope_item_count:
        scope_parts.append(f"narrowed to {state.scope_item_count} items")

    provider = rows.get("Provider")
    source = rows.get("Sources") or rows.get("RAG/source")
    recovery_required = any(
        rows.get(label) is not None for label in ("Recovery action", "Next action")
    )
    if owned.incomplete:
        run = "Inspector data incomplete"
    elif recovery_required:
        run = "Recovery required"
    elif state.pending_approval_count > 0:
        run = "Waiting for approval"
    elif provider is not None and provider.status == "blocked":
        run = "Blocked"
    elif source is not None and source.status == "blocked":
        run = "Blocked"
    elif state.run_active:
        run = "Running"
    else:
        run = "Ready"

    where = f"{workspace} › {conversation}"
    if state.ephemeral:
        where = f"{where} · Temporary"

    return ConsoleSendAuthorityProjection(
        where=where,
        scope=" · ".join(scope_parts) or "Everything available",
        run=run,
        sources=(
            f"{state.staged_source_count} staged"
            if state.staged_source_count
            else "None staged"
        ),
        approvals=(
            f"{state.pending_approval_count} pending · action required"
            if state.pending_approval_count
            else "None pending"
        ),
    )
```

Classify the supplied state once. Read rows by stable label. Where combines `Workspace` and `Selected conversation`, with explicit Default/Temporary fallback. Scope combines one-shot/pinned prefill and the snapshot's typed narrowed-scope count. Run precedence is incomplete → recovery required → pending approval → provider/source blocked → running → ready. Sources and Approvals use the snapshot's typed counts.

`ConsoleSendAuthoritySummary` holds only the last `ConsoleInspectorState` and projection needed to skip equal syncs. Compose six fixed Statics and make the summary container one focusable unit. `sync_state()` computes one complete projection, patches all five Statics, recomputes per-row truncation tooltips, then refreshes once. `Resize` recomputes the same tooltips from Rich cell widths. It does not independently query any owner.

- [ ] **Step 5: Add pinned placement RED tests in the production right rail**

Update `Tests/UI/test_console_right_rail.py` boundary inventory. Move the existing one-row project-instruction control outside the scroll owner as well, so the approved order is exact. Assert the direct rail children are:

```python
(
    "console-inspector-rail-header",
    "console-project-instruction-status",
    "console-send-authority-summary",
    "console-inspector-rail-body",
    "console-inspector-outer-scroll-hint",
)
```

and that `console-send-authority-summary` is not a descendant of `#console-inspector-rail-body`. Scroll the outer body to its maximum and assert the summary's screen region remains unchanged. At narrow and wide production sizes, assert the summary region is six rows, fully inside the right rail, and neither edge/divider clips.

- [ ] **Step 6: Mount and synchronize the summary from the same screen snapshot**

In `ConsoleInspectorRail.compose()`, give the existing header `id="console-inspector-rail-header"`, then yield the existing `ConsoleProjectInstructionStatusRow` and `ConsoleSendAuthoritySummary(self._inspector_state)` before `_InspectorOuterBody`. The test oracle is that project control and summary are both outside the scroll owner, with the summary immediately above it.

In `_sync_console_control_bar()`, after building `inspector_state` once, query both widgets and pass that exact object to:

```python
summary.sync_state(inspector_state)
inspector.sync_state(inspector_state)
```

Do not rebuild the state between calls. Add a spy test asserting object identity and one build per sync tick. Extend contextual F1 assembly so focus anywhere on `#console-send-authority-summary` appends the five complete projection facts; this is the keyboard/full-value path for visually ellipsized Where and Scope rows.

- [ ] **Step 7: Write conditional-group and More RED tests**

In `Tests/UI/test_console_run_inspector.py`, test these projection tables:

| State | Ordinary sequence after Source Readiness | More contents |
|---|---|---|
| all empty | More | Tools, Approvals, Artifacts |
| tools nonzero | Tools, More | Approvals, Artifacts |
| approval pending | Approvals, More | Tools, Artifacts |
| artifact available | Artifacts, More | Tools, Approvals |
| all actionable | Tools, Approvals, Artifacts | none |

Assert fixed promotion order regardless of input row order. `Source Readiness` remains ordinary at zero.

Test More by real pointer and keys: click, Enter, and Space toggle; Left closes; Right opens. Assert it defaults closed and posts one `MoreToggled(open=...)` message only for deliberate user changes. Programmatic `set_more_open()` must post nothing.

- [ ] **Step 8: Define actionability with owner-specific, testable rules**

Add one pure helper in `console_run_inspector.py` rather than a generic readiness framework:

```python
def inspector_group_is_actionable(owner: str, owned: InspectorOwnedContent) -> bool:
    rows = owned.rows_for(owner)
    actions = owned.actions_for(owner)
    if any(action.enabled for action in actions):
        return True
    values = {str(entry.row.value).strip().lower() for entry in rows}
    if owner == "Tools":
        return any(value not in {"", "—", "0", "0 ready"} for value in values)
    if owner == "Approvals":
        return any(entry.row.status == "blocked" for entry in rows)
    if owner == "Artifacts":
        return any(
            value not in {"", "—", "none", "unavailable", "not available for this item"}
            for value in values
        )
    raise ValueError(owner)
```

Pin every currently emitted zero/unavailable spelling in tests. Do not infer all groups through fuzzy substring matching.

- [ ] **Step 9: Implement one More boundary inside `ConsoleRunInspector`**

Keep canonical ownership unchanged. Add a render projection that partitions only Tools, Approvals, and Artifacts into promoted and More-owned groups. Render promoted groups immediately after Source Readiness in fixed order. Render remaining ordinary groups after More exactly as today.

Implement a private `_ConsoleInspectorMore` in `console_run_inspector.py` rather than a general framework. It owns one focusable `Button` and one body `Vertical`; it accepts already-built group widgets, applies `display: none` when closed, and supports:

```python
BINDINGS = [
    Binding("left", "collapse", "Collapse", show=False),
    Binding("right", "expand", "Expand", show=False),
]
```

Enter/Space use the Button's native press path. Its `Toggled` message bubbles to `ConsoleRunInspector`, which updates local display and posts `ConsoleRunInspector.MoreToggled(open)` for persistence.

If no groups remain under More, omit the boundary. If focus was on the More toggle when it disappears, let the existing right-rail keyed recovery select the next valid Inspector boundary; do not auto-focus a promoted group.

- [ ] **Step 10: Remove exact lower duplicates while retaining added detail**

Filter `Selected conversation` and `Workspace` rows from the lower Selected Conversation group because the pinned Where fact owns them. Keep `Conversation source`, `Resume state`, and Prefill rows because they add persistence/next-send detail. Keep the detailed Run recipe, recovery rows, Evidence/Authority, and actionable group rows because they add information or controls beyond the compact facts.

Update `_rendered_row_entries()` and `_structural_key()` to use the same filtered/partitioned projection as compose; in-place patches must target exactly the mounted ids. The structural key must include the ordered promoted-owner tuple, ordered More-owner tuple, and More-boundary presence in addition to row/action ids. Promotion can leave flattened row ids unchanged while moving their parent, so row ids alone are not a safe in-place-update fingerprint. More's open flag is not structural because `set_more_open()` changes body display in place.

- [ ] **Step 11: Add focus-demotion RED tests**

Mount More open, focus an enabled approval/artifact action, then sync to an empty/disabled state. Assert:

1. If the same descendant id remains mounted and focusable under open More, focus stays on it.
2. If it disappears or disables, focus goes to that group's focusable heading inside open More.
3. If More is closed, focus goes directly to `#console-inspector-more-toggle`.
4. No case moves focus to transcript, composer, or another Inspector group.
5. Promotion never proactively moves focus to the promoted group.

- [ ] **Step 12: Implement keyed focus recovery and scoped More persistence**

Before a structural recompose, capture `(owner, focused_widget_id)` only when focus is inside a conditional group. After refresh:

```python
if more_open and same_id_is_focusable:
    focus(same_id)
elif more_open and demoted_heading_is_visible:
    focus(demoted_heading)
else:
    focus(CONSOLE_INSPECTOR_MORE_TOGGLE_ID)
```

Make conditional headings focusable only while inside open More; ordinary headings retain current focus behavior.

Handle `ConsoleRunInspector.MoreToggled` in `ChatScreen` by calling:

```python
self._set_console_rail_preference(
    section_updates={CONSOLE_INSPECTOR_MORE_DISCLOSURE_ID: event.open},
    notify_on_failure=False,
)
```

Pass `rail_state.inspector_more_open` into the initial right-rail/Inspector composition and programmatically sync it from `_sync_console_rail_visibility()` without reposting the user event. This automatically follows the selected Global/Per workspace persistence key from Task 2.

- [ ] **Step 13: Run the full Inspector slice and regenerate CSS**

Run:

```bash
../../.venv/bin/python -B tldw_chatbook/css/build_css.py
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_send_authority_summary.py \
  Tests/UI/test_console_run_inspector.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_css_bundle_sync_guard.py \
  -q
```

Expected: PASS. The production boundary test must show the six-row summary above/outside the scrolling body, and conditional group tests must show no actionable group hidden under closed More.

- [ ] **Step 14: Commit the Inspector slice**

```bash
git add \
  Tests/UI/test_console_send_authority_summary.py \
  Tests/UI/test_console_run_inspector.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_console_rail_reconciliation.py \
  tldw_chatbook/Widgets/Console/console_send_authority_summary.py \
  tldw_chatbook/Widgets/Console/console_run_inspector.py \
  tldw_chatbook/Chat/console_display_state.py \
  tldw_chatbook/Widgets/Console/__init__.py \
  tldw_chatbook/UI/Console_Modules/right_rail.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "feat(console): pin next-send authority"
```

### Task 4: Documentation, review, and proportional verification

**Files:**
- Modify: `Docs/User_Guide/console/sessions-tabs-workspaces.md:90-130`
- Modify: `backlog/tasks/task-20937.6 - Verify-and-document-Console-edge-rails-and-workspace-ownership.md`
- Review: `backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md`

- [ ] **Step 1: Update the Console user guide**

Replace immediate label-click wording with a compact table documenting:

- single click selects; collapsed workspace also expands;
- double-click or Enter activates;
- Space and Left/Right control disclosure;
- Load more/Retry remain immediate;
- selected versus active glyph meaning;
- full-label context row, truncation-only pointer tooltip, and contextual F1;
- Global default and Per workspace opt-in under Settings → Console Behavior;
- exactly what layout scope persists and what remains transient;
- the pinned `What happens if I send now?` five facts;
- why zero Tools/Approvals/Artifacts sit under More and when they promote.

Keep the established workspace ownership, Default/unassigned Conversations, starred-first ordering, local/outer scrolling, 15/20/35 ceilings, and Character contain wording intact.

- [ ] **Step 2: Run focused formatting and static checks**

Run Ruff only over changed Python files:

```bash
../../.venv/bin/python -B -m ruff check \
  tldw_chatbook/Chat/console_rail_state.py \
  tldw_chatbook/Chat/console_display_state.py \
  tldw_chatbook/config.py \
  tldw_chatbook/UI/Screens/settings_screen.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/UI/Console_Modules/left_rail.py \
  tldw_chatbook/UI/Console_Modules/right_rail.py \
  tldw_chatbook/Widgets/Console/console_workspace_tree.py \
  tldw_chatbook/Widgets/Console/console_workspace_context.py \
  tldw_chatbook/Widgets/Console/console_send_authority_summary.py \
  tldw_chatbook/Widgets/Console/console_run_inspector.py \
  Tests/UI/test_console_workspace_tree.py \
  Tests/UI/test_console_workspace_context_rail.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_console_send_authority_summary.py \
  Tests/UI/test_console_run_inspector.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/Chat/test_console_rail_state.py \
  Tests/Chat/test_console_rail_state_prune.py \
  Tests/test_config_console_defaults.py

../../.venv/bin/python -B -m ruff format --check \
  tldw_chatbook/Chat/console_rail_state.py \
  tldw_chatbook/Chat/console_display_state.py \
  tldw_chatbook/config.py \
  tldw_chatbook/UI/Screens/settings_screen.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/UI/Console_Modules/left_rail.py \
  tldw_chatbook/UI/Console_Modules/right_rail.py \
  tldw_chatbook/Widgets/Console/console_workspace_tree.py \
  tldw_chatbook/Widgets/Console/console_workspace_context.py \
  tldw_chatbook/Widgets/Console/console_send_authority_summary.py \
  tldw_chatbook/Widgets/Console/console_run_inspector.py \
  Tests/UI/test_console_workspace_tree.py \
  Tests/UI/test_console_workspace_context_rail.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_console_send_authority_summary.py \
  Tests/UI/test_console_run_inspector.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/Chat/test_console_rail_state.py \
  Tests/Chat/test_console_rail_state_prune.py \
  Tests/test_config_console_defaults.py
```

Expected: both commands exit 0.

- [ ] **Step 3: Run the complete changed-functionality gate**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_workspace_tree.py \
  Tests/UI/test_console_workspace_context_rail.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_console_edge_rail_geometry.py \
  Tests/UI/test_console_rail_section_layout.py \
  Tests/UI/test_console_rail_sections.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_console_send_authority_summary.py \
  Tests/UI/test_console_run_inspector.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/Chat/test_console_rail_state.py \
  Tests/Chat/test_console_rail_state_prune.py \
  Tests/test_config_console_defaults.py \
  Tests/UI/test_css_build_integrity.py \
  Tests/UI/test_css_bundle_sync_guard.py \
  -q

../../.venv/bin/python -B -m compileall -q \
  tldw_chatbook/Chat/console_rail_state.py \
  tldw_chatbook/Chat/console_display_state.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/UI/Screens/settings_screen.py \
  tldw_chatbook/UI/Console_Modules/left_rail.py \
  tldw_chatbook/UI/Console_Modules/right_rail.py \
  tldw_chatbook/Widgets/Console/console_workspace_tree.py \
  tldw_chatbook/Widgets/Console/console_workspace_context.py \
  tldw_chatbook/Widgets/Console/console_send_authority_summary.py \
  tldw_chatbook/Widgets/Console/console_run_inspector.py

../../.venv/bin/python -B -m tldw_chatbook.css.check_bundle_sync
git diff --check
```

Expected: all tests pass, compile/check commands exit 0, the generated CSS bundle matches sources, and `git diff --check` prints nothing. Do not replace this focused gate with the unrelated full repository suite.

- [ ] **Step 4: Perform the four-persona UAT checklist in iTerm2**

At the same reported row/column sizes used by the existing task evidence, record per-persona observations:

1. First-time non-technical: can distinguish selected vs active, discover Enter open, understand Where/Scope/Run/Sources/Approvals, and find empty telemetry under More.
2. First-time technical: can predict click/double-click/Enter, inspect full truncated labels, and understand Global vs Per workspace.
3. Regular non-technical: can switch workspaces without rail-layout surprise and recover from blocked/pending state.
4. Regular technical power user: can use keyboard-only Tree/More grammar, retain focus during promotion/demotion, and verify no transient scroll/search/selection state persisted.

Also repeat the reflow click at overflowing and non-overflowing Tree geometry and verify the 15/20/35 ceilings, both edge dividers, and Character contain behavior remain unchanged.

- [ ] **Step 5: Obtain equivalent Windows Terminal evidence before closeout**

Use the same commit and equivalent reported rows/columns. Record the same checklist and captures. Physical pixels are not an oracle. If Windows evidence is missing or divergent, leave `TASK-20937.6` In Progress and record the exact open gate; do not mark Done based on iTerm2 alone.

- [ ] **Step 6: Self-review against the approved spec and ADR**

Diff every changed file and explicitly check:

- no single click activates a workspace/conversation;
- no timer or coordinate re-resolution was added;
- shared and legacy records are retained;
- responsive overrides never write preferences;
- summary and Inspector consume the same `ConsoleInspectorState` object;
- summary is exactly six rows and outside the scroll owner;
- actionable groups cannot remain under closed More;
- focus recovery never escapes Inspect;
- no footer/F1 copy advertises an unimplemented key;
- ADR-083 still matches code and no second ADR is needed.

- [ ] **Step 7: Update task evidence without overclaiming closeout**

Check AC #8, #9, and #10 only after their automated and terminal evidence passes. Add concise Implementation Notes naming the three commits, exact focused test commands/results, UAT evidence paths, persistence compatibility, and any deviation from this plan.

Do not check AC #1–#6 or mark the task Done unless all pre-existing benchmark, documentation, cross-terminal, child-task, and cleanliness requirements are also satisfied.

- [ ] **Step 8: Commit documentation and evidence**

```bash
git add \
  Docs/User_Guide/console/sessions-tabs-workspaces.md \
  'backlog/tasks/task-20937.6 - Verify-and-document-Console-edge-rails-and-workspace-ownership.md'
git commit -m "docs(console): document rail interaction authority"
```

- [ ] **Step 9: Verify the final boundary is clean**

Run:

```bash
git status --short
git log --oneline -5
```

Expected: no tracked product/spec/task changes remain unstaged or uncommitted. The pre-existing untracked `.impeccable/critique/...` artifact is not part of these implementation commits unless the task's UAT evidence section explicitly adopts it.
