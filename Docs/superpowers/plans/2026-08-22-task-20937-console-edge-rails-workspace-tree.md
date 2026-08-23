# Console Edge Rails and Workspace Tree Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Context and Inspect the Console's application edges, give Context sections stable 15/20/35-row natural ceilings, organize named-workspace conversations in a native Tree, keep Default/unassigned conversations flat, and contain Character art within a complete 35-row body.

**Architecture:** Retain the existing Console screen, controller, bounded-section, persistence, and responsive owners. Replace only Context's shared sibling-height allocator with per-section ceilings plus ordinary outer scrolling; introduce one pure workspace-Tree projection and one small native Textual Tree adapter; keep the flat Default/unassigned browser in its existing state family; split search/page concurrency at the controller boundary. Character fitting remains in the existing image pipeline and adds one measured, equality-bounded row budget rather than a second image system.

**Tech Stack:** Python 3.12, Textual 8.2.8, Rich `Text`, pytest/Textual `run_test`, TCSS and the canonical CSS builder, Backlog.md.

**Approved design:** `Docs/superpowers/specs/2026-08-22-console-edge-rails-workspace-tree-design.md`

**Backlog parent:** `TASK-20937`

**Backlog editing rule:** Backlog CLI 1.44.0 has a documented five-digit task
addressing corruption trap. Create IDs with the guarded creation workflow, but
update TASK-20937 and its children by editing their canonical Markdown files
directly. Verify them through `backlog task list --plain` plus `git status
backlog/`; never invoke `backlog task edit` against these five-digit IDs.

**ADR required:** yes

**ADR path:** `backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md`

**Reason:** This changes long-lived Console information ownership, focus exceptions, scrolling policy, and native Tree interaction while narrowly superseding clauses of ADR-017 and ADR-077.

**Live environments:** iTerm2 verification runs on the implementation macOS
host. The named Windows operator is Robert (the request owner), using his
existing Windows Terminal host that reproduced the full 15-row behavior. Task
6 records its Windows/Windows Terminal/Python/Textual versions and uses the
same reported rows/columns as iTerm2 with the capture checklist. If Robert or
that host is unavailable, Task 6 remains blocked rather than waiving parity.

---

## File responsibility map

### Existing production owners

- `tldw_chatbook/Widgets/Console/console_bounded_section.py` — one bounded local body, instance ceiling, hint, native scrolling, focus recovery.
- `tldw_chatbook/UI/Console_Modules/rail_section_layout.py` — retain only pure local/outer hint and Context focus-fallback policy; delete shared budget allocation.
- `tldw_chatbook/UI/Console_Modules/left_rail.py` — Context section descriptors, stable DOM order, local-first/outer reconciliation, outer cue/focus/offset continuity.
- `tldw_chatbook/Workspaces/conversation_browser_state.py` — flat Default/unassigned Conversations projection after Starred/Workspaces groups retire.
- `tldw_chatbook/UI/Console_Modules/workspace.py` — service loading, separate search lanes, per-workspace page attempts, selection/resume/star persistence.
- `tldw_chatbook/Widgets/Console/console_workspace_context.py` — active workspace chrome, flat Conversations UI, Tree host and messages.
- `tldw_chatbook/UI/Screens/chat_screen.py` — screen event routing, Tree/flat mutation seams, Character control measurement and avatar rebuild.
- `tldw_chatbook/UI/Console_Modules/wiring.py` — explicit controller dependencies only; no second state owner.
- `tldw_chatbook/css/components/_agentic_terminal.tcss` — edge/divider geometry, Tree/pinned chrome, Character containment.
- `tldw_chatbook/css/tldw_cli_modular.tcss` — generated canonical bundle; never hand-edit.

### New focused owners

- `tldw_chatbook/Workspaces/workspace_tree_state.py` — immutable, literal-data workspace/child projection and starred-first ordering; no Textual or service I/O.
- `tldw_chatbook/Widgets/Console/console_workspace_tree.py` — the thin `Tree` adapter: hidden root, literal labels, glyph fallback, key guards, stable keyed synchronization, and messages.
- `Tests/Workspaces/test_workspace_tree_state.py` — pure ownership/order/search/page-state tests.
- `Tests/UI/test_console_workspace_tree.py` — native Tree interaction/focus/update tests.
- `Tests/UI/test_console_edge_rail_geometry.py` — exact production CSS edge/divider/15-20-35 geometry matrix.
- `Tests/UI/test_console_workspace_tree_performance.py` — deterministic old/new adapter and report-only measurement harness, created before Task 2 production edits.
- `Tests/UI/fixtures/console_workspace_tree_old_baseline.json` — immutable pre-change raw samples, environment metadata, fixture digest, and old materialized/service counts.
- `Docs/superpowers/reports/2026-08-22-console-workspace-tree-performance.md` — raw old baseline plus final median/p95/node/reconcile evidence.

### Documentation and governance

- `backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md`
- `backlog/decisions/README.md`
- `backlog/tasks/task-20937*.md`
- `Docs/User_Guide/console.md`
- `Docs/User_Guide/console/context-and-rag.md`

---

## Task 1 — TASK-20937.1: parameterize Context ceilings and retire shared allocation

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_bounded_section.py`
- Modify: `tldw_chatbook/UI/Console_Modules/rail_section_layout.py`
- Modify: `tldw_chatbook/UI/Console_Modules/left_rail.py`
- Modify: `Tests/UI/test_console_bounded_section.py`
- Modify: `Tests/UI/test_console_rail_section_layout.py`
- Modify: `Tests/UI/test_console_rail_reconciliation.py`
- Modify: `Tests/UI/test_console_left_rail.py`
- Modify: `Tests/UI/test_console_right_rail.py`
- Modify: `backlog/tasks/task-20937.1 - Parameterize-Context-section-ceilings-and-retire-shared-height-allocation.md`

- [ ] **Step 1: Move TASK-20937.1 to In Progress and add its task-local plan**

Direct-edit the canonical TASK-20937.1 Markdown file: set `status: In
Progress` and add its five-step implementation plan before code changes. Do
not address this five-digit task through Backlog CLI 1.44.0. Verify the file is
the only intended task change with `git status --short backlog/`, then confirm
the board sees it in `backlog task list --plain`.

- [ ] **Step 2: Write ceiling RED tests before production edits**

Add parameterized harness cases for `(ceiling, demand, expected viewport,
hint)`: `(15,14,14,false)`, `(15,15,15,false)`, `(15,16,15,true)`,
`(20,19,19,false)`, `(20,20,20,false)`, `(20,21,20,true)`,
`(35,34,34,false)`, `(35,35,35,false)`, and `(35,36,35,true)`. Reuse the
existing `_Harness`/multiline `Static` helpers so the oracle is
physical `content_region.height`, viewport `max_scroll_y`, and painted hint
copy—not a mocked desired-row field.

Add a real `ConsoleLeftRail` test with every section open that asserts each body gets `min(D, descriptor.ceiling)`, the outer viewport scrolls, no header title contains `· no room`, no `[>]` exists, and open preferences/messages remain unchanged.

Add or retain a `ConsoleInspectorRail` production-path 20/21 case in
`test_console_right_rail.py` proving the shared default remains exactly 20
content rows plus the separate hint after the constructor is parameterized.

- [ ] **Step 3: Run RED and record the intended failures**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_bounded_section.py \
  Tests/UI/test_console_rail_section_layout.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_console_left_rail.py \
  Tests/UI/test_console_right_rail.py -q
```

Expected: failures show the fixed-20 constructor rejection and the old shared-budget/no-room allocation behavior—not collection/setup errors.

- [ ] **Step 4: Parameterize the bounded body minimally**

In `ConsoleBoundedSection`:

```python
if isinstance(max_content_lines, bool) or not isinstance(max_content_lines, int):
    raise TypeError("max_content_lines must be an integer")
if max_content_lines <= 0:
    raise ValueError("max_content_lines must be positive")
self.max_content_lines = max_content_lines

def _normalize_allocation(self, allocation: int | None) -> int | None:
    if allocation is None:
        return None
    if isinstance(allocation, bool) or not isinstance(allocation, int):
        raise TypeError("allocation must be an integer or None")
    if allocation < 0:
        raise ValueError("allocation must be non-negative")
    return min(allocation, self.max_content_lines)
```

Keep the default at 20 so every Inspector caller remains byte-for-byte compatible. Do not add configuration or preference state for ceilings.

- [ ] **Step 5: Replace shared-budget policy with descriptor ceilings**

Add `max_content_lines` to `CONTEXT_SECTION_DESCRIPTORS` using the approved table. Delete `_MAX_CONTENT_ROWS`, `ContextAllocationResult`, `no_room`, active row distribution, and short-height base-row allocation if they have no remaining consumer. Retain the pure `local_hint_required`, `outer_hint_required`, and active-section fallback helpers used by focus recovery.

`ConsoleLeftRail` should request each local reconcile first, leave allocation `None` (its own ceiling), then derive the outer cue from the committed complete-section geometry. Remove `_no_room_section_ids`, title suffix/tooltip mutation, `[>]` routing, and fallback-entry reveal. Preserve explicit section opening, focus ownership, deliberate reveal, offset clamping, and one coalesced outer pass.

- [ ] **Step 6: Run GREEN and mutation checks**

Run the Step 3 command. Then temporarily restore the fixed-20 constructor guard and confirm at least the 15/35 cases fail; restore production immediately. Temporarily route Context through the old budget allocator and confirm the all-open test fails; restore.

Expected final: all focused cases pass with only known dependency warnings.

- [ ] **Step 7: Run scoped static checks and commit**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Widgets/Console/console_bounded_section.py \
  tldw_chatbook/UI/Console_Modules/rail_section_layout.py \
  tldw_chatbook/UI/Console_Modules/left_rail.py \
  Tests/UI/test_console_bounded_section.py \
  Tests/UI/test_console_rail_section_layout.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_console_left_rail.py \
  Tests/UI/test_console_right_rail.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Widgets/Console/console_bounded_section.py \
  tldw_chatbook/UI/Console_Modules/rail_section_layout.py \
  tldw_chatbook/UI/Console_Modules/left_rail.py \
  Tests/UI/test_console_bounded_section.py \
  Tests/UI/test_console_rail_section_layout.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_console_left_rail.py \
  Tests/UI/test_console_right_rail.py
git diff --check
git add \
  tldw_chatbook/Widgets/Console/console_bounded_section.py \
  tldw_chatbook/UI/Console_Modules/rail_section_layout.py \
  tldw_chatbook/UI/Console_Modules/left_rail.py \
  Tests/UI/test_console_bounded_section.py \
  Tests/UI/test_console_rail_section_layout.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_console_left_rail.py \
  Tests/UI/test_console_right_rail.py \
  'backlog/tasks/task-20937.1 - Parameterize-Context-section-ceilings-and-retire-shared-height-allocation.md'
git commit -m "feat(console): apply section-specific Context ceilings"
```

Check all TASK-20937.1 ACs, add concise Implementation Notes, and move it to Done only after the commit checks are clean.

---

## Task 2 — TASK-20937.2: split workspace and Default conversation projections

**Files:**

- Create: `tldw_chatbook/Workspaces/workspace_tree_state.py`
- Create: `Tests/Workspaces/test_workspace_tree_state.py`
- Create: `Tests/UI/test_console_workspace_tree_performance.py`
- Create: `Tests/UI/fixtures/console_workspace_tree_old_baseline.json`
- Create: `Docs/superpowers/reports/2026-08-22-console-workspace-tree-performance.md`
- Modify: `tldw_chatbook/Workspaces/conversation_browser_state.py`
- Modify: `tldw_chatbook/Workspaces/display_state.py`
- Modify: `tldw_chatbook/Workspaces/__init__.py`
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `Tests/UI/test_console_workspace_controller.py`
- Modify: `Tests/UI/test_console_workspace_lifecycle.py`
- Modify: `Tests/Workspaces/test_console_conversation_browser_state.py`
- Modify: `backlog/tasks/task-20937.2 - Split-workspace-and-default-conversation-projections.md`

- [ ] **Step 1: Start TASK-20937.2 and add its task-local plan**

Direct-edit the canonical child task file before code changes, following the
five-digit rule above. Keep this task UI-free: it owns immutable projection and
controller concurrency only.

- [ ] **Step 2: Write pure ownership/order REDs**

Define expected public shapes in tests before the module exists:

```python
@dataclass(frozen=True, slots=True)
class WorkspaceTreeConversation:
    conversation_id: str
    title: str
    starred: bool
    updated_sort: str
    selected: bool
    run_marker: str

@dataclass(frozen=True, slots=True)
class WorkspaceTreeWorkspace:
    workspace_id: str
    label: str
    conversations: tuple[WorkspaceTreeConversation, ...]
    next_cursor: int | None
```

Test named/default/unassigned partition, starred-first then recency, active markers, atomic owner movement, duplicate input IDs, literal labels, and deterministic ordering.

- [ ] **Step 3: Capture the reproducible old-projection baseline before production edits**

Create the deterministic small/representative/stress fixtures and an old-state
adapter around the existing conversation-browser projection. Run three
unreported warm-ups and 20 measured iterations for initial projection, 5%
marker update, search apply/clear, and active-row selection. Commit the raw
samples plus median, p95, total service-record count, materialized-row count,
reconcile/recompose count, Python/Textual version, terminal size, machine,
fixture seed, source commit, and a SHA-256 fixture digest to the immutable JSON
baseline and report. This step must run before changing
`conversation_browser_state.py`; Task 6 validates that frozen file and appends
new Tree-adapter measurements using the same generated records and protocol.

Run the old adapter node before any production edit and copy its emitted raw
samples verbatim into the report:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_workspace_tree_performance.py::test_old_projection_baseline_is_reproducible \
  -q -s
```

After this capture, final-tree tests load and checksum the frozen JSON; they do
not invoke changed production through an "old" adapter. Deleting or changing
the baseline metadata/digest must fail its validation test.

- [ ] **Step 4: Write controller interleaving REDs**

Use deletion-sensitive fakes for two search lanes and two page attempts:

- Workspaces query A, Conversations query B, A finishes late: B remains unchanged.
- Workspace page cursor 75 fails, Retry generation replaces it, stale failure finishes: Retry state wins.
- Two requests for cursor 75 overlap: one page commits once by conversation ID.
- Search hit exists outside loaded children: result parent/row appears.
- Membership moves while a page is in flight: stale page is discarded and the record appears once in its new owner.

- [ ] **Step 5: Run meaningful RED**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Workspaces/test_workspace_tree_state.py \
  Tests/Workspaces/test_console_conversation_browser_state.py \
  Tests/UI/test_console_workspace_tree_performance.py \
  Tests/UI/test_console_workspace_controller.py \
  Tests/UI/test_console_workspace_lifecycle.py -q
```

Expected: missing new state module and current shared-lane/Starred projection assertions fail. Reject import-only RED once the module skeleton exists; capture behavioral failures before implementation.

- [ ] **Step 6: Implement the pure projection**

Keep `workspace_tree_state.py` free of Textual, timers, services, and mutable widgets. Use one stable-ID dedupe at the boundary. The flat builder in `conversation_browser_state.py` returns only Default/unassigned rows and removes the Starred and Workspaces sections. Both builders use a shared small starred-first sort key; do not duplicate records to express stars.

- [ ] **Step 7: Split controller state without a generic framework**

Use explicit private state:

```python
self._workspace_tree_search = _SearchAttemptState()
self._flat_conversation_search = _SearchAttemptState()
self._workspace_page_attempts: dict[str, _PageAttemptState] = {}
```

Each completion compares its captured generation and full request key before commit. A page attempt key includes workspace ID, membership token, query token, and cursor. Keep the existing star lock and persistence worker; do not create a second star writer.

- [ ] **Step 8: Run GREEN, mutation evidence, and commit**

Run the Step 5 suite. Mutate the workspace-search completion to write the flat lane and prove the cross-search test fails. Remove page-generation validation and prove the Retry test fails. Restore, run scoped Ruff/format/diff checks, commit:

```bash
git commit -m "feat(console): split workspace conversation projections"
```

Close TASK-20937.2 with exact counts and no speed claim.

---

## Task 3 — TASK-20937.3: move the rails to application edges

**Files:**

- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss` (generated)
- Modify: `tldw_chatbook/UI/Console_Modules/frame.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (existing Console frame/focus call sites and comments only)
- Modify: `Tests/UI/test_console_shell_regions.py`
- Modify: `Tests/UI/test_console_resize_reflow.py`
- Modify: `Tests/UI/test_console_transcript_region.py`
- Modify: `Tests/UI/test_console_keyboard_trust.py`
- Modify: `Tests/UI/test_non_obscuring_focus_contract.py`
- Modify: `Tests/UI/test_console_internals_decomposition.py`
- Create: `Tests/UI/test_console_edge_rail_geometry.py`
- Modify: `Tests/UI/test_css_build_integrity.py`
- Modify: `backlog/tasks/task-20937.3 - Move-Console-rails-to-the-application-edges.md`

- [ ] **Step 1: Start TASK-20937.3 and capture production-CSS RED**

Mount the full real ChatScreen hierarchy with `CSS_PATH = TldwCli.CSS_PATH`. At representative expanded, collapsed, focused, 100/120/150-column, and short-height cases assert:

```python
assert grid.content_region.x == 0
assert left.region.x == grid.content_region.x
assert right.region.right == grid.content_region.right
assert transcript.region.x == left.region.right
assert right.region.x == transcript.region.right
```

Sample compositor cells on both boundaries and assert exactly one divider owns each cell. Record RED against current grid padding/full borders/rounded transcript.

- [ ] **Step 2: Make the smallest truthful source-TCSS and inline-frame change**

Remove `padding: 0 1` and side framing from `#console-workspace-grid`. Assign divider ownership once (rail edge or transcript edge, never both). Remove the full `#console-left-rail:focus` border and rounded transcript frame; express focus through color/text/background on existing geometry. Preserve top/bottom separation from global chrome.

TCSS alone cannot establish this ownership: `frame_console_region()` and
`_paint_console_rail_focus_frame` write inline borders that outrank TCSS in
Textual 8.2.8. Narrow those existing seams to explicit edge ownership and
update only their Console call sites/comments. Use the smallest local API
needed (for example, an explicit `edges` tuple); do not expand this into a
generic framing framework. The grid owns top/bottom only, Context and its
collapsed handle own right only, Inspect and its collapsed handle own left
only, and the transcript owns neither divider. Focus may repaint the existing
Context-right/Inspect-left divider and add a stable non-color label/control cue,
but it must never restore removed border edges or change geometry.

- [ ] **Step 3: Rebuild the canonical bundle**

```bash
../../.venv/bin/python -B -m tldw_chatbook.css.build_css
```

Restore any unrelated whitespace-only generated artifact. Never hand-edit `tldw_cli_modular.tcss`.

- [ ] **Step 4: Run GREEN and responsive controls**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_edge_rail_geometry.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_console_resize_reflow.py \
  Tests/UI/test_css_build_integrity.py -q
```

Assert regions are identical focused/unfocused and ADR-043 explicit-open/single-pane behavior remains unchanged.

The correction pass also preserves the original transcript, keyboard-trust,
non-obscuring-focus, and decomposition regression intents while replacing
their obsolete full-frame expectations with ADR-081 single-divider ownership.
These existing regression files are in the Task 3 map because the edge-rail
implementation intentionally invalidates their old border contract; leaving
them unchanged would make the approved behavior fail the broader focused gate.
Add same-pilot resize compositor coverage across 150 → 120 → 100 and short
height, plus focused collapsed-handle and transcript title-cue coverage.

- [ ] **Step 5: Mutation-check and commit**

Temporarily restore one grid inset or the transcript rounded border and prove containment/divider ownership fails. Restore, run scoped Ruff/format/diff, and commit:

```bash
git commit -m "style(console): move rails to the application edges"
```

---

## Task 4 — TASK-20937.4: render the native Workspace conversation Tree

**Files:**

- Create: `tldw_chatbook/Widgets/Console/console_workspace_tree.py`
- Create: `Tests/UI/test_console_workspace_tree.py`
- Modify: `tldw_chatbook/Widgets/Console/console_bounded_section.py`
- Modify: `Tests/UI/test_console_bounded_section.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py`
- Modify: `tldw_chatbook/UI/Console_Modules/left_rail.py`
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss` (generated)
- Modify: `Tests/UI/test_console_workspace_context_rail.py`
- Modify: `Tests/UI/test_console_workspace_keyboard.py`
- Modify: `Tests/UI/test_console_rail_reconciliation.py`
- Modify: `Tests/UI/test_console_left_rail.py`
- Modify: `backlog/tasks/task-20937.4 - Render-the-native-Workspace-conversation-Tree.md`

- [ ] **Step 1: Start TASK-20937.4 and write native interaction REDs**

Cover hidden-root construction, `auto_expand=False`, literal `Text`, two-cell guides, Unicode/ASCII vocabulary, workspace vs conversation selection, disclosure, top-level Left no-op, shifted hidden-root guards, leaf/empty Right no-op, `s` editable exclusion, and non-overflow Tab focus.

Add a bounded-section RED proving an injected native `ScrollView` is the direct
local scroll owner: the bounded root contains that scroll owner plus the
separate hint and contains no `BoundedSectionViewport`. The ordinary default
mode must still create exactly one `BoundedSectionViewport` and preserve every
Inspector contract.

- [ ] **Step 2: Write geometry and lifecycle REDs**

Use real fixed chrome and the bounded Workspaces body:

- one workspace node hugs natural height;
- demand ≥8 yields ≥8 Tree rows while body ≤20 and chrome ≤12;
- long identity is one ellipsized row with full tooltip;
- contextual Star row cannot intercept search typing;
- Tree is the sole local scroll owner and hands wheel to Context at its boundary;
- keyed title/marker/star updates preserve node object identity and cursor;
- removal recovers same-owner next/previous, then header/outer.

- [ ] **Step 3: Run meaningful RED**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_workspace_tree.py \
  Tests/UI/test_console_bounded_section.py \
  Tests/UI/test_console_workspace_context_rail.py \
  Tests/UI/test_console_workspace_keyboard.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_console_left_rail.py -q
```

- [ ] **Step 4: Implement the thin adapter**

First extend `ConsoleBoundedSection` with one narrow optional
native-scroll-owner mode. In that mode it yields the supplied `ScrollView`
directly instead of creating `BoundedSectionViewport`, measures and clamps that
owner's virtual geometry against the instance ceiling, and keeps ownership of
only the sibling hint. Its default path remains unchanged. The Tree selects
this mode and the interactive focus policy; no other Context or Inspector
caller does.

`ConsoleWorkspaceTree(Tree[WorkspaceTreeNodeData])` must configure native behavior rather than reimplement rendering:

```python
self.show_root = False
self.auto_expand = False
self.guide_depth = 2
```

Override only the key actions whose target could be the hidden root and the glyph constants needed for fallback. Post small messages for workspace selected, conversation selected, star toggled, load-more, and retry. The host/controller remains the service and persistence owner.

- [ ] **Step 5: Replace only the Workspaces group UI**

Keep active identity and Switch/New/RAG Scope above independent Workspaces search. Mount the Tree directly as the Workspaces local scroll owner—never inside a second `VerticalScroll`. Keep the existing flat row widgets for Conversations and remove the old Starred/Workspaces grouped browser rendering/event routes.

- [ ] **Step 6: Implement keyed synchronization and temporary search disclosure**

Maintain `workspace_id -> TreeNode` and `conversation_id -> TreeNode` maps. Update labels/data in place; add/remove/move only for structural changes. Snapshot persisted expansion before the first active query, force parents for hits, ignore persistence writes during search, and restore exactly on clear. Preserve scroll/cursor unless deliberate selection requests reveal.

- [ ] **Step 7: Rebuild CSS and run GREEN**

Style the pinned strip, identity, Tree, contextual Star action, literal status rows, local hint, and focus owner without adding borders. Rebuild CSS, then run the Step 3 suite plus `Tests/UI/test_css_build_integrity.py`.

- [ ] **Step 8: Mutation checks and commit**

Prove tests fail if `Tree.can_focus` is tied to overflow, `auto_expand` is restored, the hidden-root guard is removed, or one search writes disclosure preferences. Restore and commit:

```bash
git commit -m "feat(console): add workspace conversation Tree"
```

---

## Task 5 — TASK-20937.5: fit Character in a stable 35-row body

**Files:**

- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Console_Modules/left_rail.py`
- Modify: `tldw_chatbook/UI/Console_Modules/character.py` only if it already owns the relevant state seam
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss` (generated)
- Modify: `Tests/UI/test_console_character_avatar.py`
- Modify: `Tests/UI/test_console_character_controller.py`
- Modify: `Tests/UI/test_console_rail_reconciliation.py`
- Modify: `backlog/tasks/task-20937.5 - Fit-Character-content-within-a-stable-35-row-section.md`

- [ ] **Step 1: Start TASK-20937.5 and write exact geometry REDs**

Use production hierarchy/CSS and portrait, landscape, square, very large, missing, corrupt, and unsupported inputs. Assert image + name + reaction state + action + margins are contained by the initial 35-row viewport; valid image aspect ratio is preserved in terminal-cell geometry and controls are visible without initial scroll.

- [ ] **Step 2: Add settle/deletion-sensitive REDs**

Change width, control wrapping, image, and scrollbar conditions. Count computed box changes, image replacements, local reconciles, and outer reconciles. Require one box update and at most one follow-up; disabling the post-measure fit must fail the 35-row containment test, while forcing unconditional rebuild must fail the loop/count test.

- [ ] **Step 3: Run RED**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_character_avatar.py \
  Tests/UI/test_console_character_controller.py \
  Tests/UI/test_console_rail_reconciliation.py -q
```

- [ ] **Step 4: Implement measured contain fitting**

Retain `character_avatar_box` only as the width/fallback helper needed elsewhere. At the mounted Character owner, measure non-image rows and stable content width, then calculate available rows as `max(0, 35 - controls_with_margins)`. When available rows or content width is zero, omit/hide the image and preserve the controls/recovery copy without calling `fit_image_cell_size`, `scale_image_for_cell_box`, or `mosaic_from_image`. Otherwise call the existing fit helpers with `fit="contain"`. Store the last `(source_identity, width, available_rows, fitted_box)` signature and update only on inequality.

- [ ] **Step 5: Bound reconciliation**

Schedule one post-refresh measurement after relevant mutation. If the fitted box changes, update/remount the image and allow one ordinary local-to-outer reconcile. The follow-up observes equality and stops. Never use a timer, sleep, or unbounded fixed-point retry.

- [ ] **Step 6: Run GREEN, rebuild CSS, and commit**

Run Step 3 plus `Tests/UI/test_console_left_rail.py` and `Tests/UI/test_css_build_integrity.py`. Mutation-check crop/stretch and unconditional rebuild. Run scoped static/diff checks and commit:

```bash
git commit -m "feat(console): contain Character art within 35 rows"
```

---

## Task 6 — TASK-20937.6: production verification, performance, docs, and closeout

**Files:**

- Modify: `Tests/UI/test_console_workspace_tree_performance.py`
- Verify without modifying: `Tests/UI/fixtures/console_workspace_tree_old_baseline.json`
- Modify: `Docs/superpowers/reports/2026-08-22-console-workspace-tree-performance.md`
- Modify: `Tests/UI/test_console_edge_rail_geometry.py`
- Modify: `Tests/UI/test_console_workspace_tree.py`
- Modify: `Tests/UI/test_console_rail_reconciliation.py`
- Modify: `Tests/UI/test_console_resize_reflow.py`
- Modify: `Tests/UI/test_css_build_integrity.py`
- Modify: `Docs/User_Guide/console.md`
- Modify: `Docs/User_Guide/console/context-and-rag.md`
- Modify: `backlog/tasks/task-20937.6 - Verify-and-document-Console-edge-rails-and-workspace-ownership.md`
- Modify: `backlog/tasks/task-20937 - Make-Console-rails-edge-native-and-organize-conversations-by-workspace.md`

- [ ] **Step 1: Start TASK-20937.6 and assemble the exact focused gate**

List only changed-functionality files/nodes from Tasks 1–5. Include policy, bounded widget, Context outer reconciliation, shell geometry, pure projection, controller concurrency, Tree lifecycle/keyboard, Character, resize, CSS integrity, and relevant legacy workspace/Inspector contracts. Do not run unrelated repository suites.

- [ ] **Step 2: Complete the deterministic performance harness and append new results**

Generate the exact design datasets (3×4+4, 12×12+20, 50×75+75) with fixed IDs/titles/markers/search hits. Validate the frozen old baseline's seed, source commit, raw samples, environment metadata, and SHA-256 digest without invoking changed production as the old implementation. For the new projection, perform three unreported warm-ups and 20 measured iterations of initial projection/mount, 5% marker update, search apply/clear, and active-row selection using `time.perf_counter`. Record median, p95, total service-record count, materialized node count, reconcile/recompose count, Python/Textual version, terminal size, and machine.

Reuse the raw old-projection baseline recorded before Task 2 production edits;
do not reconstruct it against changed code. Timing remains report-only. Assert
deterministic results/counts and one logical reconcile; do not assert wall-clock
thresholds in CI. If the representative new median is more than 20% slower,
investigate and either correct it or record explicit acceptance in Task 6 and
parent notes before closeout. Make no unsupported speed claim.

- [ ] **Step 3: Run the exact focused gate and static checks**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Workspaces/test_workspace_tree_state.py \
  Tests/Workspaces/test_console_conversation_browser_state.py \
  Tests/UI/test_console_bounded_section.py \
  Tests/UI/test_console_rail_section_layout.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_console_left_rail.py \
  Tests/UI/test_console_edge_rail_geometry.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_console_resize_reflow.py \
  Tests/UI/test_console_workspace_tree.py \
  Tests/UI/test_console_workspace_tree_performance.py \
  Tests/UI/test_console_workspace_context_rail.py \
  Tests/UI/test_console_workspace_keyboard.py \
  Tests/UI/test_console_workspace_controller.py \
  Tests/UI/test_console_workspace_lifecycle.py \
  Tests/UI/test_console_character_avatar.py \
  Tests/UI/test_console_character_controller.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_css_build_integrity.py -q
```

Then run Ruff check/format only on changed Python files and `git diff --check`.

- [ ] **Step 4: Run live iTerm2 verification**

At agreed row/column sizes, capture:

- both rails expanded and edge-aligned;
- all Context sections open with 15/20/35 boundaries and local/outer cues;
- Workspaces Tree selection, disclosure, search, paging, star, and Default route;
- Character portrait and landscape contain behavior;
- collapse/reopen and resize focus/offset continuity.

Record terminal version, reported size, command, screenshots/captures, and observed regions in task notes.

- [ ] **Step 5: Run the same Windows Terminal checklist**

The user/operator runs the same build/commit and reported row/column sizes on the existing Windows environment. Compare cell geometry and behavior, not physical window pixels. A missing or divergent result keeps TASK-20937.6 open.

- [ ] **Step 6: Update user documentation**

Keep `context-and-rag.md` the detailed authority and `console.md` concise. Document ownership, starred-first order, independent search, Tree keys/disclosure/paging, Default route, local versus outer cues, section ceilings, and complete Character contain behavior. Do not expose generations, selectors, benchmark dimensions, or implementation vocabulary to users.

- [ ] **Step 7: Review, close children/parent, and commit**

Use `superpowers:requesting-code-review` for a final spec/quality review of the complete feature range. Recheck every task/ADR link and task ID across fetched remotes/worktrees. Update Task 6 notes, check its ACs, mark it Done; then check parent ACs, add concise parent Implementation Notes and exact evidence, and mark TASK-20937 Done only when every child is Done.

```bash
git add \
  Tests/UI/test_console_workspace_tree_performance.py \
  Docs/superpowers/reports/2026-08-22-console-workspace-tree-performance.md \
  Tests/UI/test_console_edge_rail_geometry.py \
  Tests/UI/test_console_workspace_tree.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_console_resize_reflow.py \
  Tests/UI/test_css_build_integrity.py \
  Docs/User_Guide/console.md \
  Docs/User_Guide/console/context-and-rag.md \
  'backlog/tasks/task-20937.6 - Verify-and-document-Console-edge-rails-and-workspace-ownership.md' \
  'backlog/tasks/task-20937 - Make-Console-rails-edge-native-and-organize-conversations-by-workspace.md'
git commit -m "docs(console): complete edge rails and workspace Tree"
```

Expected: worktree clean, no unreviewed generated drift, no unsupported performance claim, and no unresolved Windows verification blocker.

---

## Execution order and review gates

1. TASK-20937.1 — bounded policy foundation.
2. TASK-20937.2 — pure ownership and async lanes (may be implemented independently after Task 1 review, but do not merge Tree UI first).
3. TASK-20937.3 — shell edge geometry after Task 1 to avoid overlapping rail/CSS edits.
4. TASK-20937.4 — Tree integration after Tasks 1–3.
5. TASK-20937.5 — Character sizing after Tasks 1 and 3; it may run in parallel with Task 4 only in a separate worktree/branch because both touch `left_rail.py`, `chat_screen.py`, tests, and CSS.
6. TASK-20937.6 — final integration evidence and closeout after Tasks 4–5.

Every task uses TDD, a meaningful runtime RED, mutation/deletion sensitivity for the load-bearing assertion, a focused changed-functionality gate, scoped static checks, one independent review, and one commit boundary. Rebase on latest `origin/dev` before each merge; regenerate CSS after rebases rather than hand-merging the bundle.
