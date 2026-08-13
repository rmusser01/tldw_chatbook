# TASK-2702 Prompt Dirty-Navigation Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or `superpowers:executing-plans` to
> implement this plan one RED→GREEN cycle at a time.

**Goal:** Make a dirty Library Prompt navigation veto explain itself and give
every dirty Prompt editor a truthful `Discard changes` recovery.

**Architecture:** Mirror the existing Skill dirty-veto/discard behavior in the
existing Prompt canvas and `LibraryScreen`. Reuse the Prompt editor's current
`dirty` state, in-place patch discipline, and clean-Back reset/list-return tail;
add no state owner, worker, modal, CSS, or shared abstraction.

**Tech Stack:** Python 3.12, Textual 8.x, pytest/Pilot, Ruff

**ADR required:** no
**ADR path:** N/A
**Reason:** Routine UX bug fix applying an existing Library pattern without
changing persistence, service contracts, ownership, security, or long-lived
structure.

---

### Task 1: Explain only a dirty Prompt navigation veto

**Files:**
- Modify: `Tests/UI/test_library_prompts_canvas.py:7440-7465`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:575-585,6003-6035`

- [x] **Step 1: Strengthen the existing mounted flush regression**

In
`test_library_prompt_flush_pending_work_vetoes_dirty_editor`, record
`app.notify`. First call `flush_pending_work()` while the editor is clean and
assert it returns `True` with no notification. Then edit the author, flush again,
and assert:

```python
assert allowed is False
assert screen.query_one("#library-prompt-author", Input).value == (
    "Changed mid switch"
)
assert screen._library_prompt_dirty is True
assert notifications == [
    (
        "Unsaved Prompt changes — Save or Discard changes first.",
        {"severity": "warning"},
    )
]
```

- [x] **Step 2: Run the exact test and verify RED**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_library_prompts_canvas.py::test_library_prompt_flush_pending_work_vetoes_dirty_editor \
  -q
```

Expected RED: the dirty refusal produces no notification. The clean assertion
must already pass, proving the future notifier is not unconditional.

- [x] **Step 3: Implement the fixed warning at the existing barrier**

Add the content-free constant
`LIBRARY_PROMPT_DIRTY_VETO_COPY = "Unsaved Prompt changes — Save or Discard changes first."`
and `_notify_prompt_dirty_veto()` beside the Skill sibling. In
`flush_pending_work()`, call it only when the awaited Prompt flush result is
false. Leave every note/skill barrier and the combined return expression
unchanged; notification failure must not change the fail-closed veto.

- [x] **Step 4: Rerun the exact test GREEN**

Run the Step 2 command and require one pass.

### Task 2: Render a truthful clean Discard action

**Files:**
- Modify: `Tests/UI/test_library_prompts_canvas.py:7897-7940`
- Modify: `tldw_chatbook/Widgets/Library/library_prompts_canvas.py:52-72,970-1020`
- Modify: `tldw_chatbook/Widgets/Library/__init__.py:10-20,60-80`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:341-360`

- [x] **Step 1: Pin the clean action and literal reason**

Extend
`test_library_prompt_editing_shows_unsaved_marker_and_save_clears_it` only as
far as the initial clean editor. Assert `#library-prompt-discard` is mounted,
disabled, and carries exactly:

```text
No unsaved Prompt changes to discard.
```

- [x] **Step 2: Run the exact test and verify RED**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_library_prompts_canvas.py::test_library_prompt_editing_shows_unsaved_marker_and_save_clears_it \
  -q
```

Expected RED: `#library-prompt-discard` is absent.

- [x] **Step 3: Add and export the Prompt tooltip constants, then render**

Define beside the Prompt canvas constants:

```python
PROMPT_DISCARD_TOOLTIP_CLEAN = "No unsaved Prompt changes to discard."
PROMPT_DISCARD_TOOLTIP_DIRTY = (
    "Return to the Prompt list without saving these changes."
)
```

Export both from `Widgets/Library/__init__.py`, import them through the existing
`Widgets.Library` screen seam, and render `Discard changes` in the fixed editor
action region for every normal, conflict, and compatibility editor:

```python
yield Button(
    "Discard changes",
    id="library-prompt-discard",
    classes="library-canvas-action",
    compact=True,
    disabled=self.mutation_in_flight or not self.dirty,
    tooltip=(
        PROMPT_DISCARD_TOOLTIP_DIRTY
        if self.dirty
        else PROMPT_DISCARD_TOOLTIP_CLEAN
    ),
)
```

Only mutation and a clean working copy disable this action.

- [x] **Step 4: Rerun the exact test GREEN through the clean assertions**

Run the Step 2 command and require the clean action assertions to pass.

### Task 3: Patch Discard live across dirty and saved transitions

**Files:**
- Modify: `Tests/UI/test_library_prompts_canvas.py:7897-7940`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:18240-18335,18610-19010`

- [x] **Step 1: Strengthen the existing no-recompose regression**

Capture the clean Discard widget identity. After editing, assert the same widget
instance is enabled with exactly:

```text
Return to the Prompt list without saving these changes.
```

After the existing real save reaches `Saved.`, assert the same widget instance
is disabled again with the clean tooltip. This node already proves the Prompt
meta and editor fields do not recompose.

- [x] **Step 2: Run the exact test and verify RED**

Run the Task 2 Step 2 command.

Expected RED: the mounted action stays in its clean disabled state after the
field change.

- [x] **Step 3: Implement one targeted Discard patcher**

Add `_set_library_prompt_discard_enabled()` beside
`_sync_library_prompt_save_action_widgets()`. It updates only `disabled` and
`tooltip`. Call it from both false→true dirty paths:

- `_mark_library_prompt_dirty()`; and
- `_capture_library_prompt_block_state()`.

Call it from the common successful create/update save settlement after
`_library_prompt_dirty = False`, so both branches re-disable the action. Paths
that already recompose project the correct state from `dirty` and need no second
mechanism.

- [x] **Step 4: Rerun the exact test GREEN**

Run the Task 2 Step 2 command and require one pass.

### Task 4: Make compatibility-only dirty editors escapable

**Files:**
- Modify: `Tests/UI/test_library_prompts_canvas.py` (new mounted regression near
  the Prompt dirty-state tests)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:19280-19335`

- [x] **Step 1: Add the compatibility dead-end regression**

Use a real Prompt database/service and mount a compatibility-only structured
artifact with blank System/User lanes. Prove `Update original` and
`Convert and save as new Prompt` are unavailable, edit metadata, and assert
Discard enables. Press it and prove all of the established clean-Back tail:

- no persistence call and the stored metadata remains unchanged;
- editor returns to the Prompt list;
- the current browse scope is requested exactly once;
- the local source snapshot refreshes exactly once; and
- deferred first-row focus lands on the first Prompt row.

- [x] **Step 2: Run the exact new node and verify RED**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_library_prompts_canvas.py::test_library_prompt_compatibility_editor_discard_returns_to_current_list \
  -q
```

Expected RED: pressing the mounted Discard action has no handler/list-return
effect.

- [x] **Step 3: Add the explicit screen handler**

Handle `#library-prompt-discard`. Refuse during mutation or while clean.
Otherwise reset the Prompt editor, request the current Prompt browse scope,
refresh the local source snapshot, and arm first-row focus using the same exact
operations and ordering as the clean Back tail. Do not persist and do not add a
confirmation modal or state.

- [x] **Step 4: Rerun the exact new node GREEN**

Run the Step 2 command and require one pass.

### Task 5: Mutation proofs and affected verification

- [x] **Step 1: Prove all behavior boundaries are non-vacuous**

Temporarily bypass the Prompt notifier; Task 1's exact dirty-flush node must go
RED while its clean assertion remains green. Restore it. Temporarily omit the
live Discard dirty patch; Task 3's exact node must go RED on the disabled action.
Restore it. Temporarily bypass the Discard handler; Task 4's exact node must go
RED on its unchanged editor/list assertions. Restore and rerun all three GREEN.

- [x] **Step 2: Run focused sibling verification**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_library_prompts_canvas.py \
  Tests/UI/test_library_skills_canvas.py \
  -q -k 'flush_pending_work_vetoes_dirty_editor or prompt_discard or editing_shows_unsaved_marker_and_save_clears_it or flush_pending_work_skill_veto_notifies'
```

- [x] **Step 3: Run the full Prompt canvas file once**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py -q
```

Classify any failure against unchanged `origin/dev`; do not weaken unrelated
tests.

- [x] **Step 4: Run static checks over every owned Python file**

```bash
../../.venv/bin/ruff check \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/Widgets/Library/__init__.py \
  Tests/UI/test_library_prompts_canvas.py
../../.venv/bin/ruff format --check tldw_chatbook/Widgets/Library/__init__.py
../../.venv/bin/ruff format --check --range=52-75 \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py
../../.venv/bin/ruff format --check --range=960-1040 \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py
../../.venv/bin/ruff format --check --range=330-365 \
  tldw_chatbook/UI/Screens/library_screen.py
../../.venv/bin/ruff format --check --range=545-590 \
  tldw_chatbook/UI/Screens/library_screen.py
../../.venv/bin/ruff format --check --range=5990-6045 \
  tldw_chatbook/UI/Screens/library_screen.py
../../.venv/bin/ruff format --check --range=18230-18340 \
  tldw_chatbook/UI/Screens/library_screen.py
../../.venv/bin/ruff format --check --range=18595-19015 \
  tldw_chatbook/UI/Screens/library_screen.py
../../.venv/bin/ruff format --check --range=19275-19340 \
  tldw_chatbook/UI/Screens/library_screen.py
../../.venv/bin/ruff format --check --range=7420-7485 \
  Tests/UI/test_library_prompts_canvas.py
../../.venv/bin/ruff format --check --range=7870-7985 \
  Tests/UI/test_library_prompts_canvas.py
../../.venv/bin/python -m py_compile \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/Widgets/Library/__init__.py
git diff --check
```

The exact changed ranges above are baseline-green before implementation. If
edits shift their end lines, widen only the affected range and record it. Do not
whole-file-format the three monolithic files.

- [x] **Step 5: Run the Impeccable detector once at final state**

```bash
node /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.agents/skills/impeccable/scripts/detect.mjs \
  --json \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/Widgets/Library/__init__.py \
  Tests/UI/test_library_prompts_canvas.py
```

### Task 6: Commit behavior and close out TASK-2702

- [x] **Step 1: Commit the verified behavior**

```bash
git add \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/Widgets/Library/__init__.py \
  Tests/UI/test_library_prompts_canvas.py
git commit -m "fix(library): explain dirty Prompt navigation veto"
```

- [x] **Step 2: Complete task hygiene**

Check all acceptance criteria, add concise Implementation Notes with exact
test/static evidence and the ADR decision, and set TASK-2702 to Done without a
CLI operation that strips existing task sections. Add a lesson only if a new,
generalizable incident actually occurred.

- [x] **Step 3: Commit closeout and verify the cumulative branch**

```bash
git add \
  'backlog/tasks/task-2702 - Library-unsaved-prompt-silently-blocks-screen-navigation.md' \
  Docs/superpowers/plans/2026-08-13-task-2702-prompt-dirty-navigation.md \
  Docs/superpowers/specs/2026-08-13-task-2702-prompt-dirty-navigation-design.md
git commit -m "docs(library): complete TASK-2702"
git diff --check origin/dev...HEAD
git status --short
```

The cumulative branch diff, not an empty post-commit working-tree diff, is the
authoritative whitespace gate.
