# TASK-2702 Prompt Dirty-Navigation Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a dirty Library Prompt navigation veto explain itself and give every dirty Prompt editor a truthful `Discard changes` recovery.

**Architecture:** Mirror the existing Skill dirty-veto/discard pattern inside the existing Prompt canvas and `LibraryScreen`. Reuse the Prompt editor's current `dirty` state, in-place patch discipline, and clean-Back reset/list-return tail; add no new state owner, worker, modal, CSS, or shared abstraction.

**Tech Stack:** Python 3.12, Textual 8.x, pytest/Pilot, Ruff

**ADR required:** no  
**ADR path:** N/A  
**Reason:** Routine UX bug fix applying an existing Library pattern without changing persistence, service contracts, ownership, security, or long-lived structure.

---

### Task 1: Pin the dirty-veto and discard contracts RED-first

**Files:**
- Modify: `Tests/UI/test_library_prompts_canvas.py:7441-7463`

- [ ] **Step 1: Strengthen the mounted dirty-navigation test**

Record `app.notify`, keep the existing real SQLite Prompt/editor flow, and assert the fixed warning exactly once at warning severity while the dirty author field and dirty flag remain unchanged:

```python
notifications: list[tuple[str, dict[str, object]]] = []
app.notify = lambda message, **kwargs: notifications.append((message, kwargs))

allowed = await screen.flush_pending_work()

assert allowed is False
assert screen.query_one("#library-prompt-author", Input).value == "Changed mid switch"
assert screen._library_prompt_dirty is True
assert notifications == [
    (
        "Unsaved Prompt changes — Save or Discard changes first.",
        {"severity": "warning"},
    )
]
```

- [ ] **Step 2: Add the clean/save state and no-convert compatibility discard test**

Use the real Prompt database/service and mounted canvas. Prove `#library-prompt-discard` is disabled with a reason on a clean editor, enables without remounting after a metadata edit, and re-disables after a successful save. Add a compatibility-only structured artifact with blank System/User lanes; prove Update/Convert are disabled, metadata can still become dirty, Discard is enabled, and pressing it returns to the list without persisting the edit.

- [ ] **Step 3: Run the exact tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_library_prompts_canvas.py \
  -q -k 'flush_pending_work_vetoes_dirty_editor or prompt_discard'
```

Expected: failures because `#library-prompt-discard` and the Prompt warning do not exist.

### Task 2: Implement the minimal Prompt sibling of the Skill pattern

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_prompts_canvas.py:50-75,980-1010`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:555-585,5994-6030,18250-18325,18588-18615,19000-19025,19290-19345`

- [ ] **Step 1: Add literal Prompt discard copy and render the action**

Define content-free clean/dirty tooltips beside the Prompt canvas constants. Render `Discard changes` in the existing editor action region for every non-mutation editor state:

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

- [ ] **Step 2: Add the fixed warning and live button patcher**

Add `LIBRARY_PROMPT_DIRTY_VETO_COPY`, `_notify_prompt_dirty_veto()`, and `_set_library_prompt_discard_enabled()`. The patcher updates both `disabled` and tooltip, matching the existing Skill implementation. Invoke it on the existing dirty false→true and save-success true→false targeted-update paths; do not recompose fields.

- [ ] **Step 3: Notify only when the Prompt barrier refuses app navigation**

In `flush_pending_work()`, after awaiting `_flush_library_prompt_save()`, call `_notify_prompt_dirty_veto()` only when that result is false. Keep the combined return expression and every note/skill barrier unchanged.

- [ ] **Step 4: Add the explicit discard handler**

Handle `#library-prompt-discard` at the screen seam. Refuse during mutation or when clean; otherwise reset the editor, request the current Prompt browse scope, refresh the local snapshot, and arm first-row focus—the established clean-Back exit tail—with no persistence call.

- [ ] **Step 5: Run focused GREEN verification**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_library_prompts_canvas.py \
  Tests/UI/test_library_skills_canvas.py \
  -q -k 'flush_pending_work_vetoes_dirty_editor or prompt_discard or flush_pending_work_skill_veto_notifies'
```

Expected: all selected tests pass.

- [ ] **Step 6: Mutation-check both new boundaries**

Temporarily bypass the Prompt notifier and confirm the dirty-navigation test fails. Restore it. Temporarily leave Discard disabled after dirty marking and confirm the compatibility test fails. Restore it and rerun both tests green.

- [ ] **Step 7: Commit the behavior**

```bash
git add \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  Tests/UI/test_library_prompts_canvas.py
git commit -m "fix(library): explain dirty Prompt navigation veto"
```

### Task 3: Verify and close out TASK-2702

**Files:**
- Modify: `backlog/tasks/task-2702 - Library-unsaved-prompt-silently-blocks-screen-navigation.md`

- [ ] **Step 1: Run bounded affected verification**

Run the full Prompt canvas file once plus the focused Skill sibling:

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py -q
../../.venv/bin/python -m pytest \
  Tests/UI/test_library_skills_canvas.py::test_library_flush_pending_work_skill_veto_notifies \
  -q
```

Classify any failure against unchanged `origin/dev`; do not weaken unrelated tests.

- [ ] **Step 2: Run static and UI-hardening gates**

```bash
../../.venv/bin/ruff check \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  Tests/UI/test_library_prompts_canvas.py
../../.venv/bin/ruff format --check Tests/UI/test_library_prompts_canvas.py
../../.venv/bin/python -m py_compile \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py
git diff --check
node /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.agents/skills/impeccable/scripts/detect.mjs \
  --json \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  Tests/UI/test_library_prompts_canvas.py
```

Use changed-range Ruff formatting for the two monolithic production files if their unchanged whole-file baselines remain unformatted.

- [ ] **Step 3: Complete task hygiene**

Check all three ACs, add concise Implementation Notes with exact test/static evidence and the ADR decision, and set TASK-2702 to Done without using a CLI operation that strips the existing task sections. No lesson entry is expected unless implementation exposes a new generalizable incident.

- [ ] **Step 4: Commit closeout**

```bash
git add \
  'backlog/tasks/task-2702 - Library-unsaved-prompt-silently-blocks-screen-navigation.md' \
  Docs/superpowers/plans/2026-08-13-task-2702-prompt-dirty-navigation.md
git commit -m "docs(library): complete TASK-2702"
```
