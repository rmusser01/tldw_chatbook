# TASK-3070.11 Wave 6 Closeout Amendment Implementation Plan

> **For Codex:** Execute this characterization plan before starting either production
> extraction. Amendment base `d20dd733b72148818f4491943136edfa68494c68`
> remains valid across unrelated `dev` commits; stop if a rebase changes the
> characterized `ChatScreen` source, candidate spans, or immutable budgets.

**Goal:** Replace the invalidated Wave 6 docs-only closeout with reviewed,
source-inspected evidence and an atomic successor sequence that clears both immutable
Console ratchets without raising either budget.

**Architecture:** Preserve every completed Wave 6 owner. Add one test-only closeout
inventory at the final delivery SHA, one binding amendment, two serial extraction
tasks, and one replacement closeout task. No production module changes in this task.

**Tech stack:** Python AST, pytest, Git revision reads, Backlog.md, Markdown.

**ADR required:** no

**ADR path:** `backlog/decisions/068-console-text-selection-and-annotations.md`

**Reason:** The task records an amendment under `DESIGN.md` section 7 and preserves
ADR-068's existing screen-owned review-note workflow; it introduces no new runtime
boundary.

---

## Task 1: Freeze the invalidated closeout evidence

**Files:**

- Create: `Tests/Architecture/test_console_wave6_closeout_inventory.py`
- Reference: `Tests/Architecture/test_screen_size_ratchet.py`
- Reference: `Tests/Architecture/test_console_wave6_inventory.py`

1. Add exact constants for final Wave 6 delivery base
   `87791f85533d883341a6b52489660c9e1a67223d` at 19,863 / 630, current amendment
   base `d20dd733b72148818f4491943136edfa68494c68` at 19,884 / 632, and the
   immutable 17,727 / 593 ceilings.
2. Source-inspect the existing post-image base and every TASK-3070.3-.10 delivery
   boundary used to calculate -4,958 / -130 task-local reduction and the current
   amendment base used to calculate +2,670 / +50 concurrent growth.
3. Assert the arithmetic yields +2,670 / +50 concurrent growth and the current
   2,157 / 39 deficit. Do not edit the
   production ratchet.
4. Run the new node and confirm it passes against the frozen revisions.

## Task 2: Lock candidate ownership and conservative margin

**Files:**

- Modify: `Tests/Architecture/test_console_wave6_closeout_inventory.py`

1. Define exact move/delegate/stay method-name sets for the 57-method realtime family
   and 26-method review/selection family; explicitly exclude provider selection.
2. Assert every name exists exactly once on the frozen `ChatScreen`, total spans are
   1,997 and 1,114 lines, the groups are disjoint, and the classifications are exactly
   56/0/1 and 15/4/7.
3. Count every stay's full 19/438-line source span and each delegate at the five-line
   ceiling. Assert the 19/458-line maximum residue projects at least 2,634 lines and
   71 methods removed.
4. Pin ADR-068's `on_console_review_notes_requested` and
   `_console_review_notes_flow` as review/selection stays.
5. Pin the three event delegates to their exact Textual `@on` bindings and the
   trajectory delegate to its `Binding` action; retain no realtime delegate because
   none has a framework binding or external caller.
6. Add a non-vacuity check that rejects a missing candidate and a projection equal to
   either current deficit.
7. Run the focused inventory file and the existing Wave 6 inventory file.

## Task 3: Amend the records and successor sequence

**Files:**

- Create: `Docs/superpowers/specs/2026-08-23-console-decomposition-wave6-closeout-amendment.md`
- Modify: `Docs/superpowers/specs/2026-08-13-console-decomposition-wave6-design.md`
- Modify: `backlog/tasks/task-3070 - chat_screen-size-ratchet-red-on-dev-after-console-decomposition-wave-3.md`
- Modify: `backlog/tasks/task-3070.11 - Characterize-Wave-6-concurrent-growth-and-amend-closeout.md`
- Create: `backlog/tasks/task-3070.12 - Extract-Console-realtime-orchestration-ownership.md`
- Create: `backlog/tasks/task-3070.13 - Extract-Console-review-and-selection-workflow-ownership.md`
- Create: `backlog/tasks/task-3070.14 - Close-amended-Console-decomposition-ratchet.md`

1. Record measured evidence, chosen boundaries, residue, task order, focused-only
   local verification, and the unchanged no-budget-increase rule.
2. Keep future task plans absent until each task is moved to In Progress.
3. Verify each child is atomic, testable, assigned, parented, and dependent only on
   an earlier task.
4. Verify the parent names TASK-3070.1-.14 and assigns final closure only to .14.

## Task 4: Review and close characterization

**Files:** all files above.

1. Request an independent specification/plan review and correct every valid finding.
2. Run the passing focused architecture gates:

   ```bash
   .venv/bin/python -m pytest -q \
     Tests/Architecture/test_console_wave6_closeout_inventory.py \
     Tests/Architecture/test_console_wave6_inventory.py
   ```

3. Record the unchanged ratchet as separate expected-RED characterization evidence:

   ```bash
   .venv/bin/python -m pytest -q Tests/Architecture/test_screen_size_ratchet.py
   ```

   Expected: the production size-ceiling node remains RED at 19,884 > 17,727 while
   its slack node passes. This diagnostic is not a green TASK-3070.11 gate and must
   not be hidden.
4. Run targeted Ruff/format on the new Python test, `git diff --check`, and the backlog
   hygiene scan for malformed task filenames.
5. Check all TASK-3070.11 acceptance criteria, add concise implementation notes, and
   mark it Done. Do not start TASK-3070.12 until this PR is reviewed and merged.

No local full-suite run is authorized.
