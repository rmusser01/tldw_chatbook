# Post-UX Roadmap Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Operationalize the post-UX product roadmap by mapping it onto the existing product-maturity tracker, Backlog task hierarchy, QA evidence model, and public roadmap docs without duplicating already-verified Phase 1/2 work.

**Architecture:** This is a docs and planning handoff slice, not a product feature implementation. It keeps `Docs/superpowers/specs/2026-05-06-post-ux-product-roadmap-design.md` as the strategy source, updates the product-maturity tracker as the execution source of truth, and creates public roadmap text as a derived view. Existing `TASK-10` through `TASK-13` remain the parent task structure unless the tracker explicitly supersedes them.

**Tech Stack:** Markdown, Backlog.md task files/MCP, pytest markdown contract tests, existing `Docs/superpowers/` tracker and QA conventions.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-05-06-post-ux-product-roadmap-design.md`
- Product tracker: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Product tracker pointer: `backlog/docs/product-maturity-roadmap.md`
- Current layout contract spec: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- Current Phase 3 task: `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`
- Current Phase 3.0 task: `backlog/tasks/task-10.0 - Product-Maturity-Phase-3.0-Destination-Layout-And-IA-Contracts.md`
- Future parent tasks: `backlog/tasks/task-11 - Product-Maturity-Phase-4-Agent-Configuration-And-Execution.md`, `backlog/tasks/task-12 - Product-Maturity-Phase-5-Server-Parity-And-Live-Integrations.md`, `backlog/tasks/task-13 - Product-Maturity-Phase-6-Release-Hardening-And-Documentation.md`

## Scope Check

This plan covers only the first post-UX operationalization slice:

- confirm current UX/UI and Phase 3.0 layout-contract work is complete before starting.
- map the post-UX roadmap onto existing `TASK-10` through `TASK-13`.
- add tracker language that prevents duplicate Product Maturity Phase 1/2 work.
- create QA evidence scaffolding for a post-UX rebaseline if the mapping says one is needed.
- publish a directional public roadmap page derived from the approved public roadmap view.
- add focused documentation regression coverage so the mapping does not drift.

This plan does not:

- implement Product Maturity Phase 3, Phase 4, Phase 5, or Phase 6 feature work.
- create a parallel Backlog phase tree.
- reopen verified Product Maturity Phase 1 or Phase 2 tasks unless a post-UX replay finds regressions.
- run the actual post-UX app replay before UX/UI completion is confirmed.

## Precondition Gate

Do not execute Task 1 until these are true:

- The current in-progress UX/UI work has been merged or explicitly closed.
- `TASK-10.0` is Done or the tracker explicitly records why the layout-contract gate is being deferred.
- `git status --short` is reviewed and unrelated untracked UX/UI files are left untouched unless they are part of the completed handoff.

If these are not true, stop after this plan and continue the active UX/UI workstream first.

## File Structure

- Create: `Tests/UI/test_post_ux_product_roadmap_handoff.py`
  - Markdown contract tests for the post-UX roadmap mapping, verified-work reuse rule, public roadmap page, and tracker handoff.

- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
  - Add a "Post-UX Roadmap Handoff" section mapping the roadmap stages to `TASK-10` through `TASK-13`.
  - Add explicit verified-work reuse rules for Product Maturity Phase 1/2.

- Modify: `backlog/docs/product-maturity-roadmap.md`
  - Add the post-UX roadmap spec as an additional planning source and explain that the tracker remains the execution source of truth.

- Create: `Docs/superpowers/qa/product-maturity/post-ux/README.md`
  - Define where post-UX rebaseline evidence belongs and when it should be created.

- Create: `Docs/superpowers/qa/product-maturity/post-ux/walkthrough-template.md`
  - Delta-focused evidence template for changed screens, changed workflows, changed layout contracts, and regressions.

- Create: `Docs/Product_Roadmap.md`
  - Public, directional roadmap with no dates, task IDs, or delivery commitments.

- Modify: Backlog task files only if needed after tracker mapping:
  - `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`
  - `backlog/tasks/task-11 - Product-Maturity-Phase-4-Agent-Configuration-And-Execution.md`
  - `backlog/tasks/task-12 - Product-Maturity-Phase-5-Server-Parity-And-Live-Integrations.md`
  - `backlog/tasks/task-13 - Product-Maturity-Phase-6-Release-Hardening-And-Documentation.md`

## Task 1: Add Failing Post-UX Roadmap Handoff Contract Tests

**Files:**
- Create: `Tests/UI/test_post_ux_product_roadmap_handoff.py`

- [ ] **Step 1: Write the failing markdown contract test**

Create `Tests/UI/test_post_ux_product_roadmap_handoff.py`:

```python
"""Post-UX product roadmap handoff documentation regressions."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = Path("Docs/superpowers/specs/2026-05-06-post-ux-product-roadmap-design.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
BACKLOG_POINTER = Path("backlog/docs/product-maturity-roadmap.md")
PUBLIC_ROADMAP = Path("Docs/Product_Roadmap.md")
POST_UX_QA_README = Path("Docs/superpowers/qa/product-maturity/post-ux/README.md")
POST_UX_QA_TEMPLATE = Path("Docs/superpowers/qa/product-maturity/post-ux/walkthrough-template.md")


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_post_ux_spec_preserves_tracker_handoff_rules():
    text = _text(SPEC)

    assert "Relationship To Existing Roadmaps" in text
    assert "Verified work reuse rule" in text
    assert "Operational handoff rule" in text
    assert "`TASK-10` through `TASK-13`" in text
    assert "Post-UX Reliability Rebaseline" in text
    assert "not reimplemented" in text


def test_product_tracker_maps_post_ux_roadmap_to_existing_tasks():
    text = _text(TRACKER)

    assert "Post-UX Roadmap Handoff" in text
    assert "Docs/superpowers/specs/2026-05-06-post-ux-product-roadmap-design.md" in text
    assert "Product Maturity Phase 1 and Phase 2 are not reopened" in text
    for task_id in ("TASK-10", "TASK-11", "TASK-12", "TASK-13"):
        assert task_id in text
    for stage in (
        "Post-UX Reliability Rebaseline",
        "Source, Knowledge, And Artifact Loops",
        "Controlled Agent Configuration And Run Loops",
        "Server Parity And Live Integrations",
        "Release Hardening And Distribution",
    ):
        assert stage in text


def test_backlog_pointer_names_execution_source_of_truth():
    text = _text(BACKLOG_POINTER)

    assert "2026-05-06-post-ux-product-roadmap-design.md" in text
    assert "canonical product-maturity tracker" in text
    assert "Do not create a parallel phase tree" in text


def test_post_ux_qa_scaffold_is_delta_focused():
    readme = _text(POST_UX_QA_README)
    template = _text(POST_UX_QA_TEMPLATE)

    assert "changed screens" in readme
    assert "changed workflows" in readme
    assert "changed layout contracts" in readme
    assert "discovered regressions" in readme
    assert "Pre-UX Baseline Evidence" in template
    assert "Post-UX Delta" in template
    assert "Regression Decision" in template


def test_public_roadmap_is_directional_and_commitment_free():
    text = _text(PUBLIC_ROADMAP)

    assert "local-first agentic knowledge console" in text
    assert "Now: Reliability And Product Confidence" in text
    assert "Next: Complete Workflow Loops" in text
    assert "Later: Server-Backed And Live Capabilities" in text
    assert "Always: Local-First Control" in text
    forbidden = ("ETA", "deadline", "will ship on", "TASK-", "Phase 1.1", "Phase 2.5")
    assert not any(term in text for term in forbidden)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_post_ux_product_roadmap_handoff.py --tb=short
```

Expected: FAIL because the tracker, public roadmap, and post-UX QA scaffold do not exist yet.

- [ ] **Step 3: Commit the failing test**

Run:

```bash
git add Tests/UI/test_post_ux_product_roadmap_handoff.py
git commit -m "Add post-UX roadmap handoff contract tests"
```

Expected: commit succeeds with only the new test file.

## Task 2: Map The Post-UX Roadmap Onto Existing Product-Maturity Tracking

**Files:**
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `backlog/docs/product-maturity-roadmap.md`

- [ ] **Step 1: Update the tracker with a post-UX handoff section**

Add this section after the current "Current Verified Baseline" section in `Docs/superpowers/trackers/product-maturity-roadmap.md`:

```markdown
## Post-UX Roadmap Handoff

Source Spec: `Docs/superpowers/specs/2026-05-06-post-ux-product-roadmap-design.md`
Status: pending UX/UI completion and tracker activation

This post-UX roadmap is a planning overlay for work that starts after the current UX/UI and destination layout-contract work is complete. It does not create a parallel task tree.

Product Maturity Phase 1 and Phase 2 are not reopened by default. Their existing QA evidence is the pre-UX baseline. New post-UX evidence should record only changed screens, changed workflows, changed layout contracts, or discovered regressions.

| Post-UX Roadmap Stage | Existing Backlog Owner | Execution Rule |
| --- | --- | --- |
| Post-UX Reliability Rebaseline | `TASK-10` only if Phase 3 UX/UI deltas require a distinct gate | Revalidate deltas against verified Phase 1/2 evidence; do not recreate Phase 1/2 child tasks. |
| Source, Knowledge, And Artifact Loops | `TASK-10` | Continue Phase 3 under existing Knowledge/Study ownership. |
| Controlled Agent Configuration And Run Loops | `TASK-11` | Create PR-sized child tasks under Phase 4 when ready. |
| Monitoring And Cross-Loop Recovery | `TASK-10` or `TASK-11` based on the concrete workflow owner | Do not create standalone recovery rewrites without a workflow gate. |
| Server Parity And Live Integrations | `TASK-12` | Prioritize parity by workflow value, not endpoint count. |
| Release Hardening And Distribution | `TASK-13` | Use only after earlier workflow gates have QA evidence. |
```

- [ ] **Step 2: Update tracker phase overview residual risks**

In the Phase 3 row, append:

```text
Post-UX roadmap execution should continue under TASK-10 unless the tracker explicitly creates a rebaseline child task for UX/UI deltas.
```

In the Phase 4 row, append:

```text
The post-UX roadmap maps controlled agent configuration and run loops to TASK-11.
```

In the Phase 5 row, append:

```text
The post-UX roadmap requires parity to be prioritized by workflow value rather than endpoint count.
```

In the Phase 6 row, append:

```text
The post-UX roadmap public-roadmap and release-readiness refresh belongs under TASK-13.
```

- [ ] **Step 3: Update the Backlog pointer doc**

Add the post-UX spec to `backlog/docs/product-maturity-roadmap.md`:

```markdown
The post-UX roadmap design lives at:

`Docs/superpowers/specs/2026-05-06-post-ux-product-roadmap-design.md`

The canonical product-maturity tracker remains the execution source of truth. Do not create a parallel phase tree from the post-UX roadmap; map new child tasks to the existing product-maturity parent tasks unless the tracker explicitly supersedes them.
```

- [ ] **Step 4: Run the focused test and verify remaining failures**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_post_ux_product_roadmap_handoff.py --tb=short
```

Expected: FAIL only for missing post-UX QA scaffold and public roadmap.

- [ ] **Step 5: Commit tracker and pointer updates**

Run:

```bash
git add Docs/superpowers/trackers/product-maturity-roadmap.md backlog/docs/product-maturity-roadmap.md
git commit -m "Map post-UX roadmap to product maturity tracker"
```

Expected: commit succeeds with only tracker and pointer docs changed.

## Task 3: Add Delta-Focused Post-UX QA Scaffold

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/post-ux/README.md`
- Create: `Docs/superpowers/qa/product-maturity/post-ux/walkthrough-template.md`

- [ ] **Step 1: Create post-UX QA README**

Create `Docs/superpowers/qa/product-maturity/post-ux/README.md`:

```markdown
# Post-UX Product Rebaseline QA

This folder stores post-UX rebaseline evidence created after the current UX/UI and destination layout-contract work is complete.

Use this folder only for:

- changed screens.
- changed workflows.
- changed layout contracts.
- discovered regressions.
- revalidation of verified Phase 1/2 behavior affected by UX/UI changes.

Do not recreate Product Maturity Phase 1 or Phase 2 evidence here unless a post-UX replay finds a regression or the tracker explicitly requests a rebaseline gate.

Every evidence file must link:

- the pre-UX baseline evidence.
- the UX/UI or layout-contract change being revalidated.
- the affected Backlog task.
- the pass/fail decision.
- any P0/P1 findings and follow-up owner.
```

- [ ] **Step 2: Create post-UX walkthrough template**

Create `Docs/superpowers/qa/product-maturity/post-ux/walkthrough-template.md`:

```markdown
# Post-UX Rebaseline Evidence

Date:
Branch/Commit:
Post-UX Roadmap Stage:
Affected Backlog Task:
Changed Screen/Workflow/Layout Contract:

## Pre-UX Baseline Evidence

- Existing tracker entry:
- Existing QA artifact:
- Existing test command:

## Post-UX Delta

- What changed:
- Why this needs revalidation:
- Screens or workflows exercised:

## Automated Evidence

- Command:
- Result:
- Notes:

## Running-App Walkthrough

- Environment:
- Terminal size:
- Entry path:
- Steps:
- Result:

## Visual/Focus Notes

## Empty/Error/Recovery Notes

## Regression Decision

- [ ] Revalidated with no regression
- [ ] Regression found and fixed
- [ ] Regression found and accepted with owner/follow-up
- [ ] Not applicable because no affected UX/UI delta exists

## Residual Risk
```

- [ ] **Step 3: Run the focused test and verify remaining failures**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_post_ux_product_roadmap_handoff.py --tb=short
```

Expected: FAIL only for missing public roadmap page.

- [ ] **Step 4: Commit QA scaffold**

Run:

```bash
git add Docs/superpowers/qa/product-maturity/post-ux
git commit -m "Add post-UX product rebaseline QA scaffold"
```

Expected: commit succeeds with only post-UX QA docs.

## Task 4: Add Directional Public Roadmap Page

**Files:**
- Create: `Docs/Product_Roadmap.md`

- [ ] **Step 1: Create public roadmap page**

Create `Docs/Product_Roadmap.md`:

```markdown
# Chatbook Product Roadmap

Chatbook is evolving into a local-first agentic knowledge console: a place to ingest sources, reason over them, configure controlled agentic work, monitor progress, and preserve useful outputs as durable artifacts.

This roadmap is directional. It describes current priorities and likely future areas, not delivery dates or commitments.

## Now: Reliability And Product Confidence

Current focus:

- first-run setup and configuration clarity.
- stable layouts across terminal sizes.
- keyboard-first navigation and focus behavior.
- understandable empty, error, and blocked states.
- clear local, server, workspace, and runtime authority labels.
- repeatable QA coverage for core workflows.

## Next: Complete Workflow Loops

Next focus:

- ask grounded questions over selected sources.
- turn answers into reusable Chatbooks and artifacts.
- organize knowledge with Workspaces and Collections.
- generate and reuse flashcards, quizzes, reports, and study outputs.
- configure personas, skills, tools, schedules, and workflows.
- launch and monitor controlled agent work through Console.
- recover cleanly when providers, runtimes, or optional capabilities are missing.

## Later: Server-Backed And Live Capabilities

Longer-term focus:

- richer source and RAG integration.
- server-assisted watchlists and collections.
- live status updates for running work.
- import and sync paths for sources, personas, skills, and artifacts.
- clearer collaboration between local and remote runtimes.
- documented residual gaps where local mode remains the better default.

## Always: Local-First Control

Across every area, Chatbook should keep source authority, runtime readiness, approvals, recovery paths, and generated outputs visible.

Console remains the live work surface. Other areas prepare, inspect, organize, resume, or preserve work.
```

- [ ] **Step 2: Run the focused test and verify it passes**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_post_ux_product_roadmap_handoff.py --tb=short
```

Expected: PASS.

- [ ] **Step 3: Commit public roadmap**

Run:

```bash
git add Docs/Product_Roadmap.md
git commit -m "Add public Chatbook product roadmap"
```

Expected: commit succeeds with only the public roadmap page.

## Task 5: Update Backlog Parent Tasks Only If The Tracker Mapping Requires It

**Files:**
- Modify only as needed:
  - `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`
  - `backlog/tasks/task-11 - Product-Maturity-Phase-4-Agent-Configuration-And-Execution.md`
  - `backlog/tasks/task-12 - Product-Maturity-Phase-5-Server-Parity-And-Live-Integrations.md`
  - `backlog/tasks/task-13 - Product-Maturity-Phase-6-Release-Hardening-And-Documentation.md`

- [ ] **Step 1: Inspect tracker mapping and existing task descriptions**

Run:

```bash
backlog task 10 --plain
backlog task 11 --plain
backlog task 12 --plain
backlog task 13 --plain
```

Expected: existing parent tasks are still compatible with the post-UX tracker mapping.

- [ ] **Step 2: Decide whether parent task edits are needed**

If the tracker mapping only refines sequencing, do not edit parent tasks.

If parent task descriptions or acceptance criteria now conflict with the tracker, update only the conflicting parent task fields. Preserve existing status, labels, priority, and completed child task history.

- [ ] **Step 3: Commit any Backlog task edits**

If edits were needed, run:

```bash
git add backlog/tasks
git commit -m "Align product maturity tasks with post-UX roadmap"
```

Expected: commit succeeds with only the necessary Backlog task files.

If no edits were needed, record that in the final handoff summary and do not make an empty commit.

## Task 6: Final Verification And Handoff

**Files:**
- No new files unless verification finds a documentation defect.

- [ ] **Step 1: Run focused documentation tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_post_ux_product_roadmap_handoff.py --tb=short
```

Expected: PASS.

- [ ] **Step 2: Run adjacent product tracker tests if present**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase1_harness.py Tests/UI/test_product_maturity_phase3_layout_contracts.py --tb=short
```

Expected: PASS, or skip/fail only if `Tests/UI/test_product_maturity_phase3_layout_contracts.py` has not been created yet in the active branch. If missing, record that Phase 3.0 implementation is still pending.

- [ ] **Step 3: Check markdown whitespace**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Confirm no unrelated files were staged**

Run:

```bash
git status --short
```

Expected: only intended docs/test files are modified or staged. Existing unrelated untracked UX/UI files remain untouched unless they were explicitly part of the completed handoff.

- [ ] **Step 5: Write final handoff summary**

Summarize:

- which post-UX roadmap stage maps to each existing Backlog parent task.
- whether a distinct post-UX rebaseline task is needed.
- which evidence paths now exist.
- which tests passed.
- which UX/UI or Phase 3.0 preconditions remain unresolved.

## Notes For Future Execution

- This plan is intentionally blocked until the current UX/UI and Phase 3.0 layout-contract work is complete or explicitly deferred.
- If execution starts before those preconditions are true, stop after Task 1 and update the tracker with a "blocked by active UX/UI work" note instead of creating new roadmap tasks.
- The first product-code implementation plan after this handoff should be scoped to one concrete workflow loop under `TASK-10` or `TASK-11`, not to the whole roadmap.
