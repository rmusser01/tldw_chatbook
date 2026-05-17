# Post-Release UX/HCI Functional Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-audit the rendered app as an actual user, verify every top-level screen and cross-screen workflow, and convert discovered breakage into prioritized correction work before advancing deferred feature implementation.

**Architecture:** Treat current Phase 3-6 closeout as a historical baseline, not proof that the app is currently usable. The next tranche starts with evidence capture and actual-use validation, then turns findings into P0/P1/P2/P3 correction tasks and only then plans deferred features such as ACP runtime launch, write sync, Workspaces, and citation/snippet carry-through.

**Tech Stack:** Python 3.11+, Textual, textual-web/CDP when available, actual terminal screenshots when textual-web is insufficient, pytest documentation regressions, Backlog.md.

---

## File Structure

- Create: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/README.md` for the audit index and current acceptance status.
- Create: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/walkthrough-template.md` for reusable screen and workflow evidence.
- Create: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/<date>-<screen-or-workflow>.md` for each completed audit slice.
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md` to list `TASK-60` and child tasks as the active post-release validation tranche.
- Modify: `backlog/tasks/task-60*.md` as audit evidence is produced and follow-up tasks are created.
- Test: `Tests/UI/test_post_release_ux_hci_validation_plan.py` to prevent the audit from becoming untracked prose.

## Task 1: Create Actual-Screen Audit Harness

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/README.md`
- Create: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/walkthrough-template.md`
- Modify: `backlog/tasks/task-60.1 - Post-release-actual-screen-UX-HCI-audit-harness.md`
- Test: `Tests/UI/test_post_release_ux_hci_validation_plan.py`

- [ ] **Step 1: Write the failing documentation regression**

Add a test that asserts the plan, tracker, parent task, and audit-template files require actual rendered screenshots, actual-use verification, NN/g heuristic notes, CDP/textual-web or terminal evidence, severity classification, and user approval before a screen is accepted.

- [ ] **Step 2: Run the focused regression and confirm failure**

Run: `python -m pytest -q Tests/UI/test_post_release_ux_hci_validation_plan.py --tb=short`

Expected: FAIL because the QA index/template do not exist yet.

- [ ] **Step 3: Create the audit harness docs**

The template must include these required fields:

- Screen or workflow name
- Evidence method: textual-web/CDP, browser automation, terminal screenshot, or manual screenshot path
- Actual screenshot path
- Screenshot approval state: pending, approved, rejected
- Primary user goal
- Steps attempted
- What worked
- What broke
- NN/g heuristic findings
- Keyboard/focus findings
- Empty/error/setup-state findings
- Cross-screen handoff findings
- P0/P1/P2/P3 severity decisions
- Follow-up Backlog task links

- [ ] **Step 4: Verify the harness regression passes**

Run: `python -m pytest -q Tests/UI/test_post_release_ux_hci_validation_plan.py --tb=short`

Expected: PASS.

- [ ] **Step 5: Commit the harness slice**

Commit message: `docs: add post-release UX HCI audit harness`

## Task 2: Audit Every Top-Level Screen By Actual Use

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/<date>-screen-<destination>.md`
- Modify: `backlog/tasks/task-60.2 - Post-release-top-level-screen-functionality-audit.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`

- [ ] **Step 1: Start from a clean launch**

Run the app from the current branch with the same Python used by the repo virtual environment. Prefer textual-web and CDP for browser-observed screenshots. If textual-web cannot capture the real terminal rendering, use an actual terminal screenshot.

- [ ] **Step 2: Capture and review each screen**

Required destinations: Home, Console, Library, Artifacts, Personas, Watchlists, Schedules, Workflows, MCP, ACP, Skills, Settings.

For each destination, record the actual screenshot, whether the user approved it, the visible IA/layout issues, whether primary controls work, whether blocked states explain recovery, and whether keyboard/focus behavior supports fast use.

- [ ] **Step 3: Create follow-up tasks for every P0/P1**

P0 means the screen blocks basic use. P1 means a core workflow is seriously degraded. Do not close the screen audit while P0/P1 findings are only prose.

- [ ] **Step 4: Update task evidence**

Update `TASK-60.2` with the audit evidence index, completion state, and remaining findings.

- [ ] **Step 5: Commit the screen-audit slice**

Commit message: `docs: record post-release screen UX audit`

## Task 3: Validate Cross-Screen Workflows End-To-End

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/<date>-workflow-<name>.md`
- Modify: `backlog/tasks/task-60.3 - Post-release-cross-screen-workflow-validation.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`

- [ ] **Step 1: Select the minimum required workflows**

Required workflows:

- Home active work opens details and routes relevant work into Console or shows a clear blocked state.
- Library Search/RAG produces evidence or a recoverable blocked state, then hands context into Console where appropriate.
- Console can accept visible composer input, preserve pasted/collapsed text behavior, send or clearly block, and save/resume a Chatbook artifact.
- Artifacts and Chatbooks can reopen or resume into Console without dead controls.
- Personas, Skills, MCP, ACP, Watchlists, Schedules, and Workflows handoffs into Console are verified or explicitly classified as blocked future work with user-visible recovery copy.

- [ ] **Step 2: Drive each workflow in the actual app**

Use CDP/textual-web/browser automation where possible, but record actual screenshots for visual acceptance. Use terminal screenshots when browser rendering cannot prove the state.

- [ ] **Step 3: Record power-user friction**

For at least five repeated-use workflows, record shortcuts, repeated steps, state persistence, missing batch actions, recovery paths, and whether the workflow supports fast repeated use.

- [ ] **Step 4: Create correction tasks**

Every P0/P1 workflow failure must have a Backlog task with concrete acceptance criteria before `TASK-60.3` can be closed.

- [ ] **Step 5: Commit the workflow-validation slice**

Commit message: `docs: record post-release workflow validation`

## Task 4: Convert Deferred Features Into Evidence-Gated Tranches

**Files:**
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `Docs/Product_Roadmap.md`
- Modify: `backlog/tasks/task-60.4 - Post-release-deferred-feature-tranche-planning.md`
- Create: new Backlog child tasks only after audit evidence justifies their order.

- [ ] **Step 1: Read the audit findings before feature planning**

Do not prioritize deferred feature implementation ahead of unresolved P0/P1 usability failures.

- [ ] **Step 2: Create staged tranches**

Required deferred tranches:

- ACP runtime launch.
- Write sync promotion from dry-run/read-only surfacing to explicitly approved write behavior.
- Workspaces and deeper Library organization, including Library-owned collection membership and deeper Import/Export.
- Citation/snippet carry-through into Chat responses, artifacts, and exported Chatbooks.
- Optional dependency installation/recovery and package metadata polish.

- [ ] **Step 3: Link each tranche to evidence**

Each tranche must name the audit finding or residual-risk evidence that justifies it, plus the screens/workflows it affects.

- [ ] **Step 4: Commit the deferred-feature planning slice**

Commit message: `docs: plan post-release deferred feature tranches`

## Task 5: Close The Tracking PR

**Files:**
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `backlog/tasks/task-60*.md`
- Test: `Tests/UI/test_post_release_ux_hci_validation_plan.py`

- [ ] **Step 1: Run focused verification**

Run: `python -m pytest -q Tests/UI/test_post_release_ux_hci_validation_plan.py Tests/UI/test_post_ux_product_roadmap_handoff.py Tests/UI/test_product_maturity_phase6_release_closeout.py --tb=short`

- [ ] **Step 2: Run diff hygiene**

Run: `git diff --check`

- [ ] **Step 3: Update implementation notes**

Record what was planned, what was intentionally deferred, and why actual-use validation gates future work.

- [ ] **Step 4: Open PR against `dev`**

The PR body must state that this does not certify the app as usable. It creates the next validation tranche because the current app may still be visually or functionally broken despite prior mounted-test closeout.
