# Phase 6 Release Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the verified product-maturity work into release-candidate usability with full replay evidence, docs alignment, packaging/config validation, and explicit release-blocker decisions.

**Architecture:** Phase 6 is a release-hardening pass, not a feature-depth rewrite. It replays the already-built product loops, fixes only release-blocking defects found by the replay, and records remaining accepted risks with owners and follow-up tasks.

**Tech Stack:** Python 3.11+, Textual, pytest, Backlog.md, existing product-maturity QA evidence, Textual Web/CDP where visual QA requires actual rendered screenshots.

---

## Source Of Truth

- Parent task: `backlog/tasks/task-13 - Product-Maturity-Phase-6-Release-Hardening-And-Documentation.md`
- Product tracker: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Product maturity design: `Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md`
- Post-UX roadmap design: `Docs/superpowers/specs/2026-05-06-post-ux-product-roadmap-design.md`
- Layout contracts: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- Design system: `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`
- QA index: `Docs/superpowers/qa/product-maturity/phase-6/README.md`

## Phase 6 Scope Boundary

Phase 6 verifies release readiness of the current product surface. It may fix P0/P1 defects discovered during replay, but it must not reopen broad feature-depth work unless a release-blocking defect makes the product unusable.

In scope:

- full first-time user replay;
- power-user workflow replay across at least five core loops;
- keyboard/focus/accessibility replay;
- visual polish against the approved terminal design system;
- error/recovery copy review;
- docs/onboarding/help alignment;
- packaging, configuration, migration, and data-safety checks;
- public roadmap refresh that stays directional and commitment-free.

Out of scope:

- full ACP runtime implementation;
- full write sync;
- new server parity beyond already-verified Phase 5 seams;
- broad rewrites of Console, Library, or ChatWindow internals;
- visual changes without actual rendered screenshot approval.

## Phase 6 Child Task Plan

### Task 1: Phase 6.1 Release Hardening Planning And Task Breakdown

**Backlog:** `TASK-13.1`

**Purpose:** Create the Phase 6 execution plan, child task tree, and QA evidence index from the current `dev` state.

**Files:**
- Create: `Docs/superpowers/plans/2026-05-16-phase-6-release-hardening.md`
- Create: `Docs/superpowers/qa/product-maturity/phase-6/README.md`
- Create: `backlog/tasks/task-13.1 - Phase-6.1-Release-hardening-planning-and-task-breakdown.md`
- Create: `backlog/tasks/task-13.2 - Phase-6.2-Full-first-time-user-release-replay.md`
- Create: `backlog/tasks/task-13.3 - Phase-6.3-Power-user-workflow-release-replay.md`
- Create: `backlog/tasks/task-13.4 - Phase-6.4-Keyboard-focus-accessibility-and-visual-sweep.md`
- Create: `backlog/tasks/task-13.5 - Phase-6.5-Recovery-setup-and-documentation-alignment.md`
- Create: `backlog/tasks/task-13.6 - Phase-6.6-Packaging-configuration-and-data-safety-validation.md`
- Create: `backlog/tasks/task-13.7 - Phase-6.7-Public-roadmap-release-closeout.md`
- Modify: `backlog/tasks/task-13 - Product-Maturity-Phase-6-Release-Hardening-And-Documentation.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Test: `Tests/UI/test_product_maturity_phase6_release_hardening_plan.py`

- [ ] **Step 1: Write failing planning regression**

Run: `python -m pytest -q Tests/UI/test_product_maturity_phase6_release_hardening_plan.py --tb=short`

Expected: FAIL because the Phase 6 plan and child tasks do not exist.

- [ ] **Step 2: Add minimal plan, QA index, and child task records**

Keep child tasks PR-sized and independent. Do not mark release hardening complete.

- [ ] **Step 3: Run focused verification**

Run: `python -m pytest -q Tests/UI/test_product_maturity_phase6_release_hardening_plan.py Tests/UI/test_post_ux_product_roadmap_handoff.py --tb=short`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add Docs/superpowers/plans/2026-05-16-phase-6-release-hardening.md Docs/superpowers/qa/product-maturity/phase-6/README.md Docs/superpowers/trackers/product-maturity-roadmap.md backlog/tasks Tests/UI/test_product_maturity_phase6_release_hardening_plan.py
git commit -m "Plan Phase 6 release hardening"
```

### Task 2: Phase 6.2 Full First-Time User Release Replay

**Backlog:** `TASK-13.2`

**Purpose:** Verify a fresh user can launch, understand the product model, identify setup blockers, and reach the correct starting workflows.

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-2-first-time-user-release-replay.md`
- Modify: `Docs/superpowers/qa/product-maturity/phase-6/README.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `backlog/tasks/task-13.2 - Phase-6.2-Full-first-time-user-release-replay.md`
- Test: focused mounted replay under `Tests/UI/`

- [ ] **Step 1: Write failing replay contract**

Assert the first-time QA evidence exists, is indexed, and records actual running-app verification.

- [ ] **Step 2: Replay clean launch and first-run orientation**

Use a clean config/data environment. Verify Home, Console, Library, Settings, and recovery/setup entry points.

- [ ] **Step 3: Capture actual rendered screenshots only if visible UI changes are made**

If any visible UI fix is required, capture the rendered screen and wait for user approval before marking the screen done.

- [ ] **Step 4: Fix P0/P1 defects only**

Keep fixes narrow. Record P2/P3 residuals with owner/follow-up.

- [ ] **Step 5: Run focused tests and commit**

Run the new replay test plus relevant first-run/navigation regressions.

### Task 3: Phase 6.3 Power-User Workflow Release Replay

**Backlog:** `TASK-13.3`

**Purpose:** Verify at least five repeated-use workflows remain fast, visible, and recoverable for experienced users.

**Required workflows:**
- grounded answer loop;
- source-to-artifact loop;
- agent run loop;
- monitoring loop;
- study loop;
- recovery loop where relevant.

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-3-power-user-workflow-release-replay.md`
- Modify: `Docs/superpowers/qa/product-maturity/phase-6/README.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `backlog/tasks/task-13.3 - Phase-6.3-Power-user-workflow-release-replay.md`
- Test: focused workflow replay under `Tests/UI/`

- [ ] **Step 1: Write failing workflow-matrix contract**

Assert evidence covers at least five named workflows with status, friction, and severity.

- [ ] **Step 2: Replay workflows in the running app**

Use existing mounted tests, Textual Web/CDP, and actual screenshots where UI changes occur.

- [ ] **Step 3: Preserve power-user speed**

Check command palette, keyboard path, state persistence, and recovery from failed starts.

- [ ] **Step 4: Fix or accept P0/P1**

No workflow can close with unaccepted P0/P1 defects.

### Task 4: Phase 6.4 Keyboard/Focus/Accessibility And Visual Sweep

**Backlog:** `TASK-13.4`

**Purpose:** Verify the release UI is navigable, readable, and visually coherent across supported terminal sizes.

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-4-keyboard-focus-accessibility-visual-sweep.md`
- Modify: affected screen/UI files only if the sweep finds defects
- Modify: `Docs/superpowers/qa/product-maturity/phase-6/README.md`
- Modify: `backlog/tasks/task-13.4 - Phase-6.4-Keyboard-focus-accessibility-and-visual-sweep.md`
- Test: focused UI/focus/visual contract tests

- [ ] **Step 1: Write failing sweep evidence contract**

Assert evidence records terminal sizes, focus surfaces, screenshots, and defect decisions.

- [ ] **Step 2: Replay supported terminal sizes**

Use actual rendered screenshots for any changed screen. Required visual changes need user approval.

- [ ] **Step 3: Replay keyboard/focus paths**

Cover top navigation, command palette, primary actions, mode bars, inspectors, and composer-style inputs.

- [ ] **Step 4: Fix visual/focus blockers**

P0/P1 defects must be fixed or explicitly accepted.

### Task 5: Phase 6.5 Recovery, Setup, And Documentation Alignment

**Backlog:** `TASK-13.5`

**Purpose:** Ensure setup, provider/model/runtime, server, optional dependency, and recovery copy are actionable and documented.

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-5-recovery-setup-docs-alignment.md`
- Modify: setup/help docs as needed
- Modify: affected recovery-copy code only if replay finds defects
- Modify: `Docs/superpowers/qa/product-maturity/phase-6/README.md`
- Modify: `backlog/tasks/task-13.5 - Phase-6.5-Recovery-setup-and-documentation-alignment.md`
- Test: focused recovery/docs alignment tests

- [ ] **Step 1: Write failing recovery evidence contract**

Assert provider, server, ACP, MCP, optional dependency, and missing-source blockers are covered.

- [ ] **Step 2: Compare UI copy with docs**

Docs must match actual labels, setup commands, and current unavailable states.

- [ ] **Step 3: Fix misleading recovery copy**

Do not add unsupported actions; prefer honest blocked states.

### Task 6: Phase 6.6 Packaging, Configuration, And Data-Safety Validation

**Backlog:** `TASK-13.6`

**Purpose:** Verify install/config/migration/data-safety paths enough for release-candidate use.

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-6-packaging-config-data-safety.md`
- Modify: packaging/config docs or tests as needed
- Modify: `Docs/superpowers/qa/product-maturity/phase-6/README.md`
- Modify: `backlog/tasks/task-13.6 - Phase-6.6-Packaging-configuration-and-data-safety-validation.md`
- Test: packaging/config/data-safety smoke tests

- [ ] **Step 1: Write failing release validation evidence contract**

Assert install/config/migration/data-safety checks are documented and linked.

- [ ] **Step 2: Verify packaging and configuration commands**

Assume virtual environment active in docs; avoid machine-specific absolute paths.

- [ ] **Step 3: Verify migration and data-safety checks**

Confirm schema/migration and write paths do not hide destructive behavior.

### Task 7: Phase 6.7 Public Roadmap Release Closeout

**Backlog:** `TASK-13.7`

**Purpose:** Close Phase 6 only after release-hardening evidence exists and public-facing docs/roadmap match current behavior.

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-7-release-closeout.md`
- Modify: `Docs/Product_Roadmap.md`
- Modify: release/onboarding docs as needed
- Modify: `Docs/superpowers/qa/product-maturity/phase-6/README.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `backlog/tasks/task-13 - Product-Maturity-Phase-6-Release-Hardening-And-Documentation.md`
- Modify: `backlog/tasks/task-13.7 - Phase-6.7-Public-roadmap-release-closeout.md`
- Test: focused closeout/roadmap/docs regressions

- [ ] **Step 1: Write failing closeout contract**

Assert all Phase 6 child tasks are done, QA evidence is indexed, public docs avoid dates/promises, and residual risks are explicit.

- [ ] **Step 2: Update public docs/roadmap**

Use directional public language. Do not expose internal task IDs or delivery commitments.

- [ ] **Step 3: Mark Phase 6 verified only after evidence exists**

Close parent `TASK-13` only when all required release-hardening gates have passed or accepted residuals.

## Release QA Evidence Requirements

Every Phase 6 QA artifact must include:

- branch and commit;
- exact command or CDP/Textual Web evidence path used;
- running-app walkthrough notes;
- screenshot references for any changed visible UI;
- workflow matrix or checklist;
- P0/P1 disposition;
- residual risks with follow-up owner/scope;
- verification commands with portable `python -m pytest ...` form.

## Verification Commands

Run during this planning slice:

```bash
python -m pytest -q Tests/UI/test_product_maturity_phase6_release_hardening_plan.py Tests/UI/test_post_ux_product_roadmap_handoff.py --tb=short
git diff --check
```

Later child tasks should add only the focused verification needed for changed seams.
