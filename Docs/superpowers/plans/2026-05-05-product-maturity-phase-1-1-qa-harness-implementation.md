# Product Maturity Phase 1.1 QA Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the first executable product-maturity planning slice: Backlog anchors plus the canonical Phase 1.1 QA harness for clean-run usability evidence.

**Architecture:** Reuse the verified Unified Shell maturity model instead of creating a parallel process. Product-maturity tracking gets its own tracker and QA evidence directory, while Backlog.md owns PR-sized execution tasks. The first implementation slice is harness-only: it creates repeatable protocol, template, and smoke evidence, but does not complete the full first-run, focus, visual, empty-state, or core-loop audit.

**Tech Stack:** Python 3.11+, pytest, Textual test pilot conventions, Backlog.md, Markdown documentation under `Docs/superpowers/`.

---

## Source Inputs

- Spec: `Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md`
- Existing QA protocol: `Docs/superpowers/qa/unified-shell/phase-1/walkthrough-protocol.md`
- Existing QA template: `Docs/superpowers/qa/unified-shell/phase-1/walkthrough-template.md`
- Existing tracker: `Docs/superpowers/trackers/unified-shell-maturity-roadmap.md`
- Backlog pointer doc: `backlog/docs/unified-shell-maturity-roadmap.md`

## Scope Check

This plan covers one small execution slice:

- create Backlog parent anchors for the six product-maturity phases.
- create one child task for Phase 1.1.
- create the product-maturity tracker and Backlog pointer doc.
- create the Phase 1.1 QA protocol, evidence template, README, and harness smoke summary.
- add a focused regression test that prevents the harness from disappearing or drifting.
- close only the Phase 1.1 Backlog child task if verification passes.

This plan does not:

- perform the full first-run audit.
- perform the full keyboard/focus sweep.
- perform the full visual broken-state audit.
- implement product module depth.
- create child tasks for later phases beyond Phase 1.1.

## File Structure

- Create: `Docs/superpowers/trackers/product-maturity-roadmap.md`
  - Product-maturity source of truth. Tracks phase goals, Backlog task IDs, QA evidence paths, and residual risks.
- Create: `Docs/superpowers/qa/product-maturity/README.md`
  - QA evidence index and rules for the new workstream.
- Create: `Docs/superpowers/qa/product-maturity/phase-1/README.md`
  - Phase 1 QA index.
- Create: `Docs/superpowers/qa/product-maturity/phase-1/walkthrough-protocol.md`
  - Canonical clean-run protocol, severity mapping, terminal-size matrix, and entry commands.
- Create: `Docs/superpowers/qa/product-maturity/phase-1/walkthrough-template.md`
  - Reusable evidence template.
- Create: `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-1-harness-smoke.md`
  - Harness-only smoke evidence. Must explicitly say it does not verify product workflows.
- Create: `backlog/docs/product-maturity-roadmap.md`
  - Backlog pointer to the canonical spec, tracker, and QA evidence.
- Create: `Tests/UI/test_product_maturity_phase1_harness.py`
  - Focused contract tests for docs, Backlog anchors, severity taxonomy, and QA boundary.
- Modify: new Backlog task files under `backlog/tasks/`
  - Create parent tasks for product-maturity phases.
  - Create one child task for Phase 1.1.
  - Do not hard-code task IDs before creation. Record the IDs returned by Backlog.md and use those IDs consistently in docs.

## Task 1: Create Backlog Product-Maturity Anchors

**Files:**
- Create: `backlog/tasks/task-<returned-id> - Product-Maturity-Phase-1-QA-Baseline-And-Usability-Guardrails.md`
- Create: `backlog/tasks/task-<returned-id> - Product-Maturity-Phase-2-Core-Agentic-Loop.md`
- Create: `backlog/tasks/task-<returned-id> - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`
- Create: `backlog/tasks/task-<returned-id> - Product-Maturity-Phase-4-Agent-Configuration-And-Execution.md`
- Create: `backlog/tasks/task-<returned-id> - Product-Maturity-Phase-5-Server-Parity-And-Live-Integrations.md`
- Create: `backlog/tasks/task-<returned-id> - Product-Maturity-Phase-6-Release-Hardening-And-Documentation.md`
- Create: `backlog/tasks/task-<returned-id> - Product-Maturity-Phase-1.1-Canonical-QA-Harness.md`

- [ ] **Step 1: Inspect current Backlog IDs**

Run:

```bash
backlog task list --plain
```

Expected: existing Unified Shell `TASK-1` through `TASK-7.3` are present and done. If the CLI is unavailable, use the Backlog MCP `task_list` tool.

- [ ] **Step 2: Create six parent tasks**

Use Backlog MCP `task_create` or the CLI. Set all parent tasks to `To Do`, priority `medium`, label `product-maturity`, and add the phase label from the spec.

Parent task titles:

```text
Product Maturity Phase 1: QA Baseline And Usability Guardrails
Product Maturity Phase 2: Core Agentic Loop
Product Maturity Phase 3: Knowledge And Study Workflows
Product Maturity Phase 4: Agent Configuration And Execution
Product Maturity Phase 5: Server-Parity And Live Integrations
Product Maturity Phase 6: Release Hardening And Documentation
```

Each parent task description should be one sentence copied from the phase purpose in the spec. Each parent task acceptance criteria should include:

```text
QA walkthrough verifies the running app is usable for this phase's target workflows.
Focused regression evidence exists for changed seams.
Repo-tracked QA evidence exists under Docs/superpowers/qa/product-maturity/.
P0/P1 findings are fixed or explicitly accepted according to the spec severity policy.
```

- [ ] **Step 3: Create the Phase 1.1 child task**

Create it under the Phase 1 parent task.

Title:

```text
Product Maturity Phase 1.1: Canonical QA Harness
```

Description:

```text
Create the reusable product-maturity QA protocol, template, evidence index, severity mapping, and smoke evidence so later usability work can be verified against the running app rather than render-only checks.
```

Acceptance criteria:

```text
Product-maturity QA protocol defines clean-run setup, entry commands, terminal-size matrix, severity mapping, and evidence rules.
Product-maturity QA template captures environment, entry path, steps, visual/focus notes, functional result, defects, evidence, residual risk, and exit decision.
Product-maturity tracker links the spec, Backlog tasks, Phase 1.1 evidence, and residual risks.
Focused pytest coverage verifies the protocol, template, tracker, and Backlog anchors exist and preserve the harness-only boundary.
Harness smoke evidence states that Phase 1.1 verifies the QA harness only and does not complete the full product walkthrough.
```

- [ ] **Step 4: Record returned task IDs**

Run:

```bash
backlog task list --plain
```

Expected: six new product-maturity parent tasks and one Phase 1.1 child task exist. Record actual IDs for later docs. If current Backlog state is unchanged, likely IDs are `TASK-8` through `TASK-13` and `TASK-8.1`, but do not assume.

- [ ] **Step 5: Commit Backlog anchors**

Run:

```bash
git add backlog/tasks
git commit -m "Add product maturity backlog anchors"
```

Expected: commit succeeds with only new Backlog task files.

## Task 2: Add Failing Harness Contract Test

**Files:**
- Create: `Tests/UI/test_product_maturity_phase1_harness.py`

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_product_maturity_phase1_harness.py`:

```python
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = Path("Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
BACKLOG_DOC = Path("backlog/docs/product-maturity-roadmap.md")
QA_ROOT = Path("Docs/superpowers/qa/product-maturity")
PHASE_1_ROOT = QA_ROOT / "phase-1"
PHASE_1_README = PHASE_1_ROOT / "README.md"
PROTOCOL = PHASE_1_ROOT / "walkthrough-protocol.md"
TEMPLATE = PHASE_1_ROOT / "walkthrough-template.md"
SMOKE = PHASE_1_ROOT / "2026-05-05-phase-1-1-harness-smoke.md"
BACKLOG_TASKS = Path("backlog/tasks")

REQUIRED_TEMPLATE_SECTIONS = {
    "Environment",
    "Task Or Phase",
    "Entry Path",
    "Terminal Size",
    "Clean-Run Setup",
    "Steps Attempted",
    "Visual/Focus Notes",
    "Keyboard Path Result",
    "Mouse/Click Path Result",
    "Functional Result",
    "Defects Found",
    "Evidence",
    "Residual Risk",
    "Exit Decision",
    "Product QA Boundary",
}

REQUIRED_SEVERITIES = {
    "blocker",
    "workflow-degradation",
    "recoverability",
    "polish",
}

REQUIRED_PRIORITY_LABELS = {"P0", "P1", "P2", "P3"}


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _task_text_containing(title_fragment: str) -> str:
    matches: list[str] = []
    for path in (REPO_ROOT / BACKLOG_TASKS).glob("*.md"):
        text = path.read_text(encoding="utf-8")
        if title_fragment in text:
            matches.append(text)
    assert len(matches) == 1, f"expected one task containing {title_fragment!r}, found {len(matches)}"
    return matches[0]


def _phase_row(markdown: str, phase_title: str) -> list[str]:
    for line in markdown.splitlines():
        if not line.startswith("|"):
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if columns and columns[0] == phase_title:
            return columns
    raise AssertionError(f"{phase_title!r} row not found")


def test_product_maturity_phase_one_harness_files_exist() -> None:
    for path in (SPEC, TRACKER, BACKLOG_DOC, QA_ROOT / "README.md", PHASE_1_README, PROTOCOL, TEMPLATE, SMOKE):
        assert (REPO_ROOT / path).exists(), f"{path} should exist"


def test_product_maturity_template_captures_required_fields_and_severity() -> None:
    text = _text(TEMPLATE)

    for section in REQUIRED_TEMPLATE_SECTIONS:
        assert f"## {section}" in text
    for severity in REQUIRED_SEVERITIES:
        assert severity in text
    for priority in REQUIRED_PRIORITY_LABELS:
        assert priority in text
    assert "usable, not merely rendered" in text


def test_product_maturity_protocol_defines_clean_run_and_terminal_matrix() -> None:
    text = _text(PROTOCOL)

    assert "python3 -m tldw_chatbook.app" in text
    assert "Fresh HOME" in text
    assert "XDG_CONFIG_HOME" in text
    assert "XDG_DATA_HOME" in text
    assert "minimum supported compact" in text
    assert "common laptop terminal" in text
    assert "large power-user workspace" in text
    assert "render-only" in text
    assert "click-event-only" in text
    assert "Tests/UI/test_product_maturity_phase1_harness.py" in text


def test_product_maturity_tracker_links_phase_one_harness_and_tasks() -> None:
    tracker = _text(TRACKER)
    phase_one_task = _task_text_containing("Product Maturity Phase 1: QA Baseline And Usability Guardrails")
    phase_one_one_task = _task_text_containing("Product Maturity Phase 1.1: Canonical QA Harness")

    assert str(SPEC) in tracker
    assert str(PROTOCOL) in tracker
    assert str(TEMPLATE) in tracker
    assert str(SMOKE) in tracker
    assert "Phase 1.1" in tracker
    assert "<PHASE_" not in tracker

    phase_one_row = _phase_row(tracker, "Phase 1: QA Baseline And Usability Guardrails")
    assert phase_one_row[2] in {"planned", "in_progress"}
    assert "Phase 1.1" in phase_one_row[3]
    assert "phase-1/" in phase_one_row[4]

    assert "QA walkthrough verifies the running app is usable" in phase_one_task
    assert "Product-maturity QA protocol defines clean-run setup" in phase_one_one_task
    assert "Harness smoke evidence states" in phase_one_one_task


def test_phase_one_one_smoke_evidence_records_harness_only_boundary() -> None:
    text = _text(SMOKE)

    assert "Phase 1.1" in text
    assert "Canonical QA Harness" in text
    assert "Product QA Boundary" in text
    assert re.search(r"harness[- ]only", text, re.IGNORECASE)
    assert "does not complete the full product walkthrough" in text
    assert "Tests/UI/test_product_maturity_phase1_harness.py" in text
    assert "<PHASE_" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q
```

Expected: FAIL because `Docs/superpowers/trackers/product-maturity-roadmap.md`, product-maturity QA docs, and Backlog pointer doc do not exist yet.

- [ ] **Step 3: Commit failing test**

Run:

```bash
git add Tests/UI/test_product_maturity_phase1_harness.py
git commit -m "Add product maturity QA harness contract test"
```

Expected: commit succeeds with one new test file.

## Task 3: Create Product-Maturity Tracker And QA Harness Docs

**Files:**
- Create: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Create: `Docs/superpowers/qa/product-maturity/README.md`
- Create: `Docs/superpowers/qa/product-maturity/phase-1/README.md`
- Create: `Docs/superpowers/qa/product-maturity/phase-1/walkthrough-protocol.md`
- Create: `Docs/superpowers/qa/product-maturity/phase-1/walkthrough-template.md`
- Create: `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-1-harness-smoke.md`
- Create: `backlog/docs/product-maturity-roadmap.md`

- [ ] **Step 1: Create QA directories**

Run:

```bash
mkdir -p Docs/superpowers/qa/product-maturity/phase-1
```

Expected: directory exists.

- [ ] **Step 2: Create Backlog pointer doc**

Create `backlog/docs/product-maturity-roadmap.md` with:

```markdown
# Product Maturity Roadmap

The canonical product-maturity design spec lives at:

`Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md`

The canonical product-maturity tracker lives at:

`Docs/superpowers/trackers/product-maturity-roadmap.md`

Durable QA evidence lives at:

`Docs/superpowers/qa/product-maturity/`

Backlog tasks in `backlog/tasks/` are PR-sized execution units. Parent tasks represent product-maturity phases; child tasks represent implementation or QA gates.

Do not mark a product-maturity task or phase complete because UI renders or a button is clickable. Completion requires automated evidence, running-app QA evidence where product behavior is in scope, and a repo-tracked QA summary.
```

- [ ] **Step 3: Create product-maturity tracker**

Create `Docs/superpowers/trackers/product-maturity-roadmap.md`. Use the actual Backlog task IDs returned in Task 1.

Minimum required content is below. The `<PHASE_...>` markers are implementation variables for this plan only; the committed tracker must contain actual Backlog task IDs and must not contain these markers.

```markdown
# Product Maturity Roadmap

Date: 2026-05-05
Status: Phase 1.1 planned
Source Branch: `dev`
Source Spec: `Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md`

## Purpose

Track product-depth maturity after Unified Shell Phase 6 so rendered screens, clickable controls, and complete usable workflows stay distinct.

## Current Verified Baseline

- Unified Shell Phase 0-6 are verified in `Docs/superpowers/trackers/unified-shell-maturity-roadmap.md`.
- Product-maturity work starts with a QA baseline before new feature depth.
- Phase 1.1 creates the reusable harness only; it does not complete the full first-run, focus, visual, empty-state, or core-loop audits.

## Severity Policy

| Priority | Taxonomy | Exit Rule |
| --- | --- | --- |
| P0 | `blocker` | Must be fixed before the phase or gate can close. |
| P1 | `workflow-degradation` | Must be fixed before phase close unless explicitly accepted with owner, rationale, and follow-up task. |
| P2 | `recoverability` | May remain only with residual risk and a scoped follow-up. |
| P3 | `polish` | May remain as backlog polish if it does not hide status, source authority, or recovery action. |

## Backlog Task Hierarchy

- Phase 1: QA Baseline And Usability Guardrails - `<PHASE_1_TASK_ID>`
- Phase 1.1: Canonical QA Harness - `<PHASE_1_1_TASK_ID>`
- Phase 2: Core Agentic Loop - `<PHASE_2_TASK_ID>`
- Phase 3: Knowledge And Study Workflows - `<PHASE_3_TASK_ID>`
- Phase 4: Agent Configuration And Execution - `<PHASE_4_TASK_ID>`
- Phase 5: Server-Parity And Live Integrations - `<PHASE_5_TASK_ID>`
- Phase 6: Release Hardening And Documentation - `<PHASE_6_TASK_ID>`

## QA Evidence Index

| Phase | Evidence Path | Status |
| --- | --- | --- |
| Phase 1 | `Docs/superpowers/qa/product-maturity/phase-1/` | planned |

## Phase Overview

| Phase | Goal | Status | Backlog Tasks | QA Evidence | Residual Risk |
| --- | --- | --- | --- | --- | --- |
| Phase 1: QA Baseline And Usability Guardrails | Establish clean-run usability guardrails before feature depth. | planned | `<PHASE_1_TASK_ID>`, `<PHASE_1_1_TASK_ID>` | `phase-1/` | Phase 1.1 is harness-only; product walkthroughs remain future Phase 1 gates. |
| Phase 2: Core Agentic Loop | Complete source/question to grounded Console to Artifact/Chatbook loop. | planned | `<PHASE_2_TASK_ID>` | not-started | Depends on Phase 1 QA baseline. |
| Phase 3: Knowledge And Study Workflows | Mature ingest, organize, retrieve, study, and reuse workflows. | planned | `<PHASE_3_TASK_ID>` | not-started | Depends on Phase 2 core loop and later task slicing. |
| Phase 4: Agent Configuration And Execution | Mature Personas, Skills, MCP, ACP, Schedules, and Workflows. | planned | `<PHASE_4_TASK_ID>` | not-started | Depends on service adapters and runtime readiness. |
| Phase 5: Server-Parity And Live Integrations | Close high-value `tldw_server2` parity gaps. | planned | `<PHASE_5_TASK_ID>` | not-started | Requires parity inventory. |
| Phase 6: Release Hardening And Documentation | Reach release-candidate usability. | planned | `<PHASE_6_TASK_ID>` | not-started | Depends on earlier phase evidence. |

## Phase 1.1 Evidence

- `Docs/superpowers/qa/product-maturity/phase-1/walkthrough-protocol.md`
- `Docs/superpowers/qa/product-maturity/phase-1/walkthrough-template.md`
- `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-1-harness-smoke.md`
```

Replace all `<...>` markers before running tests.

- [ ] **Step 4: Create product-maturity QA README**

Create `Docs/superpowers/qa/product-maturity/README.md`:

```markdown
# Product Maturity QA Evidence

This directory stores durable QA evidence for the product-maturity roadmap.

Canonical tracker:

`Docs/superpowers/trackers/product-maturity-roadmap.md`

Canonical spec:

`Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md`

Rules:

- Verify running-app behavior, not only rendered widgets.
- Record whether each workflow completed, was honestly blocked with recovery, or failed.
- Use one defect taxonomy label: `blocker`, `workflow-degradation`, `recoverability`, or `polish`.
- Record P0/P1/P2/P3 only when release or phase-exit decisions need that mapping.
- Store one markdown QA summary per gate.
```

- [ ] **Step 5: Create Phase 1 README**

Create `Docs/superpowers/qa/product-maturity/phase-1/README.md`:

```markdown
# Product Maturity Phase 1 QA

Status: planned

Phase 1 establishes QA baselines and usability guardrails before additional product-depth work.

Phase 1.1 evidence:

- `walkthrough-protocol.md`
- `walkthrough-template.md`
- `2026-05-05-phase-1-1-harness-smoke.md`

Phase 1.1 is harness-only. It does not complete the full first-run, focus, visual, empty-state, or core-loop audits.
```

- [ ] **Step 6: Create walkthrough protocol**

Create `Docs/superpowers/qa/product-maturity/phase-1/walkthrough-protocol.md`.

Required content:

````markdown
# Product Maturity Phase 1 Walkthrough Protocol

Status: active for Phase 1.1 harness setup

Use this protocol before marking product-maturity workflows usable. Run it against the actual Textual app when product behavior is in scope. Render-only checks and click-event-only checks do not prove workflow completion.

## Clean-Run Setup

Use a fresh `HOME` and `XDG_*` directory set when validating first-run or setup behavior:

- Fresh HOME
- XDG_CONFIG_HOME
- XDG_DATA_HOME
- XDG_CACHE_HOME

Record the exact directories in the QA summary. Do not use a developer's normal app state for first-run claims.

## Entry Commands

Manual app entry:

```bash
python3 -m tldw_chatbook.app
```

Focused harness contract:

```bash
python3 -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q
```

Existing shell support checks:

```bash
python3 -m pytest Tests/UI/test_master_shell_navigation.py -q
python3 -m pytest Tests/UI/test_shell_destinations.py -q
python3 -m pytest Tests/UI/test_unified_shell_qa_protocol.py -q
```

## Terminal-Size Matrix

Record the terminal size used for every visual/focus walkthrough:

- minimum supported compact
- common laptop terminal
- large power-user workspace

The implementation task that performs a visual sweep must replace these labels with exact dimensions after confirming supported sizes.

## Required Walkthrough Scope

For each product workflow or harness gate, record:

- Navigation entry path.
- Whether status and source authority are visible.
- Whether primary actions are reachable by keyboard and mouse.
- Whether disabled, blocked, unavailable, pending approval, and recovery states explain owner, reason, and next action.
- Whether the workflow completes, is honestly blocked with recovery, or fails.
- Whether repeated use remains fast enough for power users.

## Severity Labels

Use exactly one taxonomy label when a defect is found:

- `blocker` - P0; prevents basic use, traps the user, corrupts or loses user work, or makes a required workflow impossible.
- `workflow-degradation` - P1; breaks or seriously slows a core workflow but leaves a workaround.
- `recoverability` - P2; blocked/error state exists but recovery copy, ownership, or next action is unclear.
- `polish` - P3; visual or wording issue that does not block completion.

## Evidence Rules

- Do not count render-only checks as workflow completion.
- Do not count click-event-only checks as workflow completion.
- If a workflow is blocked, record the recovery path and severity instead of treating the blocked state as a pass.
- Record screenshots only when they materially clarify layout, focus, or visual defects.

## Output

Create one markdown summary per gate using `walkthrough-template.md`. Store summaries in this directory and link them from `Docs/superpowers/trackers/product-maturity-roadmap.md` and the relevant Backlog task.
````

- [ ] **Step 7: Create walkthrough template**

Create `Docs/superpowers/qa/product-maturity/phase-1/walkthrough-template.md`:

```markdown
# Product Maturity Phase 1 Walkthrough Summary

## Environment

- Date:
- Branch:
- Commit:
- Python version:
- Runtime source:
- Config/home directory:

## Task Or Phase

- Backlog task:
- Phase:
- Destination or workflow:

## Entry Path

- Launch command, direct route, command palette path, focused mounted test, or manual terminal replay:

## Terminal Size

- Category:
- Dimensions:

## Clean-Run Setup

- Fresh HOME:
- XDG_CONFIG_HOME:
- XDG_DATA_HOME:
- XDG_CACHE_HOME:
- Reused state:

## Steps Attempted

1.

## Visual/Focus Notes

- Layout:
- Clipping:
- Focus indication:
- Labels:
- Disabled or blocked states:
- Information hierarchy:

## Keyboard Path Result

- Completed, blocked with recovery, failed, or not tested:

## Mouse/Click Path Result

- Completed, blocked with recovery, failed, or not tested:

## Functional Result

- Completed workflow:
- Honest blocked state:
- Failed workflow:
- Recovery path:

## Defects Found

Use one taxonomy label and optional P-level:

- `blocker` / P0:
- `workflow-degradation` / P1:
- `recoverability` / P2:
- `polish` / P3:

## Evidence

- Screenshots:
- Logs:
- Probe JSON:
- Test commands:
- Related PRs or commits:

## Residual Risk

- Untested live server/API paths:
- Optional dependency limits:
- Environment limits:
- Follow-up tasks:

## Exit Decision

- Pass, blocked, failed, or harness-only:

## Product QA Boundary

State whether this walkthrough verifies a real product workflow, only validates the protocol/harness, or intentionally leaves product workflow verification to a later task.

This summary must prove the app is usable, not merely rendered, when product behavior is in scope.
```

- [ ] **Step 8: Create Phase 1.1 smoke evidence**

Create `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-1-harness-smoke.md`:

```markdown
# Phase 1.1 Canonical QA Harness Smoke

## Environment

- Date: 2026-05-05
- Branch:
- Commit:
- Python version: not executed in this evidence slice
- Runtime source: documentation/test harness
- Config/home directory: not applicable

## Task Or Phase

- Backlog task: `<PHASE_1_1_TASK_ID>`
- Phase: Phase 1.1
- Destination or workflow: Canonical QA Harness

## Entry Path

- Focused contract: `python3 -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q`

## Steps Attempted

1. Created product-maturity tracker.
2. Created product-maturity QA protocol.
3. Created product-maturity QA template.
4. Linked Backlog task anchors and QA evidence.
5. Ran focused harness contract test.

## Visual/Focus Notes

- Layout: not tested in this harness-only slice.
- Clipping: not tested in this harness-only slice.
- Focus indication: not tested in this harness-only slice.
- Labels: protocol and template labels verified by focused tests.
- Disabled or blocked states: not tested in this harness-only slice.
- Information hierarchy: tracker and evidence index verified by focused tests.

## Functional Result

- Completed workflow: QA harness exists and is linked.
- Honest blocked state: product workflow verification remains future Phase 1 work.
- Failed workflow: none for harness scope.
- Recovery path: future product workflow defects must use the severity policy in the protocol.

## Defects Found

- None in harness scope.

## Evidence

- Test commands: `python3 -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q`
- Related PRs or commits:

## Residual Risk

- Untested live server/API paths: all product live paths remain outside Phase 1.1.
- Optional dependency limits: not exercised.
- Environment limits: visual/focus behavior requires later manual terminal replay.
- Follow-up tasks: full first-run, navigation, visual, empty-state, and core-loop product audits.

## Exit Decision

- harness-only

## Product QA Boundary

Phase 1.1 is harness-only. It does not complete the full product walkthrough. It verifies that later product-maturity work has a reusable protocol, template, severity mapping, tracker linkage, and focused regression test.
```

Replace `<PHASE_1_1_TASK_ID>` before running tests. The committed smoke evidence must contain the actual Backlog task ID and must not contain this marker.

- [ ] **Step 9: Run focused test to verify it passes**

Run:

```bash
python3 -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit harness docs**

Run:

```bash
git add Docs/superpowers/trackers/product-maturity-roadmap.md Docs/superpowers/qa/product-maturity backlog/docs/product-maturity-roadmap.md
git commit -m "Add product maturity QA harness docs"
```

Expected: commit succeeds with new product-maturity docs.

## Task 4: Close Phase 1.1 Backlog Task

**Files:**
- Modify: `backlog/tasks/task-<phase-1-1-id> - Product-Maturity-Phase-1.1-Canonical-QA-Harness.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `Docs/superpowers/qa/product-maturity/phase-1/README.md`
- Modify: `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-1-harness-smoke.md`

- [ ] **Step 1: Re-run focused verification before closing task**

Run:

```bash
python3 -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q
git diff --check
```

Expected: pytest PASS; diff check has no output.

- [ ] **Step 2: Update Phase 1.1 smoke evidence with final commit/test output**

Edit `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-1-harness-smoke.md`:

- fill branch and commit.
- fill Python version if command was run locally.
- add focused test result.
- add related commit IDs.

- [ ] **Step 3: Update tracker status for Phase 1.1**

Edit `Docs/superpowers/trackers/product-maturity-roadmap.md`:

- mark Phase 1.1 as `verified`.
- keep Phase 1 as `planned` or `in_progress`; do not mark Phase 1 complete.
- keep Phase 1 residual risk clear: full first-run, focus, visual, empty-state, and core-loop audits remain future gates.

- [ ] **Step 4: Update Phase 1 README**

Edit `Docs/superpowers/qa/product-maturity/phase-1/README.md`:

- set Phase 1.1 harness status to `verified`.
- keep Phase 1 overall status as `planned` or `in_progress`.
- link the smoke evidence.

- [ ] **Step 5: Mark Phase 1.1 Backlog acceptance criteria complete**

Use Backlog MCP `task_edit` or the CLI:

- check every Phase 1.1 acceptance criterion.
- add implementation notes summarizing tracker, protocol, template, smoke evidence, and test coverage.
- set only the Phase 1.1 child task to `Done`.
- do not mark the Phase 1 parent done.

- [ ] **Step 6: Run final verification**

Run:

```bash
python3 -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q
git diff --check
```

Expected: pytest PASS; diff check has no output.

- [ ] **Step 7: Commit closeout**

Run:

```bash
git add backlog/tasks Docs/superpowers/trackers/product-maturity-roadmap.md Docs/superpowers/qa/product-maturity/phase-1/README.md Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-1-harness-smoke.md
git commit -m "Verify product maturity Phase 1.1 QA harness"
```

Expected: commit succeeds.

## Task 5: Final Branch Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused product-maturity verification**

Run:

```bash
python3 -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q
```

Expected: PASS.

- [ ] **Step 2: Run adjacent QA protocol regression**

Run:

```bash
python3 -m pytest Tests/UI/test_unified_shell_qa_protocol.py -q
```

Expected: PASS.

- [ ] **Step 3: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Inspect final status**

Run:

```bash
git status --short --branch
git log --oneline -5
```

Expected: only intentional product-maturity commits are ahead of `dev`; unrelated untracked files remain untouched.

## Implementation Notes Template

Use this for the Phase 1.1 Backlog task:

```markdown
## Implementation Notes

- Added product-maturity Backlog anchors for Phase 1 through Phase 6 and one Phase 1.1 child task.
- Added `Docs/superpowers/trackers/product-maturity-roadmap.md` as the product-maturity tracker.
- Added the product-maturity QA evidence root and Phase 1.1 protocol/template/smoke evidence.
- Added `Tests/UI/test_product_maturity_phase1_harness.py` to lock the harness, tracker, severity mapping, and harness-only QA boundary.
- Phase 1.1 verifies harness readiness only. Full first-run, visual/focus, empty-state, and core-loop walkthroughs remain future Phase 1 gates.
```

## PR Description Template

```markdown
## Summary

- Adds product-maturity Backlog anchors and Phase 1.1 canonical QA harness.
- Adds tracker and QA evidence docs for the product-maturity roadmap.
- Adds focused pytest coverage to prevent harness/tracker drift.

## Verification

- `python3 -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q`
- `python3 -m pytest Tests/UI/test_unified_shell_qa_protocol.py -q`
- `git diff --check`

## Residual Risk

- Phase 1.1 is harness-only.
- Full first-run, keyboard/focus, visual, empty/error state, and core-loop product audits remain future Phase 1 tasks.
```
