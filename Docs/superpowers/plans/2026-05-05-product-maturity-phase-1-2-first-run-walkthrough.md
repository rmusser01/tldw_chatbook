# Product Maturity Phase 1.2 First-Run Walkthrough Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the clean first-run launch and setup-orientation path against the running Textual app, with durable QA evidence that proves the app is usable rather than merely rendered.

**Architecture:** Reuse the Phase 1 product-maturity QA protocol and the Phase 6 first-time replay harness. Keep this slice focused on clean launch, Home orientation, Console/Library/Settings setup entry points, and honest recovery states. Do not mark Phase 1 complete and do not start Phase 2 core-loop implementation from this task.

**Tech Stack:** Python 3.11+, pytest, Textual app test pilot, temporary HOME/XDG directories, Markdown QA evidence under `Docs/superpowers/qa/product-maturity/phase-1/`, Backlog.md.

---

## Source Inputs

- Spec: `Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md`
- Tracker: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- QA protocol: `Docs/superpowers/qa/product-maturity/phase-1/walkthrough-protocol.md`
- QA template: `Docs/superpowers/qa/product-maturity/phase-1/walkthrough-template.md`
- Prior first-time replay: `Tests/UI/test_unified_shell_phase6_first_time_replay.py`
- Prior first-time evidence: `Docs/superpowers/qa/unified-shell/phase-6/2026-05-05-phase-6-1-first-time-user-replay.md`
- Backlog task: `backlog/tasks/task-8.2 - Product-Maturity-Phase-1.2-Clean-First-Run-Launch-And-Configuration-Walkthrough.md`

## Scope Check

This plan covers one small execution slice:

- run the app from fresh `HOME` and `XDG_*` directories through the Textual test pilot where possible.
- verify first-run starts on Home even when the configured default route is Console.
- verify Home exposes setup/status orientation and routes to Console, Library, Settings, and relevant recovery/setup entry points.
- record keyboard, visual/focus, functional, and residual-risk notes using the Phase 1 template.
- add focused regression coverage for the first-run launch/evidence seam.
- close only `TASK-8.2` if verification passes or accepted blockers are documented.

This plan does not:

- complete top-level navigation smoke for every destination.
- complete the full keyboard/focus sweep.
- complete the visual broken-state audit across all terminal sizes.
- implement the Phase 2 grounded Console to Artifact/Chatbook loop.
- mark the Phase 1 parent task done.

## File Structure

- Create: `Tests/UI/test_product_maturity_phase1_first_run.py`
  - Product-maturity first-run regression and evidence contract.
- Create: `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-2-first-run-walkthrough.md`
  - Durable QA summary for the clean first-run walkthrough.
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
  - Link Phase 1.2 evidence and update status after verification.
- Modify: `Docs/superpowers/qa/product-maturity/phase-1/README.md`
  - Link Phase 1.2 evidence and keep Phase 1 overall status in progress.
- Modify: `backlog/tasks/task-8.2 - Product-Maturity-Phase-1.2-Clean-First-Run-Launch-And-Configuration-Walkthrough.md`
  - Move to In Progress before implementation, then Done only after QA evidence and tests pass.

## Task 1: Start The Phase 1.2 Backlog Task

**Files:**
- Modify: `backlog/tasks/task-8.2 - Product-Maturity-Phase-1.2-Clean-First-Run-Launch-And-Configuration-Walkthrough.md`

- [ ] **Step 1: Move task to In Progress**

Run:

```bash
backlog task edit TASK-8.2 -s "In Progress" --plain
```

Expected: `TASK-8.2` is In Progress. `TASK-8` remains To Do.

- [ ] **Step 2: Add implementation plan to the task**

Run:

```bash
backlog task edit TASK-8.2 --plan "1. Add failing product-maturity first-run regression and evidence contract tests.
2. Implement the Textual first-run pilot walkthrough with fresh HOME/XDG setup and setup-entry assertions.
3. Create Phase 1.2 QA evidence from the walkthrough result using the Phase 1 template.
4. Update tracker and Phase 1 README with Phase 1.2 evidence links while keeping Phase 1 open.
5. Run focused verification and close only TASK-8.2 if the evidence is complete." --plain
```

Expected: task has an `Implementation Plan` section and unchanged acceptance criteria.

- [ ] **Step 3: Commit task start**

Run:

```bash
git add "backlog/tasks/task-8.2 - Product-Maturity-Phase-1.2-Clean-First-Run-Launch-And-Configuration-Walkthrough.md"
git commit -m "Start product maturity Phase 1.2 first-run task"
```

Expected: commit includes only the task status/plan update.

## Task 2: Add Failing First-Run Product-Maturity Test

**Files:**
- Create: `Tests/UI/test_product_maturity_phase1_first_run.py`

- [ ] **Step 1: Write the failing evidence contract**

Create `Tests/UI/test_product_maturity_phase1_first_run.py` with constants for:

```python
EVIDENCE = Path("Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-2-first-run-walkthrough.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_1_README = Path("Docs/superpowers/qa/product-maturity/phase-1/README.md")
TASK = Path("backlog/tasks/task-8.2 - Product-Maturity-Phase-1.2-Clean-First-Run-Launch-And-Configuration-Walkthrough.md")
```

Add a test that asserts the evidence file exists and contains:

```text
## Clean-Run Setup
Fresh HOME
XDG_CONFIG_HOME
XDG_DATA_HOME
XDG_CACHE_HOME
running Textual app
Home
Console
Library
Settings
usable, not merely rendered
TASK-8.2
```

Add tracker/readme/task assertions:

```text
Phase 1.2
TASK-8.2
2026-05-05-phase-1-2-first-run-walkthrough.md
status: Done
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_first_run.py -q
```

Expected: FAIL because the evidence file and test implementation do not exist yet.

- [ ] **Step 3: Commit failing test**

Run:

```bash
git add Tests/UI/test_product_maturity_phase1_first_run.py
git commit -m "Add product maturity first-run walkthrough contract"
```

Expected: commit succeeds with the failing contract test.

## Task 3: Implement The Textual First-Run Walkthrough

**Files:**
- Modify: `Tests/UI/test_product_maturity_phase1_first_run.py`

- [ ] **Step 1: Reuse the app test builder**

Import `_build_test_app` from `Tests/UI/test_screen_navigation.py`, as the Phase 6 first-time replay does.

- [ ] **Step 2: Add fresh environment helpers**

Use `tmp_path` and `monkeypatch` to set:

```python
monkeypatch.setenv("HOME", str(tmp_path / "home"))
monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
```

Create the directories before running the app. Do not use the developer's normal app state.

- [ ] **Step 3: Disable splash and force first-run routing**

Patch `tldw_chatbook.app.get_cli_setting` so `("splash_screen", "enabled")` returns `False`.

Set:

```python
app.app_config["_first_run"] = True
app._initial_tab_value = "chat"
```

- [ ] **Step 4: Verify Home first-run orientation**

Run the app test at the common laptop size first:

```python
async with app.run_test(size=(140, 40)) as pilot:
```

Wait until:

```python
app.current_tab == "home"
app.screen.__class__.__name__ == "HomeScreen"
```

Assert the visible text includes:

```text
Dashboard, notifications, status, active work, and next actions.
Set up Console model
Start in Console
More: Ctrl+P
```

- [ ] **Step 5: Verify setup and recovery entry routes**

From first-run Home, navigate via buttons to:

```text
Console
Library
Settings
```

For each route, assert the target screen appears and exposes orientation or setup copy:

```text
Console: Live work sources, provider/model readiness or setup guidance
Library: Library, Import/Export Sources, Search/RAG
Settings: Settings or configuration/provider/runtime controls
```

If a route is honestly blocked by missing optional services, assert the blocker copy names the cause and next action.

- [ ] **Step 6: Add compact and large size probes**

Run the same high-level launch assertion at:

```text
minimum supported compact: 100x32
large power-user workspace: 180x50
```

Keep the compact/large assertions shallow enough to avoid false positives from service-loading details: Home loads, nav is visible, first-run setup action is visible, and no exception is raised.

- [ ] **Step 7: Run focused test**

Run:

```bash
../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_first_run.py -q
```

Expected: first-run pilot tests pass after evidence is added in the next task; before evidence exists, only evidence-contract assertions should fail.

## Task 4: Create Phase 1.2 QA Evidence

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-2-first-run-walkthrough.md`

- [ ] **Step 1: Copy the Phase 1 template structure**

Use `Docs/superpowers/qa/product-maturity/phase-1/walkthrough-template.md` as the section source.

- [ ] **Step 2: Fill environment and clean-run setup**

Record:

```text
Date: 2026-05-05
Branch: codex/product-maturity-phase1-first-run
Commit: current branch commit after implementation
Python version: Python 3.12.11, or actual supported runtime used
Runtime source: Textual app test pilot
Fresh HOME: temp path shape, not the absolute developer path if avoidable
XDG_CONFIG_HOME: temp path shape
XDG_DATA_HOME: temp path shape
XDG_CACHE_HOME: temp path shape
```

Do not include `/Users/...` absolute paths in the evidence. Use `<tmp>/...` placeholders when documenting ephemeral test directories.

- [ ] **Step 3: Fill walkthrough result**

Record:

```text
Entry Path: clean first-run launch with `_first_run=True` and default route set to Console
Terminal Size: 100x32, 140x40, 180x50
Steps Attempted: Home, Console, Library, Settings
Keyboard Path Result: completed, blocked with recovery, failed, or not tested
Mouse/Click Path Result: Textual pilot button activation
Functional Result: first-run orientation route completed or blocker found
Defects Found: use blocker/workflow-degradation/recoverability/polish
Exit Decision: pass, blocked, or failed
Product QA Boundary: verifies first-run orientation only, not the full Phase 1 or Phase 2 core loop
```

- [ ] **Step 4: Record automated evidence**

Add:

```text
../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_first_run.py -q
../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q
```

Expected: both pass before task closeout.

## Task 5: Update Tracker And Phase 1 Index

**Files:**
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `Docs/superpowers/qa/product-maturity/phase-1/README.md`

- [ ] **Step 1: Update tracker evidence index**

Ensure the Phase 1 section links:

```text
Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-2-first-run-walkthrough.md
```

Mark Phase 1.2 as verified only if the first-run walkthrough test and evidence pass.

- [ ] **Step 2: Preserve Phase 1 residual risk**

Keep Phase 1 overall `in_progress`. Residual risk must still mention:

```text
top-level navigation smoke
keyboard/focus sweep
visual broken-state audit
empty/error/setup-state coverage
narrow core-loop proof
```

- [ ] **Step 3: Update Phase 1 README**

Add Phase 1.2 evidence under a separate subsection:

```markdown
Phase 1.2 clean first-run status: verified

- `2026-05-05-phase-1-2-first-run-walkthrough.md`
```

## Task 6: Close TASK-8.2

**Files:**
- Modify: `backlog/tasks/task-8.2 - Product-Maturity-Phase-1.2-Clean-First-Run-Launch-And-Configuration-Walkthrough.md`

- [ ] **Step 1: Check acceptance criteria**

Run:

```bash
backlog task edit TASK-8.2 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --plain
```

- [ ] **Step 2: Add implementation notes**

Use:

```text
Verified the clean first-run launch and setup-orientation path through the Textual app test pilot with fresh HOME/XDG state. Added Phase 1.2 QA evidence covering Home orientation, Console/Library/Settings entry routes, terminal-size probes, and residual risks. Phase 1 remains open for full navigation smoke, keyboard/focus, visual, empty/error/setup, and core-loop gates.
```

- [ ] **Step 3: Mark only TASK-8.2 Done**

Run:

```bash
backlog task edit TASK-8.2 -s Done --plain
```

Expected: `TASK-8.2` is Done. `TASK-8` remains To Do or In Progress.

## Task 7: Final Verification And PR

**Files:**
- No new files.

- [ ] **Step 1: Run focused verification**

Run:

```bash
../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_first_run.py -q
../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q
../../.venv/bin/python -m pytest Tests/UI/test_unified_shell_phase6_first_time_replay.py -q
git diff --check
```

Expected: all pytest commands pass and `git diff --check` has no output.

- [ ] **Step 2: Inspect final status**

Run:

```bash
git status --short --branch
git log --oneline -5
```

Expected: only Phase 1.2 task, test, tracker, README, and QA evidence changes are ahead of `dev`.

- [ ] **Step 3: Create PR**

Use title:

```text
Verify product maturity Phase 1.2 first-run walkthrough
```

PR body:

```markdown
## Summary

- Adds Phase 1.2 clean first-run product-maturity walkthrough coverage.
- Records durable QA evidence for fresh HOME/XDG launch and setup orientation.
- Keeps Phase 1 open for navigation, focus, visual, empty/error, and core-loop gates.

## Verification

- `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_first_run.py -q`
- `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q`
- `../../.venv/bin/python -m pytest Tests/UI/test_unified_shell_phase6_first_time_replay.py -q`
- `git diff --check`
```
