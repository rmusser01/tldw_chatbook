# Console Workbench Density Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the Console Workbench so the redesign visibly exposes useful controls and reduces passive empty space.

**Architecture:** Keep the accepted Workbench frame and route contracts. Update Console-owned widgets and CSS so domain state renders as compact visible controls, while legacy compatibility selectors remain mounted for existing tests.

**Tech Stack:** Python 3.11, Textual, Rich, pytest, TCSS build pipeline.

---

## Scope And ADR

ADR required: no.

ADR path: `backlog/decisions/011-chatbook-workbench-ui-system.md`.

Reason: this implements the already accepted Workbench UI System decision, especially visible workflow controls and command-palette-as-accelerator constraints.

## Files

- Modify: `Tests/UI/test_console_workbench_contract.py`
- Modify: `Tests/UI/test_workbench_visual_snapshots.py`
- Modify: `tldw_chatbook/Widgets/Console/console_control_bar.py`
- Modify: `tldw_chatbook/Widgets/Console/console_run_inspector.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console.md`
- Modify: `backlog/tasks/task-140 - Correct-Console-Workbench-visual-density-and-visible-controls.md`

### Task 1: Contract Tests For Visible Dense Controls

- [ ] **Step 1: Write failing tests**

Add tests requiring visible state chips for provider, model, persona, RAG, sources, tools, and approvals, plus visible action buttons in the Console control area.

- [ ] **Step 2: Verify red**

Run: `PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_control_bar_renders_visible_state_chips -q`

Expected: fail because the current control bar renders one summary line and hides detailed controls.

- [ ] **Step 3: Implement minimal widget change**

Render compact state chips and action affordances in `ConsoleControlBar`, keeping hidden compatibility selectors for legacy tests.

- [ ] **Step 4: Verify green**

Run: `PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_control_bar_renders_visible_state_chips -q`

Expected: pass.

### Task 2: Inspector And Composer Density Tests

- [ ] **Step 1: Write failing tests**

Add tests requiring inspector primary actions/status to appear before secondary grouped detail, and composer action/recovery state to remain visible.

- [ ] **Step 2: Verify red**

Run: `PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_inspector_prioritizes_actionable_status_before_secondary_groups -q`

Expected: fail because headings and rows currently dominate before action affordances.

- [ ] **Step 3: Implement minimal widget/CSS change**

Reorder `ConsoleRunInspector` around primary status, available actions, and compact state rows before lower-priority group headings. Tune composer/status CSS to avoid unused rows.

- [ ] **Step 4: Verify green**

Run: `PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_inspector_prioritizes_actionable_status_before_secondary_groups -q`

Expected: pass.

### Task 3: Visual Proof And QA Notes

- [ ] **Step 1: Extend visual snapshot gates**

Assert the exported SVG contains dense control labels and no large passive summary-only control row.

- [ ] **Step 2: Regenerate artifacts**

Create refreshed normal, compact, command-palette, and focus-state SVGs under `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/visual/`.

- [ ] **Step 3: Update QA notes**

Record before-after intent, artifact paths, verification commands, and residual risks in the QA document.

### Task 4: Verification And Closeout

- [ ] **Step 1: Run targeted tests**

Run: `PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py Tests/UI/test_workbench_visual_snapshots.py Tests/UI/test_console_workbench_parity_matrix.py Tests/UI/test_workbench_focus_help.py -q`

- [ ] **Step 2: Run CSS build**

Run: `PATH=.venv/bin:$PATH python3 tldw_chatbook/css/build_css.py`

- [ ] **Step 3: Run diff hygiene**

Run: `git diff --check`

- [ ] **Step 4: Update backlog task**

Check all acceptance criteria and add implementation notes before setting TASK-140 to Done.

- [ ] **Step 5: Commit**

Stage the task, plan, widget, CSS, test, QA, and visual artifact changes. Commit with message: `Refine Console Workbench visible density`.
