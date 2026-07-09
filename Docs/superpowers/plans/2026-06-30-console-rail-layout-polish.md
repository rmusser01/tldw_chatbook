# Console Rail Layout Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish the Console setup-blocked layout so the left rail uses one scroll owner, focus visuals are not a loud blue boundary box, and empty-state surfaces use their available vertical space.

**Architecture:** Keep the existing Console workbench shell, selectors, and rail framing. Convert the nested conversation browser list from a scroll container to a normal vertical region, route overflow through `#console-left-rail-body`, and make focus styling explicit in TCSS. Rebuild `tldw_cli_modular.tcss` after editing source TCSS.

**Tech Stack:** Python 3.11, Textual, TCSS, pytest async Pilot tests, Backlog.md.

---

### Task 1: Add Failing Rail Layout Tests

**Files:**
- Modify: `Tests/UI/test_console_workspace_context_rail.py`
- Modify: `Tests/UI/test_console_persistent_rails.py`

- [x] **Step 1: Add a behavior test for single-scroll ownership**

Add assertions that `#console-workspace-conversations` has no inner scroll range and that `#console-left-rail-body` owns overflow when many conversation rows exist.

- [x] **Step 2: Add CSS contract assertions**

Assert the source and generated TCSS do not give the conversation browser an inner `overflow-y: auto` scrollbar, and assert explicit quiet focus styles exist for the rail/workspace containers.

- [x] **Step 3: Run the new tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_many_conversations_keep_lower_status_reachable Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_context_grouped_browser_styles_are_declared Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_conversation_subsection_styles_are_declared Tests/UI/test_console_persistent_rails.py::test_generated_console_stylesheet_includes_rail_rules -q
```

Expected: fail because the inner conversation list still owns scroll/focus styling and the generated CSS has not changed.

### Task 2: Implement Layout and Focus Polish

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss` via build script

- [x] **Step 1: Make the conversation list a normal vertical region**

Replace the nested `VerticalScroll(id="console-workspace-conversations")` instances with `Vertical(id="console-workspace-conversations")`.

- [x] **Step 2: Keep row sizing stable**

Preserve two-line row heights and margins. Remove Python-assigned fixed scroll heights from the list.

- [x] **Step 3: Add explicit quiet focus styles**

Add TCSS for `#console-left-rail`, `#console-left-rail-body`, `#console-workspace-context`, and `#console-workspace-conversations` so focus/focus-within does not draw a high-saturation panel border.

- [x] **Step 4: Fill the transcript empty surface**

Update `#console-transcript-empty-state` / `.console-transcript-empty-panel` to use the transcript region height rather than a short auto-height panel.

- [x] **Step 5: Rebuild modular CSS**

Run:

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
```

Expected: `tldw_chatbook/css/tldw_cli_modular.tcss` includes the same source CSS changes.

### Task 3: Verify and Capture Evidence

**Files:**
- Modify: `backlog/tasks/task-149 - Polish-Console-rail-layout-and-sticky-header.md`

- [x] **Step 1: Run focused tests**

Run the red test slice again and expect pass.

- [x] **Step 2: Run broader affected UI tests**

Run:

```bash
.venv/bin/python -m pytest Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_persistent_rails.py Tests/UI/test_console_workbench_contract.py::test_console_f6_cycles_visible_workbench_panes -q
```

- [x] **Step 3: Capture rendered evidence**

Use Textual `export_screenshot()` in a focused harness to produce an SVG artifact under `/tmp`.

- [x] **Step 4: Update Backlog task notes**

Mark AC complete only after tests and screenshot evidence exist.
