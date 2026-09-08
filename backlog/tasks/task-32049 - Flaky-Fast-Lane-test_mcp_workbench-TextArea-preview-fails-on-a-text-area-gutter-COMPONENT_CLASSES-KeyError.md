---
id: TASK-32049
title: >-
  Flaky Fast Lane: test_mcp_workbench TextArea preview fails on a
  text-area--gutter COMPONENT_CLASSES KeyError
status: To Do
assignee: []
created_date: '2026-09-08 18:01'
labels:
  - mcp
  - tests
  - flaky
  - tech-debt
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Intermittent Fast Lane failure, unrelated to Library work, that has blocked PRs twice (#2502, #2515) and cleared on rerun/update-branch. `Tests/UI/test_mcp_workbench.py::test_test_tool_preview_*` raises `KeyError: "No 'text-area--gutter' key in COMPONENT_CLASSES"` under certain test orders — a Textual TextArea component-class registration race: the preview asserts `base_style`/`component_styles` before TextArea's COMPONENT_CLASSES are registered (which happens on mount). Filed as a rider; the fix belongs to whoever owns the MCP workbench tests, not this Library ▸ Media wave.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The MCP-workbench TextArea preview test registers TextArea's COMPONENT_CLASSES (mounts the widget/screen) before asserting its style, or is isolated so test order cannot leave the class unregistered
- [ ] #2 The test passes reliably across repeated Fast Lane runs (no intermittent text-area--gutter KeyError)
<!-- AC:END -->
