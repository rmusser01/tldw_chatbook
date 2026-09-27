---
id: TASK-32049
title: >-
  Flaky Fast Lane: test_mcp_workbench TextArea preview fails on a
  text-area--gutter COMPONENT_CLASSES KeyError
status: To Do
assignee: []
created_date: '2026-09-08 18:01'
updated_date: '2026-09-27 23:13'
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
Intermittent Fast Lane failure that has blocked PRs (#2502, #2515) and cleared on rerun/update-branch. Any test that removes or detaches a `TextArea` (or a screen/widget containing one) while a repaint is still queued can hit this: Textual's widget teardown detaches the widget and clears its component styles before a queued `render_lines` call runs, and `TextArea.render_lines`'s theme step looks up `text-area--gutter` in `COMPONENT_CLASSES`, raising `KeyError: "No 'text-area--gutter' key in COMPONENT_CLASSES"`. This is not limited to `Tests/UI/test_mcp_workbench.py::test_test_tool_preview_*` (the file this task was originally filed against) -- it is a general TextArea-detach race that can hit any test, or any real teardown path, that removes a mounted TextArea with a pending repaint. TASK-32114 patched one call site (the MCP Test Tool panel's Escape-close path) with `app.batch_update()`; that narrows one window without closing the class of bug for every other TextArea in the MCP module or elsewhere.

Root-cause fix (TASK-33115): `tldw_chatbook/Widgets/detach_safe_text_area.py` adds `DetachSafeTextArea`, a `TextArea` subclass whose `render_lines` returns blank strips when `self.is_attached` is False instead of calling into the (now-cleared) component-style lookup. Every `TextArea(` construction site under `tldw_chatbook/UI/MCP_Modules/` (`mcp_schema_form.py`, `mcp_inspector.py`, `mcp_profile_form.py`) now uses it, closing the race regardless of which teardown path removes the widget.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every TextArea construction site under tldw_chatbook/UI/MCP_Modules/ is detach-safe (TASK-33115), so no test in the Fast Lane suite that removes a mounted MCP-module TextArea can raise the text-area--gutter KeyError
- [ ] #2 The mechanism is understood and documented: it is a detach race (a queued repaint reaching render_lines after Textual clears the widget's component styles on teardown), not a mount-order/COMPONENT_CLASSES-registration race as originally filed
- [ ] #3 Tests/UI/test_mcp_workbench.py passes reliably across repeated Fast Lane runs with no intermittent text-area--gutter KeyError
<!-- AC:END -->
