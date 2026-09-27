---
id: TASK-32049
title: >-
  Flaky Fast Lane: test_mcp_workbench TextArea preview fails on a
  text-area--gutter COMPONENT_CLASSES KeyError
status: To Do
assignee: []
created_date: '2026-09-08 18:01'
updated_date: '2026-09-27 23:28'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified mechanism (Textual 8.2.8): `Widget._message_loop_exit` runs `_detach()` before `_component_styles.clear()`. A screen repaint already queued when the widget detaches can still reach `TextArea.render_lines`, which calls `theme.apply_css` looking up `text-area--gutter` in the (now-cleared) `_component_styles` / `COMPONENT_CLASSES` mapping, raising `KeyError: "No 'text-area--gutter' key in COMPONENT_CLASSES"`. This corrects the mechanism this task was originally filed against (a mount-order/COMPONENT_CLASSES-registration race) -- the real cause is a post-detach repaint race, confirmed by reading Textual's own teardown order and reproduced deterministically (see below), not inferred from the traceback alone.

Fix (TASK-33115): `DetachSafeTextArea` in `tldw_chatbook/Widgets/detach_safe_text_area.py` subclasses `TextArea` and overrides only `render_lines`, returning blank `Strip`s when `self.is_attached` is False instead of falling into the theme lookup. All five `TextArea(` construction sites under `tldw_chatbook/UI/MCP_Modules/` (`mcp_schema_form.py`, `mcp_inspector.py`, `mcp_profile_form.py` x3) now use it; `query_one(..., TextArea)` and `@on(TextArea.Changed, ...)` call sites are unchanged, since the subclass still posts the same messages.

Evidence:
- `Tests/Widgets/test_detach_safe_text_area.py::test_stock_text_area_raises_after_detach` is a deterministic negative control: it mounts a stock `TextArea`, removes it, and calls `render_lines` directly -- it raises `KeyError` matching `text-area--gutter`, proving the race is real on this Textual pin (not just theorized).
- `test_detach_safe_text_area_renders_blank_after_detach` confirms `DetachSafeTextArea` renders blank strips instead of raising in the same scenario.
- `test_attached_detach_safe_text_area_renders_like_stock` confirms the guard changes nothing while the widget is attached (byte-for-byte same rendered lines as stock `TextArea`).
- Full-suite comparison with the dev `.venv`: `Tests/UI/test_mcp_workbench.py` + `Tests/UI/test_mcp_tools_mode.py` was 408 passed / 0 failed / 0 errored on both this branch and an `origin/dev` base worktree -- identical (empty) FAILED/ERROR sets, no regressions.
- The 10x10 before/after loop of `Tests/UI/test_mcp_workbench.py` on the fast-lane minimal venv was inconclusive: 0 `text-area--gutter` failures in 10 runs on both `origin/dev` (before) and the fix branch (after) -- it did not reproduce the timing-dependent flake in this run, so the deterministic negative-control test above is the evidence that carries this task, not the loop.

See PR #2866 and TASK-33115 for the full change and additional detail. Status left as To Do for the controller to set Done after merge.
<!-- SECTION:NOTES:END -->
