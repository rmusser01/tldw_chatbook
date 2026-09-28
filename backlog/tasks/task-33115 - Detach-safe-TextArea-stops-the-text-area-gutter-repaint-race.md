---
id: TASK-33115
title: Detach-safe TextArea stops the text-area--gutter repaint race
status: Done
assignee: []
created_date: '2026-09-27 23:11'
updated_date: '2026-09-28 02:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every MCP-module TextArea construction site is swapped for DetachSafeTextArea, a TextArea subclass that renders blank strips instead of raising once the widget is detached. This closes TASK-32049 at its root: Textual clears a widget's component styles on detach, and a screen repaint already queued can still reach render_lines, whose theme step looks up text-area--gutter and raises KeyError.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DetachSafeTextArea renders identically to stock TextArea while attached
- [x] #2 DetachSafeTextArea renders blank strips instead of raising once detached
- [x] #3 A deterministic negative-control test proves the KeyError is real on stock TextArea against the pinned Textual version
- [x] #4 Every TextArea construction site under tldw_chatbook/UI/MCP_Modules/ uses DetachSafeTextArea; query_one/@on usages of TextArea are unchanged
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root-cause fix for TASK-32049: `tldw_chatbook/Widgets/detach_safe_text_area.py` adds `DetachSafeTextArea(TextArea)`, which overrides only `render_lines` to return blank `Strip`s when `self.is_attached` is False instead of calling into Textual's theme lookup. Every construction site under `tldw_chatbook/UI/MCP_Modules/` (`mcp_schema_form.py:244`, `mcp_inspector.py:1624`, `mcp_profile_form.py:107/132/432`) now uses it; `query_one(..., TextArea)` and `@on(TextArea.Changed, ...)` are unchanged since it still subclasses `TextArea` and posts the same messages.

Mechanism, verified against Textual 8.2.8 with a deterministic reproduction: Textual detaches a widget and clears its component styles before a screen repaint already queued can run. A queued `render_lines` call on a stock `TextArea` then raises `KeyError: "No 'text-area--gutter' key in COMPONENT_CLASSES"`. `Tests/Widgets/test_detach_safe_text_area.py::test_stock_text_area_raises_after_detach` reproduces this deterministically as a negative control (mount, detach, call `render_lines` directly). `test_attached_detach_safe_text_area_renders_like_stock` confirms `DetachSafeTextArea` renders byte-for-byte identically to stock `TextArea` while attached. 3/3 tests pass, including the negative control.

Evidence: the 10x10 before/after loop against `Tests/UI/test_mcp_workbench.py` on the fast-lane minimal venv showed 0 `text-area--gutter` failures on both `origin/dev` (before) and this branch (after) -- the loop did not reproduce the flake in this run, so the deterministic unit test is the primary evidence and the loop is supporting only. Ruling-4 comparison: the dev `.venv` full run of `Tests/UI/test_mcp_workbench.py` + `Tests/UI/test_mcp_tools_mode.py` was 408 passed / 0 failed / 0 errored on both this branch and an `origin/dev` base worktree -- identical (empty) FAILED/ERROR sets, no regressions.

The regression test (`Tests/Widgets/test_detach_safe_text_area.py`) is gated in the PR fast lane: it is in `FAST_LANE_TARGETS` (`Tests/CI/test_ci_queue_pressure_contract.py`) and in derived-artifacts.yml's fast-lane pytest invocation, added after a Qodo finding on PR #2866 that the fast lane's explicit list omitted it. The 10x10 before/after loop against the fast-lane minimal venv was inconclusive (0 `text-area--gutter` failures in 10 runs on both `origin/dev` and this branch); the deterministic negative-control test above is the evidence that carries this task. Merged as #2866.
<!-- SECTION:NOTES:END -->
