---
id: TASK-34926
title: Move-Workflows-on_mount-loads-off-the-app-message-pump
status: Done
assignee: []
created_date: 2026-10-13 14:51
labels:
- performance
- ui-perf
- wave-3-screen-switch
dependencies: []
priority: high
updated_date: 2026-10-13 17:46
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WorkflowsScreen.on_mount awaits _initialize_authoring() — which itself awaits ensure_workflow_authoring() and then a full-screen await self.recompose() DURING mount (workflows_screen.py:126-157) — and then awaits controller.load(). The repo documents this exact trap elsewhere (personas_screen.py:1938: awaiting here ran it on the App's message pump — the app stopped responding). First navigation to Workflows freezes the whole TUI until load completes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] on_mount returns before authoring init and load complete — loads run in workers or call_after_refresh
- [x] No await self.recompose() during mount
- [x] Saved selection/section/workflow-views state still restored correctly (test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read WorkflowsScreen on_mount/_initialize_authoring and dependents of _loaded/controller
2. Move authoring init + controller.load into a worker so on_mount returns fast
3. Preserve saved-state restore and session attach semantics
4. Tests + targeted runs + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- `on_mount` is now synchronous and spawns one exclusive `workflows-mount-load` worker running the unchanged sequence as `_load_authoring_after_mount`: authoring init (including the recompose, now on a mounted screen inside a worker, not during mount), draft subscription, `controller.load`, saved-state restore (selection/section/views), session attach. Every step after an await re-checks `is_mounted`; the render callbacks (`_show_status`, `_render_authoring`, `_layout_panes`) already guard on it, so a worker outliving an unmounted screen is harmless. `exit_on_error=False` keeps the bounded-recovery contract (the try/except and Retry copy are unchanged).
- Tests: new `Tests/UI/test_workflows_mount_worker.py` (on_mount spawns exactly one exclusive worker and returns without loading inline). 1/1 pass. Regression: `Tests/UI/test_workflows_run.py` + `test_workflows_stylesheet_loading.py` — 65/65 passed (these mount WorkflowsScreen through real navigation and exercise the restore/attach/retry paths end-to-end).
- ADR required: no — scheduling change to match the repo's documented worker-mount idiom.
