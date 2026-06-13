---
id: TASK-3.10
title: 'Phase 3.10: Launch active Workflows run from Workflows into Console'
status: Done
assignee: []
created_date: '2026-05-03 19:18'
updated_date: '2026-05-03 19:22'
labels:
  - unified-shell
  - phase-3
  - console
  - workflows
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md
  - Docs/superpowers/trackers/unified-shell-maturity-roadmap.md
parent_task_id: TASK-3
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Workflows destination expose a real Console launch action when the existing active-work adapter can identify an actionable workflow run, while preserving an honest disabled state when no run context exists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workflows destination keeps Console launch disabled with recovery copy when no actionable workflow run exists.
- [x] #2 Workflows destination enables Console launch when a visible Workflows active-work item has Console launch context.
- [x] #3 Clicking the enabled Workflows Console launch action routes through the existing Home active-work adapter Console launch path.
- [x] #4 Focused automated tests and Phase 3 QA evidence verify the `workflows-launch-in-console` producer and roadmap/task updates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regressions for Workflows destination disabled fallback, enabled active-run Console launch, adapter-routed click behavior, source readiness, off-thread adapter loading, and Phase 3.10 tracking evidence.
2. Reuse the existing Home active-work adapter from Workflows to discover an active Console-capable workflow run without inventing a separate workflow service.
3. Render the Workflows Console launch button as enabled only when that adapter context exists, and route clicks through the existing app-level Home Console launch method.
4. Add Phase 3.10 QA evidence plus roadmap and task updates.
5. Run focused destination, Console handoff, navigation, and diff hygiene checks before commit/PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the Workflows destination to reuse the existing Home active-work adapter for active workflow-run Console discovery. The `workflows-launch-in-console` action now stays disabled with explicit recovery copy when no active workflow run exists, becomes enabled when adapter context exists, and routes clicks through `open_active_home_item_in_console` so Home and Workflows share the same Console launch path. Updated Console source readiness to show Workflows as connected and added focused destination/Console regressions plus Phase 3.10 QA evidence and roadmap tracking. The active-work lookup runs in a Textual thread worker so Workflows composition does not block on adapter work.
<!-- SECTION:NOTES:END -->
