---
id: TASK-3.6
title: 'Phase 3.6: Show Console live-work source readiness'
status: Done
assignee: []
created_date: '2026-05-03 22:59'
updated_date: '2026-05-03 23:07'
labels:
  - unified-shell
  - phase-3
  - console
  - live-work
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md
  - Docs/superpowers/trackers/unified-shell-maturity-roadmap.md
parent_task_id: TASK-3
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Show a compact Console live-work source readiness summary so users can see connected live-work sources while ACP and MCP remain honest unavailable states until source-specific payloads exist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console source readiness state marks connected sources as connected and remaining planned sources as explicitly unavailable or not wired with recovery copy.
- [x] #2 ChatScreen renders the source readiness summary when no pending live-work launch is staged and keeps pending launch cards focused when a launch exists.
- [x] #3 The readiness summary exposes stable IDs/classes for source-specific follow-up tests without creating fake source actions.
- [x] #4 Focused Console tests and Phase 3 QA evidence verify source readiness state, mounted Console rendering, and roadmap/task updates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regressions for Console source-readiness state, mounted no-pending rendering, pending-launch suppression, and Phase 3.6 tracking evidence.
2. Verify the new tests fail before production changes.
3. Add the minimal `ConsoleLiveWorkSourceReadinessState` display model and render helper, then wire ChatScreen to show it only when no pending launch exists.
4. Add durable Phase 3 QA evidence and update the unified-shell roadmap index.
5. Run focused Console/Home/W+C/navigation regressions plus diff checks, then complete the Backlog task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `ConsoleLiveWorkSourceReadinessState` with stable source row IDs/classes, marking connected source families connected and future live-work source families as informational `Not wired` rows with recovery copy. `ChatScreen` now renders the readiness summary only when no pending Console live-work launch is staged, preserving the focused pending launch card when a launch exists. Added regression coverage for the display model, mounted Console rendering, pending-card suppression, and Phase 3.6 evidence/roadmap/task tracking.
<!-- SECTION:NOTES:END -->
