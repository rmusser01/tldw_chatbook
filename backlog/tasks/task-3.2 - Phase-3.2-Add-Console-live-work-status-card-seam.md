---
id: TASK-3.2
title: 'Phase 3.2: Add Console live-work status card seam'
status: Done
assignee: []
created_date: '2026-05-03 20:35'
updated_date: '2026-05-03 20:41'
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
Extract pending Console launch rendering into a reusable live-work status card contract so future workflows, schedules, ACP, MCP, RAG, and artifact sources can render consistent status, recovery, action, and metadata without duplicating ChatScreen markup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console live-work card state derives stable badge, title, source, status, recovery, action, and payload rows from a launch context.
- [x] #2 ChatScreen renders pending launches through the reusable status card seam while preserving one-shot pending launch consumption.
- [x] #3 The rendered card exposes stable IDs/classes for future live-work source tests without changing existing user-visible copy regressions.
- [x] #4 Focused Console live-work tests and Phase 3 QA evidence verify the card-state model, mounted Console rendering, and roadmap/task updates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regressions for Console live-work card-state derivation and mounted card rendering with stable selectors.
2. Verify the new tests fail before production changes.
3. Add the minimal reusable `ConsoleLiveWorkStatusCardState` model and render helper, then wire ChatScreen through it without changing one-shot launch semantics.
4. Add durable Phase 3 QA evidence and update the unified-shell roadmap index.
5. Run focused Console/Home/destination regressions plus diff checks, then complete the Backlog task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a reusable Console live-work status card seam for pending launch context.

- Added `ConsoleLiveWorkStatusCardState` and row state derived from `ConsoleLiveWorkLaunch`.
- Updated `ChatScreen` to render pending launches through `_render_console_live_work_status_card()` while preserving one-shot pending launch consumption and existing copy.
- Added stable row selectors/classes for status-card source, title, status, recovery, action, and payload metadata.
- Added focused regressions plus Phase 3 QA evidence and roadmap tracking for `TASK-3.2`.
<!-- SECTION:NOTES:END -->
