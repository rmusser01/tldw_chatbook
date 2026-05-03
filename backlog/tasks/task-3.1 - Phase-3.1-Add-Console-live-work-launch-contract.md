---
id: TASK-3.1
title: 'Phase 3.1: Add Console live-work launch contract'
status: Done
assignee: []
created_date: '2026-05-03 19:14'
updated_date: '2026-05-03 19:19'
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
Replace ad hoc pending Console launch dictionaries with a typed, backward-compatible live-work launch contract so source destinations can pass consistent status, recovery, action, and payload metadata into Console.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console live-work launch model normalizes source, title, payload, status, recovery, and action metadata.
- [x] #2 TldwCli.open_console_for_live_work remains backward compatible with existing source/title/payload callers while storing typed normalized launch context.
- [x] #3 ChatScreen renders pending Console launch details including source, title, status, recovery/action copy, and payload metadata, then clears the one-shot app pending value.
- [x] #4 Focused Console live-work tests and relevant Home/Console regressions pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused regression tests for the Console live-work launch payload contract, backward-compatible app helper behavior, and pending-launch card rendering.
2. Verify the new tests fail before production changes.
3. Add the smallest typed launch contract and wire TldwCli plus ChatScreen through it while preserving existing callers.
4. Add durable Phase 3 QA evidence and update the unified-shell roadmap index for this slice.
5. Run focused Console/Home regression tests plus git diff checks, then complete the Backlog task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a typed ConsoleLiveWorkLaunch contract for normalized live-work launch metadata.

- Added backward-compatible TldwCli.open_console_for_live_work handling for source/title/payload callers plus optional status, recovery, and action copy.
- Updated ChatScreen to normalize typed launches or legacy dicts, clear the one-shot pending value, and render source, title, status, recovery, action, and payload metadata.
- Added focused Console handoff regressions and Phase 3 QA evidence; updated the unified-shell roadmap to show Phase 3 as in progress.
<!-- SECTION:NOTES:END -->
