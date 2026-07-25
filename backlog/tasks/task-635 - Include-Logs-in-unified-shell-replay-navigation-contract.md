---
id: TASK-635
title: Include Logs in unified shell replay navigation contract
status: Done
assignee:
  - '@codex'
created_date: '2026-07-25 19:31'
updated_date: '2026-07-25 19:31'
labels:
  - ui
  - navigation
  - tests
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore first-run and Nielsen unified-shell replays after Logs became a canonical top-level destination by aligning their shared navigation inventory with ADR-015.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The shared replay navigation inventory includes Logs in canonical order.
- [x] #2 First-run and Nielsen replays reach their runtime assertions instead of timing out on an impossible navigation count.
- [x] #3 The focused replays and unified-shell release-gate block pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the two timeout failures and verify the shared 12-entry inventory disagrees with the 13 mounted destinations.
2. Add Logs to the shared expected inventory in canonical order.
3. Run both focused replays, the unified-shell gate block, and static checks.

ADR required: no
ADR path: backlog/decisions/015-shell-destination-ia.md
Reason: The existing ADR already establishes Logs as a top-level destination; this task only repairs a stale test oracle.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added the canonical `nav-logs` entry between Lab and Settings in the shared
  replay inventory used by both first-run and Nielsen gates.
- Both formerly timed-out replays passed and reached their full visible-state
  assertions; the complete 26-test unified-shell block also passed.
- Ruff, formatting, compile, and diff checks passed. No production behavior
  changed.
<!-- SECTION:NOTES:END -->
