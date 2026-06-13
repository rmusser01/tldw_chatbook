---
id: TASK-12.2
title: 'Phase 5.2: Active server auth live status'
status: Done
assignee: []
created_date: '2026-05-16 00:00'
updated_date: '2026-05-16 05:30'
labels:
  - product-maturity
  - phase-5-server-parity
dependencies: []
parent_task_id: TASK-12
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose active-server, auth, reachability, and recovery state in the running app while preserving local mode usability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 QA walkthrough verifies local mode, missing server, auth-required, unreachable, and ready server states are understandable in the running app.
- [x] #2 Focused regression evidence proves local mode does not require server credentials and server mode does not silently fall back to local writes.
- [x] #3 Repo-tracked QA evidence exists under Docs/superpowers/qa/product-maturity/phase-5/.
- [x] #4 P0/P1 findings are fixed or explicitly accepted according to the spec severity policy.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused failing regressions for Home dashboard server status in local, missing-server, auth-required, unreachable, and ready states.
2. Extend Home dashboard input/state summary with backend-owned runtime source/server fields from RuntimeSourceState.
3. Wire the local Home adapter to runtime_policy without storing credentials or changing source authority.
4. Add Phase 5.2 QA evidence and update the Phase 5 QA index/task notes.
5. Run focused Home/server contract tests and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added runtime-source/server/auth/reachability fields to the Home dashboard state and summary labels.
- Wired Home adapters to the app-owned runtime policy without storing credentials or changing local/server write authority.
- Added Home adapter and mounted Home screen regressions for local, missing-server, auth-required, auth-expired, unreachable, and ready states.
- Added a Home UX/HCI polish pass that groups runtime readiness into a `System Status` pane, clarifies keyboard affordances, and delineates terminal-native columns for later resizable-pane work.
- Added Phase 5.2 QA evidence and captured actual textual-web Home screenshots; visual approval was granted after reviewing the rendered Home polish screenshot.
<!-- SECTION:NOTES:END -->
