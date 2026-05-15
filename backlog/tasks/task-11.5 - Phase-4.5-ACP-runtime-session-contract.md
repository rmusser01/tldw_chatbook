---
id: TASK-11.5
title: 'Phase 4.5: ACP runtime session contract'
status: Done
assignee: []
created_date: '2026-05-12 00:00'
updated_date: '2026-05-15 04:35'
labels:
  - product-maturity
  - phase-4-agent-execution
  - acp
dependencies:
  - TASK-11.1
references:
  - Docs/superpowers/plans/2026-05-12-phase-4-agent-configuration-execution.md
parent_task_id: TASK-11
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make ACP runtime setup session readiness and Console follow states explicit without moving ACP ownership into Settings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ACP shows whether an ACP-compatible runtime is configured and what setup step is needed next.
- [x] #2 Session and Console-follow actions are enabled only when a real session payload is available.
- [x] #3 Runtime ownership remains under ACP while Settings remains limited to global defaults.
- [x] #4 QA walkthrough and focused regression evidence prove the ACP flow is usable or honestly blocked in the running app.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing mounted ACP regressions for missing runtime, configured runtime without session, and configured runtime with a real session payload.\n2. Add a small ACP-owned runtime/session state contract without moving setup into Settings.\n3. Wire ACPScreen to render configured runtime/session readiness and enable Console follow only with a session payload.\n4. Add Phase 4.5 QA evidence, update task and roadmap tracking, then run focused verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added an ACP-owned runtime/session state contract and wired `ACPScreen` to render missing-runtime, configured-runtime/no-session, and configured-runtime/session-payload states. Console follow now remains disabled until a real session payload exists, then hands off through the shared Console live-work contract with an ACP source and `local:acp_session:<id>` target. Runtime ownership copy remains in ACP and Settings is not presented as the runtime owner. Added mounted regressions and Phase 4.5 QA evidence with an approved textual-web/CDP screenshot at `Docs/superpowers/qa/product-maturity/phase-4/acp-runtime-session-2026-05-14.png`.
<!-- SECTION:NOTES:END -->
