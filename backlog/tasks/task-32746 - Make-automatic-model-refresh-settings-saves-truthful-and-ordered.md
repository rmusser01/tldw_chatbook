---
id: TASK-32746
title: Make automatic model-refresh settings saves truthful and ordered
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 18:06'
updated_date: '2026-09-17 18:26'
labels:
  - ui
  - settings
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the component migration review so automatic-refresh controls accurately show pending, saved and failed changes and preserve the latest user choice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Keyboard edits to refresh controls persist only the intended model_catalog fields and never record startup consent.
- [x] #2 Failed writes retain choices with visible recovery and an accessible retry; pane rebuilds preserve pending edits and feedback.
- [x] #3 Rapid edits are serialized so an older write cannot overwrite a newer choice, including returning to a previously saved value.
- [x] #4 Empty, invalid and non-finite refresh intervals do not write config and show how to recover.
- [x] #5 Targeted production-CSS tests, private native persistence journeys, static checks, review and migration completion tracking document the verified scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: existing backlog/decisions/020-automatic-model-catalog-refresh.md; ADR-031/150/161 govern keyboard and component patterns. Reason: preserve instant application, consent separation and Settings-owned threaded config persistence; repair ordering and truthful feedback without a new service/storage boundary.
1. Reproduce failed/overlapping writes, value roundtrips, invalid intervals and pane-rebuild loss in production-CSS tests before production edits.
2. Keep pending form values in the existing Settings owner, drain writes in order off the event loop, and show token-backed pending/saved/error/validation feedback with Retry. Reject stale completions and preserve startup consent.
3. Run relevant existing settings/catalog tests, perform isolated real-config native keyboard journeys in dark/light at wide/compact sizes, inspect one capture batch, check lifecycle/default isolation and review the diff.
4. Update the guide, audit and a requirements-to-evidence migration completion ledger; record exact remaining destinations/integration gates, close the atomic task and push the verified continuation to draft PR #2704. No full sweep or merge into dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Automatic-refresh settings now retain pending/failed choices, serialize the latest config snapshot and gate receipts by revision. Failed writes show Retry; invalid/non-finite intervals do not write; fractional hours are keyboard-editable. Existing-token compact styles keep full labels visible. Edits omit startup consent.

Verification: 223 distinct targeted tests passed (19 new production-CSS cases); four private native dark/light wide/compact journeys passed with real successful writes after four injected false returns. Eight captures inspected; eleven private databases, default fingerprints, normal shutdown and PID absence verified. The injected failure is adapter-result evidence, not OS-denial evidence. Independent review found no actionable defects. New Python files pass Ruff/format/syntax; existing Settings lint drops from 118 to 117 with no added diagnostics. CSS boot budget is 616683/634050 bytes; no ratchets changed.

Modified Settings screen, source/generated styles, focused tests, user guide, workflow audit, QA artifacts and full completion ledger. Existing ADR-020/031/150/161 apply; no new ADR. No additional generalizable lesson arose. The ledger preserves remaining Settings/destination review and current-dev integration gates. No full suite or merge into dev ran. Task-ID collision check covered 256 refs and 27 worktrees before closure.
<!-- SECTION:NOTES:END -->
