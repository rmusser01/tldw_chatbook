---
id: TASK-34916
title: Equality-gate-the-llamacpp-setup-2Hz-repaints
status: Done
assignee: []
created_date: 2026-10-13 14:45
labels:
- performance
- ui-perf
- wave-1-timers
dependencies: []
priority: medium
updated_date: 2026-10-13 16:09
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
llamacpp_setup_view arms set_interval(0.5, refresh_state) (llamacpp_setup_view.py:198) which re-renders launch state, query_ones the profile Select plus 5 buttons, snapshots owner state, and updates Statics every tick with no equality guard — unlike LabScreen.refresh_lab_status which guards its writes. Pure repaint churn at 2Hz while the Lab > llama.cpp view is mounted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Widget writes happen only when computed state differs from last rendered state
- [x] query_one and snapshot lookups skipped or cached when state is unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read llamacpp_setup_view refresh_state chain and what it repaints
2. Equality-gate writes and skip snapshot/query_one when unchanged
3. Unit-test the gate
4. Targeted run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- `_refresh_state_locked` (llamacpp_setup_view.py) now builds a gate tuple from every input the paint derives from — (active, profile-busy, profiles-loaded, snapshot code/state/target/model_ids, live claim's launch preview, local claim's exposed + cancel flags) — and skips the entire query_one/Static/Button walk when it is unchanged. Snapshot fields are read as VALUES because `LlamaCppConnectionOwner.snapshot()` returns a cached object mutated in place, so identity would never change.
- `_render_diagnostics()` stays OUTSIDE the gate (runs every tick) because its diagnostics sink changes independently and it already carries its own identity gate (`_last_diagnostics`). The duplicate `owner.snapshot()` call in the body was folded into the gate's single call.
- Tests: new self-contained `Tests/UI/test_llamacpp_setup_refresh_gate.py` (3 tests: skip-when-unchanged with diagnostics still ticking, repaint on connection state/code change, repaint on profile-load change) — 3/3 pass.
- Regression: Tests/UI/test_llamacpp_{handoff,launch_preview,setup_view}.py show the identical 5 failures with the change stashed (pre-existing on this branch HEAD); no new failures introduced.
- ADR required: no — equality gate inside one view's refresh callback.
