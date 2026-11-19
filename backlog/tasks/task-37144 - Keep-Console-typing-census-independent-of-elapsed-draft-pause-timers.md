---
id: TASK-37144
title: Keep Console typing census independent of elapsed draft-pause timers
status: Done
created_date: 2026-10-28 04:00
assignee:
- '@codex'
labels:
- ci
- tests
- performance
priority: high
references:
- TASK-36390
- TASK-33260
- TASK-33005.2
- Tests/Performance/test_console_keystroke_work_census.py
- backlog/decisions/097-boot-budget-ratchets.md
- backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
updated_date: 2026-10-28 04:39
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The PR3283 ready-event Perf run40610052389 counted79 readiness builds for24 keys and exceeded the unchanged three-per-key bound, while its synchronize run passed on the identical merge b8e2b1e735e0a4d5dfef6d58f0e1a2c3b036b11b. Native caller tracing shows one build per key plus three for each elapsed draft-pause refresh. Keep typing metrics independent of runner speed while retaining paused-work coverage. This is a distinct residual phase-attribution gap after completed TASK-36390, not a new product performance feature or duplicate of TASK-37000.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The native per-key census remains within its existing positive and upper bounds under controlled elapsed pause intervals between real Pilot key dispatches; the new regression fails before the fix without setup/teardown errors or skips.
- [x] #2 The existing storage-unit typing-pause, idle and revisit work remains exercised with unchanged callbacks and ceilings; phase separation does not discard its coverage.
- [x] #3 Production behavior, dependencies and existing guard caps/timeouts/shard counts are unchanged; the controlled regression is enrolled in the existing Perf Guard step and targeted checks plus scoped Superpowers review qualify the repair. Actual current-head CI remains a separate PR delivery gate.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the false attribution with a controlled clock around native Textual timers: retain actual Console, coordinator callbacks and Pilot dispatch; inspect actual child XML. 2. Apply the existing storage-census typing/pause separation to all typing variants without raising the three-per-key bound or caching live readiness. 3. Rerun the new regression and two existing Perf nodes; prove the storage_units=True path/callbacks and its existing hosted pause/idle guard stay unchanged. 4. Enroll the new native control in the existing Perf step and run focused style/CI checks. 5. Obtain Superpowers scoped review and inspect every current-head hosted run; record exact evidence and limitations. ADR required: no. ADR path: N/A. Reason: bounded test-harness phase attribution and existing-lane enrollment only, no production/dependency/runtime policy change. Existing ADRs: backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md.
6. Preserve paused in-memory coverage with one explicitly billed native draft refresh after the controlled 400-message key burst; assert positive readiness and unchanged zero-history/snapshot/context-row bounds. Add only a default-false census opt-in, prove the default/storage executable paths remain equivalent, re-prove controlled-clock RED with only the timer hold reverted, then run the strengthened native control GREEN. No new pause CPU budget or guard increase.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Separated typing from elapsed draft-pause work for every census variant, reusing the existing storage-mode timer hold. Added one native controlled-clock regression on a 400-message Console: real Pilot dispatch, native Timer handles/coalescing and the production coordinator callbacks remain active; every held timer is stopped and its task drained in finally. A default-false draft_pause observation separately bills one native stop/refresh and drains its publications, preserving in-memory history/snapshot coverage as well as the unchanged storage pause/idle/revisit coverage. Enrolled only this node in the existing Perf Guard keystroke step.

Native TDD evidence (actual private-child XML, not wrapper dots): ci-repair37144-red01 failed only at 97 readiness builds/24 keys > the existing 72-build ceiling, with 24 native timers/24 delivered callbacks. Initial ci-repair37144-green01 passed all three targeted children (new control, transcript snapshot guard, empty/400-message scaling); both scaling arms recorded 25 builds. After root review strengthened paused in-memory coverage, ci-repair37144-red02 again failed only at 97/24 > 3; the explicit pause independently recorded 3 readiness builds and zero history/snapshot rows. Its wrapper restored the exact final source in finally. Final ci-repair37144-green02 passed: 25 builds/24 keys, 24 timers, no due typing callbacks, plus 3 separately counted paused readiness builds; all typing/pause history, snapshot and projected-row counters zero, context-estimate maxima 0/1 <= 1. Every child had zero errors/skips; the only failures were the two intended RED assertions.

Source qualification: the final draft_pause=False executable helper AST equals the exact already-passed GREEN01 source (reconstructed SHA256 67a4e18ec403413d81b989e40fa6a6562692ecf930a4476b220f41183d6d4b59). The storage_units=True/draft_pause=False executable AST equals the original base; all 23 other existing functions and all module assignments/ceilings are unchanged. Native storage callbacks and ceilings are preserved, without claiming a fresh local storage-mode run or adding a paused CPU ceiling. Full production tree and diagnostic inventory Git objects are unchanged. Workflow comparison proves exactly one executable target addition with all other settings retained. Ruff check and whitespace checks pass; formatter comparison preserves the same five inherited regions and introduces no formatting debt.

Related completed ownership: [TASK-36390](task-36390%20-%20Tests-Performance-compact-setup-voice-latency-tab-sync-and-keystroke-tests-are-red-on-dev-and-run-in-no-PR-lane.md) owns the earlier cold-publication/fixture warming and Perf enrollment; [TASK-33260](task-33260%20-%20PERF-01-Perf-guards-that-count-admissions-helper-spawns-and-pre-import-payload.md) owns admission/helper/pre-import and paused/idle storage ratchets; [TASK-33005.2](task-33005.2%20-%20Make-Console-readiness-honour-known-connection-evidence.md) owns live readiness/evidence accounting. This repair closes the distinct residual elapsed-pause attribution gap. [TASK-37000](task-37000%20-%20Console-a-first-send-rebuilds-the-collapsed-Inspectors-run-section-and-lays-out-hidden-widgets.md) concerns first-send collapsed-Inspector layout, a separate scope.

ADR required: no. ADR path: N/A. Existing governance: [backlog/decisions/097-boot-budget-ratchets.md](../decisions/097-boot-budget-ratchets.md) and [backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md](../decisions/103-fast-pr-lane-and-required-gate-aggregation.md). No production behavior, dependency, cap, timeout or shard-count change.

Receipts: .test-tmp/textual-37031-37034/ci-repair37144-{red01,green01,red02,green02}/; consolidated actual child receipts and final static proof are ci-repair37144-child-receipts.json and ci-repair37144-final-static-proof.{py,json} in that parent. Local Python 3.12.10 / pytest 8.3.5 / Textual 8.2.8; inherited asyncio fixture-scope, Pydantic exclude_if and pydub/audioop warnings remain. Scoped Superpowers review of a0457fe1c7 is approved with zero Critical/Important/Minor findings; root independently verified all six original child XML receipts. AC3 is qualified. Actual current-head hosted CI remains a separate pending PR delivery gate, not a claimed pass. No full suite was run.

Portable evidence and full scoped review: Docs/superpowers/qa/2026-10-27-textual-followups-implementation.md and its linked evidence folder. Root adjudicated all six declined review boundaries with costs. Local repair complete; PR delivery remains pending hosted checks.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
