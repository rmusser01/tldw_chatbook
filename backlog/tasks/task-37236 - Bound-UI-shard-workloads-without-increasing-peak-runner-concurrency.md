---
id: TASK-37236
title: Bound UI shard workloads without increasing peak runner concurrency
status: In Progress
created_date: 2026-10-28 19:08
labels:
- ci
- testing
- textual
priority: high
references:
- backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
- scripts/ui_lane_shard_replay.py
- backlog/tasks/task-35834 - UI-Fast-Lane-job-9-runs-within-a-minute-of-its-20-minute-cap-on-dev.md
assignee:
- '@codex'
updated_date: 2026-10-28 19:38
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the existing UI gate finish with its documented headroom despite observed checkout delays and full setup/call/teardown work. Three17-shard matrices timed out after limited-position balancing; checkout reached289seconds and per-case teardown added substantial cost. Use measured whole-census replay to reduce each serial slice while preserving the current instantaneous runner ceiling and every selected test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The full1140-entry census is conserved exactly once, including all audit and previously split cases; no selected target, profile guard, timeout, dependency or required aggregate is relaxed.
- [x] #2 The UI matrix has28serial shards with max-parallel17, preserving the prior23-test-runner peak (24withPerf); the20-minute job and180-second test limits remain.
- [x] #3 Official all-phase/progress replay across retained CI runs plus observed checkout reserve and15percentvariance supports the layout under the existing1020-second target; projected new Media controls are explicitly budgeted until measured.
- [x] #4 Executable CI contracts distinguish total eligible jobs from peak concurrency and reject missing/raised parallel limits, missing shards and inconsistent count statements.
- [ ] #5 ADR-103, active count statements and the implementation plan document the measured capacity decision and why limited-prefix balancing was insufficient; targeted contracts and scoped Superpowers review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (amend the existing canonical ADR, do not create a duplicate).
ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md.
Reason: separate eligible UI job count from bounded instantaneous concurrency under the accepted required-gate policy.
1. Preserve all prior failed matrices and inspect actual checkout/setup/test/teardown boundaries; use existing official replay rather than call-only residual forecasts.
2. Amend ADR-103 before implementation with the measured28-shard/max-parallel17decision, all1140targets conserved, setup310seconds and15percentvariance target1020seconds.
3. Add failing contract pins for the matrix/parallel ceiling and the distinction between total eligible jobs and concurrency.
4. Reorder all census blocks with deterministic LPT using official worst-observed all-phase costs, preserving exact entries/comment blocks; set matrix28/max-parallel17 and update active statements only. Keep all timeouts, dependencies, required gate and native controls unchanged.
5. Run targeted CI contracts/replay/source conservation, retain exact receipts and fresh scoped Superpowers review. Root owns final hosted CI and task closure. Coordinate with TASK-37235 so its whole-file controls are budgeted, not duplicated.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The UI gate now has28serial shards with max-parallel17, retaining the former23ordinary-test-runner peak (24withPerf) while scheduling34eligibleapplication jobs. All1140census selectors are retained exactly once through deterministic whole-census LPT placement; comment provenance remains attached, and the replay checkpoint shape now covers1140entries rather than the old92-entry block. All20-minute job/180-second test caps, minimal dependencies, required aggregation, fail-fast=False and selected targets remain unchanged. ADR-103 was amended before implementation; it records the extra setup/later-wave cost and the eligible-versus-parallel distinction.

Three prior17-shard layouts remain failed evidence. Actual012ca checkout took289seconds in shard1 and131in13, and per-case teardown consumed50–145seconds in many jobs. Limited-prefix call-only residual models were insufficient. Existing scripts/ui_lane_shard_replay.py supplies worst-observed progress/all-phase costs across retained860/d48/48ef/012ca runs; failed/cancelled logs supply timing observations only. Loaded-local Media-control XML totals116.644seconds, padded to120 in the model, not claimed hosted capacity. Work16850.38seconds;28-way phase max602.73s, plus310s setup/checkout reserve and15percentworkvariance forecasts1003.14s below1020target. Even ideally balanced27shards lower-bound1027.70s exceeds target. Actual successor-headCI remains required.

TDD:8new policy/count tests fail against oldconfig (0errors/skips); new format checkpoint separately fails beforeitsheaderexists. First broad candidate197pass/1formatfail preserved. Corrected fullcheckpoint + current28CLI split + dispatch/count/queue/replay contracts:204passed0F/E/S97.65s; after AST-preserving style and measured-allowance reweighting,21final pins pass0F/E/S16.18s. New mutations reject missing/raised parallel limits and missing/duplicate shards. Topology pins distinguish total eligible jobs from peak concurrency at different ceilings. Allnewlintchecks pass; existing large-file formatter debt remains baseline-qualified. PureCI tests only, nofullsuite.

Evidence: .test-tmp/textual-37031-37034/capacity-full-phase/final-proof.json and capacity37236-{red01,format-red01,green01,green02,final-pins01}/. TaskremainsInProgress/AC5open for root independentSuperpowersreview and finaldelivery. ExistingADR103 plus earlierTASK35834capacity work; TASK37235controls are budgeted aswhole-file additions, notduplicated selectors.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
