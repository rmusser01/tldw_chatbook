---
id: TASK-35595
title: Avoid native storage initialization polling convoys
status: Done
assignee:
  - '@codex'
created_date: '2026-10-19 16:36'
updated_date: '2026-10-19 17:12'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Concurrent storage acquisitions repeatedly validate native paths while waiting for the same initialization owner, delaying Windows Console returns.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Contended initialization waits do not poll native filesystem checks while parked without notification
- [x] #2 Native admission, path/source/provenance checks, recursive-entry refusal, pause cancellation and physical retirement remain enforced
- [x] #3 Targeted native regressions and paired Windows profiling document improvement and remaining limits
- [x] #4 Pending-authority refusal verification establishes durable pending state before asserting refusal
- [x] #5 Initialization checks every existing ancestor creation-intent marker in one native verified traversal, retaining nofollow, ownership, mode and descriptor cleanup checks
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (amend existing ADR-126; no duplicate ADR). ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: define the bounded native directory visitor used to retain creation-intent checks while avoiding repeated root-to-parent walks. 1. Observe original timed waits and native validation under held initialization ownership. 2. Add failing native contention controls and wake/refusal cases. 3. Replace redundant timed wakeups after auditing notifications. 4. Audit existing failures; make pending-state refusal setup deterministic. 5. Reprofile; remaining initialization samples identify repeated ancestry walks. Amend ADR126, add native linear-open/refusal/cleanup controls, and perform a single verified traversal with every existing ancestor checked. 6. Independent review, targeted checks, paired original/current profiles and update PR3166.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced 10 ms initializer polling with existing notifications (parked checks 16 to 1); changed creation-intent validation to one verified ancestry walk (91 to 13 opens at depth 13), retaining native checks and descriptor retirement. Amended ADR-126 for the borrowed-descriptor visitor. Repaired the pending-authority fixture to witness durable publication before cancellation. Final focused 16 and existing 42 checks pass; 6 existing failures and 7 POSIX skips match original code, and 11 earlier lifetime failures reproduce unchanged. Identified diagnostic-only Windows polling feedback with a standalone 261,448 versus 92 loop-iteration control, then corrected both profile variants. Clock-aware six-return median input improved 940 to 744 ms and worst 1792 to 891 ms, with drafts retained, attach complete and zero network/profile guard events. Earlier short-poll results and limitations are retained. New code/tests/scripts pass targeted Ruff; nine existing exact-type findings are unchanged. Independent review found no missing notification or native check. Architecture, evidence and testing lesson updated. No full suite. PR #3166 follow-up; evidence: Docs/Development/evidence/windows-performance-2026-10-18/STORAGE_RESULTS.md.
<!-- SECTION:NOTES:END -->
