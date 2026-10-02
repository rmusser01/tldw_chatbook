---
id: TASK-33647
title: Keep legacy trace batches within the storage-admission budget
status: Done
assignee:
  - '@codex'
created_date: '2026-10-02 15:14'
updated_date: '2026-10-02 16:57'
labels:
  - ci
  - performance
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The read-only completion precheck introduced for parked trace maintenance adds a second admission when a batch still needs normalization. Keep the existing storage-unit budget without weakening the idle read-only path, mutation rechecks or finite worker ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A fresh file-backed normalization batch and completed idle batch each pay at most two storage admissions through the existing finite worker helper.
- [x] #2 Idle completion uses no immediate transaction, and normalization still rechecks maintenance and migration state under the immediate transaction.
- [x] #3 The existing Console storage-unit ratchet and relevant maintenance and worker-ownership checks pass without changing ceilings, warmup or measured work.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the exact CI admission breach and pin the first incomplete fresh-worker callback with a real file-backed regression. 2. Reuse the existing core repository operation across the read-only probe and write fallback while leaving run_owned_db_call and physical compaction unchanged. 3. Run the unchanged storage-unit ratchet, read-only/refusal/rollback/parking/worker-ownership checks, static and diagnostic checks, then obtain independent review. ADR required: no new ADR. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: reuse an existing admitted repository scope without changing SQL transaction, schema, authority or runtime boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reuse the existing core repository operation for the legacy completion probe and normalization fallback. The read-only idle transaction and immediate write rechecks stay separate; provider-active refusal occurs before admission, and run_owned_db_call and physical GC are unchanged. The real file-backed regression processes a cold legacy row, enforces a nonzero upper admission budget of two, then checks completed idle work and zero registered worker handles. Before the repair it fails at three admissions; the final predicate passes in 3.11s. The unchanged Console storage-unit guard passes in 34.33s (trace 1.875 admissions per tick; all original ceilings, phases and canaries preserved). Twelve affected maintenance cases pass in 12.08s, and eight independent refusal/rollback/read-only/parked-wake/borrowed/in-flight cleanup cases pass across 8.67s and 2.29s. Independent source review approves. Three startup/exact-schema guards pass in 22.60s, imports 679/686 and UI-ready 1032/1033. Final changed-code static checks cover 81 Python files, ten new-file Ruff/format checks pass, owned formatting and whitespace pass, and the audited diagnostic inventory remains unchanged. ADR required: no new ADR; existing backlog/decisions/126-complete-local-backup-and-recovery.md. Modified production scope, one existing test module, qualification records and testing lesson. The whole census can pass before the repair; the controlled cold-worker test is the red proof. CI failed at 17/8 on published 68646eeeaa; fresh published-head remote checks remain required. No full suite, live provider, Windows, aggregate-resource or merge result.

Preserving rebase onto dev eba4305d8389a2112c99ac19fa804e9abea394ba retains the reviewed production/test bytes. Nine targeted Delete/Undo persistence/Save/native-close/cold-admission cases pass in 26.72s; unchanged storage/startup/CSS guards pass four cases in 56.00s; independent mounted Delete/Undo and accepted completion nonreplay pass two cases in 11.62s. The 81-file static scan, ten new-file Ruff/format checks, whitespace and generated bundles pass. Incoming diagnostic inventory verifies 632 owners and 15 sinks. No new ADR; existing ADR-126 and ADR-199 apply. Published 68646eeeaa has three passing CI jobs and the repaired latency failure; its Qodo summary resolves all four findings. Fresh publication and all current-head remote gates remain before merge.

Required clean rebase onto dev 6958e8dfa99a66680b1aec09fed65df0f1a91955 after published e8474 passes all four jobs/Qodo and GitHub reports BEHIND under strict protection. Only upstream store bytes change among the 81 qualified patch Python files; TASK-33647 bytes remain unchanged. Ten first-send/Save/close/nonreplay/cold-admission cases pass in 45.18s, four unchanged storage/startup/CSS guards in 55.02s, and independent persistence/Inspector plus lifecycle review approve four and six cases in 3.90s and 3.34s. Final 81-file/new-file static, worker-contract and 633-owner diagnostic guards pass. CSS/schema/recovery/runtime bytes remain unchanged. Existing ADR-069/126/199 apply, no new ADR or runtime repair. Original pins retained at 680/686 imports and 1033/1033 UI-ready. Fresh published-head gates and protected merge remain; earlier qualification limits persist.

Required rebase onto dev 30ca4552b3b881bee6c077db08c44233e5e97e6a after all four jobs/Qodo pass on published 64d770 and upstream hook-review changes cause an actual conflict. Only testing-lesson appends conflict; both entire blocks are preserved exactly. Only upstream ChatScreen changes among 81 patch Python files; TASK-33647, runtime, schema/recovery and CSS bytes stay unchanged. Twelve real hook-refund/prepared-close/Save/native-close/nonreplay/readiness/cold-admission cases pass in 72.48s; four unchanged storage/startup/CSS guards pass in 58.06s. Independent controller and mounted modal/diagnostic reviews approve six and nine cases in 3.82s and 40.38s. Final static/new-file, worker-contract and 634-owner diagnostic guards pass. No new ADR/runtime fix; ADR-148/126/199 preserved. Original 680/686 and 1033/1033 pins retained. Fresh published-head gates and protected merge remain. Separate upstream TASK33621.28 stays open; no global live-app Ctrl+Q claim. Earlier qualification limits persist.
<!-- SECTION:NOTES:END -->
