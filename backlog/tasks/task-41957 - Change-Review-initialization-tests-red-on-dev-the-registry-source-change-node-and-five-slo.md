---
id: TASK-41957
title: "Change Review initialization tests red on dev: the registry source-change node and five slow readiness waits"
status: Done
assignee:
- '@claude'
created_date: '2026-11-19 07:30'
updated_date: '2026-11-23 23:05'
labels:
- follow-up
- tests
- workspaces
priority: low
references:
- Tests/Workspaces/test_native_change_review_initialization.py
dependencies:
- TASK-41635
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** `test_admitted_callback_cannot_publish_ready_after_original_source_changes[registry]` ends with `executions != ()` on dev and on TASK-41635's branch; five source-replacement and readiness-handoff waits in the same file also fail on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each node has a recorded cause and passes after repair or fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Find the commit that turned the file red and the nodes' common cause.
2. Fix the cause in the product code, not the tests.
3. Compare the related Change Review suites with dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The light CI tier (TASK-42521) ran this file whole on #3688 and stopped it. Checked on dev 44e7a47e4f, on macOS and on Linux (python:3.12-bookworm, non-root).

**Node count.** Exactly five nodes were red, not the six the description implies: `[registry]`, the three `handoff`/`consent_read`/`root_identity_read` cases and `test_source_replacement_while_readiness_waits_for_publication_lock`.

**Cause.** The file arrived on dev red with #3480 (merge e10ce63fe8, 2026-11-12); it was added on that branch in 47821e879a. Inside the branch, the cause landed in 11d71e5db5 ("Integrate ordinary Console physical ownership"):

- 11d71e5db5 made `_SnapshotTicket.returned()` finish its operation inside `self.activate()`. Before it, `returned()` called `operation.finish()` directly.
- For an `_InitializationSource`, `activate()` enters `storage_scope()`, which calls `check()`. `check()` raises "native initialization source changed" once the original registry or service has been replaced.
- So `operation.finish()` and `finish_root()` never ran. The tool slot stayed counted (`tool_workers=1`) after the queue drained, and still was 5 s later with no thread holding it. It is a leak, not a race.
- The local review found the same leak one step earlier. Three call sites in `Workspaces/change_review_consent.py` (the worker's `custody`, `begin_close` and `_schedule`'s `finally`) called `returned()` inside an outer `ticket.activate()`. A source that changed after admission, while the work was queued, raised there before `returned()` ran. At shutdown, `begin_close` would also have raised out of the runtime's disposal.

**Fix.**
- `_SnapshotTicket.returned()` (`Workspaces/change_review_ownership.py`) still tries the source's scope first. If the scope refuses, it logs the exception type at debug level and finishes the operation under the original owner's activation. Every callback `finish()` runs is in-memory and needs no source scope.
- The three consent-service sites call `returned()` outside any activation, as `change_review_finalization.py` already does.
- New test `test_source_change_after_admission_still_returns_the_slot` replaces the source right after admission. It fails without the consent-service change (the ticket never retires) and passes with it.
- The debug call adds one row to `Docs/security/production-diagnostic-inventory.json`. It logs only the exception type name: no path, no user content.

**Evidence.**
- The file goes from 5 failed / 21 passed to 27 passed (26 plus the new test), on macOS and on Linux.
- 14 related files (Change Review consent, git, custody, ownership and initialization; unproven cleanup; turn context and admission; workflow scheduling; the opener, runtime-ownership and Settings UI files), at `-n 4` against a clean dev export: only the repaired nodes differ. `test_native_change_review_ownership.py::test_original_opaque_claim_is_borrowed_and_bounded_claim_refuses` is flaky on both trees (run alone: dev failed 3 of 3, this branch 2 of 3).
<!-- SECTION:NOTES:END -->
