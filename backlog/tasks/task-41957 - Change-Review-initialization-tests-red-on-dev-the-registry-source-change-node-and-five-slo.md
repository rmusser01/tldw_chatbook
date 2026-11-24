---
id: TASK-41957
title: "Change Review initialization tests red on dev: the registry source-change node and five slow readiness waits"
status: Done
assignee:
- '@claude'
created_date: '2026-11-19 07:30'
updated_date: '2026-11-24 08:00'
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

**Cause.** The five red nodes (`[registry]`, the three `handoff`/`consent_read`/`root_identity_read` cases and `test_source_replacement_while_readiness_waits_for_publication_lock`) share one cause. A bisect of dev's merges puts it at the #3480 merge (e10ce63fe8, 2026-11-12); the file passed at e34daa55df.

- #3480 made `_SnapshotTicket.returned()` finish its operation inside `self.activate()`.
- For an `_InitializationSource`, `activate()` enters `storage_scope()`, which calls `check()`. `check()` raises "native initialization source changed" once the original registry or service has been replaced.
- So `operation.finish()` and `finish_root()` never ran. The tool slot stayed counted (`tool_workers=1`) after the queue drained, and still was 5 s later with no thread holding it. It is a leak, not a race.

**Fix.** `Workspaces/change_review_ownership.py::_SnapshotTicket.returned` still tries the source's scope first. If the scope refuses, the operation finishes under the original owner's activation instead, so a replaced source can no longer keep its slot.

**Evidence.**
- The file goes from 5 failed / 21 passed to 26 passed, on macOS and on Linux.
- 14 related files (Change Review consent, git, custody, ownership and initialization; unproven cleanup; turn context and admission; workflow scheduling; the opener, runtime-ownership and Settings UI files), at `-n 4` against a clean dev export: 46 failed on this branch and 54 on dev. Only the 9 repaired nodes differ, apart from `test_native_change_review_ownership.py::test_original_opaque_claim_is_borrowed_and_bounded_claim_refuses`, which is flaky on both trees (run alone: dev failed 3 of 3, this branch 2 of 3).
<!-- SECTION:NOTES:END -->
