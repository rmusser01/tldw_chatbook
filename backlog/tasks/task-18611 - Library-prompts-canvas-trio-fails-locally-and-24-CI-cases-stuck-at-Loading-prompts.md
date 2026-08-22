---
id: TASK-18611
title: >-
  Library prompts canvas: trio fails on clean dev and 24 CI cases stuck at
  "Loading prompts"
status: Done
assignee: []
created_date: '2026-08-19 15:30'
labels:
  - ui
  - library
  - testing
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two distinct failure modes in `Tests/UI/test_library_prompts_canvas.py`
found during TASK-18610 triage:

1. **Clean-dev trio (reproduces locally on macOS, pytest 9.1.1):**
   `test_library_prompt_import_blocks_undo_until_import_settles`,
   `test_cancelled_prompt_import_retains_writer_ownership_until_commit`,
   plus a rotating third (`..._delete_receipt_undo_restores_row_and_count`
   or `..._history_no_change_keeps_selection_and_retry_available` --
   order-dependent). Symptoms: an expected `PromptBatchTarget()` undo entry
   never appears, and recursion visible at
   `library_screen.py:8149 _study_count_or_none`. First appeared around
   35bb1aa98 (2026-08-18, project-skills import offer after workspace
   creation) -- bisect from there.
2. **CI-only 24 (ubuntu sharded runs, never reproduces locally):** every
   case times out after 15s waiting for `#library-prompt-row-5`; visible
   text shows "Loading prompts..." stuck -- a data-load worker that never
   settles on a headless runner (0.02s-poll loops, 517-568 polls).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The trio passes on a clean dev checkout (bisect 35bb1aa98 first).
- [x] #2 The CI-only stuck-loading mode is reproduced or instrumented (e.g. capture the load worker's state on timeout) and fixed.
<!-- AC:END -->

## Implementation Notes

**AC#1 (clean-dev trio) -- fixed and merged.** PR #1849 (`2a74a7b31`):
the retry Undo press targeted a cached `#library-prompts-delete-undo`
reference that the cancelled/settled import's recomposes had detached, so
the press silently no-op'd and the restore never ran. Both affected tests
now query the live mounted button at press time, plus a settlement wait
that drains the cancelled worker rather than trusting `import_finished`
alone. Root cause bisected to `0dfadf463` (perf(library): reconcile
snapshots below screen), NOT the `35bb1aa98` this task guessed at.
Verified 310 passing at the time of merge.

**AC#2 (CI-only stuck-at-Loading 24) -- verified RESOLVED, upstream.**
Initially deferred here as "superseded by PR #1838, re-check after it
merges"; that re-check is now DONE and is what closes this task. #1838
(task-18912) hardened `_wait_for_prompt_browse_scope` (canvas-settlement
condition + real deadline) and reworked `_open_prompts_list` in this same
file, and it merged 2026-08-20.

Evidence, run **32511976568** (2026-08-21, all 12 ubuntu UI shards,
dev@`2a15a72bb`): **338 canvas rows, 0 failures carrying the stuck-loading
signature** -- no timeout on `#library-prompt-row-5`, no "Loading prompts"
in any longrepr. The mode is gone.

Obtaining that verdict needed a workaround worth recording: no `dev` Tests
run can complete (TASK-19600 -- `cancel-in-progress` plus a 20-40 minute
merge cadence against an ~80 minute run; 25 of 40 recent runs cancelled,
zero finished). A manual `gh workflow run Tests --ref dev` forms its own
concurrency group and therefore survives; that is how this run finished,
and it is the first complete `dev` verdict in weeks.

**Remaining canvas red is a DIFFERENT failure and is deliberately not
absorbed here.** The same run shows 54 canvas failures signed
`AssertionError: Initial Prompt failure never reached the mounted pager`
(50), `NoMatches` on `#library-row-browse-prompts` (2), `LibraryHarness has
no attribute 'app_config'` (1), and `NoActiveAppError` (1). None appear in
the 2026-08-20 baseline artifacts; all trace to PR #1893's
workspace-registry work; and **TASK-19602** (In Progress) already owns them
with the same signatures.
