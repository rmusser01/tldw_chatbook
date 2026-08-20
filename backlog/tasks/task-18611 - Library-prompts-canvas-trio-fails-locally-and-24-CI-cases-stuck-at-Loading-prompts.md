---
id: TASK-18611
title: >-
  Library prompts canvas: trio fails on clean dev and 24 CI cases stuck at
  "Loading prompts"
status: To Do
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
- [ ] #2 The CI-only stuck-loading mode is reproduced or instrumented (e.g. capture the load worker's state on timeout) and fixed.
<!-- AC:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
**2026-08-20 status sweep (evidence-based):**

- **AC#1 is resolved on dev without a patch.** On clean `origin/dev` tip
  `25500ad87`, the two deterministic repro cases pass (2/2) and the full
  `Tests/UI/test_library_prompts_canvas.py` runs green: **310 passed in
  295s** (macOS, venv pytest). The trio no longer reproduces; dev's
  prompt-pager lifecycle work (`f656def50`..`989f81da3`) evolved past it.
- **The local branch `fix/task-18611-library-canvas` (worktree
  `/private/tmp/pass3`, commit `4e7966105`) is SUPERSEDED — do not PR it.**
  Its root cause (cached `#library-prompts-delete-undo` reference detaching
  across import-triggered recomposes) was independently found and fixed
  more robustly by PR #1838 (wait-for-recompose before the retry press,
  live-projection guards). Verified green on its own base (310 passed) but
  obsolete against current dev.
- **AC#2 (CI-only 24-case "stuck at Loading prompts") is owned by open
  PR #1838** (`codex/task-18912-dev-test-health`): deadline-based waits for
  a single live `#library-prompts-canvas` projection plus the
  `_open_prompts_list` helper target exactly this mode, with a fully
  accounted 47k-node baseline. Close this task when #1838 merges and the
  ubuntu shards run green on the canvas file; reopen AC#2 here only if the
  24-case mode survives that merge.
- Historical: a ZCode-session bisect pointed at `93fa11acc`; that result
  conflicts with the verified root cause above and was likely noise from
  the order-dependent rotating-third failure. Recorded so nobody re-runs
  that bisect.
<!-- SECTION:NOTES:END -->
