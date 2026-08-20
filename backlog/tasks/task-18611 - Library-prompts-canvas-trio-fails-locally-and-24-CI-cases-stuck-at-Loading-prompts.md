---
id: TASK-18611
title: >-
  Library prompts canvas: trio fails on clean dev and 24 CI cases stuck at
  "Loading prompts"
status: In Progress
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

## Implementation Notes

**AC #1 (clean-dev trio) — fixed.** Bisect landed on `0dfadf463`
(perf(library): reconcile snapshots below screen), not 35bb1aa98: its
recomposes can detach a cached `#library-prompts-delete-undo` widget
reference taken before the cancelled/settled import, so the retry press
silently no-ops and the restore never runs. Fix is test-side: query the
live mounted button at press time in both `import_blocks` and
`cancelled_import` tests, plus a hardened settlement wait (drain the
cancelled import's worker, not just `import_finished`). Verified: full
`test_library_prompts_canvas.py` run green (310 passed) on a head rebased
onto dev@25500ad87.

**AC #2 (CI-only stuck-at-Loading 24) — superseded, do not duplicate.**
Open PR #1838 (task-18912, "restore latest-dev suite health") hardens
`_wait_for_prompt_browse_scope` (canvas-settlement condition + real
deadline) and reworks `_open_prompts_list` in this same test file, which
targets exactly this mode; it also covers the audio_cpp/TTS stragglers
from TASK-18610. Re-check AC #2 against CI only after #1838 merges; close
it there if the sharded UI runs no longer show the stuck-loading cluster.
