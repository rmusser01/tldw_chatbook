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
- [ ] #1 The trio passes on a clean dev checkout (bisect 35bb1aa98 first).
- [ ] #2 The CI-only stuck-loading mode is reproduced or instrumented (e.g. capture the load worker's state on timeout) and fixed.
<!-- AC:END -->
