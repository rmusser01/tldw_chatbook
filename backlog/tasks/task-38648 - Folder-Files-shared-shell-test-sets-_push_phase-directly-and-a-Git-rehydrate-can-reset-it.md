---
id: TASK-38648
title: Folder Files shared-shell test sets _push_phase directly and a Git rehydrate can reset it
status: To Do
created_date: 2026-11-03 05:51
labels:
- follow-up
- tests
- library
priority: low
references:
- Tests/UI/test_library_file_notes_workspace.py
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
dependencies:
- TASK-38646
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-38646 (2026-11-03). `Tests/UI/test_library_file_notes_workspace.py::test_folder_files_shared_shell_retains_state_across_breakpoints` failed once with `assert 'idle' == 'needs_attention'` on `_push_phase`. That is about 1 failure in 29 loaded runs:
- two whole-file runs with `-n 4`: 1 failure;
- 16 runs on its own, 8 three-test runs and 3 more whole-file runs: none.

**Likely mechanism, not proven.** The test writes `workspace._push_phase = "needs_attention"` directly (:680 on dev 18482c1cf7). Product code sets the phase back to `"idle"` when the Git presentation rehydrates (`_rehydrate_push_state`, library_file_notes_workspace.py:4691). If a rehydrate lands after the write, the shell no longer holds the state the test set. The same direct write appears at :1105, :2892 and :2912.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is shown: the failure is reproduced by forcing the suspected order (for example a delayed rehydrate under a temporary source edit), or the mechanism is ruled out with evidence and the real one named
- [ ] #2 The test cannot lose the state it sets: it reaches it through a path a rehydrate respects, or waits for the rehydrate to settle before setting it; the forced order from #1 then passes
- [ ] #3 If the cause is a product bug (a real push-attention state lost on rehydrate), the product is fixed and a test pins it instead
- [ ] #4 The test passes 20 three-test runs with `-n 4`
<!-- AC:END -->
