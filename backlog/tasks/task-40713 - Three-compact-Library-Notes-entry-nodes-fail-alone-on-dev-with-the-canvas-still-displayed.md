---
id: TASK-40713
title: "Three compact Library Notes entry nodes fail alone on dev with the canvas still displayed"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-13 06:07'
created_date: 2026-11-13 05:15
labels:
- follow-up
- library
- tests
priority: medium
references:
- Tests/UI/test_library_shell.py
- scripts/library_shell_push_deselect.txt
- .github/workflows/derived-artifacts.yml
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down, from the TASK-39152 stream. Checked on dev 26b2e677dd, 2026-11-12.

Three TASK-39153 watch items in `Tests/UI/test_library_shell.py` now fail when run alone, 3 of 3 runs at load 59-70. Each fails with `assert True is False` on `#library-canvas`'s `display` after entering compact Notes. In round 13 each passed alone at least once.
- `test_library_note_wide_deep_link_back_clears_future_compact_intent` (~:32527)
- `test_library_note_unsafe_session_outranks_rail_focus_on_compact_entry` (~:33366)
- `test_library_note_load_failure_outranks_rail_focus_on_compact_entry` (~:33421)

None of them is in `scripts/library_shell_push_deselect.txt`. So if they are red, the `library-shell-dev` push job is red on every dev push, and real regressions in that job go unseen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes say, with evidence, whether each node is red from a product defect or from a test race, each red shown first with the same command
- [x] #2 A product defect is fixed and its node passes; a test race is fixed by waiting on a condition; each node passes 5 of 5 alone and in its file at -n 4
- [x] #3 If a fix is out of scope, the node is deselected in the push job with its owning task named, and the push job's pins still pass
<!-- AC:END -->

## Implementation Plan

1. Run the three nodes alone on base 26b2e677dd (run.sh, one process per node) and read each failure.
2. Trace what keeps `#library-canvas` displayed on a compact Notes entry: probe the compact stage, the canvas/work-pane display and the timing (does it settle later?), and `git log -S` the code that decides it.
3. Classify: a test race gets a wait on the real end condition; a product defect is fixed if small, or, if it is one of TASK-40293..40299's defects, deselected in `scripts/library_shell_push_deselect.txt` with that owner named.
4. Show each red first, then green 5 of 5 alone and in the file at `-n 4`; run the push job's pins (`Tests/CI/test_derived_artifacts_workflow.py`).

## Implementation Notes

**All three are one test race, not a product defect, and none is one of TASK-40293..40299's defects.** Each node waited on `_notes_state.compact`, then read `#library-canvas`'s `display` at once.
- A wide -> compact resize sets `compact` in `_transition_library_notes_presentation` straight away.
- The canvas is hidden later, by `AdaptivePaneShell.sync_layout` (inline `display: none` plus `disabled`), and only once the Notes reader has resolved the settled compact allocation (`_sync_library_notes_reader_layout_from_shell` returns early on a transient one).
- A probe (temporary, reverted) showed the gap: the canvas was still displayed (`styles.display` `block`) when `compact` flipped, was hidden 0.15 s later with no further input, and stayed hidden for the next 4 s.
- Siblings that start at 60x20 never cross the boundary, which is why they pass.

**Red first.** On base 26b2e677dd, each node alone through run.sh, one process per run, at load 59-70: 3 of 3 red each (`:32527`, `:33366`, `:33421`, `assert True is False`). At load ~29 the same command passed 3 of 3: it is load-dependent.
- Forced red: a temporary edit made `sync_layout` hide `#library-canvas` late (1 s for the deep-link node; 3 s for the other two, whose hide was already armed before the compact flip). The old tests then failed at the same three lines.

**Fix.** A new helper, `_wait_for_library_canvas_hidden_after_compact_resize`, waits on the real end state (the canvas hidden) after the compact wait, through `_wait_for_condition` (15 s deadline). The nodes' assertions are unchanged.
- Green under the same forced late hide.
- Negative control: `sync_layout` never hiding the canvas (temporary, reverted) fails all three with "The compact Notes shell never hid #library-canvas.", so the wait cannot hide a real regression.

**AC#2.** Each node passed 5 of 5 alone (run.sh, one process per run, load 35-45). In `Tests/UI/test_library_shell.py` whole at `-n 4` (load 37-44, 33 min), all three passed: 17 failed / 831 passed.
- 12 of those 17 failures are the nodes the push job already deselects.
- The other five each passed alone at once (load ~36). They are TASK-39153-style load-flaky nodes, not this task's: `test_library_graduation_late_restore_does_not_steal_newer_user_focus`, `test_conversation_canvas_scrolls_current_page_while_pager_stays_fixed`, `test_browse_row_error_callout_keeps_the_right_tint_across_a_retry`, `test_library_rag_scope_recovery_steady_state_snapshot_causes_no_churn` and `test_start_click_immediately_after_typing_submits`.

**AC#3.** No deselect was needed: the fix is in scope and test-only. The push job's pins pass: `Tests/CI/test_derived_artifacts_workflow.py` 70 passed, including the three `library_shell` pins.

**Files.** `Tests/UI/test_library_shell.py`.
