---
id: TASK-38648
title: Folder Files shared-shell test sets _push_phase directly and a Git rehydrate can reset it
status: Done
assignee:
- '@claude'
created_date: 2026-11-03 05:51
updated_date: 2026-11-03 07:16
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
- [x] #1 The cause is shown: the failure is reproduced by forcing the suspected order (for example a delayed rehydrate under a temporary source edit), or the mechanism is ruled out with evidence and the real one named
- [x] #2 The test cannot lose the state it sets: it reaches it through a path a rehydrate respects, or waits for the rehydrate to settle before setting it; the forced order from #1 then passes
- [x] #3 If the cause is a product bug (a real push-attention state lost on rehydrate), the product is fixed and a test pins it instead
- [x] #4 The test passes 20 three-test runs with `-n 4`
<!-- AC:END -->

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: Test synchronization fix unless the evidence shows a product bug (AC#3).

1. Instrument the test temporarily: log every `_push_phase` write with its caller, each `_refresh_session_changes` call with the owner revision it sees, each poll tick, and the test's own milestones, relative to the workspace mount.
2. Use that log to name what can reset the phase after the test's write, and show whether it is test-only or a real push-attention state the product loses.
3. AC#1: force that order under a temporary edit and show the test fail with `'idle' == 'needs_attention'`.
4. AC#2: change the test so it cannot lose the state it sets, by the smallest path the evidence supports; show the forced order from step 3 passes.
5. AC#4: 20 three-test runs with `-n 4`, each its own pytest process.

## Implementation Notes

The suspected mechanism is confirmed, and the cause is in the test only. The fix is one line in the test: it runs the refresh the poll would run before it sets any state.

**Mechanism (AC#1).** A temporary probe logged each `_push_phase` write with its caller, each `_refresh_session_changes` call with the owner revision, each poll tick, and the test's own steps. Times are relative to the workspace mount.
- `open_path` holds a path transition (`try_acquire_transition`), which moved `session_revision()` from (1, 1) to (1, 2). Nothing re-read it: the observed revision stayed (1, 1).
- The first poll tick comes at mount + 10 s (`poll_interval=10`). It runs `refresh_files`, then `_render_projection`, then `_refresh_session_changes`. That call saw (1, 2) against (1, 1) and called `_rehydrate_push_state`.
- The owner has no retained push and no `push_recovery`, so the rehydrate set the phase to `"idle"` (:4734).
- Unforced, at load average 34, the resize loop ended 9.3 s after mount, just inside the tick. In one later run at load 37, the real tick landed inside the loop at 10.0 s.

Two forced orders on the unfixed test each failed 2 of 2 runs with `assert 'idle' == 'needs_attention'`:
- `await workspace.refresh_files()` (the poll's body) right after the write;
- `asyncio.sleep(10)` right after the write, so the real 10 s tick lands before the loop.

**Fix (AC#2).** Right after `open_path`, the test now runs `assert await workspace.refresh_files()`, the same body the poll tick runs. The owed rehydrate therefore lands before the test sets its state. Every later refresh then sees an unchanged revision and returns early. With the fix, the probe showed the observed revision equal to the owner's, (1, 2), at the write and at every width. Both forced orders passed 2 of 2 runs, and in one of those the real tick also landed inside the loop. Product code is unchanged.

**Not a product bug (AC#3).** The rehydrate derives the phase from owner state: `"needs_attention"` when the owner holds a push recovery, or the settled result of a retained push it still holds. The test's phase had neither behind it. It was a bare field write, so dropping it is the designed outcome. One product path does set `"needs_attention"` with nothing in the owner behind it: `_publish_push_review_admission_failure` while the panel shows its list. By reading the code, a later rehydrate would drop that label too. That is not this failure's cause, no test here exercised it, and it is left as a proposed follow-up.

**AC#4.** 20 three-test runs with `-n 4`, each its own pytest process. The three tests were this one, `test_conflict_resolution_discard_keeps_cancel_first_confirmation` and `test_folder_files_capability_inventory_and_modes`. This test passed in all 20, and so did the conflict-discard test. Load averages were 32 to 45. 9 of the 20 runs failed only `test_folder_files_capability_inventory_and_modes`, at the Restore assertion (`restore.disabled`). That is TASK-38646's race, which PR #3356 fixes and which is not on dev yet.

**Other direct writes.** These are the three that the task description named: `test_notes_authority_round_trip_retains_both_workspaces`, `test_folder_files_authority_row_tracks_root_save_and_session_git` and `test_folder_files_authority_status_survives_in_surface_navigation`. A one-run probe logged no owed refresh at any of their writes. In two of them, the observed revision equalled the owner's. The row test has no bound session revision to compare: (0, 0), observed `None`. None of them was changed. The file has four more `_push_phase =` writes (:3128, :3843, :3891 and :7077 at 88ceaa72c0), which this task did not probe. `test_notes_authority_round_trip_retains_both_workspaces` is red at base 18482c1cf7 for an unrelated reason: `#library-notes-task-return` is not displayed at :1155.

**Checks.** Preflight passes. Ruff format is clean, and the Ruff lint count is unchanged from base (24). A lesson entry, "A 10 s `poll_interval` does not keep the poll out of a loaded UI test", was added to `backlog/docs/lessons-testing-evidence.md`.

Files: `Tests/UI/test_library_file_notes_workspace.py`, `backlog/docs/lessons-testing-evidence.md`.
