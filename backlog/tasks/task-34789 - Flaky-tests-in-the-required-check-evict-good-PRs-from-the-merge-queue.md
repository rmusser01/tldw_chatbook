---
id: TASK-34789
title: Flaky tests in the required check evict good PRs from the merge queue
status: In Progress
created_date: 2026-10-11 01:44
labels:
- ci
- tests
- flaky
priority: high
assignee:
- '@claude'
updated_date: 2026-10-11 18:26
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On 2026-10-10 the merge queue served real PRs for a day. 7 of about 11 heads it rebased failed the required check once, each on a different test, and #3059 and #2882 were evicted as failed twice on flakes alone. The queue's fail-twice rule assumes a second failure means a real defect; at this flake rate it removes good PRs. Tests seen failing then passing on a re-run: Tests/UI/test_study_flashcards_real_service_contract.py::test_real_service_create_deck_select_deck_and_add_card_in_local_mode (4 runs), Tests/Notes/test_notes_sync_source_moved_settle.py::test_a_save_in_one_folder_never_waits_for_another_folders_pass (3 of 4 runs on #3039), Tests/UI/test_library_reader_console_delete_refresh.py::test_console_delete_updates_the_open_library_reader, Tests/UI/test_console_pending_interrupt_projection.py::test_session_owned_pending_projection_journeys, Tests/UI/test_console_runtime_ownership.py::test_opening_console_during_a_headless_delivery_arms_the_poll, Tests/UI/test_console_settings_hidden_fields.py::test_applying_on_anthropic_submits_none_of_the_hidden_fields, Tests/UI/test_library_rail_switch_keeps_reader_and_note.py::test_new_note_with_saved_content_survives_a_rail_round_trip. Outside the required check, the Perf Guard module census (1034 against a limit of 1033) and Tests/CI/test_task598_external_parakeet_evidence.py::test_supervisor_terminates_a_descendant_in_a_new_session also flaked once each.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed test either passes 20 consecutive CI runs or is fixed, quarantined or removed from the required lanes with a recorded reason
- [x] #2 The required check's first-attempt failure rate on unchanged code is measured over at least 20 runs and recorded
- [ ] #3 The owner decides whether the queue's fail-twice eviction stays as it is at the measured rate
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the first-attempt failure rate and name the flaky tests from CI logs
2. Root-cause each flaky test; fix the product or the test, one PR per cause
3. Bring the owner the fail-twice decision with the measured rate
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Measurement (AC 2), from every pull_request run of the required workflow between 2026-10-08 and 2026-10-11, attempts read per run. One branch was genuinely broken (codex/console-uat-performance-fixes: 39 failed attempts on the same 9 Console tests, never green on a re-run) and is excluded. Of the other 104 runs, 48 (46%) failed their first real attempt, and 33 (32% of all runs) failed only on nine flaky tests. About 10% of good PRs would fail twice in a row and be evicted. These are PR runs, so 'unchanged code' is approximated by counting a run as flake-only when every failing test is one of the nine, six of which were also proven by a same-commit re-run going green.

The nine, with failed attempts through 2026-10-11 and their state:
- Tests/Notes/test_notes_sync_source_moved_settle.py::test_a_save_in_one_folder_never_waits_for_another_folders_pass (14): product race, fixed in this task's first PR (below).
- Tests/UI/test_study_flashcards_real_service_contract.py::test_real_service_create_deck_select_deck_and_add_card_in_local_mode (9): already fixed on dev by 4fe980f0a5 (test waits for the empty card list); no failure since it reached dev on 2026-10-11.
- Tests/UI/test_console_settings_hidden_fields.py::test_applying_on_anthropic_submits_none_of_the_hidden_fields (6): under investigation.
- Tests/UI/test_console_turn_resend_ui.py::test_console_poll_outlives_a_turn_the_controller_has_not_started (5): under investigation.
- Tests/UI/test_console_pending_interrupt_projection.py::test_session_owned_pending_projection_journeys (4): not started.
- Tests/UI/test_library_reader_console_delete_refresh.py::test_console_delete_updates_the_open_library_reader (4): under investigation.
- Tests/UI/test_library_rail_switch_keeps_reader_and_note.py::test_new_note_with_saved_content_survives_a_rail_round_trip (4): under investigation; the two latest failures were app crashes while the note editor remounts.
- Tests/UI/test_library_notes_sync_delete_restore.py::test_delete_holds_the_folder_and_undo_returns_it_to_up_to_date (2): under investigation.
- Tests/UI/test_console_runtime_ownership.py::test_opening_console_during_a_headless_delivery_arms_the_poll (1): not started.

Notes sync fix (first PR). The failure was never a slow test. A guarded replace unlinks the file it displaced, and ext4 hands that inode number to the next new file in the same allocation group; the replaced file's binding keeps recording the number until its own commit. A second folder writing in that window got a file carrying the number, and whenever it committed first the store's one-identity-per-active-binding index refused it: update_file, needs_attention, executor_failed. APFS never reuses a number, so it never failed on macOS. The fix: the one place a sync-written file gets its inode number (the temporary file PinnedSyncRoot creates for a replace or a create) asks the store whether an active binding records that number, and takes another candidate if so. A refused candidate is held as an open, already-unlinked descriptor (at most 64), so a directory walk, the watcher and a kill see at most two candidate names (the most recent refused one and the one in hand). The check is advisory: if the store cannot answer, the candidate cannot be examined or replaced, or 64 have been refused, the write goes ahead as it did before and the commit decides. The CI logs never showed the cause because every untyped exception is recorded as executor_failed with no trace; the cause was first shown on a model of ext4 numbering, and the PR adds tests on real inode numbers that pass on a filesystem that recycles the number and skip elsewhere.

Found on the way and filed: TASK-34794 (Recovery never re-runs a write that failed before it landed) and TASK-34795 (a stale identity claim from outside sync still fails a binding commit, with no named reason).
Observed on Linux, 2026-10-11: on PR #3077's run 38161635121 (PR Fast Lane, image ubuntu-24.04) all 17 tests of Tests/Notes/test_notes_sync_source_moved_settle.py passed with no skip. The two real-inode arms skip when the filesystem does not hand the second folder the recycled number, so a pass means the collision happened on the runner: the unfixed path ended in the row CI used to show and the fixed path refused the number. That test forces the order of events, so it proves the number is reused and the unfixed commit is refused; that the original test reaches the same order unforced rests on two independent models of the numbering (16 and 11 failures in two sets of 120 runs of the unfixed code; 12 in 220 in the independent review) until the original test accumulates clean CI runs. A later run (38182902782, of the redesigned search) passed both arms again with the control arm asserting the IntegrityError itself, and passed Tests/Notes/test_sync_containment.py with no skip, including its real-inode test. On macOS both arms skipped 50 of 50. Tests/Notes/test_sync_containment.py ran in no PR lane; this PR adds it to the PR fast lane so its real-inode test runs on Linux before a merge. Lesson recorded in backlog/docs/lessons-testing-evidence.md.

Also flaky under local load, not in any PR lane and not in the CI failure logs: Tests/Notes/test_notes_sync_observation_reuse.py::test_concurrent_note_writer_yields_only_committed_pairs (its writer thread records a version as committed after the database commit, so an observer can see a version not yet recorded; a test bug, not fixed here).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
