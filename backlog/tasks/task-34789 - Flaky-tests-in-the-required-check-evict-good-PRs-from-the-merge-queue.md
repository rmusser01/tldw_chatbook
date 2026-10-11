---
id: TASK-34789
title: Flaky tests in the required check evict good PRs from the merge queue
status: To Do
created_date: 2026-10-11 01:44
labels:
- ci
- tests
- flaky
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On 2026-10-10 the merge queue served real PRs for a day. 7 of about 11 heads it rebased failed the required check once, each on a different test, and #3059 and #2882 were evicted as failed twice on flakes alone. The queue's fail-twice rule assumes a second failure means a real defect; at this flake rate it removes good PRs. Tests seen failing then passing on a re-run: Tests/UI/test_study_flashcards_real_service_contract.py::test_real_service_create_deck_select_deck_and_add_card_in_local_mode (4 runs), Tests/Notes/test_notes_sync_source_moved_settle.py::test_a_save_in_one_folder_never_waits_for_another_folders_pass (3 of 4 runs on #3039), Tests/UI/test_library_reader_console_delete_refresh.py::test_console_delete_updates_the_open_library_reader, Tests/UI/test_console_pending_interrupt_projection.py::test_session_owned_pending_projection_journeys, Tests/UI/test_console_runtime_ownership.py::test_opening_console_during_a_headless_delivery_arms_the_poll, Tests/UI/test_console_settings_hidden_fields.py::test_applying_on_anthropic_submits_none_of_the_hidden_fields, Tests/UI/test_library_rail_switch_keeps_reader_and_note.py::test_new_note_with_saved_content_survives_a_rail_round_trip. Outside the required check, the Perf Guard module census (1034 against a limit of 1033) and Tests/CI/test_task598_external_parakeet_evidence.py::test_supervisor_terminates_a_descendant_in_a_new_session also flaked once each.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed test either passes 20 consecutive CI runs or is fixed, quarantined or removed from the required lanes with a recorded reason
- [ ] #2 The required check's first-attempt failure rate on unchanged code is measured over at least 20 runs and recorded
- [ ] #3 The owner decides whether the queue's fail-twice eviction stays as it is at the measured rate
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
