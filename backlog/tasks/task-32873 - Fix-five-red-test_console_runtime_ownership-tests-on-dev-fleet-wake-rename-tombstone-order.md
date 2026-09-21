---
id: TASK-32873
title: >-
  Fix five red test_console_runtime_ownership tests on dev (fleet-wake rename +
  tombstone order)
status: Done
assignee:
  - '@robert'
created_date: '2026-09-20 16:42'
updated_date: '2026-09-20 22:19'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Five tests in Tests/UI/test_console_runtime_ownership.py fail on pure origin/dev, verified 2026-09-12 during PR #2643's review: test_runtime_owned_custody_tracks_only_lifetime_handles, test_runtime_tombstones_before_shutdown_and_disposes_via_to_thread (expects to-thread before controller-shutdown; actual order differs), test_persistent_attach_sync_failure_has_bounded_backoff_and_resume_retry, test_reconciled_view_keeps_each_live_poll_reason_and_one_timer[wake], test_opening_console_during_a_headless_delivery_arms_the_poll (ConsoleFleetWakeCoordinator.delivering_session_id renamed to delivering_session_ids but the tests still use the singular). Reproduced on a clean origin/dev worktree. Not caused by PR #2643. Also: Tests/Chat/test_console_skill_script_confirm.py::test_confirm_payload_carries_timeout_and_request_id flakes under combined-run load, passes isolated.
<!-- SECTION:DESCRIPTION:END -->

### Additional dev-reds found 2026-09-13 (verified on clean origin/dev @ d1a0649cd2)

- Tests/Chat/test_console_chat_create_integration.py::test_chat_create_callbacks_reach_bridge_when_ui_sinks_wired and ::test_chat_create_callbacks_absent_without_ui_sinks — both fail with `RecoveryRequired: raw_source_selection_changed` from the Backup_Recovery bootstrap (PR #2642 chain): the suites' fixture profiles now trip a recovery gate.
- Tests/Chat/test_console_chat_store.py::test_hidden_db_legacy_cas_miss_with_unavailable_reader_marks_live_stale — same era, fails on clean dev.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All five named tests pass on dev,delivering_session_id references updated to delivering_session_ids where the coordinator renamed them,tombstone ordering expectation matches actual shutdown order or the order is fixed in code,Load-order flake root-caused or stabilized
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All five AC reds fixed with root causes; suite added to CI (separate Fast Lane invocation); machine enrollment wedge remediated (backed up first); one rotted mount parked as strict xfail; flake stabilized.
<!-- SECTION:NOTES:END -->
