---
id: TASK-38759
title: "History participant lock-order test is red on dev and the vLLM repair-path test depends on the tmp path shape"
status: Done
assignee: ['@claude']
created_date: 2026-11-05 07:11
updated_date: 2026-11-05 23:46
labels:
- follow-up
- tests
- backup-recovery
priority: medium
references:
- Tests/Backup_Recovery/test_async_file_participant_lifetimes.py
- Tests/LLM_Management/test_vllm_profiles.py
- tldw_chatbook/UI/LLM_Management/vllm_setup.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

Two nodes are red on dev, and neither file runs on a PR lane.

1. **`Tests/Backup_Recovery/test_async_file_participant_lifetimes.py::test_history_participant_validation_does_not_invert_config_storage_locks`** (:677). While holding the config file lock, it starts a thread that closes the history participant's admission. It then waits up to 2 s for `raw._participant_state` to be called for that participant. The call never comes, and `assert validating.wait(2)` fails at :710. It failed both times it ran: once under `-n 4` and once serially. Either the close no longer validates through `_participant_state`, or it blocks before it gets there; which one is not established. The test exists to catch a lock inversion between config and storage, and while it is red, nothing checks for one.

2. **`Tests/LLM_Management/test_vllm_profiles.py::test_profile_accepts_nonexistent_safe_local_directory_for_repair`** (:438). It saves a `LOCAL_DIRECTORY` profile for a path under `tmp_path`, and fails with `VllmProfileValidationError: model_value is invalid for model_source`. The cause is the path, not the code. `is_safe_local_model_path_shape` (`UI/LLM_Management/vllm_setup.py:384`) rejects any path component that starts with "-". The scratch-profile wrapper's basetemp sits under `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/...`. A probe returned False for that path and True for `/tmp/pytest-of-runner/...`. So the node is red in every local run through the wrapper, and green wherever `tmp_path` has no such component.

#3402 changes both files, but only to scope their `json.loads` stubs (TASK-38685); neither node changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The lock-order test reaches its validation point and passes; or the notes show the close path changed on purpose, and the test is re-anchored to the new validation point and still goes red under a temporary edit that inverts the locks
- [x] #2 The vLLM repair-path test passes when `tmp_path` contains a component that starts with "-", and still fails if the shape check rejects a safe absolute path
- [x] #3 Both files pass whole under `-n 4` through the scratch-profile wrapper
- [x] #4 Both files run on a PR lane, and their measured times in that lane are recorded in the notes
<!-- AC:END -->

## Implementation Plan

1. Trace `close_admission`'s history to see whether it still validates through `_participant_state`; if the change was deliberate, re-anchor the lock-order test's wait to the call it makes now.
2. Show the re-anchored test red under a temporary edit that makes the close take the config lock while holding storage, then green.
3. Give the vLLM repair-path test a safe absolute path that does not depend on `tmp_path`'s shape; show it red under a temporary shape check that rejects safe absolute paths.
4. Run both files whole under `-n 4`, add them to a PR lane step chosen from measured dev job durations, time that step with its exact command, and update the `Tests/CI` lane pins.

## Implementation Notes

**AC#1 (lock order).** The close path changed on purpose. 3e618d7119 (2026-10-15) changed `_RawParticipant.close_admission` from `_participant_state(self).closed = True` to `_participant_identity(self)`, which checks installed identity mappings only and never reads source metadata. b3874e98bd renamed it `participant_identity`. The close therefore never reaches `_participant_state`, and the test's wait could never fire. The test is re-anchored to `raw.participant_identity`, which the close calls under `storage._changed`, and it passes. Red control: a temporary edit made `close_admission` take `config._config_file_lock()` while holding the storage lock. The test then failed with `history holds storage while waiting on the config owner`.

**AC#2 (vLLM repair path).** The test now builds its absent model directory under `tmp_path`'s anchor (`/tldw-vllm-repair-<uuid>/models`), not under `tmp_path`. It passes under the scratch wrapper, whose basetemp contains the `-Users-macbook-dev-...` component that failed at base with `model_value is invalid for model_source`. Red control: a temporary `return False` for every absolute path in `is_safe_local_model_path_shape` fails the test with the same error.

**AC#3.** Both files whole under `-n 4`: 3 of 3 runs, 121 passed (7.4, 7.9 and 7.7 s; load average 10-12).

**AC#4.** Lane: dreams-boundary's sandboxed step "Run Dreams and Guardian boundary contracts". This uses the same reading of 25 dev PR runs as TASK-38758: job median 6.2 / max 7.2 min of its 20. The vLLM file is appended at the end. The lifetimes file goes FIRST. At the end of the step, 3 of its nodes (`test_history_pause_does_not_change_cache_or_create_parent`, `test_installed_history_running_cancel_holds_native_until_bookkeeping`, `test_startup_native_exclusion_and_pending_cover_event_loop_bookkeeping`) failed in `pause.drain(...)`, because a local pause drains every live lease in the process. A per-test probe of `storage._live_leases` found five Guardian nodes earlier in the step that leave one lease open each: `test_live_discovery_consent.py::test_revocation_during_cycle_wait_prevents_next_external_call[provider/search/story]`, `test_daily_native_lifecycle.py::test_daily_file_inbox_contention_keeps_ui_live_and_retires_only_new_handles[True]` and `test_crisis_resources.py::test_check_pipeline_uses_the_extracted_module`. They are listed as a follow-up, not fixed here. Measured with the step's exact command in its new order (22 targets, `--timeout=180 --tb=short`, serial, through the scratch wrapper, load 10-17): 398 passed in 133.6 s. The lifetimes file took 10.0 s (26 tests) and the vLLM file 2.1 s (95 tests). Pins updated in `DREAMS_BOUNDARY_TARGETS` and `INHERITED`; Tests/CI: 1199 passed, 1 skipped. Not verified on Linux.

**Files.** `Tests/Backup_Recovery/test_async_file_participant_lifetimes.py`, `Tests/LLM_Management/test_vllm_profiles.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_response_rules_critical_gate.py`, `backlog/docs/lessons-testing-evidence.md` (the shared-step lease trap).
