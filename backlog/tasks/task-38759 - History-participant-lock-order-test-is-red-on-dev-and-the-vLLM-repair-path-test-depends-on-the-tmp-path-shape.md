---
id: TASK-38759
title: "History participant lock-order test is red on dev and the vLLM repair-path test depends on the tmp path shape"
status: To Do
assignee: []
created_date: 2026-11-05 07:11
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
- [ ] #1 The lock-order test reaches its validation point and passes; or the notes show the close path changed on purpose, and the test is re-anchored to the new validation point and still goes red under a temporary edit that inverts the locks
- [ ] #2 The vLLM repair-path test passes when `tmp_path` contains a component that starts with "-", and still fails if the shape check rejects a safe absolute path
- [ ] #3 Both files pass whole under `-n 4` through the scratch-profile wrapper
- [ ] #4 Both files run on a PR lane, and their measured times in that lane are recorded in the notes
<!-- AC:END -->
