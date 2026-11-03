---
id: TASK-38686
title: "Prompts DB legacy concurrency test expects a conflict that update_prompt_by_id does not raise"
status: To Do
created_date: 2026-11-03 11:40
labels:
- follow-up
- tests
- database
priority: low
references:
- Tests/Prompts_DB/test_prompts_db_legacy.py
- tldw_chatbook/DB/Prompts_DB.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down, where it was reported failing 6 of 6 runs alone. Checked on dev 9bbd5c2515, 2026-11-03.

`Tests/Prompts_DB/test_prompts_db_legacy.py::TestConcurrencyAndDataIntegrity::test_concurrent_updates_to_same_prompt` failed 5 of 5 runs alone, each its own pytest process through the scratch-profile wrapper (about 1 s each), with `AssertionError: 'conflict' not found in dict_values(['success', 'success'])`.

**Why it fails.** The test starts two threads behind a barrier. Each calls `update_prompt_by_id(pid, {"details": ...})` without `expected_version`, and the test asserts that one of them raises a conflict and that the final version is 2. `update_prompt_by_id` (Prompts_DB.py:1818) compares versions only when the caller passes `expected_version` (:1890). Without it, each update reads the current version inside its own transaction (:1870-1887), so two serialized updates both succeed.

**No lane runs it.** The only Tests/Prompts_DB target on a PR lane is `test_prompt_export_mixins.py` (derived-artifacts.yml:376). The core-tests job in test.yml runs on pushes to main only, and Nightly Deep last ran on 2026-09-27.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Two concurrent updates of one prompt that pass the same expected_version produce exactly one success and one conflict, and the rewritten test pins that, passing 10 of 10 runs alone on dev
- [ ] #2 The test fails when update_prompt_by_id's version comparison is removed (shown under a temporary source edit)
- [ ] #3 The test runs on a PR lane, or the task notes record why it does not
<!-- AC:END -->
