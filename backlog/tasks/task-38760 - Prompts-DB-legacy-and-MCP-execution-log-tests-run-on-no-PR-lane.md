---
id: TASK-38760
title: "Prompts DB legacy and MCP execution-log tests run on no PR lane"
status: To Do
assignee: []
created_date: 2026-11-05 07:11
labels:
- follow-up
- ci
- tests
priority: low
references:
- Tests/Prompts_DB/test_prompts_db_legacy.py
- Tests/MCP/test_execution_log.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-38685
- TASK-38686
- TASK-38716
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

Two small files run on no PR lane. Each has reds on dev that PR #3402 fixes:
- **`Tests/Prompts_DB/test_prompts_db_legacy.py`**: 70 tests, about 4-5 s serially on this base. One is red on dev: `TestConcurrencyAndDataIntegrity::test_concurrent_updates_to_same_prompt`. It failed 3 of 3 serial runs, and passed in one `-n 4` run. TASK-38686 owns it, and #3402 fixes it. With #3402 the file has 72 tests, all green, in about 5 s.
- **`Tests/MCP/test_execution_log.py`**: 26 tests, about 9-10 s serially. Two are red on dev: `test_steady_state_appends_do_not_reparse_the_whole_log` and `test_appends_after_a_rotation_do_not_reparse_either_generation` ("appends re-parsed the log 115 times"). Their parse counter patches `json.loads` for the whole process. TASK-38685 scopes it in #3402, and the file then gives 26 passed.

Neither file is a workflow target or a UI census entry. The only `Tests/Prompts_DB` target is `test_prompt_export_mixins.py`. The `Tests/MCP` targets are four named files in `pr-fast-lane` and `test_server_manifest_parse_reuse.py`.

`pr-fast-lane` has no room: TASK-38716 measured it at its 35-min cap. `dreams-boundary` has room. Its cap is 20 min, and TASK-34986's replay put it at median 5.2 / max 6.7 min. TASK-36532 already moved a non-Dreams file there (`Tests/Chat/test_local_server_discovery.py`) for the same reason.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Once #3402 is on dev, both files run on a PR lane other than `pr-fast-lane`, and the step's comment says why they are there
- [ ] #2 Each file passes 3 of 3 runs in that lane's configuration before it is added, and its measured time in that lane is recorded in the notes
- [ ] #3 The job they join still ends at least 5 minutes under its cap in its slowest measured run
- [ ] #4 The lane contract tests under `Tests/CI` reflect the new targets
<!-- AC:END -->
