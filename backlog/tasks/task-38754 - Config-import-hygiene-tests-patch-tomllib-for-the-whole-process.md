---
id: TASK-38754
title: "Config import hygiene tests patch tomllib for the whole process"
status: To Do
assignee: []
created_date: 2026-11-05 07:11
labels:
- follow-up
- tests
priority: low
references:
- Tests/Utils/test_config_import_hygiene.py
- tldw_chatbook/config.py
dependencies:
- TASK-38685
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

`Tests/Utils/test_config_import_hygiene.py` counts TOML parses by patching the stdlib `tomllib` module itself:
- :213, `monkeypatch.setattr(config_mod.tomllib, "loads", _counting_loads)`, in `test_forced_full_reload_does_not_reparse_embedded_default`;
- :265, `monkeypatch.setattr(config_mod.tomllib, "load", _counting_load)`, in `test_user_config_file_parsed_once_across_startup_sequence`.

`config_mod.tomllib` is the stdlib module. For the length of each test, every `tomllib.loads` or `tomllib.load` call in the process goes through the counter, from any module or thread. TASK-38685 removed the same hazard for `json.loads`: there, process-wide counters picked up storage-admission parses from other modules, and the execution-log tests read `115 == 0`.

The `tomllib.loads = _counting_loads` at :171 is not a leak. It is inside `_EMBEDDED_TOML_PARSE_COUNT_SNIPPET`, which runs in a subprocess.

No miscount has been seen yet. On this base, both tests fail earlier with `RecoveryRequired: raw_source_selection_changed`, along with four other tests in the file (the class TASK-38111 owns), so their counts are never reached locally. The file runs on no PR lane. #3402's new guard (`Tests/CI/test_json_loads_patch_guard.py`) covers only `json.loads`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both tests count only parses made through `tldw_chatbook.config`'s own `tomllib` name, and each still fails when the code under test parses again (shown with a temporary source edit)
- [ ] #2 A guard under `Tests/CI` fails on any new process-wide patch of `tomllib.load` or `tomllib.loads` under `Tests/`, shown red with a planted stub
<!-- AC:END -->
