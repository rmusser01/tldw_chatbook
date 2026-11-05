---
id: TASK-38754
title: "Config import hygiene tests patch tomllib for the whole process"
status: Done
assignee:
- '@claude'
created_date: 2026-11-05 07:11
updated_date: '2026-11-05 21:34'
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
- [x] #1 Both tests count only parses made through `tldw_chatbook.config`'s own `tomllib` name, and each still fails when the code under test parses again (shown with a temporary source edit)
- [x] #2 A guard under `Tests/CI` fails on any new process-wide patch of `tomllib.load` or `tomllib.loads` under `Tests/`, shown red with a planted stub
<!-- AC:END -->

## Implementation Plan

1. Generalize TASK-38685's syntax scan in `Tests/CI/test_json_loads_patch_guard.py` to a module name and attribute set, and add a `tomllib` (`load`, `loads`) guard beside the `json` one, with planted forms it must catch and scoped forms it must allow. Show it red at base on the two hygiene patches.
2. Patch `tldw_chatbook.config`'s own `tomllib` name in both hygiene tests (`SimpleNamespace(**{**vars(tomllib), name: stub})`), as TASK-38685 did for `json`.
3. Show each test still fails when config parses again (temporary source edit), running them under a temporary `@private_profile_test` marker because plain runs stop at `RecoveryRequired` (TASK-38111).
4. Plant a stub file under `Tests/` and show the guard red, then remove it.

## Implementation Notes

**Approach.** Both hygiene tests now replace `tldw_chatbook.config`'s own `tomllib` name with `SimpleNamespace(**{**vars(tomllib), "loads"|"load": counter})`, the form TASK-38685 used for `json`. The stdlib module is left alone, so a parse from another module or thread is never counted. TASK-38685's syntax scan in `Tests/CI/test_json_loads_patch_guard.py` now takes a module name and an attribute set (`json`/`loads` by default), and the same file has a `tomllib` (`load`, `loads`) guard. The file pre-filter is `"tomllib" in source`, which every form the scan can see contains.

**Tightened bound.** `test_forced_full_reload_does_not_reparse_embedded_default` asserted `len(calls) <= 1`, while its comment said "must not call tomllib.loads() again at all". Counted through config's own name, the forced reload makes 0 calls. With one re-parse of the embedded default added, the old bound still passed (count 1), so AC#1 could not hold. It now asserts `== 0`.

**Evidence.** Plain runs of both tests stop at `RecoveryRequired: raw_source_selection_changed` before they count (6 of 13 in the file, at base and after; TASK-38111). So each check below ran the two tests under a temporary `@private_profile_test` marker, written to the profile's own selected config path instead of a tmp path. The temporary changes were then removed.
- Scoped counts: forced reload 0 `loads`; startup pairing 1 `load`. Both pass.
- Scope: with extra `tomllib.loads`/`load` calls made through the stdlib module inside each test, the scoped tests still pass (0 and 1). The old process-wide form, with the same calls, failed both tests (1 and 2).
- Re-parse (temporary edit in `config._load_settings_uncached`: parse `CONFIG_TOML_CONTENT` and re-open the user file): old `<= 1` let the `loads` test pass, and the `load` test failed (2 calls). With `== 0`, both failed (1 call; 2 calls).
- Guard: red at base on exactly `Tests/Utils/test_config_import_hygiene.py:213` and `:265`; green after the fix. A planted `Tests/CI/test_zz_planted_tomllib_stub.py` (`monkeypatch.setattr(tomllib, "loads", ...)`) turned it red on `:5`, and the file was then removed. The file has 8 planted forms the scan must catch and 3 scoped forms it must allow.
- `Tests/CI/test_json_loads_patch_guard.py`: 33 passed, serial (it is in the `Tests/CI` fast-lane target).

**Files.** `Tests/Utils/test_config_import_hygiene.py`, `Tests/CI/test_json_loads_patch_guard.py`.

