---
id: TASK-38684
title: "Perf Guard does not run when only tldw_chatbook package data changes"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-03 11:40
updated_date: '2026-11-04 01:02'
labels:
- follow-up
- ci
- performance
priority: low
references:
- .github/workflows/perf-guard.yml
- Tests/CI/test_ci_queue_pressure_contract.py
dependencies:
- TASK-36529
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (TASK-36529's review, PR #3380, 2026-11-03). Checked on dev 9bbd5c2515 and at #3380's head.

Perf Guard's `pull_request` and `push` path filters cover `tldw_chatbook/**.py` and `tldw_chatbook/css/**`. The package also ships non-Python files that the app reads at boot or on first use. A PR that changes only those files merges without Perf Guard. Tracked examples:
- `DB/migrations/` (99 files, mostly SQL). Every database opens through its migrations.
- `Config_Files/` (`default_config.toml`, `note_templates*.json`, `openai_tts_mappings.json`, ...). An empty profile loads `default_config.toml`.
- `assets/` (`persona_visual` 67 files, `characters` 57, `tiktoken_cache` 9).
- `Canvas/static` (15), `Canvas/mermaid` (8), `Web_Server/static` (14), `Plugins/migrations` (5), `Chunking/engine` (4).

#3380 adds the test helpers the steps import and `packages/tldw_profile_core/src/**`. Its trigger contract test derives Python imports and keeps a short hand-written list of data dependencies, which covers only `Tests/Plugins/fixtures/native/**`. So it cannot see package data either.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A PR that changes only a non-Python file shipped under tldw_chatbook/ that a Perf Guard step loads (for example a DB migration .sql file or Config_Files/default_config.toml) triggers Perf Guard on pull_request and on push
- [x] #2 The Perf Guard trigger contract test fails when such a package-data path is outside either trigger (shown red with the new pattern removed)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Widen both Perf Guard triggers from `tldw_chatbook/**.py` + `tldw_chatbook/css/**` to `tldw_chatbook/**`: the job installs editable, so the app reads every package file from the checkout.
2. Make the trigger contract test count the package's files as loaded data, so a package-data path outside either trigger fails it.
3. Show the contract test red with the old patterns, then green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both Perf Guard triggers now list `tldw_chatbook/**` in place of `tldw_chatbook/**.py` and `tldw_chatbook/css/**`. The job installs editable, so the app reads every package file (migrations, `Config_Files/`, `assets/`, `Canvas/static`...) from the checkout. Any file under the package now triggers Perf Guard on `pull_request` and on `push`. A package-internal README edit also triggers it now. That costs one extra run on a rare PR, and naming file types would bring back the list that went stale.

`test_perf_guard_runs_when_a_file_its_steps_load_changes` adds `tldw_chatbook` to `PERF_GUARD_DATA_DEPENDENCIES`. Every file under the package (`__pycache__` excluded) now counts as loaded data. The test also asserts that `Config_Files/default_config.toml` is counted, so the data entry cannot be dropped without notice.

**Red/green.** With dev's two old patterns restored, the contract test fails: `pull_request paths miss files its steps load`, listing 345 package files, including 103 `.sql` migrations and `Config_Files/default_config.toml`. With the new pattern it passes. `Tests/CI` at `-n 4`: 1177 passed, 1 skipped.

Files: `.github/workflows/perf-guard.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
<!-- SECTION:NOTES:END -->
