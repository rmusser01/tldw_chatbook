---
id: TASK-38684
title: "Perf Guard does not run when only tldw_chatbook package data changes"
status: To Do
created_date: 2026-11-03 11:40
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
- [ ] #1 A PR that changes only a non-Python file shipped under tldw_chatbook/ that a Perf Guard step loads (for example a DB migration .sql file or Config_Files/default_config.toml) triggers Perf Guard on pull_request and on push
- [ ] #2 The Perf Guard trigger contract test fails when such a package-data path is outside either trigger (shown red with the new pattern removed)
<!-- AC:END -->
