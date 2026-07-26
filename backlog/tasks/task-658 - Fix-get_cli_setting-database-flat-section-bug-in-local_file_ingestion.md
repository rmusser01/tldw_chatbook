---
id: TASK-658
title: Fix get_cli_setting("database", {}) flat-section bug in local_file_ingestion.py
status: To Do
assignee: []
created_date: '2026-07-25'
labels: [config, bug, ingestion]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`quick_ingest()` in `Local_Ingestion/local_file_ingestion.py:1148` reads `db_config = get_cli_setting("database", {})`, then does `db_config.get("media_db_path", ...)`. This is the second instance (found during the TASK-545 investigation) of TASK-547's bug: `get_cli_setting(section, key=None, default=None)` treats a non-string second positional argument as the `default` value, not as `key`. Concretely, `get_cli_setting("database", {})` calls with `key={}`, `default=None`; since `key` is not a string and `default is None`, `config.py`'s own fallback branch sets `default = key` (i.e. `{}`) and — because `"." not in "database"` — returns that `default` immediately, **discarding the real `[database]` TOML section every single call**. `db_config` is therefore always `{}`, so `media_db_path` always falls back to its hardcoded default and the `[database]` section can never override it through this call site.

`get_cli_setting` has no "read the whole section as a dict" call shape at all (confirmed by inspecting its implementation) — the fix is to read the section directly off the loaded config (e.g. via `load_cli_config_and_ensure_existence()` or an existing config accessor), not to keep forcing a section read through `get_cli_setting`'s key/default argument slots.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] `quick_ingest()`'s database-path resolution reads the real `[database]` section from the loaded config, not an unconditional `{}`
- [x] Setting `media_db_path` under `[database]` in `config.toml` is honored by `quick_ingest()` when no explicit `db_path` argument is given
- [x] Unit test confirms a configured `[database].media_db_path` is used, and that the hardcoded fallback only applies when the section/key is genuinely absent
- [x] A quick repo-wide grep for the same `get_cli_setting("<section>", {})` / `get_cli_setting("<section>", [])` pattern is done and any further instances are noted (fixed here or filed) so this bug class does not have a third silent occurrence
<!-- AC:END -->

## Implementation Notes

AC#4's repo-wide sweep (done during TASK-545 P3) found four further live instances of this bug class beyond `quick_ingest()`, none fixed inline — each filed as its own follow-up task so it gets its own fix/verification/test cycle:

- TASK-699 — `Widgets/splash_screen.py` and `Widgets/settings_splash_screen_viewer.py`, `[splash_screen]` (two instances)
- TASK-700 — `Web_Server/serve.py`, `[web_server]`
- TASK-701 — `TTS/backends/openai.py`, three no-key calls (`openai_api`/`API`/`app_tts`) — same symptom, slightly different call shape (no key at all rather than a stray non-string second argument)

The underlying bug class itself (rather than each individual call site) is now tracked by TASK-703, which also covers `save_setting_to_cli_config`'s sibling defect.
