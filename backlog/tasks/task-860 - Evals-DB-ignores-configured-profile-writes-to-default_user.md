---
id: TASK-860
title: >-
  Evals DB ignores the configured profile and always writes to default_user
status: Done
assignee: []
created_date: '2026-07-27 02:40'
labels:
  - evals
  - bug
  - data-isolation
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during live verification of the Evals rebuild. Every other database in the app honours the configured profile name and lands under `~/.local/share/tldw_cli/<users_name>/`. The Evals database does not — it always writes to `default_user/evals.db`, whatever the profile says.

The cause is a key-name mismatch in `Evals/eval_orchestrator.py:_initialize_database`. It resolves the profile with `settings.get("user_id", settings.get("username", "default_user"))`, but `load_settings()` publishes the profile name under the key **`USERS_NAME`**. Neither `user_id` nor `username` exists, so both lookups miss and the hardcoded `"default_user"` fallback wins every time. The same function also reads `settings.get("user_data_dir", ...)`, another key `load_settings()` does not publish, so the data root falls back too.

Two consequences:

1. **Profiles are not isolated for Evals.** A user with several profiles gets one shared `evals.db` for all of them, while every other DB is correctly separated.

2. **Test and scratch profiles write into the real user's Evals data.** This is how it was found: a verification run launched with `TLDW_CONFIG_PATH` pointing at a throwaway profile (`users_name = "evals_live"`) still created its bench, dataset and run inside `default_user/evals.db`. Every other DB the run touched was correctly created under `evals_live/`. Any agent or developer who trusts the documented scratch-profile recipe to protect real data is silently wrong for Evals alone.

Note this is pre-existing and unrelated to the word bench engine — `eval_orchestrator.py` is untouched by the rebuild.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] The Evals DB path honours the same profile name every other DB uses
- [x] Launching with a non-default profile creates `<profile>/evals.db`, not `default_user/evals.db`
- [x] The data root honours the configured value rather than silently falling back
- [x] A test asserts the resolved path changes when the profile changes
<!-- AC:END -->

## Implementation Notes

**Approach.** `EvaluationOrchestrator._initialize_database()` (`tldw_chatbook/Evals/eval_orchestrator.py`) no longer re-derives the profile/data-root from `settings.get(...)` with guessed keys. It now calls `tldw_chatbook.config.get_user_data_dir()` directly — the same shared, profile-aware helper every other DB (`get_chachanotes_db_path`, `get_media_db_path`, etc.) already uses. That helper correctly resolves `[general] users_name` (env `USERS_NAME`) and honours `[paths] data_dir` when configured.

**Existing-data handling.** Chose the notice-only option specified in the task rather than any silent copy/move. Added `EvaluationOrchestrator._warn_if_legacy_data_exists()`, called once per orchestrator construction (i.e., effectively once per app launch, since `EvalsDB.__init__` immediately creates the schema file, so the profile's own path exists on every subsequent call and the check short-circuits). It fires a `logger.warning` naming both the new (empty) path and the legacy path when: the resolved path doesn't yet exist, the legacy file does, and the two paths differ. It never touches either file.

One subtlety caught in self-review: the pre-fix code's `user_data_dir` lookup key was **never published** by `load_settings()` at all, so its fallback literal `"~/.local/share/tldw_cli"` always won — even for a user who had configured a custom `[paths] data_dir`. The true legacy location is therefore always the hardcoded `~/.local/share/tldw_cli/default_user/evals.db`, not `<configured_data_root>/default_user/evals.db`. The legacy-notice path is computed as that hardcoded literal (not derived relative to the newly-resolved path), or it would have missed exactly the users most likely to have a custom data root and lost track of an old file. Verified with a dedicated test (`test_legacy_notice_uses_the_hardcoded_legacy_location_even_with_custom_data_dir`) that fails against the naive relative-path version and passes against the shipped one.

**Tests.** New file `Tests/Evals/test_eval_orchestrator_db_path.py`, 6 tests:
- two different configured profiles resolve to two different `evals.db` paths, neither hardcoded to `default_user`
- a profile literally named `default_user` still resolves normally (not special-cased away)
- the legacy notice fires, names both paths, and leaves the legacy file byte-for-byte untouched
- no notice when there's no legacy file, and no notice once the profile has its own data
- the custom-`data_dir` edge case described above

Revert-check: reverted `eval_orchestrator.py` to `HEAD` and reran the new test file — 3 of 5 tests failed as expected (the two path-changed-with-profile assertions and the legacy-notice-fires assertion; the other two incidentally still passed for reasons orthogonal to the fix — one because the profile used was literally `default_user`, one because "no notice when nothing to notice about" holds trivially with no warning logic at all). Also reverted just the legacy-path line to the naive (buggy) relative-path form and confirmed the custom-`data_dir` test alone fails against it. Restored the fix afterward and reran clean.

**Test counts.** `pytest Tests/Evals/ -q`: baseline (pre-fix, without the new test file) 405 passed / 13 skipped / 0 failed; after (fix + 6 new tests) 411 passed / 13 skipped / 0 failed. No regressions, no pre-existing failures.

**Files modified:** `tldw_chatbook/Evals/eval_orchestrator.py`. **Files added:** `Tests/Evals/test_eval_orchestrator_db_path.py`. No schema changes; word bench engine and Evals UI untouched.
