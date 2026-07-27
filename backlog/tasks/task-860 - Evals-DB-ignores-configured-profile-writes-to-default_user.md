---
id: TASK-860
title: >-
  Evals DB ignores the configured profile and always writes to default_user
status: To Do
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
- [ ] The Evals DB path honours the same profile name every other DB uses
- [ ] Launching with a non-default profile creates `<profile>/evals.db`, not `default_user/evals.db`
- [ ] The data root honours the configured value rather than silently falling back
- [ ] A test asserts the resolved path changes when the profile changes
<!-- AC:END -->
