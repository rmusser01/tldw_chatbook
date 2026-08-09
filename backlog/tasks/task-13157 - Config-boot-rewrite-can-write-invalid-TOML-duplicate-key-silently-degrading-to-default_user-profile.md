---
id: TASK-13157
title: >-
  Config boot-rewrite can write invalid TOML (duplicate key), silently degrading
  to default_user profile
status: To Do
assignee: []
created_date: '2026-08-09 16:47'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found via live TUI verification (supervisor-fleet PR-1, Task 7). A hand-authored scratch config.toml with a single [api_settings.openrouter] api_key entry, after the app's own config-rewrite-on-boot pass ran across two launches, ended up with the SAME api_key key defined TWICE inside the same [api_settings.openrouter] table -- invalid TOML (tomllib raises 'Cannot overwrite a value'). Because the config file failed to parse, the app appears to have silently fallen back to defaults: the resolved data directory became ~/.local/share/tldw_cli/default_user/ (the REAL user's live profile) instead of the configured users_name profile, and the first-run wizard re-offered itself despite [first_run] setup_completed=true being present in the (invalid) file. Confirmed via lsof that the running process held open file handles on default_user/*.db, and via direct SQL/mtime checks that no actual data was written there in this incident -- but the mechanism is a real, reproducible, silent isolation failure: an app whose own config-normalization write path can corrupt its own config file into unparseable TOML, with no error surfaced to the user, and a silent fallback to the DEFAULT profile rather than a fail-loud error. This has real blast radius for any user whose config.toml happens to trigger the same duplicate-key condition on a rewrite pass -- their session could silently start operating against the wrong (default) profile without any indication.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Root cause identified: which config-rewrite/normalization code path in config.py can write a duplicate key into an existing TOML table, and under what preconditions
- [ ] #2 The rewrite path is fixed to be idempotent (never emits a duplicate key for a table it is re-serializing)
- [ ] #3 A regression test constructs a config.toml already containing a user-set key that coincides with a template default key, runs the rewrite path twice, and asserts the result stays valid, parseable TOML with exactly one occurrence of that key
- [ ] #4 If config loading ever fails to parse, the app fails loudly (a visible error/notification naming the file and parse error) rather than silently falling back to the default_user profile
<!-- AC:END -->
