---
id: TASK-31417
title: >-
  Screen the tldw_api auth token for placeholder values before it outranks an
  explicit api key
status: Done
assignee: []
created_date: '2026-09-04 22:41'
updated_date: '2026-09-05 23:03'
labels:
  - config
  - correctness
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A `[tldw_api]` section written with only `api_key = <real credential>` comes back from the app's first boot with `auth_token = "default-secret-key-for-single-user"` added beside it — the app's own config rewrite adds it. `RuntimeServerContextProvider._legacy_config_token` (`runtime_policy/server_context.py:771`) then resolves `auth_token or api_key or bearer_token`, so the placeholder the app wrote beats the credential the user wrote, and the only screening applied is `strip()`-to-empty.

`config.py` already screens PROVIDER keys for exactly this class of value: `resolve_provider_api_key` rejects placeholders, and `TLDW_API_PLACEHOLDER_AUTH_TOKEN = "default-secret-key-for-single-user"` is already a named constant at `config.py:1056`. The `[tldw_api]` token gets no such check on the resolution path.

Live incident (schedules-handoff PR-6 task 6, 2026-09-02): this is the sibling trap that SEEDED the keyring incident. The scratch profile was written with the real key under `api_key`; the first boot added the placeholder `auth_token` beside it; the placeholder won; and because the resolver imports whatever it resolves into the machine-global keyring, the placeholder was then cached under that `server_id` and outranked every subsequent correction. Round 2 worked around it by writing `auth_token` directly, which is a workaround for the author of a scratch profile and no help at all to a user who fills in the field the config file offers them.

The failure is silent in both directions: the user sees 401s with a valid credential in their config, and nothing reports that a placeholder was chosen over it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A [tldw_api] section carrying a real api_key and the placeholder auth_token authenticates with the real credential
- [x] #2 The placeholder screening reuses config.py's existing constant and validity check rather than a second copy of the rule
- [x] #3 A genuinely configured auth_token still outranks api_key, so deliberate token users are unaffected
- [x] #4 Choosing a credential over a rejected placeholder is visible in the log with the source named, not silent
- [x] #5 A test pins the boot-rewrite plus resolve round trip end to end, not just the screening predicate in isolation
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `resolve_tldw_api_auth_token(value)` to config.py, next to `TLDW_API_PLACEHOLDER_AUTH_TOKEN`/`resolve_provider_api_key`: it calls `resolve_provider_api_key` (the existing blank/placeholder screen) and additionally rejects a result equal to `TLDW_API_PLACEHOLDER_AUTH_TOKEN` -- one extra literal check, no duplicated rule (AC#2).

`RuntimeServerContextProvider._legacy_config_token` now resolves `auth_token` through that function first; a genuine value still returns immediately (AC#3 -- order unchanged, only the screening is new). Only when auth_token is rejected does it fall through to `api_key`/`bearer_token` (also now run through `resolve_provider_api_key`, closing the same "only strip()-screened" gap for those two fields as a small, safe generalization -- neither field can legitimately hold the tldw placeholder or a provider-key placeholder like "<API_KEY_HERE>"). When the placeholder specifically was rejected, an INFO log names the chosen source (`credential_source=config:api_key` / `config:bearer_token`) (AC#4).

The AC#5 round trip test (`test_boot_rewrite_placeholder_auth_token_does_not_outrank_real_api_key`, Tests/RuntimePolicy/test_server_context_provider.py) writes a `[tldw_api] api_key=...`-only file, loads it through the REAL `config_module.load_settings(force_reload=True)` (confirmed empirically this synthesizes `auth_token = TLDW_API_PLACEHOLDER_AUTH_TOKEN` into the merged config via `deep_merge_dicts(DEFAULT_CONFIG_FROM_TOML, user_config_from_file)` even though the placeholder is never written back to the file), then resolves through a real `RuntimeServerContextProvider` -- not a hand-built dict and not the predicate alone. A separate small predicate-only test file (Tests/test_config_tldw_api_auth_token_placeholder.py) pins `resolve_tldw_api_auth_token`'s edge cases directly.

All 4 new/changed tests were revert-checked: temporarily restoring the old `auth_token or api_key or bearer_token` (`str().strip()`-only) logic makes both the round-trip test and the placeholder-log test fail with the placeholder value returned instead of the real key, while the "genuine auth_token still wins" test keeps passing (confirms no over-fit).

Modified: tldw_chatbook/config.py (resolve_tldw_api_auth_token), tldw_chatbook/runtime_policy/server_context.py (_legacy_config_token rewritten). Tests: Tests/RuntimePolicy/test_server_context_provider.py (3 new tests), Tests/test_config_tldw_api_auth_token_placeholder.py (new, 3 tests). scripts/check_persistent_diagnostic_inventory.py --write regenerated jointly with task-31416's log-line additions (shared commit).

Landed in the same commit as task-31416 (shared file, shared incident, shared diagnostics-inventory regen) per the task-3 brief's instruction to close both together.
<!-- SECTION:NOTES:END -->
