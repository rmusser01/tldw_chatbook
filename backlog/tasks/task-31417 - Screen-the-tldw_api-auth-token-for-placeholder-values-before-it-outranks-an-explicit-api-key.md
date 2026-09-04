---
id: TASK-31417
title: >-
  Screen the tldw_api auth token for placeholder values before it outranks an
  explicit api key
status: To Do
assignee: []
created_date: '2026-09-04 22:41'
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
- [ ] #1 A [tldw_api] section carrying a real api_key and the placeholder auth_token authenticates with the real credential
- [ ] #2 The placeholder screening reuses config.py's existing constant and validity check rather than a second copy of the rule
- [ ] #3 A genuinely configured auth_token still outranks api_key, so deliberate token users are unaffected
- [ ] #4 Choosing a credential over a rejected placeholder is visible in the log with the source named, not silent
- [ ] #5 A test pins the boot-rewrite plus resolve round trip end to end, not just the screening predicate in isolation
<!-- AC:END -->
