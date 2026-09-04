---
id: TASK-31416
title: Isolate server credentials per config profile
status: To Do
assignee: []
created_date: '2026-09-04 22:40'
labels:
  - config
  - security
  - runtime-policy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`TLDW_CONFIG_PATH` isolates the config FILE. It does not isolate credentials, and neither does `users_name`.

`RuntimeServerContextProvider._resolve_auth_token` (`runtime_policy/server_context.py:681`) resolves credentials from `KeyringServerCredentialStore` — the machine-global OS keyring, service `tldw_chatbook.server_credentials`, keyed by `server_id` (the base URL) — BEFORE the `[tldw_api]` config fallback. `_import_legacy_token` then writes the config token into that keyring the first time it is used. A scratch profile's first boot therefore permanently seeds a machine-global entry under its `server_id`, and from then on that entry outranks every corrected value in the scratch config file.

Live incident (schedules-handoff PR-6 task 6, 2026-09-02): the live gate could not authenticate against a local tldw_server. Every scheduling call returned 401 while the scratch config carried the real `SINGLE_USER_API_KEY`, a direct `httpx` probe with that key returned 200, and a header-dumping listener proved the app was putting the CORRECT `X-API-KEY` on the wire when pointed at a different port. Roughly 25 minutes went into it. The cause was a keyring entry imported under `server_id = "http://127.0.0.1:8000"` during the profile's very first boot, when the token was still a placeholder. Moving the server to an unused port made the identical client and identical config authenticate on the first request. Round 2 avoided it by choosing a fresh port up front and authenticated immediately.

The trap is recorded in `backlog/docs/lessons-live-verification.md`, which makes it survivable but does not fix it. Two things are wrong independently of any workaround: an isolated profile silently shares a credential namespace with every other profile on the machine, and the failure is diagnosable only by reading the resolver — nothing in the UI or the log says which source the credential came from.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Two profiles pointed at the same base URL with different tokens each authenticate with their own token
- [ ] #2 A profile's first boot cannot silently seed a credential that outranks a later corrected config value for a different profile
- [ ] #3 The resolved credential's source is visible without reading the resolver, in the server-connection surface or the log
- [ ] #4 Existing single-profile users keep their current credentials working across the change, with no re-entry prompt
- [ ] #5 The live-verification lesson is updated to point at the fix rather than at the port-picking workaround
<!-- AC:END -->
