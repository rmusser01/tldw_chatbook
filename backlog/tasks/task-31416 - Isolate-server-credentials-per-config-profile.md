---
id: TASK-31416
title: Isolate server credentials per config profile
status: Done
assignee: []
created_date: '2026-09-04 22:40'
updated_date: '2026-09-05 23:03'
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
- [x] #1 Two profiles pointed at the same base URL with different tokens each authenticate with their own token
- [x] #2 A profile's first boot cannot silently seed a credential that outranks a later corrected config value for a different profile
- [x] #3 The resolved credential's source is visible without reading the resolver, in the server-connection surface or the log
- [x] #4 Existing single-profile users keep their current credentials working across the change, with no re-entry prompt
- [x] #5 The live-verification lesson is updated to point at the fix rather than at the port-picking workaround
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wired the existing-but-unused ServerCredentialScope machinery into RuntimeServerContextProvider instead of building a new scoping scheme. The provider now takes a `credential_profile_id: str | None` constructor param; every credential read/write/clear goes through a `_credential_scope(server_id, purpose)` helper that builds `ServerCredentialScope(server_profile_id=<profile-or-server_id>, normalized_origin=server_id, credential_type=purpose)` and calls the store's `get_scoped_secret`/`set_scoped_secret`/`delete_scoped_secret` (added to `InMemoryServerCredentialStore` and `UnavailableServerCredentialStore` to match `KeyringServerCredentialStore`'s existing scoped API; `InMemoryServerCredentialStore`'s plain set/get/delete methods now delegate through `ServerCredentialScope.legacy` for parity, same as the keyring store already did).

`credential_profile_id=None` (the default -- every existing call site) makes `server_profile_id == server_id`, byte-for-byte `ServerCredentialScope.legacy`, so nothing changes for an unconfigured provider: AC#4 holds for the entire existing test suite with zero test edits. The ONE production wiring site, `app.py:_wire_server_context_provider`, passes `default_server_credential_profile_id()` (new in server_context.py): None for the default config path, `str(get_cli_config_path())` for a `TLDW_CONFIG_PATH`-retargeted one. This check lives only at the wiring boundary, not inside `RuntimeServerContextProvider`, deliberately -- `Tests/conftest.py` sets `TLDW_CONFIG_PATH` for the entire test session, so gating on the env var inside the provider itself would have scoped every unit test's credential store by the sandbox config path and broken ~380 existing assertions that read the plain unscoped keyring API directly.

Source visibility (AC#3): `_import_legacy_token` now logs an INFO line naming the purpose and profile on a successful import; `_resolve_auth_token` logs at DEBUG when a credential-store hit resolves (kept below INFO because `get_active_context()` runs on every client-cache-key computation, not just once per boot).

Modified: tldw_chatbook/runtime_policy/server_context.py (credential_profile_id param, _credential_scope/_credential_profile_scope_id helpers, default_server_credential_profile_id, every credential_store call site switched to the scoped API), tldw_chatbook/runtime_policy/server_credentials.py (scoped methods on InMemoryServerCredentialStore/UnavailableServerCredentialStore, Protocol extended, dead _credential_ref removed), tldw_chatbook/app.py (wiring). Tests: Tests/RuntimePolicy/test_server_context_provider.py (profile isolation, back-compat regression guard, log-visibility), Tests/RuntimePolicy/test_server_credentials.py (store-level scoped-isolation tests), test doubles in test_server_context_provider.py extended with the 3 new methods. Docs: backlog/docs/lessons-live-verification.md rewritten to point at the fix. scripts/check_persistent_diagnostic_inventory.py --write regenerated (5 new diagnostic calls, none interpolate secret values).

Concern for review: a user who has been running a NON-default TLDW_CONFIG_PATH for a while (not a scratch profile, an established custom path) will get a new, previously-unpopulated profile scope on upgrade and will see one re-auth prompt -- their old credential remains reachable only via the plain/legacy scope, which nothing now reads by default once credential_profile_id is non-None. This was a deliberate tradeoff versus a read-fallback design, because a fallback-to-shared-scope read would have let a stale contaminated shared entry (the exact incident) keep winning after the fix shipped. Flagging in case a broader "TLDW_CONFIG_PATH has been in long-term use" migration path is wanted later.
<!-- SECTION:NOTES:END -->
