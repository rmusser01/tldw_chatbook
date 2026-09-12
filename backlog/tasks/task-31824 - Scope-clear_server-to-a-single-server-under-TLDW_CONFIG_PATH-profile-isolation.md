---
id: TASK-31824
title: Scope clear_server to a single server under TLDW_CONFIG_PATH profile isolation
status: Done
assignee: []
created_date: '2026-09-06 08:08'
updated_date: '2026-09-06 15:45'
labels:
  - config
  - reliability
  - runtime-policy
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from schedules close-out burndown (PR #2454) -- Qodo review + final whole-branch review both flagged it. In TLDW_CONFIG_PATH scoped mode, RuntimeServerContextProvider.clear_server (runtime_policy/server_context.py:~455) filters on server_profile_id, so signing out of ONE server clears credentials for ALL servers in that profile. Fail-safe (over-clears, never under-clears) and only reachable in the rare multi-server-per-scratch-profile case; default single-server installs are unaffected -- which is why it did not block the burndown merge (final review rated INFO, Qodo rated High). Also in-area (optional): 3 Qodo Medium docstring-completeness nits from the same PR -- resolve_tldw_api_auth_token (config.py:~1406), the per-profile helper's returns: (server_context.py:~46), and the scoped credential methods (server_credentials.py:~45).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 clear_server in scoped mode clears only the target server's credentials, not sibling servers in the same profile
- [x] #2 A test pins the scoped-clear behavior: in a multi-server profile, clearing server A leaves server B's credentials intact
- [x] #3 Default single-profile / single-server users are unaffected (no behavior change)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Grep every caller of clear_server (store-level and provider-level) to confirm no whole-profile-clearing caller exists that needs preserving.
2. Add an optional keyword-only normalized_origin param to ServerCredentialStore.clear_server (protocol + InMemory/Keyring/Unavailable implementations), filtering on server_profile_id AND normalized_origin when given; omitted keeps today's whole-profile-match semantics for backward compat with existing single-arg callers/tests.
3. Update RuntimeServerContextProvider.clear_active_server_credentials / clear_server_credentials to pass normalized_origin=server_id, matching the same scope grammar _credential_scope already uses for reads/writes (including T1's scoped bearer seam).
4. Complete the three rider docstrings (config.py resolve_tldw_api_auth_token, server_context.py default_server_credential_profile_id, server_credentials.py scoped credential Protocol methods).
5. Add revert-checked pinning tests at both layers (InMemory + Keyring store, provider clear_active_server_credentials + clear_server_credentials) and run the touched suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: RuntimeServerContextProvider.clear_active_server_credentials/clear_server_credentials passed only self._credential_profile_scope_id(server_id) to credential_store.clear_server, which in scoped (TLDW_CONFIG_PATH) mode is the SAME value for every server in the profile -- the origin was never passed to the store, so clear_server could not distinguish servers within one profile.

Fix: ServerCredentialStore.clear_server (Protocol + InMemoryServerCredentialStore + KeyringServerCredentialStore + UnavailableServerCredentialStore) gained an optional keyword-only normalized_origin param. Both stores now filter on server_profile_id AND normalized_origin when it is given; omitted, behavior is byte-identical to before (whole-profile match) -- this keeps every existing 1-arg caller (test doubles in Tests/Auth_Account, direct store-level tests) working unchanged. The two provider call sites now pass normalized_origin=server_id, using the same scope grammar _credential_scope already uses for reads/writes, so T1's scoped bearer write seam (store_scoped_credential/delete_scoped_credential, commit b4852fb0a) is cleared correctly too. KeyringServerCredentialStore's legacy-purpose cleanup loop (for un-migrated pre-task-31416 entries) is gated to only run when normalized_origin is omitted or equals the profile id, since those legacy usernames only ever existed under server_id==profile_id (default mode).

Caller inventory: grepped every call of .clear_server( / clear_server_credentials / clear_active_server_credentials -- exactly two production call sites exist (both in server_context.py), both intend single-server clears; clear_all_credentials (the only 'sign out everywhere' action) goes through the separate clear_all() method and was untouched. No whole-profile caller needed preserving, so no extra clear_profile/clear_all path was added.

Default-mode confirmation (AC#3): in un-retargeted mode _credential_profile_scope_id(server_id) == server_id, so profile-scope filtering already equals per-server filtering (server_profile_id and normalized_origin are always equal for that server, and different servers get different profile ids) -- the existing test_clear_active_server_credentials_and_clear_server_credentials_clear_per_server_secrets test (unchanged) continues to pin this.

Docstring rider: added Args/Returns to resolve_tldw_api_auth_token (config.py), a Returns section to default_server_credential_profile_id (server_context.py), and Args/Returns docstrings to the three scoped credential Protocol methods plus clear_server's new normalized_origin semantics (server_credentials.py).

Tests added (revert-checked -- confirmed each fails against the pre-fix code, restored after): test_in_memory_clear_server_with_origin_scopes_to_one_server_in_shared_profile (test_server_credentials.py), test_keyring_clear_server_with_origin_scopes_to_one_server_in_shared_profile (test_server_credentials_lane_a.py), test_clear_server_credentials_scoped_mode_leaves_sibling_server_intact and test_clear_active_server_credentials_scoped_mode_leaves_sibling_server_intact (test_server_context_provider.py) -- the latter two use a T1-style bearer purpose alongside access_token to confirm the scoped bearer seam clears correctly.

Modified files: tldw_chatbook/runtime_policy/server_credentials.py, tldw_chatbook/runtime_policy/server_context.py, tldw_chatbook/config.py, Tests/RuntimePolicy/test_server_credentials.py, Tests/RuntimePolicy/test_server_credentials_lane_a.py, Tests/RuntimePolicy/test_server_context_provider.py.

Verification: pytest Tests/RuntimePolicy/ Tests/Auth_Account/ Tests/test_config_tldw_api_auth_token_placeholder.py -q -> 433 passed.
<!-- SECTION:NOTES:END -->
