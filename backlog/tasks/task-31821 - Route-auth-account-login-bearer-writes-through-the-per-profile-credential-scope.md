---
id: TASK-31821
title: >-
  Route auth-account login bearer writes through the per-profile credential
  scope
status: Done
assignee: []
created_date: '2026-09-05 23:20'
updated_date: '2026-09-06 15:21'
labels:
  - config
  - security
  - runtime-policy
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from close-out burndown Task 3 (31416) review. RuntimeServerContextProvider credential READS are now profile-scoped, but auth_account_scope_service.py:145,157 still WRITE the login/account bearer via the plain legacy store API (server_id slot), bypassing the new scope. In scoped mode a non-default TLDW_CONFIG_PATH profile's login bearer lands in the shared slot the default profile reads first (bearer is the first-resolved purpose) -- the same cross-profile credential exposure 31416 closed on the [tldw_api] config path, on the auth-account write path. Outside 31416/31417's config-resolution AC surface (both MEET spec), so filed rather than folded in.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 auth-account login/account bearer writes go through a profile-scoped store method, not the plain legacy server_id slot
- [x] #2 A non-default profile's login bearer is not readable by the default profile in scoped mode
- [x] #3 Default single-profile users are unaffected (no re-auth, matching 31416 AC#4)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Study 31416 scoping machinery: server_credentials.py (ServerCredentialScope, set_scoped_secret/get_scoped_secret/delete_scoped_secret) and server_context.py (_credential_profile_scope_id, _credential_scope, store_auth_tokens/store_static_server_credential as the existing scoped-write precedents).
2. Confirm the two unscoped write sites: auth_account_scope_service.py _set_effective_bearer_token/_delete_effective_bearer_token call credential_store.set_secret/delete_secret directly (plain legacy API), bypassing server_profile_id.
3. Add a scoped-write seam on RuntimeServerContextProvider (store_scoped_credential/delete_scoped_credential) mirroring the store's scoped naming, reusing _credential_scope so the default profile lands byte-for-byte on the pre-31416 legacy key.
4. Route the two write sites through the new seam (duck-typed via getattr/callable, matching the file's existing defensive style), falling back to the plain API only if a provider predates the seam.
5. Sweep the repo for other plain store_secret/delete_secret callers outside the scoped path -- confirm auth_account_scope_service.py is the only offender.
6. Add tests: two-profile isolation (pins both directions, fails without the fix), default-profile legacy-slot key equivalence (AC#3), and the service's own same-profile round trip. Revert-check the isolation test against the pre-fix code.
7. Run Tests/Auth_Account/ + Tests/RuntimePolicy/ foreground.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a scoped-write seam, `RuntimeServerContextProvider.store_scoped_credential(server_id, purpose, secret)` /
`.delete_scoped_credential(server_id, purpose)`, in `runtime_policy/server_context.py`, right after the existing
scoped-write precedent `store_static_server_credential`. Both are thin wrappers around
`self.credential_store.set_scoped_secret/delete_scoped_secret(self._credential_scope(server_id, purpose), ...)` --
the same `_credential_scope`/`server_profile_id` machinery every other write on the provider already uses (task-31416),
so no parallel scoping scheme was invented.

`Auth_Account_Interop/auth_account_scope_service.py`'s `_set_effective_bearer_token`/`_delete_effective_bearer_token`
were the only two write sites in the whole repo still calling the plain legacy `credential_store.set_secret`/
`delete_secret` API (confirmed by a repo-wide grep for `.set_secret(`/`.delete_secret(` outside `Tests/` -- no other
offenders). Both now look up `store_scoped_credential`/`delete_scoped_credential` on `server_context_provider` via
`getattr`/`callable` (matching the file's existing defensive-duck-typing style) and use it when present, falling back
to the old plain call only for a hypothetical provider double that predates the seam -- production only ever wires the
real `RuntimeServerContextProvider`.

For the DEFAULT profile (`credential_profile_id is None`), `_credential_profile_scope_id(server_id) == server_id`, so
`_credential_scope(server_id, purpose)` is field-for-field identical to `ServerCredentialScope.legacy(server_id, purpose)`
-- the scoped write lands on the exact same keyring entry the old plain write used (AC#3, pinned by
`test_store_login_tokens_default_profile_bearer_write_matches_legacy_slot`, which reads back through the plain
`get_secret` API).

Root cause of the exposure: for `auth_mode="bearer"`, `_purposes_for_auth_mode` checks `SERVER_CREDENTIAL_BEARER_TOKEN`
*before* `SERVER_CREDENTIAL_ACCESS_TOKEN`. `store_login_tokens` always called the already-scoped `store_auth_tokens`
(access/refresh token pair, correctly profile-scoped since 31416) *and* the unscoped bearer write -- so a non-default
profile's ACCESS_TOKEN was safely isolated, but its BEARER_TOKEN (checked first) landed in the same slot the DEFAULT
profile's own bearer resolution reads. Pinned with
`test_store_login_tokens_isolates_effective_bearer_token_across_profiles`, which constructs two real
`RuntimeServerContextProvider`s (default + `credential_profile_id="scratch-profile"`) sharing one
`InMemoryServerCredentialStore`/server_id and asserts each resolves only its own login bearer after both log in.
Revert-checked: reverting the two source files (`git checkout --`, then `git apply` to restore) reproduces the failure
-- the default profile's `get_active_context().auth_token` resolves to the scoped profile's token instead of its own.
A third test (`test_store_login_tokens_round_trips_through_the_same_scoped_profile`) pins the service's own same-profile
round trip (login -> bearer stored -> same profile's resolver finds it), independent of the default profile.

Sweep for sibling unscoped write sites: `grep -rn "\.set_secret(\|\.get_secret(\|\.delete_secret(" tldw_chatbook/`
(excluding `Tests/`) turned up only the two now-fixed call sites in `auth_account_scope_service.py`. `server_context.py`'s
own `clear_server`/`clear_all` calls (lines ~453/460/467) already route through `_credential_profile_scope_id` --
no other offenders found. (`clear_server`'s single-server-vs-whole-profile scoping is task-31824, a separate follow-up,
not touched here.)

Files touched:
- `tldw_chatbook/runtime_policy/server_context.py` -- added `store_scoped_credential`/`delete_scoped_credential`.
- `tldw_chatbook/Auth_Account_Interop/auth_account_scope_service.py` -- route the two bearer write sites through the new seam.
- `Tests/Auth_Account/test_auth_account_scope_service.py` -- 3 new tests + a `_real_provider` helper building an actual
  `RuntimeServerContextProvider` (the existing `FakeServerContextProvider` in this file has no profile-scoping concept
  and was left untouched; its tests still pass via the fallback branch).
<!-- SECTION:NOTES:END -->
