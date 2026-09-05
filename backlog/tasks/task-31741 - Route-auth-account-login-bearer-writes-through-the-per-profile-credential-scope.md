---
id: TASK-31741
title: >-
  Route auth-account login bearer writes through the per-profile credential
  scope
status: To Do
assignee: []
created_date: '2026-09-05 23:20'
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
- [ ] #1 auth-account login/account bearer writes go through a profile-scoped store method, not the plain legacy server_id slot
- [ ] #2 A non-default profile's login bearer is not readable by the default profile in scoped mode
- [ ] #3 Default single-profile users are unaffected (no re-auth, matching 31416 AC#4)
<!-- AC:END -->
