---
id: TASK-32922
title: Server API calls no longer read the keyring on every call
status: Done
assignee:
- '@claude'
created_date: 2026-09-23 15:43
labels:
- performance
- linux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In server mode, build_client resolves the auth token before its client cache can hit, so every server API call (about 50 service entry points, some polling) read the OS keyring synchronously -- a D-Bus round trip per call on Linux, largely on the UI loop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repeated secret reads for a scope share one keyring round trip
- [x] #2 Writes, deletes, clear_server and clear_all are visible to the very next read
- [x] #3 A read racing a write or delete can never re-cache the old secret
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Per-entry read cache in KeyringServerCredentialStore.get_scoped_secret
2. Generation counter bumped after every write/delete (in finally)
3. Tests for dedup and immediate visibility of writes/deletes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
`KeyringServerCredentialStore.get_scoped_secret` now reads through a per-entry 30 s cache. `set_scoped_secret` and `delete_scoped_secret` (and so `clear_server`/`clear_all`) drop it AFTER the write in a `finally`, bumping a generation counter so a read in flight across a write can never re-cache the old secret. Failures are not cached (they surface as credentials-unavailable; recovery is re-auth). A secret rotated by another process is seen within the window (`ponytail:`).

Files: `runtime_policy/server_credentials.py`, `Tests/RuntimePolicy/test_server_credentials.py`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
