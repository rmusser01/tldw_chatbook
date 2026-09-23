---
id: TASK-32924
title: Image/Video Gen settings panels stop hitting the keyring on every compose
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
The Image and Video Gen settings panels resolve backend secrets in compose(), on the UI loop, on every open/save/revert/Test -- one keyring read per backend lacking an env/config key (up to 7 per compose). On Linux each is a D-Bus round trip; a locked keyring freezes the Settings screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repeated lookups for a backend within 10 s share one keyring round trip, including failed lookups
- [x] #2 A key added outside the app (keyring set) is picked up after the window
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TTL cache inside the shared Media_Generation keyring_get (both modalities route through it)
2. Tests with patched clock and keyring
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
`Media_Generation/config_machinery.keyring_get` (shared by image and video config) caches per (namespace, backend) for 10 s, including failed lookups. The app never writes these entries, so a TTL is the only invalidation; a key added with `keyring set` appears within the window (`ponytail:`). Tests keep patching `_keyring_get` above the cache, so no cross-test leakage. The panels still resolve config in `compose()`; moving that to a worker was skipped as a larger restructure than the cache warrants.

Files: `Media_Generation/config_machinery.py`, `Tests/Image_Generation/test_keyring_read_cache.py`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
