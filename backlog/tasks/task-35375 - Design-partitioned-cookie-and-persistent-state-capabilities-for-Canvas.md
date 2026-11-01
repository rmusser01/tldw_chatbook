---
id: TASK-35375
title: Design partitioned cookie and persistent state capabilities for Canvas
status: To Do
created_date: 2026-10-18 04:42
labels:
- canvas
- roadmap
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define a future opt-in cookie and persistent-state model for applications rendered in Canvas, with isolation between users and artifacts and a clear lifecycle for stored credentials and data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A canonical ADR and reviewed design identify supported state APIs and partition keys covering profile, browser user, conversation and Canvas; Chatbook authentication state is never readable by generated content.
- [ ] #2 The contract defines consent, storage limits, expiry, clear/revoke actions, Temporary behavior, conversation deletion and the treatment of sensitive cookie attributes and outbound credentials.
- [ ] #3 Historical profiles remain immutable, and archive/import or synchronization cannot silently copy credentials or restore active grants.
- [ ] #4 Native/served compatibility, cross-artifact isolation, CSRF/session-fixation, stale-grant and deletion scenarios have explicit qualification criteria before implementation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
