---
id: TASK-36767
title: Route the mobile API to canonical retained Console owners
status: To Do
created_date: 2026-10-25 17:39
dependencies:
- TASK-36761
- TASK-36762
- TASK-36763
- TASK-36764
- TASK-36766
labels:
- console
- web-server
- mobile
- adr-244
priority: high
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the separate mobile engine with authenticated access to existing served Console chats and permit shared mobile writes only after verified legacy cutover.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The mobile service no longer constructs a headless controller/store/bridge or selects mobile-only databases; phone-created chats, saved history and live controls use the retained canonical owner.
- [ ] #2 Authenticated API requests route exact epochs, sessions, runs and decisions through the shared transport, preserve CSRF/origin/device/TLS policies and reconcile outcomes without offline replay.
- [ ] #3 Saved conversation ownership is reserved before hydration and registered on persistence; unavailable native or served owners are shown honestly and are not resumed by a read.
- [ ] #4 Canonical mobile writes require verified cutover, including verified no-data sources, while existing Library reading and supported sign-in/device flows remain available.
- [ ] #5 API and stream failures, subscriber pressure, revoked sessions and stale protocol versions preserve the retained runtime and authoritative outcomes.
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
