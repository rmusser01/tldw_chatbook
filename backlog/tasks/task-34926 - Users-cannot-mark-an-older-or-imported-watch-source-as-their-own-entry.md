---
id: TASK-34926
title: Users cannot mark an older or imported watch source as their own entry
status: To Do
created_date: 2026-10-12 23:18
labels:
- watchlists
- security
- decision
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since the URL-provenance rework (ADR-207, amended 2026-10-12) a watch source counts as user-entered only when created or edited through the app's own entry paths. Every source a user made before the rework, and every source from an OPML or bundle import, now reads as unknown and is refused when its address is private, so intranet and NAS feeds that worked before stop. Watchlists has no way to edit a source's address, so the only remedies are the allowed-hosts setting or deleting and re-creating the source. This needs an owner decision: an explicit mark-as-mine action, or documenting allowed-hosts as the way. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision is recorded
- [ ] #2 A user whose private feed stopped after the upgrade can find in the app or the guide how to make it work again
- [ ] #3 Nothing marks a source as user-entered without an explicit user action
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
