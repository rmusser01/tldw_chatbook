---
id: TASK-34922
title: Decide whether Dreams retries tracked pages held for a private-looking address
status: To Do
created_date: 2026-10-12 23:18
labels:
- dreams
- decision
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When Dreams is switched on, a tracked page whose address resolves to a private or cloud-metadata address is held and never retried by the daily cycle; only a name that failed to resolve is retried. Both are answers from DNS, and a hotel or airport portal, or a VPN resolver, can answer every name with a private address, so switching Dreams on on such a network holds every tracked page until the user switches Dreams off and on again or re-tracks each story. A retry is the same public-address check, so it cannot enable a private address; it costs one lookup per held page per cycle. This needs the owner's decision. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision on retrying private and metadata holds is recorded
- [ ] #2 The behaviour and the user guide match the decision
- [ ] #3 A test pins the decided behaviour
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
