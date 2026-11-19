---
id: TASK-35373
title: Design an opt-in Canvas network capability
status: To Do
created_date: 2026-10-18 04:41
labels:
- canvas
- roadmap
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define narrowly authorized external data access for a future Canvas profile. Users must understand and control which destinations and data a document can reach; the current zero-egress profiles remain immutable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A canonical ADR and reviewed design select the execution/origin boundary and specify explicit opt-in grants bound to the current browser, conversation, Canvas, revision and destinations.
- [ ] #2 The contract defines methods, redirects, credentials, private/loopback addresses, DNS changes, request/response limits, timeouts, revocation and session expiry without exposing Chatbook credentials or APIs.
- [ ] #3 The design specifies source-only behavior and safe import/export, branch and profile transitions when capabilities are unavailable; imported content cannot grant itself access.
- [ ] #4 A threat model, consent/denial/revocation journeys and an adversarial qualification plan are approved before any network-enabled runtime is implemented.
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
