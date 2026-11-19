---
id: TASK-35379
title: Qualify server Canvas capabilities message trees and byte admission
status: To Do
created_date: 2026-10-18 05:56
dependencies:
- TASK-35378
labels:
- canvas
- sync
- server
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Advertise Canvas synchronization only when the companion server can preserve complete conversation branches and transport every valid bounded artifact safely. Keep partial deployments incompatible rather than claiming small-document support.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The server proves stable complete chat-message ancestry and historical soft-deleted origins; older chat-only clients cannot strip, implicitly delete or reparent Canvas records.
- [ ] #2 The complete canvas-sync-v1 capability group advertises exact domains/operations/versions, chat-message-tree-v1, sibling-sequence-ties-v1, retired-owner barriers and unchanged artifact quotas only when every prerequisite is supported.
- [ ] #3 At least 4 MiB canonical payload admission and bounded HTTP/proxy request and pull-page limits carry worst-case escaped valid 512 KiB source plus bounded metadata without count-only amplification or truncation.
- [ ] #4 Pinned server integration tests cover absent/partial negotiation, read-only or changed enrollment, unsupported versions, byte boundaries and both advertised persistence backends; deployment at-rest encryption and server-readable-source policy are documented.
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
