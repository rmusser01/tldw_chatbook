---
id: TASK-35377
title: Implement server-owned inert Canvas Sync v2 domains
status: To Do
created_date: 2026-10-18 05:55
dependencies:
- TASK-31003
labels:
- canvas
- sync
- server
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide authorized persistence for immutable Canvas documents and revisions behind the existing companion-server Sync v2 transport. This is server-repository follow-up work under approved ADR-239, not authorization to modify a server checkout from Chatbook.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The selected companion-server repository, maintainer and reviewed target build are recorded; its canonical Sync v2 contract declares canvas.document and canvas.revision adapter/schema v1 without a second CRUD surface.
- [ ] #2 Account/profile/dataset/conversation membership and exact message-origin/parent ownership are enforced before source admission; missing dependencies are retryable, foreign ownership is terminal, and incomplete roots are not advertised ready.
- [ ] #3 Exact immutable identities, bytes, hashes, metadata and sibling sequences persist inertly under quotas; identical replay is idempotent and same-ID divergent content is retained as an integrity conflict without overwrite.
- [ ] #4 Native server persistence and transport tests cover ownership, dependency order, immutable replay, invalid UTF-8/digests and source-free diagnostics; the complete Canvas capability remains unavailable until lifecycle and admission requirements qualify.
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
