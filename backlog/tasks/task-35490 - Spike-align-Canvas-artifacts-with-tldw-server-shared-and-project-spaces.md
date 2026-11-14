---
id: TASK-35490
title: 'Spike: align Canvas artifacts with tldw_server shared and project spaces'
status: To Do
created_date: 2026-10-18 14:46
dependencies:
- TASK-31003
- TASK-35376
labels:
- canvas
- sync
- server
- spaces
- spike
priority: medium
references:
- Docs/superpowers/specs/2026-10-17-canvas-server-sync-contract-design.md
documentation:
- Docs/Canvas/UAT_2026-10-18.md
updated_date: 2026-10-18 15:00
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate how conversation-owned Canvas artifacts should align with shared spaces and project spaces in tldw_server before resuming Canvas synchronization. Produce an evidence-backed recommendation and bounded follow-up scope, not a server implementation or an assumption that the existing personal-dataset sync contract grants shared-space access.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A pinned read-only inventory identifies the actual tldw_server shared-space/project-space APIs, ownership and membership roles, project/conversation relationships, artifact support and existing synchronization contracts; implemented behavior is distinguished from proposals.
- [ ] #2 A compatibility map compares ADR-239 Canvas identities, immutable revision graphs and branch projection with space/project ownership, membership changes, permissions and revocation, source readability, concurrent changes, delete/restore, retention/purge and export/import; unsupported mappings and isolation risks are explicit.
- [ ] #3 The spike evaluates practical alignment options including keeping Canvas conversation-owned, exposing references from spaces/projects and introducing explicit shared ownership; it recommends a smallest safe option or a justified no-go without silently broadening zero-egress runtime capabilities or data authority.
- [ ] #4 A documented conformance/UAT matrix covers two users/replicas, allowed and denied access, membership removal, project moves, sibling revisions and lifecycle conflicts, with prerequisites and server-owner decisions clearly separated from verified results.
- [ ] #5 The output records exact code/API evidence, open questions, ADR required yes/no and rationale, proposed atomic follow-up scope and dependency order. No server writes, activation, migration, network/filesystem/cookie capability change or production/shared-data experiment is performed; the sync workstream remains paused until explicit user resumption.
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
