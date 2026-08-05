---
id: TASK-1693
title: Promote per-file artifact URLs to a descriptor schema v2 with migration
status: Done
assignee: []
created_date: '2026-08-01 06:48'
updated_date: '2026-08-01 07:03'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-596 adds an optional per-file url to ArtifactFile additively (default None, existing manifests keep parsing, acquisition prefers it and falls back to source_url). That is the pragmatic shape, not the principled one: per-file source URLs are part of an artifact's identity and belong in a versioned schema. Bump ArtifactDescriptor.schema_version to 2 with an explicit migration for manifests already written under v1, and decide whether url becomes required for multi-file artifacts at that point.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ArtifactDescriptor schema_version 2 defines per-file url semantics explicitly
- [ ] #2 Manifests written under schema_version 1 still load, via a documented migration path
- [ ] #3 Round-trip tests cover v1-written and v2-written manifests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Superseded by TASK-1695: the parallel TASK-595 branch's descriptor + source-map contract keeps per-file URLs out of the frozen descriptor schema entirely, so a schema v2 bump and its migration are unnecessary. See Docs/superpowers/reviews/2026-08-01-task-595-duplicate-implementation-reconciliation.md.
<!-- SECTION:NOTES:END -->
