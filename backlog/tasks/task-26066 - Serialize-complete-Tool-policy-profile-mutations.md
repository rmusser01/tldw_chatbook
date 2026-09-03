---
id: TASK-26066
title: Serialize complete Tool policy profile mutations
status: Done
assignee:
  - '@codex'
created_date: '2026-08-31 17:12'
updated_date: '2026-08-31 18:03'
labels:
  - tool-packs
  - permissions
  - concurrency
  - security
dependencies:
  - TASK-26065
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent lost updates and split-brain policy changes by sharing one resolved-path mutation fence, canonical profile digests, atomic complete-profile CAS operations, and lifecycle lease coordination across permission-store instances.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All permission-store mutators and low-level replacement saves serialize through one resolved-path reentrant fence with durable atomic replacement.
- [x] #2 Complete-profile install, update, and tombstone operations enforce generation, revision, lifecycle, collision, and size/profile limits.
- [x] #3 Canonical profile digests drive profile-scoped CAS while unrelated-profile edits may coexist.
- [x] #4 Lifecycle mutation and exact-profile lease accounting are process-wide, deterministic, and covered by barrier-based concurrency tests.
- [x] #5 Focused Tool_Packs authority and existing permission-store tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add deterministic red tests using threading barriers for multi-instance lost updates and complete-profile CAS.
2. Add one resolved-path reentrant mutation fence and move all permission mutations into locked reload/change/save operations.
3. Add canonical profile digests, digest-aware field mutators, and atomic install/update/tombstone profile operations.
4. Add the lifecycle mutation coordinator and exact-profile lease accounting.
5. Run focused authority and permission-store tests, self-review, and address independent review findings.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: Accepted ADR-107 already fixes the shared fencing, CAS, digest, and lifecycle coordinator boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented one resolved-path reentrant mutation fence, durable private atomic replacement, canonical policy digests, complete-profile CAS operations, and process-wide lifecycle/lease coordination. Independent review found and fix round 1 resolved three Important issues: mandatory initial first-bind marker, private-temp cleanup/selective directory-fsync errors, and deterministic contested-operation probes. Fresh verification: 71 focused tests passed; scoped Ruff and git diff --check passed; one pre-existing RequestsDependencyWarning remains. ADR: existing ADR-107; no new ADR required.
<!-- SECTION:NOTES:END -->
