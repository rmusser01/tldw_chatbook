---
id: TASK-26065
title: Add strict lifecycle-aware permission snapshots
status: Done
assignee:
  - '@codex'
created_date: '2026-08-31 16:46'
updated_date: '2026-08-31 17:11'
labels:
  - tool-packs
  - permissions
  - security
dependencies:
  - TASK-25713
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a non-mutating schema-1 permission snapshot seam, authoritative imported/tombstone lifecycle validation, fail-closed resolver behavior, and profile-scoped raw getters so portable Tool Pack review can inspect policy without triggering legacy recovery.
<!-- SECTION:DESCRIPTION:END -->

## Renumbering provenance

Created as TASK-25836, which was already allocated on parallel branches. Renumbered
before implementation to TASK-26065; the older tasks retain TASK-25836.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Strict snapshot reads return immutable validated schema-1 policy and a generation token without creating, renaming, normalizing, backing up, resetting, or rewriting the permission file.
- [x] #2 Malformed, partial, mismatched, or unknown Tool Pack lifecycle metadata resolves Deny; valid tombstones short-circuit named-profile inheritance.
- [x] #3 Raw global/server/tool getters read the requested profile without seeding or mutating policy, while legacy load recovery and unknown legacy-profile inheritance remain compatible.
- [x] #4 Focused permission-store and resolver tests cover missing/valid/corrupt/unknown/nested-invalid snapshots, lifecycle variants, tombstones, and named raw getters.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red tests for byte-preserving strict reads, immutable snapshots, lifecycle dispositions, tombstone resolver short-circuits, and named raw getters.
2. Implement strict schema-1 snapshot parsing and immutable generation snapshots without changing legacy load recovery.
3. Implement exact imported/tombstone lifecycle structure validation and fail-closed resolver precedence.
4. Refactor raw getters to select a profile without seeding or mutating payloads.
5. Run the two focused MCP suites, self-review, and address review findings.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: Accepted ADR-107 already defines the strict read, lifecycle, resolver, and profile-access boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented strict immutable permission snapshots, lifecycle-aware fail-closed resolution, and profile-scoped raw getters. Fix round preserved corrupt-store bytes, typed deep JSON recursion failures, and enforced exact integer schema versions. Focused verification: 107 passed; existing RequestsDependencyWarning remains. ADR: existing ADR-107; no new ADR required.
<!-- SECTION:NOTES:END -->
