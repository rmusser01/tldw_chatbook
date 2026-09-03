---
id: TASK-29232
title: Design portable Tool-use Pack export and import
status: Done
assignee:
  - '@codex'
created_date: '2026-08-31 14:22'
updated_date: '2026-08-31 15:56'
labels:
  - tool-packs
  - permissions
  - design
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the portable single-profile Tool-use Pack contract, trust boundary, import lifecycle, workspace binding confirmation, profile management behavior, and verification strategy before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reviewed design spec defines the single-profile deterministic archive, complete permission-addressable snapshot, safe flattening and fallback semantics, exact/manual mapping, and excluded data.
- [x] #2 A canonical ADR records the unbound import boundary, first-bind confirmation, unresolved rule behavior, atomic profile authority, deletion tombstones, and separation from Tools+Skills installation.
- [x] #3 The design specifies modular ownership, UI flows, stable fail-closed categories, concurrency behavior, privacy constraints, performance bounds, and targeted verification including a separate Windows-support claim.
- [x] #4 The spec and ADR contain no placeholders or contradictory requirements, link each other and relevant existing ADRs, and are committed on the isolated design branch.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write the approved Tool-use Pack design specification covering the archive contract, permission-addressable inventory, flattening, mapping, activation, binding, removal, UI, privacy, failure behavior, limits, and verification.
2. Record the security and service-contract decisions in ADR-107 and link the relevant existing ADRs.
3. Self-review the spec and ADR for placeholders, contradictions, scope creep, unsafe widening, and missing lifecycle behavior.
4. Verify documentation integrity and repository diffs, commit the design artifacts, then close the Backlog task with implementation notes.

ADR required: yes

ADR path: `backlog/decisions/107-portable-tool-use-packs.md`

Reason: The design establishes portable permission-policy semantics, import activation rules, deletion behavior, and first-bind security boundaries.
<!-- SECTION:PLAN:END -->

## Renumbering provenance

This task was originally filed as TASK-25713 at 2026-08-31 14:22, ten
minutes after the older "Census warm-boot flakes on sys.modules mutation
during iteration" task claimed TASK-25713 at 14:12. The duplicate-id CI
guard detected the collision during PR #2324. Per the TASK-19601 owner rule,
the older task keeps 25713 and this younger task renumbers to TASK-29232.
Live dependencies and documentation links were updated; historical commit
messages remain unchanged.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Defined the deterministic single-profile Tool-use Pack contract and recorded its
security/runtime boundary in ADR-107. The approved revision adds strict non-mutating
permission snapshots, authoritative imported-profile lifecycle state, named-profile
provider propagation, linearized first binding, durable receipts and tombstones,
safe publication/reconciliation, stale-event-safe UI ownership, and a separate
future boundary for Windows-native support and Tools+Skills/plugin installation.

Verification covered relative links, required provisions, placeholder scans,
whitespace, exact commit contents, and a clean isolated worktree. No production code
changed, so no runtime test suite was required for this design-only task.
<!-- SECTION:NOTES:END -->
