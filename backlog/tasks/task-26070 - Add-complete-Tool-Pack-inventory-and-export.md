---
id: TASK-26070
title: Add complete Tool Pack inventory and deterministic export
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-01 00:00'
updated_date: '2026-09-01 00:00'
labels:
  - tool-packs
  - export
  - inventory
  - security
dependencies:
  - TASK-26068
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Capture a complete classified inventory of permission-addressable tools, flatten
one strict profile snapshot into portable policy, and produce deterministic,
privacy-safe Tool Pack archives for later publication.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A code-owned registry captures every classified permission namespace, reports excluded categories/counts, and fails closed for unclassified or incomplete namespaces.
- [ ] #2 Inventory snapshots are immutable and deterministic, cover all V1 builtin/local/external providers, reject exact or case-folded identity duplicates, and contain no workspace roots or admitted aliases.
- [ ] #3 Export captures each authority once from strict immutable state, rejects invalid/tombstoned lifecycles, safely normalizes reserved source ids, and flattens named inheritance, definition changes, high-risk/raw-shell floors, and unseen fallbacks.
- [ ] #4 Definitionless Denies remain pending while definitionless Ask/Allow rules are omitted and reported; receipt history, workspace/Persona/config/session gates, and runtime-only data never enter portable output.
- [ ] #5 The two-member ZIP archive is byte-deterministic with pinned canonical JSON, member order, headers, modes, timestamps, hashes, and privacy exclusions; focused tests and scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red registry/inventory tests for namespace classification, excluded counts, concrete V1 providers, duplicate identities, completeness, and path privacy.
2. Implement immutable inventory types, adapters, and deterministic digest capture.
3. Add red export tests for strict one-snapshot flattening, lifecycle/id handling, policy floors, fallbacks, and missing definitions.
4. Implement side-effect-free export capture using Task 1 strict snapshots and Task 4 contracts.
5. Add deterministic ZIP golden/header/privacy tests and implement the canonical two-member archive writer.
6. Run focused inventory/export tests, scoped static checks, self-review, and independent review.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: ADR-107 already fixes the inventory boundary, policy-only export semantics, canonical archive contract, privacy exclusions, and Windows separation.
<!-- SECTION:PLAN:END -->

