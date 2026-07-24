---
id: TASK-401.3
title: Define governed citation source locator contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 00:43'
updated_date: '2026-07-24 07:00'
labels:
  - rag
  - citations
  - security
  - resolvers
dependencies:
  - TASK-401.2
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-401
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define typed inert source locators, capability policy, and a versioned source inventory before any source-opening implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SourceLocatorEnvelope and static resolver registration reject arbitrary classes, commands, paths, URL handlers, and unknown payload versions.
- [x] #2 Storage mode and view, resolve, native-open, external-open, compare, refresh, and export capabilities are independently policy-derived.
- [x] #3 The versioned inventory classifies every enabled local and pinned server source kind including claims and snapshot-only SQL evidence.
- [x] #4 Imported and legacy free-form locators remain inert until a current authority lookup and explicit rebinding succeed.
- [x] #5 A bounded read-authorization contract binds profile or tenant scope, authority, and independent capabilities before governed payload hydration.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing hostile-envelope, authorization, inert-rebinding, and inventory contract tests plus the committed inventory fixture.
2. Implement strict frozen data-only locator envelopes, source-specific payloads, binding states, capability policies, and bounded read authorization.
3. Define the immutable runtime-to-canonical mapping and versioned local/server/derived source inventory, including SQL snapshot-only and claims parent-lineage rules.
4. Implement native validation and inert imported/legacy parsing/rebinding semantics without resolver callbacks or source opening.
5. Run focused locator, trace compatibility, lint, fixture, and diff checks; self-review and complete both independent review gates.
6. Complete acceptance criteria and implementation notes only after approval.

ADR required: yes
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: This task directly implements ADR-024’s typed locator, capability, authority, and inert rebinding boundary; no new architectural decision is introduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented strict frozen citation source locator, capability-policy, inventory, authorization, inert parsing, and explicit rebinding contracts. Source payloads are selected through a static data-only per-kind union; runtime producer mapping and the 14-entry local/server/derived inventory are pinned by a committed fixture. Local and server authority shapes are distinct, SQL remains snapshot-only, and claims current/open operations require an independently validated matching server-media parent lineage.

Hardened the hostile boundary with safe relative-path validation, independent capability checks, five-minute current-authority lookup freshness, model-copy revalidation, and bounded preflight of inert JSON before canonical serialization. The preflight rejects oversized/deep/count/cyclic/non-JSON/non-finite/subclass and hostile bigint inputs without attacker-sized intermediate allocation. No resolver, opener, navigation, URL fetch, persistence, or UI behavior was added.

Verification: 150 focused locator/trace compatibility tests passed; Ruff check/format and git diff checks passed. Independent specification and quality/security reviews approved the final implementation with no remaining Critical or Important findings.

ADR required: yes. Applied existing backlog/decisions/024-rag-citation-provenance-and-source-resolution.md; no new ADR was needed. Documentation remains the approved design, ADR, plan, and committed inventory fixture.
<!-- SECTION:NOTES:END -->
