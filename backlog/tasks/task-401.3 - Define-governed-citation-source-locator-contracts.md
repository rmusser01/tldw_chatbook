---
id: TASK-401.3
title: Define governed citation source locator contracts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 00:43'
updated_date: '2026-07-24 06:05'
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
- [ ] #1 SourceLocatorEnvelope and static resolver registration reject arbitrary classes, commands, paths, URL handlers, and unknown payload versions.
- [ ] #2 Storage mode and view, resolve, native-open, external-open, compare, refresh, and export capabilities are independently policy-derived.
- [ ] #3 The versioned inventory classifies every enabled local and pinned server source kind including claims and snapshot-only SQL evidence.
- [ ] #4 Imported and legacy free-form locators remain inert until a current authority lookup and explicit rebinding succeed.
- [ ] #5 A bounded read-authorization contract binds profile or tenant scope, authority, and independent capabilities before governed payload hydration.
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
