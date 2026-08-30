---
id: TASK-24408
title: Add governed Personal Context agent tools and durable proposals
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-30 06:59'
labels:
  - personal-context
  - agents
  - tools
  - privacy
dependencies:
  - TASK-24407
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let eligible agents read bounded profile context and submit durable reviewable profile changes without bypassing user privacy controls or mutation authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Read-only and propose catalogs expose only tools allowed by current runtime authority.
- [ ] #2 Direct writes require exact current-user evidence and optimistic concurrency.
- [ ] #3 Pending proposals remain outside agent context and are quota-bound, conflict-safe, and content-shredded when resolved.
- [ ] #4 Workspace promotion always creates a reviewable proposal with provenance and never overwrites global context.
- [ ] #5 Run-scoped authority is invalidated after lifecycle, scope, binding, or grant changes.
- [ ] #6 Production-shaped provider, catalog, and concurrency regressions pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing proposal lifecycle and tool-provider tests for catalogs, exact evidence, visibility, scope isolation, quotas, conflict freezing, promotion provenance, and authority invalidation.
2. Implement the durable proposal service over canonical encrypted proposal/record operations, including bounded receipts and content shredding on every terminal state.
3. Implement the run-scoped Personal Context ToolProvider from Shared Core request contracts and wire it into each fresh Console registry with trusted current-user message evidence.
4. Run the targeted proposal/provider/catalog/concurrency suites, inspect privacy-sensitive durable owners, and obtain independent specification and code-quality review.

ADR required: no

ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md

Reason: ADR-102 already defines separate pending proposals, runtime-local authority, canonical mutation ownership, privacy behavior, and content-shredded terminal receipts; this task implements that accepted boundary.
<!-- SECTION:PLAN:END -->
