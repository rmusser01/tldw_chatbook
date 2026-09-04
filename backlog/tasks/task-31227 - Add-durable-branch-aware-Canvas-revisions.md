---
id: TASK-31227
title: Add durable branch-aware Canvas revisions
status: In Progress
assignee: []
created_date: '2026-09-03'
updated_date: '2026-09-04 05:04'
labels:
  - canvas
  - database
  - conversations
dependencies:
  - TASK-31226
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the local Canvas domain, immutable revision graph, and persistence boundaries so conversations can own multiple named artifacts whose visible head follows the active message branch while temporary sessions remain genuinely temporary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A migration from the actual schema head adds Canvas documents, immutable revisions, revisioned titles/runtime profiles, origin-message/turn linkage, and local reopen hints
- [ ] #2 Repository operations enforce conversation ownership, same-Canvas parentage, unique sequence numbers, digests, quotas, and parameterized SQL transactionally
- [ ] #3 Active-path resolution excludes sibling branches and deterministically selects the newest eligible revision
- [ ] #4 Selecting a historical revision makes the next update or rename branch from that exact parent without mutating prior history
- [ ] #5 Stale `expected_parent_revision_id` values make no mutation and return bounded current metadata
- [ ] #6 Temporary Canvas history stays in memory, displays as temporary state, and joins conversation/message persistence atomically during existing session promotion
- [ ] #7 Failed promotion restores the complete session and Canvas state to temporary; unsaved session shutdown destroys staged history
- [ ] #8 Conversation soft delete, restore, and hard purge apply the existing lifecycle to owned Canvases without adding Canvas data to sync logs
- [ ] #9 Focused migration, repository, property-based branch, race, promotion-rollback, and lifecycle tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md
Reason: this task implements ADR-115’s durable ownership, immutable revision graph, active-branch resolution, and atomic temporary-promotion contract; no new ADR is needed unless implementation changes those accepted boundaries.

1. Allocate migration 66 from verified schema head 65 and implement the immutable Canvas repository with migration, concurrency, rollback, lifecycle, and query-plan tests.
2. Add the scoped Canvas service with active-message-path resolution, historical branching, deterministic conflicts, and centralized durable quotas.
3. Add in-memory temporary Canvas staging and join the existing conversation/message promotion transaction with complete rollback and shutdown destruction.
4. Run only the focused migration, Canvas repository/service/staging, and affected chat persistence suites plus static checks.
5. Update TASK-31227 and ADR-115 with the actual schema, constraints, quotas, query plans, rollback evidence, review outcome, and remaining limitations.
<!-- SECTION:PLAN:END -->

## Related Design

- `Docs/superpowers/specs/2026-09-03-chatbook-canvas-design.md`
- `Docs/superpowers/plans/2026-09-03-chatbook-canvas-implementation.md`
- `backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md`
