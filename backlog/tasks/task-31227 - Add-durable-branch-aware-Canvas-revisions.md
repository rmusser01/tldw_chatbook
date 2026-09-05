---
id: TASK-31227
title: Add durable branch-aware Canvas revisions
status: Done
assignee: []
created_date: '2026-09-03'
updated_date: '2026-09-04 09:40'
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
- [x] #1 A migration from the actual schema head adds Canvas documents, immutable revisions, revisioned titles/runtime profiles, origin-message/turn linkage, and local reopen hints
- [x] #2 Repository operations enforce conversation ownership, same-Canvas parentage, unique sequence numbers, digests, quotas, and parameterized SQL transactionally
- [x] #3 Active-path resolution excludes sibling branches and deterministically selects the newest eligible revision
- [x] #4 Selecting a historical revision makes the next update or rename branch from that exact parent without mutating prior history
- [x] #5 Stale `expected_parent_revision_id` values make no mutation and return bounded current metadata
- [x] #6 Temporary Canvas history stays in memory, displays as temporary state, and joins conversation/message persistence atomically during existing session promotion
- [x] #7 Failed promotion restores the complete session and Canvas state to temporary; unsaved session shutdown destroys staged history
- [x] #8 Conversation soft delete, restore, and hard purge apply the existing lifecycle to owned Canvases without adding Canvas data to sync logs
- [x] #9 Focused migration, repository, property-based branch, race, promotion-rollback, and lifecycle tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md
Reason: this task implements ADR-121’s durable ownership, immutable revision graph, active-branch resolution, and atomic temporary-promotion contract; no new ADR is needed unless implementation changes those accepted boundaries.

1. Allocate migration 66 from verified schema head 65 and implement the immutable Canvas repository with migration, concurrency, rollback, lifecycle, and query-plan tests.
2. Add the scoped Canvas service with active-message-path resolution, historical branching, deterministic conflicts, and centralized durable quotas.
3. Add in-memory temporary Canvas staging and join the existing conversation/message promotion transaction with complete rollback and shutdown destruction.
4. Run only the focused migration, Canvas repository/service/staging, and affected chat persistence suites plus static checks.
5. Update TASK-31227 and ADR-121 with the actual schema, constraints, quotas, query plans, rollback evidence, review outcome, and remaining limitations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented schema migration 65→66 and the Canvas domain across b8eec28f64 through de3c0ba41c. Added conversation-owned canvas_documents, immutable canvas_revisions, local canvas_conversation_hints, payload/ownership/lineage triggers and indexes, typed transactional repository operations, active-branch resolution, exact historical branching, source-free optimistic conflicts, and centralized quotas. Added incarnation-owned in-memory staging and an exact atomic Console promotion participant with leases, immutable native-to-durable message-ID mapping, retry-safe rollback, post-commit retirement, and lifecycle fences for close/restore/recreate/shutdown. Canvas remains local-only and produces no sync-log records under ADR-121.

Verification: focused migration/repository/service/staging and affected chat suites passed (709 passed, 2 deselected known reviewed-head baseline failures). Segmented cleanup diagnostics showed zero descriptor growth in the Canvas repository/service group; combined-gate growth was attributable to unchanged pre-existing transaction/promotion test files (159 + 87 descriptors), while the new Canvas Console tests used 12 and remained below the repository threshold. Delivery-specific Ruff check/format and git diff --check passed. Independent repository, service, and temporary-promotion reviews reported no Critical or Important findings.

Query plans on a fresh schema-66 database use primary-key lookups for conversation/message scope, idx_canvas_documents_conversation plus idx_canvas_revisions_canvas_sequence for reachable listing, and the revision primary key plus uq_canvas_documents_id_conversation for exact reads. No table scans occur; a temporary B-tree is limited to the final bounded list ordering (at most 100 revisions per Canvas).

Relevant files: tldw_chatbook/Canvas/{repository,service,staging}.py, tldw_chatbook/DB/migrations/chachanotes_v65_to_v66_canvas_revisions.sql, tldw_chatbook/Chat/{chat_persistence_service,console_chat_store,console_transaction_contribution}.py, focused Tests/Canvas, Tests/ChaChaNotesDB, and Tests/Chat coverage. ADR: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md.
<!-- SECTION:NOTES:END -->

## Related Design

- `Docs/superpowers/specs/2026-09-03-chatbook-canvas-design.md`
- `Docs/superpowers/plans/2026-09-03-chatbook-canvas-implementation.md`
- `backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md`
