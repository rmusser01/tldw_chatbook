---
id: TASK-33132
title: SQLite store kernel ADR and connection-transaction consolidation
status: In Progress
assignee:
  - '@robert'
created_date: '2026-09-27 19:46'
labels: [refactor, db]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The DB layer carries 21 independent transaction() context managers and 9 verbatim copies of the held thread-local connection plus liveness-ping idiom, because base_db.py provides almost none of what a store actually needs. The copies know they are copies: RAG_Indexing_DB.py documents copying the Workspace_DB idiom, AgentRuns_DB.py says it follows the Workspace_DB pattern, and three classes carry close() aliases to feel like BaseDB without inheriting it. One kernel in base_db (held connection with liveness ping, transaction with defined nesting semantics, column-exists helpers) makes the base class real and deletes roughly 2k+ LOC of copied plumbing. ChaChaNotes' richer transaction semantics (depth tracking, quiescence) become the canonical implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An ADR records the kernel decision with ChaChaNotes transaction semantics as canonical.
- [ ] #2 The held connection plus liveness-ping idiom is implemented once and consumed by every store that currently copies it.
- [ ] #3 Transaction nesting semantics are documented and covered by tests.
- [ ] #4 No store retains a private copy of the consolidated machinery.
- [ ] #5 Targeted DB test suites for every touched store pass.
<!-- AC:END -->

## Renumbering provenance

Filed 2026-09-27 as task-33087. origin/dev minted its own task-33087 before this branch merged, so per the landed-keeps-id rule this task moved to task-33132; every inbound reference moved with it.

## Implementation Plan

Premise re-verified against origin/dev (2026-10-12): 25 `transaction()` definitions (growing: 18→21→25), 10 held-connection copies. Two design-relevant discoveries: `Backup_Recovery/participants.py` already coordinates stores via `_core_*` hooks/decorators (the kernel must keep it observable by construction), and `RAG_Indexing_DB` is the hardened, best-documented reference port of the Workspace pattern (task-15465/15466/19566/3012/32801.5 lineage) — it becomes the pilot and the comment donor.

1. ADR-228 (backlog/decisions/228-sqlite-store-kernel.md): kernel = `DB/store_kernel.py` mixin; canonical semantics from the hardened Workspace/RAG pattern (autocommit held connections, one-place PRAGMA configuration, BEGIN IMMEDIATE with BaseException rollback, fail-loud nesting, participants integration built in, per-store `:memory:` overrides). AC #1's original wording ("ChaChaNotes semantics canonical") is refined there: the fleet's de facto canonical is the fail-loud variant; ChaChaNotes' depth-tracked manager stays the documented exception.
2. Implement `DB/store_kernel.py` with tests covering nesting failure, BaseException rollback, liveness replacement, and participants observability.
3. Pilot-migrate RAG_Indexing_DB onto the kernel (comments move verbatim where they describe kernel behavior); run its test suite.
4. Migrate remaining held-connection copyists one store per PR (Workspace, AgentRuns, Prompts, Client_Media, Library_Collections, event-state, client-notifications, notes-device, receipts...), 25 transaction() definitions shrinking accordingly.
5. Stores with genuinely different contracts (Chunking_Lab durability, Client_Media blob layer) may consume the kernel without its policy — documented per store.



Migration 3 (2026-10-14): AgentRuns_DB onto the kernel (BaseDB provenance, busy_timeout-first policy, no transaction override — kernel immediate default matches). Same PR lands the tri-state _transaction_immediate_default decision from the workspace-PR review and simplifies Workspace to the class attribute. Direct battery 125/125.


Migration 4 (2026-10-16): Library_Collections_DB onto the kernel. Deltas preserved as overrides: the WAL-retry helper (_enable_wal, concurrent-opener tolerance), the guarded read_transaction (BEGIN DEFERRED snapshot + total_changes write-guard), and the busy-safe PRAGMA policy; the standard transaction()/close()/held-connection machinery deleted (kernel default BEGIN IMMEDIATE matches the store's cross-process MCP rationale, noted at the owner attribute). Battery: collections capture migration + offline store + capture extraction/app wiring + kernel + pragma suites 135/135.
