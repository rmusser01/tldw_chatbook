---
id: TASK-31989
title: Add recovery adapters for core conversation and library stores
status: Done
assignee:
  - '@codex'
created_date: '2026-09-07 23:50'
updated_date: '2026-09-08 06:36'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31986
  - task-31987
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Core durable records and assets survive a real SQLite/WAL capture with stable identities and relationships.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Core durable records and assets survive a real SQLite/WAL capture with stable identities and relationships.
- [x] #2 Source files remain unchanged, custom paths are honored, and missing required dependencies block completeness.
- [x] #3 Schema/relocation policies and registered private backup authority are explicit for every core owner.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of the approved storage and recovery contract; current-only schema qualification, no application schema change.

1. Establish the named declaration regression and behavioral RED after importability.
2. Declare pure installed core store resolvers and explicit profile-scoped dependencies for conversations/messages/characters/notes, prompts, media, collections, and ingestion history.
3. Obtain controller ruling for native-held directional source/staging capture capability; implement fenced registered SQLite snapshots with cancellation and no sidecar copying.
4. Declare exact current schema/FTS/trigger policies and explicit unsupported historical versions; validate domain references and relocate managed paths without runtime constructors.
5. Exercise real domain fixtures, WAL, stable identities, soft deletions, FTS, BLOBs, byte assets, custom paths, missing dependencies, cancellation and source preservation.
6. Update private SQLite and recovery owner inventories; run focused round trips, private SQLite census/interop and recovery inventory guards.
7. Run scoped lint/format and diff checks, self-review, update documentation and implementation notes, and commit scoped task changes. Keep AC unchecked/In Progress for controller review.

Review fix round 1 plan (ADR-126 remains applicable): reproduce failed-constructor retention and repeated peer scans; restrict capture to default factories before construction, reuse operation-local validated peer readers, then run focused covering checks and document evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the five frozen core recovery adapters and exact current SQL policies (ChaChaNotes 42, Media 6, Prompts 4, Collections 1, Ingest Jobs 5), preserving every SQLite record/FTS/BLOB through registered committed-WAL snapshots. Pure discovery honors existing custom selectors and profile-qualified dependencies. Historical/physical schema variants are unsupported; no schema bump or runtime constructor/migration is used. Relative managed locators/BLOB identities survive relocation; external sync/ingest locators stay inert. The auxiliary dependency validator checks actual local cross-store references using only the item's exact declared profile-qualified IDs; server-origin ingestion remains a separate boundary.

ADR: [ADR-126](../decisions/126-complete-local-backup-and-recovery.md). Controller rulings extend the task narrowly with native-issued maintenance capture scopes, a frozen SQL catalog, exact dynamic owner-dispatch guard qualification, and shared-lock reuse of existing fixed admission authority. Capture requires verified bindings, source namespaces plus bootstrap.unbound, exact source read-only authority, disjoint private staging and native handle retirement. Actual process tests exposed and fixed idempotent registration blocking disjoint profiles; established authority now opens without recreation and verifies marker/registry identity under shared locking. No Complete/replacement product flow is exposed.

Changed DB recovery modules/private SQLite policy; admission, storage admission and control records; focused core tests and existing SQLite coverage/inventory guards; both owner inventories, core/startup guidance and an evidence-backed lesson. Existing domain constructor/schema modules required no mutation.

Final targeted evidence (read-only shared Python 3.12.11, private fixtures/cache roots):

- `python -m pytest Tests/Backup_Recovery/test_core_owners.py Tests/Backup_Recovery/test_admission.py Tests/Backup_Recovery/test_bootstrap.py -q`: **122 passed**, including all 59 core tests, no skips.
- `python -m pytest Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_private_sqlite_interop_owners.py -q`: **318 passed, 1 skipped** (pre-existing Windows functional-posture test only).
- `python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q`: **11 passed**.
- Scoped `ruff check --select E9,F63,F7,F82`: clean. Six focused modules pass `ruff format --check`; `git diff --check` clean.

Behavioral RED/GREEN evidence covers declaration, exact schema policies, capture/validation, dependencies/relocation, conflicting native authority, and the real disjoint private-owner process route. Real committed-WAL captures compare complete domain dumps and explicit FTS, BLOB, soft-delete and source-main/WAL preservation evidence. Tests also exercise source/staging direction, alias escapes, cancellation, copied/expired/cross-thread sessions, escaped SQLite handles and capture custom-factory refusal, missing peer/asset dependencies, and lost/corrupt/replaced authority without repair. Exact commands/results and limitations are in the controller task-6 report; core ownership/consumer contracts are documented in [core qualification](../docs/backup-recovery-core-owners.md).

Independent spec and quality review passed after one fix round; all acceptance criteria are verified. Later asset inventories, startup/participant drain, archive/staged migration budgets, activation and product flows remain their existing tasks. No full suite, user data/config/keyring, environment mutation, push, merge or publication was performed.

Review fix round 1: capture now refuses custom factories before either constructor phase or file creation, preserving ordinary owner factory compatibility. Dependency validation scans each referenced installed peer once per invocation and reuses bounded read-only connections, including its own source reader. Real file-backed constructor regressions and nine-reference validation/connection-count evidence are GREEN; the prior capture custom-close branch is superseded by default-factory-only qualification. Hard base-native-close fault injection remains a downstream qualification limit.

Fix-round targeted verification: core **62 passed**; relevant SQLite constructor/copy/close **22 passed**; ordinary custom-factory lifecycle **2 passed**; exact SQLite/recovery inventories **38 passed**. Final focused regressions after fixture-close cleanup **4 passed**. Scoped fatal lint/format and diff checks clean; baseline dependency/AST warnings deferred without environment changes.
<!-- SECTION:NOTES:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-6)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.
