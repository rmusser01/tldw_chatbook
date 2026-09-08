---
id: TASK-31986
title: Declare recovery inventory and side-effect-free profile discovery
status: Done
assignee:
  - '@codex'
created_date: '2026-09-07 23:48'
updated_date: '2026-09-08 03:34'
labels:
  - backup-recovery
dependencies:
  - task-31978
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Discovery identifies selected profiles, custom app-owned storage, shared aliases, and durable unknowns without opening services or modifying data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Discovery identifies selected profiles, custom app-owned storage, shared aliases, and durable unknowns without opening services or modifying data.
- [x] #2 Coverage states and dependency failures accurately distinguish complete, partial, unavailable, excluded, and intentional deletion.
- [x] #3 Every existing persistence producer has an explicit owner-inventory row, and new unclassified producers fail an architecture guard.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of ADR-126; preserve ADR-029/030/036/059/060 boundaries.

1. Establish behavioral RED for unknown durable ownership after importable model skeleton.
2. Census SQLite owners, canonical path resolvers, private file writers and durable roots; record explicit exclusions and unsupported adapter coverage.
3. Extract pure canonical path selection while preserving existing resolver priorities and lexical paths.
4. Implement frozen shared models, owner declarations/registration, read-only selected TOML discovery, identity, dependency and coverage classification.
5. Verify real filesystem multiple profiles, custom paths, aliases, unknowns, missing data, inactive owners, external roots and malformed configs; add producer census guard.
6. Run focused inventory tests, relevant existing guards, scoped lint/format and diff checks; self-review.
7. Record actual evidence and commit scoped files for independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-126 recovery inventory with frozen shared models, installed-owner registry, locally constructed profile context, strict selected TOML discovery, and canonical pure path selectors delegated to by normal config. Discovery identifies custom app-owned databases, shared paths and durable unknowns without config bootstrap, service/database construction, keyring access or filesystem mutation. Unsupported owners, invalid dependencies/sharing, required absence and unvalidated deletion block completeness.

StorageItem adds explicit shared_group and local deletion_validated evidence metadata. DiscoveryContext and storage_logical_id give adapters one profile-scoped owner/dependency convention; TOML cannot inject context or deletion authority. Service-heavy owner resolvers and capture/validation/relocation remain explicitly unqualified. No Complete backup or replacement capability is advertised by this foundation.

The source census records 978 exact producer-symbol/call rows and all 59 SQLite policy IDs. Independent review found writable io.open and aliased os.open bypasses; the first fix exposed shadowed-name/local-import ambiguity. The reviewed second fix retains all recognized opens and cumulative imported aliases, including read-only candidates, instead of inferring Python bindings. Existing row classifications/cohorts and SQLite declarations are preserved; added uncertain candidates remain unsupported. Scoped re-review confirms both findings addressed with no new breakage.

Evidence used Python 3.12.11 from the shared read-only interpreter and only isolated fixture data. Before guard-only fixes, python -m pytest Tests/Backup_Recovery/test_inventory.py Tests/Architecture/test_backup_owner_inventory.py -q passed 35 tests in 7.34s. Final python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q passed 8 tests in 8.33s after behavioral RED with 3 failed / 5 passed. Tests/Architecture/test_profile_owned_path_inventory.py, test_retired_profile_path_owners.py, Tests/test_config_encryption_effective_path.py, Tests/DB/test_private_sqlite_inventory.py and Tests/test_probe_import_provenance.py have 50 distinct passing guard cases after an exact sorting correction and focused rerun. Changed Python Ruff fatal checks, new-module format checks and git diff --check passed. Four pre-existing dependency/AST warnings remain recorded for final review; no full suite ran.

Real filesystem/SQLite and fresh-process tests cover custom/multiple profiles, inactive features, historical profiles, aliases/nested roots, malformed configs, immutable collections, context injection, dependencies, FIFO/link refusal and scope identity changes. Discovery scope is preview evidence; operation options/budgets and fenced capture belong to subsequent service phases.

Files: Backup_Recovery/{models,inventory,profile_paths,owner_registry}.py; config.py; exact canonical path-inventory rules; focused inventory/architecture tests; backlog/docs/backup-recovery-owner-inventory.md. Implementation commits: f804ed5b3, 2017832d8, f4d3cee70. Existing ADR: backlog/decisions/126-complete-local-backup-and-recovery.md. Independent spec/quality review and both scoped fix rounds completed; no new ADR or unrelated changes.
<!-- SECTION:NOTES:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-3)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.
