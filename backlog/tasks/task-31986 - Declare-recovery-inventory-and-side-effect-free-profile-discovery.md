---
id: TASK-31986
title: Declare recovery inventory and side-effect-free profile discovery
status: In Progress
assignee:
  - "@codex"
created_date: '2026-09-07 23:48'
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
- [ ] #1 Discovery identifies selected profiles, custom app-owned storage, shared aliases, and durable unknowns without opening services or modifying data.
- [ ] #2 Coverage states and dependency failures accurately distinguish complete, partial, unavailable, excluded, and intentional deletion.
- [ ] #3 Every existing persistence producer has an explicit owner-inventory row, and new unclassified producers fail an architecture guard.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-3)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Plan

ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of ADR-126; preserve ADR-029/030/036/059/060 boundaries.

1. Establish behavioral RED for unknown durable ownership after importable model skeleton.
2. Census SQLite owners, canonical path resolvers, private file writers and durable roots; record explicit exclusions and unsupported adapter coverage.
3. Extract pure canonical path selection while preserving existing resolver priorities and lexical paths.
4. Implement frozen shared models, owner declarations/registration, read-only selected TOML discovery, identity, dependency and coverage classification.
5. Verify real filesystem multiple profiles, custom paths, aliases, unknowns, missing data, inactive owners, external roots and malformed configs; add producer census guard.
6. Run focused inventory tests, relevant existing guards, scoped lint/format and diff checks; self-review.
7. Record actual evidence and commit scoped files; leave criteria unchecked and In Progress for controller review.

## Implementation Notes

Implemented the ADR-126 inventory foundation with frozen owner/schema/context models,
pure canonical config/database selectors, installed-owner registration, strict selected
TOML discovery, physical alias/root/dependency classification, and a checked census of
876 exact producer-symbol/call rows plus all 59 SQLite policy owners. Discovery does
not bootstrap config, instantiate services, open SQLite, read keyrings, or create
profiles. Custom database paths remain baseline app-owned content; unresolved owner
cohorts and unknown durable files explicitly block completeness.

Controller rulings preserved: non-DB service-heavy resolver paths remain unsupported
until their owner adapters extract canonical pure seams; no backup-local filename
authorities were invented. StorageItem adds shared_group and deletion_validated local
evidence fields; source discovery rejects unqualified deletion evidence. A frozen
DiscoveryContext under a reserved locally inserted mapping key supplies profile
selectors, and storage_logical_id provides explicit owner/dependency IDs. Imported
TOML cannot inject this context. These are foundation interfaces, not qualified
capture/validation/relocation or Complete-backup/replacement capabilities.

Verification used only `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`
(3.12.11) read-only, from the independent execution clone. Commands:

- `-m pytest Tests/Backup_Recovery/test_inventory.py Tests/Architecture/test_backup_owner_inventory.py -q`: 35 passed, 4 existing warnings, 7.34s; real SQLite/files/subprocesses, no required skips.
- `-m pytest Tests/Architecture/test_backup_owner_inventory.py -q`: 3 passed, 4 existing warnings, 6.67s.
- `-m pytest Tests/Architecture/test_profile_owned_path_inventory.py Tests/Architecture/test_retired_profile_path_owners.py Tests/test_config_encryption_effective_path.py Tests/DB/test_private_sqlite_inventory.py Tests/test_probe_import_provenance.py -q`: 49 passed and one ordering failure in moved resolver inventory entries; sorted entries and reran the failed guard with inventory tests, 31 passed in 0.88s. All 50 distinct affected guard cases have passing evidence.
- Changed Python: `ruff check --select E9,F63,F7,F82` passed; six new focused Python modules passed `ruff format --check`; `git diff --check` passed.

Behavioral RED/GREEN covered unknown ownership (1 failed then 1 passed), external
folder default exclusion, historical default profile detection, immutable tuples,
linked-parent non-traversal, non-regular payloads, alias-retarget scope changes and
preservation of explicit cross-owner shared declarations. Existing requests dependency
and source invalid-escape warnings remain; no full suite, shared environment mutation,
user data/keyring access, push or merge occurred. Self-review replaced quadratic root
pair comparison with ancestor-set membership and retained explicit source identity
checks. No new architecture decision was needed beyond approved ADR-126.

Changed files: Backup_Recovery/{models,inventory,profile_paths,owner_registry}.py,
config.py, two focused test modules, the owner census documentation, and the exact
canonical profile-path inventory entries required by resolver extraction.

Status and ACs intentionally remain In Progress/unchecked pending controller review.

### Review fix: producer open signatures

Independent review found that module-qualified and aliased writable opens could
evade the AST census because their filename was interpreted as a Path.open mode.
The scanner now resolves import identities before selecting a known signature and
retains ambiguous calls, dynamic modes and expanded arguments conservatively.
Synthetic regression controls cover builtin/io/os aliases and Path forms. The
reviewed census grows from 840 to 876 rows, with four existing call counts increased;
new ambiguous candidates retain unsupported coverage. Runtime production code is
unchanged. `python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q` produced
behavioral RED (2 failed, 4 passed in 6.30s), then GREEN after the fix/census refresh
(6 passed, 4 existing warnings in 5.77s). Scoped Ruff E9/F63/F7/F82 and format checks
passed; git diff --check passed. The read-only Python 3.12.11 interpreter path above
was used. Exact evidence is appended to the execution report; status and ACs remain
In Progress/unchecked for independent re-review.
