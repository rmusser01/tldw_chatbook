---
id: TASK-21246
title: >-
  Privacy - the V46 to V47 migration logs the database path at INFO
status: Done
assignee: ['@claude']
created_date: '2026-08-23'
labels:
  - privacy
  - security
  - diagnostics
  - database
dependencies: []
priority: high
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down. Surfaced during the
TASK-21116 fix round, while the diagnostic-inventory rows added by TASK-21100 (this
burn-down's own merged work) were being reviewed row by row.

`DB/ChaChaNotes_DB.py`'s `_migrate_from_v46_to_v47` interpolates `self.db_path_str` — an
absolute local filesystem path, which on a default profile contains the operating-system
username — into two **INFO**-level log lines:

- `:6068` `"Migrating schema from V46 to V47 for '{...}' in DB: {self.db_path_str}..."`
- `:6103` `"[... V46→V47] Migration completed successfully for DB: {self.db_path_str}"`

Verified present on dev `b2b1e2e0d`.

This is consistent with the surrounding file, which has 353 such interpolations — and that is
the point. The pattern was invisible until TASK-21100's new rows forced a row-by-row review of
the inventory and pinned these two. A migration runs on the **first boot after an upgrade**,
at the default log level, for **every** user, so this is the highest-traffic instance of the
pattern in the file. The repo's own rule is that user data never reaches the log, and a home
directory path naming the user is user data.

The fix is a scoped privacy repair, not a rewrite of 353 call sites: decide the treatment for
a database path in a log line (omit it, reduce it to the file name, or demote to DEBUG), apply
it to the V46→V47 pair, and record the decision so the next migration inherits it rather than
re-deriving it.

## Acceptance Criteria

- [x] No log line at INFO or above emitted by the V46→V47 migration contains a filesystem path that includes the user's home directory or username
- [x] The migration's log lines still identify which database was migrated well enough to diagnose a failed upgrade
- [x] The chosen treatment for database paths in migration logs is recorded where the next migration author will see it
- [x] `python3 scripts/check_persistent_diagnostic_inventory.py` is green and the inventory rows for these two sites reflect the change
- [x] A test fails if a migration re-introduces an absolute database path at INFO

## Implementation Plan

Premise re-verified at branch base `cddc89d3e7` (2026-10-06): **STALE for
the code fix**. Commit `d617cbfb13` (PR #2190, 2026-08-28, "fix(diagnostics):
prevent user path disclosure in live logs", TASK-19864's batch — five days
after this task was filed) already replaced both V46→V47 INFO lines'
`self.db_path_str` with `db_sha256={self._db_diagnostic_ref}` (a
`content_fingerprint` of the path: stable per database, no path disclosure),
repo-wide (102 uses in ChaChaNotes_DB.py at base), and shipped the enforcing
scanner. Remaining gap found: AC3's record — the treatment existed only in
the PR #2190 design spec and the scanner's rules, not in the migrations
authoring guide a migration author reads.

1. Verify each AC's state at base with evidence (source read, scanner probe,
   inventory run).
2. AC3: record the treatment in `tldw_chatbook/DB/migrations/README.md`'s
   "Adding a migration" list (new step 3: `db_sha256=<fingerprint>`, never
   the path; exceptions as `exception_type=`; pointer to the enforcing
   scanner), renumbering the later steps.
3. AC4: regenerate `Docs/security/production-diagnostic-inventory.json` —
   green at base for the V46→V47 sites; the only drift on this branch is
   TASK-407's one new reviewed diagnostic call in
   `Client_Media_DB_v2.fetch_content_prefixes_for_media_batch` (mirrors the
   adjacent `fetch_keywords_for_media_batch` error shape; scanner accepts,
   no path/exception-taint candidate).

## Implementation Notes

No migration code changed — the V46→V47 privacy repair itself landed in
`d617cbfb13` (PR #2190) before this branch; this task closes the authoring
record (AC3) and verifies the rest.

Evidence (venv 3.12.13):

- AC1/AC2: `ChaChaNotes_DB.py` `_migrate_from_v46_to_v47` at base logs
  `"Migrating schema from V46 to V47 ... db_sha256={self._db_diagnostic_ref}"`
  and `"[...] Migration completed successfully for DB:
  db_sha256={self._db_diagnostic_ref}."` (lines 6529-6531, 6564-6567); the
  failure path logs `exception_type={type(exc).__name__}` only. Path-free,
  still identifies the database (stable fingerprint).
- AC5 probe: `scan_path_diagnostic_candidates` over a synthetic
  `_migrate_from_v99_to_v100` flags exactly the two raw
  `self.db_path_str` INFO lines as path-privacy candidates and passes the
  two `db_sha256=` lines; a new candidate fails
  `scripts/check_persistent_diagnostic_inventory.py` (preflight +
  derived-artifacts) and
  `Tests/Architecture/test_persistent_diagnostic_inventory.py::test_
  production_diagnostic_inventory_and_sink_topology_are_unchanged`.
- AC4: check green at base for these sites; on this branch regenerated for
  TASK-407's +1 (Client_Media_DB_v2.py 342→343 calls, one row) —
  `python scripts/check_persistent_diagnostic_inventory.py` →
  "diagnostic inventory verified: 655 owners, 1445 TASK-492 calls, 56
  TASK-31551 calls, 7627 TASK-494 calls, 16 sink files", exit 0.
- Suites: `Tests/Architecture/test_persistent_diagnostic_inventory.py`
  (inventory pin) + `Tests/Architecture/test_diagnostic_path_privacy.py`
  → 99 passed.

ADR required: no — documentation-only completion of an already-shipped
privacy treatment; the governing design record is
`Docs/superpowers/specs/2026-08-28-diagnostic-path-privacy-and-guard-design.md`
(PR #2190), linked here rather than duplicated. Files:
`tldw_chatbook/DB/migrations/README.md` (step 3 added, steps renumbered),
`Docs/security/production-diagnostic-inventory.json` (TASK-407's row).
