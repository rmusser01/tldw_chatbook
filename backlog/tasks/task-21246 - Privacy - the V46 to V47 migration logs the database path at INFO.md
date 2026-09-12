---
id: TASK-21246
title: >-
  Privacy - the V46 to V47 migration logs the database path at INFO
status: To Do
assignee: []
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

- [ ] No log line at INFO or above emitted by the V46→V47 migration contains a filesystem path that includes the user's home directory or username
- [ ] The migration's log lines still identify which database was migrated well enough to diagnose a failed upgrade
- [ ] The chosen treatment for database paths in migration logs is recorded where the next migration author will see it
- [ ] `python3 scripts/check_persistent_diagnostic_inventory.py` is green and the inventory rows for these two sites reflect the change
- [ ] A test fails if a migration re-introduces an absolute database path at INFO
