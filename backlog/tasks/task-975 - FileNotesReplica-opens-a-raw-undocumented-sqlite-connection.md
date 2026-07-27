---
id: TASK-975
title: >-
  FileNotesReplica opens a raw sqlite connection that is neither registered nor
  documented
status: To Do
assignee: []
created_date: '2026-07-27 20:00'
labels:
  - bug
  - db
  - security
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`FileNotesReplica.__init__` (`tldw_chatbook/Notes/file_notes_replica.py:40`) calls `sqlite3.connect(...)` directly rather than going through `connect_private_sqlite`. So this database is opened outside every protection the app applies to all its others: the private-path guards (ownership, no shared-writable ancestor, no untrusted symlink in the path) and private file modes.

`Tests/DB/test_private_sqlite_inventory.py::test_raw_connection_census_is_qualified_and_transition_aware` fails on `origin/dev` because of it — reproduced on a clean detached checkout of `a73b9b46f` with no other changes:

```
Left contains 1 more item:
{('tldw_chatbook/Notes/file_notes_replica', 'FileNotesReplica.__init__'): 1}
```

That census exists precisely to stop a raw connection site being added silently, and it is doing its job. The inventory at `backlog/docs/sqlite-private-owner-inventory.md` has no row for this call.

Two outcomes are legitimate and the choice needs this module's context:

- **Route it through the seam.** Register an owner in `SQLITE_OWNER_REGISTRY`, add a `C`-row to the inventory, and extend the ID range assertion in `test_inventory_has_stable_unique_connection_and_backup_ids`. This is what `ensure_site_configs_schema` did (row C36) when the same census caught it.
- **Document it as a deliberate exclusion.** The inventory carries `X`-rows for sites that legitimately stay raw. If the replica's connection is genuinely outside the private-database contract, say why in an `X`-row.

What is not an option is leaving the census red, because the next person to add a connection site will find the guard already failing and learn to ignore it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A decision is recorded — route through `connect_private_sqlite`, or document as an exclusion — with the reasoning stated
- [ ] #2 `Tests/DB/test_private_sqlite_inventory.py` passes on a clean `dev` checkout
- [ ] #3 If routed: an owner is registered and the inventory carries a `C`-row, with the ID range assertion extended
- [ ] #4 If excluded: the `X`-row states why this database is outside the private-path contract
- [ ] #5 File-notes replica behaviour is unchanged — existing replica tests pass
<!-- AC:END -->
