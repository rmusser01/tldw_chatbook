---
id: TASK-20975
title: >-
  Hard-deleting a world book raises on its own entries sync emitter
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - bug
  - database
  - sync
  - latent
priority: low
dependencies: []
---

## Description

Source: found while surveying `sync_log` retention for **TASK-19564**.
Re-verified independently at `684c6aba4` against a real schema-v45 database.

`world_book_entries` has no `client_id` column of its own, so its `sync_delete`
trigger reads one from its parent:

```
CREATE TRIGGER world_book_entries_sync_delete
AFTER DELETE ON world_book_entries BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('world_book_entries', CAST(OLD.id AS TEXT), 'delete', datetime('now'),
         (SELECT client_id FROM world_books WHERE id = OLD.world_book_id), 1, ...);
END;
```

(`DB/ChaChaNotes_DB.py:1712-1718`.) The child rows are wired to the parent with
`ON DELETE CASCADE` (`:1520`). So when a `world_books` row is hard-deleted the
cascade fires the child trigger *after* the parent row is already gone, the
subselect yields `NULL`, and the insert violates `sync_log`'s `NOT NULL` on
`client_id`. The whole delete is rolled back.

Measured on a fresh v45 database with one world book and one entry:

| operation | result |
| --- | --- |
| `UPDATE world_books SET deleted=1` (the shipped path) | ok |
| `DELETE FROM world_books WHERE id=?` | `IntegrityError: NOT NULL constraint failed: sync_log.client_id` |

**This is latent, not live.** No shipped path hard-deletes a world book: the
deletion the UI performs is a soft delete, and a repository-wide search finds no
`DELETE FROM world_books` outside the FTS maintenance triggers. The defect is
reachable only by a future caller, a migration, or a maintenance script that
does the obvious thing.

It is worth fixing anyway because of how it will present. The trigger works
correctly for a *direct* entry delete and fails only under cascade, so the bug
is invisible to any test that deletes entries; and it surfaces as a constraint
violation on an unrelated table, which is a poor signpost to a parent-child
trigger ordering problem.

Recorded for accuracy: this is neither caused nor worsened by the `sync_log`
retention work. It reproduces on `dev` with TASK-19564 unmerged.

## Acceptance Criteria

- [ ] Hard-deleting a `world_books` row succeeds and removes its entries
- [ ] The `sync_log` row emitted for each cascaded entry carries a valid
      `client_id`, or the design decides deliberately that no row is emitted for
      a cascaded delete and records why
- [ ] A test covers the cascade path specifically, distinct from a direct
      `world_book_entries` delete, since the direct path already works
- [ ] The same parent-lookup pattern is checked across the other sync emitters
      and either found absent or fixed with this one
- [ ] The chosen fix does not depend on trigger firing order, which SQLite
      leaves undefined for same-kind triggers

## Notes

Filed low because it is unreachable from the shipped UI today. The reason to
fix it rather than note it is that the first caller to hard-delete a world book
will get a `NOT NULL constraint failed: sync_log.client_id` and no obvious
route from that message to a cascade-ordering cause.
