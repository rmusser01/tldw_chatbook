# ADR-216: Conversations browse-order index

## Status

Accepted (2026-10-06)

## Context

The Library's conversation browse surfaces page through `conversations` with
`ORDER BY last_modified DESC, id DESC LIMIT ? OFFSET ?`
(`DB/ChaChaNotes_DB.py::search_conversations_page`, `locate_conversation_page`,
`list_all_active_conversations`; queries unchanged by this decision). Every
page query also carries the archive scope clause
(`archived = 0` / `archived = 1`, from
`_conversation_archive_scope_clause`).

The closest pre-existing index, `idx_conversations_archive` (ADR-147, v70→v71),
is `(archived, deleted, last_modified DESC, id DESC)`. Because `deleted` sits
between the equality prefix and the sort keys, that index cannot deliver the
global `(last_modified DESC, id DESC)` order: on a 5k-row fixture,
`EXPLAIN QUERY PLAN` for the default page query showed
`SEARCH conversations USING INDEX idx_conversations_archive (archived=?)`
followed by `USE TEMP B-TREE FOR ORDER BY` — every browse page sorted the
full filtered set (measured ≈3.1 ms/query; the same fixture index-served is
≈0.3–0.6 ms/query). Unscoped browses (`archive_scope="all"`) degraded to a
full table scan plus the same sort.

## Decision

One versioned migration (schema v78 → v79,
`DB/migrations/chachanotes_v79_to_v80_conversations_browse_order_index.sql`)
adds two complementary, non-unique indexes:

1. `idx_conversations_last_modified ON conversations(last_modified DESC, id DESC)`
   — serves unscoped browses and any filter the planner cannot pair with a
   prefix column.
2. `idx_conversations_archived_browse_order ON conversations(archived, last_modified DESC, id DESC)`
   — serves the scoped browses (the dominant path): with `archived` as an
   equality prefix, the remaining index order is exactly the page order, so
   the planner drops the TEMP B-TREE (EXPLAIN-verified for active, archived,
   and workspace-filtered scopes, before and after `ANALYZE`).

The index reaches existing databases through the store's established
pattern: a versioned migration step in the chain
(`_migrate_from_v78_to_v79`, following `_migrate_from_v77_to_v78`'s
SQL-file style), not a schema-script re-run — `_initialize_schema` returns
early when the database is already at the target version, so schema-script
edits alone never reach existing files. Fresh databases gain the indexes by
replaying the same chain step. The index census literal
(`Tests/ChaChaNotesDB/test_index_census.py::EXPECTED_CHACHANOTES_INDEXES`)
is pinned to both indexes in the same change, per that test's contract.

## Alternatives considered

- **Schema-script-only addition** (`CREATE INDEX IF NOT EXISTS` in the v4
  base script): rejected — new databases would bootstrap with the index at
  v4 and then collide with the chain's own migration replay; existing
  databases would never receive it.
- **Single pure browse-order index only** (the review's original minimal
  proposal): EXPLAIN showed SQLite's planner keeps
  `idx_conversations_archive` + TEMP B-TREE for every `archived = ?` scoped
  page query (it prefers the equality-satisfying index and does not model
  LIMIT-aware early termination against the sort cost). The pure index only
  removed the sort for unscoped browses, leaving the default browse — the
  actual hot path — still sorting. The compound archived-prefixed index is
  what flips the planner for the scoped pages; the pure index is kept
  because the compound cannot serve unscoped (`1 = 1`) browses.
- **Wider covering index including further scope columns** (workspace,
  character): rejected — page filters vary per request; EXPLAIN showed the
  planner satisfies those filters with their dedicated single-column
  indexes while the archived-prefixed index still delivers the order, so
  extra prefixes would add write cost without removing the sort.
- **Replacing/reshaping `idx_conversations_archive`**: rejected — that
  shape is ADR-147's decision (its `deleted` middle column serves the
  trash-retains-archived semantics) and is relied on elsewhere; a
  destructive reshape is out of scope for a performance remediation.

## Consequences

- Browse pages across all archive scopes are index-served (no
  `TEMP B-TREE FOR ORDER BY`); measured ≈5–11x faster per page query on the
  5k-row fixture.
- Two additional indexes on a write-heavy table: `conversations` INSERTs and
  `last_modified`-bumping UPDATEs do extra index maintenance. The browse
  path is read-hot and the write delta is two narrow index entries per
  touched row; accepted.
- Other `conversations` query regions (character/flashcard regions owned by
  the sibling remediation PR) may observe new plan choices from the added
  indexes; the migration is purely additive and changes no row semantics,
  but their tests must be watched at merge time.
- Schema version bump recorded: **v78 → v79** in this PR (sibling PR remains
  at v78 as of this writing; merge order determines the final chain numbering
  and conflicts are resolved there).

## Amendment (2026-10-08)

The sibling remediation PR (#3045) landed first and took schema version 79
for ADR-224 (sargable timestamp normalization). Per the landing-order rule
in this ADR's context, the browse-order migration was renumbered to
**v79 → v80** and its catalogs/stamps stacked on the ADR-224 catalogs
(`_browse_order_catalog(_sargable_catalog(...))`).
