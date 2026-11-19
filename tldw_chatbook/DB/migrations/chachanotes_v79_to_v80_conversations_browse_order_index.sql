-- Conversations browse ordering: pages ask for
-- ORDER BY last_modified DESC, id DESC LIMIT ? OFFSET ?
-- (search_conversations_page / locate_conversation_page /
-- list_all_active_conversations). Without a sort-serving index every page
-- materialized a TEMP B-TREE over the full filtered set just to take the
-- first `limit` rows (EXPLAIN on a 5k-row fixture: 3.1 ms/query with the
-- sort vs 0.3-0.6 ms index-served).
--
-- Two complementary shapes (EXPLAIN-driven, see ADR-216):
-- * idx_conversations_last_modified serves unscoped browses (archive_scope
--   "all") and any filter SQLite can't pair with a prefix column.
-- * idx_conversations_archived_browse_order serves the scoped browses:
--   every page query carries the archive clause (archived = 0 / = 1), and
--   with `archived` as an equality prefix this index yields the exact
--   ORDER BY order. Without it the planner kept choosing
--   idx_conversations_archive(archived, deleted, last_modified DESC,
--   id DESC) plus a TEMP B-TREE sort, because that index's `deleted`
--   column sits between the equality and the sort keys.
CREATE INDEX idx_conversations_last_modified
  ON conversations(last_modified DESC, id DESC);
CREATE INDEX idx_conversations_archived_browse_order
  ON conversations(archived, last_modified DESC, id DESC);
