---
id: TASK-22206
title: >-
  Conversation resume: replace the O(N-squared) per-node message-tree build
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
labels:
  - performance
  - chat
  - database
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22206).

`Chat/chat_conversation_service.py:1136-1184`: `_build_message_tree` recurses once per
message and issues one `get_messages_for_conversation_by_parent_ids` per node
(`DB/ChaChaNotes_DB.py:9684-9716`). With `sqlite_stat1` absent (no production DB has it)
the plan uses `idx_msgs_conv_ts` and post-filters — every per-node call scans all N rows,
hydrating `content` and the `image_data` BLOB. Python adds a per-node `set(seen)` copy.
Measured in-memory with 1 KB blobs: 3.0 ms @100 msgs, 22.7 @300, 89.8 @600 (clean N^2);
the same walk with `idx_msgs_parent` forced: 2.0 ms @600 — 45x, O(N). The walk is awaited
inline on the loop (`UI/Console_Modules/workspace.py:3679`, `:3706`) on every saved-
conversation resume/session restore. Also: recursion depth equals conversation length —
a ~980-message linear conversation raises RecursionError on resume.

## Acceptance Criteria

- [x] Resume performs O(N) total work: either one conversation-scoped query with in-memory tree assembly, or a per-parent query shape proven by EXPLAIN QUERY PLAN with sqlite_stat1 ABSENT to use `idx_msgs_parent`
- [x] BLOB columns are not hydrated during tree construction
- [x] A 2000-message linear conversation resumes without RecursionError (iterative build or explicit bound with a graceful path)
- [x] Resume time measured before/after at 600+ messages; the walk runs off the event loop or its on-loop time is bounded and stated

## Implementation Plan

1. Caller census (done before writing code): `get_messages_for_conversation_by_parent_ids`
   is called in production ONLY by `_build_message_tree`; direct DB-level tests
   (`test_chat_conversation_parity.py`, `test_provider_continuation.py`) call it and are
   left untouched — the DB method's semantics do not change. The tree's only production
   consumers are `console_conversation_hydration.console_messages_from_conversation_tree`
   (reads content/image_data/usage_json/metadata_json/provider_continuation_json/
   assistant_generation_state/id/role/sender/children) and the scope/server services
   (pass-through). `image_data` IS consumed on resume (position-0 legacy attachment), so
   the tree contract keeps carrying it — it just must not be hydrated N times during the
   build.
2. New narrow DB read `get_message_tree_rows_for_conversation(conversation_id, *,
   order_by_timestamp, include_deleted_conversation=False)`: ONE conversation-scoped
   query, all columns the old per-parent query selected EXCEPT the `image_data` BLOB,
   plus `(m.image_data IS NOT NULL) AS has_image`, ordered by `m.timestamp`. EXPLAIN
   QUERY PLAN recorded on a schema-only scratch DB with `sqlite_stat1` ABSENT (repo rule
   from TASK-21126) — must drive off `idx_msgs_conv_ts (conversation_id, timestamp)`
   with no post-filter scan and no temp B-tree.
3. New batched BLOB fetch `get_message_images_by_ids(message_ids)` (chunked IN(...) at
   500 ids, mirroring `get_attachments_for_messages`) used AFTER the build, once, only
   for nodes whose row said `has_image` — zero BLOB reads for imageless conversations.
4. Rewrite `ChatConversationService.get_conversation_tree`/`_build_message_tree` to:
   fetch once, partition rows into a children-by-parent map preserving SQL order
   (per-parent order == the old per-parent query's timestamp order), page roots in
   memory (offset/limit slice of the parent-IS-NULL rows), assemble iteratively with an
   explicit stack — preserving depth_cap truncation (`depth >= depth_cap` ⇒
   `children=[]`, `truncated=True`), the seen-id cycle guard (re-encounter ⇒ truncated),
   the normalize-None row-drop (drops its subtree), and id-None rows getting
   `children=[]`. `total_root_threads` computed from the same fetched rows (identical
   predicate to the old COUNT query).
5. The resume path recurses once more in `console_messages_from_conversation_tree._walk`
   (depth == chain length ⇒ same RecursionError at ~1000): convert that walk to an
   explicit stack too, preserving pre-order.
6. Red-first tests (Tests/Chat/test_chat_conversation_service_tree_perf.py):
   (a) 2000-message linear chain through `get_conversation_tree` +
   `console_messages_from_conversation_tree` — red today with RecursionError;
   (b) equivalence: new build output == the OLD recursive algorithm (ported verbatim
   into the test as the reference oracle, running against the same real DB) on a
   branched fixture — siblings order, parents, roots, deep branch, images;
   (c) query-count probe via `sqlite3` trace callback: O(1) queries per
   `get_conversation_tree` regardless of N (and that adding an image adds exactly one
   batched query, not N).
7. Update the FakeDB in `Tests/Chat/test_chat_conversation_service.py` to serve the new
   one-shot read so `test_get_conversation_tree_wraps_root_and_child_rows` exercises the
   new seam.
8. Measure resume-build time before/after at 600 and 2000 messages (temp-file DB, 1 KB
   blobs); decide inline-vs-worker from the measured number (AC allows "bounded and
   stated").
9. Mutation test: reintroduce a per-node query into the build, confirm probe (c) reds.
10. Targeted suites + `--collect-only` sweep, tee everything; `./scripts/preflight.sh`;
    notes, ACs, Done; commit + push (no PR).

## Implementation Notes

The resume tree build is now ONE conversation-scoped, BLOB-free query plus an
in-memory iterative assembly, and the resume flattener no longer recurses.
Measured on a temp-file DB, linear chain, **1 KB image blob on every message**
(median of repeats; `.taskwork/bench_output.txt` methodology mirrors the
review's):

| N | before (legacy recursive build) | after (new build) | after (build + resume flatten) |
|---|---|---|---|
| 600 | 110.2 ms | **5.2 ms** (21x) | 9.0 ms |
| 2000 | **RecursionError** at the production recursion limit; 1892.4 ms with the limit raised to 50k | **21.0 ms** (90x) | 32.1 ms |

**Core changes**

- `DB/ChaChaNotes_DB.py`: new `get_message_tree_rows_for_conversation` — one
  query, all live rows of the conversation, the same columns as the old
  per-parent query EXCEPT `image_data`, which becomes a
  `(m.image_data IS NOT NULL) AS has_image` flag; ordered by `m.timestamp` so
  a stable partition reproduces each parent's child order. New
  `get_message_images_by_ids` — batched (500-id chunks, mirroring
  `get_attachments_for_messages`) fetch of the position-0 image columns for
  exactly the ids that need them. `get_messages_for_conversation_by_parent_ids`
  is UNCHANGED (its remaining callers are direct DB-level tests; production no
  longer calls it).
- `Chat/chat_conversation_service.py`: `get_conversation_tree` fetches once,
  partitions rows into a children-by-parent map, pages roots in memory
  (`total_root_threads` computed from the same fetch — identical predicate to
  the old COUNT query), and `_build_message_tree` assembles the nested nodes
  with an explicit stack: no recursion, no per-node queries, no per-node
  `set(seen)` copies (the visited set is global; it differs from the old
  per-path copy only on inputs a real DB cannot produce — a duplicated primary
  key). Depth-cap truncation, the seen-id truncation flag, normalize-rejected
  row subtree drops, id-less rows, and SQL LIMIT/OFFSET edge semantics
  (negative limit = unbounded, negative offset = 0) are all preserved.
  `_hydrate_tree_images` then fills `image_data` once, batched, only for
  `has_image` rows — an imageless conversation performs ZERO BLOB reads
  (trace-probe-asserted).
- `Chat/console_conversation_hydration.py`:
  `console_messages_from_conversation_tree`'s per-node recursive `_walk` was
  the SECOND RecursionError in the same resume (depth == chain length);
  it is now an explicit pre-order stack. Fixing only the service build would
  have left the 2000-message resume red — the red-first test drives the full
  path and caught it.

**EXPLAIN QUERY PLAN** (schema-only scratch DB, `sqlite_stat1` ABSENT —
`.taskwork/explain_probe.py`; repo rule from TASK-21126):

```
-- get_message_tree_rows_for_conversation (ASC and DESC identical) --
SEARCH c USING INDEX sqlite_autoindex_conversations_1 (id=?)
SEARCH m USING INDEX idx_msgs_conv_ts (conversation_id=?)
-- get_message_images_by_ids (IN chunk) --
SEARCH messages USING INDEX sqlite_autoindex_messages_1 (id=?)
```

One index-driven range scan over exactly the conversation's rows, no temp
B-tree in either direction, PK point lookups for images. (The legacy
per-parent query showed the same `idx_msgs_conv_ts (conversation_id=?)`
search — i.e. a full-conversation scan with post-filter — PER NODE, which is
what made the old walk O(N^2).)

**Event-loop decision (AC 4, "bounded and stated")**: the build stays inline
on the loop via `chat_conversation_scope_service`. On-loop cost is now linear
and small: 5.2 ms @600, 21.0 ms @2000 with a 1 KB blob on every message
(32.1 ms including the flatten) — under the repo's 100 ms worker threshold up
to roughly 6,000+ messages. The old build exceeded that threshold at ~600
messages.

**Tests** (`Tests/Chat/test_chat_conversation_service_tree_build.py`, all
red-first where the defect allowed):

- 2000-message linear chain through `get_conversation_tree` +
  `console_messages_from_conversation_tree` — was `RecursionError`, now green.
- Equivalence: the legacy recursive algorithm is ported verbatim into the test
  as the oracle; new output must equal it exactly against the same real SQLite
  file — branched fixture (two roots, siblings, image, 10-deep branch), plus
  DESC / root pagination / depth-cap variants, plus structural spot-checks so
  equality cannot pass on two empty trees.
- Query-count probes (`sqlite3` trace callback): statement count identical
  @40 vs @120 messages (was 45 vs 125 — linear); image hydration is exactly one
  batched statement, present only when the conversation has images.
- `Tests/Chat/test_chat_conversation_service.py` FakeDB updated to serve the
  new one-shot read (`tree_rows`/`images_by_message_id`); the wraps-root-and-
  child-rows assertions are unchanged.

**Mutation test**: reintroducing a per-node
`get_messages_for_conversation_by_parent_ids` call inside the new build turned
both probes red (44 vs 124 statements; BLOB probe red) — then the mutation was
reverted (Edit-based restore). Verified green after restore.

**Verification**: 188 passed / 0 failed across the affected suites
(conversation service + tree build + hydration + scope/server service + video
message + generation store + ChaChaNotes parity + provider continuation +
resume-active-path + launch wake; `.taskwork/final_run.txt`). Whole-suite
`--collect-only`: 58,548 collected; the 28 collection errors are all
pre-existing optional-dependency modules (numpy/audio/TTS/transcription/web-
scraping), none in touched areas. `./scripts/preflight.sh` all green.

**Pre-existing failures encountered (baselined at merge-base 983aa5878 with
the same venv — identical failures there, NOT introduced by this task)**:
6 integration e2e tests (`test_console_branching_e2e` etc.) fail at the FIRST
`submit_draft` with "Provider destination is incomplete";
`test_console_chat_controller.py::test_provider_switch_ignores_unrelated_completed_continuation_history`
passes but its teardown reds on attempted tiktoken-encoding download (network
guard); `test_console_workspace_dead_rows.py::test_failed_resume_marks_row_broken_with_honest_single_toast`
(ghost row never renders); 11 `test_console_native_chat_flow.py` failures
(incl. two resume tests failing on drifted inspector copy — the resume itself
succeeds there).
