---
id: TASK-22501
title: >-
  add_conversation is a DEFERRED writer and dies un-retried under any concurrent chunk commit
status: Done
assignee: [rmusser01]
created_date: '2026-08-26'
labels:
  - database
  - correctness
priority: high
dependencies: []
---

## Description

Source: close-out of the 2026-08-24 holistic performance review's burn-down (29 tasks,
TASK-22200..22228, all merged 2026-08-25/26). Evidence: `Docs/Design/2026-08-24-holistic-perf-review.md` plus the originating task's
Implementation Notes.

Found by TASK-22200's adversarial reviewer, reproduced 3/3: `add_conversation`
(`DB/ChaChaNotes_DB.py`, plain `self.transaction()`) is a read-then-write DEFERRED
transaction, so when a paced backfill chunk commits it takes the snapshot-upgrade
`database is locked` INSTANTLY — bypassing the busy handler entirely — and the UI layer does
not retry. `add_message` (IMMEDIATE) survived the same load 10/10.

This refutes one sentence of TASK-22200's own description ("every UI write is also BEGIN
IMMEDIATE"), corrected in that task file. TASK-22200's pacing shrinks the collision window
~10x but cannot close it, and TASK-22215 added a second paced backfill on a second database.

## Acceptance Criteria

- [x] `add_conversation` uses an IMMEDIATE transaction (or an equivalent retry) and survives a concurrent chunk-commit load probe 10/10
- [x] The repo is swept for sibling read-then-write DEFERRED writers on user-facing paths; each is fixed or explicitly recorded as safe
- [x] A regression test reproduces the original failure shape (concurrent committing writer + a deferred read-then-write) and reds without the fix

## Implementation Plan

1. Write the regression test first, on the base SHA, following the repo's
   existing concurrency-probe idiom (`Tests/DB/test_chachanotes_fts_backfill_pacing.py`'s
   `test_ui_write_latency_stays_bounded_while_a_backfill_is_in_flight`: real temp-file WAL DB,
   one shared `CharactersRAGDB` with thread-local connections, the real paced backfill driver on
   a worker thread, the real `add_conversation` from the foreground, with in-flight assertions so
   the probe cannot pass vacuously). Verify it is RED on the base.
2. Fix `add_conversation` to `self.transaction(immediate=True)`, with a comment in the
   task-21100 house style. Surgical edit only — migrations/trigger sections are untouched.
3. Verify the probe is GREEN 10/10 consecutive runs.
4. Sweep the repo for sibling read-then-write DEFERRED writers on user-facing paths
   (every `self.transaction()`/`with ... .transaction()` body in `tldw_chatbook/DB/` that both
   reads and writes); fix the ones on user-facing paths, record the full sweep table
   (fixed / safe + reason) in Implementation Notes.
5. Closeout: tick ACs, Implementation Notes with commands + results, `status: Done`, one commit.

ADR required: no
Reason: transaction-mode policy on existing writers was decided by TASK-21100 (immediate=True
precedent, `41a240ccd`); this task applies that standing policy to the enumerated remaining
writers and records the sweep — no new architectural decision, storage change, or interface
change is made.

## Implementation Notes

### Mechanism finding (changes the task's premise, not its fix)

Before writing the fix, the failure boundary was established empirically on real SQLite
3.49.1 (two connections on a WAL temp-file DB, busy_timeout=15000, a committing thread at
~1.4 ms/chunk, 200 writer attempts per trial):

- `BEGIN; SELECT; INSERT` (deferred read-then-write): **21/29/30 instant deaths per trial** —
  the non-retryable snapshot-upgrade SQLITE_BUSY, exactly as reported.
- `BEGIN; INSERT` (deferred blind write, incl. FK parent lookups inside the statement):
  **0/600 deaths** — plain SQLITE_BUSY, which the busy timeout retries.
- `BEGIN IMMEDIATE; ...`: **0/600 deaths**.

Consequence: depth-0 `add_conversation` (a single blind INSERT; verified unchanged at the
review-era SHA `8b61c82ed3`) cannot take the instant-busy by itself — a threaded load probe
of the real method against the real paced backfill passed 10/10 on the UNFIXED base (both
one-shared-instance and two-instance forms). The 3/3 reviewer reproduction is reproducible
only through the *composed* shape: an outer DEFERRED wrapper whose first statement is a read
(`add_message`'s conversation-existence SELECT), which neutralizes every inner writer's
IMMEDIATE (the manager honours `immediate` only at depth 0) and opens the exact
nested-composition window TASK-21100's review fixed for `ChatPersistenceService` — and which
that fix missed on two conversation-path sites. The deterministic red test drives one of
those real sites. The `add_conversation` conversion remains the right fix (mandated; makes
the depth-0 default lock-reserving against future body changes), and the probe holds it to
10/10.

### What shipped

1. `DB/ChaChaNotes_DB.py` — four conversation-writer conversions to
   `transaction(immediate=True)` (task-21100 comment style): `add_conversation` (the task's
   subject), and the three sibling read-then-write conversation CRUD writers on the same
   user-facing chat path: `update_conversation`, `soft_delete_conversation`,
   `restore_conversation` (each reads version/deleted state then UPDATEs inside one
   DEFERRED begin).
2. `Chat/chat_conversation_service.py` — `copy_conversation_active_path`'s outer unit
   (first statement: `add_message`'s read) converted to IMMEDIATE. This is the site the
   deterministic red test kills on the base.
3. `Character_Chat/Character_Chat_Lib.py` — the chat-import wrapper (the `with
   db.transaction():` around the `add_message` loop in
   `load_chat_history_from_file_and_save_to_db`) converted to IMMEDIATE; same
   read-first shape as the fork path.
4. `Tests/DB/test_chachanotes_conversation_writer_collision.py` (new): the load probe, the
   deterministic interleave regression, and structural source pins for `add_conversation`
   and the fork wrapper.

### Red -> green evidence

- RED on base (`ef831d9f38`), deterministic, 3/3 runs:
  `test_conversation_fork_survives_a_chunk_commit_inside_the_read_to_write_gap` died with
  `sqlite3.OperationalError: database is locked` raised out of the fork's outer DEFERRED
  transaction (whole unit rolled back, un-retried) after a real one-chunk backfill commit
  landed inside the read->write gap (trace callback commits a real chunk from a second
  instance at the wrapper's first INSERT).
- GREEN after the fix: all 3 tests in the new module pass; the interleave's chunk outcome
  flips to `blocked:` (the writer now holds the write lock from its BEGIN; the chunk queues
  on the shortened busy timeout instead).
- Load probe 10/10 (post-fix, consecutive runs):
  `pytest Tests/DB/test_chachanotes_conversation_writer_collision.py::test_add_conversation_survives_the_concurrent_chunk_commit_load_probe`
  => `PROBE_GREEN=10/10` (240 rows / chunk 8 / 0.05 s pause, 10 foreground
  `add_conversation` writes across the in-flight window, in-flight asserted).
- Structural pin red on base by inspection (asserts `immediate=True` in `add_conversation`
  and the fork wrapper's source).

### DEFERRED-writer sweep table

AST sweep of every `with ...transaction(...)` site in `tldw_chatbook/DB/*.py` plus the chat
wrapper services (`Character_Chat_Lib.py`, `chat_persistence_service.py`,
`chat_conversation_service.py`); each body classified by its first DB-touching statement.
Empirical backing for the "blind = safe" carve-out: the 0/600 result above (the doctrine
TASK-21100's HOT_MESSAGE_WRITERS comment states, now measured for this task).

FIXED in this task (read-then-write DEFERRED on the chat path, or the task's subject):

| Site | Shape | Action |
|---|---|---|
| `ChaChaNotes_DB.add_conversation` | depth-0 blind INSERT (see mechanism note) | IMMEDIATE (mandated; probe 10/10) |
| `ChaChaNotes_DB.update_conversation` | read version -> UPDATE | IMMEDIATE |
| `ChaChaNotes_DB.soft_delete_conversation` | read version -> UPDATE | IMMEDIATE |
| `ChaChaNotes_DB.restore_conversation` | read deleted/version -> UPDATE | IMMEDIATE |
| `chat_conversation_service.copy_conversation_active_path` (outer) | first stmt = add_message's read | IMMEDIATE (red test's subject) |
| `Character_Chat_Lib.py` import wrapper (~3471) | first stmt = add_message's read | IMMEDIATE |

SAFE — write-first ("blind") bodies, first statement is the write: no snapshot to upgrade,
plain SQLITE_BUSY honors the busy timeout (0/600 measured). Conversations/messages/character
chat: `set_conversation_context_summary`, `set_conversation_console_project_context`,
`update_message_usage_local`, `update_message_metadata_local`,
`append_message_exchanges_local` (deliberately DEFERRED per TASK-21100's own carve-out),
`upsert_transcript_annotation`, `soft_delete_transcript_annotation`,
`add_research_quick_note_owner_proof`, `remove_research_quick_note_owner_proof`,
`delete_sync_log_entries_before`, `add_character_card` (INSERT-first),
`update_character_card` (single version-guarded UPDATE),
`set_character_expression_image`, `delete_character_expression_image`,
`Character_Chat_Lib.create_conversation` wrapper at ~230 and the import branch at ~3353
(outer DEFERRED but the first statement is `add_conversation`'s INSERT), `upsert_trajectory_rows`
(already IMMEDIATE since task-21100). Other DBs: AgentRuns blind writers
(`record_change_snapshots_batch`, `delete_change_snapshots_older_than`, `add_change_note`,
`delete_change_note`, `create_agent_definition`, `update_agent_definition`,
`soft_delete_agent_definition`, `set_status`, `reconcile_orphaned_runs`,
`set_run_assistant_message_id`), Dreams_DB writers, RAG_Indexing_DB.`clear_all`,
Workspace_DB.`mark_agent_backfill_complete`, agent_worktrees `record_created`/
`mark_writer_finished`, automatic_work `pause`/`release`/`settle`/`abort_wake`/
`complete_wake`/`recover`, Subscriptions blind writers (`delete_subscription`,
`record_check_result`, `reset_subscription_errors`, `insert_briefing`,
`complete_briefing`, `insert_briefing_preset`, `delete_briefing_preset`,
`insert_briefing_script`, `create_briefing_audio`, `update_subscription_stats`,
`add_filter`, `save_template`, `set_item_briefing_queued`, `set_item_flagged`,
`transition_watchlist_run`, `mark_watchlist_run_started`).

SAFE — read-only bodies (readers never upgrade): every `get_*`/`list_*`/`search_*`/
`count_*`/`read_*` DEFERRED site, including `get_console_trace_compaction_status`,
`get_local_authority_id`, `get_character_conversation_search_revision`,
`get_conversation_archive_states`, `get_conversations_for_character`,
`search_conversations_page`, `locate_conversation_page`, `get_message_images_by_ids`,
`get_conversation_active_cursor`, `get_attachments_for_messages`,
`get_generation_metadata_for_messages`, `get_message_exchanges`,
`list_full_exchange_keys_for_conversation`, `get_next_trajectory_seq`,
`get_transcript_annotations`, `get_trajectory_rows`, `get_message_variants`,
`get_message_tombstones`, `get_note_version_states`, `list_deleted_notes`,
`list_library_*`/`search_library_*`/`get_library_*`, `read_committed_chat_*`,
`list_current_committed_chat_sync_intents`,
`chat_persistence_service.get_console_trace_fork_boundary` (capture_fork_boundary is a
SELECT) and `resolve_console_fork_commit` (docstring + body: resolves without writing),
the Prompts/Client_Media/Subscriptions reader families, and
character_conversation_search's read paths.

SAFE — boot/upgrade exclusivity: every schema step (`_migrate_from_*`, `_apply_*`,
`_initialize_schema`) in ChaChaNotes, Client_Media_DB_v2, Prompts_DB, Library_Ingest_Jobs_DB,
Library_Collections_DB, Workspace_DB, Subscriptions_DB. They run at DB open on the upgrade
path, before app mount — before the messages_fts backfill starts (it is wired at mount) and
before any live UI writer exists; single writer by construction.

SAME SHAPE, RECORDED — NOT claimed safe, NOT fixed here (out of this task's surgical
scope; the collision source is the once-per-upgrade, seconds-long first-boot backfill
window, and these paths arrive in it at far lower rates than the chat path — but the
instant-busy shape is identical and a scoped follow-up converting them is mechanical):
ChaChaNotes `soft_delete_character_card`/`restore_character_card` (read-then-write),
`replace_keywords_for_conversation`, the flashcard/quiz family (`create_flashcard`,
`update_flashcard`, `update_flashcard_review`, `update_deck`, `delete_flashcard`,
`move_flashcard`, `delete_deck`, `update_flashcard_template`,
`delete_flashcard_template`, `create_question`, `update_question`, `delete_question`,
`start_attempt`, `submit_attempt`, `update_quiz`, `delete_quiz`),
character_conversation_search `ensure_keyword_index`/`reconcile_keyword_index`;
AgentRuns_DB `update_change_snapshot_reverted`, `create_run`, `append_steps`,
`insert_steps_at_indices`, `set_terminal_with_step` (read-then-write); Prompts_DB
`_add_keyword_full`, `update_keywords_for_prompt`, `update_prompt_by_id`,
`soft_delete_keyword`; Client_Media_DB_v2's ~20 read-first writers
(`soft_delete_media`, `undelete_media`, `hard_delete_old_media`, `add_keyword`,
`update_keywords_for_media`, `update_media_metadata`, `soft_delete_keyword`,
`rename_keyword`, `merge_keywords`, `create_document_version`,
`soft_delete_document_version`, `mark_as_trash`, `restore_from_trash`,
`rollback_to_version`, `add_media_chunk`, `batch_insert_chunks`,
`soft_delete_transcript`, `clear_specific_analysis`, `clear_specific_prompt`,
`_add_media_with_keywords_impl`); automatic_work `create_chain`/`attach_run`;
RAG_Indexing_DB `mark_items_indexed`; Library_Ingest_Jobs `upsert_job`/`upsert_retry`.
Subscriptions_DB's writers are enumerated and dispositioned by TASK-21233 (same branch,
next commit).

### Verification

- New module: `pytest Tests/DB/test_chachanotes_conversation_writer_collision.py` — 3 passed.
- Touched-surface regressions: `test_chat_conversation_service.py` +
  `test_console_chat_create_integration.py` + `test_chachanotes_conversation_metadata.py` +
  `test_conversation_archive.py` (135 passed); `test_fts_soft_delete_index_witness.py` +
  `test_search_conversations_fts.py` + `test_chachanotes_v54_before_first_cursor.py` +
  `test_chachanotes_sync_log_retention.py` + the two Chat suites above (223 passed,
  4 failed); `Tests/Character_Chat/test_character_chat.py` +
  `test_character_file_operations.py` (29 passed);
  `test_assistant_generation_state_roundtrip.py` + `test_thinking_conversation_exchange.py`
  (37 passed).
- Pre-existing reds proven by A/B (`git checkout HEAD -- <files>` swaps, never stash): the
  4 failures above (`test_chachanotes_v54_before_first_cursor` x2,
  `test_chachanotes_sync_log_retention` x2 — the task-22280 historical-seeding class) and
  all 5 `test_chachanotes_v47_messages_fts_backfill` failures fail identically on the base
  SHA with this branch's changes reverted. Zero new failures introduced.

### Files changed

- `tldw_chatbook/DB/ChaChaNotes_DB.py` (4 writer conversions; no migration/trigger edits)
- `tldw_chatbook/Chat/chat_conversation_service.py` (fork wrapper conversion)
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py` (import wrapper conversion)
- `Tests/DB/test_chachanotes_conversation_writer_collision.py` (new)
- this task file

ADR: no (see plan section — standing TASK-21100 policy applied; recorded here and in the
code comments).
