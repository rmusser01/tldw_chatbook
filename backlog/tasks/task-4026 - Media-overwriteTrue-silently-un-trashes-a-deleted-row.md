---
id: TASK-4026
title: Media overwrite=True silently un-trashes a deleted row
status: Done
assignee:
  - '@claude'
created_date: '2026-08-09 22:10'
updated_date: '2026-08-11 05:14'
labels:
  - media
  - data-integrity
  - recritique-2026-08-09
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the whole-branch review of task-4022's fix wave (2026-08-09), as an out-of-scope
observation — **pre-existing, not introduced by that work**.

`Media/local_media_reading_service.py:1836` ("create local reading item" — a different function
from the `_materialize_reading_import_row` path that task-4022 protected) calls
`add_media_with_keywords(..., overwrite=True)` without `restore_trashed`. If that hits a trashed
row via the full-content-update branch, `_media_payload` (`DB/Client_Media_DB_v2.py:3610-3629`)
hardcodes `is_trash: 0, trash_date: None, deleted: 0` unconditionally — so the row is silently
un-trashed with no explicit restore decision anywhere in the call chain.

This predates task-4022 entirely: `overwrite=True` has always meant "update in place regardless of
trash state". Task-4022 made the *non*-overwrite restore explicitly opt-in
(`restore_trashed: bool = False`), which throws the remaining implicit case into relief — the
overwrite path still resurrects without asking.

Decide the intended contract and make it explicit either way:
- if `overwrite=True` should also require an explicit restore decision, gate the un-trashing in
  `_media_payload` on the same opt-in flag and update the callers that genuinely want it;
- if resurrect-on-overwrite is correct, document it at `_media_payload` and at the callers, and
  cover it with a test so it stops looking like an oversight.

Audit every `overwrite=True` caller while deciding — the reading-list creator is the one the
review named, but it may not be alone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The intended behaviour of `overwrite=True` against a trashed row is decided and stated in code, not implied
- [x] #2 Whichever way it is decided, a real-DB test pins it
- [x] #3 Every `overwrite=True` caller is audited against that decision, and any that disagrees is fixed or justified in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
**Decided contract** (revises the "hidden in-place update" direction note, with reasons):
a trashed match is NEVER mutated by `add_media_with_keywords` unless the caller passes
`restore_trashed=True` — `overwrite` governs live rows only. `overwrite=True` +
`restore_trashed=False` against a trashed row is a SKIP (`(None, None, <trash message>)`),
not a hidden update and not a resurrection. Both flags true = restore-and-overwrite
(already implemented by task-4022's `restoring_from_trash` path).

Why skip beats hidden-update, caller by caller:
- The hidden-update alternative returns a live-looking `media_id` for a row no list
  shows (no Trash surface exists until task-4025). `save_reading_item` would push a
  trashed row into read-it-later (a ghost the user cannot see); the option-driven
  import flows would report "processed" for items that never appear. Every real
  caller either already handles `media_id=None` as a skip (`add_media`,
  `process_video`, url-article ingest jobs, app.py ingest queue) or should opt into
  a real restore (`save_reading_item`). Nobody is served by a mutation they can't see.
- Skip is also what task-4022's own docstring already CLAIMS ("the row stays trashed
  and untouched"); the code only implemented it for `overwrite=False`. This makes the
  code match the documented contract instead of inventing a third behaviour.
- The current code is internally incoherent: identical-content overwrite left the row
  trashed (but still mutated title/keywords/chunks); different-content overwrite
  resurrected it via `_media_payload`'s hardcoded `is_trash: 0`. Either uniform
  alternative is better; skip is the safe one.

Steps:
1. RED: new real-file-DB pins in `Tests/Media_DB/test_media_db_v2.py`
   (class `TestOverwriteDoesNotTouchTrashedRows`): trashed+overwrite different
   content → skip/unchanged; trashed+overwrite identical content + metadata/keywords
   → untouched; chunked variant (chunks not replaced on skip); both-flags →
   restored+overwritten (+ chunked both-flags variant).
2. GREEN: route Path A on `restoring_from_trash or (overwrite and not row_is_trashed)`;
   trash-aware skip message in Case A.2; contract comment at `_media_payload` and in
   the `add_media_with_keywords` docstring.
3. Rewrite the one invalidated pin: `TestRestoreTrashedIsOptIn.
   test_default_restore_trashed_leaves_url_match_untouched` asserts the old
   "already exists" skip message; the trashed skip now names trash + the restore flag
   (the old message's "Overwrite not enabled" advice would be a lie under this contract).
4. RED→GREEN caller fix: `save_reading_item` (`local_media_reading_service.py:1844`)
   gets `restore_trashed=True` (explicit user save of a named URL = explicit restore
   intent, mirroring `persist_parsed_media`); real-DB caller test first.
5. Audit table for every other `overwrite=True` site (see Implementation Notes);
   update `Local_Ingestion/README.md` overwrite prose; run targeted suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Decided contract (AC#1)** — stated in code at `add_media_with_keywords`'s docstring,
the Path-A routing comment, and `_media_payload`'s docstring
(`DB/Client_Media_DB_v2.py`): a trashed match is NEVER mutated unless the caller
passes `restore_trashed=True`; `overwrite` governs live rows only. Trashed +
`restore_trashed=False` (any `overwrite`) → duplicate-style skip returning
`(None, None, "Media '<title>' matches an item in Trash and was not modified.
Pass restore_trashed=True to restore and update it.")`. Both flags true =
restore-and-overwrite (task-4022's `restoring_from_trash` path, unchanged).
Implementation is the Path-A branch condition
`restoring_from_trash or (overwrite and not row_is_trashed)` plus the trash-aware
skip message; `_media_payload`'s hardcoded `is_trash: 0` is now documented as safe
by routing (a trashed row only reaches it when restoring).

**Why skip, not the direction note's hidden-in-place update**: with no Trash surface
(until task-4025), a hidden update returns a live-looking `media_id` for a row no
list shows — `save_reading_item` would push a ghost into read-it-later, import flows
would count invisible items as processed. Every real caller either already treats
`media_id=None` as a skip or genuinely wants a restore. Skip is also what
task-4022's own docstring already claimed ("the row stays trashed and untouched") —
the code only delivered that for `overwrite=False`. The old behaviour was
incoherent besides: identical-content overwrite mutated title/keywords/chunks in
place while leaving the row hidden; different-content overwrite resurrected it.

**Pre-existing edge deliberately kept**: one-directional URL canonicalization
(`local://` auto-url → real url, identical content) can still fire for a trashed
match, exactly as it has for `overwrite=False` since task-4022. It changes identity
metadata only, never trash state or content. Documented at Case A.2.

**Caller audit (AC#3)** — every `overwrite=True` site in `tldw_chatbook/` (grep),
plus every `add_media_with_keywords` caller:

| Site | overwrite | Disposition |
|---|---|---|
| `Media/local_media_reading_service.py` `save_reading_item` (~1854, the named caller) | literal `True` | **FIXED**: now passes `restore_trashed=True`. Explicit user action naming one exact URL = explicit restore intent (same reasoning as `persist_parsed_media`). Also cures a pre-existing ghost: identical-content re-save used to return a still-trashed id into read-it-later. Pinned by `test_save_reading_item_restores_trashed_match`. |
| `local_media_reading_service.py` `add_media` (~657/700), `process_video` (~1140), `_execute_url_article_media_ingest_job` (~3873 via `ingest_article_to_db_new`) | user-option-driven | **JUSTIFIED**: "overwrite existing" means update live items; a trashed match now lands in each flow's existing `media_id=None` skip/error handling with the trash message — consistent with task-4022's decision that these caller families never resurrect. |
| `local_media_reading_service.py` `_materialize_reading_import_row` (~3223) | `False` | Unaffected; non-restore by design (task-4022 I1), pinned by existing caller test. |
| `Local_Ingestion/local_file_ingestion.py` `persist_parsed_media` (~1640) | absent (`False`) + `restore_trashed=True` | Unaffected — the one explicit-restore ingest writer; restore path untouched by this change. |
| `Chatbooks/chatbook_importer.py` (~1039 `False`, ~1142 absent) | `False` | Unaffected (task-4022 I1 family), suite green. |
| `UI/Console_Modules/message.py` (~1788) | `False` | Unaffected (Console "save message as media", hash-only dedup). |
| `Local_Ingestion/video_processing.py` (~1145), `audio_processing.py` (~1192) (`**media_data`, no overwrite key), `Book_Ingestion_Lib.py` (~2101, no overwrite) | default `False` | Unaffected. |
| `DB/Client_Media_DB_v2.py` wrappers `import_article_to_db` / `import_obsidian_note_to_db` | pass-through param | No production callers found; governed by the new DB contract automatically. |
| Non-media `overwrite=True`: `Prompt_Management/Prompts_Interop.py:399,510`, `prompt_scope_service.py:622`, `DB/Prompts_DB.py:4129` (Prompts DB), `TTS/backends/{chatterbox,higgs}_voice_manager.py` (voice-profile files), `UI/Screens/library_screen.py:11014` (notes-sync conflict) | n/a | **JUSTIFIED out of scope**: different subsystems with no Media trash semantics; this contract binds `MediaDatabase.add_media_with_keywords` only. |

**Pins (AC#2)** — all real file-backed SQLite (`file_db` fixture / `Database(db_path=...)`):
- NEW `TestOverwriteDoesNotTouchTrashedRows` (5 tests, `Tests/Media_DB/test_media_db_v2.py`):
  overwrite-alone skip on different content (the headline resurrection, RED pre-fix);
  overwrite-alone leaves metadata/keywords untouched on identical content (the quieter
  in-place mutation, RED pre-fix); chunked skip variant (stored chunks not replaced,
  RED pre-fix); both-flags restore-and-overwrite; chunked both-flags variant
  (chunk replacement, no UNIQUE collision).
- NEW `test_save_reading_item_restores_trashed_match`
  (`Tests/Media/test_local_media_reading_service.py`, RED between DB change and caller fix).
- **REWRITTEN pin** (deliberate, the only one the contract change invalidated):
  `TestRestoreTrashedIsOptIn::test_default_restore_trashed_leaves_url_match_untouched`
  asserted the generic "already exists. Overwrite not enabled." message for a trashed
  skip — advice that became a lie (overwrite no longer touches trashed rows). It now
  asserts the trash-naming message. No test anywhere asserted
  resurrection-by-overwrite-alone (verified by grep across `Tests/`).

**Test evidence**: `Tests/Media_DB/test_media_db_v2.py` +
`test_media_db_properties.py` 69 passed; `Tests/Media/test_local_media_reading_service.py`
68 passed; `Tests/Media/test_media_reading_scope_service.py` +
`Tests/RAG/test_ingestion_indexing.py` 120 passed; `Tests/Chatbooks/test_chatbook_importer.py`
+ `Tests/Local_Ingestion/test_ingest_parse_worker.py` + `Tests/Library/test_library_export_scope.py`
+ `Tests/tldw_api/test_media_ingest_jobs_client.py` 75 passed;
`Tests/Media_DB/test_stt_provenance_persistence.py` + `Tests/App/test_submit_library_ingest_job.py`
green. `Tests/integration/test_library_ingest_flow.py`: 2 ambient environmental
failures (fixtures require pdf/audio tooling to be ABSENT; this venv has them
installed — unrelated to this change, fail on base too by construction).

**Files**: `tldw_chatbook/DB/Client_Media_DB_v2.py`,
`tldw_chatbook/Media/local_media_reading_service.py`,
`tldw_chatbook/Local_Ingestion/README.md`, `Tests/Media_DB/test_media_db_v2.py`,
`Tests/Media/test_local_media_reading_service.py`.
<!-- SECTION:NOTES:END -->
