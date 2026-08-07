---
id: TASK-2451
title: Make the default assistant an editable sample persona
status: Done
assignee:
  - '@claude'
created_date: '2026-08-05 04:48'
updated_date: '2026-08-07 06:45'
labels: []
dependencies:
  - TASK-2450
  - TASK-951
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The seeded 'Default Assistant' character card (DB row 1, Character_Chat_Lib.DEFAULT_CHARACTER_ID) currently ships with placeholder, non-illustrative content, so a new user has nothing worth customizing on day one and no worked example of what a character card can hold. Owner rulings on the original persona-shaped design (recorded 2026-08-06) redirected this task: personas cannot carry voices until TASK-617/ADR-037, so the sample is the existing character card itself, enriched in place -- not a new Persona entity and not a second character card. Enrichment must never overwrite a user's own customization of that row, and the card cannot claim a pre-assigned voice (voice profiles live in a separate store that is empty at seed time), so the card teaches voice assignment instructionally instead.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A freshly created database seeds the 'Default Assistant' character card (id=1) with rich, documentation-grade content (description, personality, system_prompt, first_message, alternate_greetings, creator_notes) instead of the historical one-line placeholder text
- [x] #2 On an existing database, row 1 is promoted to the rich content ONLY when every user-editable content field on it is still byte-identical to the original bare-seed literals; a row with any single field ever edited by a user is left completely unmodified
- [x] #3 name stays 'Default Assistant' and creator stays 'System' in both the bare and enriched versions of the card, so the FK anchor (Character_Chat_Lib.DEFAULT_CHARACTER_ID) and provenance are unaffected
- [x] #4 The card's own content never claims a pre-assigned voice; instead it walks the user through assigning one via the character editor's Voice & Speech section and mentions Settings > Speech & TTS's Default voice profile for the app-wide default, and every UI label the content cites is verified against the real running screens
- [x] #5 Editing the seeded Default Assistant card (id=1) for the first time succeeds without error on both a fresh database and an existing pre-task-2451 database -- fixes a pre-existing FTS5 shadow-index defect (row 1's INSERT ran before its FTS5 table/triggers existed) that this task's own enrichment write would otherwise hit
- [x] #6 Schema version is bumped with an accompanying, tested migration following the existing DB/migrations pattern; ordinary character_id=1 conversation/FK behavior is unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture the exact bare-seed literals in tldw_chatbook/DB/ChaChaNotes_DB.py's character_cards row-1 INSERT as the byte-identity baseline.
2. Author rich content for description/personality/system_prompt/first_message/alternate_greetings/creator_notes; verify every UI label it cites (Roleplay nav label, Characters mode, Voice & Speech section title, Settings > Speech & TTS, Default voice profile field) against the real source.
3. Diagnose and fix the pre-existing FTS5 ordering defect discovered while prototyping the migration (row-1 INSERT precedes character_cards_fts/its triggers in _FULL_SCHEMA_SQL_V4, so any UPDATE to row 1 -- including this task's own enrichment -- raises SQLITE_CORRUPT_VTAB "database disk image is malformed"): reorder the fresh-schema SQL and rebuild the FTS index for existing databases before the migration's content UPDATE runs.
4. Add a shared conditional-enrichment routine comparing every character_cards content column (not just the ones being written) against the bare-seed literals; wire it into both the fresh-DB seed path and a new v31->v32 migration; bump _CURRENT_SCHEMA_VERSION and add the DB/migrations mirror file.
5. TDD: fresh DB seeds rich; migration enriches an untouched bare row; migration preserves a row with any single field edited (parametrized); migration/enrichment is idempotent; first edit to row 1 no longer crashes (fresh AND pre-existing DB); regular character_id=1 FK/chat behavior unchanged; schema version arithmetic correct.
6. Run the ChaChaNotes DB test files + character-lib tests, a repo-wide --collect-only sweep, and ruff; update the task's ACs (done up front per repo convention) and Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Redirected by owner rulings recorded 2026-08-06 (see conversation, not a spec file): the sample is the EXISTING seeded 'Default Assistant' character card (row 1, Character_Chat_Lib.DEFAULT_CHARACTER_ID), enriched in place -- not a new Persona entity (personas can't carry voices until TASK-617/ADR-037) and not a second character card. ACs were rewritten up front to match before implementation, per repo convention. Filing history (ADR-037 boundary, TASK-617 scope) preserved above; superseded by this note as the authoritative summary.

Approach: authored rich description/personality/system_prompt/first_message/alternate_greetings(2)/creator_notes for row 1, keeping name='Default Assistant' and creator='System' stable (FK anchor + provenance). A shared routine, `_enrich_default_assistant_card_if_bare`, runs a single conditional UPDATE gated on ALL character_cards content columns (not just the ones it writes) matching the original bare-seed literals byte-for-byte -- any single field ever edited by a user, anywhere on the row, leaves it completely untouched. This routine is called from both `_apply_schema_v4` (fresh databases enrich immediately after the bare row is inserted) and the new `_migrate_from_v31_to_v32` migration (existing databases), so both paths share one definition of "still bare" and "rich". Schema bumped 31->32; DB/migrations mirror file added (generated from the Python constants so it can't drift).

Blocking defect found and fixed: prototyping the migration's UPDATE crashed with `sqlite3.DatabaseError: database disk image is malformed` (SQLITE_CORRUPT_VTAB) on EVERY database, fresh or old -- a pre-existing bug independent of this task. Row 1's INSERT in `_FULL_SCHEMA_SQL_V4` ran BEFORE `character_cards_fts`/its `character_cards_ai` trigger existed, so row 1 was never indexed; the first-ever UPDATE to that row (character_cards_au's FTS5 'delete' special command, asking FTS5 to remove entries that were never inserted) corrupted the shadow index. This meant a real user's first edit of the Default Assistant character, via the ordinary Roleplay editor, already crashed the app before this task touched anything. Fixed two ways: (1) reordered `_FULL_SCHEMA_SQL_V4` so the row-1 INSERT runs after the FTS5 table/triggers exist (fresh installs never hit it again), and (2) the shared enrichment routine runs `INSERT INTO character_cards_fts(character_cards_fts) VALUES ('rebuild')` before its content UPDATE, which safely reconstructs the whole index from the content table for any pre-existing (already-created) database, guarded by a `character_cards` table-existence check so synthetic/legacy test fixtures without that table are a no-op.

Verified against source (not memory): "Roleplay" nav label (Constants.py TAB_PERSONAS), "Characters" mode chip (personas_screen.py), "Voice & Speech" section title (personas_character_tts_widget.py:180, mounted in the character editor), "Speech & TTS" Settings category and "Default voice profile" field label (settings_screen.py / speech_tts_settings_panel.py). Content claims about which fields feed the model (system_prompt verbatim first; personality/description/scenario/message_example labelled; creator/version/tags never sent) verified against `compose_character_card_text` in Character_Chat_Lib.py.

Tests (Tests/DB/test_chachanotes_default_assistant_enrichment_migration.py, 28 cases, TDD): fresh DB seeds rich content + version 32; migration enriches an untouched bare row and bumps its version; fields the migration doesn't write stay exactly bare; migration preserves a row with any ONE of 13 fields edited (parametrized, asserts every OTHER field is byte-identical to pre-migration, not just the edited one); a soft-deleted row 1 is left alone; enrichment/migration is idempotent (re-running is a no-op; reopening doesn't re-touch); the crash regression is fixed on both a fresh DB and a migrated pre-existing DB; FTS search actually finds row 1 post-enrichment (a plain SELECT alone can't distinguish "indexed" from "readable via content table"); the FTS rebuild doesn't disturb a second, independently-inserted character card; ordinary character_id=1 conversation/message FK behavior is unchanged; schema-version arithmetic (32, and the v31->v32 migration rejects a non-31 starting version).

Ripple: 5 pre-existing tests hard-coded "fresh DB reaches v31" (or similar) and were updated to 32, following this repo's established pattern for every prior schema bump (Tests/ChaChaNotesDB/test_message_generation_metadata.py, Tests/DB/test_chachanotes_active_leaf_migration.py, test_chachanotes_context_summary_migration.py, test_chachanotes_message_metadata_migration.py, test_chachanotes_message_usage_migration.py).

Gates: Tests/ChaChaNotesDB/ + Tests/DB/ = 903 passed, 1 skipped (Windows-only). Tests/Character_Chat/ + Tests/Chat/test_chat_functions.py + Tests/Chatbooks/ = 799 passed, 1 skipped (--run-slow gated). Repo-wide `pytest --collect-only` = 31789 collected, 0 errors. `ruff check` on all touched files = clean.

Files: tldw_chatbook/DB/ChaChaNotes_DB.py (schema version bump, FTS5 ordering fix, _enrich_default_assistant_card_if_bare + rich/bare content constants, _migrate_from_v31_to_v32, migration_steps registration); tldw_chatbook/DB/migrations/chachanotes_v31_to_v32_default_assistant_enrichment.sql (documentation mirror, not loaded at runtime); Tests/DB/test_chachanotes_default_assistant_enrichment_migration.py (new); 5 existing test files with the version-number ripple fix.

Lesson worth banking: prototype a migration's actual write path against a schema built the same way production builds it (constructor, not a hand-rolled minimal fixture) before trusting it -- a trivial single-column UPDATE with no relation to this task's content surfaced a years-old, always-crashing defect that 900+ existing green tests never exercised because nothing had ever tried to edit character id=1 as the first write after its creation.
<!-- SECTION:NOTES:END -->
