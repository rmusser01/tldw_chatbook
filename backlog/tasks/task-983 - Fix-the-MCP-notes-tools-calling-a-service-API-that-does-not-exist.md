---
id: TASK-983
title: Fix the MCP notes tools calling a service API that does not exist
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 19:33'
updated_date: '2026-07-27 20:09'
labels:
  - mcp
  - notes
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
MCP/server.py constructs NotesInteropService(self.chachanotes_db) with the wrong argument count and type, and its create_note and search_notes tools call methods and keyword arguments the real class does not define, so those tools cannot work. Found while removing the nonexistent CharacterInteropService from the same module (TASK-968) and reported rather than fixed because resolving it needs a design call about what the notes tools should expose. This is the third defect of the same shape in this one module -- the others were a config key that did not exist so a database opened in the working directory (TASK-854), and a service that was never implemented (TASK-968).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP notes tools call a real NotesInteropService API with correct arguments,create_note and search_notes work end to end against a temp database,A test exercises both tools rather than only importing the module,The module has no remaining reference to a service or method that does not exist
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the real NotesInteropService (Notes/Notes_Library.py) and the already-working construction pattern in Tools/note_management_tools.py and app.py.
2. Fix TldwMCPServer._init_databases()'s NotesInteropService construction to the real (base_db_directory, api_client_id, global_db_to_use) signature.
3. Rewrite create_note/search_notes tool bodies to call add_note/search_notes with the real keyword arguments and dict-based results, dropping the unsupported tags/template parameters.
4. Add an end-to-end regression test (Tests/MCP/test_server_notes_service.py) against a temp on-disk DB, bypassing __init__ via __new__ the way the TASK-854/968 tests do.
5. Do a deliberate pass over the rest of MCP/server.py and its collaborator modules (tools.py, resources.py, prompts.py) for the same defect shape; fix what is unambiguous, file what needs a design decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed TldwMCPServer._init_databases() constructing NotesInteropService(self.chachanotes_db) -- one positional arg bound to the real class's base_db_directory parameter -- to the real (base_db_directory, api_client_id, global_db_to_use) signature, matching the already-working construction in Tools/note_management_tools.py/app.py: base_db_directory=get_chachanotes_db_path().parent, api_client_id=CLI_APP_CLIENT_ID, global_db_to_use=self.chachanotes_db (reuses this process's own DB handle instead of opening a second one).

Rewrote the create_note and search_notes tool bodies: create_note now calls notes_service.add_note(user_id=, title=, content=) -- the real method; create_note does not exist on the class. Dropped the tags/template parameters entirely: neither NotesInteropService nor the notes table has any such concept (tags/template were never real, just assumed). search_notes now calls notes_service.search_notes(user_id=, search_term=, limit=) -- the old call passed query=/limit= with no user_id (a TypeError on the real signature) -- and reads results as dicts (note.get("id")/note.get("last_modified")) instead of attribute access (note.id/note.updated_at, an AttributeError either way; the real column is last_modified, not updated_at). user_id is resolved via a new _resolve_notes_user_id() helper mirroring Tools/note_management_tools.py's _resolve_user_id (load_settings()["USERS_NAME"] or "default_user" -- an attribution value, not a visibility partition).

Proved it end to end (not just import) in Tests/MCP/test_server_notes_service.py: bypasses __init__ via __new__ (same technique test_server_media_db_path.py/test_server_character_service.py use), runs the real _init_databases(), swaps a recording fake in for FastMCP so the actual create_note/search_notes closures can be captured and called, and creates+searches a note against a temp on-disk DB with HOME/TLDW_CONFIG_PATH redirected to a temp profile. 3/3 pass; the notes-service construction no longer needs the _PermissiveFakeService stub the TASK-854/968 tests used (their own docstrings updated to note this, stub kept there anyway to keep those files' scope narrow).

Whole-module pass (as asked, not just server.py's notes tools) turned up more of the identical defect shape in server.py's collaborator modules and fixed what was unambiguous (a real column/method exists with the obviously-intended value/semantics, no design call needed):
- tools.py::chat_with_character: get_cli_setting("API", f"{provider}_api_key", "") -> get_api_key(provider) (same defect TASK-968 fixed in server.py's chat_with_llm, sibling call site missed).
- resources.py::list_recent_conversations/list_recent_notes: get_recent_conversations/get_recent_notes (neither exists on CharactersRAGDB) -> list_all_active_conversations(limit=)/list_notes(limit=) (both already ordered by recency).
- resources.py::get_media_resource + prompts.py::analyze_media_prompt: media_db.get_media_transcript(media_id) (never existed) -> the module-level get_media_transcripts(db, media_id), most-recent transcript taken.
- prompts.py::summarize_conversation_prompt/generate_document_prompt: get_conversation_messages (never existed) -> get_messages_for_conversation(str(id)) (same fix already applied at the identical call shape in tools.py/resources.py).
- Wrong-dict-key defects (not missing methods -- wrong column names), found because they made my own new tests fail: media['media_type']/media['created_at'] (Media's real columns are type/ingestion_date) in both resources.py and prompts.py; char.get('greeting')/char.get('example_dialogue')/char.get('updated_at') (character_cards' real columns are first_message/message_example/last_modified) and conv.get('updated_at')/note.get('updated_at') (both last_modified, not updated_at) in resources.py. All fixed; covered by new tests in Tests/MCP/test_tools_resources_prompts_real_methods.py (8 tests).

Left un-fixed, needs a design decision, filed as TASK-985: tools.py::search_conversations calls chachanotes_db.search_all_content(...), which does not exist anywhere in the codebase; the real equivalent (search_conversations_by_content) returns conversation rows with no inline "content" column, so the tool's preview formatting can't be ported without deciding what a preview should show instead. resources.py::get_rag_chunk_resource calls media_db.get_chunk_by_id(int(chunk_id)), which also does not exist; the real accessor (get_chunk_text) takes a UUID string not an int id and returns bare text with no media_id/start_char/end_char/embedding_id (UnvectorizedMediaChunks has no embedding_id column at all) -- the resource's whole id scheme and metadata contract need reconciling, not guessing.

Noted but not fixed (harmless, always-empty via .get() defaults, no crash, a feature that was never built rather than a bug): notes have no tags/template columns (resources.py::get_note_resource's tags/template metadata always empty); character_cards has no message_count (tools.py::list_available_characters / resources.py::get_character_resource always report 0); Media has no duration column (resources.py::get_media_resource's duration metadata always empty).

Modified: tldw_chatbook/MCP/server.py, tldw_chatbook/MCP/tools.py, tldw_chatbook/MCP/resources.py, tldw_chatbook/MCP/prompts.py. Added: Tests/MCP/test_server_notes_service.py, Tests/MCP/test_tools_resources_prompts_real_methods.py. Updated stale scope-disclaimer comments in Tests/MCP/test_server_media_db_path.py and test_server_character_service.py now that the NotesInteropService construction they used to have to stub around is fixed for real. Filed TASK-985 for the two remaining defects that need a design call. Full Tests/MCP/ suite: 377 passed.
<!-- SECTION:NOTES:END -->
