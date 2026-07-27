---
id: TASK-968
title: Fix the missing CharacterInteropService referenced by MCP server
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 18:06'
updated_date: '2026-07-27 19:28'
labels:
  - mcp
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While fixing TASK-854's media DB lookup, MCP/server.py was found referencing a CharacterInteropService that does not exist, so that code path cannot run. Left unfixed to keep TASK-854 scoped to the database-path defect, and recorded here. Note TASK-854 also found the same file opening ./media_library.db in the working directory because it read a config key that does not exist -- this file warrants a broader look than either task gave it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP/server.py's character code path either resolves a real service or is removed,No reference to a nonexistent service remains in that module,The module's other config lookups are checked against the declared accessors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read MCP/server.py in full; confirm CharacterInteropService does not exist anywhere in the codebase (grep for the class definition).
2. Determine intent: check whether self.character_service is consumed anywhere in the MCP package, and whether MCPTools' character-related tools (chat_with_character, list_available_characters) already get character data another way.
3. Decide resolve-vs-remove based on that, then apply the fix and update tests that stub around the old bug (test_server_media_db_path.py's helper).
4. Broader look: check every other service construction, config lookup (get_cli_setting), and import in the module against its target's real signature/declared accessor; report mismatches found even if not fixed.
5. Write a before/after reproduction bypassing __init__ (which needs the optional mcp package) via __new__, calling the real _init_databases() directly, mirroring the existing test's technique.
6. Add regression tests; run Tests/MCP/, Tests/Utils/.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Decision: removed the dead code path rather than resolving it to a real service. Character_Chat_Lib.py is a free-function module -- it has never had any class in it (confirmed by listing every top-level class/def in the file), so CharacterInteropService was never renamed or moved, just never implemented. self.character_service (built from it) was never read anywhere else in MCP/server.py or the rest of the MCP package: the character-related tools that ARE wired up (chat_with_character, list_characters) go through self.tools (MCPTools), which already reads character rows directly off self.chachanotes_db via get_character_card_by_id()/list_character_cards(). There was no unfinished feature behind the reference to wire up, only a guaranteed ImportError on every call to _init_databases().

Reproduction (both commands run from the worktree, HOME/TLDW_CONFIG_PATH redirected to a scratch dir, __new__ used to bypass __init__'s optional-mcp-package guard so _init_databases() -- the real, unmodified method -- can be driven directly, same technique as the pre-existing Tests/MCP/test_server_media_db_path.py):
  Before (git show HEAD:tldw_chatbook/MCP/server.py copied into place): `_init_databases()` raised `ImportError: cannot import name 'CharacterInteropService' from 'tldw_chatbook.Character_Chat.Character_Chat_Lib'` (exit 1).
  After (this task's fix): `_init_databases()` succeeds; the constructed instance has no `character_service` attribute at all (exit 0).
Full script and both raw outputs are in this task's report file; the same assertions are now pytest-enforced in Tests/MCP/test_server_character_service.py.

Broader look at MCP/server.py, as asked (checked, not all fixed):
- FIXED (in scope, low-risk, single call site): chat_with_llm's API-key lookup called get_cli_setting("API", f"{provider}_api_key", "") directly instead of config.get_api_key(provider), the declared accessor. get_api_key() checks three tiers (newer api_settings.<provider> structure with its own env-var-name indirection, then the legacy [API] section -- the one tier the direct call covered -- then a bare {PROVIDER}_API_KEY env var); the direct call silently missed a key configured via either of the other two, returning "No API key configured" even when one existed. Repointed to get_api_key(provider); get_cli_setting import removed (no other use in the file).
- REPORTED, NOT FIXED (own design decision, out of this task's scope): self.notes_service = NotesInteropService(self.chachanotes_db) is ALSO broken, independently of the character bug. NotesInteropService.__init__ requires (base_db_directory: str|Path, api_client_id: str, global_db_to_use=None) -- the call passes only one positional arg (a CharactersRAGDB instance) where a directory path is required, and omits the required api_client_id entirely (this is exactly why test_server_media_db_path.py has to stub NotesInteropService with a permissive fake to get past _init_databases() at all -- documented in that test's docstring, updated by this task). Even a corrected construction wouldn't help: the create_note tool calls self.notes_service.create_note(title=, content=, tags=, template=) but the real class has no create_note method (only add_note(user_id, title, content, note_id=None), no tags/template params, and a required user_id neither tool call supplies); search_notes calls self.notes_service.search_notes(query=, limit=) but the real signature is search_notes(user_id, search_term, limit=10, fts_match_query=None, *, id_allowlist=None) -- wrong kwarg name (query vs search_term) and again a missing required user_id. Fixing this needs a real design decision (where does an MCP-context user_id come from?) that's beyond this task; recommend filing separately.
- Verified clean: MCPResources and MCPPrompts constructors, and every method server.py calls on self.resources/self.prompts (get_conversation_resource, get_note_resource, get_character_resource, get_media_resource, get_rag_chunk_resource, list_recent_conversations, list_recent_notes, summarize_conversation_prompt, generate_document_prompt, analyze_media_prompt, search_and_synthesize_prompt, character_writing_prompt) -- all exist with matching signatures. MCPTools' other methods used by server.py (search_conversations, get_conversation_history, export_conversation, perform_rag_search) also verified present. get_chachanotes_db_path()/get_media_db_path() (TASK-854's fix) still correct. ingest_media is an explicit TODO/placeholder, not a silent-failure case.

Files: tldw_chatbook/MCP/server.py (removed the import + self.character_service line with an explanatory comment; repointed chat_with_llm to get_api_key), Tests/MCP/test_server_media_db_path.py (dropped the now-unnecessary CharacterInteropService stub, updated docstring), Tests/MCP/test_server_character_service.py (new -- 4 tests covering the removal and the get_api_key repoint via AST-based source checks plus an executable _init_databases() regression guard). Verified via Tests/MCP/ (366 passed) and Tests/Utils/ (562 passed).
<!-- SECTION:NOTES:END -->
