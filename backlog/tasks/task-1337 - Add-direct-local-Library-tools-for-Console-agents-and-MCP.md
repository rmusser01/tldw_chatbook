---
id: TASK-1337
title: Add direct local Library tools for Console agents and MCP
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-03 03:30'
updated_date: '2026-08-03 03:41'
labels:
  - library
  - agents
  - mcp
  - privacy
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-02-local-library-agent-tools-design.md
  - backlog/decisions/030-local-library-agent-tool-boundary.md
documentation:
  - Docs/superpowers/plans/2026-08-02-local-library-agent-tools.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give Console agents and local MCP clients safe read-only access to factual local Library inventory and content without requiring semantic retrieval so users can count find and inspect their own Library items predictably.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console agents and local MCP expose list get and lexical search for Media Notes Prompts Skills Conversations and Collections
- [ ] #2 List and search return bounded pages with exact distinct totals stable IDs deterministic pagination and keyword matches
- [ ] #3 Get requires a returned stable ID and chunks large text with revision-aware continuation below 32 KiB
- [ ] #4 Direct tools never use RAG embeddings or semantic similarity and never return binary data or filesystem paths
- [ ] #5 The Console direct-tools setting defaults on and off mode exposes bounded Library RAG with visible cloud-model privacy copy
- [ ] #6 Console mode cannot be bypassed by built-in MCP overlaps while MCP client compatibility remains unchanged
- [ ] #7 Automated tests cover contracts trust boundaries MCP bootstrap Console integration and settings behavior
- [ ] #8 ADR-030 design and implementation documentation are linked
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/030-local-library-agent-tool-boundary.md
Reason: durable cross-module Console/MCP read contract, direct-versus-RAG runtime boundary, stable-ID continuation, and cloud-model privacy behavior.

Implementation plan: Docs/superpowers/plans/2026-08-02-local-library-agent-tools.md

1. Implement descriptor, ID, cursor, validation, and byte-bound contracts with tests.
2. Add exact text-only query seams for all six Library types.
3. Implement the shared 18-operation LocalLibraryToolService.
4. Integrate Console direct/RAG providers and the global privacy setting.
5. Add Console-only MCP overlap filtering and descriptor-backed MCP registration/delegation.
6. Verify cross-runtime parity, compatibility, documentation, and Definition of Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**2026-08-06 progress (@kimi, branch feat/hub-local-agent-tools):** Plan Task 1 (shared contracts) landed: `tldw_chatbook/Library/library_tool_contract.py` — the 18-descriptor table (names, trust/read-only description tails, bounded input schemas, `type.operation` routes), opaque stable-ID codec (`type:<base64url>`, ASCII, ≤128 B, fail-closed parse with wrong-type/malformed/path-like rejection), versioned continuation-cursor codec (SHA-256-checked canonical payload, tamper → `invalid_argument`, revision mismatch → `content_changed` + fresh-start hint), the six spec error codes with JSON-safe `to_payload()`, page/max-chars/query validation, control-character display normalization (160 B cap, 32 B floor), and the 32 KiB byte fitting (`fit_page_payload` with the fixed keywords→preview→metadata trim order + `fit_text_segment` largest-prefix fitting with skip/repeat-free offsets). Tests: `Tests/Library/test_library_tool_contract.py` (~30 cases). `Library/__init__.py` exports the descriptor table and error type. Remaining: plan Tasks 2–10 (storage query seams per type, `LocalLibraryToolService`, Console provider + privacy setting, MCP registration/delegation, parity/docs). No ACs ticked yet — service behavior does not exist until Task 5.

**2026-08-06 progress 2 (@kimi, branch feat/hub-local-agent-tools):** Plan Task 2 (Media + Notes query seams) landed, test-first. Media DB (`Client_Media_DB_v2.py`): `list_library_media_page` / `search_library_media_page` / `get_library_media_text` — active = `deleted=0 AND is_trash=0`, count+page in one transaction, stable `last_modified DESC, id DESC` order, 241-char preview projection (never full content, `vector_embedding`, url, or paths), keywords per page via one IN-query grouped in Python (20-value cap + exact `keyword_total` + `keywords_truncated`). Search branches OR/dedup by row: ci exact title, escaped-LIKE title/content substrings, tokenized+quoted safe FTS (user operators inert), keyword-relation substring; exact-title-first ordering; per-row hit flags derive `matched_fields`/`matched_keywords`. Detail reads only `substr(content, start+1, max_chars)` + `length(content)`. Notes DB (`ChaChaNotes_DB.py`) mirrors the same three methods against `notes`/`notes_fts` (rowid join) /`note_keywords`/`keywords` with `rowid` tie-break. Service delegates: `LocalMediaReadingService.list/search_library_media` + `get_library_media_text`, and `NotesInteropService.list/search_library_notes` + `get_library_note_text` (metric-logged like existing methods). Also fixed a real bug: `notes_scope_service.list_notes` dropped `offset` for LOCAL_NOTE (interop already accepted it). Tests: 8 media cases in `Tests/Media/test_local_media_reading_service.py`, 5 real-DB + 4 delegate cases in `Tests/Notes/test_notes_library_unit.py`, 1 offset-forward case in `Tests/Notes/test_notes_scope_service.py` — focused trio + `Tests/DB` run green (861 passed, 1 platform skip). Public-ID encoding stays out of this layer per plan (Task 5's `LocalLibraryToolService` owns it); raw uuid/version returned. ADR check: no new ADR — additive read seams under ADR-030/031/032's existing boundaries.

**2026-08-06 progress 3 (@kimi, branch feat/hub-local-agent-tools):** Plan Task 3 (Prompt + Skill seams) landed, test-first. Prompt DB (`Prompts_DB.py`): `list_library_prompts_page` / `search_library_prompts_page` / `get_library_prompt_overview` / `get_library_prompt_section` — same discipline as Media/Notes (one-transaction count+page, `last_modified DESC, id DESC`, bounded `details_preview` + per-section presence flags, keyword cap 20 + exact total). Search spans name/details/system_prompt/user_prompt/prompt_definition (escaped LIKE) + tokenized safe FTS + keyword relation; exact-name-first ranking; FTS-only hits get honest per-column probes (`name/details/system_prompt/user_prompt`, the FTS-indexed columns). Overview bounds every present section independently (exact `total_chars` + 241-char preview, never full text, never version history); the section reader is column-whitelisted (`InputError` on anything else) and reads only `substr`+`length`. `Prompts_Interop` gained thin wrappers; `LocalPromptService` async delegates forward totals/matched_fields. Skills (`local_skills_service.py`): `list_library_skills` / `search_library_skills` / `get_library_skill` / `get_library_skill_file` — one managed enumeration → exact total before slicing; blocked skills surface only safe name/description/trust fields and their bodies are never read (search skips body matching for them); detail returns exact body length + bounded preview + manifest with opaque `file:<base64url>` tokens (never file content); the file reader mirrors `read_skill_file`'s security order (trust re-verify → token parse → containment → trust-material membership → bounded read), binds `revision` = SHA-256[:16] of the file bytes, rejects traversal/garbage/binary, and never uses the eager `_read_supporting_files` path (proven by a monkeypatched-explosion test). Tests: 7 DB + 3 service prompt cases, 8 skill cases — full `Tests/Prompt_Management` + `Tests/Skills` + `Tests/Prompts_DB` run green (799 passed). ADR check: no new ADR — additive read seams under existing trust/policy boundaries.
**2026-08-07 progress 4 (@kimi, branch feat/hub-local-agent-tools):** Plan Task 4 (Conversation + Collection seams) landed, test-first (40 new tests, red-then-green; 96 passed across the three touched test files). Conversation DB (`ChaChaNotes_DB.py`): `list_library_conversations_page` / `search_library_conversations_page` / `get_library_conversation_messages` — one-transaction COUNT+page, active = `deleted=0`, stable `last_modified DESC, rowid DESC`; search OR-branches (ci exact title, escaped-LIKE title, correlated message-content LIKE EXISTS, tokenized+quoted safe FTS on `conversations_fts`/`messages_fts` via rowid, keyword-relation substring) dedupe by conversations row so multi-message matches count once; per-row hit flags → honest `matched_fields` ⊆ {title,message,keywords} + `matched_keywords`; exact-title-first ranking; keywords per page via one IN-query (cap 20 + exact `keyword_total` + `keywords_truncated`). Message reader: explicit text/metadata columns + `length(content)` + `substr(content, start+1, max_chars)` — never `SELECT *`, never `image_data`, never the full body; page mode (`timestamp ASC, rowid ASC`, exact `message_total`, `next_message_offset`) and single-message continuation mode (`message_id`+`char_start`, slices resume exactly where the previous ended); `revision` = SHA-256[:16] of `version:total_chars` (stored version is content-bound via `update_message`'s optimistic locking; hashed so agents treat it as opaque); RAG context stays in the service-owned JSON sidecar — responses always carry `include_rag_context: False` (proven by a sidecar-seeded test). `ChatConversationService` gained thin sync delegates echoing `{items,total,offset,limit}`. Collections (`library_collections_service.py`, protocol + local impl): `list_library_collections` / `search_library_collections` / `get_library_collection` — same count+page discipline; search restricted to name/description/direct stored member title (never member content or raw source identity); members page ordered `created_at ASC, membership_id ASC` with exact `member_total`; supported source types map through the shared public-ID codec (`make_public_id`, round-trip-proven for all six types, mixed-case normalized), unsupported/unencodable sources get `item_id=None` + opaque `ref:<base64url>` (raw `source_id` never exposed); member titles bounded at 160 B via `normalize_display_text`. ADR check: no new ADR — additive read seams under ADR-030/031/032's existing boundaries.
<!-- SECTION:NOTES:END -->
