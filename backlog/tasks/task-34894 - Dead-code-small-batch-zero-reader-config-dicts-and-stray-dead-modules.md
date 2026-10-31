---
id: TASK-34894
title: 'Dead-code small batch: zero-reader config dicts and stray dead modules'
status: Done
created_date: 2026-10-12 17:52
labels:
- simplification
- dead-code
assignee:
- '@zcode'
updated_date: 2026-10-12 20:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Assorted verified-dead code: config.py builds legacy provider dicts that nothing reads and several stray dead modules remain across the tree.

Evidence (verified on dev df4985e): Both access paths grepped (direct dict indexing and get_cli_setting). _LEGACY_PROVIDER_API_KEY_BRIDGE is load-bearing for live providers (Tests/Chat/test_provider_readiness.py) and must stay. Dead MCP tool calls save_conversation_from_messages and chat_with_provider which unconditionally raise (tools.py:50-61). The keywords CSV helper is pinned only by Tests/Prompts_DB/test_prompts_db_legacy.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Zero-reader legacy dict google_generative_api removed (HEAD census REFUTED the review: the other five named dicts do not exist as keys and all 17 remaining *_api dicts have live readers),Live legacy bridges for openai anthropic mistral groq cohere huggingface elevenlabs custom_openai preserved,ccv3_parser.py empty file and its stale comment reference removed,UI/ChatbookCreationWindow.py and its dedicated test removed with modal-tier entries pruned,DESCOPED: MCP chat_with_character stays - ADR-053 pins the not-available-yet contract and five test files pin the MCP surface,DB dead code removed (execute_query_with_retry _add_keyword_with_retry export_keywords_to_csv method wrapper import_obsidian_note_to_db fetch_keywords_for_media module wrapper; two tests swapped to the live batch variant),Config loads identically for live providers
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify each item at HEAD 2. Remove zero-reader legacy config dict builders (chat_with_character DESCOPED per ADR-053 - MCP surface is pinned by 5 test files and a recorded not-available-yet contract) 3. Delete ccv3_parser.py + fix stale comment 4. Delete ChatbookCreationWindow + census entries 5. Remove dead DB helpers + pinning test 6. Update AC for descope + test run vs baseline + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
HEAD re-verification REFUTED most of the review's config claim: of 18 legacy *_api dicts an exact-key census found only google_generative_api with zero readers (the review's other five names do not exist as keys; all 17 remaining dicts have live readers). Removed: that dict + its google_api_key bridge var (bridge row kept - it feeds the modern api_settings table per its docstring), the three dead Prompts_DB methods (execute_query_with_retry _add_keyword_with_retry export_keywords_to_csv METHOD wrapper - the standalone export_prompt_keywords_to_csv is live via Prompts_Interop and its test stays), Client_Media import_obsidian_note_to_db + the fetch_keywords_for_media module wrapper (two tests swapped to the live batch variant), ccv3_parser.py 0-byte ghost + stale comment, ChatbookCreationWindow.py + its dedicated test + modal-tier entries + the now-orphan wide-viewport CSS rule (source sheet edited + bundle rebuilt via build_css.py per ADR-150). DESCOPED with AC updated: MCP chat_with_character stays (ADR-053 pins the not-available-yet contract; five test files pin the surface) - needs its own decision task if removal is wanted. Verification: all affected tests green or matching the pre-existing baseline failure set exactly (modal_wide_tier 3 baseline failures prompts_db_legacy 1 css-allowlist 1; ingest worker + Chatbooks dirs proven red pre-change via stash runs: 172 vs 177 stashed).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
