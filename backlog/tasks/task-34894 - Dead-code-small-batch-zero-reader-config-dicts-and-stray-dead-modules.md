---
id: TASK-34894
title: 'Dead-code small batch: zero-reader config dicts and stray dead modules'
status: To Do
created_date: 2026-10-12 17:52
labels:
- simplification
- dead-code
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Assorted verified-dead code: config.py builds legacy provider dicts that nothing reads and several stray dead modules remain across the tree.

Evidence (verified on dev df4985e): Both access paths grepped (direct dict indexing and get_cli_setting). _LEGACY_PROVIDER_API_KEY_BRIDGE is load-bearing for live providers (Tests/Chat/test_provider_readiness.py) and must stay. Dead MCP tool calls save_conversation_from_messages and chat_with_provider which unconditionally raise (tools.py:50-61). The keywords CSV helper is pinned only by Tests/Prompts_DB/test_prompts_db_legacy.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Zero-reader legacy config dict builders removed (gemini_api open_router_api perplexity_api local_huggingface_api azona_api plus comment-only google_generative_api),Live legacy bridges for openai anthropic mistral groq cohere huggingface elevenlabs custom_openai preserved,Character_Chat/ccv3_parser.py empty file and its stale comment references removed,UI/ChatbookCreationWindow.py removed,MCP tools.py chat_with_character dead body removed or repaired,DB dead helpers removed (execute_query_with_retry _add_keyword_with_retry import_obsidian_note_to_db fetch_keywords_for_media export_keywords_to_csv),Config loads identically for live providers
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
