---
id: TASK-484
title: >-
  Fix builtin RAG profiles with invalid chunking_method values
status: To Do
assignee: []
created_date: '2026-07-22 01:45'
updated_date: '2026-07-22 01:45'
labels:
  - rag
  - profiles
  - followup
dependencies:
  - task-483
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from SP2a (task-483) Task 2 review. Three builtin profiles in `tldw_chatbook/RAG_Search/config_profiles.py` set the CORRECT field `chunking_method` but to values that are NOT in the pipeline's valid set (words/sentences/paragraphs/tokens/semantic/json/ebook_chapters/xml/rolling_summarize): `hybrid_full` and `research_papers` use `"hierarchical"`, `technical_docs` uses `"structural"`. These are silently unused today, but if the enhanced-chunking code path (`Chunker.chunk_text`) is ever exercised for a profile with these values it raises `InvalidChunkingMethodError`. This is a different bug class from the SP2a dead-attribute fix (correct field, invalid value), so it was left as a follow-up.

Decide the correct valid `chunking_method` for each (these builtins also set `preserve_structure`; "paragraphs" is the closest structure-respecting valid method) and set it, or remove the line to fall back to the default. Needs a small design decision on intended chunking behavior for each.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Every builtin profile's `chunking_method` is a value accepted by the runtime chunker (no `InvalidChunkingMethodError` possible from any builtin).
- [ ] A test asserts all builtins use a valid `chunking_method`.
<!-- AC:END -->
