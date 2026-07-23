---
id: TASK-325
title: Wire or remove the dead chat_context_limit config key
status: Done
assignee: []
created_date: '2026-07-20 18:45'
labels: [config, tech-debt]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`chat_context_limit = 10` is defined in config (`config.py:80` and the sample TOML at `config.py:2315`, under `[rag_search]`) but is never read anywhere in the codebase. A user who sets it expecting the conversation to be capped at 10 turns gets a silent no-op with no warning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `chat_context_limit` either drives a real message-count fallback cap in the Console history builder, or is removed from config and the sample TOML
- [x] #2 If kept, its effect is covered by a test; if removed, no references remain
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Removed** `chat_context_limit` (chose removal over wiring, user-approved). It was a message-count value misplaced under `[rag_search]` and never read anywhere; it is superseded by task-322's model-aware token bound (strictly better than a fixed turn cap), and wiring a second overlapping cap under the wrong config section would be confusing.

Deleted the key from `DEFAULT_RAG_SEARCH_CONFIG` and from the sample TOML `[rag_search]` block in `config.py`. Verified zero references remain dev-wide (only `config.py` contained it; no runtime consumer, no tests). No migration needed — the config loader merges over defaults, so a leftover key in an existing user file is a harmless no-op.

Testing: asserts `chat_context_limit` is absent from both the loaded `DEFAULT_RAG_SEARCH_CONFIG` dict and the `config.py` source text (catches the sample TOML too).

Files: `tldw_chatbook/config.py`, `Tests/Chat/test_config_no_chat_context_limit.py`.
<!-- SECTION:NOTES:END -->
