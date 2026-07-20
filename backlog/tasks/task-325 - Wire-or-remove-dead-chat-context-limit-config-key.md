---
id: TASK-325
title: Wire or remove the dead chat_context_limit config key
status: To Do
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
- [ ] #1 `chat_context_limit` either drives a real message-count fallback cap in the Console history builder, or is removed from config and the sample TOML
- [ ] #2 If kept, its effect is covered by a test; if removed, no references remain
<!-- AC:END -->
