---
id: TASK-433
title: Fix llama.cpp prefilled endpoint default to server root
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
labels:
  - settings
  - llm-calls
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). The Settings default/hint for llama.cpp is http://localhost:8080/completion, but the chat caller always appends v1/chat/completions unless the URL already ends with it (LLM_API_Calls_Local.py:213-222), producing .../completion/v1/chat/completions. A user who keeps the suggested default gets failures after a green-looking setup. Either fix the default to the server root or make the caller strip/tolerate the legacy /completion suffix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user who accepts the prefilled llama.cpp endpoint and a running llama-server on that port gets working chat completions
- [ ] #2 Endpoint guidance text matches what the caller actually does with the value
<!-- AC:END -->
