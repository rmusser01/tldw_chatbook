---
id: TASK-33509
title: Add Command Code as an engine preset
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-29 19:29'
updated_date: '2026-09-29 20:03'
labels:
  - providers
  - engine
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Command Code's Provider API is a pay-as-you-go multi-model gateway on OpenAI Chat Completions that oh-my-pi and Hermes both support; its Claude models are served only on the Messages API.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user with a Command Code Provider API key can chat with seeded Chat Completions models through the engine
- [x] #2 Claude models, which are Messages-only, are not offered
- [x] #3 The Settings guide documents setup
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. COMMANDCODE record: api.commandcode.ai/provider/v1, Bearer, seeded Chat Completions models (no Claude), usage always streamed.
2. Hand lists, config, tests, Settings guide.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
COMMANDCODE record (key commandcode): api.commandcode.ai/provider/v1, Bearer COMMANDCODE_API_KEY (hermes-agent's name; oh-my-pi accepts it too; Command Code's docs name none). Usage is always streamed (documented), so no stream_options and usage required. Seeded with 15 models whose /models row lists /chat/completions (hermes' fallback list + four newer); Claude (Messages-only) excluded and a test pins no claude seed. OpenAI-schema message refusal/annotations allowed (annotations: [] via the empty-list rule).
<!-- SECTION:NOTES:END -->
