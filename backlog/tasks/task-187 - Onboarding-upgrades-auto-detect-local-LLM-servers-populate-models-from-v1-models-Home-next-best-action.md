---
id: TASK-187
title: >-
  Onboarding upgrades: auto-detect local LLM servers, populate models from
  /v1/models, Home next-best-action
status: To Do
assignee: []
created_date: '2026-07-12 02:48'
labels:
  - ux
  - enhancement
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Upgrade opportunities from core-loop UAT 2026-07-11: a user running llama.cpp locally still walks a 9-step cloud-shaped setup. The app could probe common local endpoints (llama.cpp/Ollama defaults) during first-run and offer one-click connect; the model field for OpenAI-compatible local providers could be populated from /v1/models instead of free text; Home should reflect real state (provider ready -> 'Start a conversation', N conversations, last note) instead of a single import card on an empty canvas. Also reconsider the Save-as WIP destinations (task-91 made them honest; hide or wire them next).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 First-run offers a detected local server as a one-click provider option when one is reachable,Local OpenAI-compatible providers offer a model picker fed by /v1/models with free-text fallback,Home surfaces a start-a-conversation action when the provider is ready and reflects existing content counts,Save-as dialog either hides unwired destinations or ships them
<!-- AC:END -->
