---
id: TASK-188
title: >-
  Auto-detect local LLM servers during onboarding and populate models from
  /v1/models
status: To Do
assignee: []
created_date: '2026-07-12 03:05'
labels:
  - ux
  - enhancement
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Upgrade opportunity from core-loop UAT 2026-07-11: a user already running llama.cpp or Ollama locally still walks a cloud-shaped setup. During first-run (and in Settings), probe the common local endpoints (llama.cpp :8080, Ollama :11434, plus the configured api_url) and offer one-click connect; for OpenAI-compatible local providers, populate the model field from /v1/models instead of free text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 First-run setup offers a detected reachable local server as a one-click provider option,Local OpenAI-compatible providers get a model picker fed by /v1/models with free-text fallback,No probe traffic leaves localhost without user action
<!-- AC:END -->
