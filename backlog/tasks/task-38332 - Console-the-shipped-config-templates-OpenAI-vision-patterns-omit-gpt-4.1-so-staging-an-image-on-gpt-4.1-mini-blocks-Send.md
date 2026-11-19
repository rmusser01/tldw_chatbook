---
id: TASK-38332
title: >-
  Console: the shipped config template's OpenAI vision patterns omit gpt-4.1, so
  staging an image on gpt-4.1-mini blocks Send
status: To Do
assignee: []
created_date: '2026-10-31 01:16'
labels:
  - console
  - attachments
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live (TASK-33621.11 verification, isolated profile with the app's own boot-written config): staging a valid PNG with OpenAI gpt-4.1-mini showed the Send block and the log read "No capability information found for openai/gpt-4.1-mini, using defaults". The [model_capabilities.patterns] OpenAI list in the default config template (config.py ~6073) has only gpt-4 vision/gpt-4o/gpt-4 turbo patterns, and a config pattern list replaces the shipped DEFAULT_MODEL_PATTERNS (which do list ^gpt-4\.1). TASK-33620.2 fixes the same class for Claude 4+ with a family rule; OpenAI's current vision models are not covered.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Staging an image with OpenAI gpt-4.1 / gpt-4.1-mini (and the other current OpenAI vision models) in a fresh profile leaves Send ready
- [ ] #2 A profile whose config.toml carries the old pattern list is covered too, and an explicit vision = false still wins
<!-- AC:END -->
