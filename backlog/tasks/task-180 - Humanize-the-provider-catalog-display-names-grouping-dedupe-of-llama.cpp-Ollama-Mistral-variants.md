---
id: TASK-180
title: >-
  Humanize the provider catalog: display names, grouping, dedupe of
  llama.cpp/Ollama/Mistral variants
status: To Do
assignee: []
created_date: '2026-07-12 02:47'
labels:
  - ux
  - settings
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Core-loop UAT 2026-07-11: the Settings provider dropdown is a raw key-sorted dump of 27 internal keys - 'llama_cpp (llama_cpp)' with no display name, plus near-duplicates a new user cannot distinguish (local_llamacpp / local_llamafile vs llama_cpp, Local Ollama vs Ollama, Mistral vs MistralAI, Custom_2, zai). Also echoed as a raw 'Provider catalog:' line under Credentials.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every provider option has a human display name distinct from its config key,Options are grouped Cloud vs Local (or equivalent),Redundant llama.cpp/Ollama/Mistral variants are merged or explained inline,Dropdown supports type-to-filter or an equivalent quick-find
<!-- AC:END -->
