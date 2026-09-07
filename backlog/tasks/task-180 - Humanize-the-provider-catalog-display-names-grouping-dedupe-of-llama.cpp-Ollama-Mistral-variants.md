---
id: TASK-180
title: >-
  Humanize the provider catalog: display names, grouping, dedupe of
  llama.cpp/Ollama/Mistral variants
status: Done
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
- [x] #1 Every provider option has a human display name distinct from its config key
- [x] #2 Options are grouped Cloud vs Local (or equivalent)
- [x] #3 Redundant llama.cpp/Ollama/Mistral variants are merged or explained inline
- [x] #4 Dropdown supports type-to-filter or an equivalent quick-find
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on branch claude/uat-core-loop-2026-07 (PR #606, commits 6fd4a60f..88c0475b) with focused tests; re-verified live against llama.cpp on a fresh profile (remediation captures in Docs/superpowers/qa/core-loop-uat-2026-07).
<!-- SECTION:NOTES:END -->
