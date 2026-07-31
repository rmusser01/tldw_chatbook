---
id: TASK-1611
title: >-
  Target steering and second-target authoring
status: To Do
assignee: []
created_date: '2026-07-31 15:10'
labels:
  - evals
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Whole-branch review of task-1482, Important 2 (scope boundary). The UI can create exactly ONE eval_models row per install (the bench editor's create-target button renders only when zero llama_cpp models exist; the sample bench reuses rather than creates), so the Δ baseline Spread and per-target Probe comparisons stay single-column for real users. The design spec's bench mock shows steering variants ("llama+prefix") as distinct targets; `Target.prefix`/`system_prompt` and the snapshot format already support them but nothing writes them (`eval_models.config` is the natural home). This task adds: creating additional targets from the editor (name + optional prefix/system_prompt against the configured server), making `Target.is_valid_for_mode` production-reachable, and rewording the mode-revalidation copy that currently names settings with no UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A second (and Nth) target can be created from the bench editor when models already exist
- [ ] A target can carry a prefix (raw mode) or system prompt (chat mode), persisted and used by the capture request
- [ ] Prompt-mode switching revalidates steered targets with user-readable copy
- [ ] Multi-target Δ column baseline is reachable end-to-end through the UI
<!-- AC:END -->
