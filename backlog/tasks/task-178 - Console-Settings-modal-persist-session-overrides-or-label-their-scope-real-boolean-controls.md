---
id: TASK-178
title: >-
  Console Settings modal: persist session overrides or label their scope; real
  boolean controls
status: Done
assignee: []
created_date: '2026-07-12 02:47'
labels:
  - ux
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Core-loop UAT 2026-07-11: values entered in the Console Settings modal (e.g. Streaming) apply to the running session only and silently vanish on restart (config keeps streaming=false). Streaming/Reasoning/Verbosity/Thinking render as blank free-text inputs with no accepted-values hint; Streaming is a boolean presented as free text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Modal changes either persist or the modal states clearly that they are session-only with a path to persist
- [x] #2 Boolean settings use a toggle/cycle control instead of free text
- [x] #3 Enumerated fields show their accepted values
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on branch claude/uat-core-loop-2026-07 (PR #606, commits 6fd4a60f..88c0475b) with focused tests; re-verified live against llama.cpp on a fresh profile (remediation captures in Docs/superpowers/qa/core-loop-uat-2026-07).
<!-- SECTION:NOTES:END -->
