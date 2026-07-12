---
id: TASK-178
title: >-
  Console Settings modal: persist session overrides or label their scope; real
  boolean controls
status: To Do
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
- [ ] #1 Modal changes either persist or the modal states clearly that they are session-only with a path to persist,Boolean settings use a toggle/cycle control instead of free text,Enumerated fields show their accepted values
<!-- AC:END -->
