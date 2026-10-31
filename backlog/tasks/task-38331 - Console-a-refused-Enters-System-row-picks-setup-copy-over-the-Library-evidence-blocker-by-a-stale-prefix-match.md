---
id: TASK-38331
title: >-
  Console: a refused Enter's System row picks setup copy over the
  Library-evidence blocker by a stale prefix match
status: To Do
assignee: []
created_date: '2026-10-31 01:16'
labels:
  - console
  - errors-recovery
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When Enter is refused, prompt_queue chooses which sentence the System row shows by checking whether the send gate's reason starts with "Console send blocked: Library Search/RAG". The gate's evidence copy now reads "Console send blocked: Library search has no available evidence ...", so the prefix never matches and a setup sentence wins whenever both block. Found while removing substring-matched copy selection in TASK-33621.11 (not one of its named seams). The refusal should be chosen from a typed blocker, as the Send strip now is.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A refused Enter's System row names the blocker the send gate actually stopped on, chosen from a typed code rather than the sentence's wording
- [ ] #2 A test drives a send blocked by both missing Library evidence and a provider setup gap through the real gate and asserts the System row
<!-- AC:END -->
