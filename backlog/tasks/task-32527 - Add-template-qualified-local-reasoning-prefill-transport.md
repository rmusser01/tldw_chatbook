---
id: TASK-32527
title: Add template-qualified local reasoning prefill transport
status: To Do
assignee: []
created_date: '2026-09-13 03:20'
updated_date: '2026-09-13 03:31'
labels: []
dependencies:
  - TASK-32521
  - TASK-32523
  - TASK-32526
documentation:
  - Docs/superpowers/specs/2026-09-12-console-native-reasoning-prefill-design.md
  - Docs/superpowers/plans/2026-09-12-console-native-reasoning-prefill.md
  - backlog/decisions/159-console-native-reasoning-prefill.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Support native reasoning continuation for qualified local server and template combinations without enabling answer-prefill shortcuts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Local candidates validate server, template, model, thinking, tools, and response-prefill constraints and serialize one unfinished reasoning prefix.
- [ ] #2 The reasoning path never forces thinking off, bypasses tools, or rewrites endpoint settings; unknown custom endpoints remain Unverified.
- [ ] #3 Rendered-boundary and transport tests cover streaming and non-stream candidates; qualified flags require separate live evidence.
<!-- AC:END -->
