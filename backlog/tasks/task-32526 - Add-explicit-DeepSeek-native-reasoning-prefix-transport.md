---
id: TASK-32526
title: Add explicit DeepSeek native reasoning prefix transport
status: To Do
assignee: []
created_date: '2026-09-13 03:20'
updated_date: '2026-09-13 03:31'
labels: []
dependencies:
  - TASK-32521
  - TASK-32523
  - TASK-32525
documentation:
  - Docs/superpowers/specs/2026-09-12-console-native-reasoning-prefill-design.md
  - Docs/superpowers/plans/2026-09-12-console-native-reasoning-prefill.md
  - backlog/decisions/159-console-native-reasoning-prefill.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Carry native reasoning prefixes through the real DeepSeek Chat Completions wire while refusing unqualified modes and combinations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A native DeepSeek candidate preserves exact reasoning input, required prefix fields, and the configured endpoint through streaming and non-stream transport.
- [ ] #2 Unverified or unsupported targets, API modes, thinking settings, tools, and response-prefill combinations fail before network contact without fallback.
- [ ] #3 Qualification status stays Unverified until versioned native and echo evidence is recorded; ordinary unprefilled requests remain unchanged.
<!-- AC:END -->
