---
id: TASK-32528
title: Preserve reasoning prefill attribution through tool replay and budgets
status: To Do
assignee: []
created_date: '2026-09-13 03:21'
labels: []
dependencies:
  - TASK-32521
  - TASK-32525
  - TASK-32526
  - TASK-32527
documentation:
  - Docs/superpowers/specs/2026-09-12-console-native-reasoning-prefill-design.md
  - Docs/superpowers/plans/2026-09-12-console-native-reasoning-prefill.md
  - backlog/decisions/159-console-native-reasoning-prefill.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep authored seeds distinct from generated reasoning while preserving exact provider-required history and input accounting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Only the primary first model call injects the seed; later tool rounds replay the completed owner record with no extra injection and no child inheritance.
- [ ] #2 Exact adapter echo contracts produce generated-only display thinking; required continuation preserves replay bytes and optional suffix-only replay fails or omits according to policy.
- [ ] #3 Preview, direct and agent requests budget the same projection; seed input is counted once and never manufactured as output; selected-generation ownership survives retry and persistence.
<!-- AC:END -->
