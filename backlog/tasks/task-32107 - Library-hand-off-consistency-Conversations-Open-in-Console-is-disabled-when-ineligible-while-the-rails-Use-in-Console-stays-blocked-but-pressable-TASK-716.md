---
id: TASK-32107
title: >-
  Library hand-off consistency: Conversations 'Open in Console' is disabled when
  ineligible while the rail's Use-in-Console stays blocked-but-pressable
  (TASK-716)
status: To Do
assignee: []
created_date: '2026-09-08 22:43'
labels:
  - library
  - console
  - ux
  - design-decision
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32056 (PR #2523) disables the conversation reader's 'Open in Console' for an unlinked conversation and offers 'Link to workspace' beside it, which its AC required; the rail's `#library-use-in-console` deliberately stays pressable-with-reason (TASK-716). Two blocked-action grammars now coexist in Library; the critique-8 improvement list asked for one 'Send to Console' verb and placement. Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A recorded decision picks one blocked-action grammar for Console hand-offs in Library
- [ ] #2 The other surface is aligned to it, or the difference is documented with its reason
<!-- AC:END -->
