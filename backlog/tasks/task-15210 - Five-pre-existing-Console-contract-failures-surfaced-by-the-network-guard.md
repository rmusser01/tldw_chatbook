---
id: TASK-15210
title: >-
  Five pre-existing Console contract failures surfaced by the network guard
status: To Do
assignee: []
created_date: '2026-08-11 07:00'
labels:
  - console
  - tests
  - dev-baseline
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while implementing task-15111. Running the nine Console modules the socket shim had caught reaching the network gave `147 passed, 4 failed, 0 blocked-egress hits` — the four failures are **not** caused by the guard: they were re-run with the guard and both mechanism fixtures disabled and failed identically. A fifth was found in the same sweep.

1. `section:starred` collapse preference.
2. `ConsoleChatController._turn_context_provider` — `AttributeError`.
3. and 4. Two auto-RAG ordering contracts.
5. `test_console_command_popup::test_slash_opens_popup_and_typing_filters` pins a **6-item** slash list; `/generate-video` and `/stream-video` (task-3401.5, already on dev) grew it to **8**.

The fifth is the clearest and sets the pattern: a list-length pin that a legitimate feature addition broke, sitting red until something happened to run the file whole. That is the same stale-contract class as task-14920, where twenty such failures had accumulated unnoticed and one of them was hiding a real product bug that shipped.

Triage before repair, as in task-14920: a pinned count or attribute that a deliberate change moved is a test fix, but `_turn_context_provider` raising `AttributeError` and auto-RAG *ordering* changing are both shapes that can be real regressions. Do not rewrite to green without naming the commit that changed each behaviour and reading its intent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Each of the five is classified as a stale pin or a real regression, with the causing commit named and its intent read
- [ ] #2 Real regressions are fixed in the product; stale pins are updated while preserving the original claim (assert the behaviour, not a count or a class string that the next honest change breaks again)
- [ ] #3 The affected Console test modules run WHOLE with READ pass counts and no unexpected failures
<!-- AC:END -->
