---
id: TASK-36099
title: Keep newer Study quiz drafts when earlier creates complete
status: To Do
created_date: 2026-10-22 01:25
labels:
- study
- textual-patterns
- drafts
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-21-textual-book-after-pr3199.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Study quiz and question creation must not clear newer authoring text entered while an earlier create is pending. Persisting a submitted quiz or question must leave the next unfinished draft usable.

### Evidence and scope

Medium priority, audit order 3 of 4. Relevant source blobs on fetched merged dev `2a5afa03b31f60e00bdcd10a161e3e351fad023b` are identical to diagnostic candidate `8ff5f0e2d4f3016445545e84f9e30a5233b8fddd`. Two mounted StudyWindow witnesses used actual create Button.Pressed handlers, keyboard edits, real QuizScopeService/LocalQuizService and SQLite. An asyncio.Event held the scope-service await, then delegated to real persistence. Both live field pairs were edited with -new while enabled; completion saved only the original name/description or question/answer, then erased both newer fields. The suspension models an awaited service response; no live server or ordinary non-suspending local overlap was exercised.

Source: [tldw_chatbook/UI/Study_Modules/quizzes_handler.py:621](https://github.com/rmusser01/tldw_chatbook/blob/2a5afa03b31f60e00bdcd10a161e3e351fad023b/tldw_chatbook/UI/Study_Modules/quizzes_handler.py#L621); [tldw_chatbook/UI/Study_Modules/quizzes_handler.py:671](https://github.com/rmusser01/tldw_chatbook/blob/2a5afa03b31f60e00bdcd10a161e3e351fad023b/tldw_chatbook/UI/Study_Modules/quizzes_handler.py#L671); [tldw_chatbook/UI/Study_Window.py:1146](https://github.com/rmusser01/tldw_chatbook/blob/2a5afa03b31f60e00bdcd10a161e3e351fad023b/tldw_chatbook/UI/Study_Window.py#L1146).

Book basis: Chapter 7 printed 149 / PDF 158 explains awaited handlers/background work. Chapter 9 printed 170-171 / PDF 179-180 and printed 180 / PDF 189 explain ownership of editable widget state and focus. The specific newer-draft outcome comes from the current create-completion behavior.

These are audit findings, not implemented fixes. The report distinguishes demonstrated cases from future regression controls.

### Existing ownership and decisions

TASK-34752 To Do owns overlapping question-list rebuilds and writes after view teardown, not clearing editable fields on create completion. Keep that independent list-writer task intact. TASK-34751 Done owns Flashcards card-list ownership, swap guards and literal text. Its notes identify historical [PR #3057](https://github.com/rmusser01/tldw_chatbook/pull/3057) commits tagged TASK-33007; TASK-33007 itself is completed Settings Providers/Models work, not a quiz-draft owner. Neither contract covers the demonstrated create-completion drafts.

Related tasks: [TASK-34752](task-34752%20-%20Study-quizzes-one-writer-for-the-question-list.md); [TASK-34751](task-34751%20-%20Study-flashcards-one-writer-for-the-card-list-and-card-deck-text-shown-literally.md); [TASK-33007](task-33007%20-%20Phase-7-Reorder-Settings-Providers-Models-into-Connect-Default-model-Model-defaults-Advanc.md).

ADR required: no new ADR for this bounded repair.
ADR paths: [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md); [011-chatbook-workbench-ui-system](../decisions/011-chatbook-workbench-ui-system.md).
Reason: No new ADR is needed for retaining screen-local authoring state across an existing service result. Accepted ADR-033/011 govern narrow state ownership and stable editable composition. Existing service, persistence, runtime-policy, list-writer and cross-screen lifetime contracts stay independently owned.

[Post-merge book audit and evidence](../../Docs/superpowers/qa/2026-10-21-textual-book-after-pr3199.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A quiz create completing after newer name or description edits preserves those exact raw values without implicitly persisting them.
- [ ] #2 A question create completing after newer edits to the question text or correct answer preserves those values and their active editing context.
- [ ] #3 Successful unchanged submissions still persist once and clear their submitted form normally; failed creates preserve current drafts and permit retry.
- [ ] #4 Bounded mounted regressions drive both existing create handlers through controlled awaits and verify live fields separately from actual stored records. TASK-34752 question-list ownership remains independently scoped.
<!-- AC:END -->
