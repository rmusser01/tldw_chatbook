---
id: TASK-36099
title: Keep newer Study quiz drafts when earlier creates complete
status: Done
created_date: 2026-10-22 01:25
labels:
- study
- textual-patterns
- drafts
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-21-textual-save-and-source-freshness.md
assignee:
- '@codex'
updated_date: 2026-10-22 05:37
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
- [x] #1 A quiz create completing after newer name or description edits preserves those exact raw values without implicitly persisting them.
- [x] #2 A question create completing after newer edits to the question text or correct answer preserves those values and their active editing context.
- [x] #3 Successful unchanged submissions still persist once and clear their submitted form normally; failed creates preserve current drafts and permit retry.
- [x] #4 Bounded mounted regressions drive both existing create handlers through controlled awaits and verify live fields separately from actual stored records. TASK-34752 question-list ownership remains independently scoped.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md
Reason: Retain current local authoring fields across existing service results; TASK34752 continues to own question-list serialization/view teardown.
1. Read task/ADRs/design rules, current create handlers and retained controlled-await real-SQLite audit witnesses.
2. Add Tests/UI/test_study_quiz_create_drafts.py using actual create event/worker and current private-profile harness. Hold each existing create await, type newer name/description or question/answer, and assert current raw draft versus real persisted submission; include unchanged clearing, failure/retry and editing-context controls. Run red before product changes.
3. Clear only fields still matching the submitted form on successful quiz/question create, retaining later authoring text and current context. Leave service, runtime and independent question-list ownership unchanged.
4. Run targeted new/adjacent Study tests and scoped lint/format; document controlled-await versus live-server limits and notes. Root integrates census and independent Superpowers review before Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Preserve exact raw quiz/question submissions and clear each authoring field only if it still matches that submission after successful creation. Existing normalized payloads, actual SQLite/local/scope services, policy boundaries and refresh ownership remain unchanged. Four mounted regression nodes cover independent/both-field/whitespace edits, current focus/selection, unchanged success and failed-create retry. Final targeted verification: 10 passed (final-green-6f0ebe75); scoped Ruff lint/format and whitespace checks pass. Three legacy adjacent admission failures and one teardown error reproduce against original handler bytes; no guards weakened. Independent Superpowers review found no remaining issue in this task. Existing ADR-033 application-session-state and ADR-011 apply; no new ADR. Files: quizzes_handler.py and test_study_quiz_create_drafts.py. Evidence and limits: Docs/superpowers/qa/2026-10-21-textual-save-and-source-freshness.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
