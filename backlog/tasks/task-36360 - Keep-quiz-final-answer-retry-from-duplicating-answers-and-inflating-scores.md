---
id: TASK-36360
title: Keep quiz final-answer retry from duplicating answers and inflating scores
status: Done
created_date: 2026-10-22 20:42
labels:
- study
- textual-patterns
- reliability
priority: medium
references:
- backlog/decisions/033-application-session-state-ownership.md
- backlog/decisions/011-chatbook-workbench-ui-system.md
documentation:
- Docs/superpowers/qa/2026-10-22-textual-retry-and-pending-drafts.md
assignee:
- '@codex'
updated_date: 2026-10-22 23:51
modified_files:
- tldw_chatbook/UI/Study_Modules/quizzes_handler.py
- Tests/UI/test_study_quiz_attempt_retry.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A retry after a definite failed final-answer submission must retain the current answer without grading it twice or persisting a score greater than the quiz total.

### Evidence and scope

Medium priority, audit order 1 of 4. At merged dev 814becf3cabc42a622c4ed6f7eeca17d4c6266e5, an actual mounted Start/Submit journey with one one-point question and a definite pre-write OSError leaves one answer in the pending controller list. Retry appends it again, delegates to real QuizScopeService/LocalQuizService and SQLite, stores two copies and score2/total1, and displays Score: 2 / 1. Ordinary success stores one answer/1-of-1 and passes. Qualified audit: one control pass and one intended score-bound failure, zero errors, 15.419s. Only the one-question identical-answer retry was demonstrated; edited-answer/multi-question controls are future requirements.

Source: UI/Study_Modules/quizzes_handler.py:779-845, UI/Study_Window.py:1186-1192; Study_Interop/quiz_scope_service.py:381-399; local_quiz_service.py:181-205; DB/ChaChaNotes_DB.py:23966-24100 (under tldw_chatbook/).

Book basis: Chapter7 printed149/PDF158 and chapter17 printed326/PDF335 informed worker/pending-state/completion tracing; quiz grading and retry correctness follow the demonstrated application contract, not a policy prescribed by the book.

### Existing ownership and decisions

TASK34752 owns question-list rebuild and post-teardown projection,34755 Flashcards status,34751 list/teardown safety,36099 quiz create drafts and36132 Flashcards create drafts. None owns pending attempt-answer accumulation. Four open PR DB overlaps have identical actual attempt methods; no open PR touches this controller or its services.

Related tasks: [TASK-34752](task-34752%20-%20Study-quizzes-one-writer-for-the-question-list.md); [TASK-34755](task-34755%20-%20Study-the-Deck-created.-status-is-always-replaced-by-No-cards-due-for-review..md); [TASK-34751](task-34751%20-%20Study-flashcards-one-writer-for-the-card-list-and-card-deck-text-shown-literally.md); [TASK-36099](task-36099%20-%20Keep-newer-Study-quiz-drafts-when-earlier-creates-complete.md); [TASK-36132](task-36132%20-%20Preserve-newer-Flashcards-drafts-when-earlier-creates-complete.md).

ADR required: no new ADR for the bounded correction.
ADR paths: [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md); [011-chatbook-workbench-ui-system](../decisions/011-chatbook-workbench-ui-system.md).
Reason: No new ADR is expected for a controller-local pending-answer correction. Preserve existing service, grading and storage contracts; cross-server idempotency or DB duplicate-answer policy would be separate scope.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-22-textual-book-after-pr3207.md). These are findings and future regression requirements, not implemented fixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A final-answer definite pre-write failure followed by retry sends and persists exactly one answer for each quiz question, and the resulting score does not exceed the total possible score.
- [x] #2 Changing the final answer after that failure uses its current value once; earlier completed-question answers remain intact.
- [x] #3 Ordinary successful multi-question attempts preserve grading/history behavior, and definite failure leaves the current question, answer and retry usable.
- [x] #4 Bounded mounted regressions exercise actual Start/Submit dispatch and independently verify service payload, real SQLite answers/score and visible result, retaining the ordinary passing control; uncertain-commit or server-wide idempotency is not introduced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: existing ADR paths in this task. Reason: bounded correction within the current owner; no new storage, provider or application boundary.

1. Read the linked task, ADRs, current owner tests and retained audit evidence.
2. Add and run mounted behavior regressions RED, then implement the smallest owner-local correction: Build each final submission from prior completed answers plus the current answer without appending it to persistent pending state before the awaited submission succeeds. Preserve existing question progression and grading.
3. Verify targeted new/adjacent tests and static checks; obtain independent Superpowers task review.
4. Record evidence, update notes/ACs and relevant documentation, then mark Done via CLI.

Implementation plan: Docs/superpowers/plans/2026-10-22-textual-retry-and-pending-drafts.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Build final-answer submissions from prior completed answers plus the current answer, without appending the last answer to pending state before success. Retry uses the current answer exactly once and preserves prior grading/history. Changed quizzes_handler.py and added mounted test_study_quiz_attempt_retry.py with real scope/local services and SQLite. RED: three intended failures/two controls pass; focused GREEN:36pass. Broader legacy startup failures are exact-base qualified, not claimed passing. No new ADR: existing ADR-011 and ADR-033 application-session-state-ownership apply; no storage/service/idempotency policy change. Independent Superpowers task review Approved. New concrete nodes are in the required census. See Docs/superpowers/qa/2026-10-22-textual-retry-and-pending-drafts.md for source-bound receipts, static checks and integration limits.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
