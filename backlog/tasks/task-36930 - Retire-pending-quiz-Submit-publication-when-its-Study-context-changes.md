---
id: TASK-36930
title: Retire pending quiz Submit publication when its Study context changes
status: In Progress
created_date: 2026-10-25 23:33
labels:
- textual-book
- study
- ui
- async
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-25-textual-book-after-pr3248.md
assignee:
- '@codex'
updated_date: 2026-10-26 00:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep a completed response for an earlier quiz attempt from discarding a newer attempt or its unsent answer after the Study runtime or scope changes.

### Verified scope

Verified on merged dev `7a5e2513c5d7e00ef57f3d19cd8e81a01e006d05` after PR #3248. A final server Submit held across the production runtime-reset observer and a new real LocalQuizService/SQLite Start clears the newer attempt ID and draft, paints the old score and disables Submit. SQLite retains the newer attempt incomplete with no answers. The same-context control completes correctly. Mounted production Study surface and schema/router/services; controlled API client and host runtime projection, without a full TldwCli policy commit or live server claim.

Affected source: tldw_chatbook/UI/Study_Modules/quizzes_handler.py:813,850-879; UI/Screens/study_screen.py:1210-1244,1442-1449.

Book: Chapter 17 PDF335 (printed326), checking background publication authority; Chapter 7 PDF145-146/158 (printed136-137/149), independent message queues. Applying context generations is an application deduction.

### Existing ownership and decisions

TASK-36899 and TASK-36729 are completed Start-only owners; TASK-36360 covers definite final-answer failure/retry. This finding concerns a successful pending Submit after the existing runtime reset; question-list overlap and creation drafts stay with their existing owners.

Related tasks: [TASK-36899](./task-36899%20-%20Keep-pending-quiz-Start-within-its-runtime-and-scope-lifetime.md), [TASK-36729](./task-36729%20-%20Keep-Study-navigation-responsive-while-shell-Start-quiz-awaits-the-server.md), [TASK-36360](./task-36360%20-%20Keep-quiz-final-answer-retry-from-duplicating-answers-and-inflating-scores.md), [TASK-34752](./task-34752%20-%20Study-quizzes-one-writer-for-the-question-list.md), [TASK-36099](./task-36099%20-%20Keep-newer-Study-quiz-drafts-when-earlier-creates-complete.md), [TASK-643](./task-643%20-%20Make-runtime-policy-the-sole-application-runtime-source-authority.md), [TASK-644](./task-644%20-%20Move-cross-visit-screen-snapshots-behind-an-in-memory-owner.md), [TASK-646](./task-646%20-%20Complete-destination-handoff-ownership-and-ACP-target-recovery.md).

ADR required: no.
ADR paths: [ADR-011](../decisions/011-chatbook-workbench-ui-system.md), [ADR-033](../decisions/033-application-session-state-ownership.md).
Reason: A bounded correction to controller-local publication under existing runtime/session ownership; no new durable state or service contract is proposed.

[Post-merge audit and reproducible evidence](../../Docs/superpowers/qa/2026-10-25-textual-book-after-pr3248.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Submit response from a retired runtime, scope or attempt leaves the current attempt, question, status, controls and exact unsent answer unchanged.
- [ ] #2 Runtime or scope away-and-back and answer-panel replacement do not restore an obsolete Submit response publication authority.
- [ ] #3 Ordinary current-context grading/progression and the established definite pre-write failure/retry behavior remain usable without duplicate submission.
- [ ] #4 Controlled mounted tests establish current-context success and obsolete success/failure behavior, with truthful feedback for any already completed prior-context operation and no claim of remote rollback.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add controlled regression tests and observe the reported defect before production edits.
2. Capture Submit mode, scope, generation, attempt and answer-panel identity before awaiting the existing service. Reject obsolete publication after every service await, including failures, and report prior-context completion truthfully without touching current state. Preserve nonfinal progression and definite pre-write retry semantics. Test runtime reset, scope ABA, attempt replacement and panel replacement, ordinary grading and failure/retry using mounted Study and actual local SQLite.
3. Run focused regression and adjacent owner tests, lint/format checks, update Docs/Architecture/study.md, and self-review.
4. Complete Superpowers review, register new UI coverage in the existing gate and verify CI before closure.

ADR required: no
ADR path: existing canonical ADRs 011, 033 (linked in Description).
Reason: bounded implementation of existing UI/session/store ownership; no new contract or architecture.

Implementation plan: Docs/superpowers/plans/2026-10-25-textual-followup-fixes.md (Task 1).
<!-- SECTION:PLAN:END -->
