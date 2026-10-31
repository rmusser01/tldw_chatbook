---
id: TASK-36930
title: Retire pending quiz Submit publication when its Study context changes
status: Done
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
updated_date: 2026-10-26 03:37
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
- [x] #1 A Submit response from a retired runtime, scope or attempt leaves the current attempt, question, status, controls and exact unsent answer unchanged.
- [x] #2 Runtime or scope away-and-back and answer-panel replacement do not restore an obsolete Submit response publication authority.
- [x] #3 Ordinary current-context grading/progression and the established definite pre-write failure/retry behavior remain usable without duplicate submission.
- [x] #4 Controlled mounted tests establish current-context success and obsolete success/failure behavior, with truthful feedback for any already completed prior-context operation and no claim of remote rollback.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Submit-owned publication authority across runtime/scope generations, attempt and retained question-list identity, and the original answer panel. Rechecks after Submit and its history await cover obsolete success and failure without replacing the current attempt, exact unsent draft, controls or status. Ordinary grading and definite pre-write retry semantics remain intact; feedback identifies the completed prior-context operation without claiming remote rollback.

Changed UI/Study_Modules/quizzes_handler.py and Docs/Architecture/study.md; added Tests/UI/test_study_quiz_submit_lifetime.py. Final 15 mounted cases and 15 actual private-child reports pass, including history success/failure after retirement; 11 adjacent Start/retry children and 12 Study keyword/contract guards passed. Scoped lint/format checks introduce no new findings.

ADR required: no. Existing [ADR-011](../decisions/011-chatbook-workbench-ui-system.md) and [ADR-033](../decisions/033-application-session-state-ownership.md) govern this bounded controller/session correction. No new service, state or architecture boundary.

Superpowers task, whole-branch, fix-wave and CI-delta reviews are complete with no open findings; Qodo was replaced as requested. Current implementation-head hosted CI is green: [40261190139](https://github.com/rmusser01/tldw_chatbook/actions/runs/40261190139); [40261190218](https://github.com/rmusser01/tldw_chatbook/actions/runs/40261190218). Its 17 UI shards executed all 36 new exact cases once; maximum wall time 1094s under the unchanged 1200s cap. The gate preserves all 899 live-dev selectors plus 36 new cases (floor935). Rebase onto dev e554f25812 preserved all nine reviewed source/test blobs.

[Verification and explicit scope decisions](../../Docs/superpowers/qa/2026-10-25-textual-followup-fixes.md) records exact targeted evidence and the retained baseline profile/bootstrap, Chat-size and initial Chunking timing qualifications. No full local suite or native/live-server qualification is claimed.

[PR #3251](https://github.com/rmusser01/tldw_chatbook/pull/3251) targets dev.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
