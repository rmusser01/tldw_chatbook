---
id: TASK-36899
title: Keep pending quiz Start within its runtime and scope lifetime
status: To Do
created_date: 2026-10-25 19:32
labels:
- textual
- study
- correctness
- book-audit
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-25-textual-book-after-pr3247.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep a quiz Start result associated with the runtime and Study scope in which it was requested, so changing context cannot overwrite current work or send an old attempt to a different backend.

### Evidence and bounded scope

Verified after PR #3247 merged, on dev `79da83a8dca98eab6ca854a7505b8abd264eb959`. A held server Start followed by the production server-to-local callback leaves the same answer widget attached. Its eventual response publishes server attempt 21 into the local panel and clears a new-context draft; enabled Submit then calls the actual LocalQuizService and SQLite with that old attempt and fails. Global-to-workspace-B and server-to-local-to-server transitions also republish the old question after reset. Ordinary server Start and Submit work. The mounted production screen/controller and real service/SQLite boundaries are used; the API client is a controlled fake. No full TldwCli runtime-policy commit, active-server binding replacement, live server, remote rollback or durable corruption is claimed.

Affected source: tldw_chatbook/UI/Study_Modules/quizzes_handler.py:721-777,794-839; tldw_chatbook/UI/Screens/study_screen.py:1210-1246,1442-1449; tldw_chatbook/Study_Interop/quiz_scope_service.py:361-399.

Book basis: Chapter 7 PDF145-146/158 (printed136-137/149), independent queues and asynchronous operations; Chapter 17 PDF335 (printed326), cancellation-aware worker publication. The application-specific context lifetime requirement follows the witnessed reset and incorrect backend routing.

### Existing ownership and decisions

TASK-36729 covers responsive shell Start, shared admission, and removed/replaced quiz panels; this witness preserves the attached panel while resetting its runtime/scope lifetime. TASK-34752 owns question-list rebuild overlap and teardown, neither of which causes this witness. TASK-36099 owns quiz creation drafts; TASK-36360 owns final-answer retry. TASK-643/644/646 establish application authorities and handoffs rather than this controller-local pending completion. Their complete criteria were checked. No matching open PR or remote-only owner was found.

Related tasks: [TASK-36729](./task-36729%20-%20Keep-Study-navigation-responsive-while-shell-Start-quiz-awaits-the-server.md), [TASK-34752](./task-34752%20-%20Study-quizzes-one-writer-for-the-question-list.md), [TASK-36099](./task-36099%20-%20Keep-newer-Study-quiz-drafts-when-earlier-creates-complete.md), [TASK-36360](./task-36360%20-%20Keep-quiz-final-answer-retry-from-duplicating-answers-and-inflating-scores.md), [TASK-643](./task-643%20-%20Make-runtime-policy-the-sole-application-runtime-source-authority.md), [TASK-644](./task-644%20-%20Move-cross-visit-screen-snapshots-behind-an-in-memory-owner.md), [TASK-646](./task-646%20-%20Complete-destination-handoff-ownership-and-ACP-target-recovery.md).

ADR required: no.
ADR paths: [ADR-011](../decisions/011-chatbook-workbench-ui-system.md), [ADR-033](../decisions/033-application-session-state-ownership.md).
Reason: A bounded publication correction under existing domain-screen and runtime ownership. No new durable attempt store or service contract is proposed; re-evaluate if implementation changes those boundaries.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-25-textual-book-after-pr3247.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Start completion from a runtime or Study scope that has since been reset cannot replace the current attempt, question, status or answer draft, or expose an old attempt for Submit through the new backend.
- [ ] #2 Changing away and back before completion does not revive an obsolete Start; existing protection for removed/replaced quiz panels and shared duplicate admission remains effective.
- [ ] #3 Ordinary same-context Start/Submit and failure recovery remain usable, and any already-created remote attempt is described truthfully without claiming cancellation rolled it back.
- [ ] #4 Mounted production-owner regressions with held responses cover runtime change, scope change and away/back transitions, including the real local SQLite submission boundary; ordinary completion and retry controls pass without background exceptions.
<!-- AC:END -->
