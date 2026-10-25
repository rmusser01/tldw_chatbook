---
id: TASK-36729
title: Keep Study navigation responsive while shell Start quiz awaits the server
status: In Progress
created_date: 2026-10-25 03:47
labels:
- study
- textual-patterns
- responsiveness
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-24-textual-book-after-pr3243.md
assignee:
- '@codex'
updated_date: 2026-10-25 17:48
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Study shell Start quiz action must leave section navigation usable while a server attempt request is pending. Users should not wait for the remote response before Dashboard can open.

### Evidence and bounded scope

Verified after PR #3243 merged, on dev `139333920d2a87cb011e4f3aa5f8319b59c44a5c`. A mounted production StudyScreen/StudyWindow probe sends real Button.Pressed events through the actual QuizScopeService and ServerQuizService to a held fake-client request. Shell #quiz-start leaves a later Dashboard event queued until release. Under the identical hold, inner #start-quiz-attempt-button uses its existing worker and Dashboard opens before release. An independent App Ping is processed during both holds: the defect is screen-queue starvation, not a blocked asyncio event loop. Both paths start exactly once and complete. Full TldwCli boot, physical pointer gestures, natural latency, Escape and actual-unmount publication were not measured.

Affected source: tldw_chatbook/UI/Screens/study_screen.py:1635-1641; UI/Study_Window.py:1178-1184; UI/Study_Modules/quizzes_handler.py:741; Study_Interop/server_quiz_service.py:208-211.

Book basis: Chapter7 PDF158 (printed149) explains that an awaited handler defers later messages on that queue; PDF145-146 (printed136-137) explains independent widget queues. The inner Start action already supplies the local worker precedent.

### Existing ownership and decisions

TASK-1320 owns mount I/O and TASK-2834 the synchronous blocking-call guard; TASK-15471 and PERF tasks own named local costs. Full PERF-25/TASK-33284 and PERF-30/TASK-33289 criteria contain no all-Study response-time promise or pending server Start. TASK-34752 owns question-list concurrency/panel teardown; TASK-34755 and TASK-36099/36132 own separate settled statuses and draft retention. Preserve those fixes rather than duplicating them.

Related tasks: [TASK-1320](./task-1320%20-%20Move-screen-mount-IO-off-the-App-message-pump.md), [TASK-2834](./task-2834%20-%20Guard-against-blocking-IO-reachable-from-a-Textual-message-handler.md), [TASK-15471](./task-15471%20-%20Event-loop-I-O-sundries---per-click-writes-and-lookups-off-the-loop.md), [TASK-32898](./task-32898%20-%20Work-stream-give-the-scope-service-scaffold-one-home.md), [TASK-33284](./task-33284%20-%20PERF-25-Other-screens---Evals-Change-Review-Speech-video-player-splash.md), [TASK-33289](./task-33289%20-%20PERF-30-Cold-feature-and-hygiene-sweep-P2-P3-across-TTS-Audio-Evals-Chunking-interop.md), [TASK-34752](./task-34752%20-%20Study-quizzes-one-writer-for-the-question-list.md), [TASK-34755](./task-34755%20-%20Study-the-Deck-created.-status-is-always-replaced-by-No-cards-due-for-review..md), [TASK-36099](./task-36099%20-%20Keep-newer-Study-quiz-drafts-when-earlier-creates-complete.md), [TASK-36132](./task-36132%20-%20Preserve-newer-Flashcards-drafts-when-earlier-creates-complete.md).

ADR required: no.
ADR paths: [011-chatbook-workbench-ui-system](../decisions/011-chatbook-workbench-ui-system.md), [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md).
Reason: Bounded scheduling of the existing Start action within current screen/domain-service ownership. No new attempt, storage, runtime or cross-view policy is needed. Re-evaluate if implementation changes durable attempt ownership.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-24-textual-book-after-pr3243.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With the production service awaiting a controlled server Start response, the Study shell Start quiz action allows a subsequent enabled Dashboard section action to take effect before that response is released.
- [ ] #2 An ordinary successful shell Start still creates one attempt and presents its result; existing validation/refusal and failed-request feedback remain recoverable without duplicate dispatch.
- [ ] #3 The inner Start attempt path retains its current responsive behavior, and existing question-list, draft, panel-retirement and attempt ownership contracts remain unchanged.
- [ ] #4 Mounted production-owner tests compare shell and inner Start under the same held service boundary, prove independent App-queue responsiveness, and retain ordinary completion and error controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconfirm the shell handler and existing inner Start/controller/service contracts using the audit and ADRs. 2. Add actual mounted event tests at a held server-client boundary for responsive Dashboard navigation, repeated shell/inner Start admission, ordinary outcomes and failure/refusal/retry. Observe expected RED before production edits. 3. Move shell continuation into a named screen-owned worker and preserve one admitted pending attempt in the shared controller; repeated inner starts must not cancel it. 4. Run all new tests, existing attempt retry and relevant Study shell controls, scoped lint/format and source guards, then fresh Superpowers review. 5. Record evidence, check ACs and mark Done through CLI only when verification is complete. ADR required: no. ADR path: backlog/decisions/011-chatbook-workbench-ui-system.md; backlog/decisions/033-application-session-state-ownership.md. Reason: localized scheduling/admission within existing UI and service owners; no new storage/runtime/attempt ownership policy. Full plan: Docs/superpowers/plans/2026-10-25-textual-reminder-and-quiz-followups.md.
<!-- SECTION:PLAN:END -->
