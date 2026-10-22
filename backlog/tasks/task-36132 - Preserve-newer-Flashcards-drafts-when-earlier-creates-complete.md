---
id: TASK-36132
title: Preserve newer Flashcards drafts when earlier creates complete
status: To Do
created_date: 2026-10-22 15:30
labels:
- study
- textual-patterns
- drafts
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-22-textual-book-after-pr3205.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Completing a submitted deck or flashcard must retain newer unsaved authoring text entered while that create is pending.

### Evidence and scope

Medium priority; audit order 3 of 3. Audited source blobs on merged dev `7700c63a688355a2df595941b8b2b4dc8a819074` match diagnostic candidate `e41fc622e2702b3e87a1f21d83f1f328ed48b821` exactly. On e41fc622e2702b3e87a1f21d83f1f328ed48b821, 2 mounted preservation witnesses fail and 2 unchanged-submission controls pass (4 nodes, 30.84s; no errors/skips). Actual Create Deck/Create Card Button.Pressed workers run in a mounted StudyWindow. An asyncio.Event holds the existing scope-service await; keyboard -new edits land in enabled deck-name or front/back/tags fields. Release delegates to the original real StudyScopeService/LocalStudyService and in-memory SQLite. SQLite contains exactly the original normalized submission once; the same current widget instances are then cleared. Selected text resets to Selection(0,0), while focus remains. Controls leave fields unchanged during the same hold and verify one persisted record plus normal clearing. This models a supported awaited response: ordinary synchronous local service calls are not claimed to naturally suspend, and no live server was tested. Initial run initial-e41fc622 failed only in fixture construction because server_service was omitted; it is retained but is not product evidence. Mixed-field, failure, raw-whitespace and focus-move cases are future regression requirements; they were not demonstrated by this audit.

Source: [tldw_chatbook/UI/Study_Modules/flashcards_handler.py:755](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/UI/Study_Modules/flashcards_handler.py#L755); [tldw_chatbook/UI/Study_Modules/flashcards_handler.py:785](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/UI/Study_Modules/flashcards_handler.py#L785); [tldw_chatbook/UI/Study_Modules/flashcards_handler.py:797](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/UI/Study_Modules/flashcards_handler.py#L797); [tldw_chatbook/UI/Study_Modules/flashcards_handler.py:842](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/UI/Study_Modules/flashcards_handler.py#L842); [tldw_chatbook/UI/Study_Window.py:1053](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/UI/Study_Window.py#L1053); [tldw_chatbook/UI/Study_Window.py:1069](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/UI/Study_Window.py#L1069); [tldw_chatbook/Study_Interop/study_scope_service.py:986](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/Study_Interop/study_scope_service.py#L986); [tldw_chatbook/Study_Interop/study_scope_service.py:1072](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/Study_Interop/study_scope_service.py#L1072); [tldw_chatbook/Study_Interop/local_study_service.py:93](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/Study_Interop/local_study_service.py#L93); [tldw_chatbook/Study_Interop/local_study_service.py:186](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/Study_Interop/local_study_service.py#L186); [tldw_chatbook/Study_Interop/server_study_service.py:173](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/Study_Interop/server_study_service.py#L173); [tldw_chatbook/Study_Interop/server_study_service.py:243](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/Study_Interop/server_study_service.py#L243).

Book basis: Attached Creating TUI Applications with Textual and Python, SHA256 798aafc3ac58704c5053249ed228f5a45629355c3627e9611a3e78da815e9ab6. Chapter 7 printed149/PDF158 explains awaited handlers/background work; Chapter17 printed326/PDF335 demonstrates worker/result lifecycle checks. Chapter9 printed170-171/PDF179-180 and printed180/PDF189 explains editable widget state and focus. The data-loss claim comes from mounted current-code evidence, not from a book rule. Relevant extracted pages are retained in the study evidence folder.

### Existing ownership and decisions

Screened all 5618 task metadata entries, body/symbol searches and 24 open PRs/3517 complete paths from root receipts. Full close matches read: TASK34751 Done owns list writer, teardown safety and literal text; TASK34755 To Do owns settled create status and writes to a replacement panel; TASK34752 To Do owns quiz question-list writes; TASK36099 Done in PR3205 owns quiz/question drafts only. TASK20979 To Do concerns agent output and making real rows visible; TASK34000.6 Done owns real-service keyword parity and geometry. No existing task covers newer authoring fields in the still-mounted Flashcards form, and no open PR touches the five relevant handler/window/service files. Keep status, list writer and cross-panel ownership independently scoped.

Related tasks: [TASK-34751](task-34751%20-%20Study-flashcards-one-writer-for-the-card-list-and-card-deck-text-shown-literally.md); [TASK-34755](task-34755%20-%20Study-the-Deck-created.-status-is-always-replaced-by-No-cards-due-for-review..md); [TASK-34752](task-34752%20-%20Study-quizzes-one-writer-for-the-question-list.md); [TASK-36099](task-36099%20-%20Keep-newer-Study-quiz-drafts-when-earlier-creates-complete.md); [TASK-20979](task-20979%20-%20Flashcards-viewing-SRS-surface-decide-and-build-what-makes-real-rows-visible.md); [TASK-34000.6](task-34000.6%20-%20Study-hand-off-selecting-or-creating-a-deck-no-longer-crashes-Study-controls-render-copy.md).

ADR required: no new ADR for this bounded repair.
ADR paths: [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md); [011-chatbook-workbench-ui-system](../decisions/011-chatbook-workbench-ui-system.md).
Reason: ADR required: no new ADR. This bounded repair retains screen-local authoring state across an existing create result under accepted ADR033/011. It does not change service interfaces, persistence, runtime authority or cross-panel state ownership.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-22-textual-book-after-pr3205.md). The failed assertions establish an unfixed finding, not a passing implementation. Future AC variants remain implementation requirements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A successful pending deck create persists only its submitted deck name and preserves any exact newer raw name still in the current authoring field, including a whitespace-only edit.
- [ ] #2 A successful pending flashcard create persists only its submitted front, back and parsed tags; independent or combined newer raw edits to any of those fields remain in the form. Completion preserves current editing context, including caret/selection and focus, without stealing focus deliberately moved elsewhere.
- [ ] #3 Unchanged fields clear after one successful creation; a definite failed create preserves current drafts and remains retryable under the existing service contract. Uncertain-write retry/idempotency policy is outside this task.
- [ ] #4 Bounded mounted regressions use the existing create events/workers, controlled service awaits and real StudyScopeService/LocalStudyService with SQLite; live drafts are asserted separately from stored records and unchanged-success/failure-retry controls pass.
<!-- AC:END -->
