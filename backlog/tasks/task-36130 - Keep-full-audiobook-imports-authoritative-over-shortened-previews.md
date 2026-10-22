---
id: TASK-36130
title: Keep full audiobook imports authoritative over shortened previews
status: In Progress
created_date: 2026-10-22 15:29
labels:
- speech
- textual-patterns
- source-integrity
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-22-textual-book-after-pr3205.md
assignee:
- '@codex'
updated_date: 2026-10-22 17:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve the complete imported audiobook source through preview refreshes, chapter detection and the Generate action so successful import cannot silently omit the end of a document.

### Evidence and scope

Medium priority; audit order 1 of 3. Audited source blobs on merged dev `7700c63a688355a2df595941b8b2b4dc8a819074` match diagnostic candidate `e41fc622e2702b3e87a1f21d83f1f328ed48b821` exactly. A mounted native file-import witness drove the real Import button, ready file picker and original async file read, then actual TextArea.Changed, real ChapterDetector and Generate button. A 1600-character file initially reached detection intact but its shortened preview then replaced content_text with 1003 characters (first 1000 plus artificial dots). The ending chapter disappeared, and the Generate event submitted 1003 while the notice still reported 1600 imported. The 900- and 1000-character controls retained exact source and both chapters. Final receipt: two passes and one intended retention failure; original file bytes were unchanged. No synthesis, generated audio, provider billing or persisted source loss was tested. Notes/Conversation imports share the shortening/Changed path by source inspection; their journeys were not exercised.

Source: [tldw_chatbook/UI/STTS_Window.py:579](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/UI/STTS_Window.py#L579); [tldw_chatbook/UI/STTS_Window.py:872](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/UI/STTS_Window.py#L872); [tldw_chatbook/UI/STTS_Window.py:1194](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/UI/STTS_Window.py#L1194).

Book basis: Chapter 7, Preventing Messages, printed 144–145 / PDF 153–154: programmatic widget updates can emit ordinary change messages. Keep presentation refreshes from becoming unintended source edits. The application-specific source guarantee comes from the real import/Generate trace.

### Existing ownership and decisions

TASK-36100 owns stale chapter publication after deliberate source changes and explicitly excludes long-preview truncation. TASK-34394 moved file reads off-loop; TASK-19576/16471 own import reachability and service routing. TASK-15478 and TASK-15773/16849/16850 own dispatch/parser/editor correctness. TASK-32808.3 concerns truncation display conventions, and TASK-32893 enumerates other text/data losses. None owns the full imported working source being replaced by a shortened preview. Preserve those completed contracts.

Related tasks: [TASK-36100](task-36100%20-%20Reject-audiobook-chapter-results-after-their-source-text-changes.md); [TASK-34394](task-34394%20-%20Debounce-and-batch-legacy-UI-hot-paths.md); [TASK-19576](task-19576%20-%20Two%20reachable%20user%20paths%20are%20broken%20by%20retired-surface%20drift%20-%20STTS%20AudioBook%20import%20crashes%20and%20Console%20file%20attach.md); [TASK-16471](task-16471%20-%20STTS-note-conversation-import-four-DB-helper-imports-dont-exist-both-dialogs-unreachable.md); [TASK-15478](task-15478%20-%20STTS-audiobook-paste-box-queries-a-switch-that-is-never-composed.md); [TASK-15773](task-15773%20-%20ChapterEditorWidget-Select-mount-race-under-high-volume-DataTable-population.md); [TASK-16849](task-16849%20-%20Chapter-editor-in-place-edits-never-refresh-the-now-truthful-chapter-table.md); [TASK-16850](task-16850%20-%20ChapterDetector-emits-an-empty-placeholder-chapter-per-titled-heading-doubling-the-rows.md); [TASK-32808.3](task-32808.3%20-%20Adopt-or-delete-the-truncation-helper-and-reconcile-the-52-re-rolls.md); [TASK-32893](task-32893%20-%20Work-stream-data-loss-and-silent-truncation.md).

ADR required: no new ADR for this bounded repair.
ADR paths: [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md); [011-chatbook-workbench-ui-system](../decisions/011-chatbook-workbench-ui-system.md).
Reason: Repair content ownership within the existing audiobook widget. No new storage, service/provider contract or global state owner is needed; recheck ADR requirements if implementation introduces one.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-22-textual-book-after-pr3205.md). The failed assertions establish an unfixed finding, not a passing implementation. Future AC variants remain implementation requirements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Successful audiobook file import preserves all imported text in the working source and actual Generate request, including text beyond character 1000; preview ellipsis is never silently substituted into that source.
- [ ] #2 Chapter detection and its editor/table reflect the complete imported source, including a chapter beyond the preview cutoff; intentional later edits and clears retain TASK-36100 freshness guarantees.
- [ ] #3 Equivalent Notes and Conversation import previews preserve the same full-source guarantee. Intentional source editing has a consistent, explicit relationship to the text submitted by Generate, without silently discarding unseen imported content.
- [ ] #4 Bounded mounted regressions drive real import/change/Generate events for long, exactly-1000 and short sources, demonstrate the defect before the fix and preserve ordinary imports and deliberate edits without provider or synthesis effects.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md
Reason: Keep full imported content in the existing editable TextArea for file, Notes and Conversation imports. Preserve source freshness and existing import/service paths; do not let a shortened programmatic display replace source.
1. Read the task, shared source-change handler and all three import callbacks. Extend the retained real import/Generate witness into bounded maintained regressions: long file, exactly 1000 and short control; real Notes/Conversation service routes; deliberate edit/clear before detection completes. Record meaningful RED assertions, then keep the complete source in the existing editor. Verify editor/model, chapter tail and emitted Generate content separately. No synthesis or external provider calls.
2. Verify the new/adjacent mounted tests, scoped static checks and existing size ceilings; retain exact receipts and source hashes.
3. Independently review against all four acceptance criteria, update architecture/QA notes, then root integrates census/pins and completes task metadata.
<!-- SECTION:PLAN:END -->
