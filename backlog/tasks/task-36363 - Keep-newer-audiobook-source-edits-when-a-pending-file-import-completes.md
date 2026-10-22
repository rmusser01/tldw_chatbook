---
id: TASK-36363
title: Keep newer audiobook source edits when a pending file import completes
status: Done
created_date: 2026-10-22 20:43
labels:
- speech
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
updated_date: 2026-10-22 23:52
modified_files:
- tldw_chatbook/UI/STTS_Window.py
- Tests/UI/test_audiobook_import_source_integrity.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A delayed audiobook file import must not overwrite source text the user has already edited or cleared while its result is pending.

### Evidence and scope

Medium priority, audit order 4 of 4. At merged dev 814becf3cabc42a622c4ed6f7eeca17d4c6266e5, actual Import opens a ready FileOpen picker and performs the original UTF-8 asyncio.to_thread read. Holding delivery of that real result leaves the existing source TextArea enabled. An actual single x key visibly changes it to xChapter1/INITIAL_DRAFT while its Changed message is queued behind the widget callback; the model still has the older source. Completion blindly installs FILE_SOURCE into both model and editor, and actual Generate submits FILE_SOURCE. The original file is unchanged. The matched no-edit import control passes; the edit-retention assertion fails, zero errors,17.073s. Earlier fixture/plugin and Pilot-completion diagnostic attempts are excluded. Natural disk latency, other import routes, synthesis and provider effects were not measured.

Source: UI/STTS_Window.py:530-594 (picker callback/read/publication),860-889 (queued source Changed/debounce); unchanged Generate path (under tldw_chatbook/).

Book basis: Chapter7 printed149/PDF158 explicitly explains that an awaited handler holds its own message queue until it returns. Chapter17 printed326/PDF335 supplies result-applicability checks; chapter7 printed144-145/PDF153-154 covers programmatic Changed echoes. Live source authority follows observed input loss, not a new book-prescribed persistence policy.

### Existing ownership and decisions

36130 owns full source versus truncated preview;36100 chapter publication after a source edit, not pre-chapter file source publication.34394 offloads the real read,19576/16471 restore import reachability,15478/15773/16849/16850 parser/editor/dispatch behavior,32800.4 the worker/DOM guard. All are distinct completed contracts to retain. No current open PR touches STTS_Window; recent all-ref path history contains only the merged source/chapter fixes.

Related tasks: [TASK-36130](task-36130%20-%20Keep-full-audiobook-imports-authoritative-over-shortened-previews.md); [TASK-36100](task-36100%20-%20Reject-audiobook-chapter-results-after-their-source-text-changes.md); [TASK-34394](task-34394%20-%20Debounce-and-batch-legacy-UI-hot-paths.md); [TASK-19576](task-19576%20-%20Two%20reachable%20user%20paths%20are%20broken%20by%20retired-surface%20drift%20-%20STTS%20AudioBook%20import%20crashes%20and%20Console%20file%20attach.md); [TASK-16471](task-16471%20-%20STTS-note-conversation-import-four-DB-helper-imports-dont-exist-both-dialogs-unreachable.md); [TASK-15478](task-15478%20-%20STTS-audiobook-paste-box-queries-a-switch-that-is-never-composed.md); [TASK-15773](task-15773%20-%20ChapterEditorWidget-Select-mount-race-under-high-volume-DataTable-population.md); [TASK-16849](task-16849%20-%20Chapter-editor-in-place-edits-never-refresh-the-now-truthful-chapter-table.md); [TASK-16850](task-16850%20-%20ChapterDetector-emits-an-empty-placeholder-chapter-per-titled-heading-doubling-the-rows.md); [TASK-32800.4](task-32800.4%20-%20Add-a-guard-for-the-Textual-worker-and-post-await-DOM-contract.md).

ADR required: no new ADR for the bounded correction.
ADR paths: [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md); [011-chatbook-workbench-ui-system](../decisions/011-chatbook-workbench-ui-system.md).
Reason: No new ADR is expected for stale source-publication prevention within the existing widget. Retain off-loop reads, source authority and chapter contracts; no provider, storage or global lifetime framework is introduced.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-22-textual-book-after-pr3207.md). These are findings and future regression requirements, not implemented fixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When file-read completion is delayed, actual text edits, replacements or clear accepted by the enabled current source editor remain authoritative after completion; queued Changed delivery must not let the earlier import replace them.
- [x] #2 The current TextArea, working source and next Generate request agree on the retained user source, and a discarded stale import does not announce that its text became current.
- [x] #3 With no later source edit, a complete file import still replaces the previous source normally and reaches chapter detection and Generate without artificial truncation; source file bytes remain unchanged.
- [x] #4 Bounded mounted regressions use the actual Import/picker/read and keyboard-change paths, hold delivery of real file content, verify ordinary/edited/cleared controls, and retain the existing chapter freshness and full-source tests without synthesis or network effects.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: existing ADR paths in this task. Reason: bounded correction within the current owner; no new storage, provider or application boundary.

1. Read the linked task, ADRs, current owner tests and retained audit evidence.
2. Add and run mounted behavior regressions RED, then implement the smallest owner-local correction: Compare current live editor text as well as the owning source state across the awaited real file read; an editor Changed message can still be queued behind the parent callback. Discard obsolete file-import completion without its success notification. Preserve full-source import and existing chapter-publication guards.
3. Verify targeted new/adjacent tests and static checks; obtain independent Superpowers task review.
4. Record evidence, update notes/ACs and relevant documentation, then mark Done via CLI.

Implementation plan: Docs/superpowers/plans/2026-10-22-textual-retry-and-pending-drafts.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Guard delayed file import publication with current owner attachment, editor identity, live editor text and working-source snapshots across the original off-loop UTF-8 read. Accepted edits/replacement/clear remain authoritative despite queued Changed; stale import does not announce success. Changed STTS_Window.py and extended test_audiobook_import_source_integrity.py. RED3intended retention failures/1ordinarypass; final16pass (4new+7full-source+5chapterfreshness), real Import/picker/read and actual Generate without synthesis/network. Clear explicitly uses public production delete-left bound-action dispatch after Pilot selection; edit/replace use actual Pilot printable input. Optional13legacy pre-owner setup errors are excluded, with exactbase/current representative qualification. No new owner lint/format debt; existing5E721/oneformat hunk retained. No new ADR: existing ADR-011 and ADR-033 session ownership apply. All4newnodes enrolled. See Docs/superpowers/qa/2026-10-22-textual-retry-and-pending-drafts.md for source-bound evidence, review and limitations. Independent Superpowers task review Approved.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
