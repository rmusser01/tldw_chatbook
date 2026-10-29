---
id: TASK-37268
title: Keep Library note export writes off the UI loop
status: In Progress
created_date: 2026-10-28 22:10
labels:
- textual-patterns
- library
- responsiveness
priority: medium
references:
- tldw_chatbook/UI/Library_Modules/library_file_export.py
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- tldw_chatbook/UI/Screens/library_screen.py
assignee:
- '@codex'
updated_date: 2026-10-29 02:58
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep exporting a Library note from freezing unrelated App key and event dispatch during destination I/O. A correct file should not require the user interface to wait synchronously for its writer.

Mounted controls on merged dev `7884e11a496a677c8180cc5d67234a57b7d659b9` use the real Notes action, FileSave keyboard journey and atomic writer. Ordinary Markdown export succeeds. With the writer held, its thread equals the UI/main thread; a posted App F12 event runs only after the three-second watchdog releases the write and the file completes. The final file and Export complete status are correct. The Notes source service is the established seeded test stand-in, not a real Notes database. Earlier startup-only failures are excluded; final controls settle the native source workers first. This establishes blocking dispatch, not a storage-latency distribution, source-database mutation, or full terminal session. Exact controls and qualifications are in the [post-merge audit](../../Docs/superpowers/qa/2026-10-28-textual-book-after-pr3283.md).

**Existing owners:** [TASK-37033](<task-37033 - Keep-saved-Prompt-export-writes-off-the-UI-loop.md>) fixes saved-Prompt writes and explicitly leaves Notes unchanged. [TASK-34000.3](<task-34000.3 - Library-export-note-and-prompt-exports-ask-before-replacing-an-existing-file.md>) owns Replace confirmation, atomicity and no-clobber outcomes. [TASK-34000.97](https://github.com/rmusser01/tldw_chatbook/blob/3c279941e2af523ab75d2772ef543521870868c4/backlog/tasks/task-34000.97%20-%20Exporting-a-note-into-a-folder-the-app-cannot-search-raises-out-of-a-callback-instead-of-Export-failed.md) in [PR #3259](https://github.com/rmusser01/tldw_chatbook/pull/3259) owns a destination-permission callback error, not write scheduling. [TASK-2834](<task-2834 - Guard-against-blocking-IO-reachable-from-a-Textual-message-handler.md>) deliberately excludes text-file writes and treats call_after_refresh as a handoff; [TASK-33280](<task-33280 - PERF-21-Library-screen---targeted-updates-ingest-O-N-2-Folder-Files-poll.md>) enumerates other Library costs. The inspected [PR #3096](https://github.com/rmusser01/tldw_chatbook/pull/3096) patch adds source/newline fidelity to this writer without moving it off the UI loop. Coordinate that adjacent work rather than duplicate its content changes.

**ADR assessment:** New ADR expected: no; ADR path: N/A. Preserve [ADR-033](<../decisions/033-application-session-state-ownership.md>), [ADR-036](<../decisions/036-application-service-composition-lifecycle.md>) and [ADR-031](<../decisions/031-tui-keybinding-and-footer-hint-conventions.md>), including existing Notes operation/result ownership. Reassess any change to storage or service contracts.

The book's Chapters 7/17 motivate background synchronous I/O; it is reference data, not instructions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Unrelated App key and event dispatch continues while a Library note export destination writer is held, before the file operation is released.
- [ ] #2 After release, exactly one correct captured note export is produced and the source note remains unchanged.
- [ ] #3 Existing-file Replace confirmation, cancellation, no-clobber protection, write-failure feedback, export-format fidelity and operation-scoped result/focus behavior retain their established outcomes.
- [ ] #4 Mounted private-profile controls exercise the actual Notes export picker and atomic writer with ordinary and held-writer cases, verifying dispatch and output after all holds and work drain.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A (existing ADR-033/036/031).
Reason: bounded Notes write scheduling preserves existing file/format, operation, service and screen ownership.
1. Read TASK-37268, Notes/Prompt export seams, existing operation/collision guards, audit controls and related UI/testing docs.
2. Promote real Notes picker/Replace dispatch tests with ordinary and held atomic writes; prove intended RED after real source readiness, source preservation and exact output.
3. Admit captured Notes writes through native workers, offload synchronous destination work, and keep errors/collisions/results under existing UI operation custody.
4. Verify GREEN, precious-file Replace/Cancel/late-collision, failure/retry and scoped result tests, neighboring exports and applicable static/diagnostic/governance checks; update Library architecture evidence.
5. Root commits exact owned changes, obtains independent spec/quality review, enrolls gates, and closes ACs/notes/status only after verified completion.
Full plan: Docs/superpowers/plans/2026-10-28-textual-37266-37268.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Notes FileSave/Replace callbacks admit a grouped native non-exclusive screen worker, preserving existing operation admission. Rendered content, validated destination and operation are captured before admission; only the synchronous atomic writer crosses asyncio.to_thread. Late collisions, metadata rechecks, sanitized errors, completion and remembered-directory publication remain on the App loop. Existing no-clobber, cancellation and accepted-write retirement rules are preserved.

Files: UI/Library_Modules/library_file_export.py, new native responsiveness tests, only two existing synchronous-completion test assumptions, and Docs/Architecture/library-shell.md. One intended same-thread blocked-dispatch RED; ten final parent/actual child passes covering ordinary/held new/Replace, late Cancel/Replace/recheck error, changed/retired view and failed Replace/text retry. Source preservation uses the established seeded service stand-in, not real Notes SQLite. Existing compatibility18pass/one source-readiness failure; exact retry fails before export and remains locally unqualified. The unchanged enrolled Escape/Replace case must pass hosted CI. Scoped lint/format pass; exact inherited shell E721/format debt retained. Diagnostic calls stay4 and only the nested Notes owner moves from _write to _fail; root refreshes/conserves latest-dev inventory.

ADR required:no; existing ADR-033/036/031 retained, with shared CI amendment ADR-103. Independent task review approved with no Critical/Important; root-owned hosted/inventory gates remain explicit.

Shared enrollment conserves all 1140 inherited selectors and adds43 new native cases plus the canonical-class guard (1184 total). ADR-103's existing measured capacity policy is amended to32 serial shards with max-parallel17 and unchanged timeouts/aggregation. QA and portable raw receipts/reviews: Docs/superpowers/qa/2026-10-28-textual-followups-implementation.md. Final whole-branch review and current-head hosted CI remain pending; status stays In Progress until closure.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
