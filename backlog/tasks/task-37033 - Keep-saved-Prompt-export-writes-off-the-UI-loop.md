---
id: TASK-37033
title: Keep saved-Prompt export writes off the UI loop
status: To Do
created_date: 2026-10-27 18:01
labels:
- textual-patterns
- library
- responsiveness
priority: medium
references:
- tldw_chatbook/UI/Library_Modules/library_file_export.py
- tldw_chatbook/UI/Library_Modules/library_prompts_controller.py
- https://github.com/rmusser01/tldw_chatbook/blob/3c279941e2af523ab75d2772ef543521870868c4/backlog/tasks/task-34000.97%20-%20Exporting-a-note-into-a-folder-the-app-cannot-search-raises-out-of-a-callback-instead-of-Export-failed.md
- Docs/superpowers/qa/2026-10-27-textual-book-after-pr3251.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Library Prompt export destination writes from freezing key and event dispatch. Correct Markdown output should not require the UI to wait synchronously for destination I/O.

A real saved-Prompt service/SQLite journey through More actions and FileSave reaches the atomic writer on the main thread. During a bounded held writer, an actual App key dispatch waits until release; ordinary export and Markdown round-trip pass and the source row is unchanged. This is a controlled responsiveness result, not a measured storage-latency distribution. Notes shares source structure but was not dynamically qualified.

**Scope and related owners:** [TASK-32629](<task-32629 - Review-Library-Prompt-Copy-and-Markdown-export-journeys.md>) owns export reachability/fidelity/focus/retry, and [TASK-34000.3](<task-34000.3 - Library-export-note-and-prompt-exports-ask-before-replacing-an-existing-file.md>) owns explicit Replace, atomicity and no-clobber behavior; retain both. [TASK-2834](<task-2834 - Guard-against-blocking-IO-reachable-from-a-Textual-message-handler.md>) owns a call-graph guard and named launcher fixes; its notes deliberately exclude text-file writes and its analysis treats call_after_refresh as a handoff, so it does not repair this callback. [TASK-33280](<task-33280 - PERF-21-Library-screen---targeted-updates-ingest-O-N-2-Folder-Files-poll.md>) enumerates other Library costs. [TASK-34000.97](https://github.com/rmusser01/tldw_chatbook/blob/3c279941e2af523ab75d2772ef543521870868c4/backlog/tasks/task-34000.97%20-%20Exporting-a-note-into-a-folder-the-app-cannot-search-raises-out-of-a-callback-instead-of-Export-failed.md) is an unmerged Notes destination-permission callback error owner in [PR #3259](https://github.com/rmusser01/tldw_chatbook/pull/3259), distinct from this Prompt write scheduling. PRs #3096/#3166 also modify adjacent Notes source; their inspected patches do not move this Prompt writer.

**ADR assessment:** No new ADR is expected for a bounded repair preserving existing ownership and service contracts. Existing [ADR-033](<../decisions/033-application-session-state-ownership.md>), [ADR-036](<../decisions/036-application-service-composition-lifecycle.md>), [ADR-031](<../decisions/031-tui-keybinding-and-footer-hint-conventions.md>) apply; reassess if implementation changes those boundaries.

Evidence, source pins, book pages, controls and limitations: [post-merge Textual audit](../../Docs/superpowers/qa/2026-10-27-textual-book-after-pr3251.md). Audited merged dev: `416feb50ee667fdc95cb269bf50f8b1d5b3ff8ff`. The supplied book is reference data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Unrelated App key and event dispatch continues while a saved-Prompt export destination write is held, without waiting for the file operation to finish.
- [ ] #2 After release, one correct captured Prompt Markdown export is produced and the source Prompt database row remains unchanged.
- [ ] #3 Existing-file Replace confirmation, no-clobber protection, cancellation, write-failure feedback and operation-scoped result/focus behavior retain their established outcomes.
- [ ] #4 Mounted private-profile tests use the actual Prompt export picker and real local service/SQLite with ordinary and held-writer controls, verifying key dispatch, output fidelity, unchanged source data and drained work.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
