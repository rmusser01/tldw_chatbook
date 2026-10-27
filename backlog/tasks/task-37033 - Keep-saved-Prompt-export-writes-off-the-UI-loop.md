---
id: TASK-37033
title: Keep saved-Prompt export writes off the UI loop
status: In Progress
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
assignee:
- '@codex'
updated_date: 2026-10-27 22:39
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
- [x] #1 Unrelated App key and event dispatch continues while a saved-Prompt export destination write is held, without waiting for the file operation to finish.
- [x] #2 After release, one correct captured Prompt Markdown export is produced and the source Prompt database row remains unchanged.
- [x] #3 Existing-file Replace confirmation, no-clobber protection, cancellation, write-failure feedback and operation-scoped result/focus behavior retain their established outcomes.
- [x] #4 Mounted private-profile tests use the actual Prompt export picker and real local service/SQLite with ordinary and held-writer controls, verifying key dispatch, output fidelity, unchanged source data and drained work.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Promote committed real SQLite/FileSave audit into ordinary and held App dispatch regressions. 2. Move captured Prompt destination writes off the UI loop through existing worker and confirmation ownership; retain Replace/no-clobber/scoped feedback. 3. Verify targeted journeys/helper/source/worker guards, document evidence and commit exact owned files. ADR required: no. ADR path: N/A. Reason: bounded scheduling repair preserving existing service and UI ownership. Existing ADRs: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/036-application-service-composition-lifecycle.md; backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md.
4. Review fix round1 (I1): reproduce a real no-clobber collision followed by a destination metadata recheck PermissionError in a native App worker, retain safe captured feedback and retry, then place that recheck inside the existing recoverable Prompt-only boundary. Verify focused error/render/direct Prompt controls and the two real mounted racing-file Cancel/Replace controls; qualify unchanged normal success-path AST/import/source and update report. ADR required: no. ADR path: N/A. Reason: preserve the existing scoped export error contract without changing shared helper or ownership boundaries; existing canonical ADR paths above still apply.
<!-- SECTION:PLAN:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Saved Prompt export now captures Markdown and destination before submitting a native screen worker; FileSave and Replace callbacks return immediately, filesystem-only writing runs off-loop, and scoped UI receipts retain the captured Prompt identity. Late no-clobber collisions reopen the existing Replace prompt. Render/write failures keep safe feedback and retry. Added real SQLite/FileSave controls and adapted existing tests to await native export completion without weakening assertions. Final minimal ordinary/held New/Replace controls: 4 parent passes and 4 actual private-child passes, zero errors/skips. Also verified race Cancel/Replace, 34 helper cases, 3 navigation scopes, 64 size/wiring controls and unchanged native preimport budget at 558/558 modules. Fourteen existing native Windows mode/symlink/CRLF/receipt expectations and two unchanged inventory/import guard failures reproduce on original source; all raw receipts and causal witness methodology are retained in .superpowers/sdd/2026-10-27-textual-37031-37034/task-3-report.md. Updated Docs/Architecture/library-shell.md. ADR required: no; existing backlog/decisions/033-application-session-state-ownership.md, 036-application-service-composition-lifecycle.md, 031-tui-keybinding-and-footer-hint-conventions.md, 150-design-token-system-and-design-language.md and 161-component-pattern-library.md preserve existing ownership. Task remains In Progress for coordinator independent review and current-head CI; no full suite or allowance increases.
Review fix round1 (I1): placed late-collision metadata recheck inside the existing Prompt-only recoverable export error boundary. Permission failures now report the captured sanitized exception category instead of escaping as a fatal native worker failure; worker defaults, named nonexclusive ownership, generic helper and Notes/Report remain unchanged. New native App regression arms a destination Path.lstat PermissionError only after a real atomic no-clobber refusal, verifies file preservation, safe feedback, App survival and explicit same-destination Replace retry with captured content. TDD: task3-r1-red01 has one intended WorkerFailed(PermissionError) failure and no errors/skips; task3-r1-green01 passes all10 focused parent cases and both actual private racing-file children with zero failures/errors/skips. Focused Ruff/format pass. AST qualification preserves ordinary writer await, existing error feedback/success receipt, captured/modal/worker ownership and every function/import outside Prompt export. ADR required: no; existing canonical ADRs in the plan remain applicable. M1 inherited dependency/test warnings remain deferred; task stays In Progress pending coordinator scoped rereview and final combined-source/CI gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->