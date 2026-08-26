---
id: TASK-203
title: 'Library Prompts: multi-select + bulk actions in the list'
status: Done
assignee:
  - '@codex'
created_date: '2026-07-12 22:21'
updated_date: '2026-08-13 13:49'
labels:
  - ux
  - library
  - prompts
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Second-pass UX review (2026-07-12): the prompts list has no multi-select, so
bulk delete and export are impossible; a growing library will want them
(mirrors the media multi-select backlog item task-159). Selection must support
curating a batch across searches and pages. Bulk tagging is explicitly outside
this task; Prompt collections already provide the current organization surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Prompt list offers a visible selection mode with checked rows, the total selected count, the count selected on the current page, Select page, Clear all, and Done controls
- [x] #2 Selection persists across Prompt searches, pages, sort orders, collection scopes, and an Export-canvas round trip, then clears explicitly on Done, successful delete, editor/create entry, source change, or Library exit
- [x] #3 Export selected uses the existing local Chatbook export flow and includes exactly the selected active Prompt/Recipe IDs; a missing selected item aborts the Prompt-bearing archive rather than producing a partial export
- [x] #4 Bulk delete validates every selected Prompt/Recipe against its captured version and soft-deletes the complete selection atomically; any missing, changed, invalid, or failed item leaves every item undeleted and preserves the selection
- [x] #5 Successful single and bulk deletes share one mutation path and leave one in-place Undo/Dismiss receipt; Undo restores the complete receipt atomically or restores nothing and keeps the receipt available
- [x] #6 Selection and selected export remain local-only; delete and restore are policy-gated exactly once per batch; blocking export, delete, and restore work remains off the UI thread; new or modified selection/delete/restore diagnostics never contain Prompt content, names, IDs, versions, selection payloads, exception messages, or tracebacks; and selected export retains ADR-057's sanitized Prompt collection/scope diagnostics
- [x] #7 Keyboard focus, literal labels, disabled explanations, loading/error recovery, and every selection/bulk action remain reachable and readable at both 64x24 and 120x40 with exactly the existing scroll ownership
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/060-atomic-local-prompt-batch-mutations.md
Reason: TASK-203 introduces an atomic multi-row conflict policy and typed database-to-service result contract.

1. Add strict immutable batch DTOs and the pure cross-search selection basket with RED-first constructor/state tests.
2. Refactor Prompt delete/restore into transaction-local helpers and add BEGIN IMMEDIATE all-or-nothing batch APIs while preserving legacy single signatures and results.
3. Add typed local/scope batch pass-through with local-only validation and exactly one existing RuntimePolicy decision per batch.
4. Render selection mode, disabled markers/reasons, checked rows, progress, and plural receipts in the existing Prompt canvas without a new scroll owner.
5. Wire persistent selection lifecycle and exact selected ExportScope string IDs through the existing Library export canvas.
6. Route editor single delete and selected bulk delete through one screen mutation path, implement atomic Undo, and veto all Library/app navigation while the receipt owner is settling.
7. Run integrated accessibility, compositor, privacy, diagnostic-inventory, mutation, static, typing, and CSS gates; change layout only for a proven RED.
8. Update the Prompt guide, run the frozen affected and full suites, complete self/YAGNI review, record evidence, check ACs, and mark TASK-203 Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented persistent local Prompt/Recipe selection, exact selected Chatbook export, and one atomic version-checked delete/Undo family under ADR-060. Added immutable batch contracts, BEGIN IMMEDIATE database mutations with pre-commit typed results, local/scope service pass-through, existing-canvas controls, shared mutation/navigation admission, real-SQLite and mounted Textual coverage, privacy-safe diagnostics, and Prompt guide updates. Focused evidence: 1042 Prompts_DB/Prompt_Management, 278 Prompt canvas, 9 Prompt-focused shell/navigation, and 23 RuntimePolicy tests passed; CSS parity, py_compile, and diff checks passed. The TASK-203-changed diagnostic owners matched the scanner exactly: Prompts_DB call count 130 and digest 348412ed7262ef48b026; library_screen call count 94 and digest 24318415ea7bcd2bf610. Approved inherited exceptions: the repository-wide checker remains red because of unrelated TASK-492/TASK-494 owner drift and inherited persistent-sink topology drift; TASK-203 changed neither sink topology nor scanner scope; five affected-matrix failures reproduced unchanged on base 706105a2f; Ruff format, F811, and five mypy findings also reproduced on that base. The full suite was stopped at 21% after 42:24 with 8600 passed, 55 failed, 3 errors, and 76 skipped; no TASK-203 test failed. Failures were in inherited architecture, audio, ChaChaNotes schema/migrations, chat/QwenCloud, and DB migration areas; two console-provider errors were sandbox socket-bind denials and one Chunking teardown error was blocked NLTK network access. Trade-offs: stale selection blocks the whole batch, mutations reserve the SQLite writer, and navigation is briefly vetoed while delete/Undo settles. No schema change, dependency, server batch fallback, generic bulk framework, or new scroll owner was added.
<!-- SECTION:NOTES:END -->
