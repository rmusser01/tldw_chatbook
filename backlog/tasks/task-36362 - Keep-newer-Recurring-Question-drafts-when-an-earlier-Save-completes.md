---
id: TASK-36362
title: Keep newer Recurring Question drafts when an earlier Save completes
status: Done
created_date: 2026-10-22 20:43
labels:
- scheduling
- textual-patterns
- reliability
priority: medium
references:
- backlog/decisions/018-local-server-hybrid-scheduled-tasks.md
- backlog/decisions/033-application-session-state-ownership.md
documentation:
- Docs/superpowers/qa/2026-10-22-textual-retry-and-pending-drafts.md
updated_date: 2026-10-22 23:51
assignee:
- '@codex'
modified_files:
- tldw_chatbook/UI/Screens/scheduling/forms/automation_definition_form.py
- Tests/UI/test_automation_definition_pending_save_drafts.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Recurring Question form that accepts newer edits while Save is pending must keep those edits recoverable when that earlier submission succeeds.

### Evidence and scope

Medium priority, audit order 3 of 4. Actual mounted AutomationDefinitionForm with real SchedulingService and file-backed ScheduledTasksDB held the physical create/update before the original write. Keyboard typing appended LATER to enabled Name while Save was disabled and the form dirty. Release stored only the submitted name, returned saved and detached the whole form, losing the newer draft. Both create and edit witnesses plus their unchanged controls were observed: four assertions documenting existing behavior pass in12.82s, one physical write each. These are defect observations, not passing repair tests. Server queued outcomes and other fields were not dynamically measured.

Source: UI/Screens/scheduling/forms/automation_definition_form.py:942-995, editable Changed handlers679/686; scheduling service physical write2576-2580/2745-2752 and workbench entry2873/2907 (under tldw_chatbook/).

Book basis: Chapter7 printed149/PDF158 and chapter17 printed326/PDF335 guide concurrent input/completion applicability. Applying that principle to editable draft retirement is an application-specific inference, established by the real write witness.

### Existing ownership and decisions

Completed35865 owns single Save admission and failed-save retry, not later inputs lost on success.33622.10/.15 own explicit close/quit guards;31712/31414 own modal defaults/config merging and31710/31711 wording/timezones. The current open-PR census has no affected form/service/workbench overlap. Preserve all existing one-write/admission contracts.

Related tasks: [TASK-35865](task-35865%20-%20Prevent-repeated-automation-Save-from-creating-duplicate-definitions.md); [TASK-33622.10](task-33622.10%20-%20Console-CtrlQ-is-ignored-while-any-modal-is-open.md); [TASK-33622.15](task-33622.15%20-%20CtrlQ-under-a-modal-in-flight-operations-and-the-video-save-picker-can-still-lose-work.md); [TASK-31712](task-31712%20-%20Schedules-form-detail-pane-polish.md); [TASK-31414](task-31414%20-%20Stop-backfilling-unset-definition-config-keys-on-unrelated-row-edits.md); [TASK-31710](task-31710%20-%20Schedules-copy-vocabulary-pass.md); [TASK-31711](task-31711%20-%20Schedules-timestamp-timezone-display-pass.md).

ADR required: no new ADR for the bounded correction.
ADR paths: [018-local-server-hybrid-scheduled-tasks](../decisions/018-local-server-hybrid-scheduled-tasks.md); [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md).
Reason: No new ADR is expected for retaining the existing form draft and actual saved identity. Existing ADR018 owns persistence/outbox/owner semantics;033 owns narrow ephemeral state. [ADR-099 schedule editor shape](../decisions/099-schedule-editor-shape.md) and [ADR-116 schedules inspector editing](../decisions/116-schedules-inspector-editing.md) remain Proposed context, not ratified by this task.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-22-textual-book-after-pr3207.md). These are findings and future regression requirements, not implemented fixes.
<!-- SECTION:DESCRIPTION:END -->
## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 For both local create and edit, newer raw Name, Question or other editable values accepted during pending Save remain visible and dirty after that earlier Save succeeds, including invalid or whitespace-only later text.
- [x] #2 If no newer edit was accepted, successful Save still closes the form and reports its established outcome.
- [x] #3 After retaining later edits from a successful create, the next explicit Save updates the same persisted definition; existing pending admission still permits one physical operation per Save.
- [x] #4 Definite failure/refusal retains the draft and remains retryable; existing local/server owner and queued outcome semantics and explicit close/quit guards stay intact.
- [x] #5 Bounded mounted keyboard/real-SQLite create/edit witnesses cover later edits, unchanged success, next Save and definite failure without weakening storage/config admission or fabricating saved results.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: existing ADR paths in this task. Reason: bounded correction within the current owner; no new storage, provider or application boundary.

1. Read the linked task, ADRs, current owner tests and retained audit evidence.
2. Add and run mounted behavior regressions RED, then implement the smallest owner-local correction: Capture the submitted raw form snapshot, acknowledge successful persisted identity, and dismiss only if the current raw form is unchanged. A retained create form adopts the saved definition ID so another explicit Save updates that same row. Preserve one-write admission and existing local/server/queued outcomes.
3. Verify targeted new/adjacent tests and static checks; obtain independent Superpowers task review.
4. Record evidence, update notes/ACs and relevant documentation, then mark Done via CLI.

Implementation plan: Docs/superpowers/plans/2026-10-22-textual-retry-and-pending-drafts.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Compare submitted versus current raw form values before retiring a successful draft. Retained creates adopt durable ID/outcome before off-loop metadata I/O, lock only owner identity and update the same row on subsequent Save. Failed creates restore owner selection; failed postcommit reads keep saved identity and newer input. Confirmed discard returns the actual last saved/queued result for workbench refresh. Changed automation_definition_form.py and added mounted real-SchedulingService/SQLite tests. All20newoutcomes covered (18initial passes plus2corrected dialog-readiness checks);30adjacent form/quit checks pass. IDs-only refinement preserves normalized module AST and parameter data;20collected whitespace-free nodes map to passing receipts. Ruff/format/diff clean; inherited dependency warnings remain visible. No new ADR: accepted ADR-018 and ADR-033 session ownership apply; proposed099/116 remain context. Independent Astra Superpowers task review Approved. See Docs/superpowers/qa/2026-10-22-textual-retry-and-pending-drafts.md for evidence and limits. Shared integration found post-await title/help lookups: QueryError guards plus actual chrome-removal RED2/GREEN4 cover metadata/admission/rollback; full worker contract passes with2782stable inputs and scoped rereview finds no new breakage. Reviewed routing-only diagnostic and updated Docs/security/production-diagnostic-inventory.json pin, with no new sink.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
