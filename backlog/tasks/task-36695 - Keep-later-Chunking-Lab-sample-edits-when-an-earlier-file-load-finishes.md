---
id: TASK-36695
title: Keep later Chunking Lab sample edits when an earlier file load finishes
status: In Progress
created_date: 2026-10-24 17:33
labels:
- textual-patterns
- reliability
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-24-textual-book-after-pr3228.md
assignee:
- '@codex'
updated_date: 2026-10-24 19:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A pending sample file read publishes its captured text after later editing has already changed the live sample and canonical session. Preserve the newest accepted sample intent when older asynchronous reads finish.

### Evidence and scope

Verified after PR #3228 merged, against dev `96cae7eec00473bbfb5687ccfd0685eba2785add`. Real file_operation/sample-file reads a private UTF-8 file through its actual to_thread boundary. While delivery is held, a real keyboard x changes seed to xseed in both editor and canonical session. Releasing the read replaces both with selected file content. Untouched-file control passes. The file-dialog gesture and natural race frequency were not measured. Fresh merged-dev ordinary control passes. The later-edit child records the intended retention assertion failure plus a separate native-worker teardown error; it is not claimed as a clean whole-process regression result. Observational/worker-drain retries are retained as qualified or excluded evidence in the audit.

Book basis: PDF145–146 and158 / printed136–137 and149 explains independent queues and awaited handlers. Preserving newer accepted text at an older completion is an application-specific applicability requirement.

Affected owners: [chunking_lab_screen.py](../../tldw_chatbook/UI/Screens/chunking_lab_screen.py), [sample_region.py](../../tldw_chatbook/UI/Chunking_Lab_Modules/sample_region.py).

### Existing ownership and decisions

TASK-31645 is Done and implements ordinary file/clipboard imports and recovery, without a pending-read/newer-edit outcome. TASK-31640 owns recovery and TASK-31642 run lifecycle. TASK-31428 is an older planning task, not a second implementation owner. TASK-36363 protects audiobook authorship under a different owner.

Related tasks: [TASK-31645](../tasks/task-31645%20-%20Chunking-Lab-Library-screen-and-recoverable-authoring-flow.md), [TASK-31640](../tasks/task-31640%20-%20Chunking-Lab-durable-profile-local-session-checkpoints.md), [TASK-31642](../tasks/task-31642%20-%20Chunking-Lab-bounded-local-A-B-execution-lifecycle.md), [TASK-31428](../tasks/task-31428%20-%20Chunking-Lab-Library-screen-and-recoverable-authoring-flow.md), [TASK-36363](../tasks/task-36363%20-%20Keep-newer-audiobook-source-edits-when-a-pending-file-import-completes.md).

ADR required: no.
ADR paths: [118-chunking-lab-local-execution-and-recovery](../decisions/118-chunking-lab-local-execution-and-recovery.md), [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md).
Reason: Routine publication applicability repair within the existing Chunking Lab coordinator, sample ownership, validation and recovery contracts.

[Post-merge book audit and evidence](../../Docs/superpowers/qa/2026-10-24-textual-book-after-pr3228.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An older sample file or explicit-excerpt completion does not replace later accepted sample edits, including clearing the field; live text and canonical sample remain coherent.
- [x] #2 An untouched, still-applicable load publishes the selected text and truthful source provenance normally.
- [x] #3 Replacement sample intent and read failure preserve the current authored state, recovery ownership and existing file-size/UTF-8 validation.
- [x] #4 Targeted mounted controls witness real accepted keyboard input and canonical state before releasing actual asynchronous file-read publication, with ordinary-load and later-edit cases.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR paths: [ADR-118](../decisions/118-chunking-lab-local-execution-and-recovery.md) and [ADR-033](../decisions/033-application-session-state-ownership.md). Reason: applicability repair in the existing screen/coordinator/sample owner; unchanged validation, recovery and storage contracts.
1. Add bounded mounted controls for real asynchronous sample-file and explicit-excerpt reads, witnessing actual keyboard edits in both the editor and canonical session before releasing the older completion; cover clear, replacement intent, untouched success and read failure.
2. Retain attributable product RED with independently clean teardown. Reuse existing private-profile protections; use a bounded host or settle exact owners rather than hiding native-worker custody or broadly draining canceled workers.
3. Fence older read publication against current sample intent, including queued parent Changed delivery and edit-then-clear; preserve ordinary provenance, canonical recovery and existing size/UTF-8 validation.
4. Run new controls plus directly relevant existing file/excerpt/recovery tests and focused Ruff/format/whitespace checks, inspecting all private-child JUnit failures/errors/skips.
5. Obtain task-scoped Superpowers requesting-code-review, register every new regression in the required UI census, update architecture evidence and implementation notes. Overall hosted PR qualification follows all six tasks.
Master plan: Docs/superpowers/plans/2026-10-24-textual-lifecycle-and-import-followups.md (Task3).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Sample file and explicit-excerpt reads now carry screen/session/sample/editor applicability through the existing canonical edit queue. Synchronous TextArea Changed accounting protects keys accepted before parent delivery, and guards before and after transition computation prevent obsolete text from entering canonical state or Undo. Untouched content/provenance and existing validation/recovery remain unchanged.

Verification: all20 new mounted regressions passed with20 clean private children; six strengthened validation/SQLite reopen cases and three unchanged legacy UI bodies also passed with clean children. Root verified actual XMLs and final hashes. All20 exact selectors are registered in the required UI gate (floor810); eight split-census checks and census validation pass. Superpowers task review approved with no Critical/Important findings. Final hosted qualification remains pending. The original legacy batch is not green (profile-fixture setup drift and two Windows-overlong IDs); diagnostic67 parser/recovery checks passed, with eight original UI setup errors. Existing E721 exact-type check and dependency warnings remain attributed, without guard changes or broad suite claims.

ADR required: no; existing ADR-118 and ADR-033 govern unchanged canonical ownership, file validation and recovery. Updated Docs/Architecture/rag.md, UI census/ratchet and the implementation plan. Source/tests: UI/Screens/chunking_lab_screen.py, UI/Chunking_Lab_Modules/sample_region.py and Tests/UI/test_chunking_lab_sample_import_drafts.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
