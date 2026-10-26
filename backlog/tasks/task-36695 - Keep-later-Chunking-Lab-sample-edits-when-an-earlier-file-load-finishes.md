---
id: TASK-36695
title: Keep later Chunking Lab sample edits when an earlier file load finishes
status: Done
created_date: 2026-10-24 17:33
labels:
- textual-patterns
- reliability
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-24-textual-book-after-pr3228.md
assignee:
- '@codex'
updated_date: 2026-10-26 20:56
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
- [x] #5 After an admitted sample edit or Undo completes, a competing same-context status refresh cannot leave the visible sample behind canonical state; newer accepted input remains intact.
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
CI follow-up for PR #3251: ADR required: no. ADR paths: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md and backlog/decisions/033-application-session-state-ownership.md. Reason: repair final authoring publication inside the existing edit/render owner. 1. Retain controlled real-source RED for pending-worker and worker-first Undo render orderings with independently inspected private-child reports. 2. Add deterministic mounted regressions and a held-final/newer-keyboard control, keeping the original Undo assertion. 3. Let only the owned edit-complete publication survive a same-context background render generation; preserve session, input/revision, queued-edit, retirement and status controls. 4. Run targeted new/adjacent cases, lint/format and actual source guards, document the repair, obtain focused Superpowers review and verify hosted CI.
Focused CI-repair review P1: preserve newer configuration-field input while retaining final sample convergence. The initial generation exception allowed an older EditorRegion.present to replace a real Name key before parent Changed delivery; baseline comparison retained yz but the repaired child settled z. Restrict the exception to the sample echo and retain the original configuration publication generation guard, or an equally narrow verified alternative. Add a permanent held-parent real-key regression with the second key and canonical outcome, verify existing sample/newer-input/status controls, and obtain one scoped Superpowers rereview before push. Existing ADR-118/033; no new ADR.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Sample file and explicit-excerpt reads now carry screen/session/sample/editor applicability through the existing canonical edit queue. Synchronous TextArea Changed accounting protects keys accepted before parent delivery, and guards before and after transition computation prevent obsolete text from entering canonical state or Undo. Untouched content/provenance and existing validation/recovery remain unchanged.

Verification: all20 new mounted regressions passed with20 clean private children; six strengthened validation/SQLite reopen cases and three unchanged legacy UI bodies also passed with clean children. Root verified actual XMLs and final hashes. All20 exact selectors are registered in the required UI gate (floor810); eight split-census checks and census validation pass. Superpowers task review approved with no Critical/Important findings. Final hosted qualification remains pending. The original legacy batch is not green (profile-fixture setup drift and two Windows-overlong IDs); diagnostic67 parser/recovery checks passed, with eight original UI setup errors. Existing E721 exact-type check and dependency warnings remain attributed, without guard changes or broad suite claims.

ADR required: no; existing ADR-118 and ADR-033 govern unchanged canonical ownership, file validation and recovery. Updated Docs/Architecture/rag.md, UI census/ratchet and the implementation plan. Source/tests: UI/Screens/chunking_lab_screen.py, UI/Chunking_Lab_Modules/sample_region.py and Tests/UI/test_chunking_lab_sample_import_drafts.py.
Final whole-branch review follow-up: two additional file/excerpt controls hold the actual read while a valid Maximum size recipe edit is accepted canonically. The selected sample still loads with exact provenance and the complete recipe draft remains intact. No Chunking source change was needed. Together with two native Image Gen pruning regressions and two serial controls, the focused final-fix run has 6 parent passes and 6 clean private children, zero failures/errors/skips (72.279s). This task now contributes 22 exact UI cases to the overall floor843; original20-case evidence remains separately recorded. One scoped final re-review and hosted qualification remain pending.
Hosted qualification complete for PR #3243, code head087a75bab335485b2058f9a010c6f872c57ebc9b: https://github.com/rmusser01/tldw_chatbook/actions/runs/40132324399 (all25jobs success) and https://github.com/rmusser01/tldw_chatbook/actions/runs/40132324401 (Perf Guard success). Actual completed job logs prove all81 new UI selectors and20 parser cases executed, with17 successful UI shards and maximum986s, below1020s planning target/1200s cap. Pinned Canvas and all actual repository source guards pass on the hosted runner. No allowance, timeout, shard count or original census entry changed.

Superpowers requesting-code-review approved each task, the whole branch, and the single final fix re-review; M1/M2 are addressed with no remaining Critical/Important/Minor finding. All remote comment/review/thread pages were checked: no actionable finding or unresolved thread. All four acceptance criteria, implementation plan/notes, existing ADR links and relevant architecture/QA/incident documentation are complete. Native legacy/profile/Windows/static/size qualifications remain disclosed in Docs/superpowers/qa/2026-10-24-textual-followups-36693-36698.md; no full local suite or unrelated debt closure is claimed. This CLI closeout and verification-doc commit contains metadata only; its final PR head will receive fresh hosted CI before delivery.
PR #3251 CI follow-up: the edit consumer now retains its final authoring echo across a competing same-session status render. Canonical-session identity, mounted targets, input generation, synchronous editor revision and queued-edit checks remain unchanged; background guarded/save-status rendering stays active. Added five deterministic private-profile cases for both Undo render orders, held/delivered newer keyboard input, and live guarded/autosave status. Two intended Undo RED children failed the visible-editor assertion with no teardown errors; three initial control-fixture failures are excluded and documented. The repaired eight-case batch passed eight parent and eight actual child reports, including the original queued-transition failure and two adjacent controls. After hardening all pending status preparations, the final five new cases passed five parent/five actual child reports on frozen source, zero failures/errors/skips. The original three regressions are unchanged. Scoped lint/changed-range formatting pass beyond existing screen E721 at line345. Root independently inspected actual XML and source hashes; 183 integration contracts and exact worker/diagnostic guards pass. Updated UI/Screens/chunking_lab_screen.py, new Tests/UI/test_chunking_lab_edit_publication.py, Docs/Architecture/rag.md; existing ADR-118/033 apply, no new ADR. Focused Superpowers review and fresh hosted CI remain pending; status remains In Progress.
Focused Superpowers review of first CI repair d5a8 found one Important/P1 introduction: final sample publication also called older EditorRegion.present and erased a newer held-parent Name key. Root verified actual baseline/repaired children and final yz versus z canonical outcome, zero errors/skips. New AC5 is unchecked until corrected. The single feedback fix will preserve the prior configuration generation guard while exempting only the final sample echo, add the permanent two-key/canonical regression and complete one scoped rereview; no push as complete before that. Other pre-existing current-generation configuration-input windows remain unclaimed.
The focused P1 correction is implemented and independently rereviewed at 743ea2786e: only final sample echo may survive an older render generation; configuration presentation retains its exact prior generation guard. Added permanent real Name y/z held-parent regression, which failed z!=yz before correction and now preserves eventual visible/canonical yz. Final seven parent cases plus seven actual private children pass with zero failures/errors/skips on frozen source, including both Undo orders, held/delivered sample keys, live guarded/autosave status and the unchanged original queued-transition case. Root independently checked actual XML and all source hashes. Final exact worker/census guards pass; production call expressions, diagnostic inventory and sinks are unchanged. One scoped Superpowers rereview marks P1 FIXED/ADDRESSED with zero new Critical/Important/Minor findings. AC5 is checked again; fresh hosted current-head acceptance remains pending, so status remains In Progress. Existing ADR-118/033 apply; no broader current-generation configuration-input guarantee is claimed.
Final combined implementation accepted on `eb500d22dd3ab37faf18efb0a5e718c8f103b642` after rebase onto dev38623632e9: [Derived Artifacts](https://github.com/rmusser01/tldw_chatbook/actions/runs/40372196627) and [Perf Guard](https://github.com/rmusser01/tldw_chatbook/actions/runs/40372196618) are green. Successful hosted job logs establish all 46 new exact cases and five existing critical Library/Chunking cases once on their assigned shards (51 call-duration rows); all 17 UI jobs passed, maximum 1080s below the unchanged 1200s cap. All 906 current-dev census slots remain in order, plus 46 exact controls (floor952). Superpowers scoped reviews/rereviews have no open findings. [QA and evidence limits](https://github.com/rmusser01/tldw_chatbook/blob/codex/textual-book-followups-after-3248/Docs/superpowers/qa/2026-10-25-textual-followup-fixes.md) retain failed prior attempts, actual RED/GREEN controls, baseline caveats and native/live/full-suite limits. [PR #3251](https://github.com/rmusser01/tldw_chatbook/pull/3251) targets dev; final metadata-head CI is checked separately on the PR.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->