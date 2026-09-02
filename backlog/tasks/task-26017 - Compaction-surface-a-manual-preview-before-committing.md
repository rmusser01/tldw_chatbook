---
id: TASK-26017
title: 'Compaction: surface a manual preview before committing'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:45'
updated_date: '2026-09-01 16:22'
labels:
  - console
  - context
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Manual compaction commits without showing what it will do. Verified on origin/dev: plan_manual_prefix and plan_manual_range (Chat/console_context_compaction.py:1016, :1047) build a ManualMemoryPlanResult and it is invoked from Chat/console_chat_controller.py:14051, but a grep for --preview or dry_run across Chat/ returns zero - the plan object exists and is simply never shown. Hermes offers /compress --preview. This is presentation over a value already computed before the commit point.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A manual compaction can be previewed: what will be summarized, what will be retained, and the estimated token change
- [x] #2 The preview does not perform the compaction and leaves no memory record or provenance entry behind
- [x] #3 The user can commit or discard directly from the preview without re-specifying the range
- [x] #4 Preview honors the same range semantics as the commit path, so what is previewed is what happens
- [x] #5 Previewing costs no model call, or if it must call, that cost is stated before it is incurred
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pure manual_summary_preview(plan) helper + ManualSummaryPreview (RED first)\n2. Extract shared _manual_summary_planning from _summarize_manual (AC#4: one planning path)\n3. preview_summarize controller API (stops before admission/ledger/model call)\n4. ConsoleSummarizePreviewModal (bool) + gate in _summarize_console_range via push_screen_wait\n5. Modal-dismissal contract registration + CSS bundle regen + guide
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preview is a pure projection of the already-computed ManualMemoryPlan: manual_summary_preview() + ManualSummaryPreview dataclass in console_context_compaction.py (turns summarized/retained, before/after tokens, output cap). The controller's _summarize_manual was split: shared _manual_summary_planning (guards, snapshots, resolution, plan — no admission, no ledger write, no model call) feeds both the new preview_summarize() and the unchanged commit path, so what is previewed is what happens (AC#4 by construction). The screen's _summarize_console_range gates on ConsoleSummarizePreviewModal (bool modal, Escape=cancel, registered in the modal-dismissal contract suite: count pin 14->15 + adoption set + launch edge via contract table); commit re-runs full planning+admission so a changed conversation is re-validated, and the user never re-specifies the range (captured message_id/from_here). Blocked planning surfaces the same visible copy the commit would. Amended 26016 in passing: summarize_manual's own auxiliary call was ALSO unbounded (the RED test literally hung 5 minutes) — same wait_for + TIMED_OUT + reason; controller copy 'The summarizer timed out. No memory was saved.' CSS bundle regenerated (BUNDLED_CSS auto-lift). Suites: compaction 129 passed; rewind_summarize + modal-dismissal at exact baseline failure sets (stash-bisected).
<!-- SECTION:NOTES:END -->
