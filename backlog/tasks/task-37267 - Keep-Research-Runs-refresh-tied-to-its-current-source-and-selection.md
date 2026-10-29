---
id: TASK-37267
title: Keep Research Runs refresh tied to its current source and selection
status: Done
created_date: 2026-10-28 22:10
labels:
- textual-patterns
- research
- state-ownership
priority: medium
references:
- tldw_chatbook/UI/Research_Window.py
- tldw_chatbook/UI/Research_Modules/research_controller.py
assignee:
- '@codex'
updated_date: 2026-10-29 03:58
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the Research Runs action target and detail aligned with the current selected row and source when an earlier background refresh returns. Clearing selection on source change must also clear the old run detail.

On merged dev `7884e11a496a677c8180cc5d67234a57b7d659b9`, the unmodified native timer starts a real Local service/SQLite read of A. While its completed return is held, a real Enter selects B and paints B. Release restores selected A/detail A while the ListView still highlights B. A real source Select change to unavailable Server clears selection before release, then the old Local return restores A under Server. An ordinary source switch also leaves the previous Local detail visible with no selection. No erroneous run mutation or remote fallback was exercised; later actions deriving their ID from selected_run are a code-grounded risk.

**Existing owners:** [TASK-16486](<task-16486 - Auto-refresh-the-selected-research-run-in-the-window.md>) delivered the interval and normal refresh/payload guards; [TASK-21127](<task-21127 - Research runs - per-op leaked connects and loop-side periodic reads-writes while a run is active.md>) delivered service offload and connection lifetime. [TASK-19300](<task-19300 - Make-research-runs-durable-resumable-jobs.md>) owns execution durability, not view publication. [TASK-21507](<task-21507 - Add-Research-shell-authority-and-responsive-workspace-foundation.md>) and [TASK-21511](<task-21511 - Add-Research-Copy-sharing-and-Deep-Research-return.md>) concern the distinct Research Workspace/return flow; [TASK-2306](<task-2306 - Run selection populates the run detail region.md>) is Watchlists RunsPane. None owns this Runs timer/source-switch correction. Preserve the separate `research` and `research_workspace` routes.

**ADR assessment:** New ADR expected: no; ADR path: N/A. [ADR-033](<../decisions/033-application-session-state-ownership.md>), [ADR-068](<../decisions/068-local-research-execution-engine.md>), [ADR-070](<../decisions/070-research-run-lease-and-durability.md>) and [ADR-078](<../decisions/078-research-workspace-authority-and-screen-boundaries.md>) retain screen/service/execution boundaries.

Book asynchronous-publication patterns, actual native-timer observations, real local service/SQLite controls, source pins and limitations are in the [post-merge audit](../../Docs/superpowers/qa/2026-10-28-textual-book-after-pr3283.md). The supplied book is reference data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A real mounted selection of B remains selected, highlighted and rendered after an earlier refresh of A completes; current bundle/artifact/event payload state remains associated with the selected run.
- [x] #2 A Local refresh completing after a Server source switch cannot restore a Local selection/detail or change the current run-action target.
- [x] #3 Clearing selection for a source switch presents the empty-run detail state, including unavailable or empty catalog outcomes; selecting a later valid run populates its detail normally.
- [x] #4 Normal nonterminal Local refresh continues to update current progress and preserve payload state; terminal and Server selections retain their existing skip behavior.
- [x] #5 Targeted mounted controls hold the actual scope/service return, prove selection/source changes occurred before release, and verify final state with every hold released inside the App lifetime.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A (existing ADR-033/068/070/078).
Reason: bounded Research Runs view-publication fix preserving independent Workspace, service executor, storage and runtime ownership.
1. Read TASK-37267, native timer/source/selection consumers, audit controls and applicable architecture/testing docs.
2. Promote actual service-return/native interval tests; prove intended RED for newer run selection, source changes and stale detail, plus source away-and-back applicability.
3. Keep late timer results bound to the current window owner and explicitly clear source-departed detail; preserve controller cache contracts absent a concrete consumer need.
4. Verify GREEN and targeted Research progress/payload/terminal/source/controller controls plus scoped static checks; update Research architecture evidence.
5. Root commits exact owned changes, obtains independent spec/quality review, enrolls gates, and closes ACs/notes/status only after verified completion.
Full plan: Docs/superpowers/plans/2026-10-28-textual-37266-37268.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ResearchWindow now checks the captured source and selection revision after the native interval read. Newer run/source selection wins, including away-and-back changes; source clearing immediately renders empty detail even when catalogs fail or return empty. Accepted progress retains bundle/artifact/event payloads. Controller fetched-result cache remains under its existing contract because current actions use Window selection.

Files: UI/Research_Window.py, new real-service/native-timer publication tests and Docs/Architecture/research-workspace.md. Three intended assertion RED children; seven final parent/actual child passes;13 legacy behavior controls pass. Static batch46pass/8fail is not green: immutable comparisons preserve existing120 timer sites/75 roots and worker findings, including existing platform/inventory gaps. Scoped lint/format pass. Payload data is arranged from real keyword scope reads; separate artifact-action signature defect is tracked separately in the implementation QA, with existing task/ADR ownership linked.

ADR required:no; existing ADR-033/068/070/078 retained, with shared CI amendment ADR-103. Independent task review approved with no Critical/Important. Warnings and inherited static limitations remain disclosed.

Shared enrollment conserves all 1140 inherited selectors and adds43 new native cases plus the canonical-class guard (1184 total). ADR-103's existing measured capacity policy is amended to32 serial shards with max-parallel17 and unchanged timeouts/aggregation. QA and portable raw receipts/reviews: Docs/superpowers/qa/2026-10-28-textual-followups-implementation.md. Final Superpowers whole-branch review is approved. Hosted application, Perf Guard and required checks pass on 7c2b8f5aab066035ed9d06f9719a93d184f8e5c3; all44 added selections and the exact existing Notes Escape/Replace case have execution receipts. PR3290 is open against dev. Final delivery rechecks the metadata-only closure head; earlier local failures/qualifications remain historical, not relabeled passes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and independently reviewed; targeted native/contract/static evidence and final hosted qualification are linked in the implementation QA and PR3290. Existing ADR ownership retained; ADR103 extends only existing CI capacity policy.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
