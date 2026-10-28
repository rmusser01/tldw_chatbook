---
id: TASK-37267
title: Keep Research Runs refresh tied to its current source and selection
status: To Do
created_date: 2026-10-28 22:10
labels:
- textual-patterns
- research
- state-ownership
priority: medium
references:
- tldw_chatbook/UI/Research_Window.py
- tldw_chatbook/UI/Research_Modules/research_controller.py
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
- [ ] #1 A real mounted selection of B remains selected, highlighted and rendered after an earlier refresh of A completes; current bundle/artifact/event payload state remains associated with the selected run.
- [ ] #2 A Local refresh completing after a Server source switch cannot restore a Local selection/detail or change the current run-action target.
- [ ] #3 Clearing selection for a source switch presents the empty-run detail state, including unavailable or empty catalog outcomes; selecting a later valid run populates its detail normally.
- [ ] #4 Normal nonterminal Local refresh continues to update current progress and preserve payload state; terminal and Server selections retain their existing skip behavior.
- [ ] #5 Targeted mounted controls hold the actual scope/service return, prove selection/source changes occurred before release, and verify final state with every hold released inside the App lifetime.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
