---
id: TASK-37330
title: Restore Research Runs artifact loading through the real scope contract
status: To Do
created_date: 2026-10-29 00:56
labels:
- research
- bug
- service-contract
priority: medium
references:
- tldw_chatbook/UI/Research_Modules/research_controller.py
- tldw_chatbook/Research_Interop/research_scope_service.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow Research Runs to display its persisted artifacts through the real scope service. The current controller-service boundary rejects the artifact request before its content can render, while permissive service doubles do not detect the incompatibility.

**Evidence:** During mounted TASK-37267 fixture setup on baseline `8fbc9d2e1d91c31654742736f652d173b07b3ea9`, the real Local bundle method chain reached artifact loading and raised `TypeError` before the timer assertion. This is retained as an invalid timer probe, not evidence that refresh failed. Source inspection confirms the controller's positional request conflicts with the real scope's keyword-only contract; blame predates this branch. A real Load Bundle button journey has not yet been qualified for this separate defect. Receipts and qualifications belong to the [implementation QA](../../Docs/superpowers/qa/2026-10-28-textual-followups-implementation.md).

**Existing owners:** [TASK-16483](<task-16483 - Readable-bundle-and-artifact-inspection-in-the-Research-window.md>) delivered readable/default artifact presentation using a permissive fake-service auto-load test. [TASK-21127](<task-21127 - Research runs - per-op leaked connects and loop-side periodic reads-writes while a run is active.md>) owns executor/connection behavior. [TASK-19300](<task-19300 - Make-research-runs-durable-resumable-jobs.md>) owns durable execution and reopening from the separate Artifacts destination. [TASK-37267](<task-37267 - Keep-Research-Runs-refresh-tied-to-its-current-source-and-selection.md>) owns timer/source publication and deliberately leaves this call contract unchanged. Research Studio output work is a distinct surface under ADR-078. No discovered task version or ref/worktree implementation owns this specific correction at the recorded snapshot.

**ADR assessment:** New ADR expected: no; ADR path: N/A. Existing [ADR-033](../decisions/033-application-session-state-ownership.md), [ADR-068](../decisions/068-local-research-execution-engine.md) and [ADR-078](../decisions/078-research-workspace-authority-and-screen-boundaries.md) retain the window/controller/service boundaries. Correct the caller to the existing contract; do not introduce a second artifact store, change the service API, or broaden into execution durability or Workspace outputs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Loading a real Local run's bundle through the mounted Runs action displays its actual default report artifact without a calling-convention error or source change.
- [ ] #2 Choosing a specific persisted artifact displays the correct selected run/artifact content through the real scope boundary.
- [ ] #3 Targeted contract and mounted tests expose incompatible controller-service calls; permissive doubles cannot mask a mismatch with the real scope API.
- [ ] #4 Missing/unavailable artifact behavior and service executor ownership retain their established outcomes; no server fallback or storage/durability change is introduced.
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
