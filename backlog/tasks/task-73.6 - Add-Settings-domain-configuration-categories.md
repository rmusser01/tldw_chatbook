---
id: TASK-73.6
title: Add Settings domain configuration category contracts
status: Done
labels:
- settings
- library
- personas
- skills
- schedules
- workflows
dependencies:
- TASK-73.1
priority: medium
parent_task_id: TASK-73
documentation:
- Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add PR-sized Settings domain category contracts so major modules expose global defaults, status, or honest WIP/read-only ownership without replacing the destination that owns the actual workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library/RAG, Artifacts, Personas, Skills, Schedules, Watchlists, Workflows, MCP defaults, and ACP defaults have category contracts identifying source of truth, owner destination, and whether Settings can mutate anything.
- [x] #2 Categories with no PR-sized mutation source render explicit read-only/WIP ownership copy instead of save/revert controls.
- [x] #3 Library/RAG status includes later-stage citation/snippet display defaults where a source contract already exists; otherwise it records a follow-up task instead of inventing controls.
- [x] #4 Destination ownership remains explicit and no domain workflow is flattened into Settings.
- [x] #5 Mounted tests cover category presence, owner copy, and any real save/revert paths added in this contract pass.
- [x] #6 Actual CDP/Textual-web screenshot QA verifies every newly added or materially changed category and is approved before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing Settings contract tests for the required domain categories and their owner/read-only boundary copy.
2. Add domain category summaries, ownership records, state banners, inspector guidance, and read-only detail rows for Library/RAG, Artifacts, Personas, Skills, Schedules, Watchlists, Workflows, MCP defaults, and ACP defaults.
3. Keep all new domain categories read-only/WIP unless an existing PR-sized config mutation source is already identified.
4. Include Library/RAG citation/snippet later-stage language without inventing controls.
5. Run focused Settings/Destination verification, capture actual Textual-web/CDP screenshots, and update task evidence after user approval.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added read-only Settings domain contract categories for Library/RAG, Artifacts, Personas, Skills, Schedules, Watchlists, Workflows, MCP defaults, and ACP defaults.
- Each category now renders owner/source-of-truth copy, workflow boundaries, recovery guidance, and read-only state instead of fake mutation controls.
- Library/RAG explicitly records citation/snippet display defaults as follow-up `TASK-73.8` rather than inventing controls before a persisted source contract exists.
- Added mounted regressions covering category presence, ownership records, read-only copy, disabled save/revert controls, and destination ownership boundaries.
- Captured and approved actual Textual-web/CDP screenshot evidence at `Docs/superpowers/qa/product-maturity/phase-3/actual-visual-captures/settings-domain-library-rag-2026-05-31.png`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Settings now exposes the required domain configuration categories as honest ownership/status contracts. No domain workflow was flattened into Settings; all newly added categories remain read-only until a later PR identifies and wires a concrete persisted mutation source.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Unit, mounted, and integration tests relevant to changed behavior pass
- [x] #3 Static analysis and diff hygiene checks pass
- [x] #4 Actual app QA walkthrough completed with screenshots
- [x] #5 User approval recorded for visible Settings changes
- [x] #6 Documentation and task notes updated
- [x] #7 Task status moved to Done after implementation notes are added
<!-- DOD:END -->
