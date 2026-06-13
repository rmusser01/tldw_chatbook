---
id: TASK-73.6
title: Add Settings domain configuration category contracts
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-01 03:00'
labels:
  - settings
  - library
  - personas
  - skills
  - schedules
  - workflows
dependencies:
  - TASK-73.1
documentation:
  - Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md
parent_task_id: TASK-73
priority: medium
---

## Description

Add PR-sized Settings domain category contracts so major modules expose global defaults, status, or honest WIP/read-only ownership without replacing the destination that owns the actual workflow.

## Acceptance Criteria

- [x] #1 Library/RAG, Artifacts, Personas, Skills, Schedules, Watchlists, Workflows, MCP defaults, and ACP defaults have category contracts identifying source of truth, owner destination, and whether Settings can mutate anything.
- [x] #2 Categories with no PR-sized mutation source render explicit read-only/WIP ownership copy instead of save/revert controls.
- [x] #3 Library/RAG status includes later-stage citation/snippet display defaults where a source contract already exists; otherwise it records a follow-up task instead of inventing controls.
- [x] #4 Destination ownership remains explicit and no domain workflow is flattened into Settings.
- [x] #5 Mounted tests cover category presence, owner copy, and any real save/revert paths added in this contract pass.
- [x] #6 Actual CDP/Textual-web screenshot QA verifies every newly added or materially changed category and is approved before PR.

## Implementation Plan

1. Add failing Settings regressions for domain category IDs, ownership records, read-only/WIP copy, and citation/snippet follow-up copy.
2. Add domain category identifiers and summaries without enabling save/revert controls.
3. Render a generic read-only domain contract detail pane and category-specific inspector guidance.
4. Update the Settings plan/task evidence and run focused verification plus diff hygiene.
5. Capture actual Textual-web/CDP Settings screenshot evidence for user approval before PR.

## Implementation Notes

- Added read-only domain category contracts for Library/RAG, Artifacts, Personas, Skills, Schedules, Watchlists, Workflows, MCP defaults, and ACP defaults.
- Domain categories identify owner destinations, source-of-truth seams, explicit read-only status, and follow-up copy instead of inventing mutation controls.
- Library/RAG includes later-stage citations/snippets defaults as follow-up copy.
- Added mounted regressions for domain category presence, ownership records, read-only save/revert state, and rendered Library/RAG plus MCP defaults detail panes.
- Kept Diagnostics and Advanced Config visible before the longer Domain Defaults group and made the category pane scrollable.
- Captured actual textual-web/CDP PNG evidence and recorded user approval before PR closeout.

## Final Summary

Added read-only Settings domain configuration contracts for major product modules, verified them with mounted regressions, captured actual textual-web/CDP screenshot evidence, and recorded user approval.

## Definition of Done

- [x] #1 Acceptance criteria completed
- [x] #2 Unit, mounted, and integration tests relevant to changed behavior pass
- [x] #3 Static analysis and diff hygiene checks pass
- [x] #4 Actual app QA walkthrough completed with screenshots
- [x] #5 User approval recorded for visible Settings changes
- [x] #6 Documentation and task notes updated
- [x] #7 Task status moved to Done after implementation notes are added
