---
id: TASK-73.6
title: Add Settings domain configuration category contracts
status: To Do
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
- [ ] #1 Library/RAG, Artifacts, Personas, Skills, Schedules, Watchlists, Workflows, MCP defaults, and ACP defaults have category contracts identifying source of truth, owner destination, and whether Settings can mutate anything.
- [ ] #2 Categories with no PR-sized mutation source render explicit read-only/WIP ownership copy instead of save/revert controls.
- [ ] #3 Library/RAG status includes later-stage citation/snippet display defaults where a source contract already exists; otherwise it records a follow-up task instead of inventing controls.
- [ ] #4 Destination ownership remains explicit and no domain workflow is flattened into Settings.
- [ ] #5 Mounted tests cover category presence, owner copy, and any real save/revert paths added in this contract pass.
- [ ] #6 Actual CDP/Textual-web screenshot QA verifies every newly added or materially changed category and is approved before PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Unit, mounted, and integration tests relevant to changed behavior pass
- [ ] #3 Static analysis and diff hygiene checks pass
- [ ] #4 Actual app QA walkthrough completed with screenshots
- [ ] #5 User approval recorded for visible Settings changes
- [ ] #6 Documentation and task notes updated
- [ ] #7 Task status moved to Done after implementation notes are added
<!-- DOD:END -->
