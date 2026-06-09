---
id: TASK-87
title: Settings remaining category functional QA sweep
status: To Do
assignee: []
created_date: '2026-06-09 01:03'
updated_date: '2026-06-09 01:06'
labels:
  - settings
  - ux
  - qa
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit and verify the Settings screen categories not covered by recent focused Settings slices through actual rendered app use. The goal is to identify remaining broken controls, placeholder states, and category ownership gaps before claiming Settings is a real configuration hub.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Actual CDP/Textual-web walkthrough covers the remaining Settings categories that still rely on read-only contracts or incomplete mutation paths: Overview, Storage, Privacy & Security, Diagnostics, Advanced Config, Server/Sync/Workspace/Handoff, and Domain Defaults for Library/RAG, Artifacts, Personas, Skills, Schedules, Watchlists, Workflows, MCP, and ACP.
- [ ] #2 Every category records whether it is guided-editable, read-only status/recovery, owned by another destination, or an explicit WIP placeholder.
- [ ] #3 Confirmed usability blockers receive a failing regression before fixes, or become child Backlog tasks with evidence when they are outside this sweep.
- [ ] #4 Dropdown/input focus, save/revert/test feedback, and keyboard operation are verified for each editable category.
- [ ] #5 Actual screenshots and QA notes are recorded, and user approval is captured before a functional Settings PR is created.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 ADR check is documented before implementation starts; create or link an ADR only if the sweep changes storage, runtime, provider, sync, or ownership boundaries.
- [ ] #2 Focused automated regressions cover every in-scope code change made from confirmed Settings blockers.
- [ ] #3 Actual CDP/Textual-web screenshots and QA notes are attached for each approved Settings fix or category sweep result.
- [ ] #4 Verification commands and residual risks are recorded in Implementation Notes before the task can move to Done.
<!-- DOD:END -->
