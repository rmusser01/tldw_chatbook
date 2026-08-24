---
id: TASK-21351
title: Add activity views to Ctrl+K session switcher
status: To Do
assignee: []
created_date: '2026-08-23 22:37'
labels: []
dependencies:
  - TASK-20937
references:
  - >-
    Docs/superpowers/specs/2026-08-22-console-edge-rails-workspace-tree-design.md#follow-up-ctrlk-active-conversation-view
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users quickly switch to conversations that are active now without losing historical-date browsing in the Console session switcher. Define and expose an explicit activity model before implementation so open tabs, running work, and recency are not conflated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task defines whether active means an open tab, currently running work, recent activity, or a documented ranked combination.
- [ ] #2 Ctrl+K lets users browse an activity-focused view and the existing historical-date view.
- [ ] #3 Activity ordering and filtering are deterministic, keyboard accessible, and do not alter Console rail projections.
- [ ] #4 Automated tests cover mixed active and historical conversations, ordering ties, and switching behavior.
<!-- AC:END -->

## Renumbering provenance

Two delayed Backlog CLI attempts assigned TASK-21201 and TASK-21202, both already claimed on remote branches at filing time. The duplicate TASK-21201 file was removed, and the surviving TASK-21202 follow-up moved to TASK-21351 before implementation or review references were created.
