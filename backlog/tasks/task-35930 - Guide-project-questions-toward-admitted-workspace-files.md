---
id: TASK-35930
title: Guide project questions toward admitted workspace files
status: Done
assignee:
  - '@codex'
created_date: '2026-10-21 04:23'
updated_date: '2026-10-21 04:56'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-10-20-workspace-project-guidance.md
  - backlog/decisions/241-workspace-project-context-and-folder-bound-index.md
---


Current governance: [ADR-241](../decisions/241-workspace-project-context-and-folder-bound-index.md). Renumbered from project ADR-240 during 2026-10-22 dev integration; historical rulings and delivery notes below retain their original source-era labels.

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Natural project questions should inspect available workspace files while describing only the tools actually offered.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An unindexed project request receives guidance to inspect its admitted folder through offered workspace readers.
- [x] #2 Missing or goal-filtered readers produce no exploration promise; scratch and root addressing remain accurate.
- [x] #3 Saved Library policies, scopes, permissions and project-instruction authority remain unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow G1 in Docs/superpowers/plans/2026-10-20-workspace-project-guidance.md: write failing first-request/capability regressions, add conditional exploration after final narrowing, run focused tests and lint, review. ADR required: yes; ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md; Reason: implements accepted guidance contract without storage migration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented capability-aware project guidance after final plugin/goal narrowing in8e02f7883d. Twelve new capability/prompt cases pass with causal ordering RED; five real fs_list cases pass after native identity correction. Independent spec/quality review approved; exact approved paragraph verified. Saved Library controls/scopes, permissions and instruction authority unchanged; no indexing or migration. Existing broader Windows/fixture failures, dependency warnings and13 baseline bridge lint findings are recorded separately. Files: Tools/workspace_file_roots.py, Chat/console_agent_bridge.py, their targeted contract tests. ADR: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md.
<!-- SECTION:NOTES:END -->
