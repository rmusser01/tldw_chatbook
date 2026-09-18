---
id: TASK-32823
title: Keep selected MCP tool details current during catalog refresh
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-18 21:46'
updated_date: '2026-09-18 21:52'
labels:
  - mcp
  - ui
  - selection
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A refreshed Tools catalog must not leave obsolete selected-tool metadata or test controls in the inspector, and unchanged catalogs must preserve the user argument draft and focus.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Catalog refresh updates the selected tool details and clears removed tools without selecting a replacement.
- [ ] #2 Unchanged definitions preserve the mounted argument draft and focus; changed definitions retire old test controls and permission previews with visible guidance.
- [ ] #3 Targeted refresh, selection and prepared-test regressions plus bounded native evidence qualify the behavior; review ledgers and the draft PR record remaining scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/161-component-pattern-library.md and backlog/decisions/170-table-repopulation-selection-boundary.md. Reason: repair current selected-detail projection within existing ownership and permission-preview boundaries, without changing service admission or table selection contracts. 1. Reproduce metadata/schema removal and unchanged-draft cases through the mounted workbench refresh. 2. Refresh only the still-selected inspector tool under its existing refresh lock; preserve unchanged forms and retire obsolete controls with guidance. 3. Verify targeted selection/preview regressions, native private-profile controlled catalog changes and clean lifecycle. 4. Independent review, evidence/ledgers and save to draft PR2707. Fresh 230-ref/33-worktree scan found max32822 before allocation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved unfinished at the user-approved PR2707 closeout boundary. Prototype and four-case red/green evidence saved for a follow-up branch. Independent read-only review found a P2 delayed-preview mint race during awaited form teardown; fix and deterministic regression still required. Additional adjacent and native verification remain pending. Do not mark Done or merge this prototype. Resume after PR2707 merges from current dev, porting the preserved commit. Evidence: Docs/superpowers/qa/2026-09-18-mcp-inspector-refresh/README.md.
<!-- SECTION:NOTES:END -->
