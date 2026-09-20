---
id: TASK-235
title: Re-resolve inspector tool view on MCP catalog resync
status: Done
assignee:
  - '@codex'
created_date: '2026-07-16 15:19'
updated_date: '2026-09-20 18:03'
labels:
  - mcp
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The inspector's tool detail (schema, stale flag) is captured at selection time and never re-resolved when a refresh/lifecycle action changes the catalog — detail can contradict the table until reselection. Run against a vanished tool fails cleanly, so this is staleness, not a crash (PR #639 final review N2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Catalog resync re-resolves the currently shown tool by identity,Vanished tool clears the inspector tool view,Changed schema is reflected in an open Test panel or the panel is closed with a notify
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/161-component-pattern-library.md and backlog/decisions/170-table-repopulation-selection-boundary.md. Reason: administrative reconciliation of an already implemented inspector currentness repair. Verify the legacy acceptance criteria against merged PR2757, link its current-source evidence, and close the duplicate work item without adding product changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Merged PR2757 fulfills this legacy inspector refresh item: catalog resync re-resolves the selected tool by identity, missing tools clear detail, and changed schema/definition closes the old Test panel with persistent reopening guidance. Equal definition and policy facts preserve argument drafts, focus/cursor and preview ownership. Evidence: Docs/superpowers/qa/2026-09-20-mcp-inspector-refresh/qodo/README.md and pr2757-closeout.json. 316 current-source local targeted cases, 1,152 Fast Lane cases, all applicable CI and the private native journey pass. Existing Workflows dimension test debt is unchanged and outside this item. Owner-approved visuals retained and actual merged tree verified. ADR-161/170 apply; no new ADR or additional product change.
<!-- SECTION:NOTES:END -->
