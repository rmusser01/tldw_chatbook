---
id: TASK-32792
title: Keep MCP permission states visible after final table reflow
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 14:30'
updated_date: '2026-09-18 14:52'
labels:
  - mcp
  - ui
  - layout
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve complete permission labels and exact selected-row actions when scrollbar and content changes resize the matrix after its outer canvas has settled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tool and State remain fully visible at the left scroll position after final child viewport changes, including short/long matrix and outer-scrollbar transitions.
- [x] #2 Reflow preserves selected row identity, current focus, filters and exact permission context without emitting actions, reloading services or changing policies.
- [x] #3 Reflow converges without repeated unchanged rebuilds; targeted regressions and native compact/wide dark/light evidence qualify the repair and update the review ledgers.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce child-only and short/long matrix viewport changes with painted-cell assertions. 2. Observe final table resize using the existing Tools pattern and retain measured-width gating and row keys. 3. Verify selected-row visibility, filters and exact action context with targeted tests and independent review. 4. Qualify real private native dark/light compact/wide journeys and record unchanged permissions and clean shutdown. 5. Update ledgers/QA and save to draft PR2707. ADR required: no. ADR path: N/A (existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md). Reason: Bounded rendering correction preserving existing service, permission and navigation boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Observe the Permissions table final child viewport with an explicitly namespaced resize message. Width-gated rebuilding preserves row identity and current context; deferred current-focus reveal keeps the selected row visible on width and height-only changes. Clear the stale activation marker so a new Enter after rebuilding is delivered.

Evidence: 15 final focused reflow/compact cases, 26 final token/component cases, and 66 adjacent Permissions/handoff cases (the latter began before the final height-only callback). Independent review reproduced width clipping and the height-only residual, then verified the final correction. Final native run003 passed dark/light 80x24 and 170x48 with 12 visually inspected captures, real catalog inspection/filter/refresh, unchanged permission profiles, normal exit, ten healthy private databases and unchanged default-profile fingerprints. The first native failure and pre-final run002 receipts are retained honestly.

QA: Docs/superpowers/qa/2026-09-18-mcp-permission-reflow/README.md. Updated MCP/component ledgers and the existing final-scrollbar lesson. New-file lint/format and changed-method formatting pass; three inherited module lint findings remain unchanged. Backlog IDs, diagnostic inventory and diff checks pass. No full suite or tool execution. Existing ADR-150/161 apply; no new ADR required. PR2707 stays draft/unmerged pending its own visual approval. Master-toggle and remaining MCP/other component reviews remain open.
<!-- SECTION:NOTES:END -->
