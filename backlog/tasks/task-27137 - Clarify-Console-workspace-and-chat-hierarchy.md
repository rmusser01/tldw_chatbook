---
id: TASK-27137
title: Clarify Console workspace and chat hierarchy
status: Done
assignee: []
created_date: '2026-09-02 03:52'
updated_date: '2026-09-02 04:02'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make workspace and conversation rows immediately distinguishable in the Console Context rail while keeping row actions predictable and usable at narrow widths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace rows render an @ action affordance at the far-right edge.
- [x] #2 Conversation rows render the existing * action affordance in the same far-right column and are indented four cells beneath their workspace.
- [x] #3 Long and narrow labels truncate before the affordance without obscuring it, and pointer plus keyboard menu activation still target the correct row.
- [x] #4 Targeted workspace-tree tests cover the glyphs, indentation, hit zones, tooltip boundaries, and ASCII fallback.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing Textual renderer and pointer tests for the approved @/* action-column hierarchy and four-cell chat indent.
2. Implement row-kind affordances, right-edge padding, and guide-aware hit zones in ConsoleWorkspaceTree.
3. Update stale action-column tooltip fixtures and affected row-menu documentation.
4. Run targeted tree/menu tests, Ruff, Impeccable detector, and diff hygiene checks.

ADR required: no
ADR path: N/A
Reason: This is a renderer-level refinement of the existing Console tree and row-menu contract; it adds no storage, security, runtime, or cross-module boundary.

Detailed plan: Docs/superpowers/plans/2026-09-01-console-workspace-chat-hierarchy.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented distinct right-edge workspace `@` and conversation `*` action
markers, four-cell child guides, label padding/truncation, and guide-aware
pointer hit zones in `ConsoleWorkspaceTree`. Preserved the existing menu
messages, keyboard `m` action, selection behavior, and action-menu styling.

Added renderer and real-Console coverage for marker identity, shared edge
alignment, pointer activation, native indentation, tooltip boundaries, narrow
layouts, and ASCII fallback. The clean `dev` baseline exposed six failures
from TASK-25712's adjacent marker geometry and stale pre-action-column width
fixtures; this slice resolves that click collision and updates those fixtures.

Updated the workspace action-menu and Workspace Files documentation to use
the new `@` affordance. No ADR was required because storage, ownership,
security, menu payloads, and cross-module boundaries are unchanged. No new
general lesson was added; the baseline-evidence incident is already covered
by `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
