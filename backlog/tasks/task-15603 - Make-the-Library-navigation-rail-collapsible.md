---
id: TASK-15603
title: Make the Library navigation rail collapsible
status: Done
assignee:
  - '@codex'
created_date: '2026-08-12 03:21'
labels: []
dependencies: []
documentation:
  - Docs/User_Guide/library.md
  - backlog/decisions/011-chatbook-workbench-ui-system.md
priority: high
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users reclaim Library canvas width by collapsing the left navigation rail without losing its route, query, disclosure, focus, or canvas state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Wide Library layouts expose an explicit Collapse control in the navigation rail and replace the collapsed rail with a focusable Expand navigation handle
- [x] #2 Collapse and expand update the existing mounted shell in place so route selection search text section disclosure state and the active canvas survive unchanged
- [x] #3 Collapsing moves focus to the Expand navigation handle expanding moves focus into rail search and F6 includes whichever rail surface is currently visible
- [x] #4 The existing compact Notes single-stage router remains authoritative below its breakpoint and a manual wide-layout collapse is restored when the terminal returns wide
- [x] #5 The canvas receives the released width no new shortcut or footer hint is introduced and mounted wide compact resize and keyboard tests plus static checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; conform to `backlog/decisions/011-chatbook-workbench-ui-system.md`.
Reason: this reuses the established destination rail handle and keeps collapse state only in the mounted Library screen session; it adds no persisted setting, route contract, or keybinding.

1. Add a production Library shell test for collapse, state retention, focus, F6, canvas expansion, and compact breakpoint behavior.
2. Add an explicit rail heading Collapse control and the shared collapsed-rail handle.
3. Apply visibility in place through the existing Library shell stage synchronizer, with compact Notes routing taking precedence.
4. Update the Library guide and run focused Library shell and File Notes regressions plus static checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Added a restrained Navigation heading with Collapse and a three-column-wide, keyboard-focusable Nav handle built on the existing destination-rail handle. Collapse updates mounted visibility instead of recomposing the shell, so the route, query, disclosures, selection, and canvas survive; compact Notes routing remains authoritative and restores the manual collapse after returning wide.

- Updated the Library rail, screen integration, widget exports, mounted shell coverage, and Library guide.
- Verified canvas width reclamation, focus handoff, F6 cycling, state retention, compact breakpoint precedence, wide restoration, rail scrolling, and recompose behavior.
- Focused Ruff passes with the repository's pre-existing `E721` and unused-import diagnostics excluded in the two large legacy files; `git diff --check` passes.
- ADR required: no. The implementation conforms to `backlog/decisions/011-chatbook-workbench-ui-system.md` and adds no persisted setting or keybinding.
