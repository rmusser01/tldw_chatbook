---
id: TASK-16229
title: Fence stale Personas import presentation
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 09:00'
updated_date: '2026-08-14 09:05'
labels:
  - personas
  - lifecycle
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent durable character imports from refreshing or selecting through a Personas screen that navigation has already detached.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Character import persistence survives navigation away
- [x] #2 A detached initiating Personas screen receives no refresh, selection, or notification presentation
- [x] #3 A fresh Personas visit observes the durable imported character
- [x] #4 Focused ProductionApp import tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the existing failing in-flight-navigation regression as RED evidence.
2. Make the presentation-current predicate use attachment truth rather than Textual is_mounted, which remains true after pruning.
3. Run the focused ProductionApp flow, Personas tests, and static checks.

ADR required: no
ADR path: N/A
Reason: This fixes a stale-view lifecycle predicate without changing durable import ownership.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed both stale presentation paths exposed by the production flow. Durable import completion now requires a live, parented, current Personas screen before any refresh, selection, or notification work. The investigation also showed the deferred initial-load worker could reach refresh after unmount, so it now exits before touching character rows when its screen is closed, detached, or no longer current. Durable persistence remains app-owned and a fresh Personas visit loads the imported character. Verification: the complete ProductionApp Personas/Library ownership file passed 3 tests; Ruff lint/format, py_compile, and git diff --check passed.
<!-- SECTION:NOTES:END -->
