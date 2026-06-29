---
id: TASK-142
title: Address Console beginner and advanced workflow UX findings
status: Done
assignee:
  - '@codex'
created_date: '2026-06-29 20:20'
updated_date: '2026-06-29 23:46'
labels:
  - ui
  - ux
  - textual
  - workbench
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improve the Console screen after senior UX/HCI review by removing duplicated control surfaces, making blocked first-run recovery directly actionable, making empty and inspector states useful, and preserving advanced regular-user density without hiding core actions in the command palette.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console has one canonical visible state/action control strip.
- [x] #2 Blocked setup state exposes direct recovery actions.
- [x] #3 Transcript empty state provides useful workflow launch actions.
- [x] #4 Inspector shows actionable blocked/run/source/tool/approval/artifact state or collapses when not useful.
- [x] #5 Disabled actions expose nearby reasons.
- [x] #6 Beginner and advanced workflow visual evidence is captured and reviewed.
- [x] #7 Targeted Console Workbench tests and visual snapshot checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: `Docs/superpowers/plans/2026-06-29-console-beginner-advanced-ux-remediation-plan.md`

ADR required: no

ADR path: `backlog/decisions/011-chatbook-workbench-ui-system.md`

Reason: this work implements the existing Workbench UI System decision. It corrects Console screen composition, state visibility, recovery affordances, and visual evidence inside the accepted Workbench frame without introducing new storage, provider, runtime, or navigation boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Consolidated Console state/actions into the Console-owned control strip while keeping compatibility Workbench seams hidden.
- Added direct setup recovery actions, activation-oriented empty transcript actions, disabled Send reasons, tighter staged-context density, and action-first inspector state.
- Refreshed Console visual evidence with normal, compact, command-palette, focus, and standard-width inspector SVG/PNG captures under `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/visual/`.
- Strengthened visual snapshot tests and Console contract/internals tests for the updated bounded Composer and inspector evidence.
- ADR check completed: no new ADR required; this implements `backlog/decisions/011-chatbook-workbench-ui-system.md`.
<!-- SECTION:NOTES:END -->
