---
id: TASK-142
title: Address Console beginner and advanced workflow UX findings
status: To Do
assignee:
  - '@codex'
created_date: '2026-06-29 20:20'
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
- [ ] Console has one canonical visible state/action control strip.
- [ ] Blocked setup state exposes direct recovery actions.
- [ ] Transcript empty state provides useful workflow launch actions.
- [ ] Inspector shows actionable blocked/run/source/tool/approval/artifact state or collapses when not useful.
- [ ] Disabled actions expose nearby reasons.
- [ ] Beginner and advanced workflow visual evidence is captured and reviewed.
- [ ] Targeted Console Workbench tests and visual snapshot checks pass.
<!-- AC:END -->

## Implementation Plan

Plan: `Docs/superpowers/plans/2026-06-29-console-beginner-advanced-ux-remediation-plan.md`

ADR required: no

ADR path: `backlog/decisions/011-chatbook-workbench-ui-system.md`

Reason: this work implements the existing Workbench UI System decision. It corrects Console screen composition, state visibility, recovery affordances, and visual evidence inside the accepted Workbench frame without introducing new storage, provider, runtime, or navigation boundaries.
