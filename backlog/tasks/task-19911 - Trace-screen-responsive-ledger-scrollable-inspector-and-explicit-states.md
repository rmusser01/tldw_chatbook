---
id: TASK-19911
title: Trace screen responsive ledger scrollable inspector and explicit states
status: To Do
assignee: []
created_date: '2026-08-22 18:29'
labels: []
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-22-task-19907-trace-v2-exhaustive-collaboration-design.md
  - Docs/superpowers/plans/2026-08-22-task-19911-19912-trace-v2-interface.md
  - backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Trace usable for first-time, keyboard, accessibility, and narrow-terminal users while adopting Trace as the canonical Console label.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The inspector exposes every rendered field through independent keyboard scrolling and a full-pane detail mode
- [ ] #2 The ledger uses verified responsive tiers at 60x18, 80x24, 100x30, and 120x35 without hiding record identity or requiring horizontal scrolling for primary facts
- [ ] #3 Live following, paused, imported, loading, incomplete, filtered, empty, and failure states are explicitly visible and actionable
- [ ] #4 Trace is the canonical user-facing name and record kinds are humanized
- [ ] #5 Pilot geometry and compositor tests prove reachability and responsive behavior
<!-- AC:END -->
