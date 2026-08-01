---
id: TASK-1673
title: Wizard step compose() crash policy from spec §5 unimplemented
status: Done
assignee: []
labels:
  - ui
  - onboarding
  - wizard
priority: medium
---

## Description

The first-run setup wizard spec (Docs/superpowers/specs/2026-07-28-first-run-setup-wizard-design.md §5)
requires that a step whose `compose()` raises is auto-skipped with a one-line
notice and a reasoned Summary row, rather than crashing the wizard screen.
That policy was never implemented.

Originally filed as TASK-1266, which collided with an unrelated existing task
of that id (TTSPlaygroundWidget retirement); renumbered here and dev's task
restored.

## Acceptance Criteria

- [x] #1 A step whose compose raises does not crash the wizard screen
- [x] #2 The failed step is dropped from navigation and progress
- [x] #3 The Summary reports the skipped step with a reason
- [x] #4 Pilot tests force a compose failure and assert survival + summary row

## Implementation Notes

`SetupStep.compose` is now a final wrapper over per-step `compose_step()`: on
exception it logs, flags `compose_failed`, and renders a one-line skip notice;
`__init_subclass__` guards each step's own `on_mount`/`on_show` against the
gutted DOM; `_refresh_active_ids` drops failed steps from navigation (refreshed
after composition so the very first render already excludes them); `SummaryStep`
appends a reasoned ✗ row per skipped step. Two Pilot tests force
`RagStep.compose_step` to raise and assert survival + the summary row.
