---
id: TASK-17664
title: 'Watchlists: bordered compact-select frame test red on dev'
status: To Do
assignee: []
created_date: '2026-08-18'
labels:
  - watchlists
  - test-health
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_watchlists_select_option_overlays.py::test_a_bordered_compact_select_keeps_its_frame_under_focus_and_hover` fails on clean origin/dev — verified 2026-08-18 solo in a detached baseline worktree at `2b11a709e`, identical failure on a task-17663 branch that touches only two watchlists pagination tooltips. Found during task-17663's collateral sweep; fourth pre-existing dev red surfaced by this programme's sweeps (after 17656, 17660, 17663), which suggests the merge velocity is outrunning per-PR test coverage of neighboring suites.

Needs the usual decide-by-reproducing pass: either a bordered compact Select genuinely loses its frame under focus/hover (a real focus-visibility regression in a theme/CSS change) or the pin is stale against an intended styling change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The test is green on dev, with the regression fixed or the pin updated to the intended contract (decided by reproducing the styling live first — never probe a colour mechanism colorlessly)
- [ ] #2 The task records which merge introduced the red
<!-- AC:END -->
