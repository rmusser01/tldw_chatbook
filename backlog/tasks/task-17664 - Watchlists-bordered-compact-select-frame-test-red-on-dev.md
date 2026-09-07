---
id: TASK-17664
title: 'Watchlists: bordered compact-select frame test red on dev'
status: Done
assignee:
  - '@claude'
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
- [x] #1 The test is green on dev, with the regression fixed or the pin updated to the intended contract (decided by reproducing the styling live first — never probe a colour mechanism colorlessly)
- [x] #2 The task records which merge introduced the red
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce with detail (the failure was an IndexError on an EMPTY paint, not a styling assert).
2. Trace the exemplar's zero-size region to its cause; verify the contract on a living exemplar; repoint the pin.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stale exemplar, live contract. The pin (bordered compact Selects keep their painted frame at rest/focus/hover — the review-wave Critical 1 regression guard) exercised `#settings-provider-value`, and `484c74af2` ("feat: organize provider settings by user task", merged via PR #1630, 2026-08-13) replaced that visible Select with the search + OptionList picker flow, keeping the id only as a HIDDEN manual-entry compat control (`settings-provider-manual-hidden`, zero-size row) — so `_painted_rows` returned an empty list and the rest-read IndexErrored before any styling assertion ran. Not a focus-visibility regression.

The contract outlives the exemplar (~21 bordered compact Selects remain per the original docstring), so the pin now runs on `#settings-console-context-compaction-mode` (probed first: `-textual-compact` + `settings-compact-select`, paints a full frame at rest), scrolled into view the way a user reaches it, with the three-state assertions unchanged — the guarded regression (a blanket compact opt-out stripping frames on focus/hover) would still turn it red. Docstring records the swap. 11/11 green on the overlays file.

Files: `Tests/UI/test_watchlists_select_option_overlays.py`.
<!-- SECTION:NOTES:END -->
