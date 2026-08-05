---
id: TASK-1510
title: >-
  Evals rail running glyph is unreachable in the press-and-watch flow
status: To Do
assignee: []
created_date: '2026-07-30 14:00'
labels:
  - evals
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Whole-branch review finding (F3) of the 2026-07-30 UAT fix batch. Nothing recomposes the Catalog rail when a bench run starts — the press handler does not select, and progress ticks touch only the primary-action button — so the spec-advertised in-flight run row (`● HH:MM · name`) appears only if the user happens to click something mid-run. The button's "Running… (n/m)" label carries the signal, so this is a visibility gap, not a break. A start-of-run refresh is non-trivial: the run group row does not exist until `WordBenchRunner.run` creates it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Starting a bench run makes the in-flight run row (● glyph) visible in the rail without further user interaction
- [ ] The rail returns to the settled state on completion without breaking selection
<!-- AC:END -->
