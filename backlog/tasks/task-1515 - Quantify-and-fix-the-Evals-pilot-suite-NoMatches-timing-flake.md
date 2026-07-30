---
id: TASK-1515
title: >-
  Quantify and fix the Evals pilot-suite NoMatches timing flake
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
Observed twice during the 2026-07-30 UAT fix batch (Tasks 3 and 7, independent implementers): combined-suite runs of the Evals UI pilot tests intermittently fail 1-5 UNRELATED tests with `NoMatches` (e.g. '#evals-results-grid') or KeyError — a different test each run, every affected test passing in isolation, clean on rerun. Machine load is a suspected contributor (heavily shared dev box), but this repo's standing lesson is that "passes on retry" is not automatically a flake. Needs: quantification (N reruns of the combined suite, failure census), then either a real race fix (mount/recompose timing in the shared fixtures) or a documented, bounded mitigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] The flake is reproduced and its trigger identified (or bounded by a census of ≥10 combined-suite runs)
- [ ] The root cause is fixed, or a documented mitigation lands with the census attached
- [ ] Combined-suite reruns are stable afterwards
<!-- AC:END -->
