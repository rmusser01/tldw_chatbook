---
id: TASK-33160
title: 'CI throughput sub-project 1: 2-week success review (due 2026-10-11)'
status: To Do
assignee: []
created_date: '2026-09-28 02:10'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measure the outcome of CI throughput sub-project 1 (spec Docs/superpowers/specs/2026-09-27-ci-conflicts-and-waste-design.md) against its stated success measures, about two weeks after rollout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The sync-merge conflict rate, measured with `python scripts/measure_sync_merge_conflicts.py --merges 1000 --since 2026-09-27` (1000, not 300: two weeks at 23-50 merges/day is 322-700 dev merges; first confirm the scan reaches the rollout by checking that `git log origin/dev --first-parent --merges -n 1000 --format=%cI | tail -1` predates 2026-09-27), is <= 25% (baseline 50%, 558 sync merges / 281 conflicted, measured 2026-09-28 with the final meter at origin/dev 7cda012822 -- nearly all pre-change syncs; the earlier 572/297 (51%) figure came from the meter before its second-parent-on-dev filter, so do not compare against it)
- [ ] #2 Runner-minutes/day for the changed workflows (GGUF x2, the deleted guards, the nightly) is >= 80% lower than the ~1,100/day baseline
- [ ] #3 0 fast-lane failures carrying text-area--gutter since #2866 merged; this closes TASK-32049's remaining AC
- [ ] #4 The throughput set is reported: merges/day; median ready-to-merged time; required-check push-to-verdict p50/p90 split into queue and run time; re-syncs per merged PR; DIRTY share of open PRs
- [ ] #5 Results are recorded in this task's notes and in Docs/superpowers/specs/2026-09-27-ci-conflicts-and-waste-design.md
<!-- AC:END -->
