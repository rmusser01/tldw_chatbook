---
id: TASK-33160
title: 'CI throughput sub-project 1: 2-week success review (due 2026-10-11)'
status: Done
assignee: []
created_date: 2026-09-28 02:10
labels: []
dependencies: []
updated_date: 2026-10-18 18:10
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measure the outcome of CI throughput sub-project 1 (spec Docs/superpowers/specs/2026-09-27-ci-conflicts-and-waste-design.md) against its stated success measures, about two weeks after rollout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The sync-merge conflict rate, measured with `python scripts/measure_sync_merge_conflicts.py --merges 1000 --since 2026-09-27` (1000, not 300: two weeks at 23-50 merges/day is 322-700 dev merges; first confirm the scan reaches the rollout by checking that `git log origin/dev --first-parent --merges -n 1000 --format=%cI | tail -1` predates 2026-09-27), is <= 25% (baseline 50%, 558 sync merges / 281 conflicted, measured 2026-09-28 with the final meter at origin/dev 7cda012822 -- nearly all pre-change syncs; the earlier 572/297 (51%) figure came from the meter before its second-parent-on-dev filter, so do not compare against it)
- [x] #2 Runner-minutes/day for the changed workflows (GGUF x2, the deleted guards, the nightly) is >= 80% lower than the ~1,100/day baseline, or the owner accepts the measured figure (77% lower on 2026-10-18, accepted)
- [x] #3 0 fast-lane failures carrying text-area--gutter since #2866 merged; this closes TASK-32049's remaining AC
- [x] #4 The throughput set is reported: merges/day; median ready-to-merged time; required-check push-to-verdict p50/p90 split into queue and run time; re-syncs per merged PR; DIRTY share of open PRs
- [x] #5 Results are recorded in this task's notes and in Docs/superpowers/specs/2026-09-27-ci-conflicts-and-waste-design.md
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review run on 2026-10-11 at origin/dev 8332699ae5. Results table: the "Results" subsection of Docs/superpowers/specs/2026-09-27-ci-conflicts-and-waste-design.md. Window for the API measures: 2026-09-28T02:00:55Z (#2866 merged) to 2026-10-11T00:00Z, 12.92 days, 5,807 workflow runs listed in 6-hour windows (none near the 1,000-run listing cap).

**AC #1, met: 12% (9 of 70).** `python scripts/measure_sync_merge_conflicts.py --merges 1000 --since 2026-09-27`. The scan reaches 2026-08-16, so it covers the rollout. Conflicting files: the diagnostic inventory 3, lessons-testing-evidence.md 3, six others once each. Only 70 sync merges exist because the rule since 2026-09-28 is to rebase.

**AC #2, NOT met: 334 runner-minutes/day, 70% below ~1,100 (target 80%).** Sum of job time on a runner (completed minus started) for every run of the five workflows in the window; rounding each job up to a whole minute gives 370/day (66%). Nightly Deep 0 runs; each deleted guard 1 run of 0.3 min; GGUF source 230 runs / 687 jobs / 3,903 min; GGUF import 107 runs / 316 jobs / 407 min. All 337 GGUF runs are `pull_request` runs, 70 to 765 minutes on every day of the window. Against GGUF's own baseline (16% of 22.5k minutes over 8 days, about 450/day) that is 26% lower. Their path filters still include `pyproject.toml`, `Tests/conftest.py` and `Tests/UI/conftest.py`, and each run starts three OS jobs. Left unticked; closing it needs an owner decision on the GGUF triggers.

**AC #3, met: 0 of 381.** Every job named "PR Fast Lane" or "UI Fast Lane" with conclusion `failure`, across all 1,517 required-workflow runs in the window, all attempts, cancelled runs included: 381 jobs. All 381 logs were read (none expired); none contains `text-area--gutter`. Positive control: the same scan over 2026-09-25 to 2026-09-28T02:00Z finds it in 5 of 18 failed fast-lane jobs. TASK-32049's AC #3 is ticked on this evidence.

**AC #4, reported.**
- Merges into dev per day: 13.8 (180 PRs, 09-28..10-10; 3 to 28 a day). Baseline 09-19..26: 13.9 (111 PRs).
- Ready to merged (last ready-for-review event, else PR creation): median 11.6 h, p90 70.9 h. Baseline 1.8 h / 38.3 h.
- Required check push-to-verdict, 1,030 first-attempt `pull_request` runs that reached a verdict: p50 36.1 / p90 123.2 min. Queue (time with none of the run's jobs on a runner) 14.3 / 102.1; run 20.9 / 27.1. From 10-06 the daily p50 is 27 to 32 min and p90 31 to 47.
- Re-sync runs per merged PR: `python scripts/measure_required_runs_per_merge.py --prs 80` gives 421 required runs, median 2.0, 183 beyond one (2026-10-03: 531, 2.0, 218).
- DIRTY share: 15 of 25 open PRs into dev are CONFLICTING, sampled once on 2026-10-11 01:35Z. It was not sampled daily, so there is no series. On 2026-09-27 it was 4 of 34.

**Reading.** The two attributable fixes that were measured held (conflict rate, the flake). Throughput did not improve: merges per day are flat and ready-to-merged is about six times longer. The baseline ran under strict=false, the merge queue only went on on 2026-10-10, and tldw_server's shard cap merged on 10-04, so these periods are not like for like.

**Method.** AC #1 and the re-sync count use the committed meters. The other measures came from one-off read-only `gh api` scripts that were not committed. Files changed: this task, TASK-32049, and the spec's Results subsection.

**AC #2 re-measured 2026-10-18: 250 runner-minutes/day, 77% lower; the owner accepted it, so the task is Done.** Window: the 7 days after #3072 merged (2026-10-11T17:19:20Z to 2026-10-18T17:19:20Z); the three workflow files were unchanged in it, and every run had completed. Same per-job method as above: GGUF source 1,484 min (83 runs, 245 jobs), GGUF import 264 min (61 runs, 183 jobs), Nightly Deep 0 (disabled). Rounding each job up to a whole minute gives 282/day (74%). Each PR was classified by its final file list: 199/day came from PRs that touched only `Tests/conftest.py` (117/day: #3083, #3087, #3121, #3080, #2862), only `pyproject.toml` (70/day: #3084, #3127, #3138) or both (13/day: #3096). The other 51/day came from PRs touching a GGUF path of their own (#3023, #3099, #3148) plus 2 dispatches. The owner kept those shared paths on 10-11 (restored on #2860 at review; the GGUF jobs are the only macOS and Windows evidence for them). Options put to the owner: drop the root conftest (about 88% lower) or all three paths (about 95%). The owner accepted 77% instead: the 30 min/day gap is smaller than the week-to-week swing (the 10-11 replay predicted 202/day). AC #2's text now records that acceptance. Read-only scratch script, not committed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
