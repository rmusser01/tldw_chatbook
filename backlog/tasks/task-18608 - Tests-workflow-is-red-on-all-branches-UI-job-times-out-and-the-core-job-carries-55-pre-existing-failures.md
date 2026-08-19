---
id: TASK-18608
title: >-
  Tests workflow is red on all branches: UI job times out and the core job
  carries 55 pre-existing failures
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-19 07:45'
updated_date: '2026-08-19 15:06'
labels:
  - ci
  - testing
  - infrastructure
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing/merging PR #1824. The `Tests` workflow
(`.github/workflows/test.yml`) currently cannot go green on ANY branch:

1. **UI Tests job dies at the ~45-minute job cap.** 13,558 tests run on a
   2-worker (2-core runner) xdist grid; the job is cancelled at roughly 11%
   progress every time. The last 30 runs of this workflow across ALL
   branches -- feature branches and `dev` itself -- are `cancelled`; there
   is no recent green run. The job's own pytest `--timeout=180` is not the
   binding constraint; the wall-clock budget is.
2. **Core Tests job has 55 failing tests** (macOS run of PR #1824's head,
   2026-08-19: 20,500 passed / 55 failed, exit 3), concentrated in
   Notes (note_import_planner, file_notes_git_*), TTS (audio_cpp supervisor
   and request admission), LLM_Calls summarization diagnostic privacy,
   Image_Generation comfyui adapter, Architecture diagnostic inventory,
   Character_Chat visual identity, and one application-state ownership
   test. Spot-checks reproduced 10 of them identically on clean
   `origin/dev` -- none are attributable to any single recent PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both core-suite legs finish within their job budget on a PR (budget raised 60->120m: the ubuntu leg was killed at its own 60m cap mid-run while macOS finished at ~58m).
- [x] #2 The UI job finishes within its 45-minute budget, via a 12-way deterministic pytest-shard matrix (~5.8 serial-equivalent hours / 12 = ~30 min per shard), with xdist retained within each shard and per-shard result artifacts.
- [x] #3 The workflow shape contract (Tests/CI/test_github_actions_test_workflow.py) pins the sharding: ids 0..N-1 matching --num-shards, >=10 shards, per-shard artifact names.
- [x] #4 With the suite able to COMPLETE, the ubuntu leg's full failure list is captured (it died at 42% before; no artifact existed) and every failure is triaged: fixed, or filed with names for a follow-up task.
- [x] #5 The macOS leg's 55 failures (all pass on a local macOS 3.12; runner-env-specific, concentrated in subprocess/thread/SSH/fs-sensitive tests) are triaged with a filed follow-up; they must not mask NEW failures.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Shard UI job via pytest-shard matrix (suite needs ~5.8h vs 45m budget)\n2. Raise core job timeout 60->120 (ubuntu leg killed at 60m)\n3. Keep lease-shape contract test and summary script consistent\n4. Triage 55 core failures locally; fix tractable, quarantine rest with follow-ups\n5. Verify via PR CI run
<!-- SECTION:PLAN:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
Evidence: run 32256433043 (PR #1824 @ a9b4b7e26): UI job cancelled at 45m
at 11%; macOS core job exit 3 with the 55 failures listed in its
`core-test-results` artifact; `gh run list --workflow Tests` shows 30/30
`cancelled`. PR #1824 was merged on the strength of its own suites
(1,903 tests green on the rebased head) after confirming the CI red was
pre-existing on clean dev.

**Phase 1 landed here (infrastructure):** core timeout 60->120 (ubuntu was
killed by its own cap at 42% with 43 failures visible but unnameable -- no
summary, no artifact); UI job sharded 12 ways with pytest-shard (the
unsharded job died at ~11% of 13,558 tests on every run; 100+ consecutive
cancelled runs, zero green, across all branches including dev);
`pytest-shard` added to requirements-test.txt; the shape contract extended
to pin the shard partition. The macOS 55 failures all pass on a local
macOS/3.12 run of the same files (249/249 on the largest cluster) -- they
are runner-environment-specific and are AC#5, not this phase: the point of
Phase 1 is that every future run COMPLETES and produces complete,
name-level failure data for Phase 2 triage instead of interleaved xdist
progress dots.


**Phase 1 verified (PR #1826, run 32268704382):** all 12 UI shards
COMPLETED in 18-28 minutes against the 45-minute budget -- the suite's
first complete UI run in 100+ attempts -- each uploading its named
failure report (13,558 collected / 13,432 passed / 119 failed / 57 files).
The complete inventory and its triage are filed as TASK-18609 (AC#4/#5).
Core legs run under the 120-minute budget. Status -> Done with the merge
of PR #1826.

<!-- SECTION:NOTES:END -->
