---
id: TASK-18608
title: >-
  Tests workflow is red on all branches: UI job times out and the core job
  carries 55 pre-existing failures
status: To Do
assignee: []
created_date: '2026-08-19 07:45'
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
- [ ] #1 The Tests workflow completes within its job budget on a PR to `dev` (shard, split, deselect, or raise the budget -- owner call).
- [ ] #2 The 55 core-suite failures are triaged: each fixed, marked, or quarantined with a filed follow-up.
- [ ] #3 A green `Tests` run exists on `dev` and the workflow is back to being a merge signal.
<!-- AC:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
Evidence: run 32256433043 (PR #1824 @ a9b4b7e26): UI job cancelled at 45m
at 11%; macOS core job exit 3 with the 55 failures listed in its
`core-test-results` artifact; `gh run list --workflow Tests` shows 30/30
`cancelled`. PR #1824 was merged on the strength of its own suites
(1,903 tests green on the rebased head) after confirming the CI red was
pre-existing on clean dev.
<!-- SECTION:NOTES:END -->
