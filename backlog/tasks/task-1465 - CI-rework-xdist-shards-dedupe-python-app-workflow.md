---
id: TASK-1465
title: >-
  CI rework: parallel directory shards replace the 27-file -m unit matrix; dedupe python-app.yml (owner sign-off)
status: Done
assignee: []
created_date: '2026-07-30 08:55'
labels:
  - testing
  - ci
priority: high
dependencies: [task-1453]
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CI's `-m unit` job matrix (8 OS/Python runs) selects 27 of 900 test files while installing torch/chromadb/playwright each time; `-m integration` selects 40; ~590 files are exercised by no PR-triggered job in test.yml — only by `python-app.yml`'s duplicate, serial, unbounded `pytest ./Tests/` on main. No CI job uses parallelism. Restructure onto xdist directory shards with a nightly deep job.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [x] PR gate: parallel (`-n auto --dist loadscope`) jobs covering ALL of Tests/ (e.g. core = Tests minus UI, ui = Tests/UI); marker-based unit/integration jobs removed or reduced to a deliberate, documented subset
- [x] `python-app.yml` deleted or collapsed after confirming branch-protection required-check wiring with the owner
- [x] Nightly/dispatch job: serial full run + `HYPOTHESIS_PROFILE=thorough` + `--run-slow` + coverage; OS/Python matrix breadth moved here per owner decision
- [x] PR-gate wall time before/after recorded

## Implementation Plan

1. Read the CI-shape assertion test FIRST (it pins job names, block anchors, the summary needs-list) and update it in lockstep
2. Owner rulings via decision table: lean two-shard PR gate / delete python-app.yml (no branch protection exists on dev or main — verified) / nightly deep job / nightly-only coverage
3. Rework test.yml; delete python-app.yml; extend the shape test to protect the new wiring

## Implementation Notes

PR gate: `core-tests` (Tests minus UI, `-n auto --dist loadscope`, ubuntu+macos
on 3.12) + `ui-tests` (Tests/UI, same parallelism) replace the `-m unit`
8-job matrix and the `-m integration` job — every PR now runs the whole tree
instead of 67 files. Coverage moved off the PR gate. `python-app.yml`
(duplicate serial full run on main) deleted; neither branch has protection
referencing it. New `nightly-deep` (cron 08:30 UTC + dispatch, checks out dev
explicitly since schedules fire from the default branch): SERIAL (order-
regression canary), `TLDW_HYPOTHESIS_PROFILE=thorough`, `--run-slow` (those 25
tests' first-ever home), `TLDW_TEST_CSS_CACHE=0` (task-1459's soak), coverage,
and the OS/Python breadth dropped from the gate. The CI-shape test updated in
lockstep (anchors, needs-list) and EXTENDED with two assertions pinning the
new wiring; 11 passed. Known consequence, by design: the honest gate runs red
while dev's documented pre-existing rot remains — those tests previously ran
in no PR job at all.
Modified: `.github/workflows/test.yml`, `Tests/CI/test_github_actions_test_workflow.py`.
Deleted: `.github/workflows/python-app.yml`.
