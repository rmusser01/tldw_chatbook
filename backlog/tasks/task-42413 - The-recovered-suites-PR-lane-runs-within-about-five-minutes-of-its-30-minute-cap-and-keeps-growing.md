---
id: TASK-42413
title: "The recovered-suites PR lane runs within about five minutes of its 30-minute cap and keeps growing"
status: To Do
assignee: []
created_date: 2026-11-22 21:17
updated_date: 2026-11-22 21:17
labels:
- follow-up
- ci
- tests
priority: medium
references:
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
- backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22). Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths start at the repository root.

**Why.** "PR lane (recovered suites)" (`derived-artifacts.yml:1696`) is one serial job under `--storage-lease-guard`, in the required aggregate's `needs`. Its cap went from 20 to 30 minutes on 2026-11-19 (owner decision, the note at `:1713-1716`) after 5 of 24 runs hit the old cap. Every round since has added targets to it, and its time keeps rising. If a run hits the cap, the required check goes red for a reason unrelated to the PR.

**Measured.** Job durations (start to completion) of non-cancelled PR runs of the job, from the Actions API:

| Day (UTC) | Jobs | Median (min) | Max (min) |
|---|---|---|---|
| 2026-11-20 | 40 | 20.0 | 25.7 |
| 2026-11-21 | 34 | 20.9 | 24.6 |
| 2026-11-22 | 41 | 21.4 | 25.1 |

Overall: 115 jobs, median 21.0, 90th percentile 23.4, max 25.7. TASK-33620.15.61's port (round 21) adds about 22 s in CI, which puts the projected worst at about 25.5 minutes of 30. TASK-41516's two files are already in it (included in these numbers).

The job's own structure suggests a split: one invocation runs the sandboxed suites first, then the `bootstrap_profile` ones, which must not be followed by a sandboxed suite that reads config in-process (TASK-32873). Each block could be its own job. Each new job is one more runner per PR (ADR-103's notes), so the split is an owner decision; moving targets to lanes with room is the alternative.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The recovered suites' projected worst case is at least 5 minutes under the cap of every job they run in, measured over at least 20 non-cancelled PR runs after the change (or projected from per-target serial times when fewer runs exist), with the numbers in the notes
- [ ] #2 Every target that ran in the job before the change still runs in a required PR lane, with the same flags (`--storage-lease-guard`, the sandboxed-then-bootstrap order) wherever it runs
- [ ] #3 `RECOVERED_SUITES_TARGETS` and the other lane contracts in `Tests/CI/test_ci_queue_pressure_contract.py` match the workflow, and `Tests/CI` passes
- [ ] #4 The workflow note for the recovered suites states its current headroom, so the next addition can be checked against it
<!-- AC:END -->
