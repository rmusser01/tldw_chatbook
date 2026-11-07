---
id: TASK-39126
title: "The Architecture guards job's runtime-policy owner node parses all 9,001 repository Python files in one node, and its file takes up to 173 s on CI against a 180 s per-node timeout"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- ci
- tests
- performance
priority: low
references:
- Tests/Architecture/test_profile_owned_path_inventory.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-39054
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and in the notes of TASK-39054 and TASK-39060 (round-11 branch fu/r11-arch). Line numbers are dev's.

**The node.**
- The Architecture guards job runs `Tests/Architecture/test_profile_owned_path_inventory.py::test_compatibility_and_runtime_policy_constants_have_no_new_runtime_owners` with `--timeout=180` (`derived-artifacts.yml:1240`, `:1247`).
- The node (`test_profile_owned_path_inventory.py:312`) walks `REPO_ROOT.rglob("*.py")` and parses each file: 9,001 files, 214.5 MB of source.

**Measured times.** The job has no `--durations`, so CI gives only the time of the file's 14 nodes together:

| Run (2026-11-06) | The file's 14 nodes | Job total |
|---|---|---|
| 42310788537 | 172.7 s | 404.3 s |
| 42309351181 | 170.4 s | 400.2 s |
| 42306963606 | 92.0 s | 243.1 s |

- Locally, per TASK-39054's and TASK-39060's notes:
  - the node's call took 177.4 s at load about 79 on base, and 332.9 s alone at load 65-106 on the branch;
  - it timed out in a review's job run (load 84-91) and passed at load 24-57.
- A slower CI runner would fail the Architecture job on this node, not on a real regression.

**Not part of this.** The `test_persistent_diagnostic_inventory` task_15103 ledger nodes, also named in this follow-up, run on no PR lane. They read frozen revisions a shallow checkout cannot serve, and timed out only in local runs, so they do not threaten the job.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The node's own call time is measured on CI and locally, with load recorded
- [ ] #2 The node keeps at least 2x headroom under its per-node timeout across three CI job runs, recorded
- [ ] #3 Negative controls: a new `DEFAULT_RUNTIME_POLICY_PATH` assignment and a new base-data-dir CLI consumer each still turn it red
<!-- AC:END -->
