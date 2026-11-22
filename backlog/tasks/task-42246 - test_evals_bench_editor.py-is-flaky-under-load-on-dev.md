---
id: TASK-42246
title: "test_evals_bench_editor.py is flaky under load on dev"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- evals
priority: low
references:
- Tests/UI/test_evals_bench_editor.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the recompose stream (round 19). `Tests/UI/test_evals_bench_editor.py` (the stream's list gave `Tests/Evals/`) fails intermittently under machine load on dev. Not reproduced here; no node or failure message was reported, so the first step is to find which nodes flake.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The flaking nodes are named, with the failure each shows under load and how it was reproduced (for example a loaded `-n 4` run repeated)
- [ ] #2 The cause of each red node is named and fixed
- [ ] #3 The file passes five repeated runs at `-n 4` under load on a branch carrying the fix, with the load recorded
<!-- AC:END -->
