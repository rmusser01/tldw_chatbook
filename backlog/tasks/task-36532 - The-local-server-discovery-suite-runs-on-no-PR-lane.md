---
id: TASK-36532
title: The local server discovery suite runs on no PR lane
status: To Do
created_date: 2026-10-24 00:44
labels:
- follow-up
- tests
- ci
priority: low
references:
- Tests/Chat/test_local_server_discovery.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-36381
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36381's review (branch fu/h-settingsreds, 2026-10-23), checked on dev 5b5920b8e2. No PR lane runs `Tests/Chat/test_local_server_discovery.py` (59 tests):
- the UI census takes only Tests/UI;
- `derived-artifacts.yml` has a lane with a Tests/Chat target list, and this file is not on it.

TASK-36381 fixes the file's one red node (`test_probe_handles_recursive_json_as_bounded_failure`). It leaves the lane, because that burn-down could not edit the workflow. Draft PR #3036 also changes `derived-artifacts.yml`, so a PR for this task may be evicted from the merge queue for a workflow-file rebase.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tests/Chat/test_local_server_discovery.py runs on a PR-required lane, with a matching Tests/CI lane-contract pin where that lane has one
- [ ] #2 The lane's measured job time stays within its timeout, with the added time recorded in the task notes
<!-- AC:END -->
