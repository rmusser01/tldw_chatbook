---
id: TASK-38761
title: "Console native chat flow suite runs on no PR lane"
status: To Do
assignee: []
created_date: 2026-11-05 07:11
labels:
- follow-up
- ci
- tests
- console
- needs-owner-decision
priority: medium
references:
- Tests/UI/test_console_native_chat_flow.py
- scripts/ui_pr_gate_census.txt
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-38716
- TASK-38721
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

`Tests/UI/test_console_native_chat_flow.py` is the behavioural suite for the Console send path: 15,552 lines, 299 test functions, 371 collected nodes. None of them runs on a PR lane. The file has no entry in `scripts/ui_pr_gate_census.txt` and is not a target in any workflow. Only `test.yml` (on a push to main, or on dispatch) and the scheduled `nightly-deep.yml` collect it, with the whole `Tests/` tree.

Two tasks cover parts of it:
- TASK-38721: the 31 return-path nodes (the `test_conversation_settings_return_*` functions and `test_dirty_return_confirmation_survives_real_router_navigation_away_and_back`);
- TASK-35572: two transcript-restore nodes.

No task puts the other 338 nodes on a lane. Their state is unclear:
- TASK-22557 (To Do) tracked the file's reds rising from 2 to 9.
- TASK-38696's notes (PR #3404) saw three nodes (`more_delete_double_activation`, `original_attempt_delete`, `rejected_regenerate`) fail in 5 of 9 node runs at base.
- Locally most nodes fail with `RecoveryRequired: raw_source_selection_changed` (TASK-38111; see TASK-38721's notes), so a local run cannot show which nodes are healthy. Only the lane's own configuration can.

The UI lane and the fast lanes run near their caps (TASK-38716), so where 338 mounted-Console nodes run is a question of runner capacity, which is the owner's call.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision on where the file runs (UI census shards, a new job, or a subset with the rest retired) is recorded in the notes
- [ ] #2 Every node outside TASK-38721's and TASK-35572's sets runs on a PR lane, or is retired with the reason recorded
- [ ] #3 Each node passes 3 of 3 runs in its lane's configuration before it is added; a node that is red there for a reason other than RecoveryRequired is fixed or filed as its own task
- [ ] #4 Each job the nodes join still ends at least 5 minutes under its cap in its slowest measured run, and the measured times are recorded in the notes
<!-- AC:END -->
