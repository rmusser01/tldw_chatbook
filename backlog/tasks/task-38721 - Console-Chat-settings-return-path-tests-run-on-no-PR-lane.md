---
id: TASK-38721
title: "Console Chat-settings return-path tests run on no PR lane"
status: To Do
assignee: []
created_date: 2026-11-04 00:59
labels:
- follow-up
- tests
- console
- ci
priority: medium
references:
- Tests/UI/test_console_native_chat_flow.py
- scripts/ui_pr_gate_census.txt
dependencies:
- TASK-38111
- TASK-33008.1
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down. Checked on dev c2cdb6b82e, 2026-11-03.

`Tests/UI/test_console_native_chat_flow.py` holds the tests for the Console's return from Settings after provider recovery (TASK-33008.1). There are 24 test functions and 31 nodes:
- 23 `test_conversation_settings_return_*` functions (:350-:2482);
- `test_dirty_return_confirmation_survives_real_router_navigation_away_and_back` (:2204).

None of them runs on a PR lane. The file has no entry in `scripts/ui_pr_gate_census.txt` and is not a target in any workflow. Only `test.yml` (on a push to main or on dispatch) and the scheduled `nightly-deep.yml` collect it, with the whole `Tests/` tree.

Locally, at the base, two `-n 4` runs through the scratch-profile wrapper failed 28 and 27 of the 31 nodes. Every failure was `RecoveryRequired: raw_source_selection_changed` (`Backup_Recovery/raw_participants.py:307`), the plain-test trap that TASK-38111 owns. Which nodes passed varied between runs. So this task cannot tell from a local run whether the nodes are healthy; the lane's own configuration has to show it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All 31 return-path nodes run on a PR lane (census entries or a workflow target), with their measured time in that lane recorded in the task notes
- [ ] #2 They pass 3 of 3 runs in that lane's configuration before they are added; any node that is red there for a reason other than RecoveryRequired is fixed, or retired with the reason recorded
<!-- AC:END -->
