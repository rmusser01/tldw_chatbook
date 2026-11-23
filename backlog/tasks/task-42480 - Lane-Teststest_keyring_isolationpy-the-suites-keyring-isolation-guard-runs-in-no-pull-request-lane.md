---
id: TASK-42480
title: "Lane Tests/test_keyring_isolation.py: the suite's keyring-isolation guard runs in no pull-request lane"
status: To Do
assignee: []
created_date: 2026-11-23 11:13
updated_date: 2026-11-23 11:13
labels:
- follow-up
- ci
- tests
priority: low
references:
- Tests/test_keyring_isolation.py
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42477, whose guard tests had to go into `Tests/test_network_guard.py` because their natural home is not laned. Checked on `fu/r22-keychain` (62531d037a plus TASK-42477).

**Why.** `Tests/test_keyring_isolation.py` pins that the suite never reaches the developer's real OS credential store (TASK-19570 A). It checks three things: the bootstrap's null `keyring` backend, the production credential seam refusing it, and a constructed app recording the store as unavailable. No step in `.github/workflows/derived-artifacts.yml` names the file. It runs in no PR lane: only in `test.yml` (its `all-tests` and `core-tests` jobs; manual `workflow_dispatch`, or called from release qualification and publishing) and the scheduled `nightly-deep.yml` pass, which is no merge gate and usually reaches its 240-minute ceiling before root-level files. A pull request that removed the bootstrap's backend pin would merge with every lane green.

The natural lane is the "Run Architecture guard contracts" step beside `Tests/test_network_guard.py`; `Tests/CI/test_ci_queue_pressure_contract.py` pins that step's target list. Locally (2026-11-23, through the scratch wrapper), `test_a_constructed_app_records_the_store_as_unavailable` fails with the known local trap `RecoveryRequired: raw_source_selection_changed`. The other two pass. Check the app-building case in the lane before relying on it.

Coordinate with the other session's lanes work (TASK-42281), which edits `derived-artifacts.yml` and that contract file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `Tests/test_keyring_isolation.py` runs in a pull-request lane, and the CI contract that lists the lane's targets names it
- [ ] #2 All three of its tests pass in that lane, including the constructed-app case
- [ ] #3 Removing the conftest's null-backend pin turns that lane red (shown once under a temporary edit)
<!-- AC:END -->
