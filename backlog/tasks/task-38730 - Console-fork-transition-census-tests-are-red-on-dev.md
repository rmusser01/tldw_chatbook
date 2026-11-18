---
id: TASK-38730
title: "Console fork-transition census tests are red on dev"
status: To Do
created_date: 2026-11-04 02:52
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/Chat/test_console_fork_transition_census.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev d204ddce70 (2026-11-03), two tests in Tests/Chat/test_console_fork_transition_census.py fail when the file is run alone: `test_public_fork_transition_inventory_is_bidirectional` and `test_every_public_direct_fork_field_assignment_is_fenced_or_classified`. They were also red on dev 335f984bd2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both tests pass on dev, with the inventory updated to the real fork transitions rather than loosened
- [ ] #2 The task notes say which change left the census stale
- [ ] #3 The file runs on a PR lane, or the task notes say why not
<!-- AC:END -->
