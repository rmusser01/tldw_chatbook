---
id: TASK-42018
title: "Unseen Console receipts leak between bootstrap-profile tests"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/conftest.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review r17cn-rev3 (round 17). Every `bootstrap_profile` test in one process shares one ChaChaNotes database (`Tests/conftest.py` ~62, ~1906-1908), and TASK-36478 restores only config.toml between them. A mounted send that closes before its reply renders leaves an unseen receipt, so the next app announces "A Console turn completed while hidden". TASK-41900/41904 fixed their own tests locally.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Durable Console unseen receipt marks are cleared between bootstrap-profile tests, as config.toml is
- [ ] #2 A test shows a planted unseen receipt does not reach the next test
<!-- AC:END -->
