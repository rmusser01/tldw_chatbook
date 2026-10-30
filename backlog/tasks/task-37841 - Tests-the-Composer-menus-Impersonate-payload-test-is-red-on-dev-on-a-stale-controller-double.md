---
id: TASK-37841
title: >-
  Tests: the Composer menu's Impersonate payload test is red on dev on a stale
  controller double
status: To Do
assignee: []
created_date: '2026-10-30 01:31'
labels:
  - tests
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_composer_menu.py::test_impersonate_payload_obeys_the_provider_contract fails on origin/dev (ed7481a30c) under the plain command with AttributeError: '_maintenance_calls'. Its controller is built with __new__ and lacks an attribute the impersonate path now reads, so the test no longer checks the provider contract it is named for. TASK-35160 and TASK-32084 each recorded it as a known pre-existing red and moved on; TASK-15210 repaired an earlier layer of the same double. Seen again in the review of TASK-37811, where it was the only red in that file on both dev and the branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes under the plain command on dev and still fails if the impersonate payload breaks the provider contract (checked by a mutation)
- [ ] #2 The double is built from the real controller seams the impersonate path reads, so the next new attribute fails at a named assertion rather than an AttributeError
<!-- AC:END -->
