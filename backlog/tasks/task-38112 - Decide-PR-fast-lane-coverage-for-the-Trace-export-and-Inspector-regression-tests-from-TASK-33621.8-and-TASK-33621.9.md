---
id: TASK-38112
title: >-
  Decide PR fast-lane coverage for the Trace export and Inspector regression
  tests from TASK-33621.8 and TASK-33621.9
status: To Do
assignee: []
created_date: '2026-10-30 05:56'
labels:
  - tests
  - ci
  - owner-decision
  - follow-up
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner decision, raised by the TASK-33621.8/.9 review. The regression tests those tasks added are not PR fast-lane riders: only one test in Tests/UI/test_console_conversation_inspector.py is listed in scripts/ui_pr_gate_census.txt, and the new Chat tests (staged-evidence export, native-reader usage, stored-orphan export) are not in the PR gate either. On a PR they run only nightly or on pushes to main, so a change that brings back the dangling-link export failure or the unloadable Inspector call could merge before anything goes red.

Adding files to the PR gate is an owner call (lane time budget; slow files must stay out). This task records the decision and applies it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner has decided which of the TASK-33621.8 and TASK-33621.9 regression tests run on PRs, and the decision is recorded in this task
- [ ] #2 Any test added to the PR gate runs under 30 seconds per file and passes under the plain command
- [ ] #3 scripts/check_ui_pr_gate_census.py passes after the change
<!-- AC:END -->
