---
id: TASK-966
title: Fix the six failing chat API key tests on dev
status: To Do
assignee: []
created_date: '2026-07-27 18:06'
labels:
  - ui
  - tests
  - dev-baseline
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_tools_settings_window.py's six test_chat_api_key_* tests fail on pristine origin/dev with KeyError: 'openai'. They have been the standing baseline noise for every PR in the path-naming audit series and were verified pre-existing by stash bisection and by running the file on a pristine dev worktree. Standing baseline failures are corrosive: they train reviewers to skim red output, which is exactly how a real regression slips through.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The six tests pass on a clean checkout,The KeyError root cause is fixed rather than the assertion relaxed,The file has no remaining expected failures
<!-- AC:END -->
