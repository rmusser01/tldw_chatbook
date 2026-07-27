---
id: TASK-965
title: Fix the 33 failing Skills tests on dev
status: To Do
assignee: []
created_date: '2026-07-27 18:06'
labels:
  - skills
  - tests
  - dev-baseline
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Skills/ reports 33 failures on pristine origin/dev (33 failed / 342 passed). Three root causes were identified while triaging: current_runtime_backend is a read-only property while a test helper tries to set it; provider_model_resolution.py raises persisted_defaults must be a mapping; and one test does not pre-create a config parent directory. These masked signal repeatedly during the path-naming audit -- every branch touching Skills had to be separately baselined against pristine dev to prove its own failures were not among them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tests/Skills passes on a clean checkout,Each of the three root causes is fixed rather than worked around in the test,No test is relaxed merely to make it pass
<!-- AC:END -->
