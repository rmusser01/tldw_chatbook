---
id: TASK-14876
title: >-
  CI: xdist INTERNALERROR serializing websockets ServerConnection fails the Core
  Tests job
status: To Do
assignee: []
created_date: '2026-08-09 21:50'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 'Core Tests (all but UI)' CI job intermittently reports 22 failed / 12163 passed with an execnet INTERNALERROR: "DumpError: can't serialize <class 'websockets.asyncio.server.ServerConnection'>", attributed to Tests/LLM_Calls/test_openai_realtime_session.py::test_connect_sends_session_update_and_fires_ready on worker gw0. Observed on PR #1467, whose diff touches ONLY backlog/*.md (zero Python files), so the failures cannot originate from the branch. The same test file passes locally: 35 passed in 4.70s. Under pytest-xdist a test that leaves a live websockets ServerConnection reachable from something xdist tries to serialize (e.g. an assertion-rewritten repr or a failure payload) crashes the worker protocol, which can cascade into unrelated reported failures and mask real regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Root cause identified: what xdist is attempting to serialize and why the ServerConnection is reachable from it
- [ ] #2 The realtime-session test cleans up (or avoids exposing) its ServerConnection so xdist can serialize any failure payload
- [ ] #3 Core Tests job passes on an unmodified dev checkout across three consecutive runs
- [ ] #4 If the 22 failures have a cause distinct from the INTERNALERROR, they are enumerated and filed separately
<!-- AC:END -->
