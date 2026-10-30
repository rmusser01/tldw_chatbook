---
id: TASK-37763
title: The response-rules Console fixture leaves a llama.cpp endpoint in the shared bootstrap config
status: To Do
assignee: []
created_date: 2026-10-30 03:57
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/UI/response_rules_fixtures.py
- Tests/UI/test_console_response_rules.py
- Tests/UI/test_settings_footer_hints.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36514's review (PR #3242, 2026-10-24). `mounted_rules_console` in `Tests/UI/response_rules_fixtures.py` sets up a ready provider by persisting llama.cpp at `127.0.0.1:9099` (and `agent_runtime` off) to the config file. Under `bootstrap_profile`, that file is shared by every later test in the same process, and the fixture never puts it back.

So whichever test runs next in that process inherits the endpoint. When `Tests/UI/test_console_response_rules.py::test_mapped_learning_phase_and_stop_follow_live_helper` runs directly before `Tests/UI/test_settings_footer_hints.py`, the Settings test's probe connects to the server and its teardown fails the egress guard: "test attempted network egress (blocked): socket.connect -> 127.0.0.1:9099". It reproduced 3 of 3 times. The footer-hints file alone passes, and with other predecessors it passes.

Any census re-deal that puts these two in that order fails a UI Fast Lane job.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a response-rules Console test ends, the shared config is as it was before the test
- [ ] #2 The response-rules node followed by test_settings_footer_hints.py passes in one process, serially, and the same pair fails without the fix
- [ ] #3 The response-rules and footer-hints files still pass on their own
<!-- AC:END -->
