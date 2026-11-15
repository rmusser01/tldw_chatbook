---
id: TASK-41136
title: "test_settings_custom_endpoints.py and test_console_provider_gateway.py run on no PR lane"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- tests
- lanes
- providers
priority: low
references:
- Tests/UI/test_settings_custom_endpoints.py
- Tests/Chat/test_console_provider_gateway.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 endpoint stream fu/r14-endpoint 6cb8e34461 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The gateway is where Console resolves providers and endpoints, and the custom-endpoint Settings file pins how users add and fix endpoints; TASK-40300 and TASK-40302 changed both behaviours. Neither file is guarded on PRs.

**Evidence.** Neither file is named in `.github/workflows/*.yml` or `scripts/ui_pr_gate_census.txt`, at the stack top or on the endpoint branch. TASK-40300's notes record three gateway reds owned by TASK-39151.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both files' green nodes run on a PR lane (the gateway in a non-UI job, the Settings file in the UI census), with measured times
- [ ] #2 Each red node is listed with its owner
<!-- AC:END -->
