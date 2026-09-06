---
id: TASK-31824
title: Scope clear_server to a single server under TLDW_CONFIG_PATH profile isolation
status: To Do
assignee: []
created_date: '2026-09-06 08:08'
labels:
  - config
  - reliability
  - runtime-policy
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from schedules close-out burndown (PR #2454) -- Qodo review + final whole-branch review both flagged it. In TLDW_CONFIG_PATH scoped mode, RuntimeServerContextProvider.clear_server (runtime_policy/server_context.py:~455) filters on server_profile_id, so signing out of ONE server clears credentials for ALL servers in that profile. Fail-safe (over-clears, never under-clears) and only reachable in the rare multi-server-per-scratch-profile case; default single-server installs are unaffected -- which is why it did not block the burndown merge (final review rated INFO, Qodo rated High). Also in-area (optional): 3 Qodo Medium docstring-completeness nits from the same PR -- resolve_tldw_api_auth_token (config.py:~1406), the per-profile helper's returns: (server_context.py:~46), and the scoped credential methods (server_credentials.py:~45).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 clear_server in scoped mode clears only the target server's credentials, not sibling servers in the same profile
- [ ] #2 A test pins the scoped-clear behavior: in a multi-server profile, clearing server A leaves server B's credentials intact
- [ ] #3 Default single-profile / single-server users are unaffected (no behavior change)
<!-- AC:END -->
