---
id: TASK-38720
title: "Workspace default-session provider-recovery test is red on dev: its host stand-in has no _screen"
status: To Do
assignee: []
created_date: 2026-11-04 00:59
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/Chat/test_workspace_default_session.py
- tldw_chatbook/UI/Console_Modules/session.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down. Checked on dev c2cdb6b82e, 2026-11-03: red in each of two `-n 4` runs through the scratch-profile wrapper. The file's other tests pass.

`Tests/Chat/test_workspace_default_session.py::test_provider_setup_recovery_keeps_created_persona_prompt` (:276) drives `_maybe_refresh_stale_default_console_settings` (`UI/Console_Modules/session.py:4678`) with a `SimpleNamespace` host (:291-297). The method now reads `getattr(self._screen, "_console_readiness_projection_active", None)` (:4699), a line 3e618d7119 added on 2026-10-15. The stand-in has no `_screen`, so the test fails with `AttributeError: 'types.SimpleNamespace' object has no attribute '_screen'` before it reaches the persona-prompt check.

No PR lane runs this file (no census entry, no workflow target), so the red went unseen. No open task names it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_provider_setup_recovery_keeps_created_persona_prompt passes on dev and still checks that provider-setup recovery keeps the created persona prompt, through the same convergence path the product runs
- [ ] #2 The whole file passes 3 of 3 runs with -n 4
<!-- AC:END -->
