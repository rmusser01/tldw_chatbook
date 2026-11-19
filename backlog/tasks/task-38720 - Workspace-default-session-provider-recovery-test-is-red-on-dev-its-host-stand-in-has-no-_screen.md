---
id: TASK-38720
title: "Workspace default-session provider-recovery test is red on dev: its host stand-in has no _screen"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-04 00:59
updated_date: 2026-11-05 07:19
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
- [x] #1 test_provider_setup_recovery_keeps_created_persona_prompt passes on dev and still checks that provider-setup recovery keeps the created persona prompt, through the same convergence path the product runs
- [x] #2 The whole file passes 3 of 3 runs with -n 4
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Give the test's host stand-in the `_screen` the method now reads, with no readiness projection active (the state of every convergence outside a display projection), so the real `ConsoleSessionController._maybe_refresh_stale_default_console_settings` runs end to end.
2. Show the test still guards the Persona: a temporary source edit that drops the Persona carry-over must turn it red.
3. Run the whole file 3 times at `-n 4`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Test drift; test-only fix.** The `SimpleNamespace` host now has the `_screen` that `_maybe_refresh_stale_default_console_settings` reads, with `_console_readiness_projection_active=None`. That is the state of every refresh outside a readiness display projection, and it is the state in which the method converges. The test still calls the real `ConsoleSessionController` method on a real store, so it runs the product's convergence path: follows-saved-defaults check, fresh defaults, Persona carry-over, `replace_session_settings`.

**Evidence.**
- Red on the base b8608e4cdb: `AttributeError: 'types.SimpleNamespace' object has no attribute '_screen'`.
- Negative control: a temporary source edit that skipped the Persona carry-over (`if False and session.assistant_kind == "persona"`), since reverted. The node went red on `assert after.system_prompt == before.system_prompt` (`None` against the Persona prompt), so the test still guards the Persona.
- The whole file at `-n 4`, load about 29: 26/26 passed in 3 of 3 runs.

**Files:** `Tests/Chat/test_workspace_default_session.py`.
<!-- SECTION:NOTES:END -->

