---
id: TASK-31807
title: >-
  First-run wizard intermittently self-dismisses to Home with zero input when
  splash is enabled
status: Done
assignee: []
created_date: '2026-09-05 19:15'
updated_date: '2026-09-06 15:43'
labels:
  - bug
  - ui
  - wizard
  - flaky
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed twice on origin/dev during the 2026-09-05 release-gate work (by the TASK-31741 fix agent): with the splash screen enabled, the first-run wizard occasionally mounts and then self-dismisses to Home with no input, persisting setup_started. Intermittent; likely a race between splash teardown and the wizard's screen push. Needs a reproducer and fix; related surface: TASK-31226 (cancel routing) and TASK-31741 (exit-dialog settle guard).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Root cause identified with a deterministic reproducer or instrumented evidence.
- [x] #2 Wizard never dismisses without user input.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Build a splash-ON reproduction over the real app (`run_test`) with instrumentation on `_dismiss_navigation_overlays` and `post_message`.
2. Determine whether any automatic startup navigation fires (it does not) and isolate the actual trigger.
3. Fix at the navigation seam so a stray navigation can never tear down the onboarding gate.
4. Add a deterministic regression test that fails before and passes after.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Root cause (instrumented, deterministic).** The self-dismiss is a
navigation-driven teardown, not a stray dismiss inside the wizard. When any
`NavigateToScreen` is handled while the wizard is the top overlay,
`_handle_screen_navigation_locked` -> `_dismiss_navigation_overlays` calls
`overlay.dismiss(None)` on the wizard. That resolves the wizard's push callback
with `None`, which `_continue_first_run_wizard_result` treats as a cancel --
landing the user on the initial screen (Home) with zero input, while
`setup_started` was already persisted by the wizard's `on_mount`, so setup is
never re-offered cleanly.

Across 20 clean `run_test` boots the wizard never self-dismissed and NO
navigation was posted at all -- proving nothing in startup navigates on its
own. The trigger is external input: a shell-destination key (F9=settings,
F10=research, ctrl+N ..., all bound to `action_shell_destination`, which POSTs
`NavigateToScreen`) that leaks in during splash teardown. The splash's own
`on_key` consumes only the FIRST key (`not self._skip_requested`), so a second
key -- or a key pressed just after the splash is removed but before the
wizard's `call_after_refresh` push lands, when the app's global bindings are
live on the just-mounted initial screen -- reaches the app and navigates.
Directly posting `NavigateToScreen("settings")` while the wizard was up
reproduced the dismiss deterministically (final screen = HomeScreen); it is
also why the bug is splash-specific (the teardown window, plus the user
pressing keys to skip the splash).

**Fix.** `FirstRunSetupWizard` now sets a class attribute
`blocks_stray_navigation = True`, and `_handle_screen_navigation_locked`
ignores (silently, no "couldn't open" toast) any navigation that arrives while
a screen with that flag is on the stack -- mirroring the existing "initial
screen not yet mounted" early return. The wizard is only ever meant to be left
through its own Next/Back/Skip/Esc controls, which `dismiss` it directly and
post any follow-on navigation AFTER it is off the stack, so no legitimate
navigation is blocked (verified: 423 existing wizard tests + the live-contract
completion->navigation path still pass, and a dedicated test confirms
navigation resumes once the wizard leaves the stack).

**Files:** `tldw_chatbook/app.py` (guard in `_handle_screen_navigation_locked`),
`tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py` (`blocks_stray_navigation`
attribute), `Tests/UI/test_first_run_wizard_stray_navigation.py` (regression).
<!-- SECTION:NOTES:END -->
