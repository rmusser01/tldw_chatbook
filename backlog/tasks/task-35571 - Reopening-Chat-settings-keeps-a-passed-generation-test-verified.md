---
id: TASK-35571
title: Reopening Chat settings keeps a passed generation test verified
status: To Do
created_date: 2026-10-19 00:30
dependencies: []
labels:
- model-config-redesign
- follow-up
- console
priority: medium
references:
- tldw_chatbook/Widgets/Console/console_settings_modal.py
- Tests/UI/test_console_session_settings.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in Phase 9's whole-PR review (2026-10-19). It predates Phase 9: the phase base 111f8e48fc behaves the same way.

The user runs a generation test in Chat settings and it passes, so the readiness line reads 'Ready · verified HH:MM'. Then they close Chat settings and open it again without changing anything. The second open reports 'Changed since test. Prior generation evidence no longer applies.' and the line drops to 'Ready · not tested'. The user is told their settings changed when they did not, and must run the test again to get the 'verified' word back.

The cause: `_generation_number_changed` returns early only while `_restoring_suspended_draft` is set. It still handles the compose-time `Input.Changed` echoes, which arrive while `_updating_controls` is set. Every open therefore cancels the test result and marks it stale. A probe showed that also returning early under `_updating_controls` keeps 'verified' on the second open. `_generation_choice_changed` needs the same check.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a passed generation test, closing and reopening Chat settings with no edit keeps 'Ready · verified HH:MM' and shows no 'Changed since test' status
- [ ] #2 A real edit to a generation value after the test still marks the result stale, as before
- [ ] #3 A mounted test runs a test, closes, reopens and asserts both, and it fails before the fix
<!-- AC:END -->
