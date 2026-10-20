---
id: TASK-35830
title: Console send-guard tests no longer reach the bound-elsewhere state they pin
status: To Do
assignee: []
created_date: '2026-10-20 05:18'
updated_date: '2026-10-20 05:19'
labels:
  - console
  - tabs-sessions
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Five mounted tests in Tests/UI/test_console_tab_switch_binds_composer.py, added by TASK-33622.7 (#3065), exist to prove that a send (Enter, the Send button, the spoken send, Redirect) refuses and keeps the draft when the composer is still bound to a chat other than the one on screen, including after two quick switches. On dev (seen at 4de437a5c5, ce4aa4a231 and 8136af6f6f) all five fail at their setup assertion: the state they build (store moved to A during a held sync pass, A highlighted and its transcript shown, composer still holding B's draft) no longer comes out of that setup, because the composer is now bound at once. They go red before the refusal is reached, so the send/Redirect guard is pinned by one passing mounted test (test_redirect_refuses_while_the_store_has_moved_before_anything_is_painted). The guard needs real coverage again, built from a setup that still produces the bound-elsewhere state, or a recorded reason that the state can no longer occur on a path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each send path the five tests cover (Enter, Send button, spoken send, Redirect, off-screen after two switches) has a mounted test that reaches the send guard's refusal and passes under the plain pytest command
- [ ] #2 Each repaired test turns red when the send guard's refusal is removed (mutation-checked), and green again when restored
- [ ] #3 No test in the file fails at a setup assertion; any path whose bound-elsewhere state can no longer occur is retired with the reason recorded in the task notes
<!-- AC:END -->
