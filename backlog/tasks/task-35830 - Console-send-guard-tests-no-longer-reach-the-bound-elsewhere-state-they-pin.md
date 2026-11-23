---
id: TASK-35830
title: Console send-guard tests no longer reach the bound-elsewhere state they pin
status: To Do
assignee: []
created_date: '2026-10-20 05:18'
updated_date: '2026-11-23 07:09'
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

2026-11-23 (round-22 filing): all five still fail at their setup assertion on the round-22 stack (142e821116, before its rebase to 336d15338b; dev 2f7790873a plus round 22's stacked branches), at `-n 4`: the four `test_send_refuses_and_keeps_the_draft_when_the_composer_is_bound_elsewhere[enter|send_button|spoken|redirect]` cases at line 490 (`{'composer': 'a-own:'} != {'composer': 'Reply with the single word ECHO.'}` and `{'draft owner': False} != {'draft owner': True}`), and `test_send_refuses_when_the_composer_chat_is_off_screen_after_two_switches` at line 724 (the composer is bound to a different session id than the test expects). TASK-42414's covering runs saw the same at plain dev. The file's four other local reds are test-double drift, not this setup, and are TASK-42467.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each send path the five tests cover (Enter, Send button, spoken send, Redirect, off-screen after two switches) has a mounted test that reaches the send guard's refusal and passes under the plain pytest command
- [ ] #2 Each repaired test turns red when the send guard's refusal is removed (mutation-checked), and green again when restored
- [ ] #3 No test in the file fails at a setup assertion; any path whose bound-elsewhere state can no longer occur is retired with the reason recorded in the task notes
<!-- AC:END -->
