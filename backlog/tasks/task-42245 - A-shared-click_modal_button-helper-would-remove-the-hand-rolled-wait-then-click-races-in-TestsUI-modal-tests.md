---
id: TASK-42245
title: "A shared click_modal_button helper would remove the hand-rolled wait-then-click races in Tests/UI modal tests"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- ui
priority: low
references:
- Tests/UI/test_change_review_screen.py
- backlog/tasks/task-41505 - Single-pilot.pause-after-a-press-races-the-handler-under-load-in-several-Library,-Media-and-navigation-tests.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the docs2 stream (round 19), which fixed one click-before-layout race in `test_mounted_partial_and_provider_failures_name_problem_and_allow_retry`. Several `Tests/UI` modal tests hand-roll "wait for the modal, then `pilot.click(<button id>)`", and the click can land before the modal has laid out (for example `Tests/UI/test_change_review_screen.py:589` clicks `#change-revert-yes` right after the modal is found). The round-19 testsb branch adds a local `click_modal_button(pilot, selector)` next to its change-review tests; no shared one exists in `Tests/UI`. TASK-41505 covers the single-`pilot.pause()` race in other files, not this wait.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One shared `click_modal_button` helper in a `Tests/UI` support module waits until the button is laid out (non-zero region) before clicking
- [ ] #2 The hand-rolled modal clicks in `Tests/UI` that follow the wait-then-click pattern use it, or the ones left are listed with a reason
- [ ] #3 A test pins the helper: it fails if the click is sent before layout
<!-- AC:END -->
