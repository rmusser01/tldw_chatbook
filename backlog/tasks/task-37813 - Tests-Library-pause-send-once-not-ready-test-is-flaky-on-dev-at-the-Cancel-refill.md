---
id: TASK-37813
title: 'Tests: the Library-pause ''send once, provider not ready'' test is flaky on dev at the Cancel refill'
status: To Do
assignee: []
created_date: '2026-10-29 18:10'
labels:
  - tests
  - flaky
  - console
dependencies: []
references:
  - >-
    backlog/tasks/task-33621.20 -
    Console-Automatic-Library-retrieval-outcome-is-never-shown-and-a-cold-start-timeout-jams-the-conversation.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_console_library_pause_recovery.py::test_send_once_with_a_provider_that_is_not_ready_never_jams_the_chat` comes from TASK-33621.20. It failed 3 of 4 isolated runs on dev 38c9d2edcf on a loaded host (2026-10-29, load around 8-12). Each failing run timed out at `await until(lambda: composer.draft_text() == FIRST)` after the card's Cancel send on the re-paused (DESTINATION_CHANGED) send; the passing runs took about 9 s and the failing ones about 23 s.

The TASK-33621.17/.18 branch fails it at the same rate (2 of 4). That branch does not change the shelf-releasable Cancel path.

The test is not in the UI PR gate census, but it reads as a regression in any head-versus-dev comparison of that file. Find whether the refill is lost (a product race between the Cancel's draft write-back and the Console sync) or only slow (the wait is too short), and fix whichever it is.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the missing or late refill is identified and recorded on this task
- [ ] #2 The test passes 20 of 20 isolated runs on a loaded host
<!-- AC:END -->
