---
id: TASK-42136
title: "Four unbounded entered.wait() calls in test_briefing_cast.py turn an early failure into a 300 s timeout"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:06
labels:
- follow-up
- tests
- subscriptions
priority: low
references:
- Tests/Subscriptions/test_briefing_cast.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the docs stream (TASK-42027, round 18). Four tests in `Tests/Subscriptions/test_briefing_cast.py` start a generation whose `_slow_chat` sets `entered` and then waits on `release`, and then `await entered.wait()` with no bound (lines 865, 929, 1072 and 1472). If the generation fails before it calls the chat function, `entered` is never set, and the test waits for pytest-timeout's 300 s (`timeout = 300` in `pyproject.toml`) instead of failing with the generation's error.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the four waits is bounded, and when the generation task ends before `entered` is set the test fails at once with that task's error
- [ ] #2 A negative control (a temporary raise before the chat call) makes each test fail within seconds with the raised error, not at the 300 s timeout; the notes record it
- [ ] #3 The file's tests pass
<!-- AC:END -->
