---
id: TASK-41764
title: "A media return settlement whose deadline has passed arms a zero-second timer, and Textual 8.2.8 divides by its interval"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- library
- ui
priority: medium
references:
- tldw_chatbook/UI/Library_Modules/library_media_controller.py
- tldw_chatbook/Utils/ui_responsiveness.py
- Tests/UI/test_library_media_return_settlement.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`LibraryMediaController._bind_library_media_settlement_deadline` (`library_media_controller.py:1688` at dev 908e9840aa) arms `self.set_timer(max(0.0, deadline - time.monotonic()), ...)`. Once the outer deadline has passed, that is a 0-second timer.

In Textual 8.2.8, `set_timer` builds a `Timer` with `skip=True`. Whenever the next tick is already late, `Timer._run` (`textual/timer.py:159-164`) computes `int((now - start) / _interval + 1)`. With a zero interval the tick is late as soon as the clock moves, so the timer task raises `ZeroDivisionError` instead of running the expiry callback. `stop_timer` then re-raises it when the timer is stopped (`textual/timer.py:120`).

**History and evidence.**
- **First seen** on 2026-11-17, in the media series' private-profile marker A/B (1,493 ids, chain over dev 5e3c8b2bd4). It was a teardown failure of `Tests/UI/test_library_media_return_settlement.py::test_authoritative_recompose_rearms_before_replacement_geometry`.
- **Not caused by the series.** The same code sits in the controller at 5e3c8b2bd4 (`:1493`) and at 08b3c9d337, before the series (`:1492`).
- **Reproduced** on 2026-11-18 at dev 908e9840aa under the marker at `-n 3`. One of two rounds failed with "ZeroDivisionError: float division by zero" from `timer.py:120`. The other round failed earlier, on the test's known geometry wait.

**Effects.** The settlement never expires through its deadline, and the error escapes when the screen's timers are stopped.

**Same shape elsewhere.** `Utils/ui_responsiveness.py:825` arms `owner.set_timer(max(0.0, delay), ...)`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A settlement whose deadline has already passed expires, and its expiry runs, without arming a zero-second timer
- [ ] #2 A test that binds the deadline after it has passed is red before the fix and green after, and shows the expiry runs once
- [ ] #3 Utils/ui_responsiveness.py:825 and every other set_timer call in tldw_chatbook whose delay can reach 0 are fixed the same way or recorded as safe
<!-- AC:END -->
