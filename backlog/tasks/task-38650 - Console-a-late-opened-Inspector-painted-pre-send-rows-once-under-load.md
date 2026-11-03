---
id: TASK-38650
title: Console a late-opened Inspector painted pre-send rows once under load
status: To Do
created_date: 2026-11-03 05:51
labels:
- follow-up
- console
- tests
priority: low
references:
- Tests/UI/test_console_collapsed_inspector_send_layout.py
- tldw_chatbook/Widgets/Console/console_run_inspector.py
dependencies:
- TASK-37000
- TASK-38644
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-38644 implementer (2026-11-03).

`Tests/UI/test_console_collapsed_inspector_send_layout.py::test_an_inspector_opened_after_a_send_matches_one_open_throughout` pins every frame the Inspector paints when it is opened after a first send (TASK-37000). It failed once at `assert set(frames) == {((False,), ("Run history",))}` (:357 on dev 18482c1cf7). The late-opened rail's first two painted frames showed the rows from before the send, so the rebuild landed after the first paint. That is the TASK-35970 failure the assert exists to catch.

Counts so far: 1 failure in 8 whole-file runs at load about 50 (on the TASK-38644 branch), then none in 24 more runs there and none in 48 runs on dev. The TASK-38644 change runs only after the frames are captured, so it is unlikely to be the cause.

If the race is real, a user who opens the Inspector right after a first send under a busy event loop may see one frame of stale rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The failure is reproduced on dev, under load or with a temporary source edit that delays the rebuild, or the notes record the run counts that did not reproduce it and the ordering the code guarantees
- [ ] #2 If it is reproduced: the cause is named, and the fix makes the rebuild land before the opened rail's first paint (product) or corrects the test's frame capture (test), with a test shown red before the fix
- [ ] #3 If it cannot be reproduced: the notes show why the code orders the rebuild before the first paint, and the task closes with no change
<!-- AC:END -->
