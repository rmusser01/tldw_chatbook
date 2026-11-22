---
id: TASK-42421
title: "The read-barrier pin's same_session_drain case is red on dev because the held original reader is never entered"
status: To Do
assignee: []
created_date: 2026-11-22 22:10
updated_date: 2026-11-22 22:10
labels:
- follow-up
- console
- tests
- dev-red
priority: medium
references:
- Tests/UI/test_console_tab_publication_read_barrier.py
- tldw_chatbook/Chat/console_runtime.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22). Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**Why.** `Tests/UI/test_console_tab_publication_read_barrier.py::test_actual_close_publishes_real_tabs_before_stock_read_release[same_session_drain]` fails every time. It is red on dev's source too: TASK-42342's notes (line 84) list it among the 12 failures that "fail identically on dev's source". It failed in TASK-42342's two covering runs and in two runs here, one an isolated `-n 4` run of the test's three cases and eight other nodes at load 12. The `[after_close]` and `[coalesced]` cases pass. The file is in no PR step and not in the UI census, so nothing reports it.

**The failure.** `--- child failure: AssertionError: released original held reader never returned; {"entered": false, "future_capture": "not_entered", "held_frame_is_original": false, "native_calls_exact_held_ancestry": 0, "reader_returned": false, "released": true, ...}`. The `_OriginalReadGate`'s reader was never entered. That message comes from the test's `finally` block (from line 834), which runs after any earlier failure in the body and replaces it, so the first failing assertion of the `same_session_drain` branch (lines 771-830: the session switch and `_original_close_read_drain`, the held surface sync, then the close click) is not shown.

TASK-37040 (open) is about the same test taking the session lock before the screen's pump is idle, so Pilot waits can time out. That is a different mechanism: here the original reader is never admitted at all. TASK-42342 (round 21) changed how a session close drains the controller's preparation reads, which this case exercises; the case is red both on this stack, which carries that change, and on dev's source without it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The first failing assertion of the `same_session_drain` case is reported, not hidden by the `finally` block's check
- [ ] #2 Why the original reader is never entered is recorded: test drift after a named product change, or a product defect
- [ ] #3 Drift is repaired in the test; a defect is fixed in product code with a test shown red first
- [ ] #4 All three cases of the test pass, alone and in a `-n 4` run of the file
<!-- AC:END -->
