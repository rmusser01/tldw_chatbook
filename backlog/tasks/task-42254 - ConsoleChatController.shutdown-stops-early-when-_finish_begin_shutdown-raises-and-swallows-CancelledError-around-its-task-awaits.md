---
id: TASK-42254
title: "ConsoleChatController.shutdown stops early when _finish_begin_shutdown raises and swallows CancelledError around its task awaits"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- console
- lifecycle
- bug
priority: medium
references:
- tldw_chatbook/Chat/console_chat_controller.py
- Tests/Chat/test_console_chat_controller.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consoleb stream (TASK-42155, round 19). In `ConsoleChatController.shutdown` (`tldw_chatbook/Chat/console_chat_controller.py:17801`), `self.begin_shutdown()` runs before the awaits, and `_finish_begin_shutdown` (line 17994) can raise (for example the refusal "Resolve pending dispatch before changing conversation branches."). The exception leaves `shutdown` early, so scratch disposal and the awaits of the stream and submit tasks never run. Separately, `shutdown` swallows `CancelledError` around its `await task`, so a `wait_for` bound placed around `shutdown` cannot fire. Read at e5626ea043; the second claim was not run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A refusal raised from `_finish_begin_shutdown` does not skip scratch disposal or the awaits of the controller's tasks; a test is red on dev and green after
- [ ] #2 A `wait_for` around `shutdown` can time out, or the reason it cannot is recorded in the code with a test
- [ ] #3 A cancelled `shutdown` does not leave a task un-awaited, pinned by a test
<!-- AC:END -->
