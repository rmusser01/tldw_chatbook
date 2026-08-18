---
id: TASK-17610
title: 'token_counter char-fallback crashes on list-shaped message content'
status: To Do
assignee: []
created_date: '2026-08-17 17:19'
labels: [bug, tokens]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-existing bug found during TASK-16800 Task 5 (the diff-feedback auto-attach/delivery loop):
a vision-shaped user message whose `content` is a list (e.g.
`[{"type": "text", "text": "hi"}]`, the multimodal/attachment turn shape) crashes
`Utils/token_counter.py`'s char-based fallback estimator whenever no provider usage is reported
for that turn. `estimate_tokens` -> `_chars_estimate`/`_is_cjk` both iterate `text` expecting a
`str` and call `ord(ch)` per character; over a `list`, the "characters" are actually dict items,
and `ord()` on a dict raises `TypeError`. The bug is currently masked in production by the
common case (most sends have either a plain string message or real provider usage), but any
no-usage-reported call with list-shaped content hits it.

Found and worked around (not fixed, deliberately out of that task's scope) in
`Tests/Chat/test_console_diff_feedback_delivery.py`'s
`test_list_content_user_message_leaves_notes_pending_and_appends_no_disclosure`: the test
monkeypatches `agent_service_module._usage_total_tokens` to force a non-`None` usage value so
the no-usage char-estimate fallback path is never exercised, purely to isolate the diff-feedback
attach-loop behavior the test is actually about.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The char-based fallback estimator in `Utils/token_counter.py` handles list-shaped message content without raising: text parts are counted normally, non-text parts (e.g. image/attachment blocks) contribute a fixed, documented estimate instead of crashing.
- [ ] #2 The monkeypatch workaround in `Tests/Chat/test_console_diff_feedback_delivery.py::test_list_content_user_message_leaves_notes_pending_and_appends_no_disclosure` is removed -- the test exercises the real no-usage fallback path.
- [ ] #3 A regression test in `Tests/Utils/` (or the existing token_counter test module) calls the fallback estimator directly on a vision-shaped message (list content with at least one text and one non-text part) and asserts it returns a sane token count instead of raising.
<!-- AC:END -->
