---
id: TASK-17610
title: token_counter char-fallback crashes on list-shaped message content
status: Done
assignee:
  - '@claude'
created_date: '2026-08-17 17:19'
updated_date: '2026-08-18 02:04'
labels:
  - bug
  - tokens
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
- [x] #1 The char-based fallback estimator in `Utils/token_counter.py` handles list-shaped message content without raising: text parts are counted normally, non-text parts (e.g. image/attachment blocks) contribute a fixed, documented estimate instead of crashing.
- [x] #2 The monkeypatch workaround in `Tests/Chat/test_console_diff_feedback_delivery.py::test_list_content_user_message_leaves_notes_pending_and_appends_no_disclosure` is removed -- the test exercises the real no-usage fallback path.
- [x] #3 A regression test in `Tests/Utils/` (or the existing token_counter test module) calls the fallback estimator directly on a vision-shaped message (list content with at least one text and one non-text part) and asserts it returns a sane token count instead of raising.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red: regression test on vision-shaped list content through the fallback estimator\n2. Fix: content-normalization in the char estimator (text parts counted, non-text fixed contribution)\n3. Remove the delivery test's monkeypatch workaround\n4. Green: token_counter + delivery suites; PR
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed at the estimate_tokens boundary: new _flatten_message_content normalizes any content shape (str passthrough; part-list -> concatenated text parts + count of non-text parts; other shapes -> str()); each non-text part contributes NON_TEXT_PART_TOKEN_ESTIMATE = 85 (documented: the published GPT-4V low-detail image cost, used as a conservative floor consistent with the estimator's contract). This also fixes the tiktoken path, which would have crashed identically on list content. Three regression tests added to Tests/Chat/test_token_counter.py (vision-shaped text+image list, image-only list, count_tokens_messages end-to-end) — all proven red pre-fix (TypeError/ImportError). The monkeypatch workaround in test_console_diff_feedback_delivery.py's list-content test removed along with its now-orphaned agent_service_module import — the test exercises the real no-usage fallback path unpatched and passes. Regression net: token_counter 34 passed, delivery 14, cost-tracker/history-budget/session-settings/bridge 365 — all green.
<!-- SECTION:NOTES:END -->
