---
id: TASK-32104
title: >-
  Library Prompts editor: 'Use in Console' lives in the scrolling header; the
  one-page pager rule is copied three times
status: Done
assignee:
  - '@claude'
created_date: '2026-09-08 22:42'
updated_date: '2026-09-10 15:49'
labels:
  - library
  - prompts
  - cleanup
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the task-32074/32067 reviews (PR #2528): 'Use in Console' sits on its own row inside the scrolling editor header, so it leaves the viewport at the bottom of a long prompt (making it fixed means restructuring the editor shell); the hide-when-one-page pager rule is spelled out in `library_media_canvas.py`, `library_conversations_canvas.py` and `library_prompts_canvas.py` with one shared `single_page` flag — a fourth surface needs a fourth copy. Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 'Use in Console' stays reachable without scrolling, or the guide states the scroll position where it lives
- [x] #2 The one-page pager rule is one helper used by all three canvases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extract the one-page pager rule into ONE helper beside the flag it reads (`library_pager_layout` in `Library/library_pager_state.py`): status parts, boundary reasons and whether the controls render at all. Media keeps its `retry_visible` override (a FAILED fetch's Retry lives in its callout, task-31632); Prompts' 'reason without the disabled check' form is provably the same set, since a reason is only ever non-empty while its button is disabled.
2. Route `library_media_canvas`, `library_conversations_canvas` and `library_prompts_canvas` `_compose_pager` through it; pin that none of the three still carries its own `single_page` branch.
3. 'Use in Console' in the Prompts editor: take AC#1's documented branch -- state in `prompts.md` exactly where it lives -- rather than moving it. It sits in the SCROLLING `#library-prompt-editor-content`; the pinned strip is `#library-prompt-editor-actions`, which is where task-32074 moved it FROM, so moving it back is a product decision reversing that task, not a rider fix.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two halves, one of them deliberately documented rather than moved.

**The one-page pager rule is now one helper.** `library_pager_layout()` (with
its `LibraryPagerLayout` result) lives in `Library/library_pager_state.py`,
beside the `single_page` flag it reads, and answers the three questions the
rule decides: which status parts render, which boundary reasons render, and
whether the Previous/Next row renders at all. Media, Conversations and
Prompts each call it once. Two details the extraction settled:

- Media keeps its `retry_visible` override -- a FAILED fetch's Retry lives
  in its load callout (task-31632), so its pager must not hold a control row
  open just for one.
- Prompts filtered the boundary reasons by 'reason is non-empty' while the
  other two used 'control disabled AND reason non-empty'. Those are the same
  set: `build_library_pager_display` only ever sets a reason while the
  matching control is disabled. The helper uses the shorter form and says so.
- The Conversations canvas's `pager is None` fallback (hand-built states)
  now synthesises a `LibraryPagerDisplay` from its flattened fields and
  calls the same helper, rather than carrying a fourth copy of the rule.

No rendered output changes: the existing task-32067 pins for all three
surfaces pass unmodified, and a one-page Prompts list still reads "1-5 of 5"
with no Previous/Next (live-verified at 235x52).

**'Use in Console' takes AC#1's documented branch.** It sits in
`#library-prompt-editor-content`, which is the editor's `height: 1fr`
scroller, so it does leave the viewport on a long prompt. The strip that
stays put is `#library-prompt-editor-actions` (Save / Discard / More
actions) -- and that is exactly where task-32074 moved the button FROM, on
purpose, to put it under the Basic/Advanced/Info tabs. Moving it back would
reverse that task's placement decision on no new evidence, which is a
product call, not a rider fix. So `prompts.md` now states where it lives and
that the header scrolls, and names the strip that does not.

Files: `tldw_chatbook/Library/library_pager_state.py`,
`tldw_chatbook/Widgets/Library/library_media_canvas.py`,
`tldw_chatbook/Widgets/Library/library_conversations_canvas.py`,
`tldw_chatbook/Widgets/Library/library_prompts_canvas.py`,
`Tests/Library/test_library_pager_layout.py`,
`Docs/User_Guide/library/prompts.md`.
<!-- SECTION:NOTES:END -->
