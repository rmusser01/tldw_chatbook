---
id: TASK-35306
title: 'Re-baseline the stale ChatScreen review-projection ceiling test'
status: Done
assignee:
- '@claude'
updated_date: '2026-11-06 06:19'
created_date: '2026-10-17 09:11'
labels:
- follow-up
- tests
- architecture
dependencies: []
references:
- Tests/Architecture/test_console_review_selection_controller_boundary.py
- Tests/Architecture/test_screen_size_ratchet.py
- tldw_chatbook/UI/Screens/chat_screen.py
- backlog/decisions/097-boot-budget-ratchets.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen while checking PR #3144 (Phase 8); red on dev as well as on the branch. At origin/dev 111f8e48fc, run serially, `Tests/Architecture/test_console_review_selection_controller_boundary.py::test_chat_screen_remains_within_reviewed_projection_without_ratchet_raise` fails with `assert (24336, 708) <= (17215, 532)`: physical lines and unique direct ChatScreen methods against `PROJECTED_LINE_CEILING` / `PROJECTED_METHOD_CEILING`, set on 2026-08-27 (#2133, #2137). The same test also expects the screen-size ratchet row to read `("ChatScreen", 16_966, 563)`. The live budget is chat_screen.py's row in Tests/Architecture/test_screen_size_ratchet.py (24,369 lines / 740 methods), which passes. No PR lane runs this file, which is how it went stale.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test passes at dev, either with ceilings re-measured and reviewed against the screen-size ratchet row, or retired in favour of that row with the reason recorded
- [x] #2 If kept, it still fails when chat_screen.py grows past its ceiling (negative control recorded in the notes)
- [x] #3 No screen-size ratchet row rises (ADR-097)
<!-- AC:END -->

## Implementation Notes

Closed by TASK-38988, which owns the same node; see its notes.
- `test_chat_screen_remains_within_reviewed_projection_without_ratchet_raise` now holds the live screen to the measured (24,604, 709) and the ratchet row to `("ChatScreen", 24_616, 741)`. Both re-pins are dated 2026-11-06 under the owner size waiver. The 2026-08-27 projection constants remain as the historical target.
- Negative controls: one appended chat_screen.py line turns the node red, and so does raising the row to 24617.
- No screen-size ratchet row rose.

