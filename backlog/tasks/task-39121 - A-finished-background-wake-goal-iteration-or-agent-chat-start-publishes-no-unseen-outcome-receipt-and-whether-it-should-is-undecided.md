---
id: TASK-39121
title: "A finished background wake, goal iteration or agent chat start publishes no unseen-outcome receipt, and whether it should is undecided"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- console
- agents
- needs-owner-decision
priority: medium
references:
- tldw_chatbook/Chat/console_chat_controller.py
- Tests/UI/test_console_fleet_wake_ui_freshness.py
dependencies:
- TASK-39044
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and on round-11 branch fu/r11-modals, which lands TASK-39044. TASK-39044's notes leave this open and pinned neither way. Line numbers are dev's.

**Why automatic turns publish nothing.**
- The controller records an ordinary outcome id only for a turn with a preparation id (`console_chat_controller.py:23322-23323`).
- It publishes an inactive outcome receipt only with that id, and only when the run is `terminal_notification_eligible` (`:28610`, onward).
- Automatic turns (wake, goal iteration, agent chat start) have no preparation id.

**What fu/r11-modals changes.**
- TASK-39044 makes an automatic turn's terminal claim count as eligible.
- `_notify_submission_accepted` then drops stale ordinary ids for an automatic turn without a preparation, so it never publishes under the previous manual turn's receipt id. A round-11 review found it doing exactly that.
- After the branch, a finished wake gets its glyph and toast but no unseen receipt; chain turns publish at chain end.

Whether a background turn should leave an ordinary unseen receipt, and under which outcome id, is a product decision.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner decides whether a finished wake, goal iteration or agent chat start publishes an ordinary unseen receipt, and under which outcome id; the decision is recorded
- [ ] #2 The behaviour matches the decision; if it changes, a test of a chat with a prior manual turn is red before the change
- [ ] #3 An automatic turn never publishes under a manual turn's outcome id; the round-11 pin for that stays green
<!-- AC:END -->
