---
id: TASK-38698
title: "Console: check the other render-on-open deferrals for the paint-before-recompose race TASK-38650 fixed"
status: To Do
created_date: 2026-11-03 11:40
labels:
- follow-up
- console
- tests
priority: low
references:
- tldw_chatbook/Widgets/Console/console_character_context.py
- tldw_chatbook/UI/Screens/chat_screen.py
- Tests/UI/test_console_inspector_rail_deferral.py
dependencies:
- TASK-35970
- TASK-38650
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (TASK-38650's review, PR #3373, 2026-11-03).

**What TASK-38650 found.** When the Inspector rail opens, `rebuild_if_deferred` queues its held recompose on the widget's own message pump. The screen's update-timer task can already be awake in the same event-loop pass (`Timer.pause()` only clears an Event), so its tick paints the held rows before the recompose runs. #3373 holds paints for the widgets the rail cascade rebuilds. Its frame test queues the screen's tick ahead of the opening (`loop.call_soon(screen._on_timer_update)`).

**What may share the shape.** Other widgets also render a deferred snapshot when they are shown. The known one is TASK-35970's Character browser. `ConsoleCharacterContext.render_deferred` (console_character_context.py:225-252) starts the render as an asyncio task when the Character section opens, and nothing holds paints until it lands.

Checked by reading the code on dev 9bbd5c2515; no stale frame has been observed. TASK-37021 covers a different Character case: the rail reveal from `on_show`, which renders after the first frame.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Console widget that renders a deferred snapshot when it opens is listed in the task notes
- [ ] #2 Each is driven by a test that queues the screen's update tick ahead of the opening, as TASK-38650's frame test does; any that paints stale content is fixed, and its test fails on the base build
<!-- AC:END -->
