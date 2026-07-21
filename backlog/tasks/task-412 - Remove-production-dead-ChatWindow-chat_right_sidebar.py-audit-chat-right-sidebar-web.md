---
id: TASK-412
title: >-
  Remove production-dead ChatWindow + chat_right_sidebar.py + audit
  #chat-right-sidebar web
status: To Do
assignee: []
created_date: '2026-07-21 15:38'
labels:
  - tech-debt
  - dead-code
  - chat
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from P2g-3 (which scoped to the world-book UI only). ChatWindow (UI/Chat_Window.py) is never instantiated in production — the app uses ChatWindowEnhanced, which 'removed the right sidebar' — but 5 test files instantiate it (test_chat_window_tooltips.py, test_chat_window_tooltips_fixed.py, test_send_stop_button.py, test_ui_example_best_practices.py, test_chat_image_integration_real.py). Widgets/Chat_Widgets/chat_right_sidebar.py is the ONLY creator of #chat-right-sidebar, which is queried by ~5 live-ish sites (app.py:8381, chat_events_sidebar_resize.py x2, chat_events.py:4372/5020) that currently fail-gracefully. Removing these dead files requires a broader audit of the #chat-right-sidebar query web + deleting/porting the 5 ChatWindow tests — beyond the world-book scope. Surfaced during the P2g-3 spec review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Chat_Window.py and chat_right_sidebar.py are deleted (or documented as still-needed with why),The 5 ChatWindow test files are deleted or their behavior checks ported to ChatWindowEnhanced,Every #chat-right-sidebar query site is removed or confirmed to tolerate the id's absence (no new user-visible breakage of sidebar resize/toggle),import tldw_chatbook.app OK and the chat/console test suite passes
<!-- AC:END -->
