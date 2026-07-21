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

ALSO (found in the P2g-3 final review): there is a SECOND, dead-reachable world-book UI twin in `Widgets/settings_sidebar.py:1266-1337` — `create_settings_sidebar` composes a `if id_prefix == "chat":` "World Books" Collapsible with the same widget ids (f-string-composed `f"{id_prefix}-worldbook-..."`, which is why the P2g-3 literal `chat-worldbook` grep missed it). `create_settings_sidebar` is called only by the production-dead ChatWindow (the live window uses EnhancedSettingsSidebar / settings_sidebar_optimized, which have no world-book UI), so it's the same dead-reachable class — its handlers + CSS were already removed by P2g-3, leaving this section orphaned. Delete `settings_sidebar.py:1266-1337` alongside the ChatWindow removal (the spec's "composed only by create_chat_right_sidebar" premise was incorrect — there are TWO composers).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Chat_Window.py and chat_right_sidebar.py are deleted (or documented as still-needed with why),The 5 ChatWindow test files are deleted or their behavior checks ported to ChatWindowEnhanced,Every #chat-right-sidebar query site is removed or confirmed to tolerate the id's absence (no new user-visible breakage of sidebar resize/toggle),import tldw_chatbook.app OK and the chat/console test suite passes
<!-- AC:END -->
