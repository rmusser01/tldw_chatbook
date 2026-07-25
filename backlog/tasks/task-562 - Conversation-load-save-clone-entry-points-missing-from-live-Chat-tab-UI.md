---
id: TASK-562
title: Conversation load/save/clone entry points missing from live Chat tab UI
status: To Do
assignee: []
created_date: '2026-07-24 22:05'
labels:
  - chat
  - ux
  - dead-code
dependencies: []
---
## Description

Filed from task-504's scout (2026-07-24). `display_conversation_in_chat_tab_ui` was
repaired in task-504 (repointed to the live sidebar ids, scoped QueryError guards,
regression-tested), but the function remains unreachable from the live UI: its three
callers (`handle_chat_save_current_chat_button_pressed`,
`handle_chat_clone_current_chat_button_pressed`,
`handle_chat_load_selected_button_pressed`) fire only from buttons composed in
`Widgets/settings_sidebar.py`, whose `create_settings_sidebar()` has had no callers
since task-412 retired the legacy ChatWindow. The live `EnhancedSettingsSidebar` has
its own New Chat / Clone buttons but no conversation search/load surface. This is a
product decision, not just a wiring fix: either restore load/save/clone entry points
in the live Chat tab (e.g. a Conversations section in the enhanced sidebar), or record
that Chat-tab conversation loading is retired in favor of the Console workspace and
remove the dead handler chain accordingly.

Residual dead right-sidebar surfaces to sweep with whichever direction is chosen:
`app.py` watchers `watch_chat_right_sidebar_collapsed`/`watch_chat_right_sidebar_width`,
dead button/input routers (`app.py` ~:6925/:9299/:9445 referencing
`chat-conversation-*` / `chat-character-*` ids composed nowhere),
`chat_events_sidebar_resize.py`, orphan `#chat-right-sidebar` CSS blocks
(Constants.py + css/), `settings_sidebar.py` module retirement, and the
`# DEAD-ID` fixture restorations in `Tests/fixtures/event_handler_mocks.py`
(six `#chat-character-*-edit` mocks kept only for a legacy handler's tests).

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A product decision is recorded (task notes or decision doc): live Chat-tab conversation load/save/clone is either restored or explicitly retired in favor of the Console workspace
- [ ] #2 If restored: a user can search for and load a saved conversation from the live Chat tab, and the loaded title/id/message log populate (task-504's repaired path)
- [ ] #3 If retired: the dead handler chain (three handlers + tabs wrapper) and the residual dead right-sidebar surfaces listed in the description are removed, with tests updated and no live behavior regressed
<!-- AC:END -->
