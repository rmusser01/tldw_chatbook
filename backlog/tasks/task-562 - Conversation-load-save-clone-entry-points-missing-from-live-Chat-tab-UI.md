---
id: TASK-562
title: Conversation load/save/clone entry points missing from live Chat tab UI
status: Done
assignee: []
created_date: '2026-07-24 22:05'
updated_date: '2026-07-25 15:27'
labels:
  - chat
  - ux
  - dead-code
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A product decision is recorded (task notes or decision doc): live Chat-tab conversation load/save/clone is either restored or explicitly retired in favor of the Console workspace — retired; see `backlog/decisions/026-retire-chat-tab-conversation-entry-chain.md`
- [x] #2 N/A — the "restored" branch was not chosen (AC #1 recorded the retire decision), so this criterion does not apply
- [x] #3 If retired: the dead handler chain (three handlers + tabs wrapper) and the residual dead right-sidebar surfaces listed in the description are removed, with tests updated and no live behavior regressed — zero deferrals across all four implementation commits (see Implementation Notes)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the product decision (retire the dead Chat-tab conversation-entry chain in favor of the native Console) in backlog/decisions/026-retire-chat-tab-conversation-entry-chain.md, citing the 8ea71071f scout finding that ChatWindowEnhanced has been unmounted since 2026-05-06; file the follow-up task for the explicitly kept-this-cycle Chat_Window_Enhanced.py + enhanced_settings_sidebar.py retirement.
2. Unit 1 (commit 7ca5ca4dc): delete the save/clone/load-selected handlers, display_conversation_in_chat_tab_ui, and the chat_events_tabs.py wrapper region, each behind a grep gate (trigger id composed nowhere live + zero direct Python callers); strip Chat_Window_Enhanced.py's now-dead core_handlers map entries same-commit.
3. Unit 2+3 (commit ea7cd9ace): delete the new-conversation/save-details/convert-to-note handlers and the whole conversation-search stack, plus the four app.py router arms that dispatched to them.
4. Unit 4+6 (commit 1e42b2117): delete the character-load-into-sidebar handler family (including the HIGH-RISK gated handle_chat_clear_active_character_button_pressed, gated clean), the dead app.py watchers/branches, and the carried residuals (_conversation_search_timer, the fully-stale test_chat_events_integration.py).
5. Unit 5+7 (this commit): git rm the three zero-importer settings-sidebar modules (settings_sidebar.py, settings_sidebar_optimized.py, chat_events_sidebar_resize.py); strip Chat_Window_Enhanced.py's remaining references (sidebar-resize imports/actions/BINDINGS, the toggle-chat-right-sidebar id which has had zero live compose sites since task-412); sweep the orphan #chat-right-sidebar / #toggle-chat-right-sidebar CSS rules from Constants.py and the three source tcss files, regenerate the bundle via build_css.sh.
6. Add retirement-guard pins to Tests/UI/test_legacy_entrypoints_retired.py for all three retired modules plus the full verified deleted-symbol/button-id set from Units 1-4; update this task file (ACs, decision reference, status Done).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Retired the entire dead Chat-tab conversation-entry chain (ADR-026) across four commits on claude/task-562-chat-tab-conversation-entry, with zero deferrals at any unit's grep gate:

- 7ca5ca4dc (Unit 1): save/clone/load-selected handlers, display_conversation_in_chat_tab_ui, chat_events_tabs.py wrapper region. -1296/+1 LOC.
- ea7cd9ace (Units 2+3): new-conversation/save-details/convert-to-note handlers, the conversation-search stack, four app.py router arms. -826/+2 LOC.
- 1e42b2117 (Units 4+6): character-load-into-sidebar family (incl. the HIGH-RISK-gated handle_chat_clear_active_character_button_pressed, gated clean), dead app.py watchers/branches, carried residuals (_conversation_search_timer, the fully-stale test_chat_events_integration.py, 507 lines, zero surviving live-behavior coverage). -1056/+2 LOC.
- HEAD (Units 5+7, this commit): git rm'd the three zero-importer whole files (Widgets/settings_sidebar.py 1314 LOC, Widgets/settings_sidebar_optimized.py 367 LOC, Event_Handlers/Chat_Events/chat_events_sidebar_resize.py 89 LOC); stripped Chat_Window_Enhanced.py's remaining references to them (the sidebar-resize BINDINGS + action_resize_sidebar_shrink/expand methods, the CHAT_SIDEBAR_RESIZE_HANDLERS dispatch, the toggle-chat-right-sidebar id -- which has had zero live compose sites anywhere in the repo since task-412 retired its only composers); swept the orphan #chat-right-sidebar / #chat-right-sidebar.collapsed / #toggle-chat-right-sidebar id-scoped CSS rules from Constants.py and the three source tcss files (css/layout/_sidebars.tcss, css/features/_chat.tcss, css/components/_buttons.tcss) and regenerated css/tldw_cli_modular.tcss via build_css.sh (diff contains only the swept selectors + the regen timestamp). Added retirement-guard pins to Tests/UI/test_legacy_entrypoints_retired.py: the three retired modules in RETIRED_MODULES/RETIRED_FILES, plus test_task_562_conversation_entry_chain_retired pinning the full verified-absent symbol set (15 chat_events functions, 2 chat_events_tabs functions, 8 CHAT_BUTTON_HANDLERS ids) -- every symbol was re-verified via hasattr()/membership checks against the live modules before pinning, not copied from the plan.

Grand total across the campaign: roughly 5,150 lines removed (~3,200 production + ~490 test in commits 1-3, plus this commit's ~1,880 net whole-file + CSS + CWE reduction), 5 test insertions net.

Deferred units: none. Every gated candidate across all four commits (handle_chat_convert_to_note_button_pressed, is_general_history_conversation, handle_chat_clear_active_character_button_pressed, and this commit's CWE right-toggle reference) passed its gate and was deleted.

Scope explicitly kept this cycle (filed as task-577): Chat_Window_Enhanced.py itself, enhanced_settings_sidebar.py, UI/Chat_Modules/ (incl. chat_sidebar_handler.py's inert reference to the two deleted character handlers inside a never-called method), the use_enhanced_window flag, and the chat_events send-path liveness question.

Verification: Tests/UI/test_legacy_entrypoints_retired.py (5 passed, run alone per Tests/UI asyncio_mode=auto), Tests/Event_Handlers/Chat_Events/ (78 passed, 16 skipped), Tests/test_smoke.py (16 passed, 1 skipped), Tests/UI/test_chat_window_enhanced*.py (63 passed), Tests/UI/test_css_bundle_sync_guard.py + test_css_build_integrity.py (10 passed), pyflakes clean on Chat_Window_Enhanced.py and the guard test file, `python -c "import tldw_chatbook.app"` clean.

Modified/added files: tldw_chatbook/UI/Chat_Window_Enhanced.py, tldw_chatbook/Constants.py, tldw_chatbook/css/layout/_sidebars.tcss, tldw_chatbook/css/features/_chat.tcss, tldw_chatbook/css/components/_buttons.tcss, tldw_chatbook/css/tldw_cli_modular.tcss (regenerated), Tests/UI/test_legacy_entrypoints_retired.py. Deleted: tldw_chatbook/Widgets/settings_sidebar.py, tldw_chatbook/Widgets/settings_sidebar_optimized.py, tldw_chatbook/Event_Handlers/Chat_Events/chat_events_sidebar_resize.py.
<!-- SECTION:NOTES:END -->
