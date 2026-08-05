---
id: TASK-412
title: >-
  Remove production-dead ChatWindow + chat_right_sidebar.py + audit
  #chat-right-sidebar web
status: Done
assignee:
  - '@Claude'
created_date: '2026-07-21 15:38'
updated_date: '2026-07-24 01:30'
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
- [x] #1 Chat_Window.py and chat_right_sidebar.py are deleted (or documented as still-needed with why),The 5 ChatWindow test files are deleted or their behavior checks ported to ChatWindowEnhanced,Every #chat-right-sidebar query site is removed or confirmed to tolerate the id's absence (no new user-visible breakage of sidebar resize/toggle),import tldw_chatbook.app OK and the chat/console test suite passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify the ground truth from the scout report: confirm ChatWindow (UI/Chat_Window.py) has zero production callers, confirm chat_right_sidebar.py + chat_right_sidebar_optimized.py are only used by Chat_Window.py, confirm create_settings_sidebar's World Books Collapsible (settings_sidebar.py:1266-1337) is only reachable via the dead Chat_Window.py.
2. Audit every #chat-right-sidebar query site (app.py watchers, chat_events.py, chat_events_sidebar_resize.py, sidebar_events.py, Chat_Window_Enhanced.py, chat_screen.py) and classify each site's failure mode (try/except QueryError vs silent no-op vs unreachable because the triggering button/id never exists live).
3. Inspect each of the 5 named ChatWindow-instantiating test files (+ discover any others via grep) and decide delete-vs-port per file by checking for existing equivalent coverage in the ChatWindowEnhanced test suite.
4. Delete Chat_Window.py, chat_right_sidebar.py, chat_right_sidebar_optimized.py, the settings_sidebar.py World Books orphan section, the 3 stale .backup files, and the dead/redundant test files.
5. Extend Tests/UI/test_legacy_entrypoints_retired.py's RETIRED_MODULES/RETIRED_FILES tuples to lock in the new deletions (established precedent from the Conv_Char_Window retirement).
6. Verify: import tldw_chatbook.app, Tests/Event_Handlers/Chat_Events/, the 4 named Tests/UI files, and the updated retirement-guard test.
7. Update the task file and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: ChatWindow (UI/Chat_Window.py) has zero production callers -- app.py only imports ChatWindowEnhanced. chat_right_sidebar.py and chat_right_sidebar_optimized.py both compose the id "chat-right-sidebar" (the latter via f"{id_prefix}-right-sidebar") but are only ever imported by the now-dead Chat_Window.py, so both were deleted (the task named only chat_right_sidebar.py; chat_right_sidebar_optimized.py is its sibling/other branch of the same dead USE_OPTIMIZED_SIDEBARS toggle -- deleting it too avoids leaving an equally-dead file behind). settings_sidebar.py's create_settings_sidebar() also has exactly one caller (the dead Chat_Window.py), so its "World Books" Collapsible (lines 1266-1338, id_prefix=="chat" gated) was deleted and replaced with a one-line NOTE comment; settings_sidebar_optimized.py (the other branch, no World Books section) was left in place -- not named in the AC and deleting a whole second 367-line module felt like scope creep beyond "the settings_sidebar orphan section".

#chat-right-sidebar query-site audit (every site tolerates the id's absence; no NEW breakage, since the id was already never created in the live tree before this change):
- app.py watch_chat_right_sidebar_collapsed / watch_chat_right_sidebar_width: try/except QueryError, logs and returns. No change.
- chat_events.py display_conversation_in_chat_tab_ui (query at ~4281): inside a try wrapped by `except QueryError as qe_disp_main` (~4450) -- tolerates without crashing, but PRE-EXISTING bug: the whole rest of conversation-load population (title/messages) is skipped when it fires, and it always fires today because #chat-right-sidebar has been absent from the live enhanced tree since ChatWindowEnhanced replaced ChatWindow (comment in that file: "Right sidebar removed - functionality moved to settings_sidebar"). Confirmed pre-existing (identical before/after this deletion) by grepping the live composers (enhanced_settings_sidebar.py, settings_sidebar_optimized.py, chat_tab_container.py) for the character-edit-field ids referenced there -- none exist. Left unfixed (out of AC scope); worth a follow-up task.
- chat_events.py handle_chat_clear_active_character_button_pressed (query at ~4929): try/except QueryError, shows an error notify -- but the triggering button id ("chat-clear-active-character-button") is itself only composed by the dead settings_sidebar.py/Chat_Window.py/chat_right_sidebar_optimized.py, so this handler is unreachable live. No change.
- chat_events_sidebar_resize.py handle_sidebar_shrink/handle_sidebar_expand: nested try/except re-raises to an outer except that logs and swallows -- tolerates. These ARE live-reachable (ChatWindowEnhanced BINDINGS ctrl+shift+left/right -> action_resize_sidebar_shrink/expand -> these handlers directly), so Ctrl+Shift+Left/Right has been a silent no-op in production already (pre-existing, unaffected by this deletion). No change.
- sidebar_events.py (Event_Handlers/sidebar_events.py, distinct from the live Event_Handlers/Chat_Events/chat_events_sidebar.py): its SIDEBAR_BUTTON_HANDLERS map and handle_sidebar_toggle_button_pressed are never imported anywhere in production -- fully orphaned, and it doesn't even query #chat-right-sidebar directly (just flips the app reactive). No change.
- Chat_Window_Enhanced.py: "toggle-chat-right-sidebar" appears only as a button-id string (in decorator_handled_buttons and a dead branch of _handle_sidebar_buttons); no #chat-right-sidebar container query, and no button with that id is ever composed live, so on_button_pressed's decorator-skip short-circuits before that branch is reached either way. Left untouched per explicit instruction not to touch the live enhanced sidebar path.
- chat_screen.py: only reads/writes the app.chat_right_sidebar_collapsed reactive attribute, not the widget id. No change.
- CSS (_chat.tcss, _buttons.tcss, _sidebars.tcss, Constants.py's embedded css_content): left untouched. These are declarative selectors on an id that's simply never mounted (a no-op, not an error), Constants.py's css_content is already documented dead code (unrelated pre-existing note at line ~1621: "production loads css/tldw_cli_modular.tcss... NOT Constants.py's css_content"), and the project's regenerate-the-bundle-by-hand path is explicitly discouraged. Not required by the AC (which is about query *sites*, i.e. code that can raise/handle QueryError).

Test files -- disposition per file (all deleted, none ported; verified equivalent live-architecture coverage already exists for every distinguishing behavior):
- test_chat_window_tooltips.py / test_chat_window_tooltips_fixed.py: tooltip checks for toggle-chat-left-sidebar/send-stop-chat covered by test_chat_window_enhanced_integration.py::TestChatWindowEnhancedAccessibility::test_tooltips_present; respond-for-me-button tooltip covered by Tests/Widgets/test_chat_session.py (the button now lives in chat_session.py, not the top-level window); toggle-chat-right-sidebar has no live equivalent (feature doesn't exist), so nothing to port.
- test_chat_window_tooltips_simple.py: NOT in the task's named list of 5 -- discovered via grep (reads UI/Chat_Window.py source by file path, would hard-fail on FileNotFoundError post-deletion). Deleted for the same reasons as the other two tooltip files.
- test_send_stop_button.py: _update_button_state/_check_streaming_state/debounce logic is ChatWindow-specific; the live architecture delegates to Chat_Modules.ChatInputHandler, which already has its own debounce + button-state-delegation coverage in test_chat_window_enhanced_modules.py (test_input_handler_debouncing, test_update_button_state_delegation) plus send/stop tooltip-transition coverage in test_chat_window_enhanced.py / test_chat_window_enhanced_integration.py.
- test_ui_example_best_practices.py: explicitly a pedagogical "best practices" demo file (docstring: "Example UI test demonstrating best practices"), not testing chat-specific production requirements; the widget_pilot/isolated_widget_pilot patterns it demonstrates are already used throughout the rest of the UI test suite.
- test_chat_image_integration_real.py: entire module is module-level pytest.mark.skip(reason="ChatWindowEnhanced not currently in use") -- already contributes zero coverage today. Real (non-skipped) image-attachment coverage exists in test_chat_image_attachment.py, test_chat_image_unit.py, test_chat_window_enhanced*.py, test_chat_session.py.

Also extended Tests/UI/test_legacy_entrypoints_retired.py's RETIRED_MODULES/RETIRED_FILES (the existing Conv_Char_Window-retirement regression-guard pattern) to cover Chat_Window.py, chat_right_sidebar.py, chat_right_sidebar_optimized.py, and the 3 deleted .backup files, so these deletions can't silently regress.

Files deleted: tldw_chatbook/UI/Chat_Window.py, tldw_chatbook/Widgets/Chat_Widgets/chat_right_sidebar.py, tldw_chatbook/Widgets/Chat_Widgets/chat_right_sidebar_optimized.py, tldw_chatbook/UI/Chat_Window_Enhanced.py.backup, tldw_chatbook/Widgets/Chat_Widgets/chat_right_sidebar.py.backup, tldw_chatbook/Widgets/settings_sidebar.py.backup, Tests/UI/test_chat_window_tooltips.py, Tests/UI/test_chat_window_tooltips_fixed.py, Tests/UI/test_chat_window_tooltips_simple.py, Tests/UI/test_send_stop_button.py, Tests/UI/test_ui_example_best_practices.py, Tests/integration/test_chat_image_integration_real.py.
Files modified: tldw_chatbook/Widgets/settings_sidebar.py (World Books Collapsible removed, lines 1266-1338), Tests/UI/test_legacy_entrypoints_retired.py (retirement guard extended).

Verification: import tldw_chatbook.app OK; Tests/Event_Handlers/Chat_Events/ = 86 passed, 26 skipped; Tests/UI/test_chat_window_enhanced.py + test_chat_window_enhanced_integration.py + test_chat_screen_state.py + test_chat_first_run_orientation.py = 46 passed, 13 errors (all pre-existing baseline failures, confirmed via git stash on the unmodified tree: AttributeError on tldw_chatbook.Widgets.compact_model_bar.get_cli_setting in a monkeypatch target, unrelated to this task); Tests/UI/test_legacy_entrypoints_retired.py = 4 passed; Tests/UI/test_chat_window_enhanced_modules.py + Tests/Widgets/test_chat_session.py = 47 passed, 2 skipped (extra sanity check).

Follow-up worth filing (not fixed here, out of AC scope): display_conversation_in_chat_tab_ui's #chat-right-sidebar query (chat_events.py ~4281) silently aborts populating the conversation title/message log whenever a conversation is loaded in the live enhanced window, because the try/except QueryError around it swallows the whole population block. This is a pre-existing bug (the id has never existed in the live tree), not introduced or worsened by this task, but it looks like it may make "load a saved conversation" partially broken today -- deserves its own investigation task.
<!-- SECTION:NOTES:END -->
