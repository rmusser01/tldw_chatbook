---
id: TASK-25909
title: 'Console: expand the slash-command surface to reach existing actions'
status: Done
assignee: []
created_date: '2026-08-31 15:10'
updated_date: '2026-09-02 00:27'
labels:
  - console
  - ux
dependencies:
  - TASK-25908
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Most Console capabilities have no typed name. Verified on origin/dev: ten commands are registered (Chat/console_command_grammar.py:194-272) against hermes's 101, and the gap is mostly discoverability rather than capability - model selection is Alt+M, session switching is Ctrl+K, diff review is ChangeReviewScreen, theme selection is the palette ThemeProvider. Thirteen actions already exist as palette entries (UI/console_command_provider.py:33-97) with handler methods behind them. This is about giving keyboard-only users a typed route to actions that already work, not building new features. Distinct from task-18921, which is about ranking the existing popup by usage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A chosen set of existing Console actions is reachable by typed slash command, each dispatching to the action method that already implements it
- [x] #2 No new capability is introduced - every added command maps to an action reachable today by key binding or palette, and the mapping is listed in the implementation notes
- [x] #3 Added commands appear in /help and in the suggestion popup without separate registration
- [x] #4 A command whose underlying action is unavailable in the current context refuses honestly rather than failing silently
- [x] #5 Existing key bindings and palette entries continue to work unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. One shared handler_id (console-action) for a chosen set of existing palette actions\n2. Register the commands in the grammar; screen maps each name to its existing action method\n3. Generic _console_command_run_action dispatches by name, refuses honestly if unavailable\n4. Descriptions so they appear in /help + popup\n5. TDD incl. a test that every mapped target method exists on ChatScreen
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Typed routes for existing Console actions; no new capability.

Mapping (command -> existing action method, all reachable today by key/palette):
- /model    -> action_open_console_model_popover        (Alt+M)
- /sessions -> action_open_console_session_switcher      (Ctrl+K)
- /workspace-> action_open_console_workspace_switcher    (Alt+W)
- /new      -> action_new_console_tab                    (Ctrl+T)
- /temp     -> action_new_temporary_console_tab
- /settings -> action_open_console_session_settings
- /context  -> action_view_chat_context                  (Ctrl+Shift+P)

Approach: all seven share handler_id 'console-action' in the grammar (CONSOLE_ACTION_COMMANDS registered in default_console_registry); ChatScreen._CONSOLE_ACTION_COMMAND_TARGETS maps each name to the action method that already implements it, and one generic _console_command_run_action getattrs + invokes it (awaiting if async), refusing with a system message when the method is missing or raises (AC#4). They inherit /help + popup listing for free (AC#3) via their _COMMAND_DESCRIPTIONS entries. No key binding or palette entry was changed (AC#5).

Tests: Tests/Chat/test_console_action_commands.py (4) -- crucially asserts every mapped target method exists on ChatScreen, so a future rename can't leave a dangling command. Updated test_console_command_suggestions.py COMMANDS + two skill-prefix tests (/w and /m now also match /workspace and /model).

Files: tldw_chatbook/Chat/console_command_grammar.py, tldw_chatbook/Chat/console_command_suggestions.py, tldw_chatbook/UI/Screens/chat_screen.py, Tests/Chat/test_console_action_commands.py, Tests/Chat/test_console_command_suggestions.py.
<!-- SECTION:NOTES:END -->
