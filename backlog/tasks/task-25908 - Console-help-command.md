---
id: TASK-25908
title: 'Console: /help command'
status: Done
assignee: []
created_date: '2026-08-31 15:09'
updated_date: '2026-09-02 00:23'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console registers ten slash commands and offers no way to list them. Verified on origin/dev: Chat/console_command_grammar.py:194-272 is the whole registry; the only help routes are F1 (app.py:7100) and a Show Keybindings palette action (app.py:2004). Both the command names and their descriptions already exist as data - registry.available_names() and the _COMMAND_DESCRIPTIONS table at Chat/console_command_suggestions.py:34 - so this is presentation over data that ships today. Smallest item in the top-ranked slash-command gap and independently useful.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typing /help lists every registered console command with its one-line description and argument hint
- [x] #2 The listing is generated from the live registry, so a newly registered command appears without touching the help code - verified by a test that registers a command and asserts it is listed
- [x] #3 /help <name> shows the detail for one command and says so honestly when the name is unknown
- [x] #4 Output is bounded and scrollable in the transcript rather than flooding it
- [x] #5 Commands the user cannot currently invoke (unavailable or gated) are either omitted or shown with their unavailability stated - not silently listed as usable
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pure console_help.build_help_listing / build_command_detail over the live registry + descriptions\n2. TDD incl. a newly-registered-command-appears test (AC#2) and unavailable-marking (AC#5)\n3. Add registry.commands() accessor\n4. Register /help command + description; wire name->handler + dispatch_map + _console_command_help\n5. availability_fn reuses console_ephemeral.blocked_reason
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Presentation over data that already ships.

Approach:
- New pure module Chat/console_help.py: build_help_listing(commands, descriptions, availability_fn) and build_command_detail(commands, descriptions, name, availability_fn). Both take the live registry's commands (added ConsoleCommandRegistry.commands()), so a newly registered command appears with no help-code change (AC#1/#2). Detail is honest on an unknown name (AC#3). An availability_fn marks a gated command with its reason (or omits it) so it is never silently listed as usable (AC#5).
- Registered /help (name+argument_hint+handler_id) in default_console_registry, added its _COMMAND_DESCRIPTIONS entry, and wired _CONSOLE_COMMAND_NAME_TO_HANDLER_ID + the dispatch_map + _console_command_help in chat_screen.py. The handler builds one bounded block (listing, or detail when /help <name>) and appends it to the scrollable transcript via _append_native_console_system_message (AC#4). availability_fn maps command name -> handler_id -> console_ephemeral.blocked_reason(ephemeral=...), so image/video show their temporary-chat unavailability.

Tests: Tests/Chat/test_console_help.py (7). Updated test_console_command_suggestions.py COMMANDS to include /help.

Files: tldw_chatbook/Chat/console_help.py (new), tldw_chatbook/Chat/console_command_grammar.py, tldw_chatbook/Chat/console_command_suggestions.py, tldw_chatbook/UI/Screens/chat_screen.py, Tests/Chat/test_console_help.py, Tests/Chat/test_console_command_suggestions.py.

NOTE (stale branch): Tests/Chat/test_console_display_state.py fails to COLLECT due to another session's in-flight dirty edit (references estimate_console_next_send_tokens, not yet added); ~8 console dispatch/recovery tests fail at baseline with my change stashed. Neither is caused by this task.
<!-- SECTION:NOTES:END -->
