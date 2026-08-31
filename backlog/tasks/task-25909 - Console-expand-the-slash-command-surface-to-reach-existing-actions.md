---
id: TASK-25909
title: 'Console: expand the slash-command surface to reach existing actions'
status: To Do
assignee: []
created_date: '2026-08-31 15:10'
updated_date: '2026-08-31 15:11'
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
- [ ] #1 A chosen set of existing Console actions is reachable by typed slash command, each dispatching to the action method that already implements it
- [ ] #2 No new capability is introduced - every added command maps to an action reachable today by key binding or palette, and the mapping is listed in the implementation notes
- [ ] #3 Added commands appear in /help and in the suggestion popup without separate registration
- [ ] #4 A command whose underlying action is unavailable in the current context refuses honestly rather than failing silently
- [ ] #5 Existing key bindings and palette entries continue to work unchanged
<!-- AC:END -->
