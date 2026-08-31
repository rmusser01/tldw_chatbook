---
id: TASK-25908
title: 'Console: /help command'
status: To Do
assignee: []
created_date: '2026-08-31 15:09'
updated_date: '2026-08-31 15:11'
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
- [ ] #1 Typing /help lists every registered console command with its one-line description and argument hint
- [ ] #2 The listing is generated from the live registry, so a newly registered command appears without touching the help code - verified by a test that registers a command and asserts it is listed
- [ ] #3 /help <name> shows the detail for one command and says so honestly when the name is unknown
- [ ] #4 Output is bounded and scrollable in the transcript rather than flooding it
- [ ] #5 Commands the user cannot currently invoke (unavailable or gated) are either omitted or shown with their unavailability stated - not silently listed as usable
<!-- AC:END -->
