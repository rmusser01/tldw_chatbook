---
id: TASK-32284
title: 'Tool gates pane: text state, humanised labels, consistent restart note'
status: To Do
assignee: []
created_date: '2026-09-10 19:14'
labels:
  - mcp
  - tool-gates
  - accessibility
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Servers, Tool gates checkboxes carry state only by colour (Textual's ToggleButton always draws X; toggling read_file flipped the config while the glyph did not change). Labels are raw ids such as read_file while the first-run wizard shows 'Read file' with a description; the pane says changes apply on the next app restart while Tools mode says the next Console agent run. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each gate row states on or off in text, following the kill-switch label pattern.
- [ ] #2 Gate rows use the wizard's humanised names and descriptions from one shared table.
- [ ] #3 The restart or next-run note is accurate per gate.
- [ ] #4 The Permissions legend's gate-off count links to this pane.
<!-- AC:END -->
