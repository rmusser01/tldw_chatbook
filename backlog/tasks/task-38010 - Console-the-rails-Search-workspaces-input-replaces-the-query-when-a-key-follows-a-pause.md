---
id: TASK-38010
title: >-
  Console: the rail's 'Search workspaces' input replaces the query when a key
  follows a pause
status: To Do
assignee: []
created_date: '2026-10-30 05:06'
labels:
  - console
  - left-rail
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Impact.** The Workspaces section's 'Search workspaces' box in the Console left rail loses what the user typed: after a pause longer than the search debounce, the next key replaces the whole query. TASK-33622.6 fixed the same defect class for the Conversations 'Filter visible titles' box only (by keeping that one input across tray rebuilds); the Workspaces box is still composed afresh, from state, on every changed sync, and refocused with its text selected.

**Evidence (live, real app, isolated profile, 160x45).** On dev ff815f8559 and on the TASK-33622.6 branch alike: click 'Search workspaces', type 'Ab', wait 1.5 s ('Ab' kept, case preserved), type 'x' -> the box reads 'x'. Captures: devff8-21/22-workspaces-search and rebased-21/22-workspaces-search under the Console UX review session's evidence/bd5/U12/. A screen-binding letter typed at that moment was not exercised.

**Code pointer.** console_workspace_context.py composes ConsoleBrowserSearchInput(initial_value=self.state.workspace_query, ...) for '#console-workspace-search' on every rebuild; Textual Input's select_on_focus defaults to True. The TASK-33622.6 kept-row mechanism (_KeepsConversationFilterRow) is the precedent to reuse.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typing 'Ab', pausing longer than the search debounce, then typing 'x' leaves 'Search workspaces' reading exactly 'Abx' with keyboard focus still in it
- [ ] #2 A debounced refresh never selects, replaces or drops characters typed into 'Search workspaces', and letters that are screen bindings typed there never fire a screen action
- [ ] #3 A Pilot test on the real ChatScreen rail types into 'Search workspaces' across the debounce boundary and asserts the value, focus and that no screen action fired; it fails on the pre-fix code
<!-- AC:END -->
