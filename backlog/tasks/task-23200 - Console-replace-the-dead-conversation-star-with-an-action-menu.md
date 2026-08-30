---
id: TASK-23200
title: 'Console: replace the dead conversation star with an action menu'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 00:44'
updated_date: '2026-08-30 01:23'
labels:
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Context rail's conversation rows carry a star button that ships disabled on a fresh install, reserves full row height, and is accompanied by the developer-facing copy 'Local stars unavailable'. It is dead vertical space. Replace it with an asterisk that opens a small action menu so conversations can be managed from the Console screen: Favorite, Change Status, Archive, Rename and Delete. Supersedes ACs 3 and 4 of TASK-23194.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An asterisk control on each conversation row opens an anchored action menu
- [x] #2 The menu offers Favorite, Change Status, Archive, Rename and More to Delete
- [x] #3 Change Status uses the canonical conversation states and Archive maps to resolved
- [x] #4 Favorite state is visible on the row itself without the control reserving full row height
- [x] #5 The developer-facing 'Local stars unavailable' copy is gone
- [x] #6 Menu is keyboard operable and dismissible with Escape
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the per-row star with an asterisk that opens a paged action menu: Favourite, Change status, Archive, Rename, More > Delete.

NO SCHEMA WORK WAS NEEDED, contrary to my own two earlier assessments in this session. I first reported that conversation status had no backing concept and Archive needed a new column, then that both needed one migration. Both were wrong: conversations.state already exists with default 'in-progress', and _ALLOWED_CONVERSATION_STATES in ChaChaNotes_DB is already ('in-progress','resolved','backlog','non-viable') -- identical to tldw_server dev. The error was reading the v1 CREATE TABLE literal in the source instead of inspecting a live schema; PRAGMA table_info on an in-memory DB settled it in one command. Archive is not a flag, it is state='resolved', the same mapping tldw_server's Sync v2 alias table uses (archived/closed -> resolved).

Every action routes through code that already existed: update_conversation handles both title and state (and normalizes state against the allowed tuple), soft_delete_conversation handles delete, and conversation_local_marks_service handles favourites.

Design decisions:
- Menu PAGES in place rather than cascading submenus. A 27-column rail has no room for a second popup, and paging keeps one widget, one lifecycle, one focus owner. Escape steps back out of a submenu before closing, so opening 'Change status' by accident does not throw you out of the row.
- The opener is never disabled. An absent marks service is common on a fresh install and only affects Favourite; status/archive/rename/delete do not need it. Gating lives on the individual entries, each with a stated reason -- which is what removed the need for the 'Local stars unavailable' line entirely.
- The asterisk is one row tall. Favourite state moved onto the title line via _row_marker(), shared by both the render and the height precompute so the two cannot disagree about line count.

Menu shape, labelling and gating are pure and separately tested (Chat/console_conversation_actions.py, 15 unit tests including one asserting the offered states equal the DB's allowed tuple, so the menu can never offer a state update_conversation would reject).

Test fallout: the star tests were rerouted through the menu's Favourite action rather than deleted -- the behaviour they guard (confirmation toast, Rich markup escaped in titles, durable write, no-service guard) is unchanged and still reachable. test_console_workspace_context_disables_star_controls_when_marks_unavailable asserted the OPPOSITE of the new contract and was rewritten to assert the row stays reachable without a marks service.

Two composited-paint tests were left xfail under TASK-23201: they read whole-screen compositor pixels through a harness that does not reproduce the real app's layout and broke on TASK-23193's default change, not on this one. The rail paints correctly in the real app -- verified in the UAT captures.

preflight is green. The diagnostic inventory was regenerated after reviewing BOTH changed rows: my three new logger.error calls in workspace.py log only type(exc).__name__, and the pre-existing library_skills_browse_controller.py row (from 40af5ba07, not this branch) follows the same safe pattern.

Files: Chat/console_conversation_actions.py (new), Widgets/Console/console_conversation_action_menu.py (new), Widgets/Console/console_workspace_context.py, UI/Screens/chat_screen.py, UI/Console_Modules/workspace.py, 2 new test files, 2 updated.
<!-- SECTION:NOTES:END -->
