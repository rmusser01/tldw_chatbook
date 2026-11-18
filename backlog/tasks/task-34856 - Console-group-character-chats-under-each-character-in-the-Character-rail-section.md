---
id: TASK-34856
title: >-
  Console: group character chats under each character in the Character rail
  section
status: Done
assignee:
  - '@robert'
created_date: '2026-09-11 04:13'
updated_date: '2026-09-11 05:44'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Character conversations currently land flat in the Console Conversations browser's Chats pile, mixed with regular chats. Group global/default-scope character chats under their character inside the existing Character rail section (below the avatar/reaction controls), mirroring how Workspaces groups workspace chats. Character chats disappear from the flat Chats section; workspace-scoped character chats stay in their workspace group; starred character chats still appear under Starred with the character name as differentiator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Global/default character conversations render as collapsible per-character groups in the Character rail section with the same headers, rows, star toggles and run-marker glyphs as workspace groups
- [x] #2 Workspace-scoped character conversations appear only in their workspace group (workspace wins; no duplicate under the character)
- [x] #3 Flat Chats section contains no character conversations
- [x] #4 Starred section still includes starred character chats and shows the character label as the differentiator
- [x] #5 Live native character sessions appear under their character group
- [x] #6 Collapse preferences persist per character group (character:<id> keys)
- [x] #7 Missing/deleted character cards fall back to a Character <id> label
- [x] #8 Targeted unit tests cover the routing rules and tray render; docs updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no\nADR path: N/A\nReason: presentation-layer routing over existing data; no schema, storage, or interface-boundary change (direct use of existing browser-state seam).\n\n1. TDD: failing tests in Tests/Workspaces/test_conversation_browser_state*.py for routing rules: character rows with global/default scope route into per-character groups on a new character_groups state field; workspace-scoped character rows stay in workspace groups; flat chats excludes all character rows; starred includes character rows with character label.\n2. Implement: add character_id/character_label to ConsoleConversationBrowserInputRow and ConsoleConversationBrowserRow; build character groups reusing _build_workspace_groups machinery (group ids character:<id>, active-character expanded default); expose character_groups + selected character on ConsoleConversationBrowserState.\n3. TDD + implement in UI/Console_Modules/workspace.py: global persisted fetch gains character_scope=non_character; new global character-scoped fetch with character labels from the character cards DB (fallback Character <id>); native character sessions route into character groups via local_character_id().\n4. Implement tray content="character" mode in Widgets/Console/console_workspace_context.py (group headers/rows reuse existing compose helpers) and mount it in the Character section body under the reaction controls in left_rail.py.\n5. Verify persisted character conversation restore binds the session character (existing load path); wire if missing.\n6. Targeted test runs + ruff on touched files; update Docs/User_Guide/console/sessions-tabs-workspaces.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented character chat groups in the Console Character rail section (TDD; all ACs verified; see Implementation Notes). One pre-existing native-flow test failure verified not caused by this change.
<!-- SECTION:NOTES:END -->
