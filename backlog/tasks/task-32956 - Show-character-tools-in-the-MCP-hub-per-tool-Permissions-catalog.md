---
id: TASK-32956
title: Show character_* tools in the MCP hub per-tool Permissions catalog
status: To Do
created_date: 2026-09-25 23:14
dependencies:
- TASK-32954
labels:
- mcp
- characters
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32954 shipped character_search, character_get and character_save as Console-only local tools. They never appear in the MCP hub's per-tool Tools/Permissions catalog, because the catalog snapshot builds its local tool provider without a character service (the same gap ask_user has). So a user cannot set the two read tools to Allow: with the default Ask, every character search and read raises an approval card, and the Character Creator skill has had to stop telling users they can relax it. The per-tool permission rows should exist so users can choose Allow for the reads while the save keeps asking.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 character_search, character_get and character_save each appear as a row in the MCP hub per-tool Permissions catalog when the character tools gate is on, and none appear when it is off
- [ ] #2 Setting character_search or character_get to Allow in the hub lets that tool run in the Console without an approval card
- [ ] #3 character_save still raises an approval card on every call even when set to Allow (mutates floor unchanged)
- [ ] #4 The built-in Character Creator skill tells the user once that the read tools can be set to Allow in the MCP hub
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
