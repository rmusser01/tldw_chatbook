---
id: TASK-32955
title: Expose the character card tools to external MCP clients
status: To Do
created_date: 2026-09-25 18:23
dependencies:
- TASK-32954
labels:
- characters
- mcp
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32954 ships the character_search / character_get / character_save tools as Console-only (exposure CONSOLE_ONLY). External MCP clients connecting to Chatbook's MCP server should also be able to search, read, and (with approval) create or update character cards, the way the Watchlists family offers CONSOLE_AND_EXTERNAL_MCP tools. This needs its own decision on how approvals and the per-session truncation guard work for a client that has no Console session.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 External MCP clients can list and call character_search and character_get
- [ ] #2 character_save is available to external MCP clients only behind an approval path that a human answers, never auto-allowed
- [ ] #3 The truncation guard and server-mode refusal behave the same for MCP callers as for the Console
- [ ] #4 Exposure can be turned off independently of the Console tools
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
