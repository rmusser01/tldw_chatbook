---
id: TASK-35152
title: 'Palette: drop duplicate Switch-to-Settings entry from POPULAR_TABS'
status: To Do
created_date: 2026-10-16 20:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
POPULAR_TABS in tldw_chatbook/app_command_providers.py still lists TAB_MCP; after the MCP destination retirement (task-35150 / ADR-240) both TAB_MCP (now an alias) and TAB_SETTINGS render a Switch-to-Settings command in the palette's popular row. Remove the duplicate or relabel it 'Settings — Tools & MCP'.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Exactly one Settings entry in the popular palette row,Provider test updated and green
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
