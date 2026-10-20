---
id: TASK-35647
title: Settings Overview and the Tools & MCP state badge still describe the retired MCP screen
status: To Do
created_date: 2026-10-20 04:30
dependencies: []
labels:
- follow-up
- settings
- copy
priority: low
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/Screens/settings_tools_hub.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of Phase 9's rebase onto dev 88fe3108e5 (2026-10-20). task-35150 (#3140) retired the MCP destination into Settings ▸ Tools & MCP, but two pieces of Settings copy still describe the old layout. The Overview card's boundary rows (`SETTINGS_OVERVIEW_BOUNDARY_ROWS` in settings_screen.py) still say "MCP: tool servers are managed on the MCP screen", a screen that no longer exists. And `_persistence_badge` has no Tools & MCP branch, so the category's State banner falls through to "Read-only here", while the category's own summary (`TOOLS_HUB_CATEGORY_SUMMARY`) calls its save model "Immediate actions". A user reading the Overview is sent to a missing screen, and the Tools & MCP banner contradicts the category list.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Settings Overview names where tool servers are managed now (Settings ▸ Tools & MCP) and names no retired screen
- [ ] #2 The Tools & MCP State banner and the category's summary name the same save model
- [ ] #3 A test fails if either Overview row or the Tools & MCP badge names a destination that is not registered
<!-- AC:END -->
