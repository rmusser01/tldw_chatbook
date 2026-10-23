---
id: TASK-36387
title: Settings copy and the MCP user guide still describe the retired MCP screen outside the Overview
status: To Do
created_date: 2026-10-22 23:58
labels:
- follow-up
- settings
- copy
- docs
priority: low
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/Screens/settings_tools_hub.py
- Docs/User_Guide/mcp.md
- Docs/User_Guide/index.md
dependencies:
- TASK-35647
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the implementation and review of TASK-35647 (PR #3218, 2026-10-22). Once TASK-35647 lands, the Settings Overview and the Tools & MCP banner name Settings ▸ Tools & MCP. Other copy still describes the layout before task-35150 (#3140), which retired the MCP destination:
- **Tool Profiles copy.** Its scope text says "MCP owns policy editing" (settings_screen.py around :2501 and :10853).
- **The ownership record.** `TOOL_PROFILES_OWNERSHIP_RECORD` (settings_tools_hub.py :103-117) says "MCP Permissions owns policy editing".
- **The User Guide.** `Docs/User_Guide/mcp.md` and `index.md` describe a standalone MCP screen ("Ctrl+9 … ⌃9 MCP in the nav bar"). Ctrl+9 itself is kept on purpose: `MCP_COMPATIBILITY_SHORTCUT` (shell_destinations.py :216) deep-links to Settings ▸ Tools & MCP.
- **Badges.** Most category-list statuses are not save models: "Guided", "Local", "Console", "Global gate" and so on. The State badges are save models, such as "Draft — save with s" and "Auto-saved". For the categories whose list status is a save model, the two can still name different ones. Workspaces, Dreams and Guardian list "Immediate actions" but badge "Applies immediately", and Tool Profiles badges "Reviewed actions". At dev 165e31fab8, the list statuses that say how edits save are "Immediate actions" and "Read-only" (Artifacts, Personas, Skills, Watchlists, Workflows and ACP Defaults, whose badge is "Read-only here"). TASK-35647 fixed this for Tools & MCP by having the badge read the category summary's status.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No Settings copy or User Guide page describes MCP as its own screen; Ctrl+9 is documented as opening Settings ▸ Tools & MCP
- [ ] #2 For each category whose list status says how its edits save ('Immediate actions' or 'Read-only' at dev 165e31fab8), its State badge says the same, from one definition
- [ ] #3 A test fails if such a category's State badge and list status name different save models
<!-- AC:END -->
