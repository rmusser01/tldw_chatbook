---
id: TASK-35150
title: Retire MCP destination into Settings Tools & MCP hub
status: In Progress
created_date: 2026-10-16 16:30
updated_date: 2026-10-16 16:32
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The MCP destination is protocol jargon and a whole top-level destination for tool config. Fold all four workbench modes into one Settings category per the approved spec.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Settings has a Tools & MCP category hosting the four-mode MCPWorkbench
- [ ] #2 mcp/tools_settings routes alias to Settings with category context
- [ ] #3 ctrl+9 preserved as compatibility deep-link
- [ ] #4 mcp-defaults read-only category deleted
- [ ] #5 Tool Profiles handoff lands on the pane in Permissions mode
- [ ] #6 targeted tests green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pane module + TOOLS category registration (destination still live)
2. Keybindings a/r/t/1-4 + per-mode footer hints
3. Deep links + Tool Profiles handoff
4. Destination retirement + route aliases + compat hotkey
5. Test migration sweep
6. Docs + verification
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR-233 written; spec linked
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
