---
id: TASK-35153
title: 'Tools hub: per-gate effect-scope labels'
status: To Do
created_date: 2026-10-16 20:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec amendment A7 item 1 (Docs/superpowers/specs/2026-10-15-tools-hub-in-settings-design.md): most gates apply to the next chat (Console rebuilds its tool catalog per run); web_deep_search_enabled needs a restart (external MCP server). Add one effect-scope field to the all_tool_gates() table (Agents/builtin_tool_gate.py) so the hub, the Settings category, and the first-run Tools step render the same truth instead of three copies of it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 all_tool_gates() rows carry an effect-scope value,Hub/Settings/first-run render from the shared field,Tests cover next-chat vs restart scopes
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
