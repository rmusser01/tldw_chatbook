---
id: TASK-88
title: Functionalize Settings MCP defaults after Unified MCP upgrade
status: To Do
dependencies:
- TASK-73.6
labels:
- settings
- mcp
- configuration-hub
- deferred
priority: medium
documentation:
- Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md
- backlog/tasks/task-73.6 - Add-Settings-domain-configuration-category-contracts.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Settings > MCP Defaults a functional persisted-defaults category after the upgraded tldw_server Unified MCP implementation has landed and Chatbook can safely mirror the same server-first MCP configuration model without taking over MCP runtime operations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Settings can edit persisted MCP defaults that have a confirmed source-of-truth contract from the upgraded Unified MCP implementation.
- [ ] #2 Settings does not own MCP runtime/server/tool management; those operations remain in the MCP destination.
- [ ] #3 The category exposes explicit owner, source-of-truth, validation, save, revert, and recovery copy for each writable MCP default.
- [ ] #4 Existing read-only/WIP copy remains until the corresponding Unified MCP config contract is available.
- [ ] #5 Focused regressions cover config loading, validation, save/revert, runtime-ownership boundaries, and no hidden/contradictory MCP state.
- [ ] #6 Actual CDP/Textual-web screenshots are captured and approved before any UI PR is created.
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
