---
id: TASK-38611
title: MCP client settlement hardening batch from the 2026-11-02 audit
status: Done
created_date: 2026-11-02 14:30
labels:
- hardening
- mcp
priority: low
updated_date: 2026-11-03 02:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Settlement edge cases in MCP client: stop_and_settle can wait forever on an unreapable child violating the never-hangs contract, and failures of requests that were never sent tear down the whole connection. See bug-sweep-2026-11-02-dev.md MCP findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settlement wait is bounded and returns the original failure after a grace period
- [x] #2 Failures of unsent requests no longer tear down the connection
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Settlement exit waits bounded at 2s across all three loops surfacing the original failure; failures of unsent requests no longer tear down the connection. Commit 3fc22081f6.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
