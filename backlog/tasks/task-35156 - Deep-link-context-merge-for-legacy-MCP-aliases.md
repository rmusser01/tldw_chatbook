---
id: TASK-35156
title: Deep-link context merge for legacy MCP aliases
status: To Do
created_date: 2026-10-16 20:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec amendment A4, resolved as a contract note in ADR-240: the legacy mcp/tools_settings aliases carry only the default category context via the default-context map; explicit caller contexts are not merged under the alias. Either merge the default context under explicit caller contexts in app_navigation, or codify canonical-route-only producers; decide the approach inside this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Approach decided and recorded (merge vs canonical-only),Legacy-alias deep links carry caller context per the chosen approach,Test posts a legacy-route context and asserts mode and profile land
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
