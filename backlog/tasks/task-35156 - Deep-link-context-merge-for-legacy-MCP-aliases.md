---
id: TASK-35156
title: Deep-link context merge for legacy MCP aliases
status: Done
created_date: 2026-10-16 20:15
updated_date: 2026-10-21 06:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec amendment A4, resolved as a contract note in ADR-240: the legacy mcp/tools_settings aliases carry only the default category context via the default-context map; explicit caller contexts are not merged under the alias. Either merge the default context under explicit caller contexts in app_navigation, or codify canonical-route-only producers; decide the approach inside this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Approach decided and recorded (merge vs canonical-only),Legacy-alias deep links carry caller context per the chosen approach,Test posts a legacy-route context and asserts mode and profile land
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Enumerate NavigateToScreen call sites with explicit contexts vs legacy-route map (decision input)
2. RED tests: explicit context on legacy alias (mcp mode/permissions + profile payload) must still land tools category; explicit-context-on-search case
3. Implement Option A merge in app_navigation.py: fill missing default keys under caller context for routes in LEGACY_ROUTE_NAV_CONTEXTS
4. Regression: bare-route legacy defaults unchanged; ratchet check on app_navigation.py
5. Commit
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Option A (merge): legacy-route default contexts now merge UNDER explicit caller contexts in app_navigation (caller keys win, omitted keys filled); bare routes unchanged. Zero of 17 context-carrying call sites target a legacy alias (no repliers). 3 new RED->GREEN tests; ratchet held at 1110.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
