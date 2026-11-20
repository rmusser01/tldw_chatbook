---
id: TASK-41803
title: Skip unused Media totals in Library keyword Search
status: In Progress
created_date: 2026-11-18 19:58
assignee:
- '@codex'
labels:
- performance
- library
updated_date: 2026-11-18 20:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library keyword Search uses only the bounded Media results but waits for an exact total. Remove that database work from this caller while retaining search and pagination behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native Library keyword Search returns the same ordered scope-limited Media results without executing the unused total query.
- [x] #2 Existing counted Media clients retain exact totals and the Library keeps its safe MATCH, fallback and access checks.
- [x] #3 Paired Terminal runs record Return to visible results, loop gaps and ordinary quit on the same populated corpus.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin the native Library result ordering and trace the unused Media count query.
2. Let the native rows-only consumer omit the count through its existing scope and leaf. Keep policy, filters, FTS and counted callers.
3. Verify real SQLite cross-seam search and three paired Terminal searches of the same corpus.

ADR required: no
ADR path: N/A
Reason: Uses the existing ADR-289 search boundary and ADR-067 pagination contracts. This change removes unused work for one caller without changing the public counted behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The native keyword Search seam opts out of the exact Media total. The DB page, scope policy, MATCH expression and counted defaults are shared with existing callers.

The same 15 IDs arrived in all Terminal arms. Median first paint fell from 358.8 to 316.1 ms, with two of three pairs faster. The 20,000-row SQL control cut approximate VM instructions from 126,000 to 4,000. Terminal database wall time and busy-paint misses varied under load. Evidence is in Docs/QA/console-library-responsiveness/README.md. Broader qualification and hosted checks remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
