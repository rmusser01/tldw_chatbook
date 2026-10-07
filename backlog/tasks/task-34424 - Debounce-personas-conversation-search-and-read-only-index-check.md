---
id: TASK-34424
title: Debounce personas conversation search and read-only index check
status: To Do
created_date: 2026-10-07 02:42
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 5 / F14: personas conversation search fires a full search cycle with a write transaction per keystroke and media filter debounce is 0.12s vs 0.2-0.3s elsewhere
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 5-char burst fires exactly one search,READY index performs zero write transactions,Media filter debounce 0.25s
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 12 (T12)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
