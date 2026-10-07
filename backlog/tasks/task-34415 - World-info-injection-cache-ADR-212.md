---
id: TASK-34415
title: World-info injection cache ADR-212
status: To Do
created_date: 2026-10-07 02:40
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 2 / F3+F12: every user message re-fetches all attached world books from SQLite re-parses JSON and reprocesses every entry twice plus recompiles keyword regexes per key per message
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ADR-212 written before code,WorldBookManager generation counter added,Second send with unchanged books performs zero book queries,Keyword patterns compiled once per entry,Double _process_entry eliminated,Recursion dedup uses id set,Golden activation fixtures unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 3 (T3)
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
