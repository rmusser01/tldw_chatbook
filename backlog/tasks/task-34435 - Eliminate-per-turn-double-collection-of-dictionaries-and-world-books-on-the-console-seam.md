---
id: TASK-34435
title: Eliminate per-turn double collection of dictionaries and world books on the
  console seam
status: To Do
created_date: 2026-10-08 00:35
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up (deferred T3e/T4d during console maintenance): capture_prompt_transform_inputs collects the same books/entries the resolver collects, so every console send pays the collection twice. The interface seams (books=/entries=) landed in TASK-34415/34416 - wire the pre-collected bundles through the console controller appliers. Coordinate with console pipeline maintenance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One collection per turn verified by spies,Console appliers use the pre-collected bundles,Console tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 3 subtask 3e + Task 4 subtask 4d
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
