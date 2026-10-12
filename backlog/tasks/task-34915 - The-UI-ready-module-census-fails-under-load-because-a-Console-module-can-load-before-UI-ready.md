---
id: TASK-34915
title: The UI-ready module census fails under load because a Console module can load
  before UI ready
status: To Do
created_date: 2026-10-12 20:42
labels:
- tests
- performance
- flaky
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The UI-ready module census pins how many app modules are loaded when the app reports its UI ready, with no headroom (1033). On a loaded machine it counts 1034 about one run in four on clean dev (2 of 8 measured on 2026-10-12). The extra module is the Console config sync, which the Console control-bar sync imports lazily. That sync is queued to run after the Console screen's first refresh, and the app marks the UI ready right after it finishes pushing that screen, so which happens first depends on scheduling. The module arrived with TASK-34415, after the pin was set. The census runs in the performance workflow, not the required check, and no saved CI log shows it failing; a timing-dependent count still makes the pin unreliable as a ratchet. Found while investigating flaky tests for TASK-34789.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The census gives the same count on every run of unchanged code, loaded machine or not
- [ ] #2 The pin still fails when a change adds a module to the path before UI ready
- [ ] #3 The decision of whether the Console config sync belongs before UI ready is recorded with its measured import cost
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
