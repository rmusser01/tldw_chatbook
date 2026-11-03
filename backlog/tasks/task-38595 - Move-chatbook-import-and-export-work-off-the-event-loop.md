---
id: TASK-38595
title: Move chatbook import and export work off the event loop
status: In Progress
created_date: 2026-11-02 14:28
labels:
- bug
- ui
priority: medium
updated_date: 2026-11-02 14:51
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The wizards run the full sync import and export pipelines directly inside async workers so local-mode operations freeze the entire TUI with the progress bar stuck. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Importer and creator calls run via to_thread inside wizard workers
- [ ] #2 Progress UI stays responsive during local import and export
- [ ] #3 Workers use exclusive groups
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
