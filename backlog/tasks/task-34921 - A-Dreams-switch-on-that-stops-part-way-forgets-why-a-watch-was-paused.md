---
id: TASK-34921
title: A Dreams switch-on that stops part-way forgets why a watch was paused
status: To Do
created_date: 2026-10-12 23:18
labels:
- dreams
- correctness
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A watch held at switch-on (for example because its address did not resolve) records the reason, and the Tracked list shows it as paused with that reason. When a later switch-on pass stops part-way, its unfinished record drops the reason: the row loses its paused label while the source is still off, and the next start runs a lookup it is documented not to make. The closing step of the pass also skips watches whose restore failed, and nothing tests that. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A hold reason survives a switch-on pass that does not finish
- [ ] #2 The Tracked row of a held watch shows it as paused through any such pass
- [ ] #3 The closing step's handling of a watch whose restore failed is covered by a test that fails if the skip is removed
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
