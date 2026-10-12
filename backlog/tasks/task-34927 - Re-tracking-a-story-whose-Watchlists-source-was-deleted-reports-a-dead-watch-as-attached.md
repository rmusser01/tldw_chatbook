---
id: TASK-34927
title: Re-tracking a story whose Watchlists source was deleted reports a dead watch
  as attached
status: To Do
created_date: 2026-10-12 23:18
labels:
- dreams
- correctness
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Dreams notice for a source that stopped working tells the user to delete the source in Watchlists and track the story again. Tracking again finds the story's existing watch and reports it attached, with the deleted source's id, so the watch stays active with nothing behind it. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tracking a story again after its source was deleted creates a working watch with a new source
- [ ] #2 A test follows the notice's instructions end to end and fails before the fix
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
