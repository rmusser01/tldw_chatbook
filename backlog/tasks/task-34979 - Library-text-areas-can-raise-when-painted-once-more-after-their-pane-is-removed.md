---
id: TASK-34979
title: Library text areas can raise when painted once more after their pane is removed
status: To Do
created_date: 2026-10-12 04:58
labels:
- library
- stability
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When a Library pane is removed while the screen's handler is still running, the next repaint can still render a text area that has already left the DOM, and the stock text area raises on a missing component style. PR #3090 (TASK-34789) fixed the note editor's body by moving it to the app's existing detach-safe text area after it crashed a UI test in CI. About 17 other stock text areas in the Library widgets and Library screen code have the same exposure in principle; the media viewer's is the closest sibling. None of them has been seen failing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every text area in the Library that can be removed while its screen is busy renders blank once detached instead of raising
- [ ] #2 Text areas that cannot be removed that way are listed with the reason they were left alone
- [ ] #3 Rendering, selection, scrolling and bindings of each changed text area are unchanged while it is attached
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

## Renumbering provenance

Renumbered from task-34811 to TASK-34979 (2026-10-13): this ID collided
with a concurrently filed task after the branches merged; per the
2026-08-21 owner rule (TASK-19601) the older arrival kept the id.
