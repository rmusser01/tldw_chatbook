---
id: TASK-683
title: Make the ingest canvas usable on a first visit
status: Done
assignee: []
created_date: '2026-07-26 03:27'
updated_date: '2026-07-26 04:25'
labels:
  - ingest
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On first run the ingest pane is mostly empty space with no indication of what can be imported, what will happen after import, or what is currently staged. The path field cannot be cleared without selecting its contents by hand.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The empty state names the kinds of content that can be imported
- [x] #2 The user can tell what will happen to a file after it is imported
- [x] #3 The path field can be cleared in one action
- [x] #4 The pane does not leave the majority of its area blank on first visit
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
On first visit the pane was mostly empty with no indication of what could be imported or what importing would do, and a long path could only be cleared by selecting it by hand.

An orientation block now names the accepted sources (file, folder, URL) and types, and says imported items become searchable and usable as chat context. The supported-type list is derived from the same labels the pre-flight breakdown uses, so the promise and the analysis cannot drift. It is shown only while the form is untouched, so it never competes with a real summary. A Clear button appears beside Browse once the field has content.

Changed: tldw_chatbook/Library/library_ingest_state.py, tldw_chatbook/Widgets/Library/library_ingest_canvas.py, tldw_chatbook/UI/Screens/library_screen.py, and their tests
<!-- SECTION:NOTES:END -->
