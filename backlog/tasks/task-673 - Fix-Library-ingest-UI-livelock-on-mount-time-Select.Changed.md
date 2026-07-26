---
id: TASK-673
title: Fix Library ingest UI livelock on mount-time Select.Changed
status: Done
assignee: []
created_date: '2026-07-26 03:26'
updated_date: '2026-07-26 03:33'
labels:
  - ingest
  - bug
  - p0
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pointing the Library ingest path field at a PDF, audio, video or e-book file freezes the whole application at 100% CPU with no recovery, because the per-type options panel recomposes itself in an endless cycle. These are the app's primary ingest types, so the ingest screen is effectively unusable for anything but plain text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Entering a PDF path and submitting completes without the UI freezing
- [x] #2 The same holds for audio, video and e-book paths, and for a folder containing them
- [x] #3 A regression test drives a type group whose options include a select field and asserts the recompose count stays bounded
- [x] #4 Changing a per-type select value by hand still updates the panel title and dependent-field enabled states
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the freeze live and capture thread stacks
2. Instrument the option-change handler to prove the recompose cycle
3. Add a failing canvas test asserting mount posts no option changes
4. Suppress mount-time value announcements in the canvas
5. Re-verify live and run the Library/ingest suites
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Textual posts Changed when a Select mounts (and when an Input mounts with a non-empty value). The canvas forwarded those as OptionValueChanged; the screen recomposes for select/checkbox edits, which remounted the widgets, which posted again -- an unbounded recompose cycle. Only pdf/audio_video/ebook were affected because generic is the one type group with no select field, which is why plain text was the only thing that ever worked.

The canvas now records the value it rendered for each option and drops an incoming event that merely echoes it, updating the record whenever it does forward one. That is free of event-ordering assumptions and still lets a user set a field back to its original value. Proven live: a PDF submit that previously pinned a core at 100% forever now settles at 0% and renders the pre-flight summary for the first time.

Changed: tldw_chatbook/Widgets/Library/library_ingest_canvas.py, Tests/UI/test_library_ingest_canvas.py
<!-- SECTION:NOTES:END -->
