---
id: TASK-41910
title: "Library canvases rebuilt at runtime build stock Selects that can crash if replaced mid-recompose"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- ui
- library
priority: low
references:
- tldw_chatbook/Widgets/Library/library_ingest_canvas.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41502 (round 16), not reproduced. TASK-41502 moved the Prompts type filter to `PruneSafeSelect` because a stock `Select` mounted while its canvas was being replaced raises `NoMatches 'SelectOverlay'`. `library_ingest_canvas.py` builds two stock `Select`s (~1567, ~1735) in a canvas that is also rebuilt at runtime, and the other canvases on TASK-41141's list of runtime-rebuilt canvases build stock `Select`s the same way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each runtime-rebuilt Library canvas's Selects survive their canvas being replaced mid-recompose, or the notes show that cannot happen for that canvas
- [ ] #2 A pin like TASK-41502's is red before the fix if a change is made
<!-- AC:END -->
