---
id: TASK-28005
title: >-
  Library media viewer - Prev/Next item navigation over the current browse
  result
status: To Do
assignee: []
created_date: '2026-09-02 04:10'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The viewer has no forward path between items: no next/prev buttons or bindings exist (BINDINGS audit library_screen.py:1780-1871; live-tested n, p, brackets and arrows all dead). Sequentially reviewing N items - a conference, a tag-filtered set, or hand-picked videos - costs Escape, arrows, Enter per item, quadratic in list position. Add Prev/Next controls in the viewer header plus keys that walk the browse controller retained ordered page, respecting whatever scope produced the list (type filter, future tag/keyword query), so reviewing a whole set in order is one keypress per item.

Re-verified 2026-09-02 live on dev tip: still absent (n, p, ], [, Left, Right all inert; Reader controls are exactly Back / Find / Read later / Use in Console / More plus the Read-Analysis-Highlights-Info tabs; no footer hint). Two dev-tip foundations make this cheap and complete the sequential-review flow: (1) moving the LIST selection auto-loads the item in the Reader, so next/prev can simply move the selection programmatically; (2) Reader mode persists across selections (begin_selection in Library/library_media_reader_state.py preserves mode), so next/prev while in the Analysis tab reads every analysis in sequence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 From an open media item, a single keypress opens the next (and previous) item in the current list order without returning to the list
- [ ] #2 Boundary behavior at the first and last item is communicated
- [ ] #3 The keys are advertised in the viewer footer
- [ ] #4 Navigation respects an active type filter (walks the filtered result)
<!-- AC:END -->
