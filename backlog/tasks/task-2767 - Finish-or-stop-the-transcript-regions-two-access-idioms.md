---
id: TASK-2767
title: Finish or stop the transcript region's two access idioms
status: Done
assignee: []
created_date: '2026-08-07 06:41'
updated_date: '2026-08-07 19:45'
labels:
  - refactor
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 3's ConsoleTranscriptRegion left the transcript's DOM with two access idioms: 3 screen methods route through _console_transcript_region_or_none, while 9 still query_one the transcript widgets directly. Both resolve to the same widget today, so this is not a defect -- it is a transitional state a later reader could mistake for settled design. Decide deliberately: migrate the remaining 9, or document the split as intentional.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Either all screen access to the transcript's widgets routes through the region, or the mixed state is documented in DESIGN.md as deliberate with its reason
- [x] #2 No behaviour change; the geometry baseline stays byte-identical
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Decision: the split is deliberate, and is now documented in `DESIGN.md`** (AC
option 2). Migrating the remaining direct queries was considered and rejected.

The measured split is 3 region-routed vs 8 direct (the task said 9; the count
was never the point). The distinction behind it is real:
`ConsoleTranscriptRegion` defines exactly three public behaviours --
`capture_reading_state`, `restore_reading_state`, `note_follow_intent` -- all
about the *viewport*, which the region owns because the region is what scrolls.
The three region-routed sites call exactly those. The eight direct sites reach
`query_one("#console-native-transcript", ConsoleTranscript)` for the transcript
widget's *own* API, which the region does not own.

Routing the eight through the region would add eight pass-through getters whose
whole body is `return self._transcript_or_none()`, converting an owner into a
facade with no invariant behind it -- and in Textual `query_one` is transparent
across compound-widget boundaries by design, so an id lookup crossing into a
region is idiomatic, not a leak. The rule recorded in `DESIGN.md` is
**ownership, not reachability**: route through the region only when the
invariant is the region's.

No production code changed, so AC #2 holds trivially; the geometry baseline is
byte-identical to dev and green (37 passed with the transcript-region suite).
<!-- SECTION:NOTES:END -->
