---
id: TASK-2767
title: Finish or stop the transcript region's two access idioms
status: To Do
assignee: []
created_date: '2026-08-07 06:41'
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
- [ ] #1 Either all screen access to the transcript's widgets routes through the region, or the mixed state is documented in DESIGN.md as deliberate with its reason
- [ ] #2 No behaviour change; the geometry baseline stays byte-identical
<!-- AC:END -->
