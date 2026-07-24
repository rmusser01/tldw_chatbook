---
id: TASK-451
title: Footer key hints should win over debug memory stats when truncating
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 20:00'
labels: [console, ux, footer]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On narrow terminals the app footer (`AppFooterStatus`) truncates the
left-docked key hints (e.g. "Ctrl+K switch se", "F1 help | E") while the
right-docked debug memory stats ("P: 144.0 KB | C/N: 904.0 KB | M: 376.0
KB") keep their full width. Key hints are navigation the user needs;
memory stats are debug telemetry — the priority is inverted under
truncation. Split from task-346 (Console small-terminal guard): this
touches the app-wide `AppFooterStatus` shared by every screen, so it is
out of scope for a Console-layout fix. TCSS has no media queries, so the
likely fix is an `AppFooterStatus`-level resize behavior that shrinks or
hides the memory indicator below a width threshold, preserving the hints.

Source: Console UX expert review 2026-07-20 (finding
`j6-small-terminal-composer-clipped`, footer sub-clause; the composer-clip
half shipped as task-346).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When the footer is too narrow to show both, the key hints are preserved and the debug memory stats yield (shrink or hide) rather than the reverse
- [x] #2 No regression to the footer at normal widths on any screen
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the memory stats (`#internal-db-size-indicator`) are `dock: right`,
so on a narrow footer they RESERVE their width from the right and squeeze the
`dock: left` key hints (the inversion the review flagged). Fix: `AppFooterStatus`
now runs a priority reflow on resize (and when the shortcut context / DB stats
change) that hides the memory stats when there isn't room for the hints AND every
right-side item (`width < hints + word + token + stats + headroom`) -- freeing the
docked-right space for the hints. Recomputed from raw renderables so the decision
is visibility-independent (no flicker). AC#2: at normal widths everything shows
(unchanged). App-wide widget shared by every screen. Served-app-VERIFIED: at
1400px the stats show; at 560px they're gone and the hints extend into the freed
space. Deterministic resize test (wide→shown, narrow→hidden, wide→shown). The one
pre-existing `test_library_registration...` footer failure is baseline (fails
identically on clean origin/dev).
<!-- SECTION:NOTES:END -->
