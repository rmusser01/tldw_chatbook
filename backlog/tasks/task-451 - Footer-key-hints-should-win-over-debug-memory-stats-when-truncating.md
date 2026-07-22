---
id: TASK-451
title: Footer key hints should win over debug memory stats when truncating
status: To Do
assignee: []
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
- [ ] #1 When the footer is too narrow to show both, the key hints are preserved and the debug memory stats yield (shrink or hide) rather than the reverse
- [ ] #2 No regression to the footer at normal widths on any screen
<!-- AC:END -->
