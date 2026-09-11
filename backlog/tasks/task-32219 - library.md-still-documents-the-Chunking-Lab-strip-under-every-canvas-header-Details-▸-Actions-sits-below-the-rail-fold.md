---
id: TASK-32219
title: >-
  library.md still documents the Chunking Lab strip under every canvas header;
  Details ▸ Actions sits below the rail fold
status: Done
assignee: []
created_date: '2026-09-10 14:54'
updated_date: '2026-09-10 19:05'
labels:
  - library
  - docs
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Layout tour says the strip sits 'directly under the header, on every Library canvas'; task-32064 moved it to Details ▸ Actions and the same page's control table says so, so the page contradicts itself. At 52 rows the Actions group is below the rail's fold (six wheel notches to reach it) with no scroll cue. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 16.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 library.md's Layout tour matches the shipped placement
- [x] #2 The rail shows a scroll cue when Details ▸ Actions is below the fold, or Actions moves above the fold
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Delete the Layout-tour bullet; point at the control-table row; check every Chunking Lab hit on the page agrees.
2. Add a fold cue to the rail, displayed only while the rail scrolls.
3. Live-verify at 235x52, 100x30 and 60x24 on both profiles.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1: the Layout tour said the strip sits "directly under the header, on *every* Library canvas"; task-32064 moved it to Details ▸ Actions and the same page's control table already said so. The bullet is gone and the tour points at the control-table row, so the page states the placement once. The three remaining "Chunking Lab" hits (lines 768/781/797) are historical Verified-against stamps recording what past PRs did -- left alone, as the plan directed.

AC#2: the rail's last line reads "▾ scroll for more" whenever its content runs past the fold, and hides itself as soon as everything fits. Three things the live pass found that no test caught first:

1. The plan's placement (a Static at the END of the Details body) is below the very fold it describes -- it can only be read after you have scrolled to the bottom, where there is nothing left to announce. It is `dock: bottom` on the rail now. The test asserts the cue's region is inside the rail's and that it PAINTS, not merely that `display` is True.
2. `max_scroll_y` reports the PREVIOUS layout inside `on_mount` and inside the `virtual_size` watcher, so the check read "no overflow" for a rail about to overflow. Deferred with `call_after_refresh`.
3. The rail recomposes on every count/evidence/route change, and each recompose rebuilt the cue as hidden faster than the measurement could turn it on -- live it never appeared at all. The visibility decision lives on the RAIL now, so a recompose carries it and the next measurement only corrects it.

Copy is measured, not chosen: the rail's Details column is ~19 cells at the rail's own minimum width, so the plan's "▾ more below — scroll or press F6" (33 cells) wraps to two rows and costs another line of the space it is complaining about. The F6 hint lives in the tooltip.

Live at 235x52, 100x30 and 60x24 on both profiles: the cue paints docked at the rail's bottom edge; Details ▸ Actions is three wheel notches away instead of six.

Files: Docs/User_Guide/library.md, tldw_chatbook/Widgets/Library/library_rail.py, tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_library_crit9_rail.py
<!-- SECTION:NOTES:END -->
