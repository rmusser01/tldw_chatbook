---
id: TASK-31665
title: >-
  Inspect rail critique 2026-09-05: minor batch + background banding
  investigation
status: To Do
assignee: []
created_date: '2026-09-05 07:00'
updated_date: '2026-09-05 19:16'
labels:
  - console
  - inspector
  - critique-2026-09-05
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remaining findings from the 2026-09-05 dual-agent critique (18/40).
Includes one investigation: a #2d2d2d background originating in the left
rail bleeds full-width to col 233, splitting single rail rows across two
backgrounds — it is why the same secondary fg measures 3.44:1 on one line
and 5.24:1 on the next, and it overturns the 2026-08-29 refutation of the
secondary-contrast finding (the class DOES render in the right rail).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Left-rail background bleed diagnosed and fixed (or documented as intended with the contrast implications resolved)
- [ ] #2 Task rows show the frontmatter title, not the filename slug (the frontmatter is already read in the same bounded head-read)
- [ ] #3 Expansion children are visually contained (indent field or └ glyph) instead of relying on an accidental blank line
- [ ] #4 Collapsed-handle "<-Inspect" and open "▸ Inspect" share one glyph vocabulary
- [ ] #5 Refresh control is visually attached to the Environment section (not floating between sections) and carries a tooltip naming its scope
- [ ] #6 Tasks vocabulary unified ("in progress/to do" everywhere); Change Review header pluralization matches the rail ("1 file")
- [ ] #7 Change Review's transient "No file changes recorded" flash (≤0.5s) on entry is eliminated or replaced with a loading state
- [ ] #8 One canonical Change Review opener decided and documented (four exist today)
- [ ] #9 Row secondary text meets 4.5:1 on every background it actually renders over (after #1 lands)
- [ ] #10 A bound→bound workspace switch must not transiently render the new root's branch/counts beside the OLD root's PR/checks while the deferred gh fetch is in flight (per-field replace in the non-UNBOUND landing branch; review finding, TASK-31660 round 1)
- [ ] #11 A persistent UNKNOWN root (no chat controller / no active session) must not sit on "Checking workspace…" with an inert Refresh indefinitely (31660 re-review obs — the AC#4 situation one state over)
- [ ] #12 test_unknown_root_never_paints_the_unbound_copy asserts the rail is open after its toggle (vacuity guard); empty-state docs table and environment.py module docstring updated for the UNKNOWN state
- [ ] #13 The fleet section's periodic _sync_console_agent_section recompose steals focus the same way the Environment poll did (its rows ARE focusable) -- apply the 31661 capture/restore + outside-rail guard there (review finding, 31661 round 1)
<!-- AC:END -->
