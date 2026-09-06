---
id: TASK-31665
title: >-
  Inspect rail critique 2026-09-05: minor batch + background banding
  investigation
status: In Progress
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
- [x] #1 Left-rail background bleed diagnosed and fixed (or documented as intended with the contrast implications resolved)
- [x] #2 Task rows show the frontmatter title, not the filename slug (the frontmatter is already read in the same bounded head-read)
- [x] #3 Expansion children are visually contained (indent field or └ glyph) instead of relying on an accidental blank line
- [x] #4 Collapsed-handle "<-Inspect" and open "▸ Inspect" share one glyph vocabulary
- [x] #5 Refresh control is visually attached to the Environment section (not floating between sections) and carries a tooltip naming its scope
- [x] #6 Tasks vocabulary unified ("in progress/to do" everywhere); Change Review header pluralization matches the rail ("1 file") — vocabulary unified and "1 file" fixed; NOTE the collapsed Tasks header now shows only the in-progress count at the narrowest rail (21-col budget), see Implementation Notes
- [x] #7 Change Review's transient "No file changes recorded" flash (≤0.5s) on entry is eliminated or replaced with a loading state
- [x] #8 One canonical Change Review opener decided and documented (four exist today)
- [x] #9 Row secondary text meets 4.5:1 on every background it actually renders over (after #1 lands)
- [x] #10 A bound→bound workspace switch must not transiently render the new root's branch/counts beside the OLD root's PR/checks while the deferred gh fetch is in flight (per-field replace in the non-UNBOUND landing branch; review finding, TASK-31660 round 1)
- [x] #11 A persistent UNKNOWN root (no chat controller / no active session) must not sit on "Checking workspace…" with an inert Refresh indefinitely (31660 re-review obs — the AC#4 situation one state over)
- [x] #12 test_unknown_root_never_paints_the_unbound_copy asserts the rail is open after its toggle (vacuity guard); empty-state docs table and environment.py module docstring updated for the UNKNOWN state
- [x] #13 The fleet section's periodic _sync_console_agent_section recompose steals focus the same way the Environment poll did (its rows ARE focusable) -- apply the 31661 capture/restore + outside-rail guard there (review finding, 31661 round 1)
- [x] #14 row_fits_one_line measures with len() not rich.cells.cell_len — a CJK/wide-glyph title would be under-measured and ellipsize the primary (31662 review minor; one-line fix)
- [x] #15 The same invisible scrollbar thumb ($ds-grid-line on panel surface, ~1.01:1) ships on #console-left-rail-body, #console-settings-body, #settings-impact-pane-body, #library-media-viewer, #prompt-variables-scroll — apply 31663's $ds-text-muted fix (31663 review M6)
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Final task of the 2026-09-05 Inspect-rail critique burn-down. Full write-up
(with the measurement tables) in
`.superpowers/sdd/2026-09-05-inspect-rail-critique-burndown/task-31665-report.md`.

**AC#1 — the banding, diagnosed before it was touched.** Measured twice: an
in-process compositor sweep (`Screen._compositor.render_strips()`) at 235x52
and a live `tmux capture-pane -e` of the real app at the same size under an
isolated profile. Three findings, none of them the reported one:

1. *No stray full-width paint exists, and the left rail does not originate
   one.* The Console shell paints `$ds-surface-panel` (#242f38) behind BOTH
   rails; a left-rail row's own `$surface` (#1e1e1e) run ends where that
   widget ends (cols ~19-32) and the backdrop continues unbroken to col 233.
   That run IS the reported "band" -- the observer read where the widget
   stops as where the band starts, inverting the polarity.
2. *Rail rows ARE split across two backgrounds -- by their own controls.*
   Rows and containers are transparent; the `Button`/`Input` inside a row
   carries Textual's stock `background: $surface`. Documented as INTENDED
   (a control reading as a control is the affordance TASK-31664 had just
   strengthened), with the contrast implication resolved rather than waved
   through -- see AC#9.
3. *The 3.44 / 5.24 pair was two FOREGROUNDS on one background, not one
   foreground on two.* `.console-inspector-section-row-secondary` carried
   `text-style: dim` on top of the already-muted `$ds-text-muted`, painting
   #7a8086 (3.42:1) where every other muted string in the same rail painted
   #a7abaf (5.91:1). Fixing the background would not have fixed it. The
   2026-08-29 refutation is correctly overturned, but for this reason.

**AC#9.** Dropped the double `dim`. A new compositor sweep
(`Tests/UI/test_console_inspector_rail_minors.py`) asserts every painted
string in every inspector row clears 4.5:1 against the background the
compositor actually put behind it, at 80x24 and 200x50 -- so it covers both
surfaces without knowing which is which, and catches a future third one.

**Behaviour changes** (each demonstrated red on `b58877f2d7` in a throwaway
worktree first): AC#7 (Change Review's `<=0.5s` "No file changes recorded"
flash, reproduced under a blocked-detection `threading.Event`), AC#10
(bound->bound switch now resets the PR tier to PENDING so the new root's
branch never sits beside the old root's PR), AC#11 (new
`EnvSourceAvailability.UNKNOWN` landed once after 3 undetermined polls, and
immediately on an explicit Refresh, but only when nothing has ever landed),
AC#13 (the fleet section's periodic sync reuses TASK-31661's own
capture/restore rather than cloning it).

**Mechanical minors.** AC#2 (frontmatter titles out of the same bounded
head-read, block scalars folded), AC#3 (`InspectorSectionRow.indent`,
rendered as MARGIN not padding -- padding would have assumed a stylesheet a
bare harness never loads), AC#4 (one arrow vocabulary, ASCII fallback
preserved, nine cells either way), AC#5 (tail attached, tooltip names its
scope; the 80x24 Environment section is now 6 lines), AC#8 (ruling:
destination follows the surface -- Environment rows open the working tree,
run-anchored controls open that run), AC#14 (`cell_len`), AC#15 (five
scrollbar thumbs).

**Trade-off called out (AC#6).** The Tasks header adopted the backlog's own
"in progress"/"to do", but at the narrowest rail its 21-column budget cannot
hold `3 in progress · 12 to do` (24 columns), so `tasks_count_summary`
degrades to `3 in progress` rather than cutting mid-word. The to-do count is
therefore no longer visible in the COLLAPSED header at any width; expanding
the section still lists every task with its status. The alternatives were
keeping a second vocabulary or shipping `3 in progress · 12 t…`.

**Files.** `Chat/console_environment_state.py`,
`UI/Console_Modules/environment.py`, `UI/Screens/chat_screen.py`,
`UI/Screens/change_review_screen.py`,
`Widgets/Console/console_inspector_section.py`,
`Widgets/Console/console_rail_handle.py`,
`Workspaces/environment_status.py`, `css/components/_agentic_terminal.tcss`
(+ regenerated bundles), `Docs/User_Guide/console/context-and-rag.md`, and
ten test modules (one new).

**Suites + preflight green.** Pre-existing reds re-confirmed at
`b58877f2d7` before being dismissed: `color_grammar` fresh-rail kwarg,
`console_rail_handle` vertical geometry, `inspector_compact_access` x2,
gate1 library core loop, `parallel_runs` navigation guard,
`workbench_visual_snapshots` x3.
<!-- SECTION:NOTES:END -->
