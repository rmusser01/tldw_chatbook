---
id: TASK-1191
title: Retire _fit_height_to_content two-pass deferred settle
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 21:30'
updated_date: '2026-07-28 06:28'
labels:
  - console
  - ui
  - layout
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ConsoleWorkspaceContextTray._fit_height_to_content settles over two deferred call_later passes plus a 0.01s timer on every tray state sync. Investigated twice during TASK-1142: not reproducible as a click-eating race under 15 rapid sync cycles, but it is real, separately-verifiable complexity that tests must work around, and a stale-geometry window in principle. Replace with a single-pass deterministic height computation (1142's estimator now covers the hard case) or document why two passes are load-bearing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tray height settles in one deterministic pass, or an ADR-style comment explains the two-pass necessity with a pinned test.
- [x] #2 Existing tray/height tests pass without settle-window workarounds.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read _fit_height_to_content and _schedule_recomposed_content_fit; git blame/log the two-pass+timer machinery to origin commit 1115fa624 for its stated rationale.
2. Determine whether the height computation is now fully state-derived (TASK-1142/1190 estimators) or still depends on post-layout measurement.
3. If state-derived: collapse sync_state's recompose fit to a single call_after_refresh pass (matching on_mount/on_resize's existing pattern), delete the nested call_later + 0.01s timer.
4. Empirically verify via the gate test suites plus repeated targeted runs of the real-click tests (flakiness check), then simplify the test-side settle-workaround retry loop and stale docstring comments that cited the old race.
5. Run the required gate suites in blocking foreground calls; do a real-TUI tmux sanity check of the Console rail (fresh mount + resize) to confirm no visual corruption.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
### Round 1 (commit 2be955f70)

Single-pass route (not the ADR route): the two-pass call_later chain plus
0.01s timer in `_schedule_recomposed_content_fit` was collapsed to one
`call_after_refresh(fit_and_restore_scroll)` -- the same primitive
`on_mount`/`on_resize` already used for this exact job.

Origin (git blame, commit 1115fa624 "Address Console rail review
feedback"): introduced as a theorized fix for "recompose settles child
layout over more than one message turn in scrolled rails," with no
reproduction attached. TASK-1142 (rounds 1+2) investigated this area twice
looking for a click-eating race and could not reproduce one under 15 rapid
cycles; instead it found and fixed a real but *different* bug: the grouped
browser's own auto-height estimate (`_conversation_browser_list_height`)
undercounting wrapped empty-copy/row lines and clipping later siblings.
That fix (wrap-aware `_empty_copy_line_count` / `_marker_prefixed_name_lines`,
landed in TASK-1142, unaffected by TASK-1190) makes the conversation list's
own height fully state-derived and set explicitly at compose time -- no
post-layout measurement needed for it. The remaining top-level tray
children (status pairs, action rows, recovery Statics) are either
nowrap+ellipsis (always 1 line) or short fixed copy; `_fit_height_to_content`
itself still reads real `virtual_region` geometry (not a from-scratch
estimate), so the fix is "defer once, after Textual has processed the
recompose refresh" rather than "compute height with zero deferral."

Evidence the multi-pass fan-out was vestigial: after collapsing to a single
`call_after_refresh`, the full gate suite (155 tests across
test_console_workspace_context_rail.py + test_console_conversation_browser_
state.py + test_console_rail_sections.py) passes with only the one
pre-existing, unrelated failure (`test_console_workspace_context_syncs_
active_conversation_marker`, a `TypeError` on `_sync_console_workspace_
context`'s signature -- reproduced identically via `git stash` on the
unmodified branch tip). test_console_parallel_runs.py (27 tests) passes
clean. The real-click tests most likely to expose a settle race
(`test_section_header_toggles_via_real_click_and_persists_across_rebuild`,
`test_collapsing_workspaces_via_real_click_reveals_aggregate_marker_from_
busy_group`, `test_section_header_caret_is_clickable_at_its_rendered_
screen_coordinates`) were additionally run 5x back-to-back (15 executions)
after simplifying their settle-window workaround -- 15/15 passed.

Removed settle-window workarounds (AC#2): `_click_conversation_browser_
toggle`'s up-to-10-attempt re-scroll/re-check retry loop is now a single
scroll + single CPU-idle pause + assert (still scrolls into view first --
that part is unrelated to the fit-height race). Updated the stale
docstring in `test_section_header_toggles_via_real_click_and_persists_
across_rebuild` that attributed its rebuild-between-toggles structure to
"a real, pre-existing race in that unrelated fit-pass machinery" -- the
rebuild is kept (still needed for the test's own persistence-across-rebuild
coverage) but the comment no longer claims a race that no longer exists.

Real-TUI sanity (tmux, size 235x52, TLDW_CONFIG_PATH scratch config with a
fake OpenAI key so the Console setup card clears and the rail renders):
Console left rail (Session/Model sections, grouped conversation browser
with Starred/Workspaces/Chats) rendered correctly on first mount with no
clipping. Resized the tmux window down to 140x50 and back to 235x52 to
exercise `on_resize` -> `_maybe_relabel_for_width`'s conditional second
pass: text re-wrapped correctly at the narrower width (recovery copy went
from 2 lines to 4, a scrollbar thumb appeared), then cleanly reflowed back
to the original layout with no visual corruption or stuck stale geometry.

Modified files (round 1):
- tldw_chatbook/Widgets/Console/console_workspace_context.py --
  `_schedule_recomposed_content_fit` now schedules one `call_after_refresh`
  pass instead of two `call_later` hops + a 0.01s `set_timer`; docstring
  explains the TASK-1142/1190 state-derivation reasoning and what remains
  conditional (the relabel-triggered second pass).
- Tests/UI/test_console_workspace_context_rail.py --
  `_click_conversation_browser_toggle` simplified from a 10-attempt retry
  loop to a single scroll+pause+assert; stale settle-race docstring in
  `test_section_header_toggles_via_real_click_and_persists_across_rebuild`
  updated to reflect the fix.

Round-1 verification: Tests/UI/test_console_workspace_context_rail.py +
Tests/Workspaces/test_console_conversation_browser_state.py +
Tests/UI/test_console_rail_sections.py in one run -- 155 passed, 1
pre-existing unrelated failure (confirmed via git stash on base commit
18171f9a1). Tests/UI/test_console_parallel_runs.py -- 27 passed. Plus 5x
repeated runs of the three real-click tests most exposed to the old race
(15/15 passed) and a real-TUI tmux mount+resize sanity check.

### Round 2 -- fast-follow (commit d72f6fd9f)

Per coordinator review of round 1 (2be955f70): approved the single-pass
route with two protective additions.

1. Comments pinning the single-pass dependencies at their actual location
   (not just in the commit message/report): `#console-left-rail-body`'s
   `scrollbar-gutter: stable` (tldw_chatbook/css/components/
   _agentic_terminal.tcss, from unrelated commit b4ef4590f) keeps this
   rail's content width from shifting when a row add/remove toggles the
   scrollbar -- without it, a width shift could feed a relabel recompose in
   after the single pass already ran, reopening the exact race this task
   retired. Added a "do not remove" comment there, cross-referenced from
   `_fit_height_to_content`'s own docstring. Also corrected an imprecision
   in that docstring: it previously implied the tray's non-browser children
   were fixed-height/bounded; they are plain wrapping `Static`s
   (`console-workspace-recovery` etc.) with no explicit height -- one pass
   suffices because Textual resolves a `Static`'s wrapped auto-height
   synchronously within the same layout pass `call_after_refresh` already
   waits for, not because content is bounded.
2. Isolated on_resize regression test: the existing
   `test_rail_title_budget_scales_with_terminal_width` resizes the terminal
   but then also calls `sync_state()` again before re-measuring, so it
   cannot tell whether `on_resize`'s own fit pass regrew the budget by
   itself. New `test_on_resize_alone_regrows_wrap_budget_within_one_pause`
   resizes with no `sync_state()` call in between and asserts the row wrap
   budget/content width regrow (measured for this geometry: row width
   23->42, budget 17->36) and height converges within one `pilot.pause()`.
   Ran 5x standalone: 5/5 passed.

Verification: Tests/UI/test_console_workspace_context_rail.py +
Tests/Workspaces/test_console_conversation_browser_state.py +
Tests/UI/test_console_rail_sections.py in one blocking foreground run --
156 passed (155 + the new test), same single pre-existing unrelated
failure as the initial pass.

Modified files (fast-follow): console_workspace_context.py (docstrings),
css/components/_agentic_terminal.tcss (CSS comment),
css/tldw_cli_modular.tcss (regenerated bundle),
Tests/UI/test_console_workspace_context_rail.py (new isolated test).
<!-- SECTION:NOTES:END -->
