---
id: TASK-2851
title: Legacy Media Library palette route hijacks the Library tab
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 01:10'
updated_date: '2026-08-07 03:05'
labels:
  - library
  - navigation
  - ux
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-02, Assessment B anomaly 1, observed at dev `6ffa56516`).

The command palette exposes "Media & Content: Open Media Library", which opens the OLD Media
Library screen (left nav: Media Types / All Media / Analysis Review / Collections/Tags /
Multi-Item Review) rendered UNDER the active ⌃3 Library tab (toast: "Opened Media Library").
After landing there, selecting the palette's "Tab Navigation: Switch to Library" still displayed
the legacy Media Library content under the Library tab until app restart.

Several legacy screens were already retired via `_SCREEN_ALIASES` in
`UI/Navigation/screen_registry.py` (Notes/Skills/Prompts/Search alias to Library). This route
escaped that sweep: two different surfaces both answer to "Library", and the stale one wins a
sticky fight with tab activation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No palette entry opens the legacy Media Library screen as a dead-end twin: the entry is removed, or it deep-links into the canonical Library Media canvas
- [x] #2 Activating the Library tab (palette "Switch to Library" or tab click) always re-asserts the canonical LibraryScreen, even after any legacy/deep-link route
- [x] #3 A regression test covers the palette route and the tab re-assertion
- [x] #4 Live TUI verification: palette route + subsequent "Switch to Library" both land the canonical Library, no restart needed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify the repro at HEAD (dev has moved since 6ffa56516) via live tmux before changing anything.
2. Alias media -> library in screen_registry._SCREEN_ALIASES, mirroring the notes/prompts/skills/search precedent, so the palette route and any other caller land on LibraryScreen instead of the legacy MediaScreen.
3. Wire the media nav-context mode: add media to _LEGACY_ROUTE_LIBRARY_NAV_CONTEXT (app.py) and to LIBRARY_NAV_MODE_TO_ROW_ID (library_screen.py, -> LIBRARY_ROW_BROWSE_MEDIA) so the bare alias lands on the canonical Media canvas, not just generic Library.
4. Update MediaProvider's open_media toast text for accuracy post-fix.
5. TDD: write failing screen_registry + pilot round-trip regression tests first (RED), then implement, then GREEN.
6. Investigate the tab-reassertion ("sticky") half live and via MainNavigationBar/_activate_navigation_button unit tests; add a regression test either way.
7. Update Docs/User_Guide/index.md and library.md text that described Media as a still-separate screen.
8. Live tmux re-verification of both halves post-fix.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Re-verification first (mandatory before touching code):** reproduced the palette hijack live
at HEAD (`6b6c35a4b`) exactly as reported -- "Media & Content: Open Media Library" mounted the
legacy `MediaScreen` (nav: Media Types / All Media / Analysis Review / Collections/Tags /
Multi-Item Review) while the nav bar kept "⌃3 Library" highlighted, toast "Opened Media Library".
AC#1 confirmed still broken. The "sticky" half (AC#2) did **not** reproduce: both the palette's
"Switch to Library" and a direct mouse click on the Library nav button recovered the canonical
`LibraryScreen` immediately, every time, with no restart needed. Root cause of the discrepancy:
`_complete_screen_navigation` unconditionally rebuilds and `switch_screen`s a fresh screen
instance whenever a route resolves to a real screen class -- there never was an "already on this
tab" short-circuit on that path. The only "already active" guard in the whole navigation stack
lives in `MainNavigationBar._activate_navigation_button` (nav-bar click optimism), and it already
correctly distinguishes a folded subroute (`active_route="media"`) from Library's own primary
route (`active_route="library"`) -- pre-existing tests (`test_active_destination_subroute_can_
return_to_primary_route`, `test_folded_routes_highlight_owning_destination`, the latter already
parametrized with `"media"`) already pinned this. Best explanation: the original UAT's "stuck
until restart" symptom was a manifestation of the separately-filed transient-PermissionError
navigation bug (task-2720), whose recovery path (`restore_active`, notify-on-failure) landed on
dev (`9e7b757e9`) after the `6ffa56516` finding and before this branch's HEAD -- not a distinct
sticky-predicate defect. Per the branch's CAUTION, `handle_screen_navigation`'s worker/error
semantics were left untouched.

**Fix (AC#1):** followed the established `_SCREEN_ALIASES` retirement pattern exactly (mirrors
notes/prompts/skills/search/ingest/research/customize) -- `"media": "library"` added to
`screen_registry._SCREEN_ALIASES`, so every caller of `NavigateToScreen("media")` (the palette
entry, saved startup configs, etc.) now resolves to `LibraryScreen` instead of the legacy
`MediaScreen`. `MediaScreen`/`MediaWindow_v2` are not deleted -- their save_state/restore_state
unit tests keep exercising the class directly, same precedent as `SkillsScreen`. To make this a
genuine deep link rather than a generic Library landing: added `"media": {LIBRARY_NAV_CONTEXT_
MODE: "media"}` to `TldwCli._LEGACY_ROUTE_LIBRARY_NAV_CONTEXT` (app.py) and `"media": LIBRARY_
ROW_BROWSE_MEDIA` to `LIBRARY_NAV_MODE_TO_ROW_ID` (library_screen.py) -- the latter table's own
comment had flagged "`media` has no navigation-context entry point at all" as a known gap; this
task closes it. `MediaProvider.handle_media_action`'s "open_media" toast now reads "Opened
Library Media" instead of "Opened Media Library", matching the "Opened Library X" wording its
own `search_transcripts` branch already used for a Library-folded route.

**AC#2:** no code change needed -- the mechanism that was already correct is now locked in with
an explicit regression test naming the exact route the bug reported (`active_route="media"`),
and the alias fix additionally makes the specific "stale MediaScreen occupies the Library slot"
scenario structurally impossible to recur (there is no longer any route that resolves to
`MediaScreen`).

**Tests (TDD):** RED confirmed first -- `test_media_route_resolves_to_library_screen`,
`test_no_route_reaches_the_retired_media_screen`, and the rewritten `test_media_route_round_
trips_to_the_library_media_row` (replaces `test_media_screen_round_trip_restores_type_filter_and_
search_term`, whose premise -- routing to "media" opens `MediaScreen` -- the fix retires) all
failed for the right reason before the alias existed. `test_media_folded_route_returns_to_
library_primary_route` (test_master_shell_navigation.py) passed immediately, as expected for a
behavior that was already correct -- it is a regression lock, not a RED/GREEN pair. GREEN after
implementation. Targeted run: `test_screen_navigation.py` + `test_master_shell_navigation.py` +
`test_command_palette_providers.py` + `test_legacy_entrypoints_retired.py` + `test_screen_
navigation_failure_recovery.py` + `test_shell_destinations.py` + `test_workbench_route_
inventory.py` + `test_destination_shells.py` + `test_library_shell.py -k media`: 293 + 55 passed,
0 failed. `Tests/Library --collect-only`: 1076 collected, no errors. `ruff check` clean on all
five changed source/test files.

**Live TUI verification (AC#4):** tmux socket `sddT3lib<rand>`, scratch profile `/tmp/sddT3`,
`users_name = "sdd_t3"`. Library (via palette "Switch to Library") -> palette "Open Media
Library" now lands the canonical Library Media canvas directly (rail row "Media" selected,
"Media (0)" header, toast "Opened Library Media", nav bar honestly showing "⌃3 Library" active
because it now IS Library) -> palette "Switch to Library" recovers the same canonical
`LibraryScreen` (still on the Media row). No restart needed, no legacy nav rows anywhere.

**Docs:** `Docs/User_Guide/library.md` and `Docs/User_Guide/index.md` described "Media" as one of
the screens still reachable behind its own palette command -- updated both (plus their "Verified
against" stamps) to say the palette entry is a deep link into Library's Media row, matching the
"search" wording already used for the same fold.

**Files changed:** `tldw_chatbook/UI/Navigation/screen_registry.py`, `tldw_chatbook/app.py`,
`tldw_chatbook/UI/Screens/library_screen.py`, `Tests/UI/test_screen_navigation.py`, `Tests/UI/
test_master_shell_navigation.py`, `Docs/User_Guide/library.md`, `Docs/User_Guide/index.md`.
<!-- SECTION:NOTES:END -->
