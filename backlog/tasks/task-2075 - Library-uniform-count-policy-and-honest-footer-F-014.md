---
id: TASK-2075
title: 'Library: uniform count policy and honest footer (F-014)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 04:40'
labels:
  - ux-review
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
(0) appears on 3 rows but not others; footer shows DB telemetry ('Prompts: N/A | Chats/Notes: N/A | Media: N/A') in user chrome. Evidence: db_status_manager.py:69. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Counts follow one policy (dim dash while loading, count when known, none when source off),DB-size telemetry is out of the main footer (Details/Logs),Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (display policy + telemetry relocation; no schema/boundary changes). Steps: 1. RED tests: (a) shell-state -- LibraryShellInput.counts_loading propagates to source/study rows as count_loading; counts are None on lookup error (not misleading zeros); (b) rail render -- loading row shows dim '(…)', known row shows '(N)', off/error row shows no suffix; (c) footer -- update_db_sizes_display('') collapses the indicator (display False), reflow never resurrects an empty indicator; (d) DBStatusManager -- update_db_sizes caches the sizes dict on app.db_sizes_status and does NOT push to the footer; (e) Library Details -- third Status row shows the cached DB sizes when present, omitted when absent. 2. library_shell_state.py: LibraryShellInput.counts_loading + LibraryRailRow.count_loading; build_library_shell_state passes it to the six source rows + three study rows. 3. library_rail.py: count suffix precedence loading '(…)' dim > count_display > count; Details renders a 'DB sizes' row from details_lines[2] when present. 4. library_screen.py _build_library_shell_input: counts None on lookup_error, counts_loading while in flight. 5. AppFooterStatus: indicator display toggles on content; reflow guards empty. DBStatusManager: cache dict on app, keep INFO log (Logs is the telemetry home), stop footer push. 6. Update test_app_footer_shortcut_context manager test to the cache contract. 7. Run rail/shell-state/shell/footer/db tests + parity + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two halves: (a) Uniform rail count policy: LibraryShellInput.counts_loading + LibraryRailRow.count_loading; dim '(…)' while the source snapshot is in flight (six snapshot-backed browse rows + three study handoff rows), count/'+'-estimate when known, NO suffix when the source is off or the lookup errored (previously a failed snapshot rendered a wall of misleading '(0)'s). Collections deliberately has no placeholder -- its count is fetched lazily on first canvas visit, so a placeholder could sit forever; its count appears once known, per policy. (b) DB-size telemetry relocated: DBStatusManager.update_db_sizes caches app.db_sizes_status (dict) and logs the sizes (Logs home) instead of pushing to AppFooterStatus; the footer indicator starts collapsed, toggles on content, and the TASK-451 reflow keeps it down when empty; Library Details Status group renders 'DB sizes · Prompts X · Chats/Notes Y · Media Z' (#library-details-db-sizes) from the cache, omitted until computed. Footer public API and shortcut/token/word-count contract unchanged. Files: library_shell_state.py, library_rail.py, library_screen.py (_build_library_shell_input + _library_details_lines), AppFooterStatus.py, db_status_manager.py, Tests/Widgets/Library/test_library_rail.py (+2), Tests/Library/test_library_shell_state.py (+1), Tests/UI/test_app_footer_shortcut_context.py (1 rewritten + 1 new), Tests/UI/test_library_shell.py (+2). Verified: 7 new/rewritten tests RED->GREEN; rail+shell-state+footer files 39 passed; full test_library_shell.py 312 passed; destination/parity/footer/responsiveness sweep 266 passed + 1 skip (3 failures are the documented pre-existing dev-broken test_library_screen.py ones). Ruff: changed source files clean; Tests/Widgets/Library/test_library_rail.py and Tests/UI/test_library_shell.py were already ruff-red at HEAD (5 pre-existing errors: widget_pilot fixture-import F401/F811, E731 lambda, CollapsibleTitle F401) -- left alone per convention; my two new widget tests follow the same in-file fixture pattern (+2 same-pattern F811s). ADR: not required (display policy + telemetry placement; no schema/boundary changes). Commit fd11f998a.
<!-- SECTION:NOTES:END -->
