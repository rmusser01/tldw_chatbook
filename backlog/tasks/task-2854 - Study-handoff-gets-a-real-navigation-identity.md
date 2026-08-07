---
id: TASK-2854
title: Study handoff gets a real navigation identity
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 01:10'
updated_date: '2026-08-07 03:51'
labels:
  - library
  - study
  - navigation
  - ux
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-06, observed at dev `6ffa56516`).

Rail Study ▸ "Study decks (0) / opens Study" first opens a Library-local staging canvas (so the
gloss is false for the first click), whose "Continue in Study" lands a full Study screen
(Dashboard/Paths/Flashcards/…) that has NO tab in the tab bar — the bar still highlights
"⌃3 Library" — Escape is dead, and the footer offers no way back. Keyboard users are stranded on
a screen the navigation model claims doesn't exist. This directly violates the "no hidden mystery
navigation" principle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Study screen reached from Library has a truthful navigation identity: either its own tab highlight or a visible breadcrumb naming where you are and how to get back
- [x] #2 The tab bar never highlights Library while a non-Library screen is displayed
- [x] #3 Escape (or an advertised key) returns from the Study screen to the Library staging canvas
- [x] #4 The rail gloss no longer promises "opens Study" for a click that opens a Library staging canvas
- [x] #5 Live TUI verification of the full round trip Library → staging → Study → back
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify the LIB-06 repro at HEAD by reading the routing seam directly (BaseAppScreen.compose -> MainNavigationBar(active=screen_name), shell_destinations.py's Library legacy_routes fold including "study", StudyScreen.__init__ passing screen_name="study") -- confirmed the nav bar highlights Library via the SAME systemic pattern used by Evals/STTS/Writing/Chatbooks (own-name-folded-under-parent), so a global unfold of "study" in shell_destinations.py would be wrong (breaks Home label overrides, command-palette alias tests, ripples into 5+ passing tests) and inconsistent with those other destinations, which are out of scope.
2. Add a LOCAL per-screen override: BaseAppScreen gains `self.nav_bar_active` (defaults to screen_name, preserving today's behavior everywhere) and compose() passes it instead of screen_name to MainNavigationBar. StudyScreen sets it to "" so its own nav bar highlights no destination, without touching the global shell_destinations fold or any other screen.
3. Give StudyScreen a breadcrumb DestinationHeader ("Library (bullet) Study" title, subtitle naming the Esc back-hint) to satisfy the "truthful identity" AC now that the tab bar is cleared.
4. Add an Escape BINDINGS entry + action_study_back_to_library on StudyScreen that posts NavigateToScreen(TAB_LIBRARY, {LIBRARY_NAV_CONTEXT_MODE: "study"}) -- reusing the exact seam open_notes_workspace already uses for "notes" -- plus a footer shortcut registration mirroring LibraryScreen's Files-mode Escape hint (task-2850).
5. Add "study": "create-study" to LIBRARY_NAV_MODE_TO_ROW_ID in library_screen.py so that nav-context mode lands on the Study staging canvas (the row is a "handoff" row like flashcards/quizzes, but row selection does not care about target_kind).
6. Reword the rail's handoff meta line in library_rail.py from "opens Study" to "opens staging canvas" (honest about what the FIRST click does).
7. TDD: write/port failing tests first (destination-header + nav-highlight test in test_destination_headers.py, Escape round-trip test in test_screen_navigation.py mirroring the prompts/skills/search pattern, rail-gloss wording test in test_library_shell.py), confirm RED via a saved-patch revert/reapply cycle (git stash is forbidden repo-wide), then implement and confirm GREEN.
8. Run targeted suites (test_study_screen.py, test_destination_headers.py, test_master_shell_navigation.py, test_screen_navigation.py, test_library_shell.py, Tests/Library/test_library_shell_state.py) plus a Tests/Library --collect-only sanity sweep.
9. Live tmux verification of the full round trip: Library -> Study staging canvas -> Continue in Study -> Study screen (capture tab-bar state, no highlight) -> Escape -> back on the Library staging canvas.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Re-verified the LIB-06 repro at HEAD (dev has moved since 6ffa56516) before fixing: BaseAppScreen.compose() yields MainNavigationBar(active=self.screen_name), StudyScreen.__init__ passes screen_name="study", and shell_destinations.py folds "study" under the Library destination for nav-highlight purposes -- so the tab bar boxed Library while the fully separate StudyScreen (Dashboard/Paths/Flashcards/.../no Library chrome) was displayed. Escape was unbound, and library_rail.py's handoff meta line said "opens Study" for a click that only opens a Library-local staging canvas.

Investigated whether this was a genuine routing bug (Study deserving its own tab) vs. by-design folding: SHELL_DESTINATION_ORDER has no standalone "study" destination, and the identical own-screen-name-folded-under-a-parent-destination pattern is used deliberately by Evals/STTS/Writing/Chatbooks (all separate full BaseAppScreen subclasses folded under Lab/Library/Artifacts for nav-highlight purposes only) -- so unfolding "study" globally in shell_destinations.py would have been inconsistent with that systemic pattern and would have rippled into Home's "Opens:" label overrides, the command palette's search-alias tests, and 5+ other passing tests that pin the current fold. Chose a local, additive fix instead: BaseAppScreen gained `self.nav_bar_active` (defaults to screen_name, preserving every other screen's behavior byte-for-byte), and StudyScreen alone sets it to "" so its own composed nav bar highlights no destination -- the global fold (Home labels, palette aliases, screen routing) is untouched.

Added a "Library (bullet) Study" breadcrumb + "Esc: back to Library" hint to StudyScreen's existing DestinationHeader (title/subtitle only -- no new widget), an Escape BINDINGS entry + action_study_back_to_library posting NavigateToScreen(TAB_LIBRARY, {LIBRARY_NAV_CONTEXT_MODE: "study"}) (the exact seam open_notes_workspace already uses for "notes"), a footer shortcut registration mirroring LibraryScreen's Files-mode Escape hint (task-2850), a new "study" -> "create-study" entry in LIBRARY_NAV_MODE_TO_ROW_ID so that nav-context lands on the Study staging canvas, and reworded the rail's handoff meta line from "opens Study" to "opens staging canvas".

TDD: wrote/updated the tests first, confirmed RED by reverting the 4 source files via a saved git-diff patch (git stash is forbidden repo-wide) and rerunning -- all three failed for the expected reason (Library still boxed / plain "Study" title / Escape a no-op) -- then reapplied the patch and confirmed GREEN. Escape always returns to the "Study decks" row regardless of whether Study/Flashcards/Quizzes originally reached it (documented simplification: the three rows are variations of one staging surface with one "Continue in Study" exit; threading the originating row id through the whole handoff was judged out of scope).

Live tmux round trip (Global Constraints recipe, socket sddT4lib<rand>, scratch profile /tmp/sddT4/users_name sdd_t4): Library -> command palette "Library" -> clicked "Study decks" rail row -> staging canvas ("Study decks" / "Continue in Study", rail meta line reads "opens staging canvas") -> clicked "Continue in Study" -> Study screen (tab bar box gone from every tab, header reads "Library (bullet) Study" / "Esc: back to Library") -> pressed Escape -> back on the Library "Study decks" staging canvas with "⌃3 Library" boxed again. Captures kept in the report.

Tests: Tests/UI/test_destination_headers.py::test_study_screen_mounts_destination_header_and_clears_nav_highlight (renamed+rewritten from ..._and_boxes_library), Tests/UI/test_screen_navigation.py::test_study_screen_escape_returns_to_library_study_staging_canvas (new), Tests/UI/test_library_shell.py::test_rail_rows_are_one_line_by_default_with_meta_only_for_handoffs (gloss wording updated). Full targeted runs: test_study_screen.py + test_destination_headers.py + test_master_shell_navigation.py (47 passed), test_screen_navigation.py (85 passed), test_library_shell.py (344 passed, 1 pre-existing unrelated failure -- test_landing_footer_advertises_the_landing_keyboard_story fails identically on a clean revert of all 4 touched source files, confirmed via the same patch-revert technique, so it predates this change and was not investigated further), Tests/Library/test_library_shell_state.py (28 passed), Tests/Library --collect-only sanity sweep (1076 collected).

Files: tldw_chatbook/UI/Navigation/base_app_screen.py, tldw_chatbook/UI/Screens/study_screen.py, tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/Widgets/Library/library_rail.py, Tests/UI/test_destination_headers.py, Tests/UI/test_screen_navigation.py, Tests/UI/test_library_shell.py, Docs/User_Guide/library.md (rail-row gloss, hand-off section, keyboard table, quirks entry, Verified-against stamp).
<!-- SECTION:NOTES:END -->
