---
id: TASK-33788
title: >-
  Roleplay: a search with no matches says No characters yet, and filtered counts
  never say N of M
status: Done
assignee: []
created_date: '2026-10-02 04:56'
labels:
  - roleplay
  - ux-review-2026-10-01
  - bug
dependencies: []
references:
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/findings.json
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the 2026-10-01 Roleplay NN/g + HCI review: finding RP-022 (P1, severity 3, effort S). Report: `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md` (section 5 and Appendix A), with the machine-readable entry in `findings.json` beside it. The review ran on origin/dev @ 84247cb843 (worktree `.worktrees/roleplay-library-ux`).

Line refs below were re-checked on origin/dev @ ab4df99959. Paths are under `tldw_chatbook/` (`personas_screen.py` is `UI/Screens/personas_screen.py`; the `personas_*` and `persona_*` widgets are in `Widgets/Persona_Widgets/`). Capture names refer to the review's capture set; those copied into the QA folder are marked `evidence/`.

**What happens.** Type a name that matches nothing and the Characters list says "No characters yet - use New or Import to add one", as if the library were empty, while the count reads "· 0" rather than "0 of 28" and the centre still says "Pick a character from the list". Filtered counts never say "of N" in Characters or Personas: with 348 cards, a search shows "· 85" and "1-50 of 85 characters", and a tag filter shows "· 20", with nothing saying "of 348". Dictionaries and Lore already show "N of M".

**Who it hurts.** Users conclude their characters are gone, or re-import cards and create duplicates. The severity is 3 mainly because the query itself is invisible while typing (TASK-33787).

**Evidence:**
- `personas_library_pane.py:431-442`: the empty branch always renders "No {noun} yet - {hint} to add one." and ignores whether a filter is active.
- `personas_screen.py:4329-4335, 4413` versus `:3690, 4021`: Dictionaries and Lore pass the filtered flag; Characters and Personas never do. `personas_screen.py:3690-3696` and `personas_library_pane.py:493-503`: the page-bar label uses the filtered total only.
- Captures: `evidence/roleplay-characters-search-nomatch-nothing-selected-160x45.txt` rows 9, 23-24; `evidence/roleplay-characters-search-nomatch-120x36.txt` rows 9, 22-24, 29; `review-rv-gap3-search-clockwork-paged-160x45` rows 9, 40; `review-rv-gap3-tag-fantasy-applied-160x45` rows 9, 22.
- Library precedent: "No prompts match your filter." and "1-4 of 4" (`library_prompts_canvas.py:62-63, 623`).

**Out of scope:** the "· 0" shown while a list is still loading (RP-028).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When a search in any of the four modes, or a tag filter in Characters (the only mode that has one), matches nothing, the list says that nothing matches, names the query, states how many items exist in total, and offers a way to clear the filter.
- [x] #2 The "No characters yet" message, and each mode's equivalent with its New or Import hint, appears only when that mode's unfiltered total is 0.
- [x] #3 While a search (any mode) or a Characters tag filter is active, the count reads N of M (for example 0 of 28, or 20 of 348), and the page bar reports matches against the total (for example 1-50 of 85 matches).
- [x] #4 While a filter has zero results, the centre pane does not say "Pick a character from the list".
- [x] #5 Regression tests cover a no-match search in Characters and in Personas, and a no-match tag filter in Characters; they fail on the current code. Tag filtering stays Characters-only (the screen's tag-filter handler is characters-only); this task adds no tag filter to other modes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- 2026-10-29 (Roleplay frame B2b): absorbed by TASK-33910.4 (spec 5.5, S): still To Do on dev at B2b's start, so its AC#1-#5 are TASK-33910.4's AC#24-#28, delivered by B2b's search-aware empty states (Task 9). It stays To Do until B2b's close-out sets it Done with the evidence.

- 2026-10-30 (Roleplay frame B2b, TASK-33910.4): delivered by B2b's search-aware empty states (10f1fc7154), as TASK-33910.4 AC#24-#28 (spec 5.5, S, absorbed). A search or tag that matches nothing names the query, the total and "esc clears the search" (Escape on the list, and in the search box while nothing matches, now clears it; owner decisions D4 and D9), and "No <kind> yet" appears only when the kind is empty; the centre never says "Pick a character" over an empty search. Pinned by Tests/UI/test_roleplay_item_list_app.py::test_a_no_match_search_says_nothing_matches (four kinds), ::test_a_no_match_tag_filter_in_characters_says_so, ::test_no_x_yet_appears_only_when_the_kind_is_empty and ::test_the_centre_never_says_pick_a_character_while_the_search_shows_nothing, each red on the tree before B2b's Task 9, which has dev's behaviour for these states. AC#3: local titles have read "N of M" since B2a (TASK-33910.3); its page-bar half is obsolete since B2a removed the page bar; the server-source "of M" exists on dev, is not fixed by B2b, and is TASK-33910.37.
<!-- SECTION:NOTES:END -->
