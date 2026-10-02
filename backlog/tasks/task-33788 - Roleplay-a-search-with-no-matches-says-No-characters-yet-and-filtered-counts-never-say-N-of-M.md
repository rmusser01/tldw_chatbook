---
id: TASK-33788
title: >-
  Roleplay: a search with no matches says No characters yet, and filtered counts
  never say N of M
status: To Do
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
- [ ] #1 When a search or tag filter matches nothing in any of the four modes, the list says that nothing matches, names the query, states how many items exist in total, and offers a way to clear the filter.
- [ ] #2 The "No characters yet" message, and each mode's equivalent with its New or Import hint, appears only when that mode's unfiltered total is 0.
- [ ] #3 While a search or tag filter is active, the count in all four modes reads N of M (for example 0 of 28, or 20 of 348), and the page bar reports matches against the total (for example 1-50 of 85 matches).
- [ ] #4 While a filter has zero results, the centre pane does not say "Pick a character from the list".
- [ ] #5 Regression tests cover a no-match search and a matching tag filter in Characters and in Personas; they fail on the current code.
<!-- AC:END -->
