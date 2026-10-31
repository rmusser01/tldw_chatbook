---
id: TASK-33786
title: >-
  Roleplay: dictionary and lore entry forms push most controls off-screen and
  label none
status: To Do
assignee: []
created_date: '2026-10-02 04:56'
labels:
  - roleplay
  - ux-review-2026-10-01
  - bug
  - layout
dependencies: []
references:
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/findings.json
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the 2026-10-01 Roleplay NN/g + HCI review: finding RP-003 (P0, severity 4, effort M). Report: `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md` (section 5 and Appendix A), with the machine-readable entry in `findings.json` beside it. The review ran on origin/dev @ 84247cb843 (worktree `.worktrees/roleplay-library-ux`).

Line refs below were re-checked on origin/dev @ ab4df99959. Paths are under `tldw_chatbook/` (`personas_screen.py` is `UI/Screens/personas_screen.py`; the `personas_*` and `persona_*` widgets are in `Widgets/Persona_Widgets/`). Capture names refer to the review's capture set; those copied into the QA folder are marked `evidence/`.

**What happens.** When editing a dictionary rule or a lore entry, the first text box takes the whole row width and pushes the other controls off the right edge. In the dictionary form, seven options (regex, probability, group, max replacements, priority, case-sensitive, enabled) are invisible at every size, including 220x55. In the lore form, Position, Priority and Enabled are clipped, and the Case-sensitive, Selective and Regex labels show without their switches. What remains has no labels: the replacement box is unlabelled; Settings shows a bare "4" (scan depth) or "500" (token budget) with placeholder-only labels; Description boxes have no label; and the Strategy menu shows internal ids (sorted_evenly, character_lore_first, global_lore_first). Tab moves focus onto off-screen controls.

**Who it hurts.** World-info builders (J3) cannot tell the pattern box from the replacement box, and cannot see or set most per-entry options without tabbing blind. The User Guide is the only place the controls are named. The Library gives every editor field a persistent label and a one-line help text.

**Evidence:**
- `personas_dictionary_detail.py:206-241`: one Horizontal row holds the Pattern input (default width 100%), three tooltip-only switches and four inputs; the replacement TextArea is unlabelled.
- `personas_lore_detail.py:131-173, 199-212`: the Keys input comes first and Position, Priority and Enabled are clipped; the Settings Name, Scan depth and Token budget are placeholder-only.
- `personas_dictionary_detail.py:29, 265-276`: Strategy shows raw values.
- Captures: `review-rv-nng-b-dict-entry-form-160x45` rows 26-28; `review-rv-nng-b-3-lore-entry-form-220x55`; `review-rv-nng-b-2-lore-settings-unlabeled-160x45` rows 16-26; `review-vf-nng-a-dict-entry-form-tab2-160x45`.
- Docs: `Docs/User_Guide/roleplay-chat-dictionaries/chat-dictionaries.md:55-66` names every control.

**Out of scope:** persona Mode and Enabled (RP-086, persona semantics); Settings-tab scrolling (TASK-33785); entry-table column widths (RP-091); the pane height the entries get (RP-006).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every input, switch and select in the dictionary entry form, the lore entry form and both Settings tabs has a visible text label that stays visible when the control holds a value and when it has focus.
- [ ] #2 At 120x36 and 160x45, no control in either entry form is cut off at the right edge of the pane, and every control can be reached with the mouse (scrolling within the pane if needed).
- [ ] #3 When Tab moves focus to any control in these forms, the focused control is visible on screen.
- [ ] #4 The dictionary Strategy options read in plain words (for example: Spread evenly, Character entries first, Global entries first) instead of internal ids, and the stored values are unchanged.
- [ ] #5 Existing dictionaries and lore books load and save exactly the same values as before the change.
- [ ] #6 The Chat Dictionaries and Lore User Guide pages use the new on-screen labels.
- [ ] #7 Regression tests assert at 120x36 and 160x45 that every control in both entry forms has a label and lies horizontally inside the pane; they fail on the current code.
<!-- AC:END -->
