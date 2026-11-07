---
id: TASK-33784
title: >-
  Roleplay: character editor long-text fields show 0-1 lines, so edits land in
  hidden text and save corrupted
status: To Do
assignee: []
created_date: '2026-10-02 04:55'
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
Found by the 2026-10-01 Roleplay NN/g + HCI review: finding RP-002 (P0, severity 4, effort S). Report: `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md` (section 5 and Appendix A), with the machine-readable entry in `findings.json` beside it. The review ran on origin/dev @ 84247cb843 (worktree `.worktrees/roleplay-library-ux`).

Line refs below were re-checked on origin/dev @ ab4df99959. Paths are under `tldw_chatbook/` (`personas_screen.py` is `UI/Screens/personas_screen.py`; the `personas_*` and `persona_*` widgets are in `Widgets/Persona_Widgets/`). Capture names refer to the review's capture set; those copied into the QA folder are marked `evidence/`.

**What happens.** In the character editor, First message, Personality and System prompt show one line of text; Description shows two; Scenario, Post-history instructions and Creator notes show none, because the border uses the whole box, even when focused. This holds at every size, including 220x55, because the heights are fixed cell counts. A click drops the cursor mid-word into text the user cannot see. In a live run two edits were saved spliced into the old text ("The pa Rain hammers the portcullis.rty stands at the gates").

**Who it hurts.** In-depth character authors (J2) cannot read or check existing text. Imported SillyTavern cards rely on Scenario and Post-history, which are invisible. By contrast, the Library prompt editor gives each long field a label, a help line and a 4-6-row box.

**Evidence:**
- `personas_character_editor_widget.py:98-123`: First message, Personality and System prompt are `height: 3`; Description `height: 4`; Scenario, Post-history and Creator notes `height: 2`.
- `widget_defaults_self.tcss:2225-2248`: the bundled CSS mirrors the same heights.
- Captures: `review-rv-flows-2-j3-card-after-save-160x45` rows 21-25 (the corrupted saves); `evidence/roleplay-character-editor-scenario-zero-height-160x45.txt`; `review-rv-flows-j3-advanced-220x55` rows 32-40.

**Out of scope:** the editor's overall structure (one long scroll of fields with no sections, RP-016) is redesign work; the inverse problem in the persona editor (each long field fills the screen, RP-076) is separate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 120x36, 160x45 and 220x55, every long-text field in the character editor (First message, Description, Personality, System prompt, Scenario, Post-history instructions, Creator notes) shows at least 3 rows of text, both blurred and focused.
- [ ] #2 Focusing a long-text field never reduces the number of text rows it shows.
- [ ] #3 A click on empty space inside a long-text field places the cursor at the end of the text, not mid-word.
- [ ] #4 Editing the Scenario of a character that already has one, then saving, stores exactly the text shown in the field, with nothing spliced into hidden text.
- [ ] #5 A regression test asserts that every long-text field in the editor has a content region of at least 3 rows at 120x36 and 160x45, blurred and focused; it fails on the current code.
<!-- AC:END -->
