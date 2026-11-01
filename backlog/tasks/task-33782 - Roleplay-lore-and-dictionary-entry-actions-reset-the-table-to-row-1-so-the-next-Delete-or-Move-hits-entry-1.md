---
id: TASK-33782
title: >-
  Roleplay: lore and dictionary entry actions reset the table to row 1, so the
  next Delete or Move hits entry 1
status: To Do
assignee: []
created_date: '2026-10-02 04:55'
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
Found by the 2026-10-01 Roleplay NN/g + HCI review: finding RP-014 (P0, severity 4, effort S; this task is the selection half). Report: `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md` (section 5 and Appendix A), with the machine-readable entry in `findings.json` beside it. The review ran on origin/dev @ 84247cb843 (worktree `.worktrees/roleplay-library-ux`).

Line refs below were re-checked on origin/dev @ ab4df99959. Paths are under `tldw_chatbook/` (`personas_screen.py` is `UI/Screens/personas_screen.py`; the `personas_*` and `persona_*` widgets are in `Widgets/Persona_Widgets/`). Capture names refer to the review's capture set; those copied into the QA folder are marked `evidence/`.

**What happens.** Every entry action on a lore book's or dictionary's Entries tab (Add, Update, Delete, Move up, Move down) rebuilds the entry table, which puts the cursor back on row 1 and loads row 1 into the form. The next Delete, Move or Update therefore acts on entry #1, which at realistic volume is far off-screen; the only cue is the form text changing. Reproduced live at 160x45 in a 250-entry lore book: with #199 selected, pressing Delete twice deleted #199 and then entry #1, 198 rows away. Pressing Move down twice on #200 moved #200 once and then moved entry #1. It also reproduces on entry #3, so volume only hides it. After Add, the form jumps to entry #1 instead of the new entry (the RP-040 symptom has the same cause).

**Who it hurts.** World-info builders (J3) lose or reorder entries they never selected, without seeing it. Lore entries have no version history, so a wrong delete is permanent; for a dictionary the only recovery is reverting the whole dictionary.

**Evidence:**
- `personas_lore_detail.py:309-318, 405-414, 572-581`: `update_entries()` clears the table (cursor to (0,0), scroll to 0) and re-adds every row. The selected entry is simply the cursor row, and a row highlight refills the form.
- `personas_screen.py:6578-6633` (lore operations) and `:5799-5835` (dictionaries): every operation reloads entries without restoring the selection. Dictionary entry ids are positional (`:5800`). The rail already re-anchors its own selection after a reload (`mark_active_row`, `:6608`).
- `personas_dictionary_detail.py:390-399`: dictionaries share the clear-and-reload path (code-verified, not driven live).
- Captures: `review-rv-gap3-lore-after-delete2-160x45` rows 20-25; `review-vf-gap3-lore-movedown1-160x45`; `review-vf-gap3-lore-movedown2-160x45`; `review-rv-gap3-lore-after-update-200-160x45`; `review-rv-gap3-lore-after-add-160x45`.

**Out of scope:** confirmation, feedback and undo for Delete and Detach (the other half of RP-014); slow one-step reordering (RP-096); entry search and filtering (RP-056).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After Update, Move up or Move down, the entry acted on stays selected, stays loaded in the form and stays in view. The cursor never jumps to row 1 unless row 1 is that entry.
- [ ] #2 After Delete, the selection moves to the entry that took the deleted entry's place (or to the new last entry), never to row 1.
- [ ] #3 After Add, the new entry is selected, loaded in the form and scrolled into view.
- [ ] #4 In a 250-entry lore book, selecting entry #200 and pressing Delete twice removes #200 and then the entry that followed it; entry #1 is still present.
- [ ] #5 In a 250-entry lore book, selecting entry #200 and pressing Move down twice moves that same entry to position #202; entry #1 keeps position #1.
- [ ] #6 The Delete-twice and Move-twice outcomes above also hold for a dictionary with 250 entries.
- [ ] #7 An entry action pressed while the previous action's reload is still running never acts on an entry other than the one shown as selected.
- [ ] #8 At 120x36 and 160x45, the selected entry's row is visible in the table after every action (measured in a session where no other heavy Roleplay view was opened first, so the TASK-33781 slot leak does not distort the reading).
- [ ] #9 Regression tests drive the real Roleplay screen through the Delete-twice and Move-twice sequences for both lore and dictionaries; they fail on the current code.
<!-- AC:END -->
