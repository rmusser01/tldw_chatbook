---
id: TASK-33783
title: >-
  Roleplay: lore and dictionary entry Delete and character Detach act at once,
  with no confirmation or undo
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
Found by the 2026-10-01 Roleplay NN/g + HCI review: finding RP-014 (P0, severity 4, effort S; this task is the confirmation half). Report: `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md` (section 5 and Appendix A), with the machine-readable entry in `findings.json` beside it. The review ran on origin/dev @ 84247cb843 (worktree `.worktrees/roleplay-library-ux`).

Line refs below were re-checked on origin/dev @ ab4df99959. Paths are under `tldw_chatbook/` (`personas_screen.py` is `UI/Screens/personas_screen.py`; the `personas_*` and `persona_*` widgets are in `Widgets/Persona_Widgets/`). Capture names refer to the review's capture set; those copied into the QA folder are marked `evidence/`.

**What happens.** On the Entries tab of a lore book or dictionary, Delete removes the selected entry the moment it is pressed: there is no confirmation and no success message, and the status line is simply cleared. Delete has the same style as Add and Update and sits in the same row. Detaching a lore book or a dictionary from a character is just as instant; for a lorebook that arrived inside an imported card, that attachment is the only copy. DESIGN.md:127 ("Confirm irreversible actions") is the governing rule. The Library already follows it: Prompts keeps Delete under More actions with danger styling, and Notes uses Danger > Delete with an inline confirmation.

**Who it hurts.** World-info builders (J3). Combined with the row-1 reset tracked in TASK-33782, a second press deletes an entry the user never saw; on its own, a single mis-press is still unrecoverable for lore, which has no version history.

**Evidence:**
- `personas_lore_detail.py:174-199, 612-619` and `personas_dictionary_detail.py:242-266`: Add, Update, Delete and Move share one style; Delete posts directly.
- `personas_screen.py:6510-6518` (dictionary) and `:6673-6683` (lore): the handlers delete with no dialog and no success message.
- `personas_screen.py:6321-6352` (lore book) and `:6170-6200` (dictionary), via `world_book_manager.py:864-893`: Detach from a character is immediate.
- Captures: `review-rv-nng-a-lore-entry-form-before-delete-160x45` row 27; `review-rv-nng-a-lore-entry-deleted-no-confirm-160x45` rows 24-26.

**Out of scope:** soft delete or version history for lore entries (a data-model change the report lists as optional), and what attaching a lore book or dictionary to a character should mean (the separate world-info semantics work).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A lore or dictionary entry cannot be deleted by one unconfirmed press: either an inline confirmation naming the entry appears first, or the deletion can be undone from its success message for at least 8 seconds.
- [ ] #2 A completed Delete says which entry was removed, by name (for example: Deleted "Foxglove Vale Pact").
- [ ] #3 Delete is styled as a destructive action and is visually separated from Add and Update.
- [ ] #4 Detaching a lore book or a dictionary from a character asks for confirmation naming the item. When the attachment came from an imported card and no other copy exists, the confirmation says so.
- [ ] #5 Cancelling a Delete or Detach confirmation changes nothing: the entry or attachment remains, and the selection stays where it was.
- [ ] #6 At 120x36 and 160x45, the confirmation (or Undo) and its buttons are fully visible without scrolling.
- [ ] #7 Regression tests drive the real Roleplay screen: Delete then Cancel leaves the entry count unchanged; Delete then confirm removes exactly the named entry; Detach then Cancel keeps the attachment. They fail on the current code.
<!-- AC:END -->
