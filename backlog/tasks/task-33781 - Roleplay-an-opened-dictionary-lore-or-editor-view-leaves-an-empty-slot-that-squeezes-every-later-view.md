---
id: TASK-33781
title: >-
  Roleplay: an opened dictionary, lore or editor view leaves an empty slot that
  squeezes every later view
status: To Do
assignee: []
created_date: '2026-10-02 04:54'
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
Found by the 2026-10-01 Roleplay NN/g + HCI review: finding RP-001 (P0, severity 4, effort S). Report: `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md` (section 5 and Appendix A), with the machine-readable entry in `findings.json` beside it. The review ran on origin/dev @ 84247cb843 (worktree `.worktrees/roleplay-library-ux`).

Line refs below were re-checked on origin/dev @ ab4df99959. Paths are under `tldw_chatbook/` (`personas_screen.py` is `UI/Screens/personas_screen.py`; the `personas_*` and `persona_*` widgets are in `Widgets/Persona_Widgets/`). Capture names refer to the review's capture set; those copied into the QA folder are marked `evidence/`.

**What happens.** Roleplay mounts its four heavy centre views (character editor, persona editor, dictionary detail, lore detail) on first use. Each view's slot stays laid out, empty, after you move to another view, and keeps an equal share of the height. After a normal tour of the four jobs, the character editor shrinks to a 2-row window at 120x36 (5-9 rows at 160x45) with blank rows below it, and lore and dictionary tables show 0-1 rows under a blank band. One Lore visit is enough. The same leak splits the dictionary Settings tab and the conversation-transcript view, and it lasts until the screen is rebuilt.

**Who it hurts.** Character authors (J2) and world-info builders (J3) work through a peephole while most of the pane is blank. The blank space reads as a rendering fault, so nobody thinks to scroll the tiny inner region.

**Origin.** A regression from TASK-31215 (mount heavy centre views on first use; ADR-115, `backlog/decisions/115-personas-demand-mounted-center-views.md`). First-use mounting is intended and stays; only the leaked layout share is the defect.

**Evidence:**
- `personas_screen.py:2030`: `_ensure_center_view` sets the slot visible after the first mount, and nothing hides a slot again.
- `personas_screen.py` `_show_center` (from `:15909`) toggles only the `_CENTER_VIEW_IDS` roots, never the `_DEMAND_CENTER_VIEW_SLOTS` (`:643-654`, `:665-670`).
- `personas_screen.py:1634-1686`: the slots are bare Verticals (default height 1fr) inside `VerticalScroll#personas-detail-stack`, so each opened slot keeps an equal share.
- Captures: `review-rv-layout-2-editor-porthole-after-dict-lore-120x36` (editor rows 15-16, rows 21-32 blank); `review-vf-layout-editor-after-lore-only-120x36`; `review-rv-layout-editor-collapsed-after-dict-lore-160x45`. Baseline without the Lore visit: `evidence/roleplay-character-editor-160x45.txt`.

**Why first.** Several of the review's Dictionaries and Lore measurements are inflated by this leak (`metrics.md` section C; capture-report items 8 and 13). Land it before re-measuring any Roleplay layout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After each of the four heavy views (character editor, persona editor, dictionary detail, lore detail) has been opened once, in any order, the view shown next gets the full height of the work pane, and no blank band from an earlier view remains.
- [ ] #2 At 120x36 and 160x45, after Lore and Dictionaries have been visited first, the character editor fills the detail stack (its height is at least the stack height minus 2 rows), the same as when it is opened first.
- [ ] #3 At 120x36 and 160x45, after the character editor has been opened first, the lore and dictionary detail views (including their Settings tabs) and the conversation-transcript view fill the work pane.
- [ ] #4 Each heavy view is still mounted only on first use; the ADR-115 behaviour is unchanged.
- [ ] #5 Regression tests mount all four heavy views, then open each view at 120x36 and 160x45 and assert the geometry above; they fail on the current code.
<!-- AC:END -->
