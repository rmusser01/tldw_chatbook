---
id: TASK-33785
title: >-
  Roleplay: lore and dictionary Settings tabs cannot scroll, so Save settings,
  Export and Enabled are unreachable
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
Found by the 2026-10-01 Roleplay NN/g + HCI review: finding RP-005 (P0, severity 3, effort S). Report: `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md` (section 5 and Appendix A), with the machine-readable entry in `findings.json` beside it. The review ran on origin/dev @ 84247cb843 (worktree `.worktrees/roleplay-library-ux`).

Line refs below were re-checked on origin/dev @ ab4df99959. Paths are under `tldw_chatbook/` (`personas_screen.py` is `UI/Screens/personas_screen.py`; the `personas_*` and `persona_*` widgets are in `Widgets/Persona_Widgets/`). Capture names refer to the review's capture set; those copied into the QA folder are marked `evidence/`.

**What happens.** The Settings tab of a lore book or dictionary is clipped and the mouse wheel does nothing, so Save settings, the Export buttons and the Enabled switch never appear. Dictionaries are affected at 120x36 and 160x45 (they fit at 220x55); Lore at 120x36, 160x45 and 220x55. New dictionary and New lore deliberately land on this tab so the item can be renamed, and the rename can only be saved by pressing Tab five times blind and then Enter. Keyboard focus moves onto controls that are off-screen.

**Who it hurts.** World-info builders (J3) cannot save scan depth, token budget, strategy, recursion or a rename by mouse at the medium sizes. The User Guide's own step ("Type a real name and click Save settings") cannot be done.

**Evidence:**
- `personas_dictionary_detail.py:269-305` and `personas_lore_detail.py:201-232`: the Settings tab panes are plain containers; only Entries is wrapped in a scroll container.
- `personas_lore_detail.py:91-94`, `personas_dictionary_detail.py:138-141` and `personas_screen.py:1634`: the detail widgets are 1fr inside `VerticalScroll#personas-detail-stack`, so the outer scroll never overflows and the tab content is clipped.
- `personas_screen.py:8044-8048, 8242-8248`: New dictionary and New lore land on Settings "to rename immediately".
- Captures: `review-vf-nng-a-lore-settings-wheel-no-save-160x45`; `review-vf-nng-b-dict-settings-clipped-160x45`; `review-rv-nng-b-lore-settings-tab-220x55`; `review-rv-flows-2-j5-new-dictionary-160x45`.

**Out of scope:** labels for the Settings fields (RP-003); where Try it lives and how much height entries get (RP-006, redesign decision D2); the other scattered per-item actions (RP-020). The slot leak in TASK-33781 can also split this tab, so measure in a session where no other heavy view was opened first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 120x36 and 160x45 (and for Lore also at 220x55), every control on the Settings tab of a lore book and of a dictionary can be brought into view with the mouse wheel.
- [ ] #2 Save settings and the Export buttons stay visible while the Settings fields are scrolled.
- [ ] #3 When Tab moves focus to a Settings control, that control is scrolled into view.
- [ ] #4 At 120x36 and 160x45, after New dictionary or New lore, the user can type a name and save it using the mouse alone.
- [ ] #5 Regression tests assert at 120x36 and 160x45 that Save settings lies inside the visible region of both Settings tabs and that the Enabled switch can be scrolled into view; they fail on the current code.
<!-- AC:END -->
