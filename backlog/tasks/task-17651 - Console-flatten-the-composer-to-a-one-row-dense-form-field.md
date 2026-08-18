---
id: TASK-17651
title: 'Console: flatten the composer to a one-row dense-form field'
status: To Do
assignee: []
created_date: '2026-08-17'
labels:
  - console
  - ux
dependencies:
  - task-17650
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The composer spends 4 chrome rows on every draft height: `COMPOSER_CHROME_ROWS = 4` (border 2 + padding 2) is hard-coded in ConsoleComposerBar and re-asserted inline on every draft repaint, so the widget renders 5-8 total rows to show 1-4 draft rows. The 2026-08-17 audit also proved the composer's focus-border CSS is dead code: `frame_console_region()` writes the border as an inline style, which outranks every stylesheet rule (the composer renders `#6f7782` even while carrying `console-composer-focused`).

Flatten the composer to the task-1586 dense-form convention: a one-column left edge at rest, thick `$accent` edge plus focus background when focused (three concurrent signals, no dimensional change on focus). The draft keeps its existing 1-4 row auto-grow; total composer height becomes 1-4 rows instead of 5-8. Below the workspace grid's closing border, the control deck (composer, chips, footer) reads as dense-form rows rather than a framed region — the workbench frame grammar closes at the grid.

Owner decisions (2026-08-17): keep ALL send affordances (disabled-reason banner, Send, Dictate); keep the 4-row draft cap (raising it is a separate follow-up); the Composer collapse control is KEPT and repurposed as a same-height content swap to the run-status variant (status + Stop + Expand) — it is no longer a row-saving lever, and its height-economics tests change accordingly.

Also in scope (reallocated from TASK-17650 during implementation review): consolidate the FOUR stacked separator rows between the last transcript line and the composer content — `#console-native-transcript`'s own border, the `#console-transcript-region` inline frame, the `#console-workspace-grid` inline frame, and the composer's top border — down to ONE separator. The transcript's border rows are its keyboard-focus affordance (`:focus` recolors them; pinned by `test_console_transcript_focus_uses_stable_border_geometry`), so removing them requires a replacement focus treatment that is dimensionally stable and non-obscuring, designed together with the composer's dense-form focus edge. Compact mode's existing border-drop (TASK-2154.1) is the precedent for the transcript side.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The default composer renders exactly 1 row with all current children on the content row (Composer collapse control, Menu, draft, disabled-reason banner, Send, Dictate) and grows with the draft to at most 4 rows
- [ ] #2 The focus treatment follows the dense-form convention, causes no dimensional change, and demonstrably renders on the running screen (i.e. is not overridden by inline frame styles)
- [ ] #3 The composer collapse control still exists and swaps to the run-status variant at the same height; updated tests encode the new economics (collapse no longer promises extra transcript rows)
- [ ] #4 Exactly one separator row renders between the last transcript content line and the composer content row (transcript border, region frame, grid frame, and composer top border consolidated), and the transcript keeps a visible, dimensionally-stable keyboard-focus treatment at all sizes
- [ ] #5 The transcript gains at least 7 more rows at 150x44 versus the post-TASK-17650 baseline (4 composer chrome + 2 transcript border + 1 duplicate frame row)
- [ ] #6 All affected geometry pins are updated to the new contract (composer-collapse suite, internals-decomposition geometry blocks, non-obscuring-focus composer AND transcript stable-border rules), using bundle-loading harnesses wherever geometry is asserted
- [ ] #7 The setup-blocked/first-run state renders correctly with the flattened composer (the disabled-reason banner is the widest single-row competitor) and the draft's usable width at 150 columns does not regress
- [ ] #8 User Guide Console page updated
<!-- AC:END -->
