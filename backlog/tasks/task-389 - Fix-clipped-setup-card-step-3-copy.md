---
id: TASK-389
title: Fix clipped setup-card step 3 copy
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The card renders '3. ○ Send your first message  Composer unlocks after      ' — the source constant is 'Composer unlocks after setup' (console_onboarding_state.py CONSOLE_SETUP_STEP_THREE_DETAIL), so the final word is truncated even though ~6 blank cells remain before the card border, and no ellipsis marks the cut.

**Repro:** Boot with a bare home at 2050x1240 -> read row 3 of the Get-started card.

**Verifier note:** Confirmed by arithmetic + capture, but NOT a regression — it never rendered fully. The step line is 59 cells ('3. ○ Send your first message  Composer unlocks after setup'; _step_text, console_setup_modal.py:438-444; constant console_onboarding_state.py:25), while .console-setup-modal-card gives 62 − 2 border − 4 padding = 56 content cells (tcss:4463-4471); 'setup' word-wraps to a second line that .console-setup-step height:1 (tcss:4417) hides — clipped at a word boundary with blank cells left and no ellipsis, exactly as captured (j1-01/j1-42). Timeline: the modal card shipped 2026-07-04 (c50f1dee9/169a6ba04); the 'Composer unlocks after setup' copy landed 2026-07-11 (76a8b1e35, the setup-card-honest-steps remediation) into the too-narrow card, so the shipped remediation itself never displayed its last word. Trivial fix (widen card or shorten detail); P3 cosmetic but on the very first screen users see.

**Source:** Console UX expert review 2026-07-20 (finding j1-card-copy-truncated; P3, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J1 new-user cold start journey. Evidence: `j1-01-initial-console.png`, `j1-42-fresh-boot.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The full sentence fits or wraps
- [x] #2 If truncation is unavoidable, show an ellipsis
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Served-app-reproduced at 2050x1240: the first-run card row 3 read "Composer
unlocks after" with "setup" clipped (wrapped to a hidden 2nd line, ~6 blank
cells, no ellipsis). The honest step-3 line is 59 cells but `.console-setup-modal-card`
width:62 gave only 56 content (−2 border −4 padding). AC#1: widened the card to
width:66 (60 content) so the full sentence fits. AC#2: `.console-setup-step` now
`text-wrap: nowrap; text-overflow: ellipsis` so a narrow terminal that caps the
card (max-width:90%) marks the overflow instead of silently clipping. Served-app-
VERIFIED (row 40: "…Composer unlocks after setup"). Regression test ties the card
width to the actual _step_text length + asserts the ellipsis.
<!-- SECTION:NOTES:END -->
