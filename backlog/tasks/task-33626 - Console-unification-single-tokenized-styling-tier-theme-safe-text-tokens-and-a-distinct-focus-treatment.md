---
id: TASK-33626
title: >-
  Console unification: single tokenized styling tier, theme-safe text tokens and
  a distinct focus treatment
status: To Do
assignee: []
created_date: '2026-09-30 03:04'
labels:
  - console
  - ux-review-2026-09-29
  - unification
dependencies: []
references:
  - qa/console-ux-review-2026-09-29/report.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: Console styling lives in two tiers. The .tcss tier is tokenized. The Python-embedded DEFAULT_CSS is not: it holds 52 named-colour literals and 792 raw integers, so about 20 Console modals hard-code a black background and become black boxes in light themes (P1). Status tokens bind to raw hues instead of polarity-aware text colours, so blocked reasons and recovery warnings drop to about 1.3:1 in light themes. In textual-dark the accent colour equals the warning colour. Keyboard focus, the active tab, the selected message and region focus all share one style, and the real focus change measures 1.03:1. Several contrast pairs on the default screen fail: the focused placeholder, primary button labels, the header switch off state (P1), headings, the transcript scrollbar, and an unpainted default-background slab. Modals, buttons and inputs each come in several unrelated styles.

What unifying means here: all Console widget CSS lives in the tokenized .tcss tier. Text-colour tokens bind to Textual's polarity-aware text variables, and raw hues are allowed only in borders and tints. One focus-only treatment changes shape, reaches at least 3:1, and stays distinct from active and selected. Shared modal frame, modal action-row and button-tier classes replace the per-modal styles. A Textual-aware CSS lint and a theme-matrix contrast check run in CI. This follows ADR-011 (workbench UI system) and DESIGN.md. This umbrella is the parent of the theme's P0/P1 fixes (GAP3-02, G2-01).

Evidence: qa/console-ux-review-2026-09-29/report.md (theme 'Two styling tiers') and qa/console-ux-review-2026-09-29/findings.md / findings.json.

Related P2/P3 in the ledger: GAP3-01, GAP3-03, GAP3-10, GAP3-11, G4-60, G4-58, G4-54, G4-14, G4-39, G4-40, G4-41, G1-26, G1-28, G3-41, G4-56, G4-57, G4-59, GAP3-13, GAP3-08, GAP5-17, GAP3-16, G3-28, GAP3-15, G1-35.

Source: Console UX review 2026-09-29 — qa/console-ux-review-2026-09-29/report.md (themes) and findings.md (ledger).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No Console widget or modal styling uses named colours, hex values or raw status hues for text, and a CSS lint in CI fails when one is introduced
- [ ] #2 Across textual-dark, textual-light, paper_light, solarized_light and a high-contrast theme, Console text measures at least 4.5:1 and control boundaries at least 3:1, and no cell falls back to the terminal default background; a theme-matrix check in CI enforces this
- [ ] #3 Keyboard focus has one treatment that changes shape and measures at least 3:1 against its unfocused state, and it is visually distinct from the active tab and the selected message
- [ ] #4 Every Console modal uses a shared frame and a shared action-row layout, so action order, alignment, primary styling and dismiss verb are the same across modals
- [ ] #5 Every on/off control in the Console shows its state in words or shape, not by colour alone
- [ ] #6 All P0/P1 child tasks under this umbrella are Done
<!-- AC:END -->
