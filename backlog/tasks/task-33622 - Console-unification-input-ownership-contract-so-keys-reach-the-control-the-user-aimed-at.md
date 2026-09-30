---
id: TASK-33622
title: >-
  Console unification: input-ownership contract so keys reach the control the
  user aimed at
status: To Do
assignee: []
created_date: '2026-09-30 03:03'
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
Why: in the Console, keystrokes often go somewhere other than the control the user aimed at. The screen-level key handler captures Enter from every composer descendant, so Enter on a focused Menu or Dictate button sends the draft to the provider. Bare single-letter screen bindings ('y', and the transcript's c/e/f/r/s/n) fire from any non-text control, so ordinary typing opens Trace or regenerates a reply. The transcript's Esc handler always stops the event, so Esc never returns to the composer. Tab activation binds the composer asynchronously, so a prompt typed just after a tab switch is sent to the previous tab's chat. The Terminal center never takes focus, so shell input goes to the model. Dialogs return focus to their opener chip. A click outside a menu to dismiss it also activates the control underneath. All of these share one cause: no contract says who owns a key.

What unifying means here: one input-ownership contract, written once and pinned with Pilot tests. (1) A focused control owns Enter and Space. (2) Unbound printable keys from a non-text control go to the composer. (3) Single-letter commands work only in explicitly announced list modes. (4) Session activation binds the composer, tab highlight and send target synchronously, and submit checks that they match the visible tab. (5) The first outside press only dismisses. (6) Every dialog close restores focus to where typing will land. This umbrella is the parent of the theme's P0/P1 fixes (G1-07, G1-06, G4-12, G4-13, GAP1-01, GAP4-05, G2-07, GAP4-04, GAP4-06). Tab-switch timing work should coordinate with TASK-26834.

Evidence: qa/console-ux-review-2026-09-29/report.md (theme 'Input ownership') and qa/console-ux-review-2026-09-29/findings.md / findings.json.

Related P2/P3 in the ledger: GAP1-02, GAP5-11, G3-15, G1-31, G4-43, GAP1-09, G1-37, G4-46, G4-47, G4-44.

Source: Console UX review 2026-09-29 — qa/console-ux-review-2026-09-29/report.md (themes) and findings.md (ledger).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Enter or Space on any focused Console button activates that button and never sends the composer draft, verified by a Pilot test over every focusable composer and header control
- [ ] #2 Printable keys typed while a non-text control has focus land in the composer, and no single-letter command fires outside an announced list mode, verified by Pilot tests
- [ ] #3 A prompt submitted immediately after activating another tab, including during a live run, is sent to the visibly active tab's chat, and submit refuses when the bound session and the visible tab disagree
- [ ] #4 Esc from the transcript returns focus to the composer, and closing any Console dialog or menu leaves focus where the next keystroke will be typed
- [ ] #5 A first click outside an open menu or popover only dismisses it and does not activate the control underneath
- [ ] #6 All P0/P1 child tasks under this umbrella are Done
<!-- AC:END -->
