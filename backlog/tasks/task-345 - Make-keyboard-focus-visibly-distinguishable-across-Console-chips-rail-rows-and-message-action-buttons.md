---
id: TASK-345
title: >-
  Make keyboard focus visibly distinguishable across Console chips, rail rows
  and message action buttons
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-21 15:18'
labels:
  - console
  - ux
  - keyboard
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured focus styles: rail buttons ('New conversation', section ▾ toggles, conversation rows) bg #1e1e1e -> #272727 on a #242f38 panel; 'Switch' chip #1e1e1e -> #1e262d; palette selected row #141f27 -> #1e1e1e — all ~1.1:1 contrast. Several keyboard stops changed literally zero cells: Enter-to-show-actions on a selected message (focus onto Copy button, no visible change), and 8 consecutive Tab presses from the composer (no visible change at all). Consequence observed live: after Enter on a selected message I pressed Tab 3x + Enter expecting the ♻ button and instead activated the invisible-focused 'Save as...' button, opening an unintended modal.

Also observed independently in J3 attachments as `j3-weak-focus-indicator-composer-buttons`: Tab focus on composer buttons is a barely visible background shift.

**Repro:** 1) F6 to rail, Tab through items and compare cell attrs (bg deltas ~#090909). 2) Select a message, press Enter (no visible change), Tab 3x (no visible change), Enter — the 'Save as...' modal opens unexpectedly. 3) Focus composer, press Tab 8x — zero visible change anywhere.

**Verifier note:** Partially code-confirmed: many Console controls (.console-rail-collapse-button, .console-switcher-result, handle buttons) have NO :focus rule → Textual default subtle bg shift; the DS focus vocabulary itself is quiet ($ds-focus-bg = $ds-surface-raised = $surface, _variables.tcss:18). Caveat: several controls DO declare bold-underline focus rules (.console-control-chip:focus, .console-transcript-action-button:focus, .console-rail-section-toggle:focus at _agentic_terminal.tcss:1961/2361/2856), which the journey's cell scans say did not render — that discrepancy is unresolved, hence medium confidence. rail-layout-quiet-focus (task-149) settled only the rail-body boundary, not control focus contrast; no open backlog task covers it. The observed accidental 'Save as…' activation makes P1 defensible; keep P1 as the umbrella focus-visibility finding.

**Source:** Console UX expert review 2026-07-20 (finding j6-chip-focus-imperceptible-accidental-activation, j3-weak-focus-indicator-composer-buttons; P1, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J6 keyboard-only/small-terminal journey. Evidence: `j6-a08-rail-tab3.png`, `j6-a08-rail-tab10.png`, `j6-a25-enter-actions.png`, `j6-a25-tab3-recycle.png`, `j6-a26-mystery-state.png`, `j6-a31-tab-from-composer.png`, `j3-80-tab-cycle.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused controls get an unmistakable indicator (inverse video, accent bg, or bracket/underline marker) meeting a ~3:1 contrast ratio
- [x] #2 Every Tab press produces a visible focus change so Enter's target is always predictable
<!-- AC:END -->

## Implementation Notes

Root cause was the design system itself: `$ds-focus-bg` aliased
`$ds-surface-raised` = `$surface`, so the non-obscuring focus contract's
own mechanism (background + bold underline) rendered a ~1.1:1 shift on
every conforming control. Fix honors the settled contract rather than
fighting it: the token now carries a raised steel-blue (#51677e, ~3:1
against the dark control surfaces) — one change that fixes every
conforming control app-wide — plus contract-style `:focus` rules for the
Console controls that had NONE (rail collapse handles, switcher results,
rail conversation rows, rail header/action buttons). No accent, no
reverse; all contract tests pass unchanged (228 across the focus/contract
suites). Live-verified with the review's own cell-attribute method: the
focused rail control renders the raised band + bold underline where the
review measured a #090909 delta. Files: `css/core/_variables.tcss`,
`css/components/_agentic_terminal.tcss`, bundle rebuilt.
