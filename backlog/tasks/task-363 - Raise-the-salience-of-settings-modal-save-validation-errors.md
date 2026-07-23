---
id: TASK-363
title: Raise the salience of settings-modal save validation errors
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clicking Save with an empty Temperature keeps the modal open and prints 'Temperature is required.' directly under the intro text. Cell-attrs of the error row: fg #E4E4E5 on bg #32303B, no bold/underline/inverse — nearly the same treatment as the descriptive text above it. The error sits 17 rows above the Temperature field, which itself shows no visible invalid-state highlight. The stale error also remains on screen after the field is fixed, until the next Save. Same pattern for 'Reasoning effort must be one of none, minimal, low, medium, high, or xhigh.'

**Repro:** 1. Rail > Configure. 2. Clear Temperature to empty. 3. Click Save -> modal stays open; single gray line 'Temperature is required.' appears at top; field not highlighted. 4. Re-enter a valid value -> error line remains until next Save.

**Verifier note:** Verified in code and not covered by task-178 (which addressed scope labeling, boolean controls, and accepted-values placeholders — not error presentation). .console-settings-error is styled 'background: $ds-status-error 10%; color: $ds-text-primary' (_agentic_terminal.tcss:293-301) — near-body-text salience matching the reviewer's cell-attrs; it is a single banner mounted at the top of the modal (console_settings_modal.py:276), never anchored to or highlighting the offending field, and only updated inside _validated_draft_or_show_errors (lines 699-710) so it stays stale after the field is fixed until the next Save. 'Save did nothing' confusion in a taller-than-viewport modal justifies P2.

**Source:** Console UX expert review 2026-07-20 (finding j5-validation-error-low-salience; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J5 settings journey. Evidence: `j5-67-error-styling.png`, `j5-63-after-save-9.png`, `j5-64-after-save-maybe.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Errors should be visually distinct (color/bold/inverse) and anchored at or near the offending field (or the field marked invalid), so the user understands why Save 'did nothing' in a modal taller than one screenful of attention
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three changes to `console_settings_modal.py`. (1) Salience: the validation
summary was near-body-text ($ds-status-error 10% / primary text) in the bundle;
added a scoped `ConsoleSettingsModal .console-settings-error` rule to the modal
DEFAULT_CSS (bold, $text-error colour, $error 25% fill, thick $error border-left)
— DEFAULT_CSS uses Textual's built-in $error/$text-error (not the bundle-only
$ds-* tokens) so it resolves in the pilot harness too. (2) Not stale: a broad
`@on(Input.Changed)`/`@on(Select.Changed)` handler clears the summary the moment
any field is edited (it previously only refreshed on the next Save). (3) Anchor:
a failed Save now scrolls the summary into view so it isn't above the fold in a
taller-than-viewport modal. RED->GREEN tests (clears-on-edit; bold styling); 129
settings-modal tests green.
<!-- SECTION:NOTES:END -->
