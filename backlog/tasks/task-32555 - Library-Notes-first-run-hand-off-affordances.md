---
id: TASK-32555
title: >-
  Library Notes: first-run hand-off — the Console card's rows have no button
  affordance, "Library tools are now available." means nothing to a first-timer,
  and the wizard's Provider step does not advance on Enter without a key
status: Done
assignee:
  - '@claude'
created_date: '2026-09-13 06:48'
updated_date: '2026-09-14 14:47'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor A, persona Jordan, the first-run path into Notes. Task-32245 fixed the destination (Blank note focused); these are the residuals on the way there.

1. The Console first-run card's "Write a note in Library" and "Set up provider" rows are visually identical text lines two rows apart; A's first click hit "Set up provider" and landed in Settings ▸ Providers (A 03, 04).
2. On the first note the toast "Library tools are now available." fires — it describes the rail graduation in the product's words, not Jordan's (A 05).
3. Wizard step 2 (Provider) does not advance on Enter without a key; Jordan wanted notes, not a provider (A 01, 02).

**Cause.** INFERRED.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Console first-run card's actions render as buttons with a shape-based focus cue
- [x] #2 The rail-graduation toast says what changed in the user's words, or is dropped
- [x] #3 Enter on the wizard Provider step with no key advances with a visible skip, or the step states why it cannot
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live on the fresh profile (captures `handoff-11-card-before.txt` / `handoff-11-card-focus-before.txt`: the card's actions are text lines whose only focus cue is bold+underline; `handoff-13-toast-before.txt`: the first note raises "Library tools are now available."; `handoff-12-wizard-before.txt`: Enter in the empty key field does nothing).
2. RED pins: the card actions render as bordered buttons with an outline focus cue under the real Console sheet; STARTER->GRADUATED raises no toast (existing pins in `test_library_shell.py` / `test_library_entry_compose_once.py` updated to the new truth); Enter in an empty `#setup-provider-api-key` advances past the Provider step and the Summary's Provider row reads not configured.
3. Fix: drop `compact=True` from the card's action buttons (Textual's compact class forces `border: none !important`), give `.console-setup-modal-action` a round border and a heavy left/right outline on focus in the Console sheet; delete the graduation notify; the wizard's key-field Enter with no key clears the provider choice and advances (the key hint says so).
4. Bundle sync + byte budget, GREEN, live re-verify (`handoff-11-card-focus.txt`, `handoff-12-wizard-skip.txt`, `handoff-13-no-toast.txt`), guide pages `console.md` / `library.md` / `First_Run_Setup.md` + stamps.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Card actions (AC#1).** The three actions already were `Button`s -- with
`compact=True`, which Textual renders through a class carrying
`border: none !important`, so no stylesheet could give them an edge. Dropping
`compact` and giving `.console-setup-modal-action` `border: round $ds-grid-line`
plus `outline-left`/`outline-right: heavy $ds-focus-accent` on `:focus` makes
the focus cue a change of SHAPE, not of text style. The rule lives in the
Console component sheet and its generated screen sheet (bundle sync green, boot
CSS byte budget green -- the Console sheet is not the boot bundle).

**Graduation toast (AC#2, "or is dropped").** Rewording it would still have
described the rail. `_apply_graduation_notice` and its call are deleted; the
rail growing from the compact Get started list to the full rail is the
evidence. The pins that asserted it FIRED (two in
`test_library_crit8_polish_shell.py`) are replaced by one parametrised test
asserting silence from all three lifecycles they covered.

**Wizard Enter (AC#3).** `_advance_on_input_submit` returned unconditionally
for `#setup-provider-api-key`, so Enter with no key did nothing at all. It now
returns only when a key is typed; an empty Enter calls
`ProviderStep.skip_without_key()` and falls through to `action_next()`. The key
field's hint says so. One guard beyond the plan: a provider already ready
*without* a typed key (an exported env var, a local server) is not cleared --
`commit()` would have staged it, and the hint never offers the skip in that
state.

Modified: `Widgets/Console/console_setup_modal.py`,
`css/components/_agentic_terminal.tcss`, `css/screen_agentic_console.tcss`,
`UI/Screens/library_screen.py`, `UI/Wizards/FirstRunSetupWizard.py`,
`Tests/UI/test_library_notes_w4_console_handoff.py`,
`Tests/UI/test_library_crit8_polish_shell.py`,
`Tests/UI/test_library_shell.py`, `Tests/UI/test_library_entry_compose_once.py`,
`Tests/UI/test_first_run_wizard_live_contract.py`,
`Docs/User_Guide/console.md`, `Docs/User_Guide/library.md`,
`Docs/User_Guide/First_Run_Setup.md`.

Live captures: `handoff-10-console-card`, `11-card-focus`,
`12a-wizard-key-hint`, `12-wizard-skip`, `12b-wizard-summary`,
`13-no-toast`.

Known, pre-existing and NOT introduced here: Tab does not cycle between the
card's three actions (focus stays on the first). The affordance and focus cue
are correct; the tab order was the same before this change, when all three were
text lines. Worth a rider.
<!-- SECTION:NOTES:END -->
