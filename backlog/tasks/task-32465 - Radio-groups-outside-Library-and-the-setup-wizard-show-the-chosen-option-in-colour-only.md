---
id: TASK-32465
title: >-
  Radio groups outside Library and the setup wizard show the chosen option in
  colour only
status: To Do
assignee: []
created_date: '2026-09-14 16:42'
labels:
  - ux
  - accessibility
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every radio group in the app except two distinguishes its selected option by colour alone, which is invisible in a monochrome terminal, in any text capture, to a colour-blind reader, and in the plain-text layer a screen reader sees — WCAG 1.4.1 (use of colour).

Stock `textual.widgets.RadioButton` sets `BUTTON_INNER = "●"` and renders that same glyph for BOTH the on and off states (`textual/widgets/_toggle_button.py`), varying only the style. Two surfaces already fix this by overriding the `_button` property per state — `ConsoleAccessRadioButton` (task-25831) and `SetupRadioButton` (task-1497), both now reading `LIBRARY_GLYPH_RADIO_SELECTED`/`LIBRARY_GLYPH_RADIO_UNSELECTED` from `Library/library_shell_state.py` (task-32464). Everywhere else still paints ● on every option: roughly eight surfaces, including `console_capture_policy_dialog.py` (~15 radios), `console_exchange_export_dialog.py`, `trace_export_dialog.py`, `artifact_share_dialog.py`, `conversation_selection_dialog.py`, `chat_question_card.py`, `ChatbookCreationWizard.py` and `ChatbookImportWizard.py`.

Found by the task-32464 review, which measured the off state of one such radio at 1.42:1 against its track. It also makes the Library glyph legend's radio row true of only two surfaces — `Docs/User_Guide/library.md` currently documents the gap rather than claiming otherwise.

The shape this wants is ONE shared structural radio subclass that both existing overrides collapse into, not a third copy of the same six lines — which is exactly why task-32464 left it out of its own commit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A radio group's selected option is distinguishable without colour on every surface that renders one
- [ ] #2 The two existing per-state overrides and every converted surface share one implementation rather than repeating the glyph logic
- [ ] #3 The shared implementation reads the glyph pair from the existing LIBRARY_GLYPH_RADIO_* constants, with no new literals
- [ ] #4 A test fails if any radio surface regresses to the stock colour-only button
- [ ] #5 The Library glyph legend drops its 'Not yet everywhere' caveat once the sweep is complete
<!-- AC:END -->
