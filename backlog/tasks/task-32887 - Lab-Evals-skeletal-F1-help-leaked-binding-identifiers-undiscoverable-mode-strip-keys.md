---
id: TASK-32887
title: >-
  Lab Evals: skeletal F1 help, leaked binding identifiers, undiscoverable
  mode-strip keys
status: Done
assignee: []
created_date: '2026-09-21 22:31'
updated_date: '2026-09-21 23:39'
labels:
  - ux
  - evals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
HCI review A5+A6 (MEDIUM). The Evals screen's F1 help contains only two lines - 'left_square_bracket: Prev mode' / 'right_square_bracket: Next mode' - raw binding names leaked as copy, nothing about subjects/depths/models/estimates. Mode chips ignore arrow keys; the real mechanism ([ / ] then Enter) is advertised only as cryptic footer copy; Escape is deliberately unbound Lab-wide so nothing closes with the keyboard's most reflexive key.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 F1 help renders key glyphs ('[' / ']') and lists the screen's real bindings plus a short 'what this screen does' section covering skill evals
- [x] #2 Left/Right arrows move mode-chip focus (brackets remain as alias); footer copy updated
- [x] #3 No help surface anywhere shows raw binding identifiers like left_square_bracket
<!-- AC:END -->

## Implementation Notes

- `_bindings_to_shortcuts` (app.py) renders display glyphs via Textual's `KEY_DISPLAY_ALIASES` + a supplement for the repo's spelled-out punctuation keys (left/right_square_bracket, comma, period, slash, minus, equal) — no help surface can leak identifiers like `left_square_bracket` anymore.
- `EvalsScreen.action_show_workbench_help` replaces the generic fallback with a teachful panel: orientation notes (word/character/skill benches; skill evals' subject + models requirements and the two model-creation paths) and glyph shortcuts. Screens with their own handler still take precedence per the app's delegation contract.
- `LabScreen` binds left/right as aliases for the bracket mode walk (screen-level fallbacks; arrow-consuming widgets win in Textual dispatch); footer copy updated to "←/→ [ ] Move mode focus" (test_lab_frame_mode_keys pinned to the new copy).
- Tests: `test_f1_help_is_teachful_and_identifier_free`, `test_arrow_keys_move_mode_chip_focus`; `test_lab_frame_mode_keys` marked bootstrap_profile (same TASK-32628 admission signature). Screen 21/22 (known pre-existing dev red), panel 21/21, empty-states 107/107, mode-keys 4/4.
- Files: `tldw_chatbook/app.py`, `tldw_chatbook/UI/Screens/lab_frame.py`, `tldw_chatbook/UI/Screens/evals_screen.py`, `Tests/UI/test_evals_skill_eval_screen.py`, `Tests/UI/test_lab_frame_mode_keys.py`.
