---
id: TASK-19192
title: 'CSS class-coverage contract red on dev: console-changed-files-section is unstyled'
status: To Do
assignee: []
created_date: '2026-08-20'
labels:
  - test-health
  - css
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_css_class_coverage_contract.py::test_every_composed_class_is_styled_or_registered`
is red on dev `7877defba` (re-run 2026-08-20 in a pristine worktree, isolated
config). The failing set has CHANGED since TASK-19042's baseline: the six
`console-*` tokens it recorded (left_rail, console_transcript,
console_selection_menu, console_turn_file_card families) no longer fail —
resolved since — and exactly ONE token now trips the gate:

- `console-changed-files-section` (first composed in
  `tldw_chatbook/Widgets/Console/console_changed_files_section.py`, the
  changed-files rail section widget landed in 12d621071 / 9cc5b6420 /
  8f7085b0c)

The contract offers two legitimate exits: give the class a real rule in the
bundle or a DEFAULT_CSS, or register it in the test's KNOWN_UNSTYLED list with
a rationale (it may be a pure query/marker class). Pick whichever is true of
the widget — do not blind-register a class that was meant to be styled.
Note the same landing also left this file out of the diagnostic inventory
(TASK-19191's diff shows it at +2 diagnostics), so the changed-files rail
evidently shipped without running the repo's contract gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `test_every_composed_class_is_styled_or_registered` passes on dev.
- [ ] #2 The `console-changed-files-section` token is either genuinely styled (a `.class` or `#id` rule that actually applies to the composed widget) or registered as KNOWN_UNSTYLED with a recorded rationale for why it is a marker/query-only class; the choice is justified against the widget's intended appearance, not picked for gate convenience.
- [ ] #3 If styling was the intended-but-missing half, the rendered Console change-review rail is verified (painted-frame or equivalent render evidence, not a style-object probe) to show the intended appearance.
<!-- AC:END -->
