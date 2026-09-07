---
id: TASK-19192
title: 'CSS class-coverage contract red on dev: console-changed-files-section is unstyled'
status: Done
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
- [x] #1 `test_every_composed_class_is_styled_or_registered` passes on dev.
- [x] #2 The `console-changed-files-section` token is either genuinely styled (a `.class` or `#id` rule that actually applies to the composed widget) or registered as KNOWN_UNSTYLED with a recorded rationale for why it is a marker/query-only class; the choice is justified against the widget's intended appearance, not picked for gate convenience.
- [x] #3 If styling was the intended-but-missing half, the rendered Console change-review rail is verified (painted-frame or equivalent render evidence, not a style-object probe) to show the intended appearance.
<!-- AC:END -->

## Implementation Plan

1. Born-red: run the contract test at base (`63901c30d`) and record the
   failing token list (expect exactly `console-changed-files-section`).
2. Establish what the token is: `ConsoleChangedFilesSection` self-stamps
   `classes="console-changed-files-section"` in `__init__`; the token has
   no rule and no query anywhere. The container's real styling is split
   between the widget's DEFAULT_CSS type selector (height) and two inline
   Python styles at the right_rail mount site (`styles.width = "100%"`,
   `styles.min_width = 0`) — while every sibling rail section's container
   (`#console-staged-context-tray`, `#console-retrieval-scope-row`) carries
   those same declarations as an `#id` rule in
   `css/components/_agentic_terminal.tcss`.
3. Fix shape (a), matching the sibling convention: add a
   `#console-changed-files-section` rule to `_agentic_terminal.tcss` next to
   the scope-row rule, carrying the width declarations, and delete the two
   now-redundant inline styles in `UI/Console_Modules/right_rail.py` so the
   rule is load-bearing (not a decorative gate-pass duplicate). The contract
   counts a same-named `#id` rule on the same widget as styled — the
   established Console_Modules pattern (`#console-left-rail-body` etc.).
4. Regenerate the bundle via `python -m tldw_chatbook.css.build_css`
   (never hand-edit `tldw_cli_modular.tcss`).
5. Render evidence (AC#3): painted-frame probe (compositor render_strips)
   mounting the section exactly as right_rail.py does, on the real CSS
   stack — captured at base (inline styles) and after (tcss rule), diffed
   for parity.
6. Gates: the CSS coverage contract (4 tests), the bundle sync guard, the
   section's own tests, the wiring tests; repo-wide `--collect-only -q`.

## Implementation Notes

Fix shape: **styled, not registered** — the `#id` branch of the contract,
matching the sibling convention rather than the KNOWN_UNSTYLED escape.

The token was neither a pure marker nor properly styled: the section's
container declarations were split between the widget's DEFAULT_CSS type
selector (height) and two inline Python styles at the right_rail mount
site (`width = "100%"`, `min_width = 0`) — while every sibling rail
section's container (`#console-staged-context-tray`,
`#console-retrieval-scope-row`) carries exactly those declarations as an
`#id` rule in `css/components/_agentic_terminal.tcss`. Registering the
token would have blessed the one rail section whose container styling
lives outside the stylesheet.

- `css/components/_agentic_terminal.tcss`: new
  `#console-changed-files-section { width: 100%; min-width: 0; }` rule
  beside the scope-row rule (comment records the split: height from
  DEFAULT_CSS, rhythm from `.console-inspector-context-section`).
- `UI/Console_Modules/right_rail.py`: the two inline width styles deleted
  so the rule is load-bearing, not a decorative gate-pass duplicate;
  comment updated.
- `css/tldw_cli_modular.tcss`: regenerated via
  `python -m tldw_chatbook.css.build_css` (never hand-edited).

Born-red at base `63901c30d`: exactly one failing token,
`console-changed-files-section  (first composed in
tldw_chatbook/Widgets/Console/console_changed_files_section.py)`.

AC#3 render evidence: painted-frame probe (compositor `render_strips`,
the repo's painted-frame idiom) mounting the section exactly as
right_rail.py does — fixed id, `console-inspector-context-section` class,
quiet frame, real 13fr/4fr column split, real CSS stack (scoped + bundle
+ self). Frame captured at base (inline styles, pre-change bundle) and
after (tcss rule, no inline styles): byte-identical — header
"Changed files (3) · latest…", three status/delta/badge rows, pruned
tail, all painted at the same regions (section 30×5 inside the 34-col
rail). Probe detail worth keeping: a bare `#console-right-rail` outside
a Horizontal resolves `width: 4fr` to 4× the screen width in a vertical
layout, painting every row off-viewport — the probe needed the real
column split to render truthfully.

Gates: css coverage contract + registry tests 4/4; bundle sync guard 3/3;
changed-files section + wiring + right-rail suites 22/22; bundle-adjacent
CSS contracts (build integrity, staleness manifest, bundle rendering,
non-obscuring focus) 131/131; repo-wide `--collect-only -q` clean
(52,113 collected).
