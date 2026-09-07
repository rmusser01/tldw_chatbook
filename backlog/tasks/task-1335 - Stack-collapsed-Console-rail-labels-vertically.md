---
id: TASK-1335
title: Stack collapsed Console rail labels vertically
status: In Progress
assignee: []
created_date: '2026-08-03 03:05'
updated_date: '2026-08-07 21:53'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the collapsed Console context and Inspector handles read vertically so they consume less horizontal space while remaining understandable and keyboard accessible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Collapsed Context handle label reads top-to-bottom.
- [x] #2 Collapsed Inspector handle label reads top-to-bottom.
- [x] #3 Collapsed Console handles use a stable three-cell width without changing the Personas workbench handles.
- [x] #4 Expanded rail headers and handle tooltips remain horizontal and descriptive.
- [x] #5 Rail badges remain legible and retain their full tooltip text.
- [x] #6 Targeted Console rail tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: N/A

Reason: This is a reversible presentation refinement that preserves the shared
widget boundary, Console behavior, and persisted rail state.

1. Add failing widget and mounted-rail assertions for opt-in vertical labels,
   three-cell sizing, badges, tooltips, and the unchanged horizontal default.
2. Add an explicit vertical presentation option to `ConsoleRailHandle` and use
   it only from the two Console collapsed-handle call sites.
3. Add the vertical handle rules to the component TCSS and regenerate the
   bundled stylesheet.
4. Run the focused widget, Console rail, Personas, and CSS integrity checks;
   self-review the diff and record any pre-existing harness failures separately.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented an opt-in vertical presentation for only the collapsed Console
Context and Inspector handles. The shared handle keeps its horizontal default,
while the Console call sites use stable three-cell outer rails, stacked visible
text, and full descriptive tooltips for controls and badges. The component TCSS
was updated and the production modular TCSS was mechanically regenerated.

Areas changed: the Console rail-handle subclass, the two collapsed-handle call
sites, component/generated TCSS, and a dedicated focused test module. During
the pre-PR rebase, current `dev`'s shared `DestinationRailHandle` refactor was
preserved and the vertical behavior was layered onto that base. The deleted
monolithic persistent-rails test was not resurrected. The trade-off remains
deliberately opt-in: non-Console destination handles, persistence, responsive
state, and expanded headers are unchanged.

TDD/review evidence: dedicated contract tests cover vertical labels, geometry,
badge visibility/tooltips, paint containment, centered columns, and the
unchanged horizontal default. On the rebased tree, 81 focused tests passed across
`Tests/UI/test_console_rail_handle.py`, `Tests/UI/test_destination_rail.py`,
`Tests/UI/test_console_inspector_compact_access.py`,
`Tests/Chat/test_console_rail_state.py`, and
`Tests/UI/test_css_build_integrity.py`. The mounted compact-access coverage
exercises both real Console handles at narrow terminal widths.

Post-PR visual review exposed that Textual's compact `Button` still painted one
cell of line padding on each side of the declared one-cell vertical control.
That overflow was masked by the framed left handle but visibly displaced the
unframed right handle. The follow-up clears `line_pad` inline, centers the
one-cell child inside both three-cell handles, and adds a paint-level regression
that failed on the broken render before passing with the correction.

The repository-wide `pytest -q` command was run before PR creation but stopped
during collection with 28 environment/unrelated errors: the standard `[dev]`
extra does not install the optional NumPy/audio or Playwright stacks required
by several suites, and `Tests/TTS/test_profile_store_lock.py` also raised an
unrelated collection-time `TypeError`. These errors occur before the full suite
can execute; they are not counted as passing evidence for this task.

Implementation is complete and targeted verification is green, but this task
remains In Progress. The repository's strict Definition of Done also requires
the full suite, linter/formatter, and performance/security/licence gates to be
run green; those repository-wide gates have not all passed, so task closure is
withheld without exception.

ADR required: no

ADR path: N/A

Reason: The completed work is a reversible, opt-in presentation refinement;
it changes no stored state, service contract, provider boundary, or long-lived
application architecture.

Lessons learned: no duplicate lesson was added. The existing "A button's region
width proves nothing about whether its label renders" entry in
`backlog/docs/lessons-testing-evidence.md` describes this exact Textual
`line_pad` trap; the new paint-level regression applies that lesson directly.
The component source was updated first and the bundle was regenerated rather
than hand-edited.
<!-- SECTION:NOTES:END -->
