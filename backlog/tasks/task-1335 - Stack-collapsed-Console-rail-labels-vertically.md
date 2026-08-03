---
id: TASK-1335
title: Stack collapsed Console rail labels vertically
status: Done
assignee: []
created_date: '2026-08-03 03:05'
updated_date: '2026-08-03 04:03'
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

Areas changed: Console rail handle widget, the two Console collapsed-handle
call sites, component/generated TCSS, and focused Console/stylesheet tests.
The trade-off is deliberately opt-in presentation behavior: Personas handles,
persistence, responsive state, and expanded headers remain unchanged.

TDD/review evidence: dedicated contract tests cover vertical labels, geometry,
badge visibility/tooltips, and the unchanged horizontal default; Tasks 1-3
passed per-task spec and quality review. Final focused verification passed 46
tests: 41 Console rail/handle/state and Personas-selection tests, plus 5 CSS
build-integrity tests. The four changed Python modules/tests compiled with
`py_compile`, and `git diff --check 659186711..HEAD` passed. Final diff review
found only the intended Console widget/call-site/test/TCSS scope; the generated
TCSS change is mechanical and no secrets were introduced.

The two existing full-app mounted Console selections remain blocked before
screen mount during `TldwCli()` construction by `sqlite3.OperationalError:
attempt to write a readonly database` against user-data SQLite paths under
`~/.local/share/tldw_cli` (scheduled-tasks and library-collections wiring).
This is the documented pre-existing harness condition; it does not exercise or
demonstrate a regression in the rail implementation.

ADR required: no

ADR path: N/A

Reason: The completed work is a reversible, opt-in presentation refinement;
it changes no stored state, service contract, provider boundary, or long-lived
application architecture.
<!-- SECTION:NOTES:END -->
