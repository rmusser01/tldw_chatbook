# MCP Audit filter layout — TASK-32835

At 80 columns, fixed-width decision/initiator slots squeezed the Audit text
filter out of view; intermediate pane widths crowded the labels. Filters now
occupy full-width rows using existing design tokens. The execution pane scrolls
and reveals the currently focused control and table cursor after resize.
The same column works at wide sizes, avoiding a second breakpoint-specific
layout. No token values, permission policy, execution behavior or budgets change.
ADR required: no; existing ADR-150 and ADR-161 govern this bounded repair.

## Targeted verification

**107 distinct cases pass** in the [final run](tests/final-003.txt) and
[case ledger](qualified-cases.json): ten real-app label/keyboard/resize cases,
69 Audit behavior/layout cases, eight design-token guards, eighteen component
pattern guards and two CSS budget guards. Tests use private profiles; no full
suite was run. The new matrix covers dark/light at 80×24, 100×30, 120×40 and
170×48. It checks actual label paint, typing, filtering, Tab order, retained
values, focus and the last table row through resize.

Five obsolete source-literal pins are replaced with computed layout checks
for table height caps, the subview strip and selector slots. This closes the
Audit literal-pin debt recorded by TASK-32796. All [seven preflight guards](preflight.txt)
pass; byte/selector limits remain unchanged. [Static analysis](static-analysis.json)
records no introduced diagnostics (the legacy files retain their existing two
and three). New files and changed code ranges pass Ruff formatting.
[Independent review](independent-review.txt) found no actionable issue.

The evidence keeps each unsuccessful stage:

- [Initial compact failure](tests/red-001.txt): the text filter placeholder did
  not paint because the control had no usable width.
- [Layout-only run](tests/layout-001.txt): eight label cases passed; resize left
  the focused table outside the compositor. The focus/reflow callback fixes it;
  [both focused resize cases then passed](tests/focus-001.txt).
- [First combined run](tests/final-001.txt): 106 passed, one failed. A replacement
  test compared Input/Select content sizes rather than border boxes. The
  [geometry probe](tests/geometry-001.txt) shows equal 118-cell border boxes;
  Input has 112 content cells because its own border/padding occupies six.
  The corrected test compares each border box with the filter bar content box.
- [Legacy-only retry](tests/final-002.txt): 69 setup errors with
  `raw_source_selection_changed` while the shared fixture imported the app
  after configuration rebinding. No test body/UI ran. The final run restores
  the original combined collection scope; no configuration ownership check
  was changed or bypassed. Standalone legacy-module execution remains a
  harness limitation, not part of this repair.

## Native visual and lifecycle evidence

The real TldwCli runs with LinuxDriver and TTY streams in a private native
terminal. HOME, USERPROFILE, XDG and TLDW profile paths are validated before
application imports. The actual service execution log contains 48 synthetic
metadata records; no tools execute and no external server connects. Workbench
projection and filtering use the real service path.

At 80×24 and 170×48 in both themes, the journey types `review`, uses Tab and
keyboard menus to choose `Blocked (killswitch)` and `Test`, then tabs to the
table and uses Ctrl+End. All filter values persist and the last of 24 filtered
rows paints within the viewport. The first filter is focused directly; later
control/menu navigation uses terminal keystrokes.

All [18 captures](GALLERY.md) were rendered and visually inspected: typed text,
selected decision, selected initiator and last table row in four theme/size
cells, plus the compact decision menu in both themes. Full labels and values
are visible; reaching the table scrolls the compact execution pane.
The [native result](native/result.json) records every assertion and source hash.
The [lifecycle receipt](lifecycle.json) confirms exit 0, normal App.run return,
absent PID, released instance lock, ten healthy private databases, zero chats
or messages, unchanged default config/UI state/runtime policy and no errors or
faulthandler output.

Source hashes matched immediately after native exit. The only later production
change was Ruff wrapping the focus guard condition; its AST is identical, with
[both hashes recorded](source-format-verification.json). Generated CSS is
unchanged after capture. [Export hashes](export-manifest.json) record original
private artifacts and repository copies normalized only for trailing whitespace.

## Bounds and next review

Based on fresh dev `cef6bd2a3e3f0b8de0e166146acb4900ca7ea2b6` after PR #2707
merged. The separate Audit selection repair is saved in
[PR #2720](https://github.com/rmusser01/tldw_chatbook/pull/2720) and is not included
here. Inspector compact reachability and tool execution are separate reviews.
The built-in readiness guidance visible above unrelated local Audit/tool detail
remains the next bounded review. Final current-head CI and user visual approval
are still required before this follow-up can merge.

The [allocation check](allocation-owner-check.json) confirms sole TASK-32835
ownership across refs/worktrees. No full-suite or broader runtime qualification
is claimed.
