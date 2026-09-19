# Compact filtered picker — TASK-32699

Baseline: `2d4412ea12` on `feat/component-pattern-library`.

At 80×24 the filtered Open picker gave its filename input zero content columns
and laid Cancel outside the dialog. Open/Save now use the established compact
picker chrome, with a full-width labeled filename field above the filter and
actions. A resize class changes layout without remounting controls. Wide layout
is retained; styles are scoped to FileOpen/FileSave, excluding Enhanced pickers.
All dimensions use existing design tokens; the generated bundle was rebuilt.

ADR required: no. This repairs existing presentation under
[ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md),
[ADR-160](../../../../backlog/decisions/160-progressive-file-picker-listings.md) and
[ADR-161](../../../../backlog/decisions/161-component-pattern-library.md).

## Verification

[Verification](verification.json) records 12 new passing journeys and 167 passing
neighbor/governance checks. The new tests use all production app stylesheets and
consolidated widget defaults. They cover Open, Save, the optional Select folder
action, pasted paths, keyboard typing, changed filters, listing selection,
confirmation and cancellation in both themes. Compact/wide round trips preserve
editor identity, value, selection, filter, highlight and focus. Existing folder,
Enhanced, export and provider-recovery tests remain green. No full suite ran.

The first reproduction measured filename content width 0 and Cancel's right
edge at column 84. The initial native inspection then found an overlapping
filename label, which geometry checks alone missed. A painted-label assertion
failed before the margin correction and passes afterward. Fixture corrections
isolated the listing from pytest's other files and avoided assuming discovery
order is alphabetical or that parent paths are normalized.

[Static comparison](static-comparison.json) records zero new Ruff findings; five
inherited findings in the vendored file remain. The three touched Python files
pass formatting. The CSS byte budget passes at 616,506 / 634,050 bytes, with no
budget increase. Existing pytest cleanup and governance SyntaxWarnings remain.
[Independent review](review.json) found no actionable issues and separately
checked compact Open/Save validation errors and the extra folder action.

## Native evidence

[native_check.py](native_check.py) runs real TldwCli/LinuxDriver with private
configuration, databases, source fixtures and a null keyring. It opens the real
pickers directly on the app, delivers bracketed paste through the owned tmux
terminal, and resizes the terminal between 80×24 and 170×48. It checks actual
callbacks, painted labels/actions, filter results and retained editor state.

Final run-004 passed eight journeys and exited normally with status 0:
[result](result.json), [isolation](isolation.json), [lifecycle](lifecycle.json).
Eight captures were rendered and inspected in the final confirmation batch:
[inspection](inspection.json). [Persistence](persistence.json) records ten
healthy private databases, zero media/messages/ingest jobs, unchanged source
fixtures and default-profile hashes, and no app ERROR/CRITICAL log lines.
No model admission, import, export-file write or network operation was invoked.
This qualifies picker interaction, not downstream model or ingestion behavior.

| State | Dark | Light |
|---|---|---|
| Open, 80×24 | [capture](open-textual-dark-80.svg) | [capture](open-textual-light-80.svg) |
| Open, 170×48 | [capture](open-textual-dark-170.svg) | [capture](open-textual-light-170.svg) |
| Select folder focused, 80×24 | [capture](folder-textual-dark-80.svg) | [capture](folder-textual-light-80.svg) |
| Save, 80×24 | [capture](save-textual-dark-80.svg) | [capture](save-textual-light-80.svg) |
