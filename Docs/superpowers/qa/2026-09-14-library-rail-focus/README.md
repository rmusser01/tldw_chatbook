# Library rail focus repair — TASK-32598

The Library rail now reveals a newly focused control above its docked scroll cue. Before the repair, Tab from Search/RAG put Create at `(22,21,3,1)` with a blank painted crop; Enter still activated it. After the repair, one row of scrolling puts the toggle at `(22,20,3,1)` with its arrow and underline visible, while the cue stays at row 21.

Baseline: `382a6f4d53`. The earlier [Library audit](../../reports/2026-09-14-library-workflow-audit.md) retains the before evidence. This directory records the repair's after evidence.

## Change and reason

Textual's `Screen.can_view_entire` tests ancestor content-region containment without subtracting dock gutters. It returned true for the covered toggle, bypassing automatic focus scrolling. `LibraryRail.on_descendant_focus` now uses the existing dock-aware `scroll_to_widget` operation when the cue is visible and the event still names the focused widget. The scroll is immediate, retains the target, and adds no queries, styles or tokens in the handler.

The production file also received its existing Ruff import-order, redundant-annotation/noqa and formatting cleanup. No stylesheet or generated CSS was changed.

ADR required: no. Existing ADR-150, ADR-161 and ADR-086 govern this routine restoration of visible keyboard focus; no new interface or application structure was introduced.

## Verification

- The new production-styled regression failed in both themes before implementation: the focused glyph was absent while `can_view_entire` was true. Both cases passed with the repair, including Enter activation and retained focus.
- `pytest -q Tests/UI/test_library_rail_focus_visibility.py Tests/UI/test_library_crit9_rail.py Tests/UI/test_design_token_governance.py`: **29 passed**. This covers forward/reverse traversal of Browse, Create, Study, Import/Export, Details and Diagnostics, existing rail layout/copy checks, fold-cue behavior and token governance.
- The focus test was then strengthened to hold **Diagnostics** through 120×45 → 80×24 → 120×45. Its complete four-case file passed again. These four are overlapping reruns, not additional unique tests.
- The two archived audit focus probes passed in dark/light and produced the SVG/JSON evidence here. Both SVGs were rendered and visually inspected.
- Native app: private profile, all ten database paths and base data directories redirected to ignored audit scratch. At 80×24, `/`, nine Tabs to Search/RAG, then Tab revealed Create above the cue. ANSI confirms the arrow's bold/underline style. Enter collapsed Create; resizing to 120×45 retained its visible focused collapsed arrow. No external model request was made. Exit code was **0**.
- Ruff check and format-check passed for the changed production and test files. The changed diff passes whitespace checks; Backlog IDs remain unique.

The two already-known query-budget tests were rerun separately: still **23** LibraryScreen queries across three non-crossing frames and **5** per Tab, exactly the audit baseline. They remain under TASK-32599; this repair does not claim to resolve them. The native log also retains the known missing `#app-log-display` startup error, outside this repair. Pytest's two warnings concern cleanup of an older temporary directory.

## Artifacts and reproduction

- [Dark theme](rail-fold-focus-textual-dark.svg) and [light theme](rail-fold-focus-textual-light.svg): focused Create after one-row scroll; accompanying `*-detail.json` preserves exact geometry and paint.
- [Native compact focus](create-focused-80.ansi) and [native activated section after resize](create-collapsed-120.ansi).
- `startup-exit.ansi`, `native-log-review.txt`, and `test-results.json` preserve exit/log review and exact test summary lines.

Run the targeted commands above from the repository root with `.venv/bin/python -m pytest`. The full suite was not run. Raw logs and private data remain under `.superpowers/sdd/2026-09-15-library-rail-focus/`; disposable capture probes were removed from `Tests/UI/`. Indentation-only SVG lines and trailing terminal-line padding were trimmed for whitespace hygiene; measured JSON paint remains exact.
